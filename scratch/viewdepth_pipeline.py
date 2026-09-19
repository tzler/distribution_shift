"""View-conditioned, model-free distribution shift: compare IMAGES, not objects.

Stage 1  estimate each MOCHI image's camera by matching its silhouette to voxel projections
         of the known object over a pose grid (no renderer; MOCHI image used only for pose)
Stage 2  training-image bank: every training object projected at the renderer's 15 fixed
         poses -> depth+silhouette descriptor  (313k images)
Stage 3  test descriptors: each MOCHI image projected at its estimated pose, same code
Stage 4  image-level coverage / knn, per category bank, through the usual battery, head to
         head with object-level coverage; plus a view-only shift for the decomposition
"""
import argparse, ast, os, sys, math, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, G); sys.path.insert(0, f'{G}/scratch')
from voxdepth import project, sphere_eye
from geom3d import read_binvox
from geom2d import silhouette
NAV = '/vast/projects/bonnen/naturalistic-navig'; MOCHI = f'{NAV}/MOCHI'; BD = f'{NAV}/Dist-shift-data/blend-data'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
SYN = {'airplane':'02691156','bench':'02828884','cabinet':'02933112','car':'02958343','chair':'03001627','display':'03211117',
       'lamp':'03636649','loudspeaker':'03691459','sofa':'04256520','table':'04379243','telephone':'04401088','watercraft':'04530566'}
INV = {v:k for k,v in SYN.items()}
RES = 32; POSE_RES = 80; NPOSE = 800

def fib_sphere(n, radius=2.0):
    g = (1 + 5 ** .5) / 2; out = []
    for i in range(n):
        th = math.acos(1 - 2 * (i + .5) / n); ph = 2 * math.pi * i / g
        out.append((th, ph % (2 * math.pi)))
    return np.array(out)
TRAIN_POSES = fib_sphere(15)                      # identical to render_shapenet.generate_viewpoints

def surface_points(bv):
    _, occ, _, _ = read_binvox(bv)
    pad = np.pad(occ, 1); nb = (pad[:-2,1:-1,1:-1]&pad[2:,1:-1,1:-1]&pad[1:-1,:-2,1:-1]&pad[1:-1,2:,1:-1]&pad[1:-1,1:-1,:-2]&pad[1:-1,1:-1,2:])
    P = np.argwhere(occ & ~nb).astype(np.float64) + .5
    if len(P) < 32: return None
    P -= P.mean(0); P /= np.linalg.norm(P, axis=1).max() + 1e-12
    return P

def norm_sil(sil, out=64):
    """scale/translation-normalise a silhouette: crop to bbox, pad square, resize."""
    from PIL import Image
    ys, xs = np.nonzero(sil)
    if len(ys) < 8: return np.zeros((out, out), bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = sil[y0:y1, x0:x1]; h, w = c.shape; s = max(h, w); pad = np.zeros((s, s), bool)
    pad[(s-h)//2:(s-h)//2+h, (s-w)//2:(s-w)//2+w] = c
    return np.asarray(Image.fromarray(pad.astype(np.uint8)*255).resize((out, out), Image.BILINEAR)) > 127

def descriptor(P, theta, phi):
    depth, sil = project(P, sphere_eye(theta, phi), res=RES)
    d = depth.copy()
    if sil.any(): d[sil] -= np.median(d[sil])
    return np.concatenate([sil.ravel(), d.ravel()]).astype(np.float16)

# ---------------------------------------------------------------- stage 1
GRID = fib_sphere(NPOSE)
def _pose_one(arg):
    obj, images = arg
    bv = f'{BD}/shapenet_mochi_excluded/{obj}/models/model_normalized.solid.binvox'
    if not os.path.exists(bv): return []
    P = surface_points(bv)
    if P is None: return []
    sils = np.stack([norm_sil(project(P, sphere_eye(th, ph), res=POSE_RES)[1]) for th, ph in GRID])   # NPOSE x 64 x 64
    out = []
    for img in images:
        m = norm_sil(silhouette(f'{MOCHI}/images/{img}', bg=(255, 255, 255), size=POSE_RES))
        inter = (sils & m).sum((1, 2)); union = (sils | m).sum((1, 2)); iou = inter / np.maximum(union, 1)
        k = int(iou.argmax()); th, ph = GRID[k]
        # local refinement
        best = (iou[k], th, ph)
        for dth in np.linspace(-.08, .08, 5):
            for dph in np.linspace(-.08, .08, 5):
                t2, p2 = np.clip(th + dth, .02, math.pi - .02), (ph + dph) % (2 * math.pi)
                s2 = norm_sil(project(P, sphere_eye(t2, p2), res=POSE_RES)[1]); v = (s2 & m).sum() / max((s2 | m).sum(), 1)
                if v > best[0]: best = (v, t2, p2)
        srt = np.sort(iou)[::-1]
        out.append(dict(image=img, obj=obj, theta=best[1], phi=best[2], iou=best[0], iou_2nd=srt[1], iou_coarse=iou[k]))
    return out

def stage1(workers):
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv'); mm = m[m.dataset == 'shapenet']
    by = {}
    for _, r in mm.iterrows():
        for f in ast.literal_eval(r['images']): by.setdefault('/'.join(f[:-4].split('_')[:2]), set()).add(f)
    args = [(o, sorted(v)) for o, v in sorted(by.items())]
    with Pool(workers) as p: res = [x for chunk in p.imap_unordered(_pose_one, args, chunksize=4) for x in chunk]
    df = pd.DataFrame(res); df.to_csv(f'{G}/out/mochi_poses.csv', index=False)
    print(f'[stage1] {len(df)} images posed; IoU median {df.iou.median():.3f}, 10th pct {df.iou.quantile(.1):.3f}, refine gain {np.mean(df.iou-df.iou_coarse):.3f}', flush=True)
    return df

# ---------------------------------------------------------------- stage 2
def _bank_one(obj):
    bv = f'{BD}/shapenet_training/{obj}/models/model_normalized.solid.binvox'
    P = surface_points(bv)
    if P is None: return []
    return [(f'{obj}/{k:03d}', descriptor(P, th, ph)) for k, (th, ph) in enumerate(TRAIN_POSES)]

def stage2(workers):
    root = f'{BD}/shapenet_training'; objs = []
    for syn in sorted(os.listdir(root)):
        if syn not in INV: continue
        for o in sorted(os.listdir(f'{root}/{syn}')):
            if os.path.exists(f'{root}/{syn}/{o}/models/model_normalized.solid.binvox'): objs.append(f'{syn}/{o}')
    print(f'[stage2] {len(objs)} training objects x 15 poses', flush=True)
    ids, X = [], []
    with Pool(workers) as p:
        for i, chunk in enumerate(p.imap(_bank_one, objs, chunksize=16)):
            for k, v in chunk: ids.append(k); X.append(v)
            if i % 2000 == 0: print(f'   {i}/{len(objs)}', flush=True)
    X = np.stack(X); np.savez(f'{G}/bank/bank_viewdepth.npz', ids=np.array(ids), X=X)
    print(f'[stage2] bank {X.shape} {X.dtype}', flush=True)

# ---------------------------------------------------------------- stage 3
def _test_one(arg):
    obj, rows = arg
    P = surface_points(f'{BD}/shapenet_mochi_excluded/{obj}/models/model_normalized.solid.binvox')
    if P is None: return []
    return [(r['image'], descriptor(P, r['theta'], r['phi'])) for r in rows]

def stage3(workers):
    df = pd.read_csv(f'{G}/out/mochi_poses.csv'); args = [(o, g.to_dict('records')) for o, g in df.groupby('obj')]
    ids, X = [], []
    with Pool(workers) as p:
        for chunk in p.imap_unordered(_test_one, args, chunksize=4):
            for k, v in chunk: ids.append(k); X.append(v)
    X = np.stack(X); np.savez(f'{G}/bank/test_viewdepth.npz', ids=np.array(ids), X=X); print(f'[stage3] test {X.shape}', flush=True)

def _test15_one(obj):
    P = surface_points(f'{BD}/shapenet_mochi_excluded/{obj}/models/model_normalized.solid.binvox')
    if P is None: return []
    return [(f'{obj}/{k:03d}', descriptor(P, th, ph)) for k, (th, ph) in enumerate(TRAIN_POSES)]

def stage3b(workers):
    df = pd.read_csv(f'{G}/out/mochi_poses.csv'); objs = sorted(df.obj.unique()); ids, X = [], []
    with Pool(workers) as p:
        for chunk in p.imap_unordered(_test15_one, objs, chunksize=8):
            for k, v in chunk: ids.append(k); X.append(v)
    X = np.stack(X); np.savez(f'{G}/bank/test_viewdepth15.npz', ids=np.array(ids), X=X); print(f'[stage3b] test@15 fixed poses {X.shape}', flush=True)

# ---------------------------------------------------------------- stage 4
def _variant(X, v):
    """derive a descriptor variant from the saved 32x32 sil+depth maps."""
    n = len(X); sil = X[:, :1024].reshape(n, 32, 32); dep = X[:, 1024:].reshape(n, 32, 32)
    def pool(a, k): return a.reshape(n, 32 // k, k, 32 // k, k).mean((2, 4)).reshape(n, -1)
    if v == 'full': return X
    if v == 'sil': return sil.reshape(n, -1)
    if v == 'depth': return dep.reshape(n, -1)
    if v == 'pool2': return np.concatenate([pool(sil, 2), pool(dep, 2)], 1)
    if v == 'pool4': return np.concatenate([pool(sil, 4), pool(dep, 4)], 1)
    if v == 'pool2sil': return pool(sil, 2)
    if v == 'pool4sil': return pool(sil, 4)
    raise ValueError(v)

def stage4(variant='full', fixed15=False, single=None):
    from scipy import stats
    bz = np.load(f'{G}/bank/bank_viewdepth.npz', allow_pickle=True); tz = np.load(f'{G}/bank/test_viewdepth15.npz' if fixed15 else f'{G}/bank/test_viewdepth.npz', allow_pickle=True)
    keep_obj = set(np.load(f'{G}/bank/bank_voxel16.npz', allow_pickle=True)['ids'])          # the SAME 20,885 training objects as every object-level analysis
    bobj = np.array(['/'.join(i.split('/')[:2]) for i in bz['ids']]); sel = np.array([o in keep_obj for o in bobj])
    print(f'[stage4] image bank: {sel.sum()} of {len(sel)} rows kept ({len(set(bobj[sel]))} objects = the object-level bank)', flush=True)
    B = _variant(bz['X'][sel].astype(np.float32), variant); T = _variant(tz['X'].astype(np.float32), variant); bz = {'ids': bz['ids'][sel]}
    print(f'[stage4] variant {variant}: descriptor dim {B.shape[1]}', flush=True)
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-6] = 1
    B = (B - mu) / sd; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    T = (T - mu) / sd; T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    bsyn = np.array([i.split('/')[0] for i in bz['ids']]); tix = {k: i for i, k in enumerate(tz['ids'])}
    poses = pd.read_csv(f'{G}/out/mochi_poses.csv').set_index('image')
    EPSS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    # per test image, per category: knn_mean(50) and coverage counts at several eps
    per = {c: {} for c in SYN}
    for c, s in SYN.items():
        Bc = B[bsyn == s]; knn = np.empty(len(T)); cnt = {e: np.empty(len(T)) for e in EPSS}
        for i0 in range(0, len(T), 256):
            D = 1.0 - T[i0:i0+256] @ Bc.T
            knn[i0:i0+256] = np.sort(np.partition(D, 49, axis=1)[:, :50], axis=1).mean(1)
            for e in EPSS: cnt[e][i0:i0+256] = (D < e).sum(1)
        per[c]['knn'] = knn
        for e in EPSS: per[c][f'cov{e}'] = -np.log1p(cnt[e])

    # view-only shift: angular distance from the estimated MOCHI pose to the nearest of the 15 training poses (trial property)
    def ang(t1, p1, t2, p2):
        v1 = np.array([np.sin(t1)*np.cos(p1), np.sin(t1)*np.sin(p1), np.cos(t1)]); v2 = np.array([np.sin(t2)*np.cos(p2), np.sin(t2)*np.sin(p2), np.cos(t2)])
        return np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))
    poses['view_gap'] = [min(ang(r.theta, r.phi, t, p) for t, p in TRAIN_POSES) for r in poses.itertuples()]
    # assemble per (trial, category)
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv'); mm = m[m.dataset == 'shapenet']; rows = []
    for c in SYN:
        for _, r in mm.iterrows():
            ims = ast.literal_eval(r['images'])
            if not all(f in poses.index for f in ims): continue
            if fixed15 and single:
                objs_ = ['/'.join(f[:-4].split('_')[:2]) for f in ims]
                if single == 'nearest':   # snap each MOCHI image to the nearest of the 15 training poses: one exact on-grid view per image
                    ks = [int(np.argmin([ang(poses.loc[f, 'theta'], poses.loc[f, 'phi'], t, p) for t, p in TRAIN_POSES])) for f in ims]
                else:                      # a random training view per image (seeded by trial)
                    ks = list(np.random.default_rng(abs(hash(r['trial'])) % (2**32)).integers(0, 15, len(ims)))
                keys = [f'{o}/{k:03d}' for o, k in zip(objs_, ks)]
                if not all(k in tix for k in keys): continue
                ix = [tix[k] for k in keys]
            elif fixed15:
                objs_ = ['/'.join(f[:-4].split('_')[:2]) for f in ims]; keys = [f'{o}/{k:03d}' for o in objs_ for k in range(15)]
                if not all(k in tix for k in keys): continue
                ix = [tix[k] for k in keys]
            else:
                if not all(f in tix for f in ims): continue
                ix = [tix[f] for f in ims]
            rec = dict(trial=r['trial'], category=c, view_gap=float(np.mean([poses.loc[f, 'view_gap'] for f in ims])), pose_iou=float(np.mean([poses.loc[f, 'iou'] for f in ims])))
            for k, v in per[c].items(): rec[f'img_{k}'] = float(v[ix].mean())
            rows.append(rec)
    d = pd.DataFrame(rows)
    obj = pd.read_csv(f'{G}/out/coverage_shapenet_voxel16_percat.csv').rename(columns={'coverage': 'obj_cov', 'knn': 'obj_knn'})[['trial', 'category', 'obj_cov', 'obj_knn', 'ft', 'pre', 'fc', 'pc', 'own']]
    d = d.merge(obj, on=['trial', 'category']); d = d[d.groupby('trial').trial.transform('size') == 12].copy()
    tag = variant + ('_fixed15' if fixed15 else '') + (f'_{single}' if single else ''); d.to_csv(f'{G}/out/viewdepth_long_{tag}.csv', index=False)
    print(f'\n[stage4] n={len(d)} trials={d.trial.nunique()}   pose IoU mean {d.pose_iou.mean():.3f}   view gap to nearest training pose: median {d.view_gap.median():.1f} deg', flush=True)
    def cen(v, g): return v - v.groupby(g).transform('mean')
    w = d[d.category == d.own]
    print(f'\n{"metric":14s} | {"within-trial r":>14s} {"%neg":>6s} | {"pooled r / ctrl":>16s} | {"on-cat r / ctrl":>16s} | {"2-way centred":>13s}')
    out = []
    for col in ['obj_knn', 'obj_cov', 'img_knn'] + [f'img_cov{e}' for e in EPSS]:
        xc = cen(d[col], d.trial); yc = cen(d.ft, d.trial); rw = stats.pearsonr(xc, yc)[0]
        sl = np.array([stats.linregress(s[col], s.ft).slope for _, s in d.groupby('trial') if s[col].std() > 0])
        rp, rpc = stats.pearsonr(d[col], d.ft)[0], stats.pearsonr(d[col], d.pre)[0]
        ro, roc = stats.pearsonr(w[col], w.ft)[0], stats.pearsonr(w[col], w.pre)[0]
        x2 = d[col] - d.groupby('trial')[col].transform('mean') - d.groupby('category')[col].transform('mean') + d[col].mean()
        y2 = d.ft - d.groupby('trial').ft.transform('mean') - d.groupby('category').ft.transform('mean') + d.ft.mean(); r2 = stats.pearsonr(x2, y2)[0]
        print(f'{col:14s} | {rw:+14.3f} {100*(sl<0).mean():5.1f}% | {rp:+7.3f} / {rpc:+6.3f} | {ro:+7.3f} / {roc:+6.3f} | {r2:+13.3f}', flush=True)
        out.append(dict(metric=col, within_trial_r=rw, pct_neg=100*(sl<0).mean(), pooled_r=rp, pooled_ctrl=rpc, oncat_r=ro, oncat_ctrl=roc, twoway_r=r2))
    pd.DataFrame(out).to_csv(f'{G}/out/viewdepth_battery_{tag}.csv', index=False)
    if not fixed15:
        print('\nactual-view run, split by pose-estimation quality (trial mean IoU):')
        for lo, hi in [(0, .8), (.8, .88), (.88, 1.01)]:
            sub = d[(d.pose_iou >= lo) & (d.pose_iou < hi)]
            if sub.trial.nunique() < 40: continue
            for col in ['obj_cov', 'img_cov0.3', 'img_cov0.4']:
                xc = cen(sub[col], sub.trial); yc = cen(sub.ft, sub.trial)
                print(f'  IoU [{lo:.2f},{hi:.2f}) n_trials={sub.trial.nunique():3d}  {col:10s} within-trial r = {stats.pearsonr(xc, yc)[0]:+.3f}')
    # view-only shift: trial-constant, so only pooled/on-cat rows are meaningful
    print(f'\nview-only shift (deg to nearest training pose), one value per trial:')
    t1 = d.groupby('trial').agg(vg=('view_gap', 'first'), ft_on=('ft', lambda s: s[d.loc[s.index, 'category'] == d.loc[s.index, 'own']].mean()), pre=('pre', 'first'))
    print(f'  r(view gap, on-category fine-tuned margin) = {stats.pearsonr(t1.vg, t1.ft_on)[0]:+.3f}   r(view gap, pretrained margin) = {stats.pearsonr(t1.vg, t1.pre)[0]:+.3f}   n={len(t1)}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--variant', default='full'); ap.add_argument('--fixed15', action='store_true'); ap.add_argument('--single', default=None)
    a = ap.parse_args()
    if a.smoke:
        m = pd.read_csv(f'{MOCHI}/mochi_trials.csv'); mm = m[m.dataset == 'shapenet'].head(4); by = {}
        for _, r in mm.iterrows():
            for f in ast.literal_eval(r['images']): by.setdefault('/'.join(f[:-4].split('_')[:2]), set()).add(f)
        for o, v in list(by.items())[:3]:
            for rec in _pose_one((o, sorted(v))): print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rec.items()})
        sys.exit()
    {'1': lambda: stage1(a.workers), '2': lambda: stage2(a.workers), '3': lambda: stage3(a.workers), '3b': lambda: stage3b(a.workers), '4': lambda: stage4(a.variant, a.fixed15, a.single)}[a.stage]()
