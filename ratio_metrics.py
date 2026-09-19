"""
A ratio family:            shift(t) = sim(X_test, X_train) / Y

  numerator   how similar / far the trial's objects are from the training bank
              (oddity-blind: computed per object, averaged over the trial's images)
  Y = 1       the plain train-test comparison (what we have been using)
  Y = sim(A,B) normalise by how similar the trial's own two objects are

The point of Y: trial difficulty is carried by sim(A,B). Dividing it out puts the
normalisation INSIDE the metric, so the pooled (trial-level) analysis might become
valid on its own, without needing the 12-model within-trial design.

Decisive test — the PRETRAINED margin. The pretrained encoder never saw any of these
training sets, so a valid shift measure must NOT predict its margin. The un-normalised
shift fails this: pooled r = +0.193. If a ratio drives that toward 0 while keeping the
fine-tuned effect, the normaliser is doing real work.

Both orientations are computed, since they are not monotone transforms of each other:
  dist_ratio  =  d(test, train) / d(A, B)
  sim_ratio   =  s(test, train) / s(A, B)          [s = 1 - d]
"""
import ast, os, sys
import numpy as np
import pandas as pd
from scipy import stats

G = os.path.dirname(os.path.abspath(__file__))
NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
K = 50
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}
REPS = [('voxel16', f'{G}/bank/bank_voxel16.npz', f'{G}/bank/test_voxel16.npz', None),
        ('voxel8', f'{G}/bank/bank_voxel8.npz', f'{G}/bank/test_voxel8.npz', None),
        ('full57', f'{G}/bank/bank3d_shapenet_trained.npz',
         f'{G}/bank/test3d_shapenet.npz', None),
        ('bbox7', f'{G}/bank/bank_bbox.npz', f'{G}/bank/test_bbox.npz', None),
        ('d2only', f'{G}/bank/bank3d_shapenet_trained.npz',
         f'{G}/bank/test3d_shapenet.npz', np.arange(0, 32)),
        ('shellonly', f'{G}/bank/bank3d_shapenet_trained.npz',
         f'{G}/bank/test3d_shapenet.npz', np.arange(32, 48))]
EPS = 1e-6


def prep(bf, tf, cols):
    bz = np.load(bf, allow_pickle=True); tz = np.load(tf, allow_pickle=True)
    B, T = bz['X'].astype(np.float64), tz['X'].astype(np.float64)
    if cols is not None:
        B, T = B[:, cols], T[:, cols]
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd
    BN = BZ / (np.linalg.norm(BZ, axis=1, keepdims=True) + 1e-12)
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    return list(bz['ids']), BN, {k: i for i, k in enumerate(tz['ids'])}, TN


def trials_list():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    out = []
    for _, r in mm.iterrows():
        ims = ast.literal_eval(r['images']); o = int(r['oddity_index'])
        out.append((r['trial'], [oid(f) for f in ims],
                    oid(ims[o]), [oid(f) for j, f in enumerate(ims) if j != o]))
    return out


def behaviour():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in SYN:
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['ft'] = o['fine_tuned_oddity_margin']
        o['pre'] = o['pretrained_oddity_margin']
        o['adv'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'ft', 'pre', 'adv']])
    return pd.concat(rows)


def build(rep):
    name, bf, tf, cols = rep
    bids, BN, tix, TN = prep(bf, tf, cols)
    syn = np.array([i.split('/')[0] for i in bids])
    tl = trials_list()
    recs = []
    for cat, s in SYN.items():
        sel = syn == s
        if sel.sum() < K:
            continue
        D = 1.0 - TN @ BN[sel].T
        kk = min(K, D.shape[1])
        d_train = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1).mean(1)
        for trial, keys, kb, ka in tl:
            if not all(k in tix for k in keys) or kb not in tix:
                continue
            num_d = float(np.mean([d_train[tix[k]] for k in keys]))   # oddity-blind
            a = TN[[tix[k] for k in ka]].mean(0)
            a = a / (np.linalg.norm(a) + 1e-12)
            s_ab = float(a @ TN[tix[kb]])
            d_ab = 1.0 - s_ab
            recs.append(dict(trial=trial, category=cat,
                             plain=num_d,
                             dist_ratio=num_d / (abs(d_ab) + EPS),
                             sim_ratio=(1.0 - num_d) / (abs(s_ab) + EPS),
                             d_ab=d_ab, s_ab=s_ab))
    return pd.DataFrame(recs)


def evaluate(d, name):
    beh = behaviour()
    d = d.merge(beh, on=['trial', 'category'])
    out = []
    for metric in ['plain', 'dist_ratio', 'sim_ratio']:
        # pooled: average the metric over the 12 banks -> one value per trial
        t = d.groupby('trial').agg(x=(metric, 'mean'), ft=('ft', 'mean'),
                                   pre=('pre', 'first'), s_ab=('s_ab', 'first')).reset_index()
        r_ft = stats.pearsonr(t.x, t.ft)
        r_pre = stats.pearsonr(t.x, t.pre)
        r_ab = stats.pearsonr(t.x, t.s_ab)
        # within-trial (two-way), for comparison
        w = d.copy()
        for c in [metric, 'ft']:
            w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                    - d.groupby('category')[c].transform('mean') + d[c].mean())
        r_w = stats.pearsonr(w[metric], w['ft'])
        out.append(dict(rep=name, metric=metric, pooled_ft=r_ft[0], p_ft=r_ft[1],
                        pooled_pre=r_pre[0], p_pre=r_pre[1], r_with_sAB=r_ab[0],
                        within_ft=r_w[0]))
    return out


if __name__ == '__main__':
    rows = []
    for rep in REPS:
        if not (os.path.exists(rep[1]) and os.path.exists(rep[2])):
            print(f'skip {rep[0]} (missing bank)'); continue
        rows += evaluate(build(rep), rep[0])
        print(f'  {rep[0]} done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f'{G}/out/ratio_metrics.csv', index=False)
    print()
    hdr = (f'{"representation":12s}{"metric":12s}{"POOLED ft":>11s}{"POOLED pre":>12s}'
           f'{"|r| with s(A,B)":>17s}{"within-trial":>14s}')
    print(hdr); print('-' * len(hdr))
    for _, r in out.iterrows():
        flag = '  <-- pretrained leak' if abs(r.pooled_pre) > .10 else ''
        print(f'{r.rep:12s}{r.metric:12s}{r.pooled_ft:>+11.3f}{r.pooled_pre:>+12.3f}'
              f'{abs(r.r_with_sAB):>17.3f}{r.within_ft:>+14.3f}{flag}')
    print(f'\nwrote {G}/out/ratio_metrics.csv')
