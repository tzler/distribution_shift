"""
Is the result specific to the 57-d descriptor we started with?

Eight model-free object representations, spanning almost the whole range of what
"geometric" could mean, evaluated identically:

  voxel16     raw 16^3 occupancy grid (4096 dims) — NO designed features at all
  voxel8      raw 8^3 occupancy grid (512 dims)
  full57      the descriptor used so far (D2 + shell + scalars)
  d2only      just the 32-bin D2 pairwise-distance histogram
  shellonly   just the 16-bin centroid-distance histogram
  scalars9    just 9 scalars (elongation, flatness, convexity, ...)
  bbox7       7 numbers from model_normalized.json — extents, aspect, volume, vertices
  SHUFFLED    full57 with test objects randomly re-assigned  <- negative control

Each gets the same oddity-blind k=50 cosine shift, the same per-category banks, and the
same within-trial evaluation against the fine-tuned oddity margin.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
EMP = 'trial_distance_(L1_not_normalized)'
K = 50
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}


def load(bank_f, test_f, cols=None, shuffle=False, seed=0):
    bz = np.load(bank_f, allow_pickle=True); tz = np.load(test_f, allow_pickle=True)
    B, T = bz['X'].astype(np.float64), tz['X'].astype(np.float64)
    if cols is not None:
        B, T = B[:, cols], T[:, cols]
    if shuffle:
        T = T[np.random.default_rng(seed).permutation(len(T))]
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd
    BN = BZ / (np.linalg.norm(BZ, axis=1, keepdims=True) + 1e-12)
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    return list(bz['ids']), BN, {k: i for i, k in enumerate(tz['ids'])}, TN


def shift_panel(bids, BN, tix, TN, trials_imgs):
    syn = np.array([i.split('/')[0] for i in bids])
    recs = []
    for cat, s in SYN.items():
        sel = syn == s
        if sel.sum() < K:
            continue
        D = 1.0 - TN @ BN[sel].T
        kk = min(K, D.shape[1])
        per = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1).mean(1)
        for trial, keys in trials_imgs:
            if all(k in tix for k in keys):
                recs.append((trial, cat, float(np.mean(per[[tix[k] for k in keys]]))))
    return pd.DataFrame(recs, columns=['trial', 'category', 'shift'])


def behaviour():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in SYN:
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'ft_margin', 'advantage']])
    return pd.concat(rows)


def evaluate(d, dv='ft_margin'):
    w = d.copy()
    for c in ['shift', dv]:
        w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                - d.groupby('category')[c].transform('mean') + d[c].mean())
    r = stats.pearsonr(w['shift'], w[dv])
    sl = np.array([stats.linregress(s['shift'], s[dv]).slope
                   for _, s in d.groupby('trial') if s['shift'].std() > 0])
    t = stats.ttest_1samp(sl, 0)
    return r[0], r[1], t.statistic, 100 * (sl < 0).mean()


def main():
    import ast
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    trials_imgs = [(r['trial'], [oid(f) for f in ast.literal_eval(r['images'])])
                   for _, r in mm.iterrows()]
    beh = behaviour()

    F57 = (f'{G}/bank/bank3d_shapenet_trained.npz', f'{G}/bank/test3d_shapenet.npz')
    VARIANTS = [
        ('voxel16   (4096d, raw voxels)', f'{G}/bank/bank_voxel16.npz',
         f'{G}/bank/test_voxel16.npz', None, False),
        ('voxel8    (512d, raw voxels)', f'{G}/bank/bank_voxel8.npz',
         f'{G}/bank/test_voxel8.npz', None, False),
        ('full57    (current)', *F57, None, False),
        ('d2only    (32d histogram)', *F57, np.arange(0, 32), False),
        ('shellonly (16d histogram)', *F57, np.arange(32, 48), False),
        ('scalars9  (9 numbers)', *F57, np.arange(48, 57), False),
        ('bbox7     (7 from JSON)', f'{G}/bank/bank_bbox.npz',
         f'{G}/bank/test_bbox.npz', None, False),
        ('SHUFFLED  (neg. control)', *F57, None, True),
    ]
    print(f'{"representation":32s}{"dims":>6s}{"r(margin)":>11s}{"t":>9s}'
          f'{"%neg":>7s}{"r(advantage)":>14s}')
    rows = []
    for name, bf, tf, cols, sh in VARIANTS:
        if not (os.path.exists(bf) and os.path.exists(tf)):
            print(f'{name:32s}  [missing bank]'); continue
        bids, BN, tix, TN = load(bf, tf, cols, sh)
        sp = shift_panel(bids, BN, tix, TN, trials_imgs).merge(
            beh, on=['trial', 'category'])
        r, p, t, pct = evaluate(sp, 'ft_margin')
        ra, pa, ta, pcta = evaluate(sp, 'advantage')
        print(f'{name:32s}{BN.shape[1]:>6d}{r:>+11.3f}{t:>+9.1f}{pct:>6.0f}%'
              f'{ra:>+14.3f}')
        rows.append(dict(representation=name.split()[0], dims=BN.shape[1],
                         r_margin=r, p_margin=p, t_margin=t, pct_neg=pct,
                         r_advantage=ra, p_advantage=pa, t_advantage=ta))
    out = pd.DataFrame(rows)
    out.to_csv(f'{G}/out/variant_representations.csv', index=False)
    print(f'\nwrote {G}/out/variant_representations.csv')


if __name__ == '__main__':
    main()
