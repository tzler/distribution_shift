"""
Oddity-blind, hub-robust, norm-free MODEL-FREE GEOMETRIC distribution shift.

    shift(t) = (1/|t|) sum_{x in t}  (1/k) sum_{i in kNN_k(x)} ( 1 - g_hat(x) . C_hat_i )

g_hat(x) : L2-normalised geometric descriptor of trial image x
           shapenet -> 3D voxel shape descriptor of the object x depicts
           shapegen -> 2D silhouette descriptor of the image itself
C_hat_i  : L2-normalised descriptors of the training bank (MOCHI objects excluded)
k        : 50 by default

The three design choices, transplanted from the encoder version:
  1. ODDITY-BLIND. Average a per-image quantity over ALL images of the trial; never
     form a difference between them. The shared-neighbour form min_C 1/2[d(A,C)+d(B,C)]
     is bounded below by 1/2 d(A,B) -- the oddity task's own decision variable -- so
     trial difficulty leaks into the shift. Averaging removes that bound structurally.
  2. HUB-ROBUST. Mean over the k nearest training objects, not a single argmin.
  3. NORM-FREE. Cosine distance on L2-normalised descriptors, so the measure cannot
     simply track descriptor magnitude.

Descriptors are z-scored against the bank first (the raw dimensions are histogram
densities and unitless ratios on wildly different scales), then L2-normalised.

NOTE ON THE RULER. The encoder version needs a "neutral ruler" caveat: if f_hat comes
from the model being evaluated, fine-tuning reshapes the space in which shift is
measured and the sign inverts. A geometric ruler cannot be reshaped by training --
it is neutral by construction, for every model and every training stage.
"""
import argparse, ast, os
import numpy as np
import pandas as pd

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
MOCHI = f'{NAV}/MOCHI'
KS = (1, 10, 50, 200)
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}


def prep(bank_npz, test_npz):
    """z-score against the bank, then L2-normalise both sides."""
    bz = np.load(bank_npz, allow_pickle=True)
    tz = np.load(test_npz, allow_pickle=True)
    B, T = bz['X'].astype(float), tz['X'].astype(float)
    mu, sd = B.mean(0), B.std(0)
    sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd
    BN = BZ / (np.linalg.norm(BZ, axis=1, keepdims=True) + 1e-12)
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    return list(bz['ids']), BN, {k: i for i, k in enumerate(tz['ids'])}, TN


def knn_mean(TN, BN, ks):
    """Per test item: mean cosine distance to its k nearest bank items."""
    D = 1.0 - TN @ BN.T
    kmax = min(max(ks), D.shape[1])
    part = np.partition(D, kmax - 1, axis=1)[:, :kmax]
    part = np.sort(part, axis=1)
    return {k: part[:, :min(k, part.shape[1])].mean(1) for k in ks}


def trial_keys(dataset, images):
    """Map each trial image to the key used in the descriptor table."""
    if dataset == 'shapenet':                       # object-level (3D descriptors)
        return ['/'.join(f[:-4].split('_')[:2]) for f in images]
    return list(images)                             # image-level (2D silhouettes)


def build(dataset, bank_npz, test_npz, per_category=False):
    bids, BN, tix, TN = prep(bank_npz, test_npz)
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == dataset]

    banks = {'ALL': np.ones(len(bids), bool)}
    if per_category:
        syn = np.array([i.split('/')[0] for i in bids])
        for cat, s in SYN.items():
            sel = syn == s
            if sel.sum() >= max(KS):
                banks[cat] = sel

    out_wide, out_long = [], []
    for name, sel in banks.items():
        per = knn_mean(TN, BN[sel], KS)              # per test item, per k
        for _, r in mm.iterrows():
            keys = trial_keys(dataset, ast.literal_eval(r['images']))
            if not all(k in tix for k in keys):
                continue
            rows = [tix[k] for k in keys]
            rec = {'trial': r['trial'], 'bank': name}
            for k in KS:                             # ODDITY-BLIND: mean over images
                rec[f'blind_k{k}'] = float(np.mean(per[k][rows]))
            (out_wide if name == 'ALL' else out_long).append(rec)
    return pd.DataFrame(out_wide), pd.DataFrame(out_long)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['shapenet', 'shapegen'])
    a = ap.parse_args()
    if a.dataset == 'shapenet':
        bank, test, percat = (f'{G}/bank/bank3d_shapenet_trained.npz',
                              f'{G}/bank/test3d_shapenet.npz', True)
    else:
        bank, test, percat = (f'{G}/bank/bank2d_shapegen.npz',
                              f'{G}/bank/mochi2d.npz', False)
    wide, long = build(a.dataset, bank, test, percat)
    wide = wide.drop(columns=['bank'])
    wide.to_csv(f'{G}/out/blindshift_{a.dataset}.csv', index=False)
    print(f'{a.dataset}: {len(wide)} trials -> out/blindshift_{a.dataset}.csv')
    print(wide[[f'blind_k{k}' for k in KS]].describe().loc[['mean', 'std', 'min', 'max']]
          .round(4).to_string())
    if len(long):
        long = long.rename(columns={'bank': 'category'})
        long.to_csv(f'{G}/out/blindshift_{a.dataset}_percat.csv', index=False)
        print(f'  per-category panel: {len(long)} rows, '
              f'{long.category.nunique()} categories -> '
              f'out/blindshift_{a.dataset}_percat.csv')
