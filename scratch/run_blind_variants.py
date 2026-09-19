"""Rebuild the oddity-blind geometric shift under different object representations.

Everything about the estimator is held fixed -- oddity-blind averaging, k-NN mean,
z-score then L2-normalise, per-category banks. The ONLY thing that changes is which
descriptor bank phi comes from:

  d57      geom3d 57-d summary  -- ROTATION-INVARIANT (pose thrown away)
  voxel8   raw 8^3 occupancy    -- pose-SENSITIVE
  voxel16  raw 16^3 occupancy   -- pose-SENSITIVE
  voxel32  raw 32^3 occupancy   -- pose-SENSITIVE

Writes out/blindshift_shapenet_<rep>_percat.csv in the same schema as the original.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, G)
import blind_shift as bs                                              # noqa: E402

REPS = {'d57':     ('bank3d_shapenet_trained', 'test3d_shapenet'),
        'voxel8':  ('bank_voxel8',  'test_voxel8'),
        'voxel16': ('bank_voxel16', 'test_voxel16'),
        'voxel32': ('bank_voxel32', 'test_voxel32')}


def prep32(bank_npz, test_npz):
    """Same as blind_shift.prep but float32 throughout -- voxel32 is 20885 x 32768,
    which is 5.5 GB in float64 before any copies."""
    bz = np.load(bank_npz, allow_pickle=True)
    tz = np.load(test_npz, allow_pickle=True)
    B = bz['X'].astype(np.float32); T = tz['X'].astype(np.float32)
    mu = B.mean(0); sd = B.std(0); sd[sd < 1e-9] = 1.0
    B -= mu; B /= sd
    T -= mu; T /= sd
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    return list(bz['ids']), B, {k: i for i, k in enumerate(tz['ids'])}, T


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', nargs='+', default=list(REPS))
    a = ap.parse_args()
    bs.prep = prep32                                  # patch in the float32 version
    for rep in a.reps:
        b, t = REPS[rep]
        bp, tp = f'{G}/bank/{b}.npz', f'{G}/bank/{t}.npz'
        nb = np.load(bp, allow_pickle=True)['X'].shape
        print(f'\n=== {rep}: bank {nb[0]:,} x {nb[1]:,} dims ===', flush=True)
        wide, long = bs.build('shapenet', bp, tp, per_category=True)
        long = long.rename(columns={'bank': 'category'})
        wide.drop(columns=['bank']).to_csv(f'{G}/out/blindshift_shapenet_{rep}.csv',
                                           index=False)
        long.to_csv(f'{G}/out/blindshift_shapenet_{rep}_percat.csv', index=False)
        print(f'  {len(wide)} trials, per-cat panel {len(long)} rows '
              f'({long.category.nunique()} categories)', flush=True)
        print(f'  blind_k50: mean {long.blind_k50.mean():.4f} '
              f'sd {long.blind_k50.std():.4f}', flush=True)
