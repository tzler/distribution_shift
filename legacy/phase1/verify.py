"""
Sanity checks for the geometric shift pipeline. Run after any change to the descriptors.

1. train/test object disjointness (guards the degenerate Delta ~ 0 failure)
2. output tables join 1:1 with the trials table and contain no NaN
3. 3D descriptor is invariant to voxel resolution (128^3 vs downsampled 64^3)
4. 2D descriptor is invariant to image resolution (128 / 256 / 512 px)
5. regression test: the descriptor recovers shapegen's known generative parameter K
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from geom3d import read_binvox, descriptor as d3
from geom2d import silhouette, descriptor as d2
from PIL import Image

P = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


print('1. train/test object disjointness')
mochi = set(l.strip() for l in open(f'{P}/mochi/mochi_shapenet_object_ids.txt') if l.strip())
tr = set(np.load(f'{G}/bank/bank3d_shapenet.npz', allow_pickle=True)['ids'])
te = set(np.load(f'{G}/bank/test3d_shapenet.npz', allow_pickle=True)['ids'])
check('training bank shares no object with MOCHI', len(tr & mochi) == 0,
      f'|overlap|={len(tr & mochi)}, |bank|={len(tr)}')
check('test descriptors are all MOCHI objects', te <= mochi, f'|test|={len(te)}')

print('\n2. output tables')
m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
for f, ds, n_exp in [('shift3d_shapenet.csv', 'shapenet', 706),
                     ('shift2d_shapenet.csv', 'shapenet', 865),
                     ('shift2d_shapegen.csv', 'shapegen', 548)]:
    d = pd.read_csv(f'{G}/out/{f}')
    j = d.merge(m[m.dataset == ds][['trial']], on='trial', validate='1:1')
    num = d.select_dtypes(include=[np.number])
    check(f'{f}: n={len(d)}, joins 1:1, no NaN',
          len(d) == n_exp and len(j) == len(d) and not num.isna().any().any())

print('\n3. 3D descriptor invariance to voxel resolution')
fs = sorted(glob.glob(f'{P}/blend-data/shapenet_training/02691156/*/models/*.solid.binvox'))[:12]
rng = np.random.default_rng(0)
A, B = [], []
for f in fs:
    _, occ, _, _ = read_binvox(f)
    half = occ[::2, ::2, ::2]                      # 64^3 downsample
    a, b = d3(occ, np.random.default_rng(0)), d3(half, np.random.default_rng(0))
    if a is not None and b is not None:
        A.append(a); B.append(b)
A, B = np.array(A), np.array(B)
sd = A.std(0) + 1e-9
rel = np.abs(A - B).mean(0) / sd
r = stats.pearsonr(A.ravel(), B.ravel())[0]
check('128^3 vs 64^3 descriptors agree', r > 0.99,
      f'r={r:.4f}, mean|delta|={rel.mean():.3f} sd units')

print('\n4. 2D descriptor invariance to image resolution')
fs = sorted(glob.glob(f'{P}/shapegen/base_images/extrusions_4/smoothness_0/*/0*.png'))[:200:10]
V = {}
for size in (128, 256, 512):
    rows = []
    for f in fs:
        im = Image.open(f)
        a = np.asarray(im.split()[-1].resize((size, size), Image.BILINEAR)) > 127
        v = d2(a)
        if v is not None:
            rows.append(v)
    V[size] = np.array(rows)
r_all = []
for s in (128, 512):
    n = min(len(V[256]), len(V[s]))
    r_all.append(stats.pearsonr(V[256][:n].ravel(), V[s][:n].ravel())[0])
check('descriptor stable across 128/256/512 px', min(r_all) > 0.98,
      f'r(256,128)={r_all[0]:.4f}  r(256,512)={r_all[1]:.4f}')

print('\n5. regression test: descriptor recovers shapegen K (extrusion count)')
X, y = [], []
for K in (2, 3, 4, 5, 6, 7, 8, 9):
    for f in sorted(glob.glob(
            f'{P}/shapegen/base_images/extrusions_{K}/smoothness_0/*/0*.png'))[:900:60]:
        v = d2(silhouette(f))
        if v is not None:
            X.append(v); y.append(K)
X, y = np.array(X), np.array(y)
sol = abs(stats.pearsonr(X[:, 1], y)[0])      # solidity
com = abs(stats.pearsonr(X[:, 2], y)[0])      # compactness
check('solidity tracks K', sol > 0.4, f'|r|={sol:.3f} (n={len(X)})')
check('compactness tracks K', com > 0.4, f'|r|={com:.3f}')

print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
sys.exit(0 if ok else 1)
