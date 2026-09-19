"""
Model-free features that go BEYOND intrinsic 3D shape, motivated by the finding that
rotation-invariant descriptors miss the within-category signal while pose-sensitive
voxels find it. If models learn a proxy for geometry rather than geometry itself, the
proxy is probably "shape as seen from a viewpoint".

All computed from the 128^3 binvox grid. No renderer, no network.

  multiview   Project the grid along K=13 directions (axes, face diagonals, corners).
              For each projected silhouette: area, extent, fill ratio, eccentricity,
              perimeter/compactness.  -> pose-referenced appearance, 13 x 5 = 65 dims.

  volatility  How much the silhouette CHANGES across those viewpoints: the sd across
              views of each silhouette statistic, plus the mean pairwise IoU between
              projections. Low IoU = the object looks very different from different
              angles = a hard multi-view inference. 6 dims.

  structure   Topology and symmetry, which no distance-to-training measure captures:
              connected components, holes (Euler characteristic proxy), and mirror
              symmetry along each canonical axis. 7 dims.
"""
import argparse, os, sys
import numpy as np
from multiprocessing import Pool
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom3d import read_binvox

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
B = f'{NAV}/Dist-shift-data/blend-data'

# 13 viewing directions: 3 axes + 6 face diagonals + 4 corner diagonals
DIRS = np.array([
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=float)
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)


def project(pts, d, n=48):
    """Orthographic projection of occupied voxel coords onto the plane normal to d."""
    a = np.array([0.0, 0.0, 1.0])
    if abs(d @ a) > .9:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    xy = np.stack([pts @ u, pts @ v], 1)
    xy -= xy.min(0)
    rng = xy.max(0)
    rng[rng < 1e-9] = 1.0
    ij = np.clip((xy / rng * (n - 1)).astype(int), 0, n - 1)
    m = np.zeros((n, n), bool)
    m[ij[:, 0], ij[:, 1]] = True
    return m


def sil_stats(m):
    a = float(m.sum())
    if a < 4:
        return [0, 0, 0, 0, 0]
    ys, xs = np.nonzero(m)
    h = np.ptp(ys) + 1; w = np.ptp(xs) + 1
    fill = a / (h * w)
    ecc = min(h, w) / max(h, w)
    per = float((m & ~ndimage.binary_erosion(m)).sum())
    comp = per ** 2 / (4 * np.pi * a)
    return [a / m.size, fill, ecc, comp, per / (a + 1e-9)]


def features(occ):
    idx = np.argwhere(occ).astype(np.float32)
    if idx.shape[0] < 64:
        return None
    idx -= idx.mean(0)
    idx /= (np.sqrt((idx ** 2).sum(1).mean()) + 1e-9)
    mats = [project(idx, d) for d in DIRS]
    S = np.array([sil_stats(m) for m in mats])          # 13 x 5

    # volatility: variation of appearance across viewpoints
    ious = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            inter = (mats[i] & mats[j]).sum(); uni = (mats[i] | mats[j]).sum()
            ious.append(inter / (uni + 1e-9))
    vol = np.concatenate([S.std(0), [np.mean(ious)]])   # 6

    # structure: topology + symmetry
    lab, ncomp = ndimage.label(occ)
    filled = ndimage.binary_fill_holes(occ)
    hole_frac = (filled.sum() - occ.sum()) / (occ.sum() + 1e-9)
    sym = [float((occ & np.flip(occ, ax)).sum() / (occ.sum() + 1e-9)) for ax in range(3)]
    big = (lab == (np.bincount(lab.ravel())[1:].argmax() + 1)).sum() / (occ.sum() + 1e-9)
    struct = np.array([ncomp, hole_frac, *sym, big, occ.mean()])   # 7
    return S.ravel(), vol, struct


def _work(arg):
    oid, root = arg
    try:
        _, occ, _, _ = read_binvox(f'{root}/{oid}/models/model_normalized.solid.binvox')
        f = features(occ)
        if f is None:
            return oid, None
        return oid, f
    except Exception:
        return oid, None


def build(ids, root, tag, workers=64):
    out = {'multiview': ([], []), 'volatility': ([], []), 'structure': ([], [])}
    with Pool(workers) as p:
        for oid, f in p.imap_unordered(_work, [(i, root) for i in ids], chunksize=8):
            if f is None:
                continue
            for k, v in zip(['multiview', 'volatility', 'structure'], f):
                if np.all(np.isfinite(v)):
                    out[k][0].append(oid); out[k][1].append(v.astype(np.float32))
    for k, (kid, X) in out.items():
        X = np.array(X, np.float32)
        np.savez_compressed(f'{G}/bank/{tag}_{k}.npz', ids=np.array(kid), X=X,
                            names=np.array([f'{k}_{i}' for i in range(X.shape[1])]))
        print(f'  {tag}_{k}: {X.shape}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=64)
    a = ap.parse_args()
    bank_ids = list(np.load(f'{G}/bank/bank3d_shapenet_trained.npz',
                            allow_pickle=True)['ids'])
    test_ids = list(np.load(f'{G}/bank/test3d_shapenet.npz', allow_pickle=True)['ids'])
    print('bank:', flush=True)
    build(bank_ids, f'{B}/shapenet_training', 'bank', a.workers)
    print('test:', flush=True)
    build(test_ids, f'{B}/shapenet_mochi_excluded', 'test', a.workers)
