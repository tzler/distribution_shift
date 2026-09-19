"""
Alternative model-free object representations, to test whether the result depends on
the particular 57-d descriptor we started with.

  voxel16   raw 16^3 occupancy grid (4096 dims) — downsampled binvox, NO designed features
  voxel8    raw 8^3 occupancy grid (512 dims)
  bbox      7 numbers from ShapeNet's model_normalized.json (extents, aspect ratios,
            volume, vertex count) — about the crudest shape summary available

Written as npz banks in the same format as bank3d_shapenet_trained.npz so the existing
blind-shift machinery can consume them unchanged.
"""
import argparse, json, os, sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom3d import read_binvox

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
B = f'{NAV}/Dist-shift-data/blend-data'


def voxel_feats(occ, side):
    """Mean occupancy in side^3 blocks of the 128^3 grid."""
    f = occ.shape[0] // side
    return occ.reshape(side, f, side, f, side, f).mean(axis=(1, 3, 5)).ravel()


def bbox_feats(jpath):
    j = json.load(open(jpath))
    mn, mx = np.array(j['min'], float), np.array(j['max'], float)
    ext = np.sort(mx - mn)[::-1]                     # sorted -> rotation-order invariant
    ext = ext / (ext[0] + 1e-12)                     # scale invariant: 1, mid, small
    vol = float(np.prod(mx - mn))
    cen = np.array(j.get('centroid', [0, 0, 0]), float)
    off = float(np.linalg.norm(cen - (mn + mx) / 2) / (np.linalg.norm(mx - mn) + 1e-12))
    nv = float(np.log10(max(j.get('numVertices', 1), 1)))
    return np.array([ext[1], ext[2], ext[1] * ext[2], vol ** (1 / 3), off, nv,
                     float(np.log10(vol + 1e-9))])


def _work(arg):
    oid, root, kind = arg
    try:
        if kind == 'bbox':
            return oid, bbox_feats(f'{root}/{oid}/models/model_normalized.json')
        _, occ, _, _ = read_binvox(f'{root}/{oid}/models/model_normalized.solid.binvox')
        return oid, voxel_feats(occ, {'voxel32': 32, 'voxel16': 16}.get(kind, 8))
    except Exception:
        return oid, None


def build(kind, ids, root, out, workers=64):
    tasks = [(i, root, kind) for i in ids]
    keep, X = [], []
    with Pool(workers) as p:
        for oid, v in p.imap_unordered(_work, tasks, chunksize=16):
            if v is not None and np.all(np.isfinite(v)):
                keep.append(oid); X.append(v.astype(np.float32))
    X = np.array(X, np.float32)
    np.savez_compressed(out, ids=np.array(keep), X=X,
                        names=np.array([f'{kind}_{i}' for i in range(X.shape[1])]))
    print(f'  {kind:8s} -> {out}  {X.shape}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=64)
    a = ap.parse_args()
    bank_ids = list(np.load(f'{G}/bank/bank3d_shapenet_trained.npz',
                            allow_pickle=True)['ids'])
    test_ids = list(np.load(f'{G}/bank/test3d_shapenet.npz', allow_pickle=True)['ids'])
    for kind in ['voxel32']:
        print(f'{kind}:', flush=True)
        build(kind, bank_ids, f'{B}/shapenet_training',
              f'{G}/bank/bank_{kind}.npz', a.workers)
        build(kind, test_ids, f'{B}/shapenet_mochi_excluded',
              f'{G}/bank/test_{kind}.npz', a.workers)
