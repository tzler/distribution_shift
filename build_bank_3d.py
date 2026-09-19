"""
Compute model-free 3D shape descriptors for a directory of ShapeNet objects.

Usage:
    python build_bank_3d.py --root <shapenet-style dir> --out <file.npz> [--workers 64]

The root is expected to be laid out as <root>/<synset>/<objectid>/models/*.solid.binvox
(this is how blend-data/shapenet_training and blend-data/shapenet_mochi_excluded are
organised). Writes an npz with `ids` (synset/objectid) and `X` (n x DIM descriptors).

Progress is checkpointed every CHUNK objects so a killed job loses at most one chunk.
"""
import argparse, os, sys, time, glob
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom3d import describe_path, DIM, NAMES

CHUNK = 2000


def find_objects(root):
    """Return sorted [(id, binvox_path)] for every object under root."""
    out = []
    for syn in sorted(os.listdir(root)):
        sd = os.path.join(root, syn)
        if not os.path.isdir(sd):
            continue
        for obj in sorted(os.listdir(sd)):
            p = os.path.join(sd, obj, 'models', 'model_normalized.solid.binvox')
            if os.path.exists(p):
                out.append((f'{syn}/{obj}', p))
    return out


def _work(arg):
    oid, path = arg
    try:
        d = describe_path(path, seed=0)
        if d is None or not np.all(np.isfinite(d)):
            return oid, None
        return oid, d.astype(np.float32)
    except Exception:
        return oid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    objs = find_objects(a.root)
    if a.limit:
        objs = objs[:a.limit]
    print(f'[{time.strftime("%H:%M:%S")}] {a.root}: {len(objs)} objects with binvox',
          flush=True)

    ids, X, n_fail = [], [], 0
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (oid, d) in enumerate(pool.imap_unordered(_work, objs, chunksize=8), 1):
            if d is None:
                n_fail += 1
            else:
                ids.append(oid)
                X.append(d)
            if k % CHUNK == 0 or k == len(objs):
                el = time.time() - t0
                rate = k / el
                eta = (len(objs) - k) / rate / 60
                print(f'[{time.strftime("%H:%M:%S")}] {k}/{len(objs)} '
                      f'ok={len(ids)} fail={n_fail} {rate:.1f} obj/s ETA {eta:.1f} min',
                      flush=True)
                np.savez_compressed(a.out + '.partial.npz',
                                    ids=np.array(ids), X=np.array(X, np.float32),
                                    names=np.array(NAMES))

    X = np.array(X, np.float32)
    np.savez_compressed(a.out, ids=np.array(ids), X=X, names=np.array(NAMES))
    p = a.out + '.partial.npz'
    if os.path.exists(p):
        os.remove(p)
    print(f'[{time.strftime("%H:%M:%S")}] wrote {a.out}  shape={X.shape} '
          f'failed={n_fail}  total {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
