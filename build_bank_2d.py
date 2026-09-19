"""
Compute silhouette descriptors for a render bank or for the MOCHI test images.

Specs:
  shapegen_bank  : Dist-shift-data/shapegen/base_images/extrusions_K/smoothness_S/<obj>/NNN.png
                   RGBA -> alpha silhouette. Subsampled: --objs per stratum, --views per object.
  shapenet_bank  : Dist-shift-data/shapenet_rendered/no_background/<cat>/<obj>/NNN.png
                   RGBA -> alpha silhouette. All 15 views. MOCHI-colliding ids excluded.
  mochi          : MOCHI/images/*.png  (shapegen + shapenet subsets only)
                   RGB -> border-flood-fill silhouette. Kept PER IMAGE, because a trial
                   needs phi(A) = mean over the matched views and phi(B) = the oddity view.

Bank objects are stored per-object as the mean descriptor over their views, matching
the paper's phi(C_i) = mean over all rendered viewpoints (Sec 3.5).
"""
import argparse, os, sys, time, glob, ast
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom2d import silhouette, descriptor, DIM, NAMES

P = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
WHITE = (255, 255, 255)


def _mochi_ids():
    return set(l.strip() for l in open(f'{P}/mochi/mochi_shapenet_object_ids.txt') if l.strip())


def tasks_shapegen_bank(n_objs, n_views):
    base = f'{P}/shapegen/base_images'
    out = []
    for K in sorted(os.listdir(base)):
        if not K.startswith('extrusions_'):
            continue
        for S in sorted(os.listdir(f'{base}/{K}')):
            objs = sorted(os.listdir(f'{base}/{K}/{S}'))[:n_objs]
            for o in objs:
                fs = sorted(glob.glob(f'{base}/{K}/{S}/{o}/*.png'))
                if not fs:
                    continue
                step = max(1, len(fs) // n_views)
                for f in fs[::step][:n_views]:
                    out.append((f'{K}/{S}/{o}', f, None))
    return out


def tasks_shapenet_bank(n_objs, n_views):
    base = f'{P}/shapenet_rendered/no_background'
    mo = {i.split('/')[1] for i in _mochi_ids()}
    out = []
    for cat in sorted(os.listdir(base)):
        d = f'{base}/{cat}'
        if not os.path.isdir(d):
            continue
        objs = [o for o in sorted(os.listdir(d)) if o not in mo][:n_objs] if n_objs \
            else [o for o in sorted(os.listdir(d)) if o not in mo]
        for o in objs:
            fs = sorted(glob.glob(f'{d}/{o}/*.png'))
            for f in fs[:n_views] if n_views else fs:
                out.append((f'{cat}/{o}', f, None))
    return out


def tasks_mochi():
    df = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    keep = df[df.dataset.isin(['shapegen', 'shapenet'])]
    imgs = sorted({f for s in keep.images for f in ast.literal_eval(s)})
    return [(f, f'{MOCHI}/images/{f}', WHITE) for f in imgs]


def _work(arg):
    key, path, bg = arg
    try:
        d = descriptor(silhouette(path, bg=bg))
        if d is None or not np.all(np.isfinite(d)):
            return key, None
        return key, d.astype(np.float32)
    except Exception:
        return key, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True,
                    choices=['shapegen_bank', 'shapenet_bank', 'mochi'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--objs', type=int, default=0)
    ap.add_argument('--views', type=int, default=0)
    ap.add_argument('--per-image', action='store_true',
                    help='store one row per image instead of averaging per object')
    a = ap.parse_args()

    if a.spec == 'shapegen_bank':
        tasks = tasks_shapegen_bank(a.objs or 800, a.views or 16)
    elif a.spec == 'shapenet_bank':
        tasks = tasks_shapenet_bank(a.objs, a.views)
    else:
        tasks = tasks_mochi()
        a.per_image = True

    print(f'[{time.strftime("%H:%M:%S")}] spec={a.spec} tasks={len(tasks)} '
          f'per_image={a.per_image}', flush=True)

    acc, n_fail, t0 = {}, 0, time.time()
    with Pool(a.workers) as pool:
        for k, (key, d) in enumerate(pool.imap_unordered(_work, tasks, chunksize=16), 1):
            if d is None:
                n_fail += 1
            else:
                acc.setdefault(key, []).append(d)
            if k % 20000 == 0 or k == len(tasks):
                el = time.time() - t0
                print(f'[{time.strftime("%H:%M:%S")}] {k}/{len(tasks)} fail={n_fail} '
                      f'{k/el:.0f} img/s ETA {(len(tasks)-k)/(k/el)/60:.1f} min', flush=True)

    ids = sorted(acc)
    X = np.array([np.mean(acc[i], axis=0) for i in ids], np.float32)
    np.savez_compressed(a.out, ids=np.array(ids), X=X, names=np.array(NAMES))
    print(f'[{time.strftime("%H:%M:%S")}] wrote {a.out} shape={X.shape} '
          f'failed={n_fail} total {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
