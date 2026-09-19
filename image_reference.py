"""
An IMAGE-based training reference, with spatially-resolved silhouettes.

Two corrections to what came before:

  1. The reference should be the images the model actually saw, not the 3D objects
     they came from. The model was trained on 311,370 renders; the training
     distribution is a distribution over IMAGES.
  2. The earlier 2D bank averaged each object's 15 views into one descriptor, which
     destroys the viewpoint structure. Here every image is its own reference point.

Also: keep the silhouette as a MAP rather than collapsing it to summary statistics.
The pose-feature experiment showed that summarising 13 projections into 5 numbers each
threw away whatever made raw voxels work, so we keep spatial layout.

  representation:  32x32 binary silhouette, centred and scale-normalised -> 1024 dims

Silhouettes:
  training renders  shapenet_rendered/no_background/*  (RGBA -> alpha is exact)
  MOCHI test images MOCHI/images/*.png (uniform white ground -> border flood fill)
Both are background-invariant by construction, which is what makes the two rendering
pipelines comparable at all.
"""
import argparse, ast, os, sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
REND = f'{NAV}/Dist-shift-data/shapenet_rendered/no_background'
MOCHI = f'{NAV}/MOCHI'
MOCHI_IDS = f'{NAV}/Dist-shift-data/mochi/mochi_shapenet_object_ids.txt'
N = 32


def silhouette_map(path, bg=None):
    """Centred, scale-normalised NxN binary silhouette."""
    im = Image.open(path)
    if bg is None:
        a = np.asarray(im.split()[-1].resize((160, 160), Image.BILINEAR)) > 127
    else:
        rgb = np.asarray(im.convert('RGB').resize((160, 160), Image.BILINEAR)).astype(np.int16)
        isbg = np.abs(rgb - np.array(bg, np.int16)).max(2) <= 12
        lab, n = ndimage.label(isbg)
        if n == 0:
            a = np.ones((160, 160), bool)
        else:
            border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
            border.discard(0)
            a = ~np.isin(lab, list(border)) if border else ~isbg
    ys, xs = np.nonzero(a)
    if xs.size < 30:
        return None
    # crop to the object's bounding box, then resample to N x N -> scale + position free
    sub = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = sub.shape
    s = max(h, w)
    pad = np.zeros((s, s), bool)                       # square-pad to keep aspect ratio
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = sub
    small = np.asarray(Image.fromarray(pad.astype(np.uint8) * 255).resize(
        (N, N), Image.BILINEAR)) > 127
    return small.ravel().astype(np.float32)


def _work(arg):
    key, path, bg = arg
    try:
        v = silhouette_map(path, bg)
        return (key, v) if v is not None else (key, None)
    except Exception:
        return key, None


def build_train(workers, per_obj):
    mo = {i.strip().split('/')[1] for i in open(MOCHI_IDS) if i.strip()}
    tasks = []
    for cat in sorted(os.listdir(REND)):
        d = f'{REND}/{cat}'
        if not os.path.isdir(d):
            continue
        for o in sorted(os.listdir(d)):
            if o in mo:
                continue                                   # never reference a MOCHI object
            ims = sorted(x for x in os.listdir(f'{d}/{o}') if x.endswith('.png'))
            for im in ims[:per_obj] if per_obj else ims:
                tasks.append((f'{cat}/{o}/{im}', f'{d}/{o}/{im}', None))
    print(f'training images: {len(tasks)}', flush=True)
    return tasks


def build_test():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    keep = m[m.dataset == 'shapenet']
    imgs = sorted({f for s in keep.images for f in ast.literal_eval(s)})
    print(f'MOCHI test images: {len(imgs)}', flush=True)
    return [(f, f'{MOCHI}/images/{f}', (255, 255, 255)) for f in imgs]


def run(tasks, out, workers):
    ids, X, fail = [], [], 0
    with Pool(workers) as p:
        for k, (key, v) in enumerate(p.imap_unordered(_work, tasks, chunksize=32), 1):
            if v is None:
                fail += 1
            else:
                ids.append(key); X.append(v)
            if k % 50000 == 0:
                print(f'  {k}/{len(tasks)}', flush=True)
    X = np.array(X, np.float32)
    np.savez_compressed(out, ids=np.array(ids), X=X)
    print(f'wrote {out}  {X.shape}  failed={fail}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--per-obj', type=int, default=0, help='0 = all 15 views')
    a = ap.parse_args()
    run(build_test(), f'{G}/bank/imgtest_sil{N}.npz', a.workers)
    run(build_train(a.workers, a.per_obj), f'{G}/bank/imgbank_sil{N}.npz', a.workers)
