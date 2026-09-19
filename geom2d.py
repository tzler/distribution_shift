"""
Model-free 2D silhouette shape descriptors.

Used for (a) shapegen, where no meshes/voxelizations exist (only .blend files), and
(b) shapenet, where it can be cross-checked against the true 3D voxel metric on the
same objects -- that agreement is what licenses the 2D metric for shapegen.

Every descriptor is invariant to translation, rotation and SCALE. Scale-invariance is
mandatory here: MOCHI images are 1000x1000, shapegen renders 518x518, shapenet
renders 224x224, and camera radius is scaled to object size, so absolute extent is
not comparable across pipelines. It also keeps the descriptor from degenerating into
a restatement of `coverage`, which is the confound that sinks pix_l1 in the audit.

Silhouette extraction:
  * RGBA renders  -> alpha channel is the exact silhouette (free).
  * MOCHI RGB     -> border-connected-component flood fill. NOT thresholding: objects
                     reach gray 255 against a white ground, so a threshold would eat
                     specular highlights and punch holes in the mask.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

N_ANG = 64
N_FFT = 12
WORK = 256


def silhouette(path, bg=None, size=WORK):
    """Return a boolean silhouette mask at `size` x `size`.

    bg=None  -> use the alpha channel (RGBA renders)
    bg=(r,g,b) -> flood fill the border-connected background of that colour
    """
    im = Image.open(path)
    if bg is None:
        if im.mode != 'RGBA':
            raise ValueError(f'expected RGBA for alpha silhouette: {path}')
        a = np.asarray(im.split()[-1].resize((size, size), Image.BILINEAR))
        return a > 127

    rgb = np.asarray(im.convert('RGB').resize((size, size), Image.BILINEAR)).astype(np.int16)
    isbg = (np.abs(rgb - np.array(bg, np.int16)).max(axis=2) <= 12)
    lab, n = ndimage.label(isbg)
    if n == 0:
        return np.ones((size, size), bool)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    if not border:
        return ~isbg
    outside = np.isin(lab, list(border))
    return ~outside


def descriptor(m):
    """Scale/translation/rotation-invariant silhouette shape vector, or None."""
    ys, xs = np.nonzero(m)
    A = float(xs.size)
    if A < 64:
        return None
    cy, cx = ys.mean(), xs.mean()

    # boundary pixels (4-connectivity erosion)
    er = ndimage.binary_erosion(m)
    bnd = m & ~er
    by, bx = np.nonzero(bnd)
    per = float(by.size)
    if per < 16:
        return None

    # --- inertia tensor -> eccentricity (rotation-invariant) ---------------
    y, x = ys - cy, xs - cx
    m20, m02, m11 = (x * x).mean(), (y * y).mean(), (x * y).mean()
    ev = np.sort(np.linalg.eigvalsh(np.array([[m20, m11], [m11, m02]])))[::-1]
    ecc = float(np.sqrt(max(ev[0] - ev[1], 0.0) / ev[0])) if ev[0] > 0 else 0.0

    # --- solidity + compactness -------------------------------------------
    try:
        hull = ConvexHull(np.stack([bx, by], 1).astype(float)).volume  # 2D: area
        solidity = A / hull if hull > 0 else np.nan
    except Exception:
        solidity = np.nan
    compact = (per * per) / (4.0 * np.pi * A)

    # --- radial profile over N_ANG angular bins (vectorised) --------------
    r = np.hypot(by - cy, bx - cx) / np.sqrt(A / np.pi)   # scale-normalised
    ang = np.arctan2(by - cy, bx - cx)
    b = np.clip(((ang + np.pi) / (2 * np.pi) * N_ANG).astype(np.int64), 0, N_ANG - 1)
    prof = np.zeros(N_ANG)
    np.maximum.at(prof, b, r)                              # max radius per sector
    filled = prof > 0
    if filled.sum() < N_ANG // 2:
        return None
    if not filled.all():                                   # interpolate empty sectors
        idx = np.arange(N_ANG)
        prof[~filled] = np.interp(idx[~filled], idx[filled], prof[filled], period=N_ANG)

    mu = prof.mean()
    fft = np.abs(np.fft.rfft(prof - mu))[1:1 + N_FFT] / (mu + 1e-9)
    radial_cv = prof.std() / (mu + 1e-9)

    # --- turning-angle / curvature roughness ------------------------------
    d1 = np.diff(np.concatenate([prof, prof[:1]]))
    rough = float(np.abs(d1).mean() / (mu + 1e-9))

    return np.concatenate([
        np.array([ecc, solidity, compact, radial_cv, rough, mu, prof.min() / (mu + 1e-9),
                  prof.max() / (mu + 1e-9)]),
        fft,
    ])


DIM = 8 + N_FFT
NAMES = (['ecc', 'solidity', 'compact', 'radial_cv', 'rough', 'radial_mean',
          'radial_min', 'radial_max'] + [f'fft{i+1}' for i in range(N_FFT)])


def describe(path, bg=None):
    m = silhouette(path, bg=bg)
    return descriptor(m)
