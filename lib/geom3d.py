"""
Model-free 3D shape descriptors from ShapeNet .solid.binvox voxelizations.

No learned parameters, no network, no rendering. Every descriptor is invariant to
translation and scale; the histogram-based ones (D2, shell) are additionally
invariant to rotation.

Representation choice: ShapeNet ships `model_normalized.solid.binvox`, a 128^3 solid
voxelization (~31 KB) alongside `model_normalized.obj` (~1.7 MB). The voxel grid is
54x smaller, already canonical, and needs no mesh library -- which matters because
trimesh/open3d are not installed anywhere on this cluster.
"""
import numpy as np
from scipy.spatial import ConvexHull

D2_BINS = 32
SHELL_BINS = 16
N_PAIRS = 200_000
N_PTS = 4096


def read_binvox(path):
    """Parse a binvox file -> (dims, occupancy bool array in xyz order).

    Format: ascii header ('#binvox 1', dim/translate/scale lines, 'data'),
    then run-length encoded byte pairs (value, count). Raw voxel order is
    x-major then z then y, so we transpose (0,2,1) to get xyz.
    """
    with open(path, 'rb') as f:
        line = f.readline().strip()
        if not line.startswith(b'#binvox'):
            raise ValueError(f'not a binvox file: {path}')
        dims = translate = scale = None
        while True:
            line = f.readline().strip()
            if line.startswith(b'dim'):
                dims = tuple(int(v) for v in line.split()[1:])
            elif line.startswith(b'translate'):
                translate = tuple(float(v) for v in line.split()[1:])
            elif line.startswith(b'scale'):
                scale = float(line.split()[1])
            elif line.startswith(b'data'):
                break
            elif line == b'':
                raise ValueError(f'truncated header: {path}')
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    values, counts = raw[::2], raw[1::2]
    n = min(len(values), len(counts))
    occ = np.repeat(values[:n], counts[:n]).astype(bool)
    total = int(np.prod(dims))
    if occ.size < total:                      # tolerate short final run
        occ = np.concatenate([occ, np.zeros(total - occ.size, bool)])
    occ = occ[:total].reshape(dims)
    return dims, np.transpose(occ, (0, 2, 1)), translate, scale


def _normalize(pts):
    """Center on centroid, scale to unit RMS radius. Translation+scale invariant."""
    c = pts.mean(0)
    q = pts - c
    rms = np.sqrt((q ** 2).sum(1).mean())
    return q / rms if rms > 0 else q


def descriptor(occ, rng):
    """Return a fixed-length rotation/translation/scale-invariant shape vector."""
    idx = np.argwhere(occ)
    if idx.shape[0] < 64:
        return None
    pts_all = idx.astype(np.float32)

    # ---- subsample occupied voxels (volume points) -------------------------
    k = min(N_PTS, pts_all.shape[0])
    sel = rng.choice(pts_all.shape[0], k, replace=False)
    P = _normalize(pts_all[sel])

    # ---- D2: histogram of pairwise distances (Osada et al.) ---------------
    i = rng.integers(0, k, N_PAIRS)
    j = rng.integers(0, k, N_PAIRS)
    d = np.linalg.norm(P[i] - P[j], axis=1)
    d2, _ = np.histogram(d, bins=D2_BINS, range=(0.0, 4.0), density=True)

    # ---- shell: histogram of centroid distances ---------------------------
    r = np.linalg.norm(P, axis=1)
    shell, _ = np.histogram(r, bins=SHELL_BINS, range=(0.0, 3.0), density=True)

    # ---- inertia eigenvalues -> elongation / flatness ----------------------
    ev = np.sort(np.linalg.eigvalsh(np.cov(P.T)))[::-1]
    ev = ev / (ev.sum() + 1e-12)
    elong, flat = ev[1] / (ev[0] + 1e-12), ev[2] / (ev[0] + 1e-12)

    # ---- convexity and compactness (scale-free ratios) --------------------
    try:
        hull_vol = ConvexHull(P).volume
    except Exception:
        hull_vol = np.nan
    vox_vol = pts_all.shape[0]
    # surface voxels: occupied voxels with a non-occupied 6-neighbour
    pad = np.pad(occ, 1)
    nb = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] &
          pad[1:-1, :-2, 1:-1] & pad[1:-1, 2:, 1:-1] &
          pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    n_surf = float((occ & ~nb).sum())
    # unit-RMS-normalized volume: how much space the shape occupies at fixed spread
    rms_scale = np.sqrt(((pts_all - pts_all.mean(0)) ** 2).sum(1).mean())
    norm_vol = vox_vol / (rms_scale ** 3 + 1e-12)
    compact = n_surf / (vox_vol ** (2.0 / 3.0) + 1e-12)
    convexity = (vox_vol / (rms_scale ** 3)) / (hull_vol + 1e-12) if hull_vol == hull_vol else np.nan

    extra = np.array([elong, flat, compact, convexity, norm_vol,
                      r.mean(), r.std(), d.mean(), d.std()], dtype=np.float64)
    return np.concatenate([d2, shell, extra])


DIM = D2_BINS + SHELL_BINS + 9
NAMES = ([f'd2_{i}' for i in range(D2_BINS)] +
         [f'shell_{i}' for i in range(SHELL_BINS)] +
         ['elong', 'flat', 'compact', 'convexity', 'norm_vol',
          'r_mean', 'r_std', 'd2_mean', 'd2_std'])


def describe_path(path, seed=0):
    _, occ, _, _ = read_binvox(path)
    return descriptor(occ, np.random.default_rng(seed))
