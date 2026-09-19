"""Depth-from-voxels: a model-free, view-conditioned image descriptor.

Replicates the training renderer's camera (pyrender PerspectiveCamera yfov=pi/3, look-at
from radius 2 with Y-up, mesh centred and scaled to unit max-norm) on the 128^3 solid
voxel grid, and returns a small depth map + silhouette. Numpy only.
"""
import numpy as np, pickle, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geom3d import read_binvox
YFOV = np.pi / 3.0

def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    eye, target, up = (np.array(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6: r = np.cross(f, np.array([1., 0, 0]))
    r /= np.linalg.norm(r); u = np.cross(r, f)
    M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M

def voxel_points(binvox_path, perm=(0, 1, 2), flip=(1, 1, 1)):
    """Occupied voxel centres, centred and scaled to unit max-norm like the renderer's mesh."""
    _, occ, _, _ = read_binvox(binvox_path)
    P = np.argwhere(occ).astype(np.float64) + 0.5
    P = P[:, list(perm)] * np.array(flip, float)
    P -= P.mean(0)
    P /= np.linalg.norm(P, axis=1).max() + 1e-12
    return P

def sphere_eye(theta, phi, radius=2.0):
    return np.array([radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)])

def project(P, eye, res=32, yfov=YFOV):
    """Perspective-project points; return (depth map [res,res], silhouette). Depth = 0 where empty."""
    M = look_at(eye); R = M[:3, :3].T; t = -R @ M[:3, 3]
    C = P @ R.T + t                                # camera frame: -z forward
    z = -C[:, 2]; ok = z > 1e-6; C, z = C[ok], z[ok]
    f = 1.0 / np.tan(yfov / 2)
    x = (C[:, 0] / z) * f; y = (C[:, 1] / z) * f    # NDC in [-1,1]
    px = ((x + 1) / 2 * res).astype(int); py = ((1 - y) / 2 * res).astype(int)
    keep = (px >= 0) & (px < res) & (py >= 0) & (py < res)
    px, py, z = px[keep], py[keep], z[keep]
    depth = np.full(res * res, np.inf); idx = py * res + px
    np.minimum.at(depth, idx, z)
    depth = depth.reshape(res, res); sil = np.isfinite(depth)
    depth = np.where(sil, depth, 0.0)
    return depth, sil

def descriptor(depth, sil):
    """Normalised depth map descriptor: silhouette-masked, depth relative to the object's own median."""
    d = depth.copy(); med = np.median(d[sil]) if sil.any() else 0.0
    d[sil] = d[sil] - med
    return np.concatenate([sil.ravel().astype(np.float32), d.ravel().astype(np.float32)])

if __name__ == '__main__':
    # ---------- calibration: reproduce actual training-render silhouettes at their stored poses
    from PIL import Image
    NAV = '/vast/projects/bonnen/naturalistic-navig'; R = f'{NAV}/Dist-shift-data/shapenet_rendered'
    B = f'{NAV}/Dist-shift-data/blend-data/shapenet_training'
    import glob
    cats = ['airplane', 'chair', 'car', 'lamp']
    SYN = {'airplane': '02691156', 'chair': '03001627', 'car': '02958343', 'lamp': '03636649'}
    perms = [((0,1,2),(1,1,1)),((0,2,1),(1,1,1)),((0,2,1),(1,1,-1)),((0,2,1),(1,-1,1)),((1,0,2),(1,1,1)),((2,1,0),(1,1,1)),((0,1,2),(1,1,-1)),((0,2,1),(-1,1,1)),((0,2,1),(-1,1,-1))]
    res_out = {}
    for perm, flip in perms:
        ious = []
        for cat in cats:
            objs = sorted(os.listdir(f'{R}/camera_info/{cat}'))[:6]
            for o in objs:
                bv = f'{B}/{SYN[cat]}/{o}/models/model_normalized.solid.binvox'
                if not os.path.exists(bv): continue
                P = voxel_points(bv, perm, flip)
                for v in ['000', '004', '008', '012']:
                    c = pickle.load(open(f'{R}/camera_info/{cat}/{o}/{v}.pkl', 'rb'))
                    _, sil = project(P, np.array(c['position'], float), res=112)
                    im = np.array(Image.open(f'{R}/no_background/{cat}/{o}/{v}.png').convert('RGBA').resize((112, 112)))[:, :, 3] > 64
                    ious.append((sil & im).sum() / max((sil | im).sum(), 1))
        res_out[(perm, flip)] = np.mean(ious)
        print(f'perm={perm} flip={flip}: mean IoU vs actual renders = {np.mean(ious):.3f}  (n={len(ious)})')
    best = max(res_out, key=res_out.get); print('\nBEST', best, f'{res_out[best]:.3f}')
