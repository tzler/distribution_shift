"""
Novel ways to COMPUTE the shift, holding the features fixed.

Task 1 varied the object representation. This varies the estimator: given the same
L2-normalised descriptors and the same training bank, how should "distance to the
training distribution" be defined? All of these are oddity-blind — computed per object,
then averaged over the trial's images, never differenced.

  nn_min        distance to the single nearest training object            (k=1)
  knn_mean      mean of the k nearest                          (k=50, the current metric)
  knn_kth       the k-th nearest distance alone       (Sun et al. 2022 deep-kNN OOD)
  centroid      distance to the training-set mean                (crudest baseline)
  mahalanobis   covariance-aware distance to the mean   (Lee et al. 2018), shrinkage 0.1
  pca_recon     reconstruction error from a PCA fitted on the bank
                  — distance to the training MANIFOLD, not to its points
  energy_lse    soft-min over ALL bank objects: -tau*logsumexp(-d/tau)
                  — an energy score; no hard neighbour cut-off
  local_norm    NN distance divided by that neighbour's own mean distance to its k
                  nearest bank neighbours — corrects for local density of the bank
  rank_pct      percentile of the test object's NN distance within the bank's own
                  internal NN-distance distribution — unitless and calibrated
"""
import ast, os, sys
import numpy as np
import pandas as pd

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
K = 50
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}
ESTIMATORS = ['nn_min', 'knn_mean', 'knn_kth', 'centroid', 'mahalanobis',
              'pca_recon', 'energy_lse', 'local_norm', 'rank_pct']


def per_object(TN, BN, which):
    """Per test object, a scalar 'distance to this training bank'."""
    D = 1.0 - TN @ BN.T                                   # cosine distance
    kk = min(K, D.shape[1])
    part = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1)
    if which == 'nn_min':
        return part[:, 0]
    if which == 'knn_mean':
        return part[:, :kk].mean(1)
    if which == 'knn_kth':
        return part[:, kk - 1]
    if which == 'centroid':
        c = BN.mean(0, keepdims=True)
        c = c / (np.linalg.norm(c) + 1e-12)
        return (1.0 - TN @ c.T)[:, 0]
    if which == 'mahalanobis':
        mu = BN.mean(0)
        C = np.cov(BN.T)
        C = 0.9 * C + 0.1 * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
        P = np.linalg.pinv(C)
        d = TN - mu
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, P, d), 0))
    if which == 'pca_recon':
        mu = BN.mean(0)
        X = BN - mu
        ncomp = min(10, X.shape[0] - 1, X.shape[1])
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:ncomp].T
        d = TN - mu
        return np.linalg.norm(d - d @ V @ V.T, axis=1)
    if which == 'energy_lse':
        tau = 0.05
        mn = D.min(1, keepdims=True)
        return (mn[:, 0] - tau * np.log(np.exp(-(D - mn) / tau).sum(1)))
    if which in ('local_norm', 'rank_pct'):
        # bank's own internal kNN-distance profile
        DB = 1.0 - BN @ BN.T
        np.fill_diagonal(DB, np.inf)
        kb = min(K, DB.shape[1] - 1)
        bank_knn = np.sort(np.partition(DB, kb - 1, axis=1)[:, :kb], axis=1).mean(1)
        nn_idx = np.argmin(D, axis=1)
        if which == 'local_norm':
            return part[:, 0] / (bank_knn[nn_idx] + 1e-9)
        return np.searchsorted(np.sort(bank_knn), part[:, 0]) / len(bank_knn)
    raise ValueError(which)


def build_panels(bank_npz, test_npz, out_csv):
    bz = np.load(bank_npz, allow_pickle=True)
    tz = np.load(test_npz, allow_pickle=True)
    B, T = bz['X'].astype(np.float64), tz['X'].astype(np.float64)
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd
    BN = BZ / (np.linalg.norm(BZ, axis=1, keepdims=True) + 1e-12)
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    tix = {k: i for i, k in enumerate(tz['ids'])}
    syn = np.array([i.split('/')[0] for i in bz['ids']])

    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    trials = [(r['trial'], [oid(f) for f in ast.literal_eval(r['images'])])
              for _, r in mm.iterrows()]

    recs = []
    for cat, s in SYN.items():
        sel = syn == s
        if sel.sum() < K:
            continue
        vals = {e: per_object(TN, BN[sel], e) for e in ESTIMATORS}
        for trial, keys in trials:
            if not all(k in tix for k in keys):
                continue
            rows = [tix[k] for k in keys]
            rec = {'trial': trial, 'category': cat}
            for e in ESTIMATORS:                 # oddity-blind: mean over trial images
                rec[e] = float(np.mean(vals[e][rows]))
            recs.append(rec)
        print(f'  {cat:12s} done', flush=True)
    d = pd.DataFrame(recs)
    d.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}  {d.shape}')
    return d


if __name__ == '__main__':
    build_panels(f'{G}/bank/bank3d_shapenet_trained.npz',
                 f'{G}/bank/test3d_shapenet.npz',
                 f'{G}/out/estimator_panel.csv')
