"""
Geometric distribution shift per MOCHI trial.

Drop-in substitution into the paper's Eqs. 1-3 -- only phi changes, from a DINOv2
feature to a hand-computed geometric descriptor:

    D(trial, C_i) = 1/2 [ d(phi(A), phi(C_i)) + d(phi(B), phi(C_i)) ]
    Delta         = min_i D(trial, C_i)

phi(A) = the trial's matched object(s), phi(B) = the oddity.

  mode 3d : phi is the voxel shape descriptor of the OBJECT (view-independent, so the
            two views of A collapse to one descriptor). shapenet only.
  mode 2d : phi(A) = mean silhouette descriptor over the matched images,
            phi(B) = silhouette descriptor of the oddity image. shapenet + shapegen.

Descriptors are z-scored using the TRAINING BANK's mean/sd before distances, since raw
dimensions have wildly different scales (histogram densities vs. convexity ratios).

Also emitted, to separate the two things the incumbent metric conflates (the audit
notes it is bounded below by 1/2 d_AB):
  d_AB     - within-trial geometric dissimilarity of the two objects
  shift_sep- separate-neighbour variant, 1/2[min_i d(A,C_i) + min_j d(B,C_j)]
  nn_id    - identity of the nearest training object (interpretable)
"""
import argparse, ast, os, sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

D = os.path.dirname(os.path.abspath(__file__))
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'


def obj_id_shapenet(fn):
    s = fn[:-4].split('_')
    return f'{s[0]}/{s[1]}'


def load(npz):
    z = np.load(npz, allow_pickle=True)
    return list(z['ids']), z['X'].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['3d', '2d'])
    ap.add_argument('--dataset', required=True, choices=['shapenet', 'shapegen'])
    ap.add_argument('--bank', required=True)
    ap.add_argument('--test', required=True,
                    help='3d: object descriptors npz; 2d: per-image descriptors npz')
    ap.add_argument('--out', required=True)
    ap.add_argument('--metric', default='cityblock')
    a = ap.parse_args()

    bids, BX = load(a.bank)
    tids, TX = load(a.test)
    tix = {k: i for i, k in enumerate(tids)}

    mu, sd = BX.mean(0), BX.std(0)
    sd[sd < 1e-9] = 1.0
    BZ = (BX - mu) / sd
    TZ = (TX - mu) / sd

    df = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    df = df[df.dataset == a.dataset].reset_index(drop=True)

    # ---- assemble phi(A), phi(B) per trial -------------------------------
    rows, phiA, phiB = [], [], []
    n_skip = 0
    for _, r in df.iterrows():
        ims = ast.literal_eval(r['images'])
        oi = int(r['oddity_index'])
        odd, matched = ims[oi], [f for k, f in enumerate(ims) if k != oi]
        if a.mode == '3d':
            ka, kb = obj_id_shapenet(matched[0]), obj_id_shapenet(odd)
            if ka not in tix or kb not in tix:
                n_skip += 1
                continue
            va, vb = TZ[tix[ka]], TZ[tix[kb]]
        else:
            mk = [f for f in matched if f in tix]
            if not mk or odd not in tix:
                n_skip += 1
                continue
            va, vb = TZ[[tix[f] for f in mk]].mean(0), TZ[tix[odd]]
        rows.append(r['trial'])
        phiA.append(va)
        phiB.append(vb)

    phiA, phiB = np.array(phiA), np.array(phiB)
    print(f'{a.dataset}/{a.mode}: {len(rows)} trials usable, {n_skip} skipped '
          f'(missing descriptors)', flush=True)

    # ---- Eqs. 1-3 --------------------------------------------------------
    dA = cdist(phiA, BZ, metric=a.metric)
    dB = cdist(phiB, BZ, metric=a.metric)
    tot = 0.5 * (dA + dB)
    j = tot.argmin(1)
    shift = tot[np.arange(len(j)), j]
    shift_sep = 0.5 * (dA.min(1) + dB.min(1))
    d_AB = np.array([cdist(phiA[[i]], phiB[[i]], metric=a.metric)[0, 0]
                     for i in range(len(rows))])

    # k-NN density: mean distance to the k nearest training objects. A standard OOD
    # score, and -- unlike the shared-neighbour min -- NOT bounded below by d_AB, so it
    # measures distance-to-training rather than within-trial dissimilarity.
    def knn_mean(Dm, k):
        idx = np.argpartition(Dm, k, axis=1)[:, :k]
        return np.take_along_axis(Dm, idx, axis=1).mean(1)

    cols = {
        'trial': rows,
        f'geom_shift_{a.mode}': shift,
        f'geom_shift_sep_{a.mode}': shift_sep,
        f'geom_dAB_{a.mode}': d_AB,
        f'geom_dA_nn_{a.mode}': dA.min(1),
        f'geom_dB_nn_{a.mode}': dB.min(1),
        f'geom_nn_{a.mode}': [bids[k] for k in j],
    }
    for k in (10, 50, 200):
        ka, kb = knn_mean(dA, k), knn_mean(dB, k)
        cols[f'geom_dA_knn{k}_{a.mode}'] = ka
        cols[f'geom_dB_knn{k}_{a.mode}'] = kb
        cols[f'geom_knn{k}_{a.mode}'] = 0.5 * (ka + kb)
    out = pd.DataFrame(cols)
    out.to_csv(a.out, index=False)
    print(f'wrote {a.out}  n={len(out)}', flush=True)
    print(out[[f'geom_shift_{a.mode}', f'geom_dAB_{a.mode}']].describe().loc[
        ['mean', 'std', 'min', 'max']].to_string())


if __name__ == '__main__':
    main()
