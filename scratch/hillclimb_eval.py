"""Score the hill-climb candidates honestly.

Three numbers matter, in this order:
  1. HELD-OUT r   -- 5-fold CV over trials. Select the winner on 4 folds, score it on
                     the fold it never saw. This is the number that generalises.
  2. PERMUTATION  -- the null distribution of the MAX |r| over all candidates, from
                     shuffling y. Answers 'how big a max would we see by chance from a
                     search this wide?'. The naive p-value of the winner does not.
  3. In-sample r  -- the number the search maximised. Always the largest; report last.
The base-DINOv2 control r is printed for every candidate but never filters anything.
"""
import argparse, os, sys
import numpy as np
from scipy import stats

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def zs(A):
    A = A - A.mean(0)
    s = A.std(0); s[s < 1e-12] = 1.0
    return A / s


def corr(Mz, yz):
    """Column-wise Pearson r between standardised M and standardised y."""
    return (Mz.T @ yz) / len(yz)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--y', default='margin_adv')  # margin_adv|ft|rt|acc|gap
    ap.add_argument('--nperm', type=int, default=2000)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--top', type=int, default=20)
    a = ap.parse_args()

    z = np.load(f'{G}/out/hillclimb_candidates.npz', allow_pickle=True)
    M = z['M']; keys = [str(k) for k in z['keys']]
    y = z[f'y_{a.y}'].astype(float)
    ctrl = z['y_pre'].astype(float)
    n, C = M.shape
    print(f'{n} trials x {C:,} candidates   objective: |r| with {a.y}\n')

    Mz = zs(M); yz = zs(y.reshape(-1, 1))[:, 0]; cz = zs(ctrl.reshape(-1, 1))[:, 0]
    r_full = corr(Mz, yz)
    r_ctrl = corr(Mz, cz)
    order = np.argsort(-np.abs(r_full))

    # ---------- 1. cross-validated: select on 4 folds, score on the 5th ----------
    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % a.folds
    held, picks = [], []
    for f in range(a.folds):
        tr, te = fold != f, fold == f
        rt = corr(zs(M[tr]), zs(y[tr].reshape(-1, 1))[:, 0])
        j = int(np.argmax(np.abs(rt)))
        rte = corr(zs(M[te]), zs(y[te].reshape(-1, 1))[:, 0])[j]
        held.append(rte); picks.append(keys[j])
        print(f'  fold {f}: picked {keys[j]:<46s} train r {rt[j]:+.3f}  '
              f'HELD-OUT r {rte:+.3f}')
    held = np.array(held)
    print(f'\n  mean HELD-OUT r = {held.mean():+.3f}  (sd {held.std():.3f}, '
          f'range {held.min():+.3f} to {held.max():+.3f})')
    print(f'  winner stable across folds: {len(set(picks))} distinct pick(s)')

    # ---------- 2. permutation null on the MAX statistic ----------
    best = np.abs(r_full).max()
    mx = np.empty(a.nperm)
    for b in range(a.nperm):
        yp = zs(rng.permutation(y).reshape(-1, 1))[:, 0]
        mx[b] = np.abs(corr(Mz, yp)).max()
    p_max = (1 + (mx >= best).sum()) / (a.nperm + 1)
    print(f'\n  observed max |r| = {best:.3f}')
    print(f'  null max |r| over {C:,} candidates: median {np.median(mx):.3f}, '
          f'95th pct {np.percentile(mx, 95):.3f}, max {mx.max():.3f}')
    print(f'  permutation p (corrected for the whole search) = {p_max:.4f}')

    # ---------- 3. the leaderboard ----------
    print(f'\n{"rank":>4s}  {"representation | estimator | metric":<48s}'
          f'{"r":>8s}{"p":>10s}{"ctrl r":>9s}{"ctrl p":>10s}')
    for i, j in enumerate(order[:a.top]):
        pv = stats.pearsonr(M[:, j], y)[1]
        pc = stats.pearsonr(M[:, j], ctrl)[1]
        print(f'{i+1:>4d}  {keys[j]:<48s}{r_full[j]:>+8.3f}{pv:>10.1e}'
              f'{r_ctrl[j]:>+9.3f}{pc:>10.2f}')

    # baseline: what we have been reporting
    for kk in ['voxel16|knn_mean_k50|both_mean', 'd57|knn_mean_k50|both_mean']:
        if kk in keys:
            j = keys.index(kk)
            print(f'\n  reference {kk}: r = {r_full[j]:+.3f}, '
                  f'control r = {r_ctrl[j]:+.3f}')
