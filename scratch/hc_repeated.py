"""Repeated 5-fold CV: does SEARCHING beat the fixed baseline out of sample?

5 folds gives 5 paired numbers and no power. Repeat the whole split-select-score cycle
over many random partitions and compare the distributions.
"""
import numpy as np, os
from scipy import stats
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'voxel16|knn_mean_k50|both_mean'
NREP = 50


def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-12] = 1.0
    return A / s


z = np.load(f'{G}/out/hillclimb_candidates.npz', allow_pickle=True)
M = z['M']; keys = [str(k) for k in z['keys']]; n = M.shape[0]
jb = keys.index(BASE)
ctrl = z['y_pre'].astype(float)
r_ctrl_all = (zs(M).T @ zs(ctrl.reshape(-1, 1))[:, 0]) / n

for yname in ['margin_adv', 'rt']:
    y = z[f'y_{yname}'].astype(float)
    S, B, allpicks = [], [], []
    for rep in range(NREP):
        rng = np.random.default_rng(rep)
        fold = rng.permutation(n) % 5
        s_, b_ = [], []
        for f in range(5):
            tr, te = fold != f, fold == f
            rt_ = (zs(M[tr]).T @ zs(y[tr].reshape(-1, 1))[:, 0]) / tr.sum()
            j = int(np.argmax(np.abs(rt_)))
            rte = (zs(M[te]).T @ zs(y[te].reshape(-1, 1))[:, 0]) / te.sum()
            s_.append(abs(rte[j])); b_.append(abs(rte[jb])); allpicks.append(keys[j])
        S.append(np.mean(s_)); B.append(np.mean(b_))
    S, B = np.array(S), np.array(B)
    d = S - B
    t = stats.ttest_rel(S, B)
    print(f'\n===== y = {yname}   ({NREP} repeats x 5 folds) =====')
    print(f'  SEARCHED  mean |held-out r| = {S.mean():.3f}  (sd {S.std():.3f})')
    print(f'  BASELINE  mean |held-out r| = {B.mean():.3f}  (sd {B.std():.3f})')
    print(f'  gain      = {d.mean():+.3f}  95% CI [{np.percentile(d,2.5):+.3f}, '
          f'{np.percentile(d,97.5):+.3f}]   searching wins {100*(d>0).mean():.0f}% of repeats')
    print(f'  paired t  = {t.statistic:+.2f}, p = {t.pvalue:.4f}')
    from collections import Counter
    c = Counter(allpicks).most_common(4)
    print(f'  most-picked candidates across all {NREP*5} selections:')
    for k, v in c:
        print(f'      {100*v/(NREP*5):5.1f}%  {k:<44s} control r = {r_ctrl_all[keys.index(k)]:+.3f}')
