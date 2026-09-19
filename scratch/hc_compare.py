"""Two corrections to the CV summary, then the comparison that actually matters.

(1) Averaging SIGNED held-out r across folds is wrong when the search maximises |r| and
    different folds pick candidates of opposite sign -- the average then cancels a real
    effect to near zero. Align on |r|.
(2) 'Held-out r of the selected candidate' means nothing on its own. The question is
    whether SEARCHING beats just using the fixed baseline we already had, so the
    baseline must be scored on the SAME held-out folds.
"""
import numpy as np, os
from scipy import stats
G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'voxel16|knn_mean_k50|both_mean'


def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-12] = 1.0
    return A / s


z = np.load(f'{G}/out/hillclimb_candidates.npz', allow_pickle=True)
M = z['M']; keys = [str(k) for k in z['keys']]; n = M.shape[0]
jb = keys.index(BASE)
for yname in ['margin_adv', 'rt']:
    y = z[f'y_{yname}'].astype(float)
    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % 5
    sel, bas, picks = [], [], []
    for f in range(5):
        tr, te = fold != f, fold == f
        rt_ = (zs(M[tr]).T @ zs(y[tr].reshape(-1, 1))[:, 0]) / tr.sum()
        j = int(np.argmax(np.abs(rt_)))
        rte = (zs(M[te]).T @ zs(y[te].reshape(-1, 1))[:, 0]) / te.sum()
        sel.append(abs(rte[j])); bas.append(abs(rte[jb])); picks.append(keys[j])
    sel, bas = np.array(sel), np.array(bas)
    w = stats.wilcoxon(sel, bas)[1] if len(set(sel - bas)) > 1 else 1.0
    print(f'\n===== y = {yname} =====')
    print(f'  SEARCHED  mean |held-out r| = {sel.mean():.3f}  (folds: '
          f'{", ".join(f"{v:.3f}" for v in sel)})')
    print(f'  BASELINE  mean |held-out r| = {bas.mean():.3f}  (folds: '
          f'{", ".join(f"{v:.3f}" for v in bas)})')
    print(f'  gain from searching = {sel.mean()-bas.mean():+.3f}   '
          f'(wins {int((sel>bas).sum())}/5 folds, wilcoxon p={w:.3f})')
    print(f'  winner stability: {len(set(picks))} distinct pick(s) across 5 folds')
    for p in sorted(set(picks)):
        print(f'      {picks.count(p)}x  {p}')
