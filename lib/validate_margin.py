"""
THE PRIMARY ANALYSIS: does the oddity margin recover a MODEL-FREE measure of
train-test distance?

Paper Sec 4.2 / Fig 1 (right) validates the oddity-margin proxy against the "empirical"
distribution shift -- but that target is itself computed in a pretrained DINOv2 feature
space, so a model-based proxy is being validated against a model-based ground truth.
Here we substitute a purely geometric target computed from the 3D meshes / silhouettes,
with no encoder anywhere, and re-run the same validation.

Published values (proxy vs DINOv2 l1 empirical distance, 30 quantile bins):
    shapenet r = .91      shapegen r = .62

Margins: Distribution-shift-Analyses/data/all_model_margins.csv, 2019 rows x 8 encoders,
positionally aligned to MOCHI/mochi_trials.csv (verified: r(dinov2-giant margin,
DINOv2G_avg) = +0.548 vs -0.027 shuffled).

Sign convention (paper eq. 7): s_hat = 1 + eps*m. eps=-1 for pretrained encoders. We
report signed r, but |r| is the comparable quantity across eps conventions.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
L = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/L1norm_vs_distshift'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
MARG = ('/vast/projects/bonnen/naturalistic-navig/Human-3D-generalization-copy/'
        'Distribution-shift-Analyses/data/all_model_margins.csv')
INC = 'trial_distance_(L1_not_normalized)'
OUT = f'{G}/out'
ENCODERS = ['dinov2-base', 'dinov2-large', 'dinov2-giant',
            'clip-b16', 'clip-l14', 'clip-h14', 'clip-g14']


def binned(x, y, nb):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    bx = np.array([x[q == i].mean() for i in range(q.max() + 1)])
    by = np.array([y[q == i].mean() for i in range(q.max() + 1)])
    return bx, by


def binned_r(x, y, nb=30):
    bx, by = binned(x, y, nb)
    r, p = stats.pearsonr(bx, by)
    return r, p, len(bx)


def build():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mg = pd.read_csv(MARG)
    assert len(m) == len(mg), 'margin file not aligned to mochi_trials'
    for c in mg.columns:
        m[f'margin_{c}'] = mg[c].values
    t = pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv')
    m = m.merge(t[['trial', INC]], on='trial', validate='1:1')
    return m


def main():
    m = build()
    rows = []
    for ds, mode in [('shapenet', '3d'), ('shapegen', '2d')]:
        g = pd.read_csv(f'{OUT}/shift{mode}_{ds}.csv')
        d = g.merge(m, on='trial', validate='1:1')
        targets = [(f'geom_shift_{mode}', 'GEOMETRIC shift'),
                   (f'geom_knn200_{mode}', 'GEOMETRIC knn200'),
                   (f'geom_dAB_{mode}', 'GEOMETRIC d(A,B)'),
                   (INC, 'DINOv2 l1 (incumbent)')]
        print(f'\n{"="*94}\n{ds}: does the oddity margin recover the geometric distance?'
              f'   n={len(d)}   [30 quantile bins]\n{"="*94}')
        print(f'{"encoder":16s}' + ''.join(f'{n:>24s}' for _, n in targets))
        for enc in ENCODERS:
            col = f'margin_{enc}'
            if col not in d:
                continue
            shat = 1.0 - d[col]                      # eps = -1
            cells = []
            for tgt, nm in targets:
                r, p, nb = binned_r(d[tgt].values, shat.values, 30)
                st = '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''
                cells.append(f'{r:+.3f}{st}')
                rows.append(dict(dataset=ds, mode=mode, encoder=enc, target=nm,
                                 n_trials=len(d), n_bins=nb, r=r, p=p))
            print(f'{enc:16s}' + ''.join(f'{c:>24s}' for c in cells))

        # bin-granularity robustness (paper's supplemental check)
        print(f'\n-- bin-granularity robustness, dinov2-large vs GEOMETRIC shift')
        shat = 1.0 - d['margin_dinov2-large']
        for nb in (10, 20, 30, 50, 100):
            r, p, n = binned_r(d[f'geom_shift_{mode}'].values, shat.values, nb)
            rt = stats.pearsonr(d[f'geom_shift_{mode}'], shat)[0]
            print(f'   {n:3d} bins: r={r:+.3f} (p={p:.1e})     [trial-level r={rt:+.3f}]')
            rows.append(dict(dataset=ds, mode=mode, encoder='dinov2-large',
                             target=f'GEOM shift @{n} bins', n_trials=len(d),
                             n_bins=n, r=r, p=p))
        figure(d, ds, mode)

    pd.DataFrame(rows).to_csv(f'{OUT}/margin_vs_geometric.csv', index=False)
    print(f'\nwrote {OUT}/margin_vs_geometric.csv')


def figure(d, ds, mode):
    encs = ['dinov2-large', 'dinov2-giant', 'clip-g14']
    targets = [(f'geom_shift_{mode}', 'geometric shift (model-free)'),
               (INC, 'DINOv2 $\\ell_1$ shift (incumbent)')]
    fig, ax = plt.subplots(len(targets), len(encs),
                           figsize=(4.6 * len(encs), 4.0 * len(targets)))
    for i, (tgt, tlab) in enumerate(targets):
        for j, enc in enumerate(encs):
            a = ax[i, j]
            shat = 1.0 - d[f'margin_{enc}']
            bx, by = binned(d[tgt].values, shat.values, 30)
            r, p = stats.pearsonr(bx, by)
            a.scatter(bx, by, s=34, color='#5B3E96' if i == 0 else '#9A9A9A',
                      edgecolor='white', zorder=3)
            b1, b0 = np.polyfit(bx, by, 1)
            xs = np.linspace(bx.min(), bx.max(), 50)
            a.plot(xs, b1 * xs + b0, color='#C1440E', lw=2, zorder=2)
            a.set_title(f'{enc}\nr={r:+.3f}, p={p:.1e}', fontsize=10)
            a.set_xlabel(tlab); a.set_ylabel('oddity margin proxy  $\\hat{s}=1-m$')
            a.grid(alpha=.25)
    fig.suptitle(f'{ds}: the oddity margin recovers a model-free geometric train-test '
                 f'distance\n(30 quantile bins, n={len(d)} trials)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = f'{OUT}/fig_margin_vs_geometric_{ds}.png'
    plt.savefig(p, dpi=140)
    plt.close()
    print(f'[figure] {p}')


if __name__ == '__main__':
    main()
