"""
EXACT Fig 1 (right) reproduction, with the model-free geometric target substituted.

Uses the manuscript's own artifacts:
  fine_tuned_oddity_margin      <- the Fig 1 proxy (eps=+1, adversarially-mined training set)
  trial_distance_(L1_not_norm)  <- the Fig 1 empirical target, computed against the HIDA
                                   training bank (anchors are shapegen/shapenet objects)
from  Dist-shift/logs/vit_large_patch14_reg4_dinov2_.../ood_analysis/ood_analysis_results.csv

Published (proxy vs empirical l1, 30 quantile bins):  shapegen r=.62   shapenet r=.91

We re-run the identical fit with Delta_geom in place of the l1 distance. If the margin
tracks the geometric distance as well as it tracks the encoder-based distance, the
validation no longer depends on a pretrained encoder to define its ground truth.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
OUT = f'{G}/out'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
OOD = ('/vast/projects/bonnen/naturalistic-navig/Dist-shift/logs/'
       'vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_multi_similarity_seed42_'
       'train:_val:_lora_r16_alpha8_dropout0.1_|45|---most-results-ood/'
       'ood_analysis/ood_analysis_results.csv')
EMP = 'trial_distance_(L1_not_normalized)'


def binned(x, y, nb):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    bx = np.array([x[q == i].mean() for i in range(q.max() + 1)])
    by = np.array([y[q == i].mean() for i in range(q.max() + 1)])
    return bx, by


def fit(x, y, nb=30):
    bx, by = binned(x, y, nb)
    r, p = stats.pearsonr(bx, by)
    F = r * r / (1 - r * r) * (len(bx) - 2)
    return r, p, F, len(bx)


def main():
    o = pd.read_csv(OOD)
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    assert len(o) == len(m), f'{len(o)} vs {len(m)}'
    o['trial'] = m['trial'].values                 # positional alignment, same source order
    # the ood file calls the majaj subset 'hvm'; otherwise row order is identical.
    # Verified independently: condition and oddity_index agree on 100% of rows.
    assert (o['dataset'].replace({'hvm': 'majaj'}).values == m['dataset'].values).all()
    assert (o['oddity_index'].values == m['oddity_index'].values).all()
    assert (o['condition'].values == m['condition'].values).all()

    o['s_ft'] = 1.0 + o['fine_tuned_oddity_margin']       # eps=+1 (fine-tuned, Fig 1)
    o['s_pre'] = 1.0 - o['pretrained_oddity_margin']      # eps=-1 (pretrained)
    o['train_adv'] = o['fine_tuned_correct'] - o['pretrained_correct']

    rows = []
    for ds, mode in [('shapegen', '2d'), ('shapenet', '3d')]:
        g = pd.read_csv(f'{OUT}/shift{mode}_{ds}.csv')
        d = g.merge(o, on='trial', validate='1:1')
        S, KNN = f'geom_shift_{mode}', f'geom_knn200_{mode}'
        print(f'\n{"="*92}\n{ds}   n={d.shape[0]}   [30 quantile bins, as published]\n{"="*92}')
        print(f'{"proxy":26s}{"target":34s}{"r":>9s}{"F":>9s}{"p":>11s}')
        for proxy, pn in [('s_ft', 'fine-tuned margin (Fig 1)'),
                          ('s_pre', 'pretrained margin')]:
            for tgt, tn in [(EMP, 'empirical l1 (DINOv2, published)'),
                            (S, 'GEOMETRIC shift (model-free)'),
                            (KNN, 'GEOMETRIC knn200 (model-free)')]:
                r, p, F, nb = fit(d[tgt].values, d[proxy].values)
                st = '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''
                print(f'{pn:26s}{tn:34s}{r:+9.3f}{F:9.1f}{p:11.1e} {st}')
                rows.append(dict(dataset=ds, proxy=pn, target=tn, n_trials=len(d),
                                 n_bins=nb, r=r, F=F, p=p))
        # training advantage vs each target (Fig 1 left)
        print(f'  -- training advantage (fine-tuned minus pretrained correct) vs target')
        for tgt, tn in [(EMP, 'empirical l1'), (S, 'GEOMETRIC shift')]:
            b = stats.linregress(d[tgt], d['train_adv'])
            print(f'     {tn:32s} beta={b.slope:+.2e}  r={b.rvalue:+.3f}  p={b.pvalue:.1e}')
            rows.append(dict(dataset=ds, proxy='training advantage', target=tn,
                             n_trials=len(d), n_bins=0, r=b.rvalue, F=np.nan, p=b.pvalue))
        figure(d, ds, S)
    pd.DataFrame(rows).to_csv(f'{OUT}/fig1_margin_vs_geometric.csv', index=False)
    print(f'\nwrote {OUT}/fig1_margin_vs_geometric.csv')


def figure(d, ds, S):
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for a, (tgt, lab, col) in zip(ax, [
            (EMP, 'empirical $\\ell_1$ distance (DINOv2)  — published', '#9A9A9A'),
            (S, 'geometric distance (model-free)  — this work', '#5B3E96')]):
        bx, by = binned(d[tgt].values, d['s_ft'].values, 30)
        r, p = stats.pearsonr(bx, by)
        a.scatter(bx, by, s=40, color=col, edgecolor='white', zorder=3)
        b1, b0 = np.polyfit(bx, by, 1)
        xs = np.linspace(bx.min(), bx.max(), 50)
        a.plot(xs, b1 * xs + b0, color='#C1440E', lw=2, zorder=2)
        a.set_xlabel(lab); a.set_ylabel('fine-tuned oddity margin  $\\hat{s}=1+m$')
        a.set_title(f'r={r:+.3f},  p={p:.1e}', fontsize=11)
        a.grid(alpha=.25)
    fig.suptitle(f'{ds}: validating the oddity margin against a model-based vs a '
                 f'model-free target\n(Fig 1 right, 30 quantile bins, n={len(d)})',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p = f'{OUT}/fig1_repro_{ds}.png'
    plt.savefig(p, dpi=145); plt.close()
    print(f'[figure] {p}')


if __name__ == '__main__':
    main()
