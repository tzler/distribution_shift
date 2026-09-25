"""
Two figures:

fig15  the PRETRAINED margin as a structural control.
       The pretrained encoder is identical in all 12 per-category files — it never saw
       any of those training sets. Its within-trial variance is 2.9e-35 (floating-point
       zero), so the within-trial design MUST return null for it. Pooled, however, it
       correlates with geometric shift MORE strongly than the fine-tuned margin does.
       That dissociation is the cleanest evidence that the pooled analysis measures a
       stimulus property and the within-trial design measures training.

fig16  nine ways of computing the shift from the same descriptors and the same bank.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from estimators import ESTIMATORS                                   # noqa: E402

NAV = '/vast/projects/bonnen/naturalistic-navig'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
FT, PRE, MUTED = '#2a78d6', '#eb6834', '#a8a6a0'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 9.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})
NICE = {'nn_min': 'nearest neighbour  (k=1)', 'knn_mean': 'k-NN mean  (k=50, current)',
        'knn_kth': 'k-th NN distance', 'centroid': 'distance to bank centroid',
        'mahalanobis': 'Mahalanobis to bank mean', 'pca_recon': 'PCA reconstruction error',
        'energy_lse': 'energy / soft-min (logsumexp)', 'local_norm': 'density-normalised NN',
        'rank_pct': 'percentile within bank NN dist.'}


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def load():
    d = pd.read_csv(f'{G}/out/estimator_panel.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in d.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        rows.append(o[['trial', 'category', 'pretrained_oddity_margin',
                       'fine_tuned_oddity_margin']])
    return d.merge(pd.concat(rows), on=['trial', 'category'])


def rank_curve(d, X, dv):
    d = d.copy(); d['rank'] = d.groupby('trial')[X].rank(method='first')
    g = d.groupby('rank')
    return (g[X].mean().values, g[dv].mean().values,
            (g[dv].std() / np.sqrt(g[dv].size())).values)


def binned(x, y, nb=20):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    return (np.array([x[q == i].mean() for i in range(q.max() + 1)]),
            np.array([y[q == i].mean() for i in range(q.max() + 1)]))


def fig_pretrained(d):
    PREM, FTM = 'pretrained_oddity_margin', 'fine_tuned_oddity_margin'
    tl = d.groupby('trial').agg(shift=('knn_mean', 'mean'), pre=(PREM, 'first'),
                                ft=(FTM, 'mean')).reset_index()
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 8.0))
    fig.subplots_adjust(left=.09, right=.98, top=.75, bottom=.08, hspace=.46, wspace=.28)

    for a, (col, lab, c) in zip(ax[0], [('pre', 'PRETRAINED margin', PRE),
                                        ('ft', 'FINE-TUNED margin', FT)]):
        style(a)
        bx, by = binned(tl['shift'].values, tl[col].values)
        a.plot(bx, by, 'o-', color=c, lw=2, ms=7, mfc=c, mec=SURFACE, mew=1.8, zorder=3)
        r = stats.pearsonr(tl['shift'], tl[col])
        a.set_title(f'POOLED across trials — {lab}\nr = {r[0]:+.3f},  p = {r[1]:.0e}',
                    loc='left')
        a.set_xlabel('geometric distance to training set'); a.set_ylabel('oddity margin')

    for a, (col, lab, c) in zip(ax[1], [(PREM, 'PRETRAINED margin', PRE),
                                        (FTM, 'FINE-TUNED margin', FT)]):
        style(a)
        bx, by, se = rank_curve(d, 'knn_mean', col)
        a.errorbar(bx, by, yerr=se, fmt='o-', color=c, lw=2, ms=7, mfc=c, mec=SURFACE,
                   mew=1.8, ecolor='#d8d7d2', zorder=3)
        wv = (d[col] - d.groupby('trial')[col].transform('mean')).var()
        a.set_title(f'WITHIN TRIAL — {lab}\nwithin-trial variance = {wv:.1e}', loc='left')
        a.set_xlabel('geometric distance to training set (raw)')
        a.set_ylabel('oddity margin (raw)')
        rng = by.max() - by.min()
        a.set_ylim(by.mean() - max(rng, .12) * .7, by.mean() + max(rng, .12) * .7)

    fig.suptitle('The pretrained model is the control: it never saw any of these '
                 'training sets', fontsize=13.5, x=.09, ha='left', y=.965, color=INK)
    fig.text(.09, .845,
             'TOP — pooled across trials, the PRETRAINED margin tracks geometric shift '
             'MORE strongly than the fine-tuned one, though it cannot possibly depend on\n'
             'these training sets. The pooled correlation is therefore a stimulus '
             'property, not a distributional effect.\n'
             'BOTTOM — within trial the pretrained margin is a constant (variance ~1e-35), '
             'so the design returns a flat line; only the fine-tuned margin responds.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig15_pretrained_control.png'
    fig.savefig(p, dpi=300); plt.close(fig); print('[fig]', p)


def fig_estimators(d):
    FTM = 'fine_tuned_oddity_margin'
    fig, ax = plt.subplots(3, 3, figsize=(13.8, 10.2), sharey=True)
    fig.subplots_adjust(left=.07, right=.985, top=.83, bottom=.06, hspace=.50, wspace=.14)
    rows = []
    for i, e in enumerate(ESTIMATORS):
        a = ax.ravel()[i]; style(a)
        bx, by, se = rank_curve(d, e, FTM)
        a.errorbar(bx, by, yerr=se, fmt='o-', color=FT, lw=1.8, ms=6, mfc=FT,
                   mec=SURFACE, mew=1.5, ecolor='#dcdbd6', zorder=3)
        w = d.copy()
        for c in [e, FTM]:
            w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                    - d.groupby('category')[c].transform('mean') + d[c].mean())
        r = stats.pearsonr(w[e], w[FTM])
        sl = np.array([stats.linregress(s[e], s[FTM]).slope
                       for _, s in d.groupby('trial') if s[e].std() > 0])
        t = stats.ttest_1samp(sl, 0)
        a.set_title(f'{NICE[e]}\nwithin-trial r = {r[0]:+.3f},  '
                    f'{100*(sl<0).mean():.0f}% neg', loc='left')
        a.set_xlabel('distance to training set (raw)')
        if i % 3 == 0:
            a.set_ylabel('fine-tuned oddity margin (raw)')
        rows.append(dict(estimator=e, r_within=r[0], p_within=r[1],
                         t_slope=t.statistic, pct_neg=100 * (sl < 0).mean()))
    fig.suptitle('Nine ways to compute the shift from the same descriptors and the same '
                 'bank', fontsize=13.5, x=.07, ha='left', y=.972, color=INK)
    fig.text(.07, .885,
             'Same features throughout — only the estimator changes. Each panel: 12 '
             'within-trial ranks, raw axes, shared y; all 12 points hold the same 706 '
             'trials.\nFrom a single nearest neighbour, through k-NN and density-corrected '
             'variants, to distance-to-the-manifold (PCA reconstruction) and a covariance-'
             'aware\nMahalanobis distance.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig16_estimator_variants.png'
    fig.savefig(p, dpi=300); plt.close(fig); print('[fig]', p)
    out = pd.DataFrame(rows).sort_values('r_within')
    out.to_csv(f'{G}/out/estimator_variants.csv', index=False)
    print()
    print(f'{"estimator":34s}{"within-trial r":>15s}{"t":>9s}{"%neg":>7s}')
    for _, r in out.iterrows():
        print(f'{NICE[r.estimator]:34s}{r.r_within:>+15.3f}{r.t_slope:>+9.1f}'
              f'{r.pct_neg:>6.0f}%')


if __name__ == '__main__':
    d = load()
    fig_pretrained(d)
    fig_estimators(d)
