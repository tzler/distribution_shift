"""
fig7 rebuilt with RAW axes.

The original binned the trial- and model-demeaned distance, which put negative values
on an axis that is a distance — conceptually wrong. This version bins by each model's
RANK WITHIN ITS OWN TRIAL (1 = the training set geometrically closest to that trial's
objects ... 12 = the furthest).

Why that controls the confound without transforming either axis: each of the 706 trials
contributes exactly one model to every rank. So all 12 bins contain the same 706 trials,
with the same objects, the same d(A,B) and the same intrinsic difficulty. The trial
composition is identical across the x-axis by construction, so a trend across ranks
cannot be produced by some trials being harder than others.

Plotted: x = the mean RAW distance of the models at that rank
         y = the mean RAW fine-tuned oddity margin at that rank
Nothing is demeaned, and the x axis is a real cosine distance.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
EMP = 'trial_distance_(L1_not_normalized)'
GEOM, DINO = '#2a78d6', '#eb6834'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9.5, 'axes.titlesize': 10.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def panel_data():
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values
        o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        rows.append(o[['trial', 'category', 'ft_margin', EMP,
                       'd_AB_(L1_not_normalized)']])
    return p.merge(pd.concat(rows), on=['trial', 'category'])


def rank_curve(d, X):
    """Mean raw x and raw y at each within-trial rank of X."""
    d = d.copy()
    d['rank'] = d.groupby('trial')[X].rank(method='first').astype(int)
    g = d.groupby('rank')
    out = pd.DataFrame({
        'x': g[X].mean(),
        'y': g['ft_margin'].mean(),
        'yse': g['ft_margin'].std() / np.sqrt(g['ft_margin'].size()),
        'n': g['ft_margin'].size(),
        'dab': g['d_AB_(L1_not_normalized)'].mean(),
    }).reset_index()
    return out


def main():
    d = panel_data()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.2))
    fig.subplots_adjust(left=.085, right=.98, top=.62, bottom=.20, wspace=.30)

    for a, (X, lab, col, unit) in zip(ax, [
            ('blind_k50',
             'how far this trial\'s objects are from the training set\n'
             'GEOMETRIC cosine, mesh-to-mesh — no encoder', GEOM, ''),
            (EMP,
             'how far this trial\'s objects are from the training set\n'
             'DINOv2 $\\ell_1$ feature distance', DINO, '')]):
        c = rank_curve(d, X)
        a.errorbar(c.x, c.y, yerr=c.yse, fmt='o-', color=col, lw=2, ms=9, mfc=col,
                   mec=SURFACE, mew=2, ecolor='#d8d7d2', zorder=4)
        r = stats.pearsonr(c.x, c.y)
        # per-trial slope on raw values, the statistic behind the curve
        sl = np.array([stats.linregress(s[X], s['ft_margin']).slope
                       for _, s in d.groupby('trial') if s[X].std() > 0])
        t = stats.ttest_1samp(sl, 0)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
        a.set_xlabel(lab)
        a.set_ylabel('oddity margin of the model fine-tuned\non that same training set')
        a.set_title(f'across-rank r = {r[0]:+.3f}\n'
                    f'per-trial slope t = {t.statistic:+.1f},  '
                    f'{100*(sl<0).mean():.0f}% of trials negative', loc='left')
        # annotate the balance guarantee
        a.text(.98, .95, f'each point = 706 trials\nsame trials in every point',
               transform=a.transAxes, ha='right', va='top', fontsize=8.5, color=INK2)

    fig.suptitle('Models trained further from a trial\'s objects show a smaller '
                 'oddity margin', fontsize=13, x=.085, ha='left', y=.975, color=INK)
    fig.text(.085, .80,
             'Each of 706 trials appears 12 times — once per ShapeNet training category; '
             'each category fine-tuned one model, so training set ↔ model is 1:1.\n'
             'Points are ordered by rank within each trial (1 = its closest training set … '
             '12 = its furthest), so all 12 contain the same 706 trials —\n'
             'identical objects, identical d(A,B), identical difficulty. Both axes raw.',
             fontsize=9.5, color=INK2, ha='left')
    p = f'{OUT}/fig9_margin_rank.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)

    for X, nm in [('blind_k50', 'geometric'), (EMP, 'DINOv2')]:
        c = rank_curve(d, X)
        print(f'\n{nm}: mean raw distance rank1={c.x.iloc[0]:.4f} -> '
              f'rank12={c.x.iloc[-1]:.4f}')
        print(f'   mean raw margin  rank1={c.y.iloc[0]:.4f} -> rank12={c.y.iloc[-1]:.4f}'
              f'   (drop {100*(1-c.y.iloc[-1]/c.y.iloc[0]):.0f}%)')
        print(f'   d(A,B) across ranks: {c.dab.min():.2f}–{c.dab.max():.2f} '
              f'(identical by construction: {c.dab.std():.2e} sd)')


if __name__ == '__main__':
    main()
