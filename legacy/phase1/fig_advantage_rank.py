"""
Geometric distribution shift (x) vs FINE-TUNING ADVANTAGE (fine-tuned - pretrained).

Same design and same raw axes as fig9: binned by each model's rank within its own
trial, so all 12 points contain the identical 706 trials and trial difficulty is held
constant by construction. Nothing is demeaned.

The advantage is the causally meaningful DV here: it asks whether training on a set
that is geometrically close to a trial's objects actually HELPS on that trial, relative
to the same pretrained starting point.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from fig_margin_rank import panel_data, EMP, GEOM, DINO, SURFACE, INK, INK2, OUT  # noqa

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9.5, 'axes.titlesize': 10.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})
DV = 'advantage'


def rank_curve(d, X, dv):
    d = d.copy()
    d['rank'] = d.groupby('trial')[X].rank(method='first').astype(int)
    g = d.groupby('rank')
    return pd.DataFrame({
        'x': g[X].mean(), 'y': g[dv].mean(),
        'yse': g[dv].std() / np.sqrt(g[dv].size()),
        'dab': g['d_AB_(L1_not_normalized)'].mean(),
    }).reset_index()


def main():
    d = panel_data()
    # advantage = fine-tuned correct minus the shared pretrained baseline
    S = ('/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune/'
         'ShapeNet_OOD_Analyses')
    m = pd.read_csv('/vast/projects/bonnen/naturalistic-navig/MOCHI/mochi_trials.csv')
    rows = []
    for cat in d.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values
        o['category'] = cat
        o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'advantage']])
    d = d.merge(pd.concat(rows), on=['trial', 'category'])

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.subplots_adjust(left=.085, right=.98, top=.68, bottom=.15, wspace=.28)
    for a, (X, lab, col) in zip(ax, [
            ('blind_k50', 'geometric distance to training set\n(cosine — model-free)', GEOM),
            (EMP, 'DINOv2 $\\ell_1$ distance to training set\n(model-based)', DINO)]):
        c = rank_curve(d, X, DV)
        a.axhline(0, color='#c9c7c1', lw=1.2, zorder=2)
        a.errorbar(c.x, c.y, yerr=c.yse, fmt='o-', color=col, lw=2, ms=9, mfc=col,
                   mec=SURFACE, mew=2, ecolor='#d8d7d2', zorder=4)
        r = stats.pearsonr(c.x, c.y)
        sl = np.array([stats.linregress(s[X], s[DV]).slope
                       for _, s in d.groupby('trial') if s[X].std() > 0])
        t = stats.ttest_1samp(sl, 0)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
        a.set_xlabel(lab)
        a.set_ylabel('fine-tuning advantage  (fine-tuned − pretrained)')
        a.set_title(f'across-rank r = {r[0]:+.3f}\n'
                    f'per-trial slope t = {t.statistic:+.1f},  '
                    f'{100*(sl<0).mean():.0f}% of trials negative', loc='left')
        a.text(.98, .95, 'each point = 706 trials\nsame trials in every point',
               transform=a.transAxes, ha='right', va='top', fontsize=8.5, color=INK2)
        print(f'{lab.splitlines()[0]}: advantage {c.y.iloc[0]:+.4f} (closest) -> '
              f'{c.y.iloc[-1]:+.4f} (furthest);  '
              f'crosses zero at rank {int(c[c.y < 0]["rank"].min()) if (c.y < 0).any() else None}')

    fig.suptitle('Fine-tuning helps only when the training set is geometrically close '
                 'to the trial\'s objects', fontsize=13, x=.085, ha='left', y=.975,
                 color=INK)
    fig.text(.085, .80,
             'Binned by each model\'s rank within its own trial (1 = closest training set '
             '… 12 = furthest); all 12 points contain the same 706 trials, so d(A,B) and\n'
             'trial difficulty are identical across the axis. Both axes raw. Above zero = '
             'fine-tuning improved on the pretrained baseline.',
             fontsize=9.5, color=INK2, ha='left')
    p = f'{OUT}/fig10_advantage_rank.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)


if __name__ == '__main__':
    main()
