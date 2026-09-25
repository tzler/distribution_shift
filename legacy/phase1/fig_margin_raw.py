"""
Margin vs distance-to-training, with RAW axes on both sides.

Nothing is demeaned. x is an actual cosine distance (>= 0); y is the actual oddity
margin. The trial-difficulty confound is handled by the DESIGN, not by transforming
either axis: within a single trial, d(A,B) is a fixed property of the stimulus, so any
variation in the margin across that trial's 12 category-specific models cannot be
caused by it.

Each faint line is one trial, connecting its 12 (distance, margin) points.
The statistic is the per-trial slope, fitted on raw values.
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


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def panel_data():
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values
        o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        rows.append(o[['trial', 'category', 'ft_margin', EMP]])
    return p.merge(pd.concat(rows), on=['trial', 'category'])


def slopes(d, X):
    out = {}
    for t, sub in d.groupby('trial'):
        if sub[X].std() > 0:
            out[t] = stats.linregress(sub[X], sub['ft_margin']).slope
    return pd.Series(out)


def main():
    d = panel_data()
    sl_g, sl_d = slopes(d, 'blind_k50'), slopes(d, EMP)

    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.9))
    fig.subplots_adjust(left=.06, right=.985, top=.70, bottom=.16, wspace=.28)

    # --- A: raw spaghetti ------------------------------------------------
    a = ax[0]; style(a)
    for t, sub in d.groupby('trial'):
        s = sub.sort_values('blind_k50')
        a.plot(s['blind_k50'], s['ft_margin'], '-', color=GEOM, lw=.6, alpha=.055,
               zorder=2)
    hi = sl_g.sort_values().index[:3]
    for t in hi:
        s = d[d.trial == t].sort_values('blind_k50')
        a.plot(s['blind_k50'], s['ft_margin'], 'o-', color=GEOM, lw=2, ms=5,
               mec=SURFACE, mew=1.2, alpha=.95, zorder=4)
    a.set_xlabel('geometric distance to training set  (cosine, raw)')
    a.set_ylabel('fine-tuned oddity margin  (raw)')
    a.set_title('A  One faint line per trial, across its 12 models\n'
                '3 example trials highlighted', loc='left')

    # --- B, C: per-trial slope distributions -----------------------------
    for a, sl, col, lab, unit in [(ax[1], sl_g, GEOM, 'geometric (model-free)', ''),
                                  (ax[2], sl_d, DINO, 'DINOv2 $\\ell_1$ (model-based)',
                                   '')]:
        style(a)
        a.hist(sl, bins=40, color=col, alpha=.75, edgecolor=SURFACE, linewidth=.8,
               zorder=3)
        a.axvline(0, color='#8a8985', lw=1.4, zorder=4)
        a.axvline(sl.mean(), color=INK, lw=2, zorder=5)
        t = stats.ttest_1samp(sl, 0)
        a.set_xlabel(f'per-trial slope of margin on distance{unit}')
        a.set_ylabel('number of trials')
        a.set_title(f'{"B" if col == GEOM else "C"}  {lab}\n'
                    f'mean {sl.mean():+.4f},  t = {t.statistic:+.1f},  '
                    f'{100*(sl<0).mean():.0f}% negative', loc='left')

    fig.suptitle('Within each trial, models trained further from the trial\'s objects '
                 'show a smaller oddity margin\n'
                 'Both axes raw — nothing demeaned. The confound is handled by the design: '
                 'within a trial, d(A,B) is fixed, so it cannot drive the effect.',
                 fontsize=12.5, x=.06, ha='left', y=.965, color=INK)
    p = f'{OUT}/fig8_margin_raw_axes.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)
    for nm, sl in [('geometric', sl_g), ('DINOv2', sl_d)]:
        t = stats.ttest_1samp(sl, 0)
        print(f'  {nm:10s} mean slope {sl.mean():+.4f}  t={t.statistic:+.2f}  '
              f'p={t.pvalue:.1e}  {100*(sl<0).mean():.0f}% negative  (n={len(sl)} trials)')


if __name__ == '__main__':
    main()
