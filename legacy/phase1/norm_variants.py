"""
How does the estimate change with the centring / plotting choice?

Six ways to relate geometric distance-to-training to the fine-tuned oddity margin, on
the same 706 trials x 12 category-models panel:

  1 pooled              no centring at all — the naive trial-level correlation
  2 x centred by trial  only the predictor is trial-centred
  3 y centred by trial  only the outcome is trial-centred
  4 both, trial only    one-way fixed effects
  5 two-way (used)      trial + training-set fixed effects
  6 within-trial rank   raw values, balanced by rank (fig9)
  7 per-trial slope     raw values, slope fitted inside each trial (fig8)

1 is confounded: between-trial variation carries trial difficulty. 4-7 all remove it,
by different routes, and should agree if the effect is real.
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
GEOM, DINO, MUTED = '#2a78d6', '#eb6834', '#8a8985'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 9.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def panel():
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        rows.append(o[['trial', 'category', 'ft_margin', EMP]])
    return p.merge(pd.concat(rows), on=['trial', 'category'])


def cen(d, cols, by):
    o = d.copy()
    for c in cols:
        o[c] = d[c] - d.groupby(by)[c].transform('mean')
    return o


def binned(x, y, nb=20):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    return (np.array([x[q == i].mean() for i in range(q.max() + 1)]),
            np.array([y[q == i].mean() for i in range(q.max() + 1)]))


def variants(d, X):
    out = []
    out.append(('1  pooled (no centring)', d[X], d['ft_margin']))
    a = cen(d, [X], 'trial');            out.append(('2  x centred by trial', a[X], a['ft_margin']))
    b = cen(d, ['ft_margin'], 'trial');  out.append(('3  y centred by trial', b[X], b['ft_margin']))
    c = cen(d, [X, 'ft_margin'], 'trial')
    out.append(('4  both centred by trial', c[X], c['ft_margin']))
    e = c.copy()
    for col in [X, 'ft_margin']:
        e[col] = c[col] - c.groupby('category')[col].transform('mean')
    out.append(('5  two-way (trial + set)', e[X], e['ft_margin']))
    return out


def main():
    d = panel()
    rows = []
    fig, ax = plt.subplots(2, 4, figsize=(16.4, 7.6))
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.09, hspace=.46, wspace=.30)

    for i, (name, x, y) in enumerate(variants(d, 'blind_k50')):
        a = ax.ravel()[i]
        r = stats.pearsonr(x, y)
        bx, by = binned(x.values, y.values)
        a.plot(bx, by, 'o-', color=GEOM, lw=2, ms=6, mfc=GEOM, mec=SURFACE, mew=1.6,
               zorder=3)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
        a.set_title(f'{name}\nr = {r[0]:+.3f},  p = {r[1]:.0e}', loc='left')
        a.set_xlabel('geometric distance'); a.set_ylabel('oddity margin')
        rows.append(dict(method=name, r=r[0], p=r[1]))
        print(f'{name:28s} r = {r[0]:+.4f}   p = {r[1]:.2e}')

    # 6: within-trial rank, raw
    a = ax.ravel()[5]
    dd = d.copy(); dd['rank'] = dd.groupby('trial')['blind_k50'].rank(method='first')
    g = dd.groupby('rank')
    bx, by = g['blind_k50'].mean(), g['ft_margin'].mean()
    r = stats.pearsonr(bx, by)
    a.plot(bx, by, 'o-', color=GEOM, lw=2, ms=7, mfc=GEOM, mec=SURFACE, mew=1.6, zorder=3)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
    a.set_title(f'6  within-trial rank (raw axes)\nacross-rank r = {r[0]:+.3f}', loc='left')
    a.set_xlabel('geometric distance (raw)'); a.set_ylabel('oddity margin (raw)')
    rows.append(dict(method='6  within-trial rank', r=r[0], p=r[1]))
    print(f'{"6  within-trial rank":28s} r = {r[0]:+.4f}')

    # 7: per-trial slopes, raw
    a = ax.ravel()[6]
    sl = np.array([stats.linregress(s['blind_k50'], s['ft_margin']).slope
                   for _, s in d.groupby('trial') if s['blind_k50'].std() > 0])
    t = stats.ttest_1samp(sl, 0)
    a.hist(sl, bins=36, color=GEOM, alpha=.75, edgecolor=SURFACE, lw=.8, zorder=3)
    a.axvline(0, color=MUTED, lw=1.4, zorder=4); a.axvline(sl.mean(), color=INK, lw=2, zorder=5)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
    a.set_title(f'7  per-trial slopes (raw axes)\nt = {t.statistic:+.1f},  '
                f'{100*(sl<0).mean():.0f}% negative', loc='left')
    a.set_xlabel('per-trial slope'); a.set_ylabel('trials')
    rows.append(dict(method='7  per-trial slope', r=np.nan, p=t.pvalue))
    print(f'{"7  per-trial slope":28s} t = {t.statistic:+.2f}  {100*(sl<0).mean():.0f}% neg')

    # 8: summary bars
    a = ax.ravel()[7]
    lab = [r['method'].split('  ')[1] for r in rows[:6]]
    val = [r['r'] for r in rows[:6]]
    cols = [MUTED if i < 3 else GEOM for i in range(6)]
    a.barh(np.arange(6), val, color=cols, height=.62, zorder=3)
    a.axvline(0, color='#d8d7d2', lw=1)
    a.set_yticks(np.arange(6)); a.set_yticklabels(lab, fontsize=8)
    a.invert_yaxis()
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis='x', color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
    a.set_xlabel('r with the oddity margin')
    a.set_title('8  summary\ngrey = confounded, blue = difficulty removed', loc='left')

    fig.suptitle('Does the conclusion depend on how we centre the axes?  '
                 '(geometric metric, 706 trials × 12 training sets)',
                 fontsize=13, x=.055, ha='left', y=.965, color=INK)
    fig.text(.055, .875,
             'Methods 1–3 leave between-trial variation in play, so trial difficulty can '
             'contribute. Methods 4–7 remove it by different routes and should agree.',
             fontsize=9.5, color=INK2, ha='left')
    p = f'{OUT}/fig12_centring_variants.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('\n[fig]', p)
    pd.DataFrame(rows).to_csv(f'{G}/out/centring_variants.csv', index=False)


if __name__ == '__main__':
    main()
