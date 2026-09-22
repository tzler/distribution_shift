"""Centre the X AXIS ONLY. The margin (y) stays raw throughout.

Within a trial the centred x sums to zero, so its covariance with any trial-constant is
exactly zero. Therefore the SLOPE is identical whether or not y is also centred; only r
changes, because raw y retains between-trial variance that x cannot explain. Shown here.
"""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV = '/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

G=_RESOLVED_G
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OUT = f'{G}/out/figures'
EMP = 'trial_distance_(L1_not_normalized)'
GEOM, DINO = '#2a78d6', '#eb6834'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans', 'text.color': INK,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d8d7d2', 'axes.linewidth': .8, 'font.size': 9,
    'axes.titlesize': 9.5, 'legend.frameon': False})


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
m = pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows = []
for cat in p.category.unique():
    o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
    o['trial'] = m['trial'].values; o['category'] = cat
    o['ft'] = o['fine_tuned_oddity_margin']
    rows.append(o[['trial', 'category', 'ft', EMP]])
d = p.merge(pd.concat(rows), on=['trial', 'category'])
X = 'blind_k50'


def cx(df, col, bys):
    """Centre ONLY the x column, by the given grouping(s). y untouched."""
    o = df.copy()
    for by in bys:
        o[col] = o[col] - o.groupby(by)[col].transform('mean')
    return o


def qbin(x, y, nb=30):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    return (np.array([x[q == i].mean() for i in range(q.max() + 1)]),
            np.array([y[q == i].mean() for i in range(q.max() + 1)]))


SPECS = [('A  x raw', d, X, GEOM),
         ('B  x centred by TRIAL only', cx(d, X, ['trial']), X, GEOM),
         ('C  x centred by trial + set', cx(d, X, ['trial', 'category']), X, GEOM),
         ('D  x centred by trial, DINOv2 target', cx(d, EMP, ['trial']), EMP, DINO)]
fig, ax = plt.subplots(1, 4, figsize=(16.4, 4.9), sharey=True)
fig.subplots_adjust(left=.055, right=.985, top=.66, bottom=.16, wspace=.12)
print(f'{"variant":40s}{"slope":>12s}{"r":>9s}{"slope, y also centred":>24s}')
for a, (ttl, dat, xx, c) in zip(ax, SPECS):
    style(a)
    bx, by = qbin(dat[xx].values, dat.ft.values)
    a.plot(bx, by, 'o', color=c, ms=6.5, mfc=c, mec=SURF, mew=1.5, zorder=3)
    b1, b0 = np.polyfit(bx, by, 1)
    xs = np.linspace(bx.min(), bx.max(), 40)
    a.plot(xs, b1 * xs + b0, color=c, lw=2, alpha=.55, zorder=2)
    lr = stats.linregress(dat[xx], dat.ft)
    # same but with y also centred, to show the slope is unchanged
    dy = dat.copy()
    for by_ in (['trial'] if 'TRIAL only' in ttl or 'DINOv2' in ttl
                else (['trial', 'category'] if 'trial + set' in ttl else [])):
        dy['ft'] = dy['ft'] - dy.groupby(by_)['ft'].transform('mean')
    lr2 = stats.linregress(dy[xx], dy.ft)
    a.set_title(f'{ttl}\nslope = {lr.slope:+.4f}   r = {lr.rvalue:+.3f}', loc='left')
    a.set_xlabel('distance (x as transformed)')
    print(f'{ttl:40s}{lr.slope:>+12.4f}{lr.rvalue:>+9.3f}{lr2.slope:>+24.4f}')
ax[0].set_ylabel('fine-tuned oddity margin  (RAW — never centred)')
fig.suptitle('Centring the x axis only: the margin stays raw',
             fontsize=13.5, x=.055, ha='left', y=.955, color=INK)
fig.text(.055, .775,
         'y is the untransformed oddity margin in every panel, so the vertical scale is '
         'identical across all four. Only the x axis changes.\n'
         'Within a trial the centred x sums to zero, so its covariance with any '
         'trial-constant is zero — the SLOPE is the same whether or not y is centred too '
         '(right-hand column of the printout);\nonly r differs, because raw y keeps '
         'between-trial variance that x cannot explain.',
         fontsize=9, color=INK2, ha='left')
q = f'{OUT}/fig33_centring_x_only.png'
fig.savefig(q, dpi=300); plt.close(fig); print('\n[fig]', q)
