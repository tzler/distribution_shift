import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
NAV = '/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

G=_RESOLVED_G
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OUT = f'{G}/out/figures'
EMP = 'trial_distance_(L1_not_normalized)'; DABm = 'd_AB_(L1_not_normalized)'
GEOM, DINO, MUT = '#2a78d6', '#eb6834', '#a8a6a0'
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
    rows.append(o[['trial', 'category', 'ft', EMP, DABm]])
d = p.merge(pd.concat(rows), on=['trial', 'category'])
X = 'blind_k50'


def qbin(x, y, nb):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    return (np.array([x[q == i].mean() for i in range(q.max() + 1)]),
            np.array([y[q == i].mean() for i in range(q.max() + 1)]), q)


fig, ax = plt.subplots(2, 4, figsize=(16.2, 8.4))
fig.subplots_adjust(left=.055, right=.985, top=.78, bottom=.08, hspace=.48, wspace=.30)

# ---------- ROW 1: binning ----------
a = ax[0, 0]; style(a)
dd = d.copy(); dd['rk'] = dd.groupby('trial')[X].rank(method='first')
g = dd.groupby('rk')
bx, by = g[X].mean(), g.ft.mean()
a.plot(bx, by, 'o-', color=GEOM, lw=2, ms=7, mfc=GEOM, mec=SURF, mew=1.6, zorder=3)
dab = g[DABm].mean()
a.set_title(f'A  RANK bins (fig 9) — only 12 possible\nd(A,B) sd across bins = {dab.std():.0e}',
            loc='left')
a.set_xlabel('geometric distance (raw)'); a.set_ylabel('oddity margin (raw)')
for k, nb in enumerate([12, 30, 60]):
    a = ax[0, k + 1]; style(a)
    bx, by, q = qbin(d[X].values, d.ft.values, nb)
    a.plot(bx, by, 'o-', color=MUT, lw=1.8, ms=5, mfc=MUT, mec=SURF, mew=1.3, zorder=3)
    dabs = np.array([d[DABm].values[q == i].mean() for i in range(q.max() + 1)])
    a.set_title(f'{"BCD"[k]}  RAW-VALUE bins, {nb}\nd(A,B) sd across bins = {dabs.std():.1f}',
                loc='left')
    a.set_xlabel('geometric distance (raw)')
    if k == 0:
        a.set_ylabel('oddity margin (raw)')

# ---------- ROW 2: centring ----------
def cen(df, cols, by):
    o = df.copy()
    for c in cols:
        o[c] = df[c] - df.groupby(by)[c].transform('mean')
    return o


tw = cen(cen(d, [X, 'ft'], 'trial'), [X, 'ft'], 'category')
SPECS = [('E  NO centring', d, X),
         ('F  ONE subtraction — trial only', cen(d, [X, 'ft'], 'trial'), X),
         ('G  TWO subtractions — trial + set (fig 7)', tw, X),
         ('H  ONE subtraction, DINOv2 target', cen(d, [EMP, 'ft'], 'trial'), EMP)]
for k, (ttl, dat, xx) in enumerate(SPECS):
    a = ax[1, k]; style(a)
    bx, by, _ = qbin(dat[xx].values, dat.ft.values, 30)
    r = stats.pearsonr(bx, by); rt = stats.pearsonr(dat[xx], dat.ft)
    c = DINO if 'DINOv2' in ttl else GEOM
    a.plot(bx, by, 'o', color=c, ms=6, mfc=c, mec=SURF, mew=1.4, zorder=3)
    b1, b0 = np.polyfit(bx, by, 1)
    xs = np.linspace(bx.min(), bx.max(), 40)
    a.plot(xs, b1 * xs + b0, color=c, lw=2, alpha=.55, zorder=2)
    a.set_title(f'{ttl}\nbinned r = {r[0]:+.3f}   trial-level r = {rt[0]:+.3f}', loc='left')
    a.set_xlabel('distance (as transformed)')
    if k == 0:
        a.set_ylabel('oddity margin')

fig.suptitle('Binning and centring choices, side by side', fontsize=13.5, x=.055,
             ha='left', y=.965, color=INK)
fig.text(.055, .855,
         'TOP — fig 9 uses RANK bins, so there can only ever be 12, and every bin holds '
         'the same 706 trials: d(A,B) is identical across the axis (sd ~1e-13).\n'
         'Binning on the raw VALUE allows any number of bins, but each bin then holds '
         'different trials, so d(A,B) varies and the difficulty confound returns.\n'
         'BOTTOM — centring the x axis. Trial-centring is what removes d(A,B); the second '
         'subtraction only removes per-model offsets.',
         fontsize=9, color=INK2, ha='left')
q = f'{OUT}/fig32_binning_and_centring.png'
fig.savefig(q, dpi=300); plt.close(fig); print('[fig]', q)
for ttl, dat, xx in SPECS:
    bx, by, _ = qbin(dat[xx].values, dat.ft.values, 30)
    print(f'{ttl:44s} binned {stats.pearsonr(bx, by)[0]:+.3f}   '
          f'trial-level {stats.pearsonr(dat[xx], dat.ft)[0]:+.3f}')
