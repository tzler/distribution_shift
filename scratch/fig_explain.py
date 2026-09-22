"""What is one data point? Walking from a single trial up to the aggregate plot."""
import ast, numpy as np, pandas as pd
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
GEOM, HL, MUT = '#2a78d6', '#eb6834', '#a8a6a0'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans', 'text.color': INK,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d8d7d2', 'axes.linewidth': .8, 'font.size': 9,
    'axes.titlesize': 10, 'legend.frameon': False})
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}


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
    rows.append(o[['trial', 'category', 'ft']])
d = p.merge(pd.concat(rows), on=['trial', 'category'])
X = 'blind_k50'
INV = {v: k for k, v in SYN.items()}
own = {}
for _, r in m[m.dataset == 'shapenet'].iterrows():
    s = {f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s) == 1:
        own[r['trial']] = INV.get(list(s)[0])
d['own'] = d.trial.map(own)
d['xc'] = d[X] - d.groupby('trial')[X].transform('mean')     # centre x only

fig, ax = plt.subplots(1, 4, figsize=(17.0, 4.9))
fig.subplots_adjust(left=.055, right=.99, top=.62, bottom=.16, wspace=.28)

# ---- A: ONE trial, raw axes, every point labelled
t0 = d[d.trial == d.trial.iloc[0]].sort_values(X)
a = ax[0]; style(a)
for _, r in t0.iterrows():
    c = HL if r.category == r.own else GEOM
    a.scatter(r[X], r.ft, s=70, color=c, edgecolor=SURF, lw=1.5, zorder=4)
    a.annotate(r.category, (r[X], r.ft), textcoords='offset points', xytext=(6, 3),
               fontsize=7, color=INK if c == HL else INK2)
a.set_xlabel('distance from THIS trial\'s objects\nto that category\'s training set')
a.set_ylabel('that category-model\'s\noddity margin on THIS trial')
a.set_title(f'A  ONE trial = 12 points\ntrial "{t0.trial.iloc[0]}", own category in orange',
            loc='left')

# ---- B: same trial, x centred
a = ax[1]; style(a)
a.axvline(0, color='#d8d7d2', lw=1.2, zorder=2)
for _, r in t0.iterrows():
    c = HL if r.category == r.own else GEOM
    a.scatter(r.xc, r.ft, s=70, color=c, edgecolor=SURF, lw=1.5, zorder=4)
a.set_xlabel('x now CENTRED: distance minus\nthis trial\'s own mean across its 12')
a.set_ylabel('oddity margin (raw)')
a.set_title('B  Same 12 points, x centred\nthe trial now sits at x = 0 on average', loc='left')

# ---- C: 25 trials overlaid
a = ax[2]; style(a)
a.axvline(0, color='#d8d7d2', lw=1.2, zorder=2)
for t in d.trial.drop_duplicates().iloc[:25]:
    s = d[d.trial == t].sort_values('xc')
    a.plot(s.xc, s.ft, '-', color=GEOM, lw=.9, alpha=.35, zorder=3)
    a.scatter(s.xc, s.ft, s=10, color=GEOM, alpha=.5, linewidths=0, zorder=3)
a.set_xlabel('centred distance'); a.set_ylabel('oddity margin (raw)')
a.set_title('C  25 trials overlaid\neach faint line is one trial', loc='left')

# ---- D: all trials, binned
a = ax[3]; style(a)
q = pd.qcut(d.xc, 30, labels=False, duplicates='drop')
bx = np.array([d.xc[q == i].mean() for i in range(q.max() + 1)])
by = np.array([d.ft[q == i].mean() for i in range(q.max() + 1)])
a.axvline(0, color='#d8d7d2', lw=1.2, zorder=2)
a.scatter(bx, by, s=45, color=GEOM, edgecolor=SURF, lw=1.4, zorder=4)
b1, b0 = np.polyfit(bx, by, 1)
xs = np.linspace(bx.min(), bx.max(), 40)
a.plot(xs, b1 * xs + b0, color=GEOM, lw=2, alpha=.55, zorder=3)
lr = stats.linregress(d.xc, d.ft)
a.set_xlabel('centred distance'); a.set_ylabel('oddity margin (raw)')
a.set_title(f'D  All {len(d):,} points, 30 bins\nslope = {lr.slope:+.4f}, r = {lr.rvalue:+.3f}',
            loc='left')

fig.suptitle('What is one data point in these plots?', fontsize=14, x=.055, ha='left',
             y=.95, color=INK)
fig.text(.055, .74,
         'Every MOCHI trial is run through 12 separate fine-tuned models, one per ShapeNet '
         'training category. So one trial contributes 12 points: the x is how far that '
         'trial\'s objects sit\nfrom that category\'s training objects, and the y is that '
         'model\'s oddity margin on that trial. The trial\'s images, its two objects and its '
         'difficulty are identical in all 12 —\nonly the training set changes. Centring '
         'subtracts each trial\'s own mean x, which is what makes trial difficulty drop out.',
         fontsize=9, color=INK2, ha='left')
q_ = f'{OUT}/fig34_what_is_a_point.png'
fig.savefig(q_, dpi=300); plt.close(fig); print('[fig]', q_)

# ================= per-condition small multiples =================
cats = sorted(SYN)
fig, ax = plt.subplots(3, 4, figsize=(15.4, 10.0), sharey=True)
fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.06, hspace=.44, wspace=.10)
print(f'\n{"condition":13s}{"trials":>7s}{"obs":>7s}{"slope":>10s}{"r":>8s}{"p":>10s}')
for i, cat in enumerate(cats):
    a = ax.ravel()[i]; style(a)
    sub = d[d.own == cat].copy()
    sub['xc'] = sub[X] - sub.groupby('trial')[X].transform('mean')
    nb = 10
    q = pd.qcut(sub.xc, nb, labels=False, duplicates='drop')
    bx = np.array([sub.xc[q == j].mean() for j in range(q.max() + 1)])
    by = np.array([sub.ft[q == j].mean() for j in range(q.max() + 1)])
    se = np.array([sub.ft[q == j].sem() for j in range(q.max() + 1)])
    lr = stats.linregress(sub.xc, sub.ft)
    a.axvline(0, color='#e6e4df', lw=1.1, zorder=2)
    a.errorbar(bx, by, yerr=se, fmt='o', color=GEOM, ms=6, mfc=GEOM, mec=SURF, mew=1.4,
               ecolor='#dcdbd6', zorder=4)
    xs = np.linspace(bx.min(), bx.max(), 30)
    a.plot(xs, lr.slope * xs + lr.intercept, color=GEOM, lw=2, alpha=.55, zorder=3)
    a.set_title(f'{cat}  ({sub.trial.nunique()} trials)\n'
                f'slope {lr.slope:+.3f}   r {lr.rvalue:+.3f}', loc='left')
    a.set_xlabel('centred distance')
    if i % 4 == 0:
        a.set_ylabel('oddity margin (raw)')
    print(f'{cat:13s}{sub.trial.nunique():>7d}{len(sub):>7d}{lr.slope:>+10.3f}'
          f'{lr.rvalue:>+8.3f}{lr.pvalue:>10.1e}')
fig.suptitle('One condition at a time: each ShapeNet category separately',
             fontsize=14, x=.055, ha='left', y=.965, color=INK)
fig.text(.055, .875,
         'Same plot as before but restricted to the trials whose objects belong to that '
         'category. Within each panel every trial still contributes 12 points, one per\n'
         'training-set model. x is centred within trial (one subtraction); y is the raw '
         'oddity margin, shared scale across panels. 10 quantile bins.',
         fontsize=9, color=INK2, ha='left')
q_ = f'{OUT}/fig35_per_condition.png'
fig.savefig(q_, dpi=300); plt.close(fig); print('[fig]', q_)
