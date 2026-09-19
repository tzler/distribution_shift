"""The margin moves with the distribution — as a WITHIN-TRIAL manipulation.

Every trial is run through 12 models that differ only in which ShapeNet category they
were fine-tuned on. The trial's images, its objects, its difficulty d(A,B), its human
accuracy and RT, and the base DINOv2 margin are all IDENTICAL across those 12 (verified:
within-trial sd = 0.000e+00 for each). So holding the trial fixed and varying the
training set isolates training exposure from everything else by construction, not by
statistical adjustment: trial-centred x has exactly zero covariance with any of them
(measured r with the base margin = 3.9e-18).
"""
import ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OUT = f'{G}/out/figures'
GEOM, HL, GREY = '#2a78d6', '#eb6834', '#8a8884'
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
INV = {v: k for k, v in SYN.items()}


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


p = pd.read_csv(f'{G}/out/blindshift_shapenet_voxel16_percat.csv')
m = pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows = []
for cat in p.category.unique():
    o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
    o['trial'] = m['trial'].values; o['category'] = cat
    o['ft'] = o['fine_tuned_oddity_margin']; o['pre'] = o['pretrained_oddity_margin']
    rows.append(o[['trial', 'category', 'ft', 'pre']])
d = p.merge(pd.concat(rows), on=['trial', 'category'])
own = {}
for _, r in m[m.dataset == 'shapenet'].iterrows():
    s = {f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s) == 1:
        own[r['trial']] = INV.get(list(s)[0])
d['own'] = d.trial.map(own); d = d.dropna(subset=['own'])
d['rank'] = d.groupby('trial').blind_k50.rank(method='first').astype(int)
d['xc'] = d.blind_k50 - d.groupby('trial').blind_k50.transform('mean')
BASE = d.pre.mean()

fig, ax = plt.subplots(1, 5, figsize=(21.5, 4.7))
fig.subplots_adjust(left=.042, right=.99, top=.60, bottom=.165, wspace=.30)

# ---- A: one trial
a = ax[0]; style(a)
t0 = d[d.trial == d.trial.iloc[3]].sort_values('blind_k50')
a.plot(t0.blind_k50, t0.ft, '-', color=GEOM, lw=1.4, alpha=.5, zorder=3)
for _, r in t0.iterrows():
    c = HL if r.category == r.own else GEOM
    a.scatter(r.blind_k50, r.ft, s=95, color=c, edgecolor=SURF, lw=1.6, zorder=4)
a.axhline(t0.pre.iloc[0], color=GREY, lw=1.8, ls='--', zorder=2)
a.text(t0.blind_k50.max(), t0.pre.iloc[0], ' base DINOv2', va='bottom', ha='right',
       fontsize=8, color=GREY)
a.set_title('A  ONE trial, 12 models\nsame images; only the training set differs',
            loc='left')
a.set_xlabel('geometric distance from this trial\'s\nobjects to that model\'s training set')
a.set_ylabel('that model\'s oddity margin')

# ---- B: rank profile, all 12
g = d.groupby('rank').agg(ft=('ft', 'mean'), ftsem=('ft', 'sem'))
for k, (a, rr, ttl) in enumerate([
        (ax[1], range(1, 13), 'B  Rank the 12 training sets, nearest → farthest'),
        (ax[2], range(2, 13), 'C  Same, EXCLUDING the nearest training set')]):
    style(a)
    gg = g.loc[list(rr)]
    a.errorbar(gg.index, gg.ft, yerr=gg['ftsem'], fmt='o-', color=GEOM, ms=8, mfc=GEOM,
               mec=SURF, mew=1.5, lw=1.6, ecolor='#bcd0ea', elinewidth=1.6, zorder=4)
    a.axhline(BASE, color=GREY, lw=1.8, ls='--', zorder=2)
    a.text(list(rr)[-1], BASE, ' base DINOv2 (flat by construction)', va='bottom',
           ha='right', fontsize=8, color=GREY)
    sp = stats.spearmanr(gg.index, gg.ft)[0]
    a.set_title(f'{ttl}\nevery point averages all 706 trials   Spearman = {sp:+.3f}',
                loc='left')
    a.set_xlabel('rank of training set within the trial\n(1 = nearest)')
    a.set_ylabel('mean oddity margin')

# ---- D: within-trial slopes
a = ax[3]; style(a)
sl = np.array([stats.linregress(s.blind_k50, s.ft).slope
               for _, s in d.groupby('trial') if s.blind_k50.std() > 0])
a.hist(sl, bins=45, color=GEOM, alpha=.75, edgecolor=SURF, lw=.6, zorder=3)
a.axvline(0, color=INK2, lw=1.6, zorder=4)
a.axvline(np.median(sl), color=HL, lw=2, zorder=5)
a.text(np.median(sl), a.get_ylim()[1] * .96, ' median', color=HL, fontsize=8, va='top')
pv = stats.binomtest((sl < 0).sum(), len(sl), 0.5).pvalue
a.set_title(f'D  One slope per trial (n = {len(sl)})\n'
            f'{100*(sl<0).mean():.1f}% negative, sign test p = {pv:.0e}', loc='left')
a.set_xlabel('within-trial slope of margin on distance')
a.set_ylabel('number of trials')

# ---- E: trial-centred aggregate
a = ax[4]; style(a)
q = pd.qcut(d.xc, 50, labels=False, duplicates='drop')
bx = np.array([d.xc[q == i].mean() for i in range(q.max() + 1)])
by = np.array([d.ft[q == i].mean() for i in range(q.max() + 1)])
se = np.array([d.ft[q == i].sem() for i in range(q.max() + 1)])
a.errorbar(bx, by, yerr=se, fmt='o', color=GEOM, ms=7.5, mfc=GEOM, mec=SURF, mew=1.4,
           ecolor='#bcd0ea', elinewidth=1.4, zorder=4)
a.axhline(BASE, color=GREY, lw=1.8, ls='--', zorder=2)
a.axvline(0, color='#dcdad5', lw=1.2, zorder=1)
r_, p_ = stats.pearsonr(d.xc, d.ft)
rc = stats.pearsonr(d.xc, d.pre)[0]
a.set_title(f'E  All {len(d):,} observations, x trial-centred\n'
            f'r = {r_:+.3f}, p = {p_:.0e}   control r = {rc:.0e}', loc='left')
a.set_xlabel('distance minus this trial\'s own mean\n(centred within trial)')
a.set_ylabel('oddity margin')

fig.suptitle('The oddity margin moves with the training distribution — within trial',
             fontsize=15, x=.042, ha='left', y=.955, color=INK)
fig.text(.042, .715,
         'Each MOCHI ShapeNet trial is evaluated by 12 models differing only in which '
         'category they were fine-tuned on. The trial\'s images, its objects, its d(A,B), '
         'its human accuracy and RT, and the base DINOv2\nmargin are identical across all '
         '12 (within-trial sd = 0 for each), so varying the training set isolates training '
         'exposure by construction. In E the trial-centred x has exactly zero covariance '
         'with any trial-level\nproperty, so the base-DINOv2 control is algebraically zero '
         '— not merely small. B and C show the effect is mostly a step at the nearest '
         'training set, plus a shallow graded decline beyond it.',
         fontsize=9.2, color=INK2, ha='left')
q_ = f'{OUT}/fig45_margin_moves.png'
fig.savefig(q_, dpi=300); plt.close(fig); print('[fig]', q_)
