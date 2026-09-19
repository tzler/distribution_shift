"""Same raw scatter, restricted to the ON-CATEGORY model for each trial.

For each MOCHI ShapeNet trial there is exactly one of the 12 fine-tuned models whose
training category matches the trial's own objects. Keeping only those rows gives one
point per trial. Raw axes, no centring, no lines.
"""
import ast, numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OUT = f'{G}/out/figures'
GEOM = '#2a78d6'
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


p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
m = pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows = []
for cat in p.category.unique():
    o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
    o['trial'] = m['trial'].values; o['category'] = cat
    o['ft'] = o['fine_tuned_oddity_margin']
    rows.append(o[['trial', 'category', 'ft']])
d = p.merge(pd.concat(rows), on=['trial', 'category'])

own = {}
for _, r in m[m.dataset == 'shapenet'].iterrows():
    s = {f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s) == 1:
        own[r['trial']] = INV.get(list(s)[0])
d['own'] = d.trial.map(own)
oc = d[d.category == d.own].dropna(subset=['own'])
x = oc['blind_k50'].values; y = oc['ft'].values
lr = stats.linregress(x, y)
print(f'ON-CATEGORY only: n={len(x)} trials  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  '
      f'p={lr.pvalue:.2e}')

fig, ax = plt.subplots(1, 5, figsize=(19.5, 4.6), sharex=True, sharey=True)
fig.subplots_adjust(left=.048, right=.99, top=.63, bottom=.17, wspace=.10)

a = ax[0]; style(a)
a.scatter(x, y, s=17, color=GEOM, alpha=.32, linewidths=0, zorder=3)
xs = np.linspace(x.min(), x.max(), 40)
a.plot(xs, lr.slope * xs + lr.intercept, color=GEOM, lw=2, alpha=.6, zorder=4)
a.set_title(f'on-category model only: one point per trial\nn = {len(x)}   '
            f'slope {lr.slope:+.3f}, r {lr.rvalue:+.3f}, p {lr.pvalue:.1e}', loc='left')
a.set_xlabel('geometric distance from the trial\'s objects\nto its OWN category\'s training set (raw)')
a.set_ylabel('on-category model\'s oddity\nmargin on that trial (raw)')

mu = np.array([x.mean(), y.mean()]); sd = np.array([x.std(), y.std()])
Zs = (np.column_stack([x, y]) - mu) / sd
for a, k in zip(ax[1:], [10, 30, 50, 70]):
    style(a)
    cent, lab = kmeans2(Zs, k, minit='++', seed=0, iter=60)
    cnt = np.bincount(lab, minlength=k); keep = cnt > 0
    C = cent[keep] * sd + mu; n = cnt[keep]
    a.scatter(x, y, s=12, color=GEOM, alpha=.14, linewidths=0, zorder=2)
    a.scatter(C[:, 0], C[:, 1], s=20 + 300 * n / n.max(), color=GEOM,
              edgecolor=SURF, lw=1.3, alpha=.92, zorder=4)
    b1, b0 = np.polyfit(C[:, 0], C[:, 1], 1, w=n)
    xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, b1 * xs + b0, color=GEOM, lw=2, alpha=.55, zorder=3)
    rc = stats.pearsonr(C[:, 0], C[:, 1])
    a.set_title(f'k-means, {int(keep.sum())} clusters\nacross-centroid r = {rc[0]:+.3f}',
                loc='left')
    a.set_xlabel('geometric distance (raw)')
    print(f'  k={k:>3d}: {int(keep.sum())} non-empty, centroid r={rc[0]:+.3f}, '
          f'weighted slope={b1:+.4f}')

fig.suptitle('On-category trials only: distance to the training set the model actually saw',
             fontsize=14, x=.048, ha='left', y=.95, color=INK)
fig.text(.048, .755,
         f'Of the 12 fine-tuned models, exactly one was trained on each trial\'s own object '
         f'category. Keeping only that model gives {len(x)} points, one per trial — the '
         'condition where the\nmodel has genuinely seen objects of this kind. Raw axes, '
         'nothing centred, no lines. Right-hand panels summarise the same points with '
         'k-means in 2-D, marker area ∝ cluster size.',
         fontsize=9, color=INK2, ha='left')
q = f'{OUT}/fig37_scatter_oncategory.png'
fig.savefig(q, dpi=300); plt.close(fig); print('\n[fig]', q)
