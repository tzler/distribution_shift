"""All 706 trials x 12 category-models as one raw scatter, then summarised by k-means
at several granularities. Raw axes throughout: no centring, no lines."""
import numpy as np, pandas as pd
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
x = d['blind_k50'].values
y = d['ft'].values
lr = stats.linregress(x, y)
print(f'all points: n={len(x):,}  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  p={lr.pvalue:.1e}')

fig, ax = plt.subplots(1, 5, figsize=(19.5, 4.6), sharex=True, sharey=True)
fig.subplots_adjust(left=.048, right=.99, top=.63, bottom=.17, wspace=.10)

# ---- raw scatter
a = ax[0]; style(a)
a.scatter(x, y, s=5, color=GEOM, alpha=.06, linewidths=0, zorder=3)
a.set_title(f'every point: one trial × one model\nn = {len(x):,}   '
            f'slope {lr.slope:+.3f}, r {lr.rvalue:+.3f}', loc='left')
a.set_xlabel('geometric distance to that\nmodel\'s training set (raw)')
a.set_ylabel('that model\'s oddity margin\non that trial (raw)')

# ---- k-means summaries, clustered in standardised 2-D then drawn in raw units
mu = np.array([x.mean(), y.mean()]); sd = np.array([x.std(), y.std()])
Z = np.column_stack([x, y])
Zs = (Z - mu) / sd
for a, k in zip(ax[1:], [10, 30, 50, 70]):
    style(a)
    cent, lab = kmeans2(Zs, k, minit='++', seed=0, iter=60)
    cnt = np.bincount(lab, minlength=k)
    keep = cnt > 0
    C = cent[keep] * sd + mu
    n = cnt[keep]
    a.scatter(x, y, s=4, color=GEOM, alpha=.035, linewidths=0, zorder=2)
    a.scatter(C[:, 0], C[:, 1], s=18 + 320 * n / n.max(), color=GEOM,
              edgecolor=SURF, lw=1.3, alpha=.9, zorder=4)
    # fit through the cluster centroids, weighted by cluster size
    b1, b0 = np.polyfit(C[:, 0], C[:, 1], 1, w=n)
    xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, b1 * xs + b0, color=GEOM, lw=2, alpha=.55, zorder=3)
    rc = stats.pearsonr(C[:, 0], C[:, 1])
    a.set_title(f'k-means, {int(keep.sum())} clusters\nacross-centroid r = {rc[0]:+.3f}',
                loc='left')
    a.set_xlabel('geometric distance (raw)')
    print(f'  k={k:>3d}: {int(keep.sum())} non-empty, centroid r={rc[0]:+.3f}, '
          f'weighted slope={b1:+.4f}')

fig.suptitle('Every trial × every model, unaggregated — then summarised by clustering',
             fontsize=14, x=.048, ha='left', y=.95, color=INK)
fig.text(.048, .755,
         'Left: all 8,472 observations at low opacity, raw axes, nothing centred and no '
         'lines joining trials. Right: the same cloud summarised by k-means in 2-D '
         '(distance, margin),\nmarker area proportional to cluster size, with a '
         'size-weighted fit through the centroids. Increasing k trades smoothing for '
         'resolution without imposing bins on x.',
         fontsize=9, color=INK2, ha='left')
q = f'{OUT}/fig36_scatter_and_clusters.png'
fig.savefig(q, dpi=300); plt.close(fig); print('\n[fig]', q)
