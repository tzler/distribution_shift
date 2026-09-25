"""fig36 (all trial x model) and fig37 (on-category only) for a chosen representation."""
import argparse, ast, numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
from _repo import G as _RESOLVED_G

G=_RESOLVED_G
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
DESC = {'d57': 'geom3d 57-d summary (ROTATION-INVARIANT — pose discarded)',
        'voxel8': 'raw 8³ occupancy grid, 512-d (pose-SENSITIVE)',
        'voxel16': 'raw 16³ occupancy grid, 4,096-d (pose-SENSITIVE)',
        'voxel32': 'raw 32³ occupancy grid, 32,768-d (pose-SENSITIVE)'}


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def load(rep):
    f = (f'{G}/out/blindshift_shapenet_percat.csv' if rep == 'orig'
         else f'{G}/out/blindshift_shapenet_{rep}_percat.csv')
    p = pd.read_csv(f)
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
        s = {fn[:-4].split('_')[0] for fn in ast.literal_eval(r['images'])}
        if len(s) == 1:
            own[r['trial']] = INV.get(list(s)[0])
    d['own'] = d.trial.map(own)
    return d.dropna(subset=['own'])


def panel_set(x, y, title0, xlab0, ylab0, suptitle, blurb, path, alpha0, s0):
    lr = stats.linregress(x, y)
    fig, ax = plt.subplots(1, 5, figsize=(19.5, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.048, right=.99, top=.63, bottom=.17, wspace=.10)
    a = ax[0]; style(a)
    a.scatter(x, y, s=s0, color=GEOM, alpha=alpha0, linewidths=0, zorder=3)
    xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, lr.slope * xs + lr.intercept, color=GEOM, lw=2, alpha=.6, zorder=4)
    a.set_title(f'{title0}\nn = {len(x):,}   slope {lr.slope:+.3f}, r {lr.rvalue:+.3f}, '
                f'p {lr.pvalue:.1e}', loc='left')
    a.set_xlabel(xlab0); a.set_ylabel(ylab0)
    mu = np.array([x.mean(), y.mean()]); sd = np.array([x.std(), y.std()])
    Zs = (np.column_stack([x, y]) - mu) / sd
    for a, k in zip(ax[1:], [10, 30, 50, 70]):
        style(a)
        cent, lab = kmeans2(Zs, k, minit='++', seed=0, iter=60)
        cnt = np.bincount(lab, minlength=k); keep = cnt > 0
        C = cent[keep] * sd + mu; n = cnt[keep]
        a.scatter(x, y, s=max(4, s0 - 5), color=GEOM, alpha=alpha0 * .45,
                  linewidths=0, zorder=2)
        a.scatter(C[:, 0], C[:, 1], s=20 + 300 * n / n.max(), color=GEOM,
                  edgecolor=SURF, lw=1.3, alpha=.92, zorder=4)
        b1, b0 = np.polyfit(C[:, 0], C[:, 1], 1, w=n)
        a.plot(xs, b1 * xs + b0, color=GEOM, lw=2, alpha=.55, zorder=3)
        rc = stats.pearsonr(C[:, 0], C[:, 1])
        a.set_title(f'k-means, {int(keep.sum())} clusters\nacross-centroid r = {rc[0]:+.3f}',
                    loc='left')
        a.set_xlabel('geometric distance (raw)')
    fig.suptitle(suptitle, fontsize=14, x=.048, ha='left', y=.95, color=INK)
    fig.text(.048, .755, blurb, fontsize=9, color=INK2, ha='left')
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f'  [fig] {path}')
    return lr


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rep', default='voxel16')
    a_ = ap.parse_args(); rep = a_.rep
    d = load(rep); X = 'blind_k50'
    tag = DESC.get(rep, rep)
    print(f'\n########## {rep}: {tag} ##########')

    # ---- fig36 equivalent: all trial x model
    lr = panel_set(
        d[X].values, d.ft.values,
        'every point: one trial × one model',
        'geometric distance to that\nmodel\'s training set (raw)',
        'that model\'s oddity margin\non that trial (raw)',
        f'Every trial × every model — representation: {rep}',
        f'Descriptor: {tag}. All {len(d):,} observations at low opacity, raw axes, '
        'nothing centred, no lines joining trials.\nRight: the same cloud summarised by '
        'k-means in 2-D (distance, margin), marker area ∝ cluster size, size-weighted fit '
        'through the centroids.',
        f'{OUT}/fig38_allpoints_{rep}.png', .07, 5)
    print(f'ALL POINTS      n={len(d):,}  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  '
          f'p={lr.pvalue:.2e}')

    # ---- fig37 equivalent: on-category only
    oc = d[d.category == d.own]
    lr = panel_set(
        oc[X].values, oc.ft.values,
        'on-category model only: one point per trial',
        'geometric distance from the trial\'s objects\nto its OWN category\'s training set (raw)',
        'on-category model\'s oddity\nmargin on that trial (raw)',
        f'On-category trials only — representation: {rep}',
        f'Descriptor: {tag}. Exactly one of the 12 fine-tuned models was trained on each '
        f'trial\'s own object category; keeping only that model gives {len(oc)} points,\n'
        'one per trial. Raw axes, nothing centred, no lines. Right-hand panels summarise '
        'the same points with k-means in 2-D, marker area ∝ cluster size.',
        f'{OUT}/fig39_oncategory_{rep}.png', .32, 17)
    x, y, pre = oc[X].values, oc.ft.values, oc.pre.values
    Z = np.column_stack([np.ones(len(oc)), pre])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    off = d[d.category != d.own]
    print(f'ON-CATEGORY     n={len(oc)}  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  '
          f'p={lr.pvalue:.2e}')
    print('  r(dist, PRETRAINED margin) %+.3f  p=%.3f   <- control, should be ~0'
          % stats.pearsonr(x, pre))
    print('  partial (ft | pretrained)  %+.3f  p=%.4f' % stats.pearsonr(rx, ry))
    print('  r(dist, advantage ft-pre)  %+.3f  p=%.4f' % stats.pearsonr(x, y - pre))
    print('OFF-CATEGORY    n=%d  r=%+.3f  p=%.2e' % ((len(off),)
          + stats.pearsonr(off[X], off.ft)))
    print('  group means: on  dist %.3f margin %.3f | off dist %.3f margin %.3f'
          % (oc[X].mean(), oc.ft.mean(), off[X].mean(), off.ft.mean()))
