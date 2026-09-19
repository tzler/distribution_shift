"""fig38/39 rebuilt: raw scatter + quantile bins on x. No clustering.

Bins are quantile bins on the x axis alone, so each bin's y IS the conditional mean
E[y | x in bin] -- unlike k-means, which partitions in x and y jointly and therefore
returns local modes of the joint density rather than a conditional mean.
"""
import argparse, ast, numpy as np, pandas as pd
from scipy import stats
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
DESC = {'d57': 'geom3d 57-d summary (ROTATION-INVARIANT — pose discarded)',
        'voxel8': 'raw 8³ occupancy grid, 512-d (pose-SENSITIVE)',
        'voxel16': 'raw 16³ occupancy grid, 4,096-d (pose-SENSITIVE)',
        'voxel32': 'raw 32³ occupancy grid, 32,768-d (pose-SENSITIVE)'}
NBINS = [10, 30, 50, 70]


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


def qbin(x, y, nb):
    q = pd.qcut(pd.Series(x), nb, labels=False, duplicates='drop').values
    ks = np.arange(q.max() + 1)
    bx = np.array([x[q == i].mean() for i in ks])
    by = np.array([y[q == i].mean() for i in ks])
    se = np.array([y[q == i].std(ddof=1) / np.sqrt((q == i).sum()) for i in ks])
    return bx, by, se, np.array([(q == i).sum() for i in ks])


def panel_set(x, y, title0, xlab0, ylab0, suptitle, blurb, path, alpha0, s0):
    lr = stats.linregress(x, y)
    fig, ax = plt.subplots(1, 5, figsize=(19.5, 4.6), sharex=True, sharey=False)
    fig.subplots_adjust(left=.048, right=.99, top=.63, bottom=.17, wspace=.26)
    a = ax[0]; style(a)
    a.scatter(x, y, s=s0, color=GEOM, alpha=alpha0, linewidths=0, zorder=3)
    xs = np.linspace(x.min(), x.max(), 40)
    a.plot(xs, lr.slope * xs + lr.intercept, color=GEOM, lw=2.2, alpha=.85, zorder=4)
    a.set_title(f'{title0}\nn = {len(x):,}   slope {lr.slope:+.3f}, r {lr.rvalue:+.3f}, '
                f'p {lr.pvalue:.1e}', loc='left')
    a.set_xlabel(xlab0); a.set_ylabel(ylab0)
    for a, nb in zip(ax[1:], NBINS):
        style(a)
        bx, by, se, cnt = qbin(x, y, nb)
        a.errorbar(bx, by, yerr=se, fmt='o', color=GEOM, ms=6, mfc=GEOM, mec=SURF,
                   mew=1.3, ecolor='#c6d6ec', elinewidth=1.4, capsize=0, zorder=4)
        a.plot(xs, lr.slope * xs + lr.intercept, color=GEOM, lw=2, alpha=.5, zorder=3)
        lo, hi = (by - se).min(), (by + se).max()
        pad = .12 * (hi - lo)
        a.set_ylim(lo - pad, hi + pad)
        rb = stats.pearsonr(bx, by)
        a.set_title(f'{len(bx)} quantile bins  (~{int(np.median(cnt))} pts/bin)\n'
                    f'binned r = {rb[0]:+.3f}', loc='left')
        a.set_xlabel('geometric distance (raw)')
        a.tick_params(labelleft=True)
    fig.suptitle(suptitle, fontsize=14, x=.048, ha='left', y=.95, color=INK)
    fig.text(.048, .755, blurb, fontsize=9, color=INK2, ha='left')
    for a in ax[1:]:
        a.set_ylabel('oddity margin (raw)')
    fig.savefig(path, dpi=300); plt.close(fig)
    print(f'  [fig] {path}')
    for nb in NBINS:
        bx, by, _, _ = qbin(x, y, nb)
        print(f'    {nb:>3d} bins -> binned r {stats.pearsonr(bx, by)[0]:+.3f}')
    return lr


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rep', default='voxel16')
    rep = ap.parse_args().rep
    d = load(rep); X = 'blind_k50'; tag = DESC.get(rep, rep)
    print(f'\n########## {rep}: {tag} ##########')

    lr = panel_set(
        d[X].values, d.ft.values,
        'every point: one trial × one model',
        'geometric distance to that\nmodel\'s training set (raw)',
        'that model\'s oddity margin\non that trial (raw)',
        f'Every trial × every model — representation: {rep}',
        f'Descriptor: {tag}. Left: all {len(d):,} observations, raw axes, nothing centred, '
        'no lines joining trials.\nRight: the same points in quantile bins on the x axis '
        '(error bars = SEM within bin), so each marker is the conditional mean margin at '
        'that distance. The line is the point-level fit — the SAME line in every panel; only the y RANGE is zoomed per panel, so read the tick labels.',
        f'{OUT}/fig40_allpoints_bins_{rep}.png', .16, 5)
    print(f'ALL POINTS      n={len(d):,}  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  '
          f'p={lr.pvalue:.2e}')

    oc = d[d.category == d.own]
    lr = panel_set(
        oc[X].values, oc.ft.values,
        'on-category model only: one point per trial',
        'geometric distance from the trial\'s objects\nto its OWN category\'s training set (raw)',
        'on-category model\'s oddity\nmargin on that trial (raw)',
        f'On-category trials only — representation: {rep}',
        f'Descriptor: {tag}. Exactly one of the 12 fine-tuned models was trained on each '
        f'trial\'s own object category; keeping only that model gives {len(oc)} points, one '
        'per trial.\nRaw axes, nothing centred, no lines. Right: quantile bins on x '
        '(error bars = SEM within bin); the line is the point-level fit — the SAME line in '
        'every panel. Only the y RANGE is zoomed per panel, so read the tick labels.',
        f'{OUT}/fig41_oncategory_bins_{rep}.png', .45, 18)
    x, y, pre = oc[X].values, oc.ft.values, oc.pre.values
    Z = np.column_stack([np.ones(len(oc)), pre])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    off = d[d.category != d.own]
    print(f'ON-CATEGORY     n={len(oc)}  slope={lr.slope:+.4f}  r={lr.rvalue:+.3f}  '
          f'p={lr.pvalue:.2e}')
    print('  r(dist, PRETRAINED margin) %+.3f  p=%.3f   <- control' % stats.pearsonr(x, pre))
    print('  partial (ft | pretrained)  %+.3f  p=%.4f' % stats.pearsonr(rx, ry))
    print('OFF-CATEGORY    n=%d  r=%+.3f  p=%.2e' % ((len(off),)
          + stats.pearsonr(off[X], off.ft)))
