"""3 x 4 grid: rows = trial subset, columns = raw scatter then 100 / 50 / 10 bins.

Rows    top    within category  (the model trained on this trial's own category)
        middle across category  (the 11 models trained on other categories)
        bottom all trials       (both of the above pooled)
Columns a) every point, b) 100 quantile bins, c) 50 bins, d) 10 bins

Bins are quantile bins on x alone, so each marker is E[y | x in bin]. No trend lines.
y autoscales per panel; x is shared down each column-of-a-row via the row's own range.
"""
import argparse, ast, numpy as np, pandas as pd
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
NBINS = [100, 50, 10]


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
    se = np.array([y[q == i].std(ddof=1) / np.sqrt(max((q == i).sum(), 2)) for i in ks])
    return bx, by, se, np.array([(q == i).sum() for i in ks])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rep', default='voxel16')
    rep = ap.parse_args().rep
    d = load(rep); X = 'blind_k50'; tag = DESC.get(rep, rep)
    ROWS = [('within category', d[d.category == d.own], .34, 16),
            ('across category', d[d.category != d.own], .10, 7),
            ('all trials',      d,                      .10, 7)]
    print(f'\n########## {rep}: {tag} ##########')

    fig, ax = plt.subplots(3, 4, figsize=(17.6, 12.4))
    fig.subplots_adjust(left=.055, right=.988, top=.845, bottom=.055,
                        hspace=.34, wspace=.24)
    for i, (lbl, sub, alpha0, s0) in enumerate(ROWS):
        x, y = sub[X].values, sub.ft.values
        lr = stats.linregress(x, y)
        xlo, xhi = x.min(), x.max()
        pad = .04 * (xhi - xlo)

        a = ax[i, 0]; style(a)
        a.scatter(x, y, s=s0, color=GEOM, alpha=alpha0, linewidths=0, zorder=3)
        a.set_xlim(xlo - pad, xhi + pad)
        a.set_title(f'{"abc"[i]}1  {lbl} — every point\nn = {len(x):,}   '
                    f'r = {lr.rvalue:+.3f}, p = {lr.pvalue:.1e}', loc='left')
        a.set_ylabel(f'{lbl}\noddity margin (raw)')
        a.set_xlabel('geometric distance (raw)')
        print(f'{lbl:16s} n={len(x):>6,}  r={lr.rvalue:+.3f}  p={lr.pvalue:.2e}', end='')

        for j, nb in enumerate(NBINS):
            a = ax[i, j + 1]; style(a)
            bx, by, se, cnt = qbin(x, y, nb)
            a.errorbar(bx, by, yerr=se, fmt='o', color=GEOM, ms=9.5, mfc=GEOM,
                       mec=SURF, mew=1.5, ecolor='#bcd0ea', elinewidth=1.5,
                       capsize=0, zorder=4)
            lo, hi = (by - se).min(), (by + se).max()
            p2 = .12 * (hi - lo)
            a.set_ylim(lo - p2, hi + p2)
            a.set_xlim(xlo - pad, xhi + pad)
            rb = stats.pearsonr(bx, by)[0]
            a.set_title(f'{"abc"[i]}{j+2}  {len(bx)} bins  (~{int(np.median(cnt))} pts/bin)'
                        f'\nbinned r = {rb:+.3f}', loc='left')
            a.set_xlabel('geometric distance (raw)')
            a.set_ylabel('oddity margin (raw)')
            print(f' | {nb}b {rb:+.3f}', end='')
        print(flush=True)

    fig.suptitle('Geometric distance to the training set vs oddity margin — '
                 f'representation: {rep}', fontsize=15, x=.055, ha='left', y=.975,
                 color=INK)
    fig.text(.055, .893,
             f'Descriptor: {tag}. Every MOCHI ShapeNet trial is run through 12 models, one '
             'fine-tuned per ShapeNet category, so one trial contributes 12 observations.\n'
             'WITHIN CATEGORY keeps only the model trained on that trial\'s own object '
             'category (706 points, one per trial); ACROSS CATEGORY keeps the other 11 '
             '(7,766); ALL TRIALS pools both (8,472).\n'
             'Column 1 shows every observation on raw axes — nothing centred, no lines. '
             'Columns 2-4 are quantile bins on the x axis alone, so each marker is the mean '
             'margin of the trials in that distance bin\n'
             'and the error bar is its SEM. No trend lines. The y range is free in every '
             'panel; read the tick labels.',
             fontsize=9.2, color=INK2, ha='left')
    q = f'{OUT}/fig42_grid_{rep}.png'
    fig.savefig(q, dpi=300); plt.close(fig)
    print(f'\n[fig] {q}')
