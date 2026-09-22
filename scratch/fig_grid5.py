"""3 x 5 grid: rows = trial subset; columns = raw scatter, 100 / 50 / 10 bins, then a
grey CONTROL column showing base DINOv2 (never fine-tuned) at 50 bins.

Rows    top    within category  (the model trained on this trial's own category)
        middle across category  (the 11 models trained on other categories)
        bottom all trials       (both pooled)
Columns a) every point  b) 100 bins  c) 50 bins  d) 10 bins  e) BASE DINOv2, 50 bins

The grey column is the control: base DINOv2 never saw any of the fine-tuning sets, so
any dependence on geometric distance there is NOT attributable to training exposure.
A flat grey column licenses the blue ones; a sloped grey column means that row's
effect is confounded with how atypical the objects are.
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
GEOM, GREY = '#2a78d6', '#8a8884'
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
# y variable -> (column, axis label, short tag for the filename)
YS = {'margin':     ('fine_tuned_oddity_margin', 'fine-tuned oddity margin', 'margin'),
      'acc':        ('fine_tuned_correct',       'fine-tuned accuracy (0/1)', 'acc'),
      'margin_adv': ('margin_adv',   'margin advantage  (fine-tuned − base)', 'marginadv'),
      'acc_adv':    ('acc_adv',      'accuracy advantage  (fine-tuned − base)', 'accadv'),
      'human_acc':  ('human_accuracy', 'human accuracy', 'humanacc'),
      'human_rt':   ('human_rt',     'human RT (s)', 'humanrt'),
      'gap':        ('human_minus_model', 'human − model accuracy', 'gap')}
NBINS = [100, 50, 10]
KEEP = ['human_accuracy', 'human_rt', 'pretrained_correct', 'pretrained_oddity_margin',
        'fine_tuned_correct', 'fine_tuned_oddity_margin']


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
        rows.append(o[['trial', 'category'] + KEEP])
    d = p.merge(pd.concat(rows), on=['trial', 'category'])
    own = {}
    for _, r in m[m.dataset == 'shapenet'].iterrows():
        s = {fn[:-4].split('_')[0] for fn in ast.literal_eval(r['images'])}
        if len(s) == 1:
            own[r['trial']] = INV.get(list(s)[0])
    d['own'] = d.trial.map(own)
    d = d.dropna(subset=['own'])
    d['margin_adv'] = d.fine_tuned_oddity_margin - d.pretrained_oddity_margin
    d['acc_adv'] = d.fine_tuned_correct - d.pretrained_correct
    d['human_minus_model'] = d.human_accuracy - d.fine_tuned_correct
    return d


def qbin(x, y, nb):
    q = pd.qcut(pd.Series(x), nb, labels=False, duplicates='drop').values
    ks = np.arange(q.max() + 1)
    bx = np.array([x[q == i].mean() for i in ks])
    by = np.array([y[q == i].mean() for i in ks])
    se = np.array([y[q == i].std(ddof=1) / np.sqrt(max((q == i).sum(), 2)) for i in ks])
    return bx, by, se, np.array([(q == i).sum() for i in ks])


def draw_bins(a, x, y, nb, colour, xlim):
    bx, by, se, cnt = qbin(x, y, nb)
    a.errorbar(bx, by, yerr=se, fmt='o', color=colour, ms=9.5, mfc=colour, mec=SURF,
               mew=1.5, ecolor=('#bcd0ea' if colour == GEOM else '#d6d4cf'),
               elinewidth=1.5, capsize=0, zorder=4)
    lo, hi = (by - se).min(), (by + se).max()
    pad = .12 * (hi - lo)
    a.set_ylim(lo - pad, hi + pad); a.set_xlim(*xlim)
    return stats.pearsonr(bx, by)[0], int(np.median(cnt)), len(bx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', default='voxel16')
    ap.add_argument('--y', default='margin', choices=list(YS))
    a_ = ap.parse_args(); rep, ykey = a_.rep, a_.y
    ycol, ylab, ytag = YS[ykey]
    d = load(rep); X = 'blind_k50'; tag = DESC.get(rep, rep)
    CTRL, CTRLLAB = 'pretrained_oddity_margin', 'BASE DINOv2 oddity margin'
    ROWS = [('within category', d[d.category == d.own], .34, 16),
            ('across category', d[d.category != d.own], .10, 7),
            ('all trials',      d,                      .10, 7)]
    print(f'\n########## {rep} | y = {ylab} ##########')

    fig, ax = plt.subplots(3, 5, figsize=(22.4, 12.4))
    fig.subplots_adjust(left=.046, right=.988, top=.845, bottom=.055,
                        hspace=.34, wspace=.26)
    for i, (lbl, sub, alpha0, s0) in enumerate(ROWS):
        x, y = sub[X].values, sub[ycol].values
        lr = stats.linregress(x, y)
        xlo, xhi = x.min(), x.max(); pad = .04 * (xhi - xlo)
        xlim = (xlo - pad, xhi + pad)

        a = ax[i, 0]; style(a)
        a.scatter(x, y, s=s0, color=GEOM, alpha=alpha0, linewidths=0, zorder=3)
        a.set_xlim(*xlim)
        a.set_title(f'{"abc"[i]}1  {lbl} — every point\nn = {len(x):,}   '
                    f'r = {lr.rvalue:+.3f}, p = {lr.pvalue:.1e}', loc='left')
        a.set_ylabel(f'{lbl}\n{ylab}'); a.set_xlabel('geometric distance (raw)')
        print(f'{lbl:16s} n={len(x):>6,}  r={lr.rvalue:+.3f}  p={lr.pvalue:.2e}', end='')

        for j, nb in enumerate(NBINS):
            a = ax[i, j + 1]; style(a)
            rb, med, nn = draw_bins(a, x, y, nb, GEOM, xlim)
            a.set_title(f'{"abc"[i]}{j+2}  {nn} bins  (~{med} pts/bin)\n'
                        f'binned r = {rb:+.3f}', loc='left')
            a.set_xlabel('geometric distance (raw)'); a.set_ylabel(ylab)
            print(f' | {nb}b {rb:+.3f}', end='')

        # ---- grey control column: base DINOv2, never fine-tuned, 50 bins
        a = ax[i, 4]; style(a)
        yc = sub[CTRL].values
        rb, med, nn = draw_bins(a, x, yc, 50, GREY, xlim)
        rp = stats.pearsonr(x, yc)
        a.set_title(f'{"abc"[i]}5  CONTROL — base DINOv2, {nn} bins\n'
                    f'binned r = {rb:+.3f}   point r = {rp[0]:+.3f}, p = {rp[1]:.0e}',
                    loc='left', color=INK2)
        a.set_xlabel('geometric distance (raw)'); a.set_ylabel(CTRLLAB)
        print(f' || CONTROL 50b {rb:+.3f}  point {rp[0]:+.3f} p={rp[1]:.1e}', flush=True)

    fig.suptitle(f'Geometric distance to the training set vs {ylab}   —   '
                 f'representation: {rep}', fontsize=15, x=.046, ha='left', y=.975,
                 color=INK)
    fig.text(.046, .893,
             f'Descriptor: {tag}. Every MOCHI ShapeNet trial is run through 12 models, one '
             'fine-tuned per ShapeNet category, so one trial contributes 12 observations.\n'
             'WITHIN CATEGORY keeps only the model trained on that trial\'s own object '
             'category (706 points, one per trial); ACROSS CATEGORY keeps the other 11 '
             '(7,766); ALL TRIALS pools both (8,472).\n'
             'Column 1 shows every observation on raw axes — nothing centred, no lines. '
             'Columns 2-4 are quantile bins on the x axis alone, so each marker is the mean '
             'of the trials in that distance bin and the error bar is its SEM.\n'
             'Column 5 (GREY) is the control: base DINOv2, which never saw any fine-tuning '
             'set. A flat grey panel licenses the blue ones in that row; a sloped one means '
             'the effect is not attributable to training exposure.',
             fontsize=9.2, color=INK2, ha='left')
    q = f'{OUT}/fig43_grid5_{ytag}_{rep}.png'
    fig.savefig(q, dpi=300); plt.close(fig)
    print(f'\n[fig] {q}')
