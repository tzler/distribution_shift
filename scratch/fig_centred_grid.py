"""The 3-row grid re-plotted with the X AXIS CENTRED. y is never touched.

Columns: raw x | x centred by TRIAL | x centred by CATEGORY | x centred by BOTH.

IMPORTANT. The within-category row holds exactly ONE observation per trial (the single
model trained on that trial's own category), so subtracting each trial's own mean x
sends every point to exactly zero. Trial-centring is undefined for that row -- those
panels are marked rather than drawn. Category-centring is defined everywhere.

Centring only x is deliberate: within a trial the centred x sums to zero, so its
covariance with any trial-constant is zero and the SLOPE is identical whether or not y
is centred too. Only r changes. y therefore stays raw and the vertical scale stays
comparable across the row.
"""
import argparse, ast, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'

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
YS = {'margin': ('fine_tuned_oddity_margin', 'fine-tuned oddity margin'),
      'margin_adv': ('margin_adv', 'margin advantage (fine-tuned − base)')}
NB = 50


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def load(rep):
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_{rep}_percat.csv')
    m = pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['margin_adv'] = (o['fine_tuned_oddity_margin']
                           - o['pretrained_oddity_margin'])
        rows.append(o[['trial', 'category', 'fine_tuned_oddity_margin', 'margin_adv']])
    d = p.merge(pd.concat(rows), on=['trial', 'category'])
    own = {}
    for _, r in m[m.dataset == 'shapenet'].iterrows():
        s = {fn[:-4].split('_')[0] for fn in ast.literal_eval(r['images'])}
        if len(s) == 1:
            own[r['trial']] = INV.get(list(s)[0])
    d['own'] = d.trial.map(own)
    return d.dropna(subset=['own'])


def centre(df, col, bys):
    v = df[col].values.astype(float).copy()
    for by in bys:
        s = pd.Series(v, index=df.index)
        v = v - s.groupby(df[by]).transform('mean').values
    return v


def qbin(x, y, nb):
    q = pd.qcut(pd.Series(x), nb, labels=False, duplicates='drop').values
    ks = np.arange(q.max() + 1)
    bx = np.array([x[q == i].mean() for i in ks])
    by = np.array([y[q == i].mean() for i in ks])
    se = np.array([y[q == i].std(ddof=1) / np.sqrt(max((q == i).sum(), 2)) for i in ks])
    return bx, by, se


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rep', default='voxel16'); ap.add_argument('--y', default='margin')
    a_ = ap.parse_args()
    ycol, ylab = YS[a_.y]
    d = load(a_.rep); X = 'blind_k50'
    ROWS = [('within category', d[d.category == d.own]),
            ('across category', d[d.category != d.own]),
            ('all trials',      d)]
    COLS = [('x RAW', []), ('x centred by TRIAL', ['trial']),
            ('x centred by CATEGORY', ['category']),
            ('x centred by BOTH', ['trial', 'category'])]
    print(f'\n##### {a_.rep} | y = {ylab} | {NB} bins #####')
    print(f'{"row":17s}' + ''.join(f'{c:>26s}' for c, _ in COLS))

    fig, ax = plt.subplots(3, 4, figsize=(17.6, 12.0))
    fig.subplots_adjust(left=.056, right=.988, top=.845, bottom=.055,
                        hspace=.34, wspace=.26)
    for i, (rlbl, sub) in enumerate(ROWS):
        y = sub[ycol].values.astype(float)
        line = f'{rlbl:17s}'
        for j, (clbl, bys) in enumerate(COLS):
            a = ax[i, j]; style(a)
            x = centre(sub, X, bys)
            if np.std(x) < 1e-9:                    # trial-centring a 1-point-per-trial row
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                a.text(.5, .55, 'UNDEFINED', ha='center', va='center',
                       fontsize=15, color=GREY, weight='bold', transform=a.transAxes)
                a.text(.5, .40, 'this row has exactly 1 point per trial,\n'
                                'so subtracting the trial mean\nsends every x to zero',
                       ha='center', va='top', fontsize=9, color=INK2,
                       transform=a.transAxes)
                a.set_title(f'{"abc"[i]}{j+1}  {clbl}', loc='left', color=GREY)
                line += f'{"undefined":>26s}'
                continue
            lr = stats.linregress(x, y)
            bx, by, se = qbin(x, y, NB)
            a.errorbar(bx, by, yerr=se, fmt='o', color=GEOM, ms=9, mfc=GEOM, mec=SURF,
                       mew=1.5, ecolor='#bcd0ea', elinewidth=1.5, capsize=0, zorder=4)
            if bys:
                a.axvline(0, color='#dcdad5', lw=1.2, zorder=1)
            lo, hi = (by - se).min(), (by + se).max(); pad = .12 * (hi - lo)
            a.set_ylim(lo - pad, hi + pad)
            rb = stats.pearsonr(bx, by)[0]
            a.set_title(f'{"abc"[i]}{j+1}  {clbl}\npoint r = {lr.rvalue:+.3f}, '
                        f'p = {lr.pvalue:.1e}   binned r = {rb:+.3f}', loc='left')
            a.set_xlabel('geometric distance' + (' (centred)' if bys else ' (raw)'))
            a.set_ylabel(f'{rlbl}\n{ylab}' if j == 0 else ylab)
            line += f'{f"{lr.rvalue:+.3f} / {rb:+.3f}":>26s}'
        print(line, flush=True)

    fig.suptitle(f'Centring the x axis — {ylab}, {a_.rep}, {NB} quantile bins',
                 fontsize=15, x=.056, ha='left', y=.975, color=INK)
    fig.text(.056, .888,
             'y is the untransformed margin in every panel — only x changes. Within a '
             'trial the centred x sums to zero, so its covariance with any trial-constant '
             'is zero and the SLOPE is\nidentical whether or not y is centred too; only r '
             'differs. Trial-centring removes trial difficulty (d(A,B) is constant within '
             'a trial); category-centring removes per-model offsets.\n'
             'The within-category row holds ONE observation per trial, so trial-centring '
             'is undefined there — those panels are marked, not drawn. '
             'Titles give point-level r then binned r.',
             fontsize=9.2, color=INK2, ha='left')
    q = f'{OUT}/fig44_centred_{a_.y}_{a_.rep}.png'
    fig.savefig(q, dpi=300); plt.close(fig); print(f'\n[fig] {q}')
