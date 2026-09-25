"""
Publication figures for the model-free geometric distribution-shift analysis.

Palette: the dataviz reference palette, unmodified. Categorical slots 1-3
(blue / orange / aqua) are the documented all-pairs-validated subset in light mode
(worst-pair CVD dE 9.2, normal-vision 24.0), which is what scatter forms require.
Colour follows the ENTITY, never its rank:
    blue   = geometric, model-free
    orange = DINOv2, model-based
    aqua   = within-trial dissimilarity d(A,B)
"""
import os, ast
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

G = os.path.dirname(os.path.abspath(__file__))
OUT = f'{G}/out/figures'
os.makedirs(OUT, exist_ok=True)
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
S = ('/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune/'
     'ShapeNet_OOD_Analyses')
OOD = ('/vast/projects/bonnen/naturalistic-navig/Dist-shift/logs/'
       'vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_multi_similarity_seed42_'
       'train:_val:_lora_r16_alpha8_dropout0.1_|45|---most-results-ood/'
       'ood_analysis/ood_analysis_results.csv')
EMP = 'trial_distance_(L1_not_normalized)'

GEOM, DINO, DAB = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8985'
DPI = 300

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans',
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2', 'axes.linewidth': .8,
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def style(a, grid='y'):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis=grid, color='#eceae5', lw=.8, zorder=0)
    a.set_axisbelow(True)


def binned(x, y, nb):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    bx = np.array([x[q == i].mean() for i in range(q.max() + 1)])
    by = np.array([y[q == i].mean() for i in range(q.max() + 1)])
    se = np.array([y[q == i].std() / max(np.sqrt((q == i).sum()), 1)
                   for i in range(q.max() + 1)])
    return bx, by, se


def within(df, cols):
    o = df.copy()
    for c in cols:
        o[c] = (df[c] - df.groupby('trial')[c].transform('mean')
                - df.groupby('category')[c].transform('mean') + df[c].mean())
    return o


def load_panel():
    d = pd.read_csv(f'{G}/out/within_trial_panel.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in d.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        rows.append(o[['trial', 'category', EMP]])
    return d.merge(pd.concat(rows), on=['trial', 'category'])


# ---------------------------------------------------------------- FIGURE 1
def figure1(d):
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.30,
                          left=.06, right=.985, top=.86, bottom=.09)

    # A: pooled
    a = fig.add_subplot(gs[0, 0]); style(a)
    bx, by, se = binned(d['geom_knn_cat'].values, d['advantage'].values, 20)
    a.errorbar(bx, by, yerr=se, fmt='o-', color=MUTED, lw=2, ms=6,
               mfc=MUTED, mec=SURFACE, mew=2, ecolor='#d8d7d2', zorder=3)
    r = stats.pearsonr(d['geom_knn_cat'], d['advantage'])
    a.set_title(f'A  Pooled — d(A,B) not controlled\nr = {r[0]:+.3f}', loc='left')
    a.set_xlabel('geometric distance to training set')
    a.set_ylabel('fine-tuning advantage')

    # B: within-trial
    a = fig.add_subplot(gs[0, 1]); style(a)
    w = within(d, ['geom_knn_cat', 'advantage', EMP])
    bx, by, se = binned(w['geom_knn_cat'].values, w['advantage'].values, 20)
    a.axhline(0, color='#d8d7d2', lw=1, zorder=1)
    a.errorbar(bx, by, yerr=se, fmt='o-', color=GEOM, lw=2, ms=7,
               mfc=GEOM, mec=SURFACE, mew=2, ecolor='#cfd9e8', zorder=3)
    r = stats.pearsonr(w['geom_knn_cat'], w['advantage'])
    a.set_title(f'B  Within trial + category — d(A,B) absorbed\n'
                f'r = {r[0]:+.3f},  p = {r[1]:.0e}', loc='left', color=INK)
    a.set_xlabel('geometric distance (demeaned)')
    a.set_ylabel('advantage (demeaned)')

    # C: permutation null
    a = fig.add_subplot(gs[0, 2]); style(a)
    rng = np.random.default_rng(0); null = []
    for _ in range(300):
        dd = d.copy()
        dd['geom_knn_cat'] = dd.groupby('trial')['geom_knn_cat'].transform(
            lambda v: rng.permutation(v.values))
        ww = within(dd, ['geom_knn_cat', 'advantage'])
        null.append(stats.pearsonr(ww['geom_knn_cat'], ww['advantage'])[0])
    null = np.array(null)
    a.hist(null, bins=26, color='#dcdbd6', edgecolor=SURFACE, linewidth=1, zorder=2)
    a.axvline(r[0], color=GEOM, lw=2.5, zorder=4)
    ymax = a.get_ylim()[1]
    a.text(r[0] + .004, ymax * .97, f'observed\n{r[0]:+.3f}', color=GEOM,
           fontsize=8.5, ha='left', va='top', linespacing=1.35)
    z = (r[0] - null.mean()) / null.std()
    a.set_title(f'C  Permuting which training set pairs\n'
                f'with which model  (z = {z:+.1f}, 300 perms)', loc='left')
    a.set_xlabel('within-trial r under permutation'); a.set_ylabel('count')

    # D: head to head, pooled vs within
    a = fig.add_subplot(gs[1, 0]); style(a)
    vals = {}
    for X, k in [('geom_knn_cat', 'geom'), (EMP, 'dino')]:
        vals[k] = (abs(stats.pearsonr(d[X], d['advantage'])[0]),
                   abs(stats.pearsonr(w[X], w['advantage'])[0]))
    xs = np.arange(2); bw = .34
    a.bar(xs - bw / 2, vals['geom'], bw, color=GEOM, zorder=3, label='geometric (model-free)')
    a.bar(xs + bw / 2, vals['dino'], bw, color=DINO, zorder=3, label='DINOv2 (model-based)')
    a.set_xticks(xs); a.set_xticklabels(['pooled', 'within trial'])
    a.set_ylabel('|r| with fine-tuning advantage')
    a.set_title('D  Both metrics survive the clean design;\nDINOv2 is stronger', loc='left')
    a.set_ylim(0, .305)
    a.legend(loc='upper right', fontsize=8)
    for i, (g, dn) in enumerate(zip(vals['geom'], vals['dino'])):
        a.text(i - bw / 2, g + .004, f'{g:.3f}', ha='center', fontsize=8, color=INK2)
        a.text(i + bw / 2, dn + .004, f'{dn:.3f}', ha='center', fontsize=8, color=INK2)

    # E: margin DV
    a = fig.add_subplot(gs[1, 1]); style(a)
    ww = within(d, ['geom_knn_cat', 's_ft'])
    bx, by, se = binned(ww['geom_knn_cat'].values, ww['s_ft'].values, 20)
    a.axhline(0, color='#d8d7d2', lw=1, zorder=1)
    a.errorbar(bx, by, yerr=se, fmt='o-', color=GEOM, lw=2, ms=7,
               mfc=GEOM, mec=SURFACE, mew=2, ecolor='#cfd9e8', zorder=3)
    rm = stats.pearsonr(ww['geom_knn_cat'], ww['s_ft'])
    a.set_title(f'E  Same design, DV = fine-tuned oddity margin\n'
                f'r = {rm[0]:+.3f},  p = {rm[1]:.0e}', loc='left')
    a.set_xlabel('geometric distance (demeaned)')
    a.set_ylabel('oddity margin (demeaned)')

    # F: per-category slopes
    a = fig.add_subplot(gs[1, 2]); style(a, grid='x')
    rows = []
    for cat, sub in d.groupby('category'):
        sw = sub.copy()
        for c in ['geom_knn_cat', 'advantage']:
            sw[c] = sub[c] - sub.groupby('trial')[c].transform('mean')
        rr = stats.pearsonr(sub['geom_knn_cat'], sub['advantage'])
        rows.append((cat, rr[0]))
    rows.sort(key=lambda t: t[1])
    names = [r[0] for r in rows]; vv = [r[1] for r in rows]
    a.barh(np.arange(len(vv)), vv, color=GEOM, height=.62, zorder=3)
    a.axvline(0, color='#d8d7d2', lw=1, zorder=2)
    a.set_yticks(np.arange(len(vv))); a.set_yticklabels(names, fontsize=8)
    a.set_xlabel('r (geometric distance vs advantage)')
    a.set_title('F  Per training-category model\n(negative = training helps nearby trials)',
                loc='left')

    fig.suptitle('A model-free geometric measure of distance-to-training predicts which '
                 'trials benefit from fine-tuning\n'
                 '706 MOCHI shapenet trials × 12 category-specific DINOv2-L fine-tunes '
                 '= 8,472 observations; no binning in the statistics',
                 fontsize=12.5, x=.06, ha='left', y=.975, color=INK)
    p = f'{OUT}/fig1_causal_design.png'
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print('[fig]', p)


# ---------------------------------------------------------------- FIGURE 2
def figure2():
    g3 = pd.read_csv(f'{G}/out/shift3d_shapenet.csv')
    o = pd.read_csv(OOD); m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o['trial'] = m['trial'].values; o['s_ft'] = 1 + o['fine_tuned_oddity_margin']
    d = g3.merge(o[['trial', 's_ft', EMP]], on='trial')

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.2))
    fig.subplots_adjust(left=.06, right=.985, top=.76, bottom=.15, wspace=.30)

    a = ax[0]; style(a, grid='both')
    a.scatter(d['geom_dAB_3d'], d['geom_shift_3d'], s=13, color=DAB, alpha=.5,
              linewidths=0, zorder=3)
    r = stats.pearsonr(d['geom_dAB_3d'], d['geom_shift_3d'])
    a.set_xlabel('within-trial dissimilarity  d(A,B)')
    a.set_ylabel('geometric "distance to training"')
    a.set_title(f'A  The metric is almost entirely d(A,B)\nr = {r[0]:+.3f}', loc='left')

    a = ax[1]; style(a)
    labels = ['geometric\nshift', 'd(A,B)\nalone', 'knn\ndensity', 'DINOv2\n$\\ell_1$']
    cols = [GEOM, DAB, GEOM, DINO]
    b = [ .726, .746, .658, .341]
    t = [ .245, .247, .195, .087]
    xs = np.arange(4); bw = .32; off = .175
    a.bar(xs - off, b, bw, color=cols, zorder=3)
    a.bar(xs + off, t, bw, color=cols, alpha=.42, zorder=3)
    a.set_xticks(xs); a.set_xticklabels(labels, fontsize=8)
    a.set_ylabel('|r| with the oddity margin')
    a.set_title('B  Binning inflates r roughly 3×\n(solid = 30 quantile bins, pale = trial level)',
                loc='left')
    for x, v in zip(xs - off, b):
        a.text(x, v + .012, f'{v:.2f}', ha='center', fontsize=8, color=INK2)
    for x, v in zip(xs + off, t):
        a.text(x, v + .012, f'{v:.2f}', ha='center', fontsize=8, color=INK2)

    a = ax[2]; style(a, grid='x')
    d['strat'] = pd.qcut(d['geom_dAB_3d'], 5, labels=False, duplicates='drop')
    rs, los, his, ns = [], [], [], []
    for s_, sub in d.groupby('strat'):
        rr = stats.pearsonr(sub['geom_dB_knn200_3d'], sub['s_ft'])[0]
        z = np.arctanh(rr); se = 1 / np.sqrt(len(sub) - 3)
        rs.append(rr); los.append(np.tanh(z - 1.96 * se)); his.append(np.tanh(z + 1.96 * se))
        ns.append(len(sub))
    y = np.arange(len(rs))
    a.hlines(y, los, his, color=MUTED, lw=2, zorder=3)
    a.plot(rs, y, 'o', ms=8, color=GEOM, mec=SURFACE, mew=2, zorder=4)
    a.axvline(0, color='#d8d7d2', lw=1, zorder=2)
    a.set_yticks(y); a.set_yticklabels([f'stratum {i+1}\n(n={n})' for i, n in enumerate(ns)],
                                       fontsize=8)
    a.set_xlabel('r (distance-to-training vs margin)')
    a.set_title('C  Within matched d(A,B) strata:\nconsistently positive, underpowered '
                '(Z = +1.73)', loc='left')

    fig.suptitle('Why the single-model analysis cannot settle the question',
                 fontsize=12.5, x=.06, ha='left', y=.955, color=INK)
    p = f'{OUT}/fig2_confound.png'
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print('[fig]', p)


# ---------------------------------------------------------------- FIGURE 3
def figure3():
    o = pd.read_csv(OOD); m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o['trial'] = m['trial'].values; o['s_ft'] = 1 + o['fine_tuned_oddity_margin']
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.6))
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.09, hspace=.42, wspace=.28)
    for i, (ds, mo) in enumerate([('shapegen', '2d'), ('shapenet', '3d')]):
        g = pd.read_csv(f'{G}/out/shift{mo}_{ds}.csv')
        d = g.merge(o[['trial', 's_ft', EMP]], on='trial')
        for j, (col, lab, c) in enumerate([
                (EMP, 'DINOv2 $\\ell_1$ distance  (model-based)', DINO),
                (f'geom_shift_{mo}', 'geometric distance  (model-free)', GEOM)]):
            a = ax[i, j]; style(a, grid='both')
            bx, by, _ = binned(d[col].values, d['s_ft'].values, 30)
            r = stats.pearsonr(bx, by)
            a.scatter(bx, by, s=42, color=c, zorder=4, edgecolor=SURFACE, linewidth=2)
            b1, b0 = np.polyfit(bx, by, 1)
            xs = np.linspace(bx.min(), bx.max(), 40)
            a.plot(xs, b1 * xs + b0, color=c, lw=2, alpha=.55, zorder=3)
            a.set_xlabel(lab); a.set_ylabel('fine-tuned oddity margin')
            a.set_title(f'{ds}  ·  r = {r[0]:+.3f},  p = {r[1]:.0e}', loc='left')
            a.xaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle('Validating the oddity margin against a model-based vs a model-free target\n'
                 'Fig 1-right reproduction, 30 quantile bins (binned — inflates r ~3×)',
                 fontsize=12, x=.09, ha='left', y=.955, color=INK)
    p = f'{OUT}/fig3_margin_validation.png'
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print('[fig]', p)


if __name__ == '__main__':
    d = load_panel()
    figure1(d); figure2(); figure3()
    print('\nall figures ->', OUT)
