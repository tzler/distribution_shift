"""
Binned figures for (1) the oddity-blind geometric distribution shift and
(2) geometric A-B similarity.

Bins are for DISPLAY ONLY. Every quoted r/p is trial-level (unbinned), because
quantile-binning inflates correlations roughly threefold.

Palette: dataviz reference palette, unmodified, colour by entity.
"""
import ast, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
os.makedirs(OUT, exist_ok=True)

SHIFT, PRE, ABSIM = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8985'
DPI = 300
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 9.5,
    'axes.labelsize': 9, 'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def binned(x, y, nb=20):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    bx = np.array([x[q == i].mean() for i in range(q.max() + 1)])
    by = np.array([y[q == i].mean() for i in range(q.max() + 1)])
    se = np.array([y[q == i].std() / max(np.sqrt((q == i).sum()), 1)
                   for i in range(q.max() + 1)])
    return bx, by, se


def panel(a, x, y, col, xlab, ylab, title):
    bx, by, se = binned(x.values, y.values)
    a.errorbar(bx, by, yerr=se, fmt='o-', color=col, lw=2, ms=7, mfc=col,
               mec=SURFACE, mew=2, ecolor='#d8d7d2', zorder=3)
    r, p = stats.pearsonr(x, y)
    style(a)
    a.set_xlabel(xlab); a.set_ylabel(ylab)
    a.set_title(f'{title}\nr = {r:+.3f}, p = {p:.0e}   (trial level, n={len(x)})',
                loc='left')


def load(dataset='shapenet'):
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o = pd.read_csv(f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_'
                    'lr2e-06_ep30_multi_similarity_seed42_train:_val:_lora_r16_alpha8_'
                    'dropout0.1_|45|---most-results-ood/ood_analysis/'
                    'ood_analysis_results.csv')
    o['trial'] = m['trial'].values
    o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
    d = pd.read_csv(f'{G}/out/absim_{dataset}.csv')
    # absim_*.csv already carries the behavioural columns; take only what is missing
    want = ['pretrained_oddity_margin', 'fine_tuned_oddity_margin']
    d = d.merge(o[['trial'] + want], on='trial', validate='1:1')
    return d.rename(columns={'ft_margin': 'ft_margin_dup'})


# ------------------------------------------------------------------ FIGURE A
def fig_shift(d):
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.2))
    fig.subplots_adjust(left=.06, right=.985, top=.83, bottom=.09, hspace=.52, wspace=.30)
    X = d['blind_k50']
    panel(ax[0, 0], X, d['fine_tuned_oddity_margin'], SHIFT,
          'geometric distribution shift (k=50)', 'fine-tuned oddity margin',
          'A  Model margin  ← the key relationship')
    panel(ax[0, 1], X, d['pretrained_oddity_margin'], PRE,
          'geometric distribution shift (k=50)', 'pretrained oddity margin',
          'B  Pretrained margin')
    panel(ax[0, 2], X, d['advantage'], SHIFT,
          'geometric distribution shift (k=50)', 'fine-tuning advantage',
          'C  Fine-tuning advantage')
    panel(ax[1, 0], X, d['human_accuracy'], SHIFT,
          'geometric distribution shift (k=50)', 'human accuracy', 'D  Human accuracy')
    panel(ax[1, 1], X, d['human_rt'], SHIFT,
          'geometric distribution shift (k=50)', 'human RT (ms)', 'E  Human RT')
    panel(ax[1, 2], X, d['fine_tuned_correct'], SHIFT,
          'geometric distribution shift (k=50)', 'fine-tuned accuracy',
          'F  Fine-tuned accuracy')
    fig.suptitle('Oddity-blind, hub-robust, norm-free GEOMETRIC distribution shift '
                 '— shapenet, n=706\n'
                 'no encoder in the measure; 20 quantile bins shown, statistics are '
                 'trial-level',
                 fontsize=12.5, x=.06, ha='left', y=.965, color=INK)
    p = f'{OUT}/fig4_blind_shift.png'
    fig.savefig(p, dpi=DPI); plt.close(fig); print('[fig]', p)


# ------------------------------------------------------------------ FIGURE B
def fig_absim(d):
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.2))
    fig.subplots_adjust(left=.06, right=.985, top=.83, bottom=.09, hspace=.52, wspace=.30)
    X = d['ab_sim']
    panel(ax[0, 0], X, d['human_accuracy'], ABSIM,
          'geometric similarity of A and B', 'human accuracy',
          'A  Human accuracy')
    panel(ax[0, 1], X, d['human_rt'], ABSIM,
          'geometric similarity of A and B', 'human RT (ms)', 'B  Human RT')
    panel(ax[0, 2], X, d['fine_tuned_oddity_margin'], ABSIM,
          'geometric similarity of A and B', 'fine-tuned oddity margin',
          'C  Fine-tuned margin')
    panel(ax[1, 0], X, d['pretrained_oddity_margin'], PRE,
          'geometric similarity of A and B', 'pretrained oddity margin',
          'D  Pretrained margin  ← insensitive')
    panel(ax[1, 1], X, d['fine_tuned_correct'], ABSIM,
          'geometric similarity of A and B', 'fine-tuned accuracy',
          'E  Fine-tuned accuracy')
    panel(ax[1, 2], X, d['pretrained_correct'], PRE,
          'geometric similarity of A and B', 'pretrained accuracy',
          'F  Pretrained accuracy  ← insensitive')
    fig.suptitle('Geometric similarity between the two trial objects — shapenet, n=706\n'
                 'a pure trial property (never touches the training bank); more similar '
                 '→ harder. Fine-tuning installs the sensitivity (top vs bottom row)',
                 fontsize=12.5, x=.06, ha='left', y=.965, color=INK)
    p = f'{OUT}/fig5_ab_similarity.png'
    fig.savefig(p, dpi=DPI); plt.close(fig); print('[fig]', p)


# ------------------------------------------------- FIGURE C: within-trial margin
def fig_within(d):
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    panel_df = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    rows = []
    for cat in panel_df.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'advantage', 'fine_tuned_oddity_margin']])
    p = panel_df.merge(pd.concat(rows), on=['trial', 'category'])

    def wthn(df, cols):
        out = df.copy()
        for c in cols:
            out[c] = (df[c] - df.groupby('trial')[c].transform('mean')
                      - df.groupby('category')[c].transform('mean') + df[c].mean())
        return out

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(left=.09, right=.98, top=.76, bottom=.15, wspace=.30)
    for a, dv, lab in [(ax[0], 'fine_tuned_oddity_margin', 'fine-tuned oddity margin'),
                       (ax[1], 'advantage', 'fine-tuning advantage')]:
        w = wthn(p, ['blind_k50', dv])
        a.axhline(0, color='#d8d7d2', lw=1, zorder=1)
        panel(a, w['blind_k50'], w[dv], SHIFT,
              'geometric shift (trial- and model-demeaned)', f'{lab} (demeaned)',
              f'{lab}')
    fig.suptitle('Within trial × 12 category-models: d(A,B) absorbed by construction\n'
                 '706 trials × 12 models = 8,472 observations',
                 fontsize=12, x=.09, ha='left', y=.955, color=INK)
    q = f'{OUT}/fig6_within_trial_blind.png'
    fig.savefig(q, dpi=DPI); plt.close(fig); print('[fig]', q)


if __name__ == '__main__':
    d = load('shapenet')
    fig_shift(d); fig_absim(d); fig_within(d)
