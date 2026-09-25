"""
The two distribution-shift metrics, in their final form, side by side.

TOP ROW    geometric, model-free:  oddity-blind / hub-robust / norm-free
             shift(t) = (1/|t|) sum_x (1/k) sum_{i in kNN_k(x)} (1 - g(x).C_i),  k=50
             g = 57-d binvox shape descriptor, z-scored then L2-normalised, cosine
BOTTOM ROW incumbent, DINOv2 ViT-L/14 reg4:
             shift = min_i 1/2[ ||phi(A)-C_i||_1 + ||phi(B)-C_i||_1 ],  raw L1

Columns:
  1  fine-tuned oddity margin        2  fine-tuning advantage        3  what it is confounded with

Columns 1-2 use the within-trial rank design: 12 points, each containing the same 706
trials, so d(A,B) and trial difficulty are identical across the axis. Raw axes.
Column 3 is trial level, |r| with the quantities a shift measure should NOT be tracking.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
L = f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
EMP = 'trial_distance_(L1_not_normalized)'
DABm = 'd_AB_(L1_not_normalized)'
GEOM, DINO = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8985'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 10,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def style(a, grid='both'):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis=grid, color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def panel_data():
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'ft_margin', 'advantage', EMP, DABm]])
    return p.merge(pd.concat(rows), on=['trial', 'category'])


def trial_level():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o = pd.read_csv(f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_'
                    'lr2e-06_ep30_multi_similarity_seed42_train:_val:_lora_r16_alpha8_'
                    'dropout0.1_|45|---most-results-ood/ood_analysis/'
                    'ood_analysis_results.csv')
    o['trial'] = m['trial'].values
    d = pd.read_csv(f'{G}/out/blindshift_shapenet.csv')[['trial', 'blind_k50']]
    d = d.merge(o[['trial', EMP, DABm]], on='trial', validate='1:1')
    ab = pd.read_csv(f'{G}/out/absim_shapenet.csv')[['trial', 'ab_sim']]
    d = d.merge(ab, on='trial', validate='1:1')
    t = pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv')
    return d.merge(t[['trial', 'feat_l1', 'pix_l1', 'coverage', 'contrast']],
                   on='trial', validate='1:1')


def rank_curve(d, X, dv):
    d = d.copy()
    d['rank'] = d.groupby('trial')[X].rank(method='first').astype(int)
    g = d.groupby('rank')
    return pd.DataFrame({'x': g[X].mean(), 'y': g[dv].mean(),
                         'yse': g[dv].std() / np.sqrt(g[dv].size())}).reset_index()


def main():
    d = panel_data()
    tl = trial_level()
    CONF = [('ab_sim', 'A–B similarity'), (DABm, 'd(A,B) in DINOv2 space'),
            ('feat_l1', 'feature magnitude'), ('pix_l1', 'pixel $\\ell_1$'),
            ('coverage', 'coverage'), ('contrast', 'contrast')]

    fig, ax = plt.subplots(2, 3, figsize=(14.6, 8.4))
    fig.subplots_adjust(left=.105, right=.985, top=.80, bottom=.08,
                        hspace=.46, wspace=.31)

    for r, (X, name, col, xlab) in enumerate([
            ('blind_k50', 'GEOMETRIC  (model-free)', GEOM,
             'geometric distance to training set  (cosine)'),
            (EMP, 'INCUMBENT  (DINOv2 $\\ell_1$)', DINO,
             'DINOv2 $\\ell_1$ distance to training set')]):

        for c, (dv, dlab) in enumerate([('ft_margin', 'fine-tuned oddity margin'),
                                        ('advantage', 'fine-tuning advantage')]):
            a = ax[r, c]; style(a)
            cur = rank_curve(d, X, dv)
            if dv == 'advantage':
                a.axhline(0, color='#c9c7c1', lw=1.2, zorder=2)
            a.errorbar(cur.x, cur.y, yerr=cur.yse, fmt='o-', color=col, lw=2, ms=8,
                       mfc=col, mec=SURFACE, mew=2, ecolor='#d8d7d2', zorder=4)
            sl = np.array([stats.linregress(s[X], s[dv]).slope
                           for _, s in d.groupby('trial') if s[X].std() > 0])
            t = stats.ttest_1samp(sl, 0)
            a.set_xlabel(xlab); a.set_ylabel(dlab)
            a.set_title(f'{cur.y.iloc[0]:.3f} → {cur.y.iloc[-1]:.3f}   '
                        f'(t = {t.statistic:+.1f})', loc='left')

        # column 3: confound profile
        a = ax[r, 2]; style(a, grid='x')
        vals = [abs(stats.pearsonr(tl[X], tl[c])[0]) for c, _ in CONF]
        y = np.arange(len(CONF))
        a.barh(y, vals, color=col, height=.62, zorder=3)
        a.axvline(0, color='#d8d7d2', lw=1)
        a.set_yticks(y); a.set_yticklabels([l for _, l in CONF], fontsize=8.5)
        a.invert_yaxis()
        a.set_xlim(0, 1.0)
        a.set_xlabel('|r|  (trial level)')
        a.set_title('what it is confounded with\n(lower is better)', loc='left')
        for yi, v in zip(y, vals):
            a.text(v + .02, yi, f'{v:.2f}', va='center', fontsize=8, color=INK2)

        # row label in the left margin, rotated: cannot collide with any panel text
        pos = ax[r, 0].get_position()
        fig.text(.022, (pos.y0 + pos.y1) / 2, name, fontsize=12.5, color=col,
                 ha='center', va='center', rotation=90, weight='bold')

    fig.suptitle('The two distribution-shift metrics in final form',
                 fontsize=14, x=.105, ha='left', y=.975, color=INK)
    fig.text(.105, .885,
             'Columns 1–2: within-trial rank design — 12 points, each containing the same '
             '706 trials, so d(A,B) and trial difficulty are identical across the axis. '
             'Raw axes, nothing demeaned.\n'
             'Column 3: trial level. A distribution-shift measure should not be tracking '
             'these.  n = 706 trials × 12 category-specific fine-tunes.',
             fontsize=9.5, color=INK2, ha='left')

    p = f'{OUT}/fig11_final_metric_comparison.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)
    print('\nconfound profile (|r|, trial level):')
    print(f'  {"":26s}{"GEOMETRIC":>12s}{"INCUMBENT":>12s}')
    for c, lab in CONF:
        print(f'  {lab:26s}{abs(stats.pearsonr(tl["blind_k50"], tl[c])[0]):>12.3f}'
              f'{abs(stats.pearsonr(tl[EMP], tl[c])[0]):>12.3f}')


if __name__ == '__main__':
    main()
