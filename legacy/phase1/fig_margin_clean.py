"""
Fig 1-right / "fig3" WITHOUT the confound.

Same question and same visual form as the original margin-validation figure — does the
oddity margin track distance-to-training? — but the trial is held fixed and the TRAINING
SET is varied across the 12 per-category fine-tunes.

Why this fixes it: d(A,B), the oddity task's own decision variable, is a property of the
stimulus. It has exactly ZERO within-trial variance, so trial fixed effects remove it by
construction rather than by statistical control. In the original single-model form it
cannot be removed at all — three attempts (oddity-blind estimator, stratification,
residualisation) all leave the relationship either leak-dominated or null.

    single model, 30 bins : geometric +0.522, but A-B similarity ALONE gives -0.803
    stratified by A-B sim : Stouffer Z = -0.6 / -1.1  (null)
    residualised          : binned -0.036 (null)
    THIS DESIGN           : geometric -0.845,  DINOv2 -0.968
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
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
EMP = 'trial_distance_(L1_not_normalized)'

GEOM, DINO, INKC = '#2a78d6', '#eb6834', '#1baf7a'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9.5, 'axes.titlesize': 10.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})


def panel_data():
    p = pd.read_csv(f'{G}/out/blindshift_shapenet_percat.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in p.category.unique():
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values
        o['category'] = cat
        o['ft_margin'] = o['fine_tuned_oddity_margin']
        rows.append(o[['trial', 'category', 'ft_margin', EMP, 'd_AB_(L1_not_normalized)']])
    return p.merge(pd.concat(rows), on=['trial', 'category'])


def within(df, cols):
    o = df.copy()
    for c in cols:
        o[c] = (df[c] - df.groupby('trial')[c].transform('mean')
                - df.groupby('category')[c].transform('mean') + df[c].mean())
    return o


def binned(x, y, nb=30):
    q = pd.qcut(x, nb, labels=False, duplicates='drop')
    bx = np.array([x[q == i].mean() for i in range(q.max() + 1)])
    by = np.array([y[q == i].mean() for i in range(q.max() + 1)])
    se = np.array([y[q == i].std() / max(np.sqrt((q == i).sum()), 1)
                   for i in range(q.max() + 1)])
    return bx, by, se


def main():
    d = panel_data()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 6.4))
    fig.subplots_adjust(left=.095, right=.98, top=.60, bottom=.20, wspace=.34)

    for a, (X, lab, col) in zip(ax, [
            ('blind_k50',
             'how far this trial\'s objects are from the training set\n'
             'GEOMETRIC, mesh-to-mesh — no encoder   (centred within trial & training set)',
             GEOM),
            (EMP,
             'how far this trial\'s objects are from the training set\n'
             'DINOv2 $\\ell_1$ feature distance   (centred within trial & training set)',
             DINO)]):
        w = within(d, [X, 'ft_margin'])
        bx, by, se = binned(w[X].values, w['ft_margin'].values)
        rb = stats.pearsonr(bx, by)
        rt = stats.pearsonr(w[X], w['ft_margin'])
        a.errorbar(bx, by, yerr=se, fmt='o', color=col, ms=8, mfc=col, mec=SURFACE,
                   mew=2, ecolor='#d8d7d2', zorder=4)
        b1, b0 = np.polyfit(bx, by, 1)
        xs = np.linspace(bx.min(), bx.max(), 50)
        a.plot(xs, b1 * xs + b0, color=col, lw=2.5, alpha=.55, zorder=3)
        a.axhline(0, color='#e6e4df', lw=1, zorder=1)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
        a.set_xlabel(lab)
        a.set_ylabel('oddity margin of the model fine-tuned\non that same training set\n'
                     '(centred within trial & training set)')
        a.set_title(f'binned r = {rb[0]:+.3f},  p = {rb[1]:.0e}\n'
                    f'trial level r = {rt[0]:+.3f},  n = {len(w):,}', loc='left')

    fig.suptitle('The oddity margin recovers distance-to-training,\n'
                 'with trial difficulty removed by design',
                 fontsize=13, x=.085, ha='left', y=.975, color=INK)
    fig.text(.085, .795,
             'Each of 706 trials appears 12 times — once per ShapeNet training category. '
             'Each category was used to fine-tune one model, so training set ↔ model is 1:1.\n'
             'x is a property of the STIMULI and the training set only (no encoder in the '
             'geometric panel); y is that model\'s behaviour.  d(A,B) has zero within-trial\n'
             'variance, so trial difficulty cannot contribute. Centring is what makes the '
             'x values negative: they are distances relative to each trial\'s own average.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig7_margin_validation_clean.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)

    # report the guarantee explicitly
    w = within(d, ['d_AB_(L1_not_normalized)'])
    print(f"  check: within-trial variance of d(A,B) = "
          f"{w['d_AB_(L1_not_normalized)'].var():.3e}  (zero by construction)")


if __name__ == '__main__':
    main()
