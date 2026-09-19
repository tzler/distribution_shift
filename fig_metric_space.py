"""
Visualising the metric space.

fig18  one panel per metric: the within-trial rank curve against the fine-tuned oddity
       margin, raw axes. Panels are coloured by whether the metric leaks — i.e. whether
       it also predicts the PRETRAINED margin, which it must not, since that encoder
       never saw any of these training sets.

fig19  the leak plot. x = how well the metric predicts the fine-tuned margin,
       y = how much it predicts the pretrained margin. The useful corner is bottom-left
       of the shaded band: predicts the fine-tuned model, ignores the pretrained one.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from metric_space import build, behaviour, METRICS, DEFN                # noqa: E402

OUT = f'{G}/out/figures'
OK, LEAK, DINO = '#2a78d6', '#a8a6a0', '#eb6834'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 8.5, 'axes.titlesize': 9,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})
SHORT = {
    'both_mean': 'average of both objects', 'oddity_only': 'oddity object only',
    'matched_only': 'matched object only', 'closer_one': 'the nearer object',
    'further_one': 'the further object',
    'odd_minus_matched': 'oddity − matched (asymmetry)',
    'abs_asymmetry': '|oddity − matched|',
    'over_dAB': '÷ distance between the two objects',
    'minus_dAB': '− distance between the two objects',
    'over_dTT': "÷ training set's own spread",
    'minus_dTT': "− training set's own spread",
    'train_vs_partner': 'training vs partner (= minus_dAB)',
}


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def panel(d, metric, dv='ft'):
    dd = d.copy()
    dd['rank'] = dd.groupby('trial')[metric].rank(method='first')
    g = dd.groupby('rank')
    return (g[metric].mean().values, g[dv].mean().values,
            (g[dv].std() / np.sqrt(g[dv].size())).values)


def main():
    stats_df = pd.read_csv(f'{G}/out/metric_space.csv')
    d = build(f'{G}/bank/bank3d_shapenet_trained.npz',
              f'{G}/bank/test3d_shapenet.npz').merge(behaviour(),
                                                     on=['trial', 'category'])
    sd = stats_df[stats_df.rep == '57-number shape descriptor'].set_index('metric')

    # ---------------- fig18: curves ----------------
    order = sd.sort_values('pooled_ft').index.tolist()
    fig, ax = plt.subplots(3, 4, figsize=(15.6, 9.4), sharey=True)
    fig.subplots_adjust(left=.06, right=.985, top=.80, bottom=.06, hspace=.55, wspace=.14)
    for i, mname in enumerate(order):
        a = ax.ravel()[i]; style(a)
        bx, by, se = panel(d, mname)
        leaks = abs(sd.loc[mname, 'pooled_pre']) > .10
        c = LEAK if leaks else OK
        a.errorbar(bx, by, yerr=se, fmt='o-', color=c, lw=1.8, ms=5.5, mfc=c,
                   mec=SURFACE, mew=1.4, ecolor='#dcdbd6', zorder=3)
        a.set_title(f'{SHORT[mname]}\n'
                    f'pooled {sd.loc[mname, "pooled_ft"]:+.3f}   '
                    f'pretrained {sd.loc[mname, "pooled_pre"]:+.3f}'
                    f'{"  LEAK" if leaks else "  ✓ clean"}\n'
                    f'within-trial {sd.loc[mname, "within_ft"]:+.3f}',
                    loc='left', color=INK if not leaks else INK2)
        a.set_xlabel('metric value (raw)')
        if i % 4 == 0:
            a.set_ylabel('fine-tuned oddity margin')
    fig.suptitle('Twelve model-free shift metrics built from the same four geometric '
                 'quantities', fontsize=13.5, x=.06, ha='left', y=.972, color=INK)
    fig.text(.06, .875,
             'Each panel: 12 within-trial ranks, raw axes, shared y — every point holds '
             'the same 706 trials.  Grey = the metric also predicts the PRETRAINED '
             'margin,\nwhich it must not: that encoder never saw any of these training '
             'sets, so such a metric is reading a stimulus property, not distance to '
             'training.\nBlue = clean.  57-number shape descriptor.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig18_metric_space_curves.png'
    fig.savefig(p, dpi=300); plt.close(fig); print('[fig]', p)

    # ---------------- fig19: leak plot ----------------
    order19 = sd.sort_values('pooled_ft').index.tolist()
    num = {mname: i + 1 for i, mname in enumerate(order19)}
    fig = plt.figure(figsize=(15.8, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .62], left=.055, right=.99,
                          top=.72, bottom=.13, wspace=.26)
    for j, (rep, col) in enumerate([('57-number shape descriptor', OK),
                                    ('raw 16^3 voxel grid', DINO)]):
        a = fig.add_subplot(gs[0, j]); style(a)
        srep = stats_df[stats_df.rep == rep]
        a.axhspan(-.10, .10, color='#e8f0fa', zorder=1)
        a.axhline(0, color='#c9c7c1', lw=1, zorder=2)
        a.axvline(0, color='#c9c7c1', lw=1, zorder=2)
        for _, r in srep.iterrows():
            clean = abs(r.pooled_pre) <= .10
            a.scatter(r.pooled_ft, r.pooled_pre, s=210,
                      color=col if clean else LEAK, edgecolor=SURFACE, lw=1.8, zorder=4)
            a.text(r.pooled_ft, r.pooled_pre, str(num[r.metric]), ha='center',
                   va='center', fontsize=8, color='white', zorder=5, weight='bold')
        a.set_xlabel('predicts the FINE-TUNED margin  (want: negative)')
        a.set_ylabel('predicts the PRETRAINED margin  (want: zero)')
        a.set_title(rep, loc='left')
        a.set_xlim(-.30, .30); a.set_ylim(-.30, .30)

    a = fig.add_subplot(gs[0, 2]); a.axis('off')
    a.text(0, 1.0, 'key', fontsize=10, color=INK, va='top', weight='bold')
    for mname, i in num.items():
        clean = abs(sd.loc[mname, 'pooled_pre']) <= .10
        a.text(0, .955 - i * .073, f'{i:>2d}   {SHORT[mname]}', fontsize=8.6,
               color=INK if clean else INK2, va='top', family='DejaVu Sans')
    a.text(0, .955 - 13 * .073, 'shaded band = |r| < 0.10 with the pretrained margin',
           fontsize=8, color=INK2, va='top', style='italic')

    fig.suptitle('Which metrics are actually measuring distance to training?',
                 fontsize=13.5, x=.055, ha='left', y=.955, color=INK)
    fig.text(.055, .825,
             'The pretrained encoder never saw any of these 12 training sets, so a valid '
             'shift measure must sit inside the shaded band (y \u2248 0) while still\n'
             'predicting the fine-tuned model (x < 0). Useful corner = bottom-left, inside '
             'the band. Grey = leaking.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig19_leak_plot.png'
    fig.savefig(p, dpi=300); plt.close(fig); print('[fig]', p)


if __name__ == '__main__':
    main()
