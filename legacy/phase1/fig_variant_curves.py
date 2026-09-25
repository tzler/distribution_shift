"""
The actual margin relationship for every model-free representation.

Small multiples: one panel per representation, each showing the within-trial rank curve
on RAW axes (x = mean raw distance at that rank, y = mean raw fine-tuned oddity margin).
All 12 points in every panel contain the same 706 trials, so trial difficulty is held
constant by construction. Shared y axis so the panels are directly comparable.
"""
import ast, os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from variant_eval import load, shift_panel, behaviour, SYN            # noqa: E402

NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
GEOM, NEG = '#2a78d6', '#a8a6a0'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 8.5, 'axes.titlesize': 9.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})

F57 = (f'{G}/bank/bank3d_shapenet_trained.npz', f'{G}/bank/test3d_shapenet.npz')
VARIANTS = [
    ('raw 16³ voxels', '4096d — no designed features', f'{G}/bank/bank_voxel16.npz',
     f'{G}/bank/test_voxel16.npz', None, False),
    ('raw 8³ voxels', '512d', f'{G}/bank/bank_voxel8.npz',
     f'{G}/bank/test_voxel8.npz', None, False),
    ('full descriptor', '57d — used throughout', *F57, None, False),
    ('bounding box JSON', '7d — extents, aspect, volume', f'{G}/bank/bank_bbox.npz',
     f'{G}/bank/test_bbox.npz', None, False),
    ('D2 histogram only', '32d', *F57, np.arange(0, 32), False),
    ('shell histogram only', '16d', *F57, np.arange(32, 48), False),
    ('9 scalars only', '9d', *F57, np.arange(48, 57), False),
    ('SHUFFLED', 'negative control', *F57, None, True),
]


def main():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    trials_imgs = [(r['trial'], [oid(f) for f in ast.literal_eval(r['images'])])
                   for _, r in mm.iterrows()]
    beh = behaviour()

    fig, ax = plt.subplots(2, 4, figsize=(15.4, 7.4), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.76, bottom=.10, hspace=.52, wspace=.16)

    for i, (name, sub, bf, tf, cols, sh) in enumerate(VARIANTS):
        a = ax.ravel()[i]
        bids, BN, tix, TN = load(bf, tf, cols, sh)
        d = shift_panel(bids, BN, tix, TN, trials_imgs).merge(beh, on=['trial', 'category'])
        d['rank'] = d.groupby('trial')['shift'].rank(method='first')
        g = d.groupby('rank')
        bx, by = g['shift'].mean(), g['ft_margin'].mean()
        se = g['ft_margin'].std() / np.sqrt(g['ft_margin'].size())
        col = NEG if sh else GEOM
        a.errorbar(bx, by, yerr=se, fmt='o-', color=col, lw=1.8, ms=6, mfc=col,
                   mec=SURFACE, mew=1.5, ecolor='#dcdbd6', zorder=3)
        sl = np.array([stats.linregress(s['shift'], s['ft_margin']).slope
                       for _, s in d.groupby('trial') if s['shift'].std() > 0])
        t = stats.ttest_1samp(sl, 0)
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)
        a.set_title(f'{name}   ({sub})\n'
                    f't = {t.statistic:+.1f},  {100*(sl<0).mean():.0f}% of trials negative',
                    loc='left', color=INK if not sh else INK2)
        a.set_xlabel('distance to training set (raw)')
        if i % 4 == 0:
            a.set_ylabel('fine-tuned oddity margin (raw)')

    fig.suptitle('The same relationship under eight different model-free object '
                 'representations', fontsize=13.5, x=.055, ha='left', y=.965, color=INK)
    fig.text(.055, .855,
             'Each panel: 12 within-trial ranks (each trial\'s closest training set … its '
             'furthest); all 12 points hold the same 706 trials, so d(A,B) and trial\n'
             'difficulty are identical across every x-axis. Raw axes, shared y. The margin '
             'falls with distance under every representation — including raw voxel grids '
             'with no designed\nfeatures — and is flat when objects are randomly '
             're-assigned (bottom right).',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig14_variant_curves.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)


if __name__ == '__main__':
    main()
