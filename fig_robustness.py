"""
Robustness of the within-trial result to (a) the object representation and
(b) the centring / plotting choice.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
OUT = f'{G}/out/figures'
GEOM, DINO, MUTED, NEG = '#2a78d6', '#eb6834', '#8a8985', '#c9c7c1'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 10,
    'legend.frameon': False,
})
LABEL = {'voxel16': 'raw 16³ voxels  (4096d)', 'voxel8': 'raw 8³ voxels  (512d)',
         'full57': 'full descriptor  (57d)', 'd2only': 'D2 histogram only  (32d)',
         'shellonly': 'shell histogram only  (16d)', 'scalars9': '9 scalars only',
         'bbox7': 'bounding box JSON  (7d)', 'SHUFFLED': 'SHUFFLED  (control)'}


def style(a, grid='x'):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis=grid, color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def main():
    v = pd.read_csv(f'{G}/out/variant_representations.csv')
    c = pd.read_csv(f'{G}/out/centring_variants.csv')
    v['label'] = v.representation.map(LABEL)
    v = v.iloc[::-1].reset_index(drop=True)          # so voxel16 ends up on top

    fig, ax = plt.subplots(1, 3, figsize=(15.6, 5.2))
    fig.subplots_adjust(left=.155, right=.985, top=.74, bottom=.13, wspace=.62)

    for a, col, lab, ttl in [
            (ax[0], 'r_margin', 'r with the fine-tuned oddity margin',
             'A  Eight model-free representations\n(within-trial, two-way centred)'),
            (ax[1], 'r_advantage', 'r with the fine-tuning advantage',
             'B  Same, predicting the fine-tuning advantage')]:
        style(a)
        cols = [NEG if r == 'SHUFFLED' else GEOM for r in v.representation]
        a.barh(np.arange(len(v)), v[col], color=cols, height=.62, zorder=3)
        a.axvline(0, color='#b8b6b0', lw=1.2, zorder=4)
        a.set_yticks(np.arange(len(v)))
        a.set_yticklabels(v.label, fontsize=8.5)
        a.set_xlabel(lab); a.set_title(ttl, loc='left')
        for i, x in enumerate(v[col]):
            a.text(x - .012 if x < 0 else x + .004, i, f'{x:+.3f}',
                   va='center', ha='right' if x < 0 else 'left', fontsize=8, color=INK2)
        a.set_xlim(min(v[col]) * 1.35, max(max(v[col]) * 1.6, .05))

    a = ax[2]; style(a)
    lab = [m.split('  ')[1] for m in c.method[:6]]
    val = c.r[:6].values
    cols = [MUTED, MUTED, MUTED, GEOM, GEOM, GEOM]
    a.barh(np.arange(6), val, color=cols, height=.62, zorder=3)
    a.axvline(0, color='#b8b6b0', lw=1.2, zorder=4)
    a.set_yticks(np.arange(6)); a.set_yticklabels(lab, fontsize=8.5)
    a.invert_yaxis()
    a.set_xlabel('r with the oddity margin')
    a.set_title('C  Six ways to centre the axes\ngrey = difficulty still in play', loc='left')
    for i, x in enumerate(val):
        a.text(x - .015, i, f'{x:+.3f}', va='center', ha='right', fontsize=8, color=INK2)
    a.set_xlim(min(val) * 1.30, .04)

    fig.suptitle('The result does not depend on the descriptor, or on how the axes are '
                 'centred', fontsize=13.5, x=.155, ha='left', y=.955, color=INK)
    fig.text(.155, .845,
             'A–B: every representation from raw 16³ voxels (no designed features) down to '
             '7 numbers in a bounding-box JSON gives the same answer; randomly re-assigning\n'
             'objects kills it.   C: every centring choice is negative and significant; the '
             'effect strengthens as trial difficulty is removed, so difficulty was masking '
             'it, not causing it.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig13_robustness.png'
    fig.savefig(p, dpi=300); plt.close(fig)
    print('[fig]', p)


if __name__ == '__main__':
    main()
