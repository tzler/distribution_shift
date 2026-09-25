"""
Qualitative check: show MOCHI trials alongside the training object the geometric
metric selected as their nearest neighbour. If the metric means anything, the
retrieved training object should look geometrically like the trial's objects.
"""
import ast, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

G = os.path.dirname(os.path.abspath(__file__))
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
REND = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/shapenet_rendered/white'
SYN = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet',
       '02958343': 'car', '03001627': 'chair', '03211117': 'display',
       '03636649': 'lamp', '03691459': 'loudspeaker', '04256520': 'sofa',
       '04379243': 'table', '04401088': 'telephone', '04530566': 'watercraft'}


def nn_render(nn_id):
    syn, obj = nn_id.split('/')
    cat = SYN.get(syn)
    for c in ([cat] if cat else []) + list(SYN.values()):
        p = f'{REND}/{c}/{obj}'
        if os.path.isdir(p):
            fs = sorted(os.listdir(p))
            if fs:
                return f'{p}/{fs[len(fs) // 2]}'
    return None


def main():
    g = pd.read_csv(f'{G}/out/shift3d_shapenet.csv')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    d = g.merge(m[['trial', 'images', 'oddity_index', 'human_avg', 'RT_avg']],
                on='trial', validate='1:1').sort_values('geom_shift_3d')

    picks = pd.concat([d.head(4), d.tail(4)])
    rows = len(picks)
    fig, ax = plt.subplots(rows, 4, figsize=(12.5, 3.0 * rows))
    for i, (_, r) in enumerate(picks.iterrows()):
        ims = ast.literal_eval(r['images'])
        oi = int(r['oddity_index'])
        for j, f in enumerate(ims[:3]):
            a = ax[i, j]
            try:
                a.imshow(Image.open(f'{MOCHI}/images/{f}').convert('RGB'))
            except Exception:
                a.text(.5, .5, 'missing', ha='center')
            a.set_xticks([]); a.set_yticks([])
            lab = 'ODDITY (B)' if j == oi else 'matched (A)'
            a.set_title(lab, fontsize=9,
                        color='#C1440E' if j == oi else '#333333')
        a = ax[i, 3]
        p = nn_render(r['geom_nn_3d'])
        if p:
            a.imshow(Image.open(p).convert('RGB'))
        else:
            a.text(.5, .5, 'render not found', ha='center')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f'nearest TRAINING object\n{r["geom_nn_3d"].split("/")[0]}', fontsize=9,
                    color='#2F6F4E')
        ax[i, 0].set_ylabel(f'shift={r["geom_shift_3d"]:.1f}\nacc={r["human_avg"]:.2f}\n'
                            f'RT={r["RT_avg"]:.0f}ms', fontsize=8)
    fig.suptitle('Lowest (top 4) and highest (bottom 4) geometric shift trials,\n'
                 'with the nearest training object retrieved by the metric',
                 fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f'{G}/out/fig_qualitative_nn.png'
    plt.savefig(out, dpi=115)
    print(f'[figure] {out}')


if __name__ == '__main__':
    main()
