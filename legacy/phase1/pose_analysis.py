"""
Do pose-referenced, model-free features predict the oddity margin?

Two questions, both plotted:

  (a) DISTANCE TO TRAINING measured in each new feature space
      multiview / volatility / structure, same within-trial design as before.

  (b) The features as RAW TRIAL PROPERTIES, with no training bank at all.
      Viewpoint volatility is the interesting one: how much an object's silhouette
      changes across viewpoints. If models learn "appearance from a pose" rather than
      shape, a volatile object should be hard whether or not it resembles the training
      set. This needs no bank, so it is a pure stimulus measure -- and the PRETRAINED
      margin is the control that says whether it is about training at all.
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
from metric_space import build, behaviour, SYN, prep                    # noqa: E402

NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
OUT = f'{G}/out/figures'
FT, PRE, MUT = '#2a78d6', '#eb6834', '#a8a6a0'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': '#d8d7d2',
    'axes.linewidth': .8, 'font.size': 9, 'axes.titlesize': 9.5,
    'legend.frameon': False, 'lines.solid_capstyle': 'round',
})
REPS = [('multiview  (13 viewpoints × 5 silhouette stats)', 'multiview'),
        ('viewpoint volatility  (how much it changes across views)', 'volatility'),
        ('structure  (parts, holes, symmetry)', 'structure'),
        ('raw 16³ voxels  (reference)', None)]


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


def own_map():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    inv = {v: k for k, v in SYN.items()}
    own = {}
    for _, r in mm.iterrows():
        s = {f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
        if len(s) == 1:
            own[r['trial']] = inv.get(list(s)[0])
    return own


def rank_curve(d, X, dv):
    dd = d.copy(); dd['rank'] = dd.groupby('trial')[X].rank(method='first')
    g = dd.groupby('rank')
    return (g[X].mean().values, g[dv].mean().values,
            (g[dv].std() / np.sqrt(g[dv].size())).values)


def raw_trial_property(tag):
    """Mean feature value over the trial's objects — no training bank involved."""
    tz = np.load(f'{G}/bank/test_{tag}.npz', allow_pickle=True)
    X = tz['X'].astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    tix = {k: i for i, k in enumerate(tz['ids'])}
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    rows = []
    for _, r in mm.iterrows():
        keys = [oid(f) for f in ast.literal_eval(r['images'])]
        if all(k in tix for k in keys):
            v = X[[tix[k] for k in keys]].mean(0)
            rows.append(dict(trial=r['trial'], **{f'f{i}': v[i] for i in range(len(v))}))
    return pd.DataFrame(rows)


def main():
    own = own_map()
    beh = behaviour()
    rows = []
    fig, ax = plt.subplots(2, 4, figsize=(16.0, 8.0))
    fig.subplots_adjust(left=.06, right=.985, top=.79, bottom=.08, hspace=.50, wspace=.26)

    for j, (nice, tag) in enumerate(REPS):
        bf = (f'{G}/bank/bank_{tag}.npz' if tag else f'{G}/bank/bank_voxel16.npz')
        tf = (f'{G}/bank/test_{tag}.npz' if tag else f'{G}/bank/test_voxel16.npz')
        if not (os.path.exists(bf) and os.path.exists(tf)):
            print(f'skip {nice} (bank missing)'); continue
        d = build(bf, tf).merge(beh, on=['trial', 'category'])
        d['own'] = d.trial.map(own); d = d.dropna(subset=['own'])
        oc = d[d.category == d.own]

        # ---- top row: distance-to-training, all 12 models, within-trial ranks
        a = ax[0, j]; style(a)
        bx, by, se = rank_curve(d, 'both_mean', 'ft')
        a.errorbar(bx, by, yerr=se, fmt='o-', color=FT, lw=2, ms=6.5, mfc=FT,
                   mec=SURFACE, mew=1.6, ecolor='#dcdbd6', zorder=3)
        w = d.copy()
        for c in ['both_mean', 'ft']:
            w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                    - d.groupby('category')[c].transform('mean') + d[c].mean())
        r_all = stats.pearsonr(w['both_mean'], w['ft'])
        a.set_title(f'{nice}\nall 12 models, within-trial r = {r_all[0]:+.3f}', loc='left')
        a.set_xlabel('distance to training set'); a.set_ylabel('fine-tuned oddity margin')

        # ---- bottom row: within category only, with the pretrained control
        a = ax[1, j]; style(a)
        q = pd.qcut(oc.both_mean, 10, labels=False, duplicates='drop')
        bx2 = [oc.both_mean[q == i].mean() for i in range(q.max() + 1)]
        by2 = [oc.ft[q == i].mean() for i in range(q.max() + 1)]
        se2 = [oc.ft[q == i].sem() for i in range(q.max() + 1)]
        bp = [oc.pre[q == i].mean() for i in range(q.max() + 1)]
        a.errorbar(bx2, by2, yerr=se2, fmt='o-', color=FT, lw=2, ms=6.5, mfc=FT,
                   mec=SURFACE, mew=1.6, ecolor='#dcdbd6', zorder=3,
                   label='fine-tuned')
        a.plot(bx2, bp, 'o--', color=PRE, lw=1.6, ms=5, mfc=PRE, mec=SURFACE, mew=1.2,
               zorder=3, label='pretrained (control)')
        rf = stats.pearsonr(oc.both_mean, oc.ft)
        rp = stats.pearsonr(oc.both_mean, oc.pre)
        a.set_title(f'WITHIN CATEGORY (n={len(oc)})\n'
                    f'fine-tuned r = {rf[0]:+.3f}   pretrained r = {rp[0]:+.3f}', loc='left')
        a.set_xlabel('distance to own-category training set')
        a.set_ylabel('oddity margin')
        if j == 0:
            a.legend(loc='best', fontsize=8)
        rows.append(dict(rep=nice, all12_within=r_all[0], wc_ft=rf[0], wc_ft_p=rf[1],
                         wc_pre=rp[0]))
        print(f'{nice:56s} all12 {r_all[0]:+.3f}   within-cat ft {rf[0]:+.3f} '
              f'(p={rf[1]:.3f})  pre {rp[0]:+.3f}', flush=True)

    fig.suptitle('Pose-referenced model-free features vs the oddity margin',
                 fontsize=13.5, x=.06, ha='left', y=.965, color=INK)
    fig.text(.06, .845,
             'TOP: distance to the training set measured in each feature space, all 12 '
             'category models, within-trial ranks (raw axes).\n'
             'BOTTOM: restricted to each trial\'s OWN category, so the category-match '
             'binary is held constant. Orange dashed = the PRETRAINED model, which saw '
             'none of these\ntraining sets — it should stay flat. Every feature family is '
             'computed from the voxel grid alone; no renderer, no network.',
             fontsize=9, color=INK2, ha='left')
    p = f'{OUT}/fig22_pose_features.png'
    fig.savefig(p, dpi=300); plt.close(fig); print('\n[fig]', p)
    pd.DataFrame(rows).to_csv(f'{G}/out/pose_features_summary.csv', index=False)


if __name__ == '__main__':
    main()
