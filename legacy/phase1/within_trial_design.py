"""
THE DISENTANGLING DESIGN.

Problem: Delta_geom and d(A,B) are collinear (r~0.85-0.997), so no amount of statistical
control on a single model can separate "distance to training" from "the two objects are
geometrically different". Partialling over-controls; stratifying is underpowered at n=706.

Solution: hold the TRIAL fixed and vary the TRAINING SET. d(A,B) is a property of the
stimulus, so it is *constant* across models and is absorbed exactly by trial fixed
effects. Distance-to-training varies across models. Any residual effect is purely
distributional.

Data: 12 per-category DINOv2-L fine-tunes (+ all_categories) at
  Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses/<category>/ood_analysis_results.csv
each with all 2019 MOCHI trials and a shared pretrained baseline.

Model:  advantage[trial, cat] ~ beta * Delta_geom[trial, cat] + alpha_trial + gamma_cat

estimated by two-way within (demeaning) so alpha_trial absorbs d(A,B), difficulty,
category of the test objects, and every other stimulus property.
"""
import os, ast
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
P = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
S = ('/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune/'
     'ShapeNet_OOD_Analyses')
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}


def geometric_distance_per_category():
    """Delta_geom(trial, category-bank) for every trial x category."""
    bz = np.load(f'{G}/bank/bank3d_shapenet_trained.npz', allow_pickle=True)
    tz = np.load(f'{G}/bank/test3d_shapenet.npz', allow_pickle=True)
    B, T = bz['X'].astype(float), tz['X'].astype(float)
    bids = np.array([i.split('/')[0] for i in bz['ids']])
    tix = {k: i for i, k in enumerate(tz['ids'])}
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd

    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    trials, A, Bq = [], [], []
    for _, r in mm.iterrows():
        ims = ast.literal_eval(r['images']); o = int(r['oddity_index'])
        ka = oid([f for k, f in enumerate(ims) if k != o][0]); kb = oid(ims[o])
        if ka in tix and kb in tix:
            trials.append(r['trial']); A.append(TZ[tix[ka]]); Bq.append(TZ[tix[kb]])
    A, Bq = np.array(A), np.array(Bq)

    recs = []
    for cat, syn in SYN.items():
        sel = bids == syn
        if sel.sum() == 0:
            print(f'  [warn] no bank objects for {cat} ({syn})'); continue
        bank = BZ[sel]
        dA = cdist(A, bank, 'cityblock'); dB = cdist(Bq, bank, 'cityblock')
        shift = (0.5 * (dA + dB)).min(1)
        k = min(200, bank.shape[0])
        knn = 0.5 * (np.sort(dA, 1)[:, :k].mean(1) + np.sort(dB, 1)[:, :k].mean(1))
        for i, t in enumerate(trials):
            recs.append(dict(trial=t, category=cat, geom_shift_cat=shift[i],
                             geom_knn_cat=knn[i], n_bank=int(sel.sum())))
    return pd.DataFrame(recs)


def load_advantages():
    rows = []
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    for cat in SYN:
        f = f'{S}/{cat}/ood_analysis_results.csv'
        if not os.path.exists(f):
            continue
        o = pd.read_csv(f)
        assert len(o) == len(m)
        o['trial'] = m['trial'].values
        o['category'] = cat
        o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
        o['s_ft'] = 1.0 + o['fine_tuned_oddity_margin']
        rows.append(o[['trial', 'category', 'advantage', 's_ft',
                       'fine_tuned_correct', 'pretrained_correct']])
    return pd.concat(rows, ignore_index=True)


def two_way_within(d, cols):
    """Demean each column by trial and by category (two-way fixed effects)."""
    out = d.copy()
    for c in cols:
        out[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                  - d.groupby('category')[c].transform('mean') + d[c].mean())
    return out


def main():
    print('computing per-category geometric distances ...', flush=True)
    gd = geometric_distance_per_category()
    adv = load_advantages()
    d = gd.merge(adv, on=['trial', 'category'], how='inner')
    print(f'panel: {d.trial.nunique()} trials x {d.category.nunique()} category-models '
          f'= {len(d)} observations\n')

    for X in ['geom_shift_cat', 'geom_knn_cat']:
        print(f'--- predictor: {X}')
        r_raw = stats.pearsonr(d[X], d['advantage'])
        print(f'    pooled (no FE)              r={r_raw[0]:+.4f}  p={r_raw[1]:.2e}')
        w = two_way_within(d, [X, 'advantage', 's_ft'])
        r_fe = stats.pearsonr(w[X], w['advantage'])
        n_par = d.trial.nunique() + d.category.nunique()
        t = r_fe[0] * np.sqrt((len(w) - n_par) / max(1 - r_fe[0] ** 2, 1e-12))
        print(f'    TWO-WAY WITHIN (trial + cat FE)  r={r_fe[0]:+.4f}  '
              f't={t:+.2f}  p={2*(1-stats.t.cdf(abs(t), len(w)-n_par)):.2e}')
        b = stats.linregress(w[X], w['advantage'])
        print(f'      beta={b.slope:+.5f}   (advantage per unit geometric distance)')
        r_m = stats.pearsonr(w[X], w['s_ft'])
        print(f'    same design, DV = fine-tuned margin: r={r_m[0]:+.4f}  p={r_m[1]:.2e}\n')

    # per-trial slope sign test: within each trial, does the closest category help most?
    slopes = []
    for t_, sub in d.groupby('trial'):
        if len(sub) >= 8 and sub['geom_shift_cat'].std() > 0:
            slopes.append(stats.linregress(sub['geom_shift_cat'], sub['advantage']).slope)
    slopes = np.array(slopes)
    ts = stats.ttest_1samp(slopes, 0)
    print(f'per-trial slopes (advantage on geometric distance, within trial): '
          f'n={len(slopes)}')
    print(f'  mean={slopes.mean():+.5f}  t={ts.statistic:+.2f}  p={ts.pvalue:.2e}  '
          f'{(slopes<0).mean()*100:.0f}% negative (paper predicts negative)')

    d.to_csv(f'{G}/out/within_trial_panel.csv', index=False)
    figure(d)
    print(f'\nwrote {G}/out/within_trial_panel.csv')


def figure(d):
    w = two_way_within(d, ['geom_shift_cat', 'advantage'])
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    q = pd.qcut(d['geom_shift_cat'], 20, labels=False, duplicates='drop')
    bx = [d['geom_shift_cat'][q == i].mean() for i in range(q.max() + 1)]
    by = [d['advantage'][q == i].mean() for i in range(q.max() + 1)]
    ax[0].plot(bx, by, 'o-', color='#9A9A9A', lw=2)
    ax[0].set_title('pooled (d(A,B) NOT controlled)', fontsize=11)
    ax[0].set_xlabel('geometric distance to that category\'s training set')
    ax[0].set_ylabel('fine-tuning advantage')

    q = pd.qcut(w['geom_shift_cat'], 20, labels=False, duplicates='drop')
    bx = [w['geom_shift_cat'][q == i].mean() for i in range(q.max() + 1)]
    by = [w['advantage'][q == i].mean() for i in range(q.max() + 1)]
    r = stats.pearsonr(w['geom_shift_cat'], w['advantage'])
    ax[1].plot(bx, by, 'o-', color='#5B3E96', lw=2)
    ax[1].set_title(f'WITHIN TRIAL + category FE  (d(A,B) absorbed)\nr={r[0]:+.4f}, '
                    f'p={r[1]:.1e}', fontsize=11)
    ax[1].set_xlabel('geometric distance (trial- and category-demeaned)')
    ax[1].set_ylabel('advantage (demeaned)')
    for a in ax:
        a.grid(alpha=.25); a.axhline(0, color='k', lw=.6)
    fig.suptitle('Does fine-tuning help most on trials geometrically CLOSE to the '
                 'training set?\n12 per-category models x the same trials', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = f'{G}/out/fig_within_trial.png'
    plt.savefig(p, dpi=140); plt.close()
    print(f'[figure] {p}')


if __name__ == '__main__':
    main()
