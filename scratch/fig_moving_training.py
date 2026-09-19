"""FIGURE: "Moving the training data moves the margin" — within-trial causal analysis.

Each of 706 ShapeNet trials is evaluated by 12 models differing only in the ShapeNet
category they were fine-tuned on. x = geometric distance from the trial to that model's
training set (voxel16 | knn_mean k=50 | both_mean, the fixed baseline estimator);
y = fine_tuned_oddity_margin.

The pretrained margin is trial-constant, so within-trial slopes are IDENTICAL for the
raw margin and the margin advantage. B and C therefore use the raw margin; D aggregates
cell means across trials, so it must difference the base model out and uses the
advantage.
"""
import ast, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OUT = f'{G}/out/figures'
GEOM, HL, GREY = '#2a78d6', '#eb6834', '#8a8884'
GB = '#9fb6d4'                                    # grey-blue, off-category
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans', 'text.color': INK,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d8d7d2', 'axes.linewidth': .8, 'font.size': 9,
    'axes.titlesize': 10.5, 'legend.frameon': False})
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}
INV = {v: k for k, v in SYN.items()}
DAB = 'd_AB_(L1_not_normalized)'
NPERM, NBOOT = 10_000, 2_000


def style(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(color='#eceae5', lw=.8, zorder=0); a.set_axisbelow(True)


# ---------------------------------------------------------------- load
p = pd.read_csv(f'{G}/out/blindshift_shapenet_voxel16_percat.csv')
m = pd.read_csv(f'{NAV}/MOCHI/mochi_trials.csv')
rows = []
for cat in p.category.unique():
    o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
    o['trial'] = m['trial'].values; o['category'] = cat
    o['ft'] = o['fine_tuned_oddity_margin']; o['pre'] = o['pretrained_oddity_margin']
    o['adv'] = o.ft - o.pre
    rows.append(o[['trial', 'category', 'ft', 'pre', 'adv', DAB]])
d = p.merge(pd.concat(rows), on=['trial', 'category'])
own = {}
for _, r in m[m.dataset == 'shapenet'].iterrows():
    s = {f[:-4].split('_')[0] for f in ast.literal_eval(r['images'])}
    if len(s) == 1:
        own[r['trial']] = INV.get(list(s)[0])
d['own'] = d.trial.map(own)
d = d.dropna(subset=['own'])
CATS = sorted(SYN)

# ---------------------------------------------------------------- sanity checks
print('SANITY CHECKS')
sd_dab = d.groupby('trial')[DAB].std().max()
print(f'  within-trial sd of d(A,B): max over trials = {sd_dab:.3e}   '
      f'{"PASS" if sd_dab < 1e-9 else "FAIL"}')
sd_pre = d.groupby('trial')['pre'].std().max()
print(f'  within-trial sd of base margin: max = {sd_pre:.3e}   '
      f'{"PASS" if sd_pre < 1e-9 else "FAIL"}')
n_per = d.groupby('trial').size()
print(f'  rows = {len(d):,}, trials = {d.trial.nunique()}, '
      f'models per trial = {n_per.min()}-{n_per.max()}')

# wide arrays, trials x 12, columns in fixed CATS order
X = d.pivot(index='trial', columns='category', values='blind_k50')[CATS].values
Y = d.pivot(index='trial', columns='category', values='ft')[CATS].values
A = d.pivot(index='trial', columns='category', values='adv')[CATS].values
OWNC = d.groupby('trial').own.first().reindex(
    d.pivot(index='trial', columns='category', values='ft').index).values
ONMASK = np.array([[c == o for c in CATS] for o in OWNC])
nT = X.shape[0]

Xc = X - X.mean(1, keepdims=True)
Sxx = (Xc ** 2).sum(1)
slopes = ((Xc * (Y - Y.mean(1, keepdims=True))).sum(1)) / Sxx
xc_all = Xc.ravel(); yc_all = (Y - Y.mean(1, keepdims=True)).ravel()
r_tc = stats.pearsonr(xc_all, yc_all)[0]
print(f'  mean within-trial slope = {slopes.mean():+.4f}   '
      f'trial-centred r (margin) = {r_tc:+.4f}   '
      f'{"PASS (both negative)" if slopes.mean() < 0 and r_tc < 0 else "FAIL"}')

# ---------------------------------------------------------------- panel C stats
frac_neg = (slopes < 0).mean()
# (1) permutation: shuffle x across the 12 models WITHIN each trial.
#     Sxx is invariant to a within-row permutation, so only the cross term moves.
rng = np.random.default_rng(0)
Yc = Y - Y.mean(1, keepdims=True)
null = np.empty(NPERM)
for b in range(NPERM):
    idx = np.argsort(rng.random((nT, 12)), axis=1)
    null[b] = (((np.take_along_axis(Xc, idx, axis=1) * Yc).sum(1)) / Sxx).mean()
p_perm = (1 + (null <= slopes.mean()).sum()) / (NPERM + 1)
# (2) cluster bootstrap over the 12 TRAINING CATEGORIES (not trials)
bs = []
for b in range(NBOOT):
    cols = rng.integers(0, 12, 12)
    xb = X[:, cols]; yb = Y[:, cols]
    xbc = xb - xb.mean(1, keepdims=True)
    sxx = (xbc ** 2).sum(1)
    ok = sxx > 1e-12
    if ok.sum() < 50:
        continue
    bs.append((((xbc * (yb - yb.mean(1, keepdims=True))).sum(1))[ok] / sxx[ok]).mean())
bs = np.array(bs)
ci = np.percentile(bs, [2.5, 97.5])
print(f'\nPANEL C  mean slope = {slopes.mean():+.4f}')
print(f'  {100*frac_neg:.1f}% of {nT} trials negative')
print(f'  permutation null (within-trial x shuffles, {NPERM:,}): '
      f'mean {null.mean():+.5f}, 2.5-97.5% [{np.percentile(null,2.5):+.4f}, '
      f'{np.percentile(null,97.5):+.4f}], p = {p_perm:.4f}')
print(f'  cluster bootstrap over 12 training categories: 95% CI '
      f'[{ci[0]:+.4f}, {ci[1]:+.4f}]  ({len(bs)} draws)')

# secondary: exclude the on-category model -> 11 points per trial
Xo = np.where(ONMASK, np.nan, X); Yo = np.where(ONMASK, np.nan, Y)
xo = Xo - np.nanmean(Xo, 1, keepdims=True); yo = Yo - np.nanmean(Yo, 1, keepdims=True)
sxx_o = np.nansum(xo ** 2, 1)
sl_off = np.nansum(xo * yo, 1) / sxx_o
null_off = np.empty(NPERM // 5)
Xo_f = np.where(ONMASK, np.nan, X)
for b in range(len(null_off)):
    idx = np.argsort(rng.random((nT, 12)) + np.where(np.isnan(Xo_f), 10, 0), axis=1)
    xs = np.take_along_axis(xo, idx[:, :11], axis=1)
    null_off[b] = (np.nansum(xs * yo[:, :11], 1) / sxx_o).mean()
p_off = (1 + (null_off <= sl_off.mean()).sum()) / (len(null_off) + 1)
r_off = stats.pearsonr(xo[~np.isnan(xo)], yo[~np.isnan(yo)])
print(f'\nPANEL C secondary  EXCLUDING the on-category model (11 points/trial)')
print(f'  mean slope = {sl_off.mean():+.4f}   {100*(sl_off<0).mean():.1f}% negative')
print(f'  permutation p = {p_off:.4f}   pooled trial-centred r = {r_off[0]:+.4f} '
      f'(p = {r_off[1]:.2e})')

# ---------------------------------------------------------------- panel D
cellA = np.full((12, 12), np.nan); cellD = np.full((12, 12), np.nan)
cellN = np.zeros((12, 12), int)
for i, tc in enumerate(CATS):
    sub = d[d.own == tc]
    for j, trc in enumerate(CATS):
        s2 = sub[sub.category == trc]
        if len(s2):
            cellA[i, j] = s2.adv.mean(); cellD[i, j] = s2.blind_k50.mean()
            cellN[i, j] = len(s2)
fx, fy = cellD.ravel(), cellA.ravel()
ok = ~np.isnan(fx)
r_all = stats.pearsonr(fx[ok], fy[ok])
offd = ~np.eye(12, dtype=bool)
r_off_diag = stats.pearsonr(cellD[offd], cellA[offd])
bt = []
for b in range(NBOOT):
    ri = rng.integers(0, 12, 12)
    xx, yy = cellD[ri].ravel(), cellA[ri].ravel()
    k = ~np.isnan(xx)
    if k.sum() > 10 and np.std(xx[k]) > 0:
        bt.append(stats.pearsonr(xx[k], yy[k])[0])
bt = np.array(bt); ci_d = np.percentile(bt, [2.5, 97.5])
print(f'\nPANEL D  144 cell means (12 test x 12 training)')
print(f'  r(mean distance, mean advantage), ALL cells   = {r_all[0]:+.3f} '
      f'(p = {r_all[1]:.2e})')
print(f'  r, DIAGONAL REMOVED (132 cells)               = {r_off_diag[0]:+.3f} '
      f'(p = {r_off_diag[1]:.2e})')
print(f'  cluster bootstrap over test categories: 95% CI '
      f'[{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]')
print(f'  diagonal mean advantage {np.diag(cellA).mean():+.4f}  vs '
      f'off-diagonal {cellA[offd].mean():+.4f}')

# ================================================================ FIGURE
fig = plt.figure(figsize=(18.6, 12.6))
gs = gridspec.GridSpec(2, 3, figure=fig, left=.052, right=.985, top=.805, bottom=.055,
                       hspace=.36, wspace=.26, height_ratios=[1, 1.06])

# ---- A schematic
a = fig.add_subplot(gs[0, 0]); a.set_aspect('equal'); a.axis('off')
ex = 3
dists = X[ex]; owncat = OWNC[ex]
rad = .30 + 2.5 * (dists - dists.min()) / (dists.max() - dists.min() + 1e-9)
for rr in [1.0, 1.9, 2.8]:
    a.add_patch(Circle((0, 0), rr, fill=False, ec='#e6e4df', lw=1, ls=(0, (4, 4)),
                       zorder=1))
ang = np.linspace(0, 2 * np.pi, 12, endpoint=False) + .18
for k, cat in enumerate(CATS):
    px, py = rad[k] * np.cos(ang[k]), rad[k] * np.sin(ang[k])
    on = cat == owncat
    a.plot([0, px], [0, py], '-', color='#e0dedaa0', lw=.9, zorder=2)
    a.scatter(px, py, s=340 if on else 190, color=HL if on else GB,
              edgecolor=SURF, lw=1.6, zorder=4)
    a.text(px, py - (.30 if on else .26), cat, ha='center', va='top', fontsize=7.4,
           color=INK if on else INK2, weight='bold' if on else 'normal')
a.scatter([0], [0], s=520, marker='s', color=INK, edgecolor=SURF, lw=2, zorder=5)
a.text(0, .46, 'ONE trial\n(fixed triplet)', ha='center', va='bottom', fontsize=8.6,
       color=INK, weight='bold')
a.set_xlim(-3.7, 3.7); a.set_ylim(-3.7, 3.9)
a.set_title('A  Stimulus fixed, training varied\nradius = geometric distance from the '
            'trial to\nthat model\'s training set; orange = on-category', loc='left')

# ---- B dose-response
a = fig.add_subplot(gs[0, 1]); style(a)
samp = rng.choice(nT, 100, replace=False)
for t in samp:
    o = np.argsort(X[t])
    a.plot(X[t][o], Y[t][o], '-', color=GB, lw=.8, alpha=.15, zorder=2)
    a.scatter(X[t][~ONMASK[t]], Y[t][~ONMASK[t]], s=7, color=GB, alpha=.30,
              linewidths=0, zorder=3)
    a.scatter(X[t][ONMASK[t]], Y[t][ONMASK[t]], s=17, color=HL, alpha=.55,
              linewidths=0, zorder=4)
gm = Y.mean()
q = pd.qcut(pd.Series(X.ravel()), 10, labels=False, duplicates='drop').values
yc = (Y - Y.mean(1, keepdims=True)).ravel()
bx = np.array([X.ravel()[q == i].mean() for i in range(q.max() + 1)])
by = np.array([yc[q == i].mean() + gm for i in range(q.max() + 1)])
be = np.array([yc[q == i].std(ddof=1) / np.sqrt((q == i).sum())
               for i in range(q.max() + 1)])
a.errorbar(bx, by, yerr=be, fmt='o-', color=GEOM, ms=10, mfc=GEOM, mec=SURF, mew=1.8,
           lw=2.6, ecolor=GEOM, elinewidth=2, capsize=0, zorder=6)
a.set_title('B  Within-trial dose–response\n100 sampled trials (thin); '
            'decile means of\ntrial-centred margin, regrand-meaned (thick)', loc='left')
a.set_xlabel('geometric distance to that model\'s training set')
a.set_ylabel('fine-tuned oddity margin')

# ---- C slopes
a = fig.add_subplot(gs[0, 2]); style(a)
a.hist(null, bins=60, color=GREY, alpha=.45, edgecolor='none', zorder=2,
       density=True, label='permutation null (within-trial x shuffles)')
a.hist(slopes, bins=50, color=GEOM, alpha=.72, edgecolor=SURF, lw=.5, zorder=3,
       density=True, label=f'observed per-trial slopes (n={nT})')
a.axvline(0, color=INK2, lw=1.5, zorder=5)
a.axvline(slopes.mean(), color=HL, lw=2.4, zorder=6)
a.annotate(f'mean {slopes.mean():+.3f}\n95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]',
           (slopes.mean(), a.get_ylim()[1] * .82), xytext=(-8, 0),
           textcoords='offset points', ha='right', fontsize=8.4, color=HL)
a.legend(loc='upper left', fontsize=7.8)
a.set_title(f'C  One OLS slope per trial (12 points each)\n'
            f'{100*frac_neg:.1f}% negative   permutation p = {p_perm:.4f}\n'
            f'CI from cluster bootstrap over the 12 training categories', loc='left')
a.set_xlabel('within-trial slope of margin on distance')
a.set_ylabel('density')

# ---- C secondary
a = fig.add_subplot(gs[1, 2]); style(a)
a.hist(null_off, bins=50, color=GREY, alpha=.45, edgecolor='none', density=True,
       zorder=2, label='permutation null')
a.hist(sl_off, bins=50, color=GEOM, alpha=.72, edgecolor=SURF, lw=.5, density=True,
       zorder=3, label='observed slopes')
a.axvline(0, color=INK2, lw=1.5, zorder=5)
a.axvline(sl_off.mean(), color=HL, lw=2.4, zorder=6)
a.legend(loc='upper left', fontsize=7.8)
a.set_title(f'C2  Same, EXCLUDING the on-category model\n'
            f'11 points per trial — tests graded structure\nbeyond category match:  '
            f'mean {sl_off.mean():+.4f}, {100*(sl_off<0).mean():.1f}% neg, '
            f'p = {p_off:.4f}', loc='left')
a.set_xlabel('within-trial slope (on-category model removed)')
a.set_ylabel('density')

# ---- D matrix
a = fig.add_subplot(gs[1, 0])
v = np.nanmax(np.abs(cellA))
im = a.imshow(cellA, cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
a.set_xticks(range(12)); a.set_xticklabels(CATS, rotation=90, fontsize=7.6)
a.set_yticks(range(12)); a.set_yticklabels(CATS, fontsize=7.6)
for i in range(12):
    a.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.8))
cb = fig.colorbar(im, ax=a, fraction=.046, pad=.03)
cb.set_label('mean margin advantage (fine-tuned − base)', fontsize=8)
cb.ax.tick_params(labelsize=7.5)
a.set_xlabel('TRAINING category'); a.set_ylabel('TEST category')
a.set_title('D  Transfer matrix, 12 × 12\ncell = mean margin advantage; '
            'diagonal boxed', loc='left')

# ---- D scatter
a = fig.add_subplot(gs[1, 1]); style(a)
dg = np.eye(12, dtype=bool)
a.scatter(cellD[offd], cellA[offd], s=52, color=GB, edgecolor=SURF, lw=1.1,
          zorder=3, label='off-diagonal (132)')
a.scatter(cellD[dg], cellA[dg], s=110, color=HL, edgecolor=SURF, lw=1.5, zorder=4,
          label='diagonal: trained on the test category (12)')
a.axhline(0, color='#dcdad5', lw=1.2, zorder=1)
a.legend(loc='upper right', fontsize=7.8)
a.set_title(f'D  144 cell means\nall cells r = {r_all[0]:+.3f}   '
            f'diagonal removed r = {r_off_diag[0]:+.3f}\n'
            f'95% CI [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}] '
            f'(bootstrap over test categories)', loc='left')
a.set_xlabel('mean geometric distance, test-category objects → training set')
a.set_ylabel('cell mean margin advantage')

fig.suptitle('Moving the training data moves the margin — within-trial causal analysis',
             fontsize=16, x=.052, ha='left', y=.975, color=INK)
fig.text(.052, .905,
         '706 MOCHI ShapeNet trials × 12 category-fine-tuned models = 8,472 observations. '
         'x is the geometric distance from the trial to that model\'s training set '
         '(voxel16 | knn_mean k=50 | both_mean); y is the fine-tuned oddity margin.\n'
         'The trial\'s images, its objects, its d(A,B) and the base DINOv2 margin are '
         'identical across all 12 models (verified within-trial sd = 0), so varying the '
         'training set isolates training exposure by construction rather than by '
         'statistical adjustment.\n'
         'Because the base margin is trial-constant, within-trial slopes are identical '
         'for the raw margin and the margin advantage; B and C use the raw margin, D '
         'aggregates across trials and so uses the advantage.',
         fontsize=9.2, color=INK2, ha='left')
q_ = f'{OUT}/fig46_moving_training.png'
fig.savefig(q_, dpi=300); plt.close(fig)
print(f'\n[fig] {q_}')
