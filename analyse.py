"""
Validation battery for the geometric distribution-shift metric.

Applies exactly the bar set by ../L1norm_vs_distshift/README.md: correlations against
the behavioural DVs, the low-level confound battery (feat_l1 / pix_l1 / coverage /
contrast), partial correlations, nested comparisons against the incumbent metric, a
within-category breakdown, and the d_AB decomposition that the audit identifies as the
structural weakness of the shared-neighbour form.

Writes stats CSVs and figures into out/.
"""
import os, sys, ast
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = os.path.dirname(os.path.abspath(__file__))
L = '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/L1norm_vs_distshift'
MOCHI = '/vast/projects/bonnen/naturalistic-navig/MOCHI'
OUT = f'{G}/out'
ENC = 'vit_base_patch16_224.dino'
CONF = ['feat_l1', 'pix_l1', 'coverage', 'contrast']
INC = 'trial_distance_(L1_not_normalized)'


def load(dataset, mode):
    f = f'{OUT}/shift{mode}_{dataset}.csv'
    if not os.path.exists(f):
        return None
    g = pd.read_csv(f)
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    d = g.merge(m[['trial', 'dataset', 'condition', 'human_avg', 'RT_avg',
                   'DINOv2G_avg', 'n_subjects']], on='trial', validate='1:1')
    t = pd.read_csv(f'{L}/trials_{ENC}.csv')
    cols = ['trial', INC, 'pretrained_correct', 'd_AB_(L1_not_normalized)'] + CONF
    d = d.merge(t[cols], on='trial', validate='1:1')
    return d


def partial(df, x, y, ctrl):
    Z = np.column_stack([np.ones(len(df))] + [df[c].values for c in ctrl])
    rx = df[x].values - Z @ np.linalg.lstsq(Z, df[x].values, rcond=None)[0]
    ry = df[y].values - Z @ np.linalg.lstsq(Z, df[y].values, rcond=None)[0]
    return stats.pearsonr(rx, ry)


def binned(x, y, nbin=10):
    q = pd.qcut(x, nbin, labels=False, duplicates='drop')
    bx = np.array([np.mean(x[q == i]) for i in range(q.max() + 1)])
    by = np.array([np.mean(y[q == i]) for i in range(q.max() + 1)])
    se = np.array([np.std(y[q == i]) / max(np.sqrt((q == i).sum()), 1)
                   for i in range(q.max() + 1)])
    return bx, by, se


def run(d, dataset, mode):
    tag = f'{dataset}_{mode}'
    S, DAB = f'geom_shift_{mode}', f'geom_dAB_{mode}'
    SEP, DA, DB = f'geom_shift_sep_{mode}', f'geom_dA_nn_{mode}', f'geom_dB_nn_{mode}'
    rows = []
    dvs = {'human_avg': 'human accuracy', 'RT_avg': 'human RT',
           'DINOv2G_avg': 'DINOv2-G acc', 'pretrained_correct': 'DINO-B correct'}
    KNN = [f'geom_knn{k}_{mode}' for k in (10, 50, 200)]
    ivs = [S, SEP, DAB, DA, DB] + KNN + [INC] + CONF

    print(f'\n{"="*78}\n{tag}   n={len(d)}\n{"="*78}')
    print(f'{"predictor":34s}' + ''.join(f'{v:>17s}' for v in dvs.values()))
    for iv in ivs:
        cells = []
        for dv in dvs:
            r, p = stats.pearsonr(d[iv], d[dv])
            st = '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''
            cells.append(f'{r:+.3f}{st:3s}')
            rows.append(dict(dataset=dataset, mode=mode, predictor=iv, dv=dv,
                             n=len(d), r=r, p=p))
        print(f'{iv:34s}' + ''.join(f'{c:>17s}' for c in cells))

    # --- confound collinearity (the audit's central diagnostic) -----------
    print(f'\n-- collinearity of {S} with the quantities that sink the incumbent')
    for c in CONF + [INC, 'd_AB_(L1_not_normalized)', DAB]:
        r, p = stats.pearsonr(d[S], d[c])
        print(f'   r({S}, {c:28s}) = {r:+.3f}  p={p:.1e}')
        rows.append(dict(dataset=dataset, mode=mode, predictor=S, dv=f'CONFOUND:{c}',
                         n=len(d), r=r, p=p))

    # --- decomposition: does anything survive controlling d_AB? ----------
    print(f'\n-- partial correlations controlling within-trial dissimilarity ({DAB})')
    for iv in [S, SEP, DA, DB] + KNN:
        for dv in ['human_avg', 'RT_avg']:
            raw = stats.pearsonr(d[iv], d[dv])[0]
            pr, pp = partial(d, iv, dv, [DAB])
            print(f'   {iv:24s} -> {dv:9s} raw={raw:+.3f}  partial|d_AB={pr:+.3f} (p={pp:.1e})')
            rows.append(dict(dataset=dataset, mode=mode, predictor=f'{iv}|dAB', dv=dv,
                             n=len(d), r=pr, p=pp))

    # --- does geometry add over incumbent + low-level confounds? ---------
    print(f'\n-- incremental value over the incumbent metric + low-level confounds')
    for dv in ['human_avg', 'RT_avg']:
        r1, p1 = partial(d, S, dv, [INC] + CONF)
        r2, p2 = partial(d, INC, dv, [S] + CONF)
        print(f'   {dv:9s}: geom | (incumbent+conf) = {r1:+.3f} (p={p1:.1e})   '
              f'incumbent | (geom+conf) = {r2:+.3f} (p={p2:.1e})')
        rows.append(dict(dataset=dataset, mode=mode, predictor='geom|inc+conf', dv=dv,
                         n=len(d), r=r1, p=p1))
        rows.append(dict(dataset=dataset, mode=mode, predictor='inc|geom+conf', dv=dv,
                         n=len(d), r=r2, p=p2))

    # --- within-category breakdown ---------------------------------------
    print(f'\n-- within-condition breakdown (guards against a between-category artifact)')
    for cond, sub in d.groupby('condition'):
        if len(sub) < 25:
            continue
        ra = stats.pearsonr(sub[S], sub['human_avg'])
        rt = stats.pearsonr(sub[S], sub['RT_avg'])
        print(f'   {cond:16s} n={len(sub):4d}  acc r={ra[0]:+.3f} (p={ra[1]:.0e})   '
              f'RT r={rt[0]:+.3f} (p={rt[1]:.0e})')
        rows.append(dict(dataset=dataset, mode=mode, predictor=f'{S}@{cond}',
                         dv='human_avg', n=len(sub), r=ra[0], p=ra[1]))
        rows.append(dict(dataset=dataset, mode=mode, predictor=f'{S}@{cond}',
                         dv='RT_avg', n=len(sub), r=rt[0], p=rt[1]))

    # --- within-category centring (guards the between-category artifact) ---
    dc = d.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            dc[c] = d[c] - d.groupby('condition')[c].transform('mean')
    grand = d[S].mean()
    ss_bet = sum(len(s_) * (s_[S].mean() - grand) ** 2 for _, s_ in d.groupby('condition'))
    ss_tot = ((d[S] - grand) ** 2).sum()
    print(f'\n-- within-category centred  (between-category share of shift variance = '
          f'{ss_bet/ss_tot:.3f})')
    for iv in [S, SEP] + KNN + [INC]:
        for dv in ['human_avg', 'RT_avg']:
            raw = stats.pearsonr(d[iv], d[dv])[0]
            r, p = stats.pearsonr(dc[iv], dc[dv])
            print(f'   {iv:24s} -> {dv:9s} raw={raw:+.3f}  within-category={r:+.3f} (p={p:.1e})')
            rows.append(dict(dataset=dataset, mode=mode, predictor=f'{iv}|within_category',
                             dv=dv, n=len(d), r=r, p=p))

    pd.DataFrame(rows).to_csv(f'{OUT}/stats_{tag}.csv', index=False)
    figure(d, tag, S, DAB, SEP)
    return rows


def figure(d, tag, S, DAB, SEP):
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.6))
    fig.suptitle(f'Model-free geometric distribution shift — {tag} (n={len(d)})',
                 fontsize=14, y=0.98)

    for j, (dv, lab) in enumerate([('human_avg', 'human accuracy'),
                                   ('RT_avg', 'human RT (ms)')]):
        a = ax[j, 0]
        bx, by, se = binned(d[S].values, d[dv].values)
        a.errorbar(bx, by, yerr=se, fmt='o-', color='#5B3E96', lw=2, ms=6)
        r, p = stats.pearsonr(d[S], d[dv])
        a.set_xlabel('geometric shift'); a.set_ylabel(lab)
        a.set_title(f'GEOMETRIC\nr={r:+.3f}, p={p:.1e}', fontsize=10)

        a = ax[j, 1]
        bx, by, se = binned(d[INC].values, d[dv].values)
        a.errorbar(bx, by, yerr=se, fmt='o-', color='#9A9A9A', lw=2, ms=6)
        r, p = stats.pearsonr(d[INC], d[dv])
        a.set_xlabel('incumbent L1 shift'); a.set_ylabel(lab)
        a.set_title(f'INCUMBENT (model-based)\nr={r:+.3f}, p={p:.1e}', fontsize=10)

        a = ax[j, 2]
        a.scatter(d[DAB], d[dv], s=7, alpha=.35, color='#C1440E')
        r, p = stats.pearsonr(d[DAB], d[dv])
        a.set_xlabel('within-trial geometric d(A,B)'); a.set_ylabel(lab)
        a.set_title(f'within-trial dissimilarity\nr={r:+.3f}, p={p:.1e}', fontsize=10)

        a = ax[j, 3]
        pr, pp = partial(d, S, dv, [DAB])
        raw = stats.pearsonr(d[S], d[dv])[0]
        a.bar(['raw', 'partial\n| d(A,B)'], [raw, pr],
              color=['#5B3E96', '#B0A6CC'])
        a.axhline(0, color='k', lw=.8)
        a.set_ylabel(f'r with {lab}')
        a.set_title(f'does shift survive d(A,B)?\npartial r={pr:+.3f} (p={pp:.1e})',
                    fontsize=10)

    # bottom-left of row 0 replaced: orthogonality panel
    a = ax[0, 0]
    for spine in a.spines.values():
        spine.set_linewidth(2); spine.set_color('#5B3E96')

    for a in ax.ravel():
        a.grid(alpha=.25)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = f'{OUT}/fig_{tag}.png'
    plt.savefig(p, dpi=140)
    plt.close()
    print(f'\n[figure] {p}')

    # --- second figure: the confound battery ------------------------------
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.9))
    fig.suptitle(f'Is the geometric metric a restatement of something trivial? — {tag}',
                 fontsize=13)
    for a, c in zip(ax, CONF + [INC]):
        a.scatter(d[c], d[S], s=7, alpha=.35, color='#2F6F4E')
        r, p = stats.pearsonr(d[c], d[S])
        a.set_xlabel(c); a.set_ylabel('geometric shift')
        a.set_title(f'r={r:+.3f}, p={p:.1e}', fontsize=10)
        a.grid(alpha=.25)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p2 = f'{OUT}/fig_{tag}_confounds.png'
    plt.savefig(p2, dpi=140)
    plt.close()
    print(f'[figure] {p2}')


def main():
    allrows = []
    for dataset, mode in [('shapenet', '3d'), ('shapenet', '2d'), ('shapegen', '2d')]:
        d = load(dataset, mode)
        if d is None:
            print(f'\n[skip] {dataset}/{mode}: no shift csv yet')
            continue
        allrows += run(d, dataset, mode)
    if allrows:
        pd.DataFrame(allrows).to_csv(f'{OUT}/stats_all.csv', index=False)
        print(f'\nwrote {OUT}/stats_all.csv')


if __name__ == '__main__':
    main()
