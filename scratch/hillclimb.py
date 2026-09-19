"""Hill climb over (representation x estimator x k x trial-metric).

OBJECTIVE   |r| against MARGIN ADVANTAGE (fine-tuned - base) on the 706 within-category
            trials -- the only row that survives its base-DINOv2 control, and a DV that
            differences that control away by construction.
CONTROL     r against the BASE DINOv2 margin is reported for every candidate, never
            used to filter.
SCOPE       single combinations only: one representation, one estimator, one k, one
            trial-level metric. No concatenation, no fitted weights.

Because the max of ~2,000 candidates on n=706 is inflated by construction, nothing is
reported without (a) 5-fold cross-validation over TRIALS -- select on 4 folds, score on
the held-out fold -- and (b) a permutation null on the max statistic itself.
"""
import ast, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

G = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, G)
NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
EPS = 1e-6
KS = [1, 10, 50, 200]
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}
INV = {v: k for k, v in SYN.items()}
REPS = {'d57': 'bank3d_shapenet_trained|test3d_shapenet',
        'voxel8': 'bank_voxel8|test_voxel8',
        'voxel16': 'bank_voxel16|test_voxel16',
        'voxel32': 'bank_voxel32|test_voxel32',
        'bbox': 'bank_bbox|test_bbox',
        'multiview': 'bank_multiview|test_multiview',
        'structure': 'bank_structure|test_structure',
        'volatility': 'bank_volatility|test_volatility'}
NOK = ['centroid', 'mahalanobis', 'pca_recon', 'energy_lse']
WITHK = ['knn_mean', 'knn_kth', 'local_norm', 'rank_pct']
METRICS = ['both_mean', 'oddity_only', 'matched_only', 'closer_one', 'further_one',
           'odd_minus_matched', 'abs_asymmetry', 'over_dAB', 'minus_dAB',
           'over_dTT', 'minus_dTT', 'train_vs_partner']


def prep(bf, tf):
    bz = np.load(bf, allow_pickle=True); tz = np.load(tf, allow_pickle=True)
    B = bz['X'].astype(np.float32); T = tz['X'].astype(np.float32)
    mu = B.mean(0); sd = B.std(0); sd[sd < 1e-9] = 1.0
    B -= mu; B /= sd; T -= mu; T /= sd          # in place: no full-size copies
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    ids = list(bz['ids']); tids = {k: i for i, k in enumerate(tz['ids'])}
    bz.close(); tz.close()
    return ids, B, tids, T


def estimators_for_bank(TN, Bc):
    """All estimator x k variants at once, from one distance matrix. -> {name: vec}"""
    D = (1.0 - TN @ Bc.T).astype(np.float32)
    n = D.shape[1]
    kmax = min(max(KS), n)
    part = np.sort(np.partition(D, kmax - 1, axis=1)[:, :kmax], axis=1)
    out = {}
    for k in KS:
        kk = min(k, n)
        out[f'knn_mean_k{k}'] = part[:, :kk].mean(1)
        out[f'knn_kth_k{k}'] = part[:, kk - 1]
    c = Bc.mean(0, keepdims=True); c = c / (np.linalg.norm(c) + 1e-12)
    out['centroid'] = (1.0 - TN @ c.T)[:, 0]
    dim = Bc.shape[1]
    if dim <= 2000:                                   # covariance is dim x dim
        mu = Bc.mean(0); C = np.cov(Bc.T.astype(np.float64))
        C = 0.9 * C + 0.1 * np.eye(dim) * np.trace(C) / dim
        P = np.linalg.pinv(C); d = (TN - mu).astype(np.float64)
        out['mahalanobis'] = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', d, P, d), 0))
    mu = Bc.mean(0); X = Bc - mu
    ncomp = min(10, X.shape[0] - 1, dim)
    Gm = (X @ X.T).astype(np.float64)                  # n x n, not dim x dim
    w, U = np.linalg.eigh(Gm)
    idx = np.argsort(w)[::-1][:ncomp]
    w = np.maximum(w[idx], 1e-12); U = U[:, idx]
    V = (X.T @ U) / np.sqrt(w)                         # dim x ncomp
    V = V.astype(np.float32)
    d = TN - mu
    out['pca_recon'] = np.linalg.norm(d - (d @ V) @ V.T, axis=1)
    del Gm, U, V, X
    tau = 0.05; mn = D.min(1, keepdims=True)
    out['energy_lse'] = mn[:, 0] - tau * np.log(np.exp(-(D - mn) / tau).sum(1))
    DB = (1.0 - Bc @ Bc.T).astype(np.float32)
    np.fill_diagonal(DB, np.inf)
    kbmax = min(max(KS), n - 1)
    bpart = np.sort(np.partition(DB, kbmax - 1, axis=1)[:, :kbmax], axis=1)
    nn_idx = np.argmin(D, axis=1)
    for k in KS:
        kb = min(k, n - 1)
        prof = bpart[:, :kb].mean(1)
        out[f'local_norm_k{k}'] = part[:, 0] / (prof[nn_idx] + 1e-9)
        out[f'rank_pct_k{k}'] = (np.searchsorted(np.sort(prof), part[:, 0])
                                 / len(prof))
    dTT = {k: float(bpart[:, :min(k, n - 1)].mean()) for k in KS}
    del D, DB, part, bpart
    return out, dTT


def trials_list():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    out, own = [], {}
    for _, r in mm.iterrows():
        ims = ast.literal_eval(r['images']); o = int(r['oddity_index'])
        out.append((r['trial'], oid(ims[o]),
                    [oid(f) for j, f in enumerate(ims) if j != o]))
        s = {f[:-4].split('_')[0] for f in ims}
        if len(s) == 1:
            own[r['trial']] = INV.get(list(s)[0])
    return out, own


def behaviour():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in SYN:
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['margin_adv'] = (o['fine_tuned_oddity_margin']
                           - o['pretrained_oddity_margin'])
        o['pre'] = o['pretrained_oddity_margin']
        o['ft'] = o['fine_tuned_oddity_margin']
        o['human_rt'] = o['human_rt']; o['human_acc'] = o['human_accuracy']
        o['gap'] = o['human_accuracy'] - o['fine_tuned_correct']
        rows.append(o[['trial', 'category', 'margin_adv', 'pre', 'ft',
                       'human_rt', 'human_acc', 'gap']])
    return pd.concat(rows)


def build_candidates(only=None):
    tl, own = trials_list()
    beh = behaviour()
    cand = {}                                  # (rep,est,metric) -> {trial: value}
    for rep, spec in REPS.items():
        if only and rep != only:
            continue
        b, t = spec.split('|')
        t0 = time.time()
        bids, BN, tix, TN = prep(f'{G}/bank/{b}.npz', f'{G}/bank/{t}.npz')
        syn = np.array([i.split('/')[0] for i in bids])
        # dAB and the trial->row lookups do NOT depend on the estimator, so build them
        # once per representation instead of once per (category, estimator) pair.
        tinfo = []
        for trial, kb_id, ka_ids in tl:
            if own.get(trial) is None:
                continue
            if kb_id not in tix or not all(k in tix for k in ka_ids):
                continue
            rb = tix[kb_id]; ra = [tix[k] for k in ka_ids]
            a = TN[ra].mean(0); a = a / (np.linalg.norm(a) + 1e-12)
            tinfo.append((trial, own[trial], rb, ra, float(1.0 - a @ TN[rb])))
        for cat, s_ in SYN.items():
            sel = syn == s_
            if sel.sum() < 200:
                continue
            sub = [ti for ti in tinfo if ti[1] == cat]      # WITHIN CATEGORY only
            if not sub:
                continue
            est, dTT = estimators_for_bank(TN, BN[sel])
            for ename, dvec in est.items():
                kk = int(ename.split('_k')[-1]) if '_k' in ename else 50
                dtt = dTT[kk]
                for trial, _c, rb, ra, dAB in sub:
                    dB = float(dvec[rb]); dA = float(dvec[ra].mean())
                    both = .5 * (dA + dB)
                    vals = dict(both_mean=both, oddity_only=dB, matched_only=dA,
                                closer_one=min(dA, dB), further_one=max(dA, dB),
                                odd_minus_matched=dB - dA, abs_asymmetry=abs(dB - dA),
                                over_dAB=both / (abs(dAB) + EPS), minus_dAB=both - dAB,
                                over_dTT=both / (dtt + EPS), minus_dTT=both - dtt,
                                train_vs_partner=both - dAB)
                    for mname, v in vals.items():
                        cand.setdefault((rep, ename, mname), {})[trial] = v
            del est
        del BN, TN
        print(f'  {rep:11s} done in {time.time()-t0:6.1f}s  '
              f'({len(cand):,} candidate columns so far)', flush=True)
    return cand, beh, own


if __name__ == '__main__':
    import argparse
    _ap = argparse.ArgumentParser(); _ap.add_argument('--rep', default=None)
    _rep = _ap.parse_args().rep
    cand, beh, own = build_candidates(_rep)
    trials = sorted({t for v in cand.values() for t in v})
    keys = sorted(cand)
    M = np.full((len(trials), len(keys)), np.nan)
    for j, k in enumerate(keys):
        v = cand[k]
        M[:, j] = [v.get(t, np.nan) for t in trials]
    ok = ~np.isnan(M).any(0) & (np.nanstd(M, 0) > 1e-12)
    M = M[:, ok]; keys = [k for k, o in zip(keys, ok) if o]
    b = beh[beh.apply(lambda r: own.get(r['trial']) == r['category'], axis=1)]
    b = b.set_index('trial').loc[trials]
    _out = (f'{G}/out/hillclimb_candidates.npz' if not _rep
            else f'{G}/out/hc_shard_{_rep}.npz')
    np.savez_compressed(_out,
                        M=M, keys=np.array(['|'.join(k) for k in keys]),
                        trials=np.array(trials),
                        y_margin_adv=b.margin_adv.values, y_pre=b.pre.values,
                        y_ft=b.ft.values, y_rt=b.human_rt.values,
                        y_acc=b.human_acc.values, y_gap=b.gap.values)
    print(f'\n{M.shape[0]} trials x {M.shape[1]:,} candidates -> {_out}')
