"""
A SPACE of model-free shift metrics, built from four geometric quantities.

Building blocks (all shape-only; no encoder anywhere):
    dA   = distance from the MATCHED object to the training objects  (mean of k=50 NN)
    dB   = distance from the ODDITY  object to the training objects  (mean of k=50 NN)
    dAB  = distance between the trial's two objects
    dTT  = the training set's own internal spread (mean k-NN distance among its members)

The metrics, each stated in one sentence:

  RAW — how far is this trial from the training set?
    both_mean      average of dA and dB                       (what we have been using)
    oddity_only    just the oddity object's distance
    matched_only   just the matched object's distance
    closer_one     whichever of the two objects is nearer to training
    further_one    whichever of the two objects is further from training

  ASYMMETRY — do the two objects differ in how familiar they are?
    odd_minus_matched   dB - dA   (is the oddity the unfamiliar one?)
    abs_asymmetry       |dB - dA|

  NORMALISED BY THE TRIAL — far from training relative to this trial's own difficulty
    over_dAB       (dA+dB)/2 divided by dAB
    minus_dAB      (dA+dB)/2 minus dAB

  NORMALISED BY THE TRAINING SET — far relative to how spread out training itself is
    over_dTT       (dA+dB)/2 divided by dTT
    minus_dTT      (dA+dB)/2 minus dTT

  CONTRASTIVE — is training a better match for this object than its own partner is?
    train_vs_partner   mean over the trial's objects of  d(object,Train) - dAB

Evaluated on three DVs with the same battery, and crucially against the PRETRAINED
margin: that encoder never saw any of these training sets, so a valid shift measure
must not predict it.
"""
import ast, os, sys
import numpy as np
import pandas as pd
from scipy import stats

G = os.path.dirname(os.path.abspath(__file__))
NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
K = 50
EPS = 1e-6
SYN = {'airplane': '02691156', 'bench': '02828884', 'cabinet': '02933112',
       'car': '02958343', 'chair': '03001627', 'display': '03211117',
       'lamp': '03636649', 'loudspeaker': '03691459', 'sofa': '04256520',
       'table': '04379243', 'telephone': '04401088', 'watercraft': '04530566'}

DEFN = {
    'both_mean':        'average distance of both objects to training',
    'oddity_only':      'oddity object distance to training',
    'matched_only':     'matched object distance to training',
    'closer_one':       'the nearer object of the two',
    'further_one':      'the further object of the two',
    'odd_minus_matched': 'oddity distance minus matched distance',
    'abs_asymmetry':    '|oddity distance - matched distance|',
    'over_dAB':         'distance to training / distance between the two objects',
    'minus_dAB':        'distance to training - distance between the two objects',
    'over_dTT':         "distance to training / training set's own spread",
    'minus_dTT':        "distance to training - training set's own spread",
    'train_vs_partner': 'is training a closer match than the trial partner?',
}
METRICS = list(DEFN)


def prep(bf, tf, cols=None):
    bz = np.load(bf, allow_pickle=True); tz = np.load(tf, allow_pickle=True)
    B, T = bz['X'].astype(np.float64), tz['X'].astype(np.float64)
    if cols is not None:
        B, T = B[:, cols], T[:, cols]
    mu, sd = B.mean(0), B.std(0); sd[sd < 1e-9] = 1.0
    BZ, TZ = (B - mu) / sd, (T - mu) / sd
    BN = BZ / (np.linalg.norm(BZ, axis=1, keepdims=True) + 1e-12)
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    return list(bz['ids']), BN, {k: i for i, k in enumerate(tz['ids'])}, TN


def trials_list():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    oid = lambda f: '/'.join(f[:-4].split('_')[:2])
    out = []
    for _, r in mm.iterrows():
        ims = ast.literal_eval(r['images']); o = int(r['oddity_index'])
        out.append((r['trial'], oid(ims[o]),
                    [oid(f) for j, f in enumerate(ims) if j != o]))
    return out


def behaviour():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for cat in SYN:
        o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
        o['trial'] = m['trial'].values; o['category'] = cat
        o['ft'] = o['fine_tuned_oddity_margin']
        o['pre'] = o['pretrained_oddity_margin']
        o['adv'] = o['fine_tuned_correct'] - o['pretrained_correct']
        rows.append(o[['trial', 'category', 'ft', 'pre', 'adv']])
    return pd.concat(rows)


def build(bf, tf, cols=None):
    bids, BN, tix, TN = prep(bf, tf, cols)
    syn = np.array([i.split('/')[0] for i in bids])
    tl = trials_list()
    recs = []
    for cat, s in SYN.items():
        sel = syn == s
        if sel.sum() < K:
            continue
        Bc = BN[sel]
        D = 1.0 - TN @ Bc.T
        kk = min(K, D.shape[1])
        d_train = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1).mean(1)
        DB = 1.0 - Bc @ Bc.T
        np.fill_diagonal(DB, np.inf)
        kb = min(K, DB.shape[1] - 1)
        dTT = float(np.sort(np.partition(DB, kb - 1, axis=1)[:, :kb], axis=1).mean())
        for trial, kb_id, ka_ids in tl:
            if kb_id not in tix or not all(k in tix for k in ka_ids):
                continue
            dB = float(d_train[tix[kb_id]])
            dA = float(np.mean([d_train[tix[k]] for k in ka_ids]))
            a = TN[[tix[k] for k in ka_ids]].mean(0)
            a = a / (np.linalg.norm(a) + 1e-12)
            dAB = float(1.0 - a @ TN[tix[kb_id]])
            both = 0.5 * (dA + dB)
            recs.append(dict(
                trial=trial, category=cat, dA=dA, dB=dB, dAB=dAB, dTT=dTT,
                both_mean=both, oddity_only=dB, matched_only=dA,
                closer_one=min(dA, dB), further_one=max(dA, dB),
                odd_minus_matched=dB - dA, abs_asymmetry=abs(dB - dA),
                over_dAB=both / (abs(dAB) + EPS), minus_dAB=both - dAB,
                over_dTT=both / (dTT + EPS), minus_dTT=both - dTT,
                train_vs_partner=both - dAB))
    return pd.DataFrame(recs)


def evaluate(d, tag):
    d = d.merge(behaviour(), on=['trial', 'category'])
    rows = []
    for mname in METRICS:
        t = d.groupby('trial').agg(x=(mname, 'mean'), ft=('ft', 'mean'),
                                   pre=('pre', 'first'), adv=('adv', 'mean'),
                                   dAB=('dAB', 'first')).reset_index()
        if t.x.std() < 1e-12:
            continue
        r_ft = stats.pearsonr(t.x, t.ft)
        r_pre = stats.pearsonr(t.x, t.pre)
        r_ab = stats.pearsonr(t.x, t.dAB)[0]
        w = d.copy()
        for c in [mname, 'ft']:
            w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                    - d.groupby('category')[c].transform('mean') + d[c].mean())
        r_w = stats.pearsonr(w[mname], w['ft'])[0] if w[mname].std() > 1e-12 else np.nan
        rows.append(dict(rep=tag, metric=mname, defn=DEFN[mname],
                         pooled_ft=r_ft[0], p_ft=r_ft[1], pooled_pre=r_pre[0],
                         p_pre=r_pre[1], r_dAB=r_ab, within_ft=r_w))
    return rows


def report(df, tag):
    print(f'\n{"="*118}\n{tag}\n{"="*118}')
    print(f'{"metric":20s}{"what it is":52s}{"pooled":>9s}{"pretrained":>12s}'
          f'{"|r| dAB":>9s}{"within":>9s}')
    print('-' * 118)
    for _, r in df.sort_values('pooled_ft').iterrows():
        leak = ' LEAK' if abs(r.pooled_pre) > .10 else ''
        print(f'{r.metric:20s}{r.defn:52s}{r.pooled_ft:>+9.3f}{r.pooled_pre:>+12.3f}'
              f'{abs(r.r_dAB):>9.2f}{r.within_ft:>+9.3f}{leak}')


if __name__ == '__main__':
    allr = []
    for tag, bf, tf, cols in [
            ('57-number shape descriptor', f'{G}/bank/bank3d_shapenet_trained.npz',
             f'{G}/bank/test3d_shapenet.npz', None),
            ('raw 16^3 voxel grid', f'{G}/bank/bank_voxel16.npz',
             f'{G}/bank/test_voxel16.npz', None)]:
        rows = evaluate(build(bf, tf, cols), tag)
        report(pd.DataFrame(rows), tag)
        allr += rows
    out = pd.DataFrame(allr)
    out.to_csv(f'{G}/out/metric_space.csv', index=False)
    print(f'\nwrote {G}/out/metric_space.csv')
    print('\npooled     = correlation with the fine-tuned margin, one value per trial')
    print('pretrained = same, but against the PRETRAINED margin. That encoder never saw')
    print('             any of these training sets, so anything non-zero here is a leak.')
    print('|r| dAB    = how much the metric is just measuring the two objects\' difference')
    print('within     = correlation using the 12-model design (difficulty held constant)')
