"""
Shift computed against an IMAGE-based training reference.

Reference = the individual training renders the model actually saw (not per-object
means, not the 3D objects). Test = the actual MOCHI images. Representation = a 32x32
silhouette map, spatially resolved rather than collapsed to summary statistics.

    shift(t) = (1/|t|) sum over the trial's images of
               the mean cosine distance to that image's k=50 nearest TRAINING IMAGES

Evaluated exactly as before: all-12-models within-trial, then within-category with the
pretrained margin as the control.
"""
import ast, os, sys
import numpy as np
import pandas as pd
from scipy import stats

G = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, G)
from metric_space import behaviour, SYN                                 # noqa: E402

NAV = '/vast/projects/bonnen/naturalistic-navig'
MOCHI = f'{NAV}/MOCHI'
K = 50
CAT = {v: k for k, v in SYN.items()}


def load():
    b = np.load(f'{G}/bank/imgbank_sil32.npz', allow_pickle=True)
    t = np.load(f'{G}/bank/imgtest_sil32.npz', allow_pickle=True)
    B, T = b['X'].astype(np.float32), t['X'].astype(np.float32)
    mu = B.mean(0)
    B = B - mu; T = T - mu
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    bcat = np.array([i.split('/')[0] for i in b['ids']])
    return B, bcat, T, {k: i for i, k in enumerate(t['ids'])}


def main():
    B, bcat, T, tix = load()
    print(f'training image reference: {B.shape[0]:,} images, {B.shape[1]} dims')
    print(f'MOCHI test images:        {T.shape[0]:,}\n')
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == 'shapenet']
    trials = [(r['trial'], ast.literal_eval(r['images'])) for _, r in mm.iterrows()]

    recs = []
    for cat in sorted(set(bcat)):
        sel = bcat == cat
        if sel.sum() < K:
            continue
        Bc = B[sel]
        D = 1.0 - T @ Bc.T                       # every MOCHI image vs every train image
        kk = min(K, D.shape[1])
        per = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1).mean(1)
        for trial, ims in trials:
            if all(f in tix for f in ims):
                recs.append(dict(trial=trial, category=cat,
                                 shift=float(np.mean([per[tix[f]] for f in ims]))))
        print(f'  {cat:12s} {sel.sum():>7,} reference images', flush=True)
    d = pd.DataFrame(recs).merge(behaviour(), on=['trial', 'category'])
    d.to_csv(f'{G}/out/image_shift_panel.csv', index=False)

    # own-category map
    oid = lambda f: f[:-4].split('_')[0]
    own = {}
    for trial, ims in trials:
        s = {oid(f) for f in ims}
        if len(s) == 1:
            own[trial] = CAT.get(list(s)[0])
    d['own'] = d.trial.map(own)

    w = d.copy()
    for c in ['shift', 'ft']:
        w[c] = (d[c] - d.groupby('trial')[c].transform('mean')
                - d.groupby('category')[c].transform('mean') + d[c].mean())
    r_all = stats.pearsonr(w['shift'], w['ft'])
    sl = np.array([stats.linregress(s['shift'], s['ft']).slope
                   for _, s in d.groupby('trial') if s['shift'].std() > 0])
    print(f'\nALL 12 MODELS, within-trial:  r = {r_all[0]:+.3f}  p = {r_all[1]:.1e}   '
          f'({100*(sl<0).mean():.0f}% of trials negative, n={len(d):,} obs)')

    oc = d[d.category == d.own]
    sh = oc['shift'].values
    Z = np.column_stack([np.ones(len(oc)), oc.pre.values])
    rx = sh - Z @ np.linalg.lstsq(Z, sh, rcond=None)[0]
    ry = oc.ft.values - Z @ np.linalg.lstsq(Z, oc.ft.values, rcond=None)[0]
    pr = stats.pearsonr(rx, ry)
    r_ft = stats.pearsonr(sh, oc.ft.values)
    r_pre = stats.pearsonr(sh, oc.pre.values)
    r_adv = stats.pearsonr(sh, oc.adv.values)
    print(f'\nWITHIN CATEGORY (n={len(oc)}):')
    print(f'  r(shift, fine-tuned margin) = {r_ft[0]:+.3f} (p={r_ft[1]:.4f})')
    print(f'  r(shift, PRETRAINED margin) = {r_pre[0]:+.3f}   <- control, should be ~0')
    print(f'  partial (ft | pretrained)   = {pr[0]:+.3f} (p={pr[1]:.4f})')
    print(f'  r(shift, advantage)         = {r_adv[0]:+.3f} (p={r_adv[1]:.4f})')
    print(f'\nwrote {G}/out/image_shift_panel.csv')


if __name__ == '__main__':
    main()
