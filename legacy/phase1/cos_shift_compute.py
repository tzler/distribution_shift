"""
Oddity-blind, hub-robust, norm-free distribution shift.

    shift(t) = (1/|t|) sum_{x in t}  (1/k) sum_{i in kNN_k(x)} ( 1 - f_hat(x) . C_hat_i )

f_hat(x) : L2-normalised DINOv2 ViT-L/14 reg4 CLS feature of trial image x
C_hat_i  : L2-normalised features of the HIDA training bank (MOCHI objects excluded)
k        : 50 by default

Oddity-blind: averages a PER-IMAGE quantity over all images of the trial and never
forms a difference between them, so it carries no 1/2 d(A,B) lower bound.
Hub-robust:  mean over k nearest, not a single argmin.
Norm-free:   cosine on L2-normalised features.

Run once per arm:
  NEUTRAL RULER -> features from the fixed pretrained encoder
  OWN SPACE     -> features from the evaluated fine-tuned model
"""
import argparse, ast, os
import numpy as np
import pandas as pd
from scipy import stats

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
MOCHI = f'{NAV}/MOCHI'
OOD = (f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_'
       'multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|45|'
       '---most-results-ood/ood_analysis/ood_analysis_results.csv')


def per_image_shift(F, idx, ks, chunk=512):
    """Mean cosine distance to the k nearest bank images, for every MOCHI image."""
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    is_bank = idx.source.str.startswith('bank').values
    B = F[is_bank]
    q = np.where(~is_bank)[0]
    kmax = max(ks)
    out = {k: np.empty(len(q)) for k in ks}
    for s in range(0, len(q), chunk):
        sl = q[s:s + chunk]
        sim = F[sl] @ B.T                     # cosine similarity
        part = np.partition(-sim, kmax - 1, axis=1)[:, :kmax]
        part = -np.sort(-part, axis=1)        # descending similarity = ascending distance
        for k in ks:
            out[k][s:s + len(sl)] = (1.0 - part[:, :k]).mean(1)
    keys = idx.key.values[~is_bank]
    return {k: dict(zip(keys, out[k])) for k in ks}, int(is_bank.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ks', default='1,10,50,200')
    ap.add_argument('--feat', default=f'{G}/cosfeat')
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(',')]

    F = np.load(f'{a.feat}/{a.tag}/features.npy')
    idx = pd.read_csv(f'{a.feat}/{a.tag}/index.csv')
    per_img, n_bank = per_image_shift(F, idx, ks)
    print(f'{a.tag}: {F.shape[0]} images, bank={n_bank}, dim={F.shape[1]}', flush=True)

    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    rows = []
    for _, r in m.iterrows():
        ims = ast.literal_eval(r['images'])
        rec = {'trial': r['trial'], 'dataset': r['dataset'], 'n_img': len(ims)}
        ok = True
        for k in ks:
            vals = [per_img[k][f] for f in ims if f in per_img[k]]
            if len(vals) != len(ims):
                ok = False
                break
            rec[f'cosshift_k{k}'] = float(np.mean(vals))   # oddity-blind average
        if ok:
            rows.append(rec)
    d = pd.DataFrame(rows)
    out = f'{G}/out/cosshift_{a.tag}.csv'
    d.to_csv(out, index=False)
    print(f'wrote {out}  n={len(d)}  ({len(m)-len(d)} trials missing an image)')
    print(d.groupby('dataset')[f'cosshift_k50'].describe()[['count', 'mean', 'std']]
          .round(4).to_string())


if __name__ == '__main__':
    main()
