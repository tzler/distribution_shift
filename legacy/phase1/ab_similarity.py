"""
Geometric similarity between the two objects in a trial, and how it relates to
behaviour.

    sim(A,B) = g_hat(A) . g_hat(B)          (cosine, on L2-normalised descriptors)
    dist(A,B) = 1 - sim(A,B)

phi(A) = mean descriptor of the trial's matched images (renormalised), phi(B) = the
oddity's descriptor -- the same phi construction the manuscript uses, with the
geometric descriptor substituted for the encoder feature.

This is deliberately NOT a distribution-shift measure: it is a property of the trial
alone and never touches the training bank. It is the quantity the shared-neighbour
shift was leaking (that estimator is bounded below by 1/2 d(A,B)), so it is worth
characterising on its own terms.

Prediction if it indexes task difficulty: HIGH similarity -> harder -> lower accuracy,
longer RT.
"""
import ast, os
import numpy as np
import pandas as pd
from scipy import stats

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
MOCHI = f'{NAV}/MOCHI'
OOD = (f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_'
       'multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|45|'
       '---most-results-ood/ood_analysis/ood_analysis_results.csv')
st = lambda p: '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''


def descriptors(dataset):
    """L2-normalised, bank-z-scored descriptors for the test items."""
    if dataset == 'shapenet':
        bank, test = (f'{G}/bank/bank3d_shapenet_trained.npz',
                      f'{G}/bank/test3d_shapenet.npz')
    else:
        bank, test = (f'{G}/bank/bank2d_shapegen.npz', f'{G}/bank/mochi2d.npz')
    B = np.load(bank, allow_pickle=True)['X'].astype(float)
    tz = np.load(test, allow_pickle=True)
    T = tz['X'].astype(float)
    mu, sd = B.mean(0), B.std(0)
    sd[sd < 1e-9] = 1.0
    TZ = (T - mu) / sd
    TN = TZ / (np.linalg.norm(TZ, axis=1, keepdims=True) + 1e-12)
    return {k: i for i, k in enumerate(tz['ids'])}, TN


def build(dataset):
    tix, TN = descriptors(dataset)
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    mm = m[m.dataset == dataset]
    key = (lambda f: '/'.join(f[:-4].split('_')[:2])) if dataset == 'shapenet' else (lambda f: f)
    rows = []
    for _, r in mm.iterrows():
        ims = ast.literal_eval(r['images'])
        o = int(r['oddity_index'])
        kb = key(ims[o])
        ka = [key(f) for j, f in enumerate(ims) if j != o]
        if kb not in tix or not all(k in tix for k in ka):
            continue
        a = TN[[tix[k] for k in ka]].mean(0)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = TN[tix[kb]]
        sim = float(a @ b)
        rows.append({'trial': r['trial'], 'ab_sim': sim, 'ab_dist': 1.0 - sim})
    return pd.DataFrame(rows)


def main():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o = pd.read_csv(OOD)
    o['trial'] = m['trial'].values
    o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
    o['ft_margin'] = o['fine_tuned_oddity_margin']
    o['pre_margin'] = o['pretrained_oddity_margin']

    for dataset in ['shapenet', 'shapegen']:
        d = build(dataset)
        d = d.merge(o[['trial', 'human_accuracy', 'human_rt', 'pretrained_correct',
                       'fine_tuned_correct', 'ft_margin', 'pre_margin', 'advantage']],
                    on='trial', validate='1:1')
        b = pd.read_csv(f'{G}/out/blindshift_{dataset}.csv')[['trial', 'blind_k50']]
        d = d.merge(b, on='trial', validate='1:1')
        d.to_csv(f'{G}/out/absim_{dataset}.csv', index=False)

        print('=' * 88)
        print(f'{dataset}   n = {len(d)}    geometric A-B similarity')
        print('=' * 88)
        print(f'  sim range [{d.ab_sim.min():+.3f}, {d.ab_sim.max():+.3f}]  '
              f'mean {d.ab_sim.mean():+.3f}')
        print(f'\n  {"DV":28s}{"r with SIMILARITY":>20s}{"r with DISTANCE":>18s}')
        for dv, lab in [('human_accuracy', 'human accuracy'),
                        ('human_rt', 'human RT'),
                        ('pretrained_correct', 'pretrained accuracy'),
                        ('fine_tuned_correct', 'fine-tuned accuracy'),
                        ('pre_margin', 'pretrained margin'),
                        ('ft_margin', 'fine-tuned margin'),
                        ('advantage', 'fine-tuning advantage')]:
            rs, ps = stats.pearsonr(d['ab_sim'], d[dv])
            rd, pd_ = stats.pearsonr(d['ab_dist'], d[dv])
            print(f'  {lab:28s}{f"{rs:+.3f}{st(ps)}":>20s}{f"{rd:+.3f}{st(pd_)}":>18s}')
        r = stats.pearsonr(d['ab_sim'], d['blind_k50'])
        print(f'\n  r(A-B similarity, oddity-blind shift k=50) = {r[0]:+.3f} (p={r[1]:.1e})')
        print(f'  -> the two are largely separable, which is the point of the '
              f'oddity-blind estimator.\n')


if __name__ == '__main__':
    main()
