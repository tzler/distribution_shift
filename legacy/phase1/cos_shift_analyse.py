"""
Test the two claims about the oddity-blind cosine-kNN shift measure.

1. The design choices do what they are supposed to:
     - oddity-blind  -> shift is NOT bounded below by 1/2 d(A,B), so its correlation
                        with d(A,B) should be far below the 0.99+ of a shared-neighbour min
     - hub-robust    -> k=50 differs from, and is more stable than, k=1
     - norm-free     -> low correlation with feature magnitude (feat_l1) and pixel stats

2. The neutral-ruler vs own-space inversion:
     NEUTRAL RULER (fixed pretrained encoder): accuracy, margin and fine-tuning
       advantage all FALL as shift rises.
     OWN SPACE (the evaluated fine-tuned model's own features): all three RISE.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
L = f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
MOCHI = f'{NAV}/MOCHI'
OOD = (f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_'
       'multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|45|'
       '---most-results-ood/ood_analysis/ood_analysis_results.csv')
EMP = 'trial_distance_(L1_not_normalized)'
DABm = 'd_AB_(L1_not_normalized)'
KS = [1, 10, 50, 200]


def load():
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o = pd.read_csv(OOD)
    o['trial'] = m['trial'].values
    o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
    o['ft_margin'] = o['fine_tuned_oddity_margin']
    o['pre_margin'] = o['pretrained_oddity_margin']
    keep = ['trial', 'dataset', 'pretrained_correct', 'fine_tuned_correct',
            'advantage', 'ft_margin', 'pre_margin', 'human_accuracy', 'human_rt',
            EMP, DABm]
    d = o[keep].copy()
    t = pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv')
    d = d.merge(t[['trial', 'feat_l1', 'pix_l1', 'coverage', 'contrast']],
                on='trial', validate='1:1')
    for tag in ['neutral', 'ownspace']:
        f = f'{G}/out/cosshift_{tag}.csv'
        if os.path.exists(f):
            c = pd.read_csv(f)
            c = c.rename(columns={f'cosshift_k{k}': f'{tag}_k{k}' for k in KS})
            d = d.merge(c[['trial'] + [f'{tag}_k{k}' for k in KS]], on='trial', how='left')
    return d


def star(p):
    return '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''


def main():
    d = load()
    arms = [a for a in ['neutral', 'ownspace'] if f'{a}_k50' in d]
    print(f'n = {len(d)} trials;  arms present: {arms}\n')

    # ---- 1. did the design choices work? --------------------------------
    print('=' * 96)
    print('1. DESIGN CHECKS  (all trials, trial level)')
    print('=' * 96)
    for a in arms:
        X = f'{a}_k50'
        print(f'\n  [{a}]  r({X}, ...)')
        for c, lab in [(DABm, 'd_AB  (oddity decision variable)'),
                       (EMP, 'shared-neighbour l1 shift'),
                       ('feat_l1', 'feature magnitude'),
                       ('pix_l1', 'pixel l1'), ('coverage', 'coverage'),
                       ('contrast', 'contrast')]:
            r, p = stats.pearsonr(d[X], d[c])
            print(f'      {lab:34s} r={r:+.3f}{star(p):3s}  p={p:.1e}')
        print(f'      -- k sensitivity (hub robustness)')
        for k in KS:
            r = stats.pearsonr(d[f'{a}_k{k}'], d[DABm])[0]
            print(f'         k={k:<4d} r with d_AB = {r:+.3f}   '
                  f'r with k=50 = {stats.pearsonr(d[f"{a}_k{k}"], d[X])[0]:+.3f}')

    # ---- 2. the neutral vs own-space inversion --------------------------
    print('\n' + '=' * 96)
    print('2. NEUTRAL RULER vs OWN SPACE   (k=50)')
    print('=' * 96)
    dvs = [('pretrained_correct', 'pretrained accuracy'),
           ('fine_tuned_correct', 'fine-tuned accuracy'),
           ('ft_margin', 'fine-tuned oddity margin'),
           ('advantage', 'fine-tuning advantage'),
           ('human_accuracy', 'human accuracy'),
           ('human_rt', 'human RT')]
    print(f'\n{"dependent variable":28s}' + ''.join(f'{a.upper():>22s}' for a in arms))
    for dv, lab in dvs:
        cells = []
        for a in arms:
            r, p = stats.pearsonr(d[f'{a}_k50'], d[dv])
            cells.append(f'{r:+.3f}{star(p)}')
        print(f'{lab:28s}' + ''.join(f'{c:>22s}' for c in cells))

    if len(arms) == 2:
        r = stats.pearsonr(d['neutral_k50'], d['ownspace_k50'])
        print(f'\n  r(neutral, ownspace) = {r[0]:+.3f}  p={r[1]:.1e}')
        print('  -> if the two arms disagree in SIGN on the DVs above while the two shift')
        print('     measures themselves are positively correlated, the inversion is a')
        print('     property of the ruler, not of the trials.')

    # ---- per dataset ----------------------------------------------------
    print('\n' + '=' * 96)
    print('3. BY DATASET (k=50)')
    print('=' * 96)
    for a in arms:
        print(f'\n  [{a}]')
        print(f'    {"dataset":10s}{"n":>6s}' +
              ''.join(f'{l:>26s}' for _, l in dvs[:4]))
        for ds, sub in d.groupby('dataset'):
            cells = []
            for dv, _ in dvs[:4]:
                r, p = stats.pearsonr(sub[f'{a}_k50'], sub[dv])
                cells.append(f'{r:+.3f}{star(p)}')
            print(f'    {ds:10s}{len(sub):>6d}' + ''.join(f'{c:>26s}' for c in cells))

    d.to_csv(f'{G}/out/cosshift_merged.csv', index=False)
    print(f'\nwrote {G}/out/cosshift_merged.csv')


if __name__ == '__main__':
    main()
