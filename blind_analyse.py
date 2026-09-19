"""
Does the oddity-blind geometric shift behave the way a distribution-shift measure
should, once the d(A,B) leak is closed?

Tests, in order:
  1. did the three design choices work (d_AB leak, hub robustness, norm-freeness)
  2. the plain trial-level relationships to model and human behaviour
  3. the within-trial x per-category-model design, which controls d(A,B) by construction
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

NAV = '/vast/projects/bonnen/naturalistic-navig'
G = f'{NAV}/Dist-shift-data/geometric_shift'
L = f'{NAV}/Dist-shift-data/L1norm_vs_distshift'
MOCHI = f'{NAV}/MOCHI'
S = f'{NAV}/Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses'
OOD = (f'{NAV}/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_'
       'multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|45|'
       '---most-results-ood/ood_analysis/ood_analysis_results.csv')
EMP = 'trial_distance_(L1_not_normalized)'
DABm = 'd_AB_(L1_not_normalized)'
KS = (1, 10, 50, 200)
st = lambda p: '***' if p < 1e-3 else '**' if p < .01 else '*' if p < .05 else ''


def base(dataset, mode):
    m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
    o = pd.read_csv(OOD)
    o['trial'] = m['trial'].values
    o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
    o['ft_margin'] = o['fine_tuned_oddity_margin']
    d = pd.read_csv(f'{G}/out/blindshift_{dataset}.csv')
    d = d.merge(o[['trial', 'dataset', 'pretrained_correct', 'fine_tuned_correct',
                   'advantage', 'ft_margin', 'human_accuracy', 'human_rt',
                   EMP, DABm]], on='trial', validate='1:1')
    old = pd.read_csv(f'{G}/out/shift{mode}_{dataset}.csv')
    keep = ['trial', f'geom_shift_{mode}', f'geom_dAB_{mode}', f'geom_knn200_{mode}']
    d = d.merge(old[keep], on='trial', validate='1:1')
    t = pd.read_csv(f'{L}/trials_vit_base_patch16_224.dino.csv')
    return d.merge(t[['trial', 'feat_l1', 'pix_l1', 'coverage', 'contrast']],
                   on='trial', validate='1:1')


def main():
    for dataset, mode in [('shapenet', '3d'), ('shapegen', '2d')]:
        d = base(dataset, mode)
        DABg, OLD = f'geom_dAB_{mode}', f'geom_shift_{mode}'
        print('=' * 94)
        print(f'{dataset}   n = {len(d)}')
        print('=' * 94)

        print('\n1. DESIGN CHECKS — did oddity-blind averaging close the d(A,B) leak?')
        print(f'   {"metric":24s}{"r with d(A,B)_geom":>21s}{"r with d(A,B)_DINOv2":>22s}')
        print(f'   {"shared-neighbour min (old)":24s}'
              f'{stats.pearsonr(d[OLD], d[DABg])[0]:>21.3f}'
              f'{stats.pearsonr(d[OLD], d[DABm])[0]:>22.3f}')
        for k in KS:
            print(f'   {"oddity-blind k=" + str(k):24s}'
                  f'{stats.pearsonr(d[f"blind_k{k}"], d[DABg])[0]:>21.3f}'
                  f'{stats.pearsonr(d[f"blind_k{k}"], d[DABm])[0]:>22.3f}')

        print('\n   norm-freeness / low-level confounds (k=50)')
        for c in ['feat_l1', 'pix_l1', 'coverage', 'contrast', EMP]:
            r, p = stats.pearsonr(d['blind_k50'], d[c])
            print(f'      r(blind_k50, {c:34s}) = {r:+.3f}{st(p)}')

        print('\n2. TRIAL-LEVEL RELATIONSHIPS  (no fixed effects, no binning)')
        dvs = [('pretrained_correct', 'pretrained accuracy'),
               ('fine_tuned_correct', 'fine-tuned accuracy'),
               ('ft_margin', 'fine-tuned oddity margin'),
               ('advantage', 'fine-tuning advantage'),
               ('human_accuracy', 'human accuracy'),
               ('human_rt', 'human RT')]
        print(f'   {"DV":26s}{"oddity-blind k=50":>19s}{"old shared-min":>17s}'
              f'{"DINOv2 empirical":>19s}')
        for dv, lab in dvs:
            cells = []
            for X in ['blind_k50', OLD, EMP]:
                r, p = stats.pearsonr(d[X], d[dv])
                cells.append(f'{r:+.3f}{st(p)}')
            print(f'   {lab:26s}{cells[0]:>19s}{cells[1]:>17s}{cells[2]:>19s}')

        # ---- 3. within-trial design (shapenet only: needs per-category models)
        pc = f'{G}/out/blindshift_{dataset}_percat.csv'
        if not os.path.exists(pc):
            print()
            continue
        panel = pd.read_csv(pc)
        rows = []
        m = pd.read_csv(f'{MOCHI}/mochi_trials.csv')
        for cat in panel.category.unique():
            o = pd.read_csv(f'{S}/{cat}/ood_analysis_results.csv')
            o['trial'] = m['trial'].values
            o['category'] = cat
            o['advantage'] = o['fine_tuned_correct'] - o['pretrained_correct']
            o['ft_margin'] = o['fine_tuned_oddity_margin']
            rows.append(o[['trial', 'category', 'advantage', 'ft_margin',
                           'fine_tuned_correct', EMP]])
        p = panel.merge(pd.concat(rows), on=['trial', 'category'])

        def wthin(df, cols):
            out = df.copy()
            for c in cols:
                out[c] = (df[c] - df.groupby('trial')[c].transform('mean')
                          - df.groupby('category')[c].transform('mean') + df[c].mean())
            return out

        print(f'\n3. WITHIN-TRIAL x CATEGORY-MODEL  '
              f'({p.trial.nunique()} trials x {p.category.nunique()} models = {len(p)} obs)')
        print(f'   {"predictor":24s}{"DV":24s}{"pooled r":>11s}{"within r":>11s}{"p":>11s}')
        for X, xl in [('blind_k50', 'oddity-blind k=50'), (EMP, 'DINOv2 empirical')]:
            for dv, dl in [('advantage', 'fine-tuning advantage'),
                           ('ft_margin', 'fine-tuned margin')]:
                w = wthin(p, [X, dv])
                rp = stats.pearsonr(p[X], p[dv])[0]
                rw, pw = stats.pearsonr(w[X], w[dv])
                print(f'   {xl:24s}{dl:24s}{rp:+11.3f}{rw:+11.3f}{pw:>11.1e}')
        print()


if __name__ == '__main__':
    main()
