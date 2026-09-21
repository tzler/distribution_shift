# evidence/ — admission rule: only figures cited from STATE.md or a REASONING entry

Every file: the script that regenerates it (run from the repo root with the `dev` conda env
unless noted), the data it reads, and the entry that cites it. All thirteen files were admitted at commit 69f2534 (2026-09-19); later additions
name their commit in this table.

| file | script | data | cited by |
|---|---|---|---|
| fig7_margin_validation_clean.png | `fig_margin_clean.py` | `out/blindshift_shapenet_percat.csv` (d57) + category csvs | D02 |
| fig9_margin_rank.png | `fig_margin_rank.py` | same | D02 |
| fig46_moving_training.png | `scratch/fig_moving_training_simple.py` (stats from `scratch/fig_moving_training.py`) | `out/blindshift_shapenet_voxel16_percat.csv` + category csvs; 10k permutations, 2k bootstraps | D02, STATE |
| fig48_accuracy_within_trial.png | `scratch/fig_accuracy_within_trial.py` | same | D02 |
| fig49_entanglement.png | `scratch/fig_entanglement_and_coverage.py` | `out/shift{2d,3d}_*.csv`, `out/blindshift_{shapegen,shapenet}.csv`, `all_model_margins.csv` | D06 |
| fig50_coverage.png | `scratch/fig_entanglement_and_coverage.py` | `out/coverage_sweep_{voxel16,d57}.csv` (from `scratch/coverage_sweep.py`) | D05 |
| fig52_coverage_pooled_control.png | `scratch/fig_coverage_gallery.py` | `bank/{bank,test}_voxel16.npz`, ε = 0.12 | D05 |
| fig54_oncat_three_metrics.png | `scratch/fig_three_metrics.py` | `out/coverage_shapenet_voxel16_percat.csv` + `trial_distance_(L1_not_normalized)` per category | D07 |
| fig56_pooled_three_metrics.png | `scratch/fig_coverage_recipe.py` | same | D05, D06 |
| fig57_oncat_anatomy.png | `scratch/fig_oncat_recipe.py` | `out/coverage_shapenet_voxel16_percat.csv` | D07 |
| fig58_viewdepth_ladder.png | `scratch/fig_coverage_recipe.py` | `out/viewdepth_battery_*.csv` (from `scratch/viewdepth_pipeline.py`, SLURM 8515465/8515516/8515529/8515554/8515600/8515622) | D08 |
| fig60_why_wrong.png | `scratch/fig_why_wrong.py` | `../L1norm_vs_distshift/trials_*.csv`; three-metric numbers from `scratch/fig_three_metrics.py` | D06 |
| manuscript_fig1.png | rendered from `Human-3D-generalization-copy/paper-to-follow-*/distributionshift_neurips2026-3.pdf` p.4 (pymupdf), cropped | the submitted manuscript | State 0 |
| fig62_encoder_space.png | `scratch/fig_encoder_circularity.py` | `knockout/eval/encoder_space_battery.csv` (from `scratch/encoder_space_battery.py`, SLURM 8522238; features from job 8519397) | D15, State 2 |
| fig61_triangle.png | `scratch/fig_triangle.py` | `../L1norm_vs_distshift/trials_vit_base_patch16_224.dino.csv` | D06 |

Numbers quoted in STATE that have no figure: hill-climb (`out/hillclimb_candidates.npz`,
`scratch/hc_repeated.py`); coverage variants (`scratch/coverage_variants.py`);
level-3 residualised test (session record 2026-09-19; script to be added).
| fig64_knockout_batch1.png | `scratch/analyze_knockout.py` | `knockout/eval/*_k10`, `chair_full`; `out/knockout_long.csv` | D18 |
| fig65_knockin_random.png | `scratch/analyze_knockin_random.py` | `knockout/eval/chair_random_100_*`; `out/knockin_random_long.csv` | D18 |
| fig66_chair_clusters_montage.png | inline in `knockout/scripts/make_cluster_exp.py` (clustering) | `design_clusters_chair.json` | D19 |
| fig67_cluster_matrix.png | `scratch/analyze_clusters.py` | `knockout/eval_bank/chair_c*_*`; `banktrials_chair.csv` | D21 |
| fig68_ref_on_bank.png | `scratch/analyze_clusters.py` | `knockout/eval_bank/ref_*` | D21 |
| fig69_cluster_shift.png | `scratch/analyze_clusters_shift.py` | `out/cluster_shift_long.csv` | D22 |
| fig70_shift_vs_setsize.png | `scratch/fig_shift_vs_setsize.py` | `out/shift_vs_setsize.csv` | D23 |
| fig71_margin_vs_dose.png | `scratch/fig_margin_vs_dose.py` (re-run as evaluations land) | `out/margin_vs_dose.csv` | D24 |
| fig72_distance_margin_by_N.png | `scratch/fig_distance_margin_by_N.py` (re-run as evaluations land) | `out/distance_margin_by_N.csv` | D25 |
| fig73_why_margin_rises.png | `scratch/fig_why_margin_rises.py` | `out/banktrials_dAB.csv` | D26 |
| fig75_one_axis_controlled.png | `scratch/analyze_all_categories.py` (first cut, 8 models; re-run when round 3 completes) | `out/all_categories_long.csv` | D28 |
| fig76_round3_moving_training.png | `scratch/analyze_round3_simple.py` | `out/all_categories_long.csv` (round-3 models on `banktrials_all.csv`) | D29 |
| fig77_raw_margin_vs_distance.png | `scratch/fig_raw_margin_vs_distance.py` | `out/all_categories_long.csv` | D30 |
| fig78_absolute_vs_relative.png | `scratch/fig_absolute_vs_relative.py` | `out/all_categories_long.csv` | D31 |
| fig79_one_trial_many_models.png | `scratch/fig_one_trial_many_models.py` | `out/all_categories_long.csv` | D32 |
| step1_manuscript_metric.png | `scratch/fig_story_steps.py` | `L1norm_vs_distshift/trials_vit_base_patch16_224.dino.csv` | story figure, step 1 |
| step2_fix_the_form.png | `scratch/fig_story_steps.py` | `out/shift2d_shapegen.csv`, `out/blindshift_shapegen.csv` | story figure, step 2 |
| step4_across_not_within.png | `scratch/fig_story_steps.py` | `out/coverage_shapenet_voxel16_percat.csv` | story figure, step 4 |
| fig80_distance_search.png | `scratch/distance_search_fast.py` → `out/distance_search.csv` | round-3 margins × 11 feature sets × 8 comparisons | D36 |
| fig81_gradient.png | `scratch/fig_gradient.py` | `eval_bank_all/chair_c*_n25{,_fullft}`, bbox + DINOv2 banks | D38 |
