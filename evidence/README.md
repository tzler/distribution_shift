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
| fig57_oncat_anatomy.png | `scratch/fig_oncat_anatomy.py` | `out/coverage_shapenet_voxel16_percat.csv` | D07 |
| fig58_viewdepth_ladder.png | `scratch/fig_coverage_recipe.py` | `out/viewdepth_battery_*.csv` (from `scratch/viewdepth_pipeline.py`, SLURM 8515465/8515516/8515529/8515554/8515600/8515622) | D08 |
| fig60_why_wrong.png | `scratch/fig_why_wrong.py` | `../L1norm_vs_distshift/trials_*.csv`; three-metric numbers from `scratch/fig_three_metrics.py` | D06 |
| manuscript_fig1.png | rendered from `Human-3D-generalization-copy/paper-to-follow-*/distributionshift_neurips2026-3.pdf` p.4 (pymupdf), cropped | the submitted manuscript | State 0 |
| fig62_encoder_space.png | `scratch/fig_encoder_circularity.py` | `knockout/eval/encoder_space_battery.csv` (from `scratch/encoder_space_battery.py`, SLURM 8522238; features from job 8519397) | D15, State 2 |
| fig61_triangle.png | `scratch/fig_triangle.py` | `../L1norm_vs_distshift/trials_vit_base_patch16_224.dino.csv` | D06 |

Numbers quoted in STATE that have no figure: hill-climb (`out/hillclimb_candidates.npz`,
`scratch/hc_repeated.py`); coverage variants (`scratch/coverage_variants.py`);
level-3 residualised test (session record 2026-09-19; script to be added).
