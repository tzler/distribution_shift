# data/ — result tables

Enough to rebuild the main figures without the cluster. Bulk inputs (rendered images,
descriptor banks, per-model evaluation outputs) stay on the lab share; these are what the
analysis scripts consume.

| file | rows | what |
|---|---|---|
| `all_categories_long.csv.gz` | 465,360 | the core table: every (trial × model) pair from round 3 and the ladder. `dist` = distance from the trial's objects to that model's 25 training objects (16³ voxels, mean of the 10 nearest, cosine); `pair` = distance between the trial's own two objects; `ft`/`pre` = the fine-tuned and pretrained oddity margins; `test_cluster`/`train_cluster` name the sub-population |
| `distance_search.csv` | 69 | the measure search: descriptor × comparison rule, scored on fitted and held-out trials, for all models / own-category / other-category |
| `transfer_mochi.csv` | 12 | the same measures re-scored on MOCHI images (unseen, different renderer) |
| `cluster_summary.csv` | 3 | the sub-category matrix by training-set size |
| `margin_vs_dose.csv` | 25 | every chair model by training-set size, on both trial sets |
| `shift_vs_setsize.csv` | 70 | the shift estimate as a function of training-set size (no training involved) |
| `knockout_summary.csv`, `knockin_random_long.csv`, `random100_bank_long.csv` | — | batch 1 |
| `banktrials_dAB.csv` | 882 | per-trial pair distance and margins, for the test-set confound |
| `coverage_*.csv`, `blindshift_*.csv` | 8,472 | the original twelve category models: coverage, distance and margins per trial × category |

Margins are on the same scale everywhere: mean similarity of the two matched images minus
the odd one's, in the model's feature space. A trial's `pre` is identical across models of
the same architecture — that is the control the whole design rests on.
