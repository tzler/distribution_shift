# INFLIGHT — live jobs and how to resume
*Machine-maintained by `scratch/inflight.py`; last refreshed 2026-09-19 14:12. Paths relative to `../knockout/`.*

## Queue right now (23 jobs)
| job | name | kind | partition | state | elapsed | log | expected output |
|---|---|---|---|---|---|---|---|
| 8520577 | chair_random_100_0 | fine-tune | b200-mig45 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_0_8520577.log` | `eval/chair_random_100_0/ood_analysis_results.csv` |
| 8521030 | airplane_g1_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_airplane_g1_k10_8521030.log` | `eval/airplane_g1_k10/ood_analysis_results.csv` |
| 8521029 | chair_random_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_k10_8521029.log` | `eval/chair_random_k10/ood_analysis_results.csv` |
| 8521028 | chair_g4_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_g4_k10_8521028.log` | `eval/chair_g4_k10/ood_analysis_results.csv` |
| 8521027 | chair_g3_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_g3_k10_8521027.log` | `eval/chair_g3_k10/ood_analysis_results.csv` |
| 8521026 | chair_g2_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_g2_k10_8521026.log` | `eval/chair_g2_k10/ood_analysis_results.csv` |
| 8521035 | table_g1_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_table_g1_k10_8521035.log` | `eval/table_g1_k10/ood_analysis_results.csv` |
| 8521034 | airplane_random_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_airplane_random_k10_8521034.log` | `eval/airplane_random_k10/ood_analysis_results.csv` |
| 8521033 | airplane_g4_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_airplane_g4_k10_8521033.log` | `eval/airplane_g4_k10/ood_analysis_results.csv` |
| 8521032 | airplane_g3_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_airplane_g3_k10_8521032.log` | `eval/airplane_g3_k10/ood_analysis_results.csv` |
| 8521031 | airplane_g2_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_airplane_g2_k10_8521031.log` | `eval/airplane_g2_k10/ood_analysis_results.csv` |
| 8521041 | chair_random_100_2 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_2_8521041.log` | `eval/chair_random_100_2/ood_analysis_results.csv` |
| 8521040 | chair_random_100_1 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_1_8521040.log` | `eval/chair_random_100_1/ood_analysis_results.csv` |
| 8521039 | table_random_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_table_random_k10_8521039.log` | `eval/table_random_k10/ood_analysis_results.csv` |
| 8521038 | table_g4_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_table_g4_k10_8521038.log` | `eval/table_g4_k10/ood_analysis_results.csv` |
| 8521037 | table_g3_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_table_g3_k10_8521037.log` | `eval/table_g3_k10/ood_analysis_results.csv` |
| 8521036 | table_g2_k10 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_table_g2_k10_8521036.log` | `eval/table_g2_k10/ood_analysis_results.csv` |
| 8521046 | chair_random_100_7 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_7_8521046.log` | `eval/chair_random_100_7/ood_analysis_results.csv` |
| 8521045 | chair_random_100_6 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_6_8521045.log` | `eval/chair_random_100_6/ood_analysis_results.csv` |
| 8521044 | chair_random_100_5 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_5_8521044.log` | `eval/chair_random_100_5/ood_analysis_results.csv` |
| 8521043 | chair_random_100_4 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_4_8521043.log` | `eval/chair_random_100_4/ood_analysis_results.csv` |
| 8521042 | chair_random_100_3 | fine-tune | dgx-b200 | PENDING (Priority) | 0:00  | `logs/train_chair_random_100_3_8521042.log` | `eval/chair_random_100_3/ood_analysis_results.csv` |
| 8519397 | enc_extract | extract | dgx-b200 | RUNNING  | 2:16:15  | `logs/extract_8519397.log` | `eval/encoder_features/*.npz` |

Completed evaluations: chair_g1_k10. (`chair_g1_k10`'s csv from the eval test is an epoch-10 checkpoint; the job's own chained eval overwrites it.)

## Batches
- **Batch 1** (submitted 19 Sep ~13:40, dgx-b200, 4.5 h limit): pilot `chair_g1_k10` + 21 jobs in `logs/batch1_jobs.txt` — knockout k = 10 for every group, `random_k10` per category, `chair_random_100_1..7`. Each trains 30 epochs (~3 min/epoch) then runs `scripts/eval_one.sh` → `eval/<cat>_<cond>/ood_analysis_results.csv`.
- **mig45 timing run** `chair_random_100_0` (`b200-mig45`, `--qos=normal`, 12 h limit): compare epoch time with ~3 min; ≤ 2.5× slower ⇒ mig45 is cheaper per run ⇒ batch 2 goes there.
- **Batch 2** (not submitted): `chair_target_100_<trial>` × 6, `airplane_cross_100_<trial>` × 6; subsets and sims built. `sbatch --job-name=<cat>_<cond> scripts/run_train[_mig45].sbatch <cat> <cond>`.
- **Batch 3** (held): `<cat>_g<1-4>_k50`, `<cat>_random_k50` (15); decide from batch-1 results.
- **Track A** `enc_extract` → `eval/encoder_features/{pretrained,ft_chair,ft_airplane,ft_table}.npz`; then write and run `scratch/encoder_space_battery.py` (within-category, category-centred, in each model's space).

## How to check / resume
```
squeue -u bonnen -o "%i %j %T %M %P %R"
python /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/scratch/inflight.py                    # refresh this file
ls /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/eval/*/ood_analysis_results.csv
bash /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/scripts/eval_one.sh <cat> <cond>          # re-run only the evaluation (GPU job)
python /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/scratch/compute_ledger.py               # refresh the cost ledger
```
A job that died after training leaves checkpoints in `logs/<cat>_<cond>/*/checkpoints/` — re-run only the eval. A job that died in training: just resubmit (the csv dir is keyed by a random exp tag; nothing to clean).

## Analysis to write while jobs run
`scratch/analyze_knockout.py` — per category, per trial: x = knn_mean / coverage of the trial to each model's actual training set (recompute from `design.json` removals against `bank_voxel16`); y = fine-tuned margin from each `eval/` csv; within-trial regression of Δmargin on Δx; own-group vs other-group vs size-matched-random contrast.
`scratch/analyze_knockin.py` — Δmargin from pretrained vs x over the 8 random subsets, per trial; candidate-metric comparison on those runs; targeted vs random vs cross for the 6 target trials; the single-trial figure.
