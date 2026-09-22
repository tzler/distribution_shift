# experiments/ — the interventions

Everything needed to rebuild the training conditions, submit the fine-tunes and evaluate
them. The analysis of their outputs lives in `../scratch/`; nothing here analyses anything.

```
scripts/    condition builders, evaluators, SLURM submitters
designs/    what each condition contains (JSON, written by the builders)
trials/     the oddity trials we built and scored models on (CSV, MOCHI's format)
patches/    our changes to the collaborator's training / evaluation code
logs/       the job lists for each batch, as submitted
```

## What was run

| batch | what | outcome |
|---|---|---|
| 1 | knockout: remove the 10 training objects nearest a group of test objects (chair, airplane, table), plus random-removal controls and eight random 100-chair models; scored on MOCHI | null — the dose is invisible to this pipeline ([D18](../REASONING.md#D18)) |
| 2 | three chair sub-populations × N ∈ {all, 50, 25}; 882 trials from held-out chairs | the effect, and the distance predicts it ([D21](../REASONING.md#D21), [D22](../REASONING.md#D22)) |
| 3 | the same design in all twelve categories at N = 25; 11,634 trials | reproduces ([D29](../REASONING.md#D29)) |
| ladder | the three chair models at three fine-tuning strengths, plus one model from scratch | "relative" is a property of the regime ([D34](../REASONING.md#D34), [D35](../REASONING.md#D35)) |

## Pipeline

```bash
# 1 · design: cluster each category, split in half, list the training objects, build trials
python scripts/make_cluster_exp_all.py          # → designs/design_clusters_all.json, trials/banktrials_all.csv
python scripts/make_train_trials.py             # → trials/traintrials_all.csv  (trials from the TRAINING objects)

# 2 · materialise a condition: symlink its objects, filter the similarity table
python scripts/make_subset.py <cat> <cond>      # data/<cat>_<cond>/{white,black,random}/
python scripts/make_sims.py   <cat>_<cond>      # data/<cat>_<cond>/sim/

# 3 · train + evaluate (SLURM; one job per condition, evaluation chained)
sbatch --job-name=<cat>_<cond> scripts/run_train_all.sbatch <cat> <cond>

# 4 · evaluate an existing checkpoint on another trial set
bash scripts/eval_bank_all.sh <cat> <cond>      # → eval_bank_all/<cat>_<cond>/ood_analysis_results.csv
bash scripts/eval_train.sh    <cat> <cond>      # → eval_train/…   (the training-object anchor)
```

Paths inside the scripts are absolute to our share
(`/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout`); change the `K=` line
at the top of each to relocate.

## The patched pipeline

Training and evaluation use a collaborator's repository, which is read-only for us and is
not redistributed here. We ran patched copies; `patches/` holds the diffs:

| patch | against | what it changes |
|---|---|---|
| `train_knockout.py.patch` | `hida-tune/train.py` | writable log/augmentation directories, absolute dataset paths, the test-split CSV key |
| `ood_eval_knockout.py.patch` | `hida-tune/evaluation/ood_distance_analysis.py` | trial CSV and image root as arguments; backbone selectable for the ViT-small runs |

To rebuild them:

```bash
cp <hida-tune>/train.py train_knockout.py && patch train_knockout.py < patches/train_knockout.py.patch
cp <hida-tune>/evaluation/ood_distance_analysis.py ood_eval_knockout.py && patch ood_eval_knockout.py < patches/ood_eval_knockout.py.patch
```

## Things that cost us time

- The pipeline draws a **fixed number of image triplets per epoch** whatever the bank size,
  so removing 10 % of a 2,000-object bank changes almost nothing the model sees. Check that a
  manipulation changes the sampling, not just the directory ([D18](../REASONING.md#D18)).
- **Replicate the reference before comparing to it.** Our patched pipeline reaches a mean
  margin of 0.155 where the collaborator's reached 0.241 on the same chairs; nothing we train
  is compared to their models.
- Evaluation scripts glob the run directory as `vit_*`; a ViT-small run is not `vit_large*`.
- MIG partitions need `--qos=normal`; `dgx-b200` needs 28 CPUs per GPU.
