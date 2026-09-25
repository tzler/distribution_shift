# Distribution shift and the oddity margin

Does a model's **oddity margin** — how confidently it picks the odd object out of three —
track how far the test objects are from what the model was trained on? And can "how far"
be measured without a neural network in the ruler?

This repository is the project: what we believe now, how that changed, the evidence for
each step, and the code to redo it.

## Read it in this order

| | |
|---|---|
| [`STATE.md`](STATE.md) | what we believe **now** — a seven-step walk, ~5 minutes |
| [`states/`](states/README.md) | STATE as it stood at each earlier turn (nine snapshots) — the story in order |
| [`REASONING.md`](REASONING.md) | why: every decision and result, newest first, `[Dxx]` |
| [`evidence/`](evidence/README.md) | every cited figure, with the script and data that made it |
| [`RESOURCES.md`](RESOURCES.md) | what was available at each point (compute, data, models) |
| [`agent/`](agent/README.md) | notes for the next agent: what is running, what to check, what broke before |

The format is [`tzler/state`](https://github.com/tzler/state); a browsable version of the
nine states is [`tzler/state_distributionshift`](https://github.com/tzler/state_distributionshift).

## The finding, in six lines

1. The manuscript's shift estimate can never be smaller than half the difference between the
   trial's two objects, and is read off the same network whose margin it explains — so it
   measures the trial and the network, not the training set.
2. Fix the form (average over the trial's images; never difference the two objects) and the
   space (measure on the objects' 3-D shape, no network).
3. With twelve category-trained models scored on the same trial, the margin follows distance
   to the training set — the pretrained model flat because it is the same model on every point.
4. Inside one category, with one training set per category, the question is not identifiable.
   So we trained new models on chosen 25-object subsets: the effect is there, and the distance
   predicts which model wins a trial without being told the subsets.
5. The margin is a **relative** measure: 92 % of it is where pretraining left the trial;
   distance moves it from there. Overwrite more of the prior and that share falls (85 → 45 %).
6. Searching 69 distance measures against those margins: within a category everything
   reasonable ties (a seven-number bounding box included); across categories only a frozen
   pretrained network's features put different training sets on one scale.

## Reproduce

[`REPRODUCE.md`](REPRODUCE.md) maps every figure to the script and the table that made it.
Analysis runs on CPU from the tables in [`data/`](data/README.md) — no GPU, no cluster:

```bash
python scratch/analyze_round3_simple.py     # the main result, from data/all_categories_long.csv.gz
python scratch/fig_absolute_vs_relative.py  # relative, not absolute
python scratch/distance_search_fast.py      # the 69-measure search (needs the descriptor banks, see below)
```

Re-running the **experiments** (fine-tuning models, evaluating them) needs the cluster, the
render bank and the collaborator's training pipeline: see [`experiments/`](experiments/README.md).

## Layout

```
STATE.md        what we believe now          ┐
REASONING.md    why, entry by entry          │ the trace — read these
states/         how STATE looked before      │
evidence/       every cited figure           │
RESOURCES.md    what we had at each point    │
agent/          notes for the next agent     ┘

README.md       start here
REPRODUCE.md    figure → script → data, for all 36 cited figures
data/           the tables the analysis consumes (runs on CPU, no cluster)
scratch/        the analysis — one script per figure or battery; `from _repo import G`
lib/            shared modules the analysis imports (descriptors, estimators)
experiments/    the interventions: designs, trial sets, SLURM scripts, patches
scripts/        state_lint.py + the pre-commit installer (documentation hygiene)
docs/           earlier write-ups, kept for the record
legacy/         first-phase code and superseded passes, kept for provenance
```

## What is not here

- **Rendered images and descriptor banks** (~100 GB): on the lab share at
  `naturalistic-navig/Dist-shift-data/{shapenet_rendered,geometric_shift/bank}`.
- **Model checkpoints** (62 fine-tunes): `Dist-shift-data/knockout/logs/<condition>/`.
- **The training pipeline** — a collaborator's repository, read-only for us. We ran patched
  copies; the patches are in `experiments/patches/`.
- **Figures at full resolution**: `out/figures/` on the share; `evidence/` holds the cited ones.

## Status

Active. Open items are in `STATE.md` → Next steps, and the immediate ones in
`agent/INFLIGHT.md`. Costs to date: ~100 GPU-hours, about $450 unsubsidised / $100
subsidised (`background/compute_ledger.csv`).
