# Checkpoint 3 — The within-category question, the search, and the pessimism

*Written from the vantage point of the hill-climb results.*

## The second question

The rank curve is mostly a step: the on-category model sits far above the other eleven.
So: *holding category membership fixed, does graded geometric distance still predict the
margin?* That is the **within-category row** — only the model trained on each trial's
own category, one point per trial, n = 706.

Answer: barely. voxel16 r = −0.097 (p = 0.01), control −0.004. d57 −0.026. The accuracy
version −0.031. Binned r *degrades* as bins get finer (−0.56 at 10 → −0.30 at 70), the
opposite of a real graded relationship (`fig41`).

## The search

`scratch/hillclimb.py`, SLURM job 8249529: 1,895 candidates (8 representations × ~20
estimator/k variants × 12 trial-level metrics), objective = within-category margin
advantage, 50 × 5-fold CV over trials, permutation null on the max statistic.

| | held-out mean \|r\| |
|---|---|
| searched winner | 0.113 |
| fixed baseline `voxel16 \| knn_mean_k50 \| both_mean` | **0.133** |

Gain −0.020 [−0.072, +0.013]; search wins 28 % of repeats. In-sample the max reached 0.184
with corrected p = 0.001 — correct and irrelevant. **No single model-free metric makes the
within-category relationship strong; ≈0.13 is the ceiling.**

For human RT the search *did* generalise (0.368 vs 0.318, 100 % of repeats,
`structure | pca_recon | minus_dTT`) — later set aside: MOCHI's ShapeNet trials were
selected adversarially, so trial-geometry ↔ human-performance relationships may be
properties of trial construction.

## Category-centring

Within-category r goes −0.097 → **+0.014** when each category's mean is removed. In that
row "which model" and "which category" are one variable. The effect is carried by ~12
category means, not 706 trials.

## Where the pessimism came from

Every figure after the within-trial design was about this second question, and every one
came back weak. It began to feel as though the whole thing had not worked.

## Revisions to Checkpoint 2

None to the within-trial result. What changed is the *scope* claimed: the within-trial
effect is mostly on-category vs off-category (coverage), not a graded distance law.

## What we believed

That the metric was weak. This was half right (the graded within-category claim is not
supported) and half a conflation (the within-trial claim was already strong). Checkpoint 4
separates them.
