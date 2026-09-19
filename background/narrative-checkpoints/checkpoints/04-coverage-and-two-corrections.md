# Checkpoint 4 — Coverage, and two things we had wrong

*Written 19 September 2026, after the coverage estimator and the Act-2 re-check.*

## Coverage, not distance

Every estimator so far asked *how far* the nearest training objects are. Coverage asks
*how much* training mass is near the object: count the category's training objects within
a fixed cosine radius ε of each image, take −log(1 + count), average over the trial.
The difference is the ceiling: "no training objects nearby" is the most shift there is,
so object atypicality saturates instead of spreading — and stops leaking into the pooled
row. (`scratch/coverage_sweep.py`, `fig50–56`)

| voxel16 | knn_mean | coverage (ε = 0.10–0.15) |
|---|---|---|
| within-trial r | −0.329 | **−0.424** |
| pooled r / base-DINOv2 control | −0.223 / **−0.137** | −0.268 / **−0.030** |
| two-way centred | −0.350 | −0.436 |

The pooled control passing is the qualitative gain — the first single-model pooled
measure that is clean. Holds for ε 0.03–0.20 on d57 and voxel16; split-half (ε chosen on
one half, scored on the other) wins 20/20, held-out −0.414 vs −0.328, control −0.041 vs
−0.143. Soft kernel-mass versions behave like knn_mean: the hard cut-off does the work.

## Correction 1: the Checkpoint-1 shapegen headline is withdrawn

The Eq. 1–3 shift correlates with d(A,B) at **r = +0.993** on shapegen. It *is* object
dissimilarity, and d(A,B) alone predicts the encoder proxy better (0.79) than the "shift"
did (0.74). With the oddity-blind metric the shapegen relationship goes to |r| = 0.12.
The incumbent's correlation with d(A,B) is 0.07 — it was never riding this confound; we
beat it by being more entangled. On shapenet a modest blind relationship survives (0.43,
incumbent 0.55). (`fig49`)

Spelled out (`fig61`): for any metric d(A,B) ≤ d(A,C) + d(C,B), so the shift is
½·d(A,B) + excess. Every distance obeys it (L1 raw / unit-norm / cosine: floor holds
100 %, r with d(A,B) 0.73 / 0.75 / 0.78, cosine worst). Decomposed on DINO ViT-B: the
floor predicts the pretrained margin at +0.57 (wrong sign), the excess at −0.17 (right
sign), sum +0.32. Changing the *form* (each object to its own neighbour) takes r with
d(A,B) to 0.66 in encoder space; leaving the encoder's space finishes it (+0.45 / −0.36).

## Correction 2: the on-category coverage curve is not a within-category result

Coverage's on-category row looked like the graded law: binned r −0.78 (`fig54`). Its bins
are sorted by category: the ">100 neighbours" bins are airplane/bench/car/lamp/telephone/
watercraft (homogeneous shapes), the "0" bin is chair/table/sofa/cabinet/display/
loudspeaker (diverse shapes, biggest banks). Category-centre and it is gone: r = +0.013,
p = 0.72; inside each category, mean r = +0.04, four of twelve negative. Twelve category
points give r = +0.30, p = 0.34, with chair the clear counterexample (lowest coverage,
highest margin). (`fig57`) Seven coverage variants including category-calibrated
percentile all give category-centred r ≈ 0. **The design ties training set to category
one-to-one in that row; no estimator can pull them apart.**

## The original metric, on our data

`trial_distance_(L1_not_normalized)` from each category's results file: pooled r with the
fine-tuned margin −0.01, with the *pretrained* margin **+0.38**; on-category +0.33 vs
**+0.61**. It predicts the model that never saw the training set better than the one that
did, with the sign backwards, and behaves only after two-way centring (−0.448). (`fig56`,
`fig60`)

## Revisions to earlier checkpoints

- Checkpoint 1: the shapegen encoder comparison is withdrawn (Correction 1).
- Checkpoint 3: the within-category search was already exhaustive; coverage does not
  change its answer (Correction 2). The pessimism was half right.
- README §5.1 claimed the rotation-invariant descriptor was null: true only in the
  within-category row; within-trial d57 gives −0.270, 82 % of slopes negative.

## What we believe now

Q1 — does the margin respond to the training distribution, measured model-free, with a
control that cannot be gamed? **Yes**, and coverage is the best ruler. Q2 — beyond category
membership, does graded distance within a category's training set predict the margin?
**Not resolvable with these data**, not because of the metric but because of the design.
