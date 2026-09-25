# Model-free geometric distribution shift — approach and findings

Working notes for the ICLR resubmission of *Human perception under distribution shift*.
Everything here is computed in `/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/`.

---

## 1. Motivation

The manuscript's empirical distribution-shift metric (§3.5) is model-based twice over:
the trial-to-training distance is an L1 nearest-neighbour search in a **pretrained
DINOv2 feature space**, and the test-time proxy `ŝ = 1 + εm` is that same encoder's
oddity margin. The in-repo audit (`Dist-shift-data/L1norm_vs_distshift/README.md`)
shows this is a substantive problem, not a stylistic one: `r(shift, ‖φ‖₁)` = +0.949 for
ResNet-50 and +0.893 for DeiT-III-B, a 0.4% subsample of ImageNet reproduces the full
search at r ≥ 0.96, and for DeiT the shift adds nothing over `‖φ‖₁` in a nested LR
(χ² = 0.3, p = 0.59).

MOCHI's shapenet and shapegen trials are rendered from known 3D assets, so stimulus
geometry is *known* rather than inferred. The goal was a distribution-shift measure with
**zero learned parameters**, evaluated head-to-head against the incumbent on the same
trial table.

---

## 2. Data

| item | count |
|---|---|
| MOCHI trials total | 2019 (barense 140, majaj/hvm 464, shapegen 548, shapenet 867) |
| ShapeNet trials with a single-category object set | 706 |
| Fine-tuned models | 12, one per ShapeNet category (+ an `all_categories` model) |
| Trial × model observations | 8,472 (706 × 12) |
| Training bank (MOCHI objects excluded) | 20,885 objects |
| MOCHI test objects | 897 |

Behavioural and model columns come from
`Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses/<category>/ood_analysis_results.csv`
(2019 rows each): `fine_tuned_oddity_margin`, `pretrained_oddity_margin`,
`fine_tuned_correct`, `pretrained_correct`, `human_accuracy`, `human_rt`.

**Design.** Every trial is run through all 12 category-models. The trial's images, its
objects, and its difficulty are identical across those 12 — only the training set
changes. This is what separates distance-to-training from trial difficulty. Note that
`d(A,B)` has exactly zero within-trial variance (verified: sd = 1.2e-13).

---

## 3. Object representations (all model-free)

Built from ShapeNet's `model_normalized.solid.binvox` (128³ solid voxelization). No mesh
library, no renderer, no network.

| name | dims | pose | description |
|---|---|---|---|
| `d57` | 57 | **invariant** | 32 D2 shape-distribution bins + 16 shell bins + 9 scalars (elongation, flatness, compactness, convexity, normalised volume, radial mean/sd, D2 mean/sd) |
| `voxel8` | 512 | sensitive | raw 8³ occupancy grid |
| `voxel16` | 4,096 | sensitive | raw 16³ occupancy grid |
| `voxel32` | 32,768 | sensitive | raw 32³ occupancy grid |
| `bbox` | 7 | invariant | extents, aspect ratios, volume, vertex count from `model_normalized.json` |
| `multiview` | 65 | sensitive | 13 viewing directions × 5 silhouette statistics |
| `volatility` | 6 | sensitive | sd across views of each silhouette statistic + mean pairwise IoU |
| `structure` | 7 | — | connected components, hole/Euler proxy, mirror symmetry per canonical axis |

Also built but not used in the final analyses: 2D silhouette descriptors (20-d) and a
32×32 silhouette map over 311,332 individual training renders.

---

## 4. The estimator

Oddity-blind, hub-robust, norm-free (`blind_shift.py`):

```
shift(t) = (1/|t|) Σ_{x∈t}  (1/k) Σ_{i∈kNN_k(x)}  ( 1 − ĝ(x) · Ĉ_i )        k = 50
```

- **Oddity-blind.** Average a per-image quantity over *all* the trial's images; never
  difference them. The shared-neighbour form `min_C ½[d(A,C)+d(B,C)]` is bounded below
  by `½·d(A,B)` — the oddity task's own decision variable — so trial difficulty leaks in.
  Averaging removes that bound structurally.
- **Hub-robust.** Mean over k nearest training objects, not a single argmin.
- **Norm-free.** Cosine distance on L2-normalised descriptors, so it cannot simply track
  descriptor magnitude (the failure mode the audit identifies).

Descriptors are z-scored against the bank, then L2-normalised. A geometric ruler cannot
be reshaped by training, so it is neutral for every model and every training stage.

**Controls used throughout:**
- *base DINOv2* (`pretrained_oddity_margin`) — never saw any fine-tuning set, so must
  not be predicted by a valid shift measure;
- *random-init* — 11 architectures with `__random_init` twins;
- *permutation* — on the max statistic, for the search.

---

## 5. Results

### 5.1 Pose sensitivity, not resolution, is what matters

Within-category (n = 706, one point per trial), against fine-tuned margin:

| representation | r | p | partial (ft \| pretrained) | pretrained control |
|---|---|---|---|---|
| `d57` (rotation-invariant) | −0.026 | 0.50 | −0.039 | +0.011 |
| `voxel8` | −0.096 | 0.011 | −0.108 | −0.011 |
| `voxel16` | −0.097 | 0.0096 | −0.116 | −0.004 |
| `voxel32` | −0.097 | 0.0099 | −0.118 | −0.000 |

The rotation-invariant descriptor is null **in this row only**; in the within-trial design
(§5.5 of the packet) `d57` gives r = −0.270 with 82 % of slopes negative, nearly matching the
voxels. All three voxel resolutions agree to within 0.001, so resolution is irrelevant above 8³.

### 5.2 Only the within-category row survives its control

Using `voxel16`, y = fine-tuned margin:

| row | n | r | binned r (50 bins) | **base DINOv2 control** |
|---|---|---|---|---|
| within category | 706 | −0.097 | −0.359 | **−0.004, p = 0.92** |
| across category | 7,766 | −0.116 | −0.731 | **−0.159, p = 6e−45** |
| all trials | 8,472 | −0.223 | −0.771 | **−0.137, p = 1e−36** |

The across-category and pooled rows have larger correlations, and both fail the control:
base DINOv2, which never saw any fine-tuning set, tracks geometric distance there at
least as strongly as the fine-tuned models do. Those rows measure object atypicality, not
training exposure. Group means (voxel16): on-category distance 0.173 / margin 0.175;
off-category 0.573 / 0.060.

### 5.3 Margin advantage

y = `fine_tuned_oddity_margin − pretrained_oddity_margin`, `voxel16`:

| row | r | p |
|---|---|---|
| within category | **−0.115** | 0.0023 |
| across category | +0.045 | 6e−05 |
| all trials | −0.121 | 3e−29 |

The within-category value is the cleanest causal statement available: greater geometric
distance from the training set → less benefit from having trained on that category. The
across-category value is positive and its binned form is non-monotone (a hump peaking
near x ≈ 0.5).

### 5.4 Humans move in the opposite direction from models

Against on-category geometric distance (n = 706, `voxel16`):

| DV | r | p |
|---|---|---|
| human accuracy | **+0.199** | 9.3e−08 |
| human RT | **−0.305** | 1.1e−16 |
| human − model accuracy | +0.139 | 2.1e−04 |

Farther from the training set → humans are *more* accurate and *faster*, while models get
worse. Note these are trial-constant across the 12 models, so their honest home is the
within-category row (one point per trial); against the mean distance over all 12 models
they are null (+0.015 and −0.067).

### 5.5 Hill climb over the metric space

Search space: 8 representations × ~20 estimator/k variants (`nn_min`, `knn_mean`,
`knn_kth`, `centroid`, `mahalanobis`, `pca_recon`, `energy_lse`, `local_norm`,
`rank_pct`; k ∈ {1,10,50,200}) × 12 trial-level metrics (`both_mean`, `oddity_only`,
`closer_one`, `over_dAB`, `minus_dTT`, …) = **1,895 candidates** on 706 trials. Single
combinations only — no concatenation, no fitted weights. SLURM job 8249529,
`genoa-std-mem`, 64 CPUs, 3 min wall.

Validation: 50 repeats × 5-fold CV **over trials** (select on 4 folds, score on the
held-out fold), plus a 2,000-shuffle permutation null on the *max* statistic.

**Objective — within-category margin advantage: the search loses.**

| | mean \|held-out r\| |
|---|---|
| searched winner | 0.113 |
| fixed baseline `voxel16 \| knn_mean_k50 \| both_mean` | **0.133** |
| gain | **−0.020**, 95% CI [−0.072, +0.013]; searching wins 28% of repeats; t = −5.95, p < 1e−4 |

In-sample the best candidate reached −0.184 vs the baseline's −0.133 (a 38% apparent
gain) with permutation-corrected p = 0.0010. That p-value is correct and irrelevant: it
says the *max* of 1,895 candidates exceeds chance, not that the *selected* candidate
generalises. Out of sample it does not. Top picks were unstable (best taken in 22% of
selections).

**Secondary — human RT: the search wins.**

| | mean \|held-out r\| |
|---|---|
| searched winner `structure \| pca_recon \| minus_dTT` | **0.368** |
| fixed baseline | 0.318 |
| gain | **+0.050**, 95% CI [+0.019, +0.074]; wins 100% of 50 repeats; t = +22.6 |

Picked in 66% of all 250 selections; control r = −0.026 (clean). The winner is a
**7-dimensional topology-and-symmetry descriptor**, scored as distance to the training
*manifold* (10-component PCA reconstruction error) minus the training set's own internal
spread. Caveat: runners-up `voxel16 | energy_lse | over_dTT` (control −0.102) and
`multiview | energy_lse | over_dTT` (−0.137) fail the control; the leaderboard was
reported, not filtered.

### 5.6 Centring the x axis

y left raw throughout; only x is transformed. Point r / binned r (50 bins), `voxel16`,
y = fine-tuned margin:

| row | x raw | trial-centred | category-centred | both |
|---|---|---|---|---|
| within category | −0.097 / −0.359 | *undefined* | **+0.014 / +0.034** | *undefined* |
| across category | −0.116 / −0.731 | −0.049 / −0.360 | −0.116 / −0.659 | −0.052 / −0.385 |
| all trials | −0.223 / −0.771 | −0.214 / −0.868 | −0.230 / −0.791 | −0.224 / −0.852 |

**Trial-centring is undefined for the within-category row**: it holds exactly one
observation per trial, so subtracting each trial's mean x sends every point to exactly
zero (verified sd = 0.000e+00).

**Category-centring annihilates the within-category effect**: −0.097 → +0.014, p = 0.70.

For margin advantage, trial-centring *strengthens* the all-trials row: −0.121 → −0.252
(binned −0.735).

Algebraic note: within a trial the centred x sums to zero, so its covariance with any
trial-constant is zero and the **slope is identical** whether or not y is also centred.
Only r changes. Hence y is never centred in these figures.

### 5.6b Entanglement check on the Act-2 encoder comparison (added 19 Sep)

`out/fig_margin_vs_geometric_*.png` use the Eq. 1–3 shift, r(shift, d_AB) = +0.993
(shapegen) / +0.996 (shapenet). With the oddity-blind `blind_k50` (ALL bank), dinov2-large
30-bin |r|: shapegen 0.74 → **0.12**; shapenet 0.44 → 0.43 (incumbent 0.55 on both). The
shapegen "beats the incumbent" claim is withdrawn. `out/figures/fig49_entanglement.png`.

### 5.6c Coverage estimator (added 19 Sep)

`coverage = −log(1 + #bank_C objects within cosine ε)`, mean over trial images. ε = 0.10–0.15.
voxel16: within-trial r −0.42 (knn_mean −0.33), 85 % negative, pooled −0.26 with control
−0.02 (knn_mean −0.22 / −0.14), partial beyond on-category binary −0.12 (−0.07). Same on
d57. Split-half 20/20. Soft kernel mass does not reproduce it. `scratch/coverage_sweep.py`,
`out/coverage_sweep_*.csv`, `out/figures/fig50_coverage.png`.

### 5.7 Negative results worth keeping

- **Image-based reference failed its control.** Using the 311,332 individual training
  renders with 32×32 silhouette maps gave r = +0.220 against the *pretrained* margin vs
  +0.124 against the fine-tuned one — the control is stronger than the effect. Cause is
  most likely pipeline mismatch (MOCHI renders at 1000², pyrender bank at 224²).
- **88% of "nearest training set" is the object's own category**, so the pooled effect is
  largely a category-match contrast: binary own-category match gives r = +0.474 vs
  −0.319 for the graded distance.
- **Could not reproduce the published shapenet r = .91 for Fig. 1**; got +0.341. Unresolved.

### 5.8 Random-init control (manuscript §5)

Across 11 architectures with `__random_init` twins: trained margins predict human
accuracy at −0.295 and human RT at +0.323; random-init twins give −0.014 and −0.006.
§5 passes its control cleanly.

---

## 6. Files

```
geometric_shift/
  geom3d.py                 binvox parser + 57-d rotation-invariant descriptor
  geom2d.py                 silhouette extraction + 20-d descriptor
  variant_descriptors.py    voxel8/16/32 + bbox banks
  pose_features.py          multiview (65d), volatility (6d), structure (7d)
  build_bank_3d.py          multiprocessing bank extraction
  blind_shift.py            the oddity-blind estimator (§4 above)
  compute_shift.py          Eqs 1–3 with φ swapped
  metric_space.py           12 trial-level metrics from dA, dB, dAB, dTT
  estimators.py             9 estimators
  image_reference.py        311,332-image silhouette reference (failed its control)
  verify.py                 5 sanity checks, all pass
  run_hillclimb.sbatch      SLURM driver for the search
  scratch/
    hillclimb.py            candidate matrix builder (one rep per process)
    merge_shards.py         concatenates shards, asserts trial-order identity
    hillclimb_eval.py       CV + permutation null + leaderboard
    hc_compare.py           corrected CV: searched vs fixed baseline
    hc_repeated.py          50 × 5-fold repeated CV
    fig_grid5.py            3×5 grid with base-DINOv2 control column
    fig_centred_grid.py     3×4 grid, x centred four ways
  bank/                     *.npz descriptor banks
  out/                      shift tables, hillclimb_candidates.npz, figures/
```

Interpreter: `/vast/home/b/bonnen/.conda/envs/dev/bin/python` (numpy 2.4.6, scipy 1.17.1,
matplotlib 3.11.1, pandas; no torch needed). Run compute on SLURM `genoa-std-mem`, not on
the login node — it routinely sits at load average 65 with 336 users.

Key figures in `out/figures/`: `fig42_grid_voxel16.png` (3×4 rows × bins),
`fig43_grid5_margin_voxel16.png` and `fig43_grid5_marginadv_voxel16.png` (with the grey
base-DINOv2 control column), `fig44_centred_*.png` (centring), `fig31_accuracy_by_category.png`.

---

## 7. Interpretation

**The model-free metric works, and its scope is narrower than it first appears.**
A distribution-shift measure computed from stimulus geometry alone — no network, no
learned parameters — predicts how much a model benefits from fine-tuning on a category
(r = −0.115, p = 0.0023), and it does so while leaving base DINOv2 completely unpredicted
(−0.004, p = 0.92). That control is the thing worth defending: it is the direct analogue
of the audit's complaint about the encoder-based metric, and the geometric version passes
it where the image-based version did not.

**Pose sensitivity is the substantive finding about representation.** A descriptor that
discards orientation sees nothing (−0.026, p = 0.50); one that retains it sees the effect
at every resolution from 8³ to 32³. Whatever the fine-tuned models are picking up is
"shape as seen from a viewpoint," not intrinsic shape. This is a claim about what
training buys, and it is cheap to state and hard to argue with.

**But the effect is largely between-category, not within.** Category-centring the x axis
takes the within-category correlation from −0.097 to +0.014. Since each trial contributes
exactly one on-category observation, "which model" and "which category" are the same
variable in that row, and removing category means removes the effect. The relationship is
therefore carried by roughly 12 category means rather than 706 independent trials, and the
effective n is far smaller than it looks. Any claim built on this row should be worded as
a statement about categories, and its uncertainty should be computed accordingly — a
cluster-robust or category-level analysis, not a t-test on 706 points.

**The hill climb's most useful output is a negative result.** Across 1,895 model-free
combinations, nothing beat the metric we already had for within-category margin: held-out
0.113 versus 0.133 for the fixed baseline, losing in 72% of repeated cross-validations,
despite an in-sample maximum that looked like a 38% improvement with a permutation-corrected
p of 0.001. Two things follow. First, the reported effect is not an artifact of one
descriptor choice — every top-20 candidate landed in the same narrow band, which is a form
of robustness. Second, |r| ≈ 0.13 appears to be the ceiling for this family of measures at
this n, and no amount of metric engineering moves it. Whether that ceiling is the
phenomenon or the noise floor cannot be settled from these data: there is one fine-tuned
model per category, hence no repeat measurements and no way to estimate the margin's
reliability. Establishing it needs re-running the fine-tunes under different seeds.

**The strongest empirical result is on the human side, and it was not the one being
optimised.** Geometric distance predicts human RT at −0.305 and human accuracy at +0.199,
with the sign *opposite* to the model effect: farther from the training distribution,
humans get faster and more accurate while models degrade. That opposition is the best
available evidence that the x axis is not simply trial difficulty in disguise — a
difficulty confound would push humans and models the same way. And the search *did* improve
here, reliably (held-out 0.368 vs 0.318, winning 100% of 50 repeats), with a 7-dimensional
topology-and-symmetry descriptor beating 32,768-dimensional voxel grids. Objects that are
topologically unusual relative to the training manifold are the ones humans resolve quickly.

**Recommended framing.** Lead with the human/model divergence (§5.4), which is
assumption-light and directly on-title. Use the within-category margin result (§5.3) as
the mechanistic claim, worded at the category level and shipped with its base-DINOv2
control (§5.2) — that control is what distinguishes it from the pooled rows, which look
better and mean less. Report the hill climb as a robustness check rather than a headline:
its value is that the result does not depend on the metric, not that it produced a better
one.
