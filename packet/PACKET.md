# Model-free distribution shift — the packet

*September 2026. Figures are numbered in reading order. Every number quoted here was
either read off the figure or recomputed in this session; where a figure's caption is
the best description, it is quoted.*

Two short pieces of context before the figures:

**The problem.** The manuscript's shift metric measures trial-to-training distance in
the pretrained encoder's own feature space. The audit (`L1norm_vs_distshift/README.md`)
shows this correlates with the encoder's feature norm at r = 0.95, i.e. it is close to a
restatement of a model property. We wanted a distance with **no learned parameters**.

**The design that makes it clean.** Every MOCHI ShapeNet trial was run through 12
models that differ only in the ShapeNet category they were fine-tuned on. The trial's
images, objects, difficulty d(A,B) and base-DINOv2 margin are identical across those 12
(within-trial sd = 0 for each). So "which training set" is a within-item manipulation.
Most of the figures below use this.

---

## Part 1 — The model-free measure works, and it beats the incumbent

### 01, 02 — geometric shift vs the incumbent, three pretrained encoders  ⚠ CORRECTED
`01_shapegen_geometric_vs_incumbent.png`, `02_shapenet_geometric_vs_incumbent.png`, `A3_entanglement.png`

**The shapegen headline is withdrawn.** These figures use the manuscript's Eq. 1–3 shift with
φ swapped: `min_i ½[d(A,C_i)+d(B,C_i)]`, bounded below by ½·d(A,B). On shapegen that shift
correlates with d(A,B) at **r = +0.993** — it is object dissimilarity — and d(A,B) alone
predicts the encoder proxy better (|r| = 0.79) than the "shift" did (0.74). With the
oddity-blind metric the shapegen relationship goes to |r| = 0.12 (trial-level 0.02). The
incumbent's correlation with d(A,B) is only +0.07; it was never riding this confound.

| dinov2-large, 30-bin \|r\| | shapegen | shapenet |
|---|---|---|
| d(A,B) alone | 0.79 | 0.46 |
| entangled shift (fig 01/02) | 0.74 | 0.44 |
| incumbent DINOv2-ℓ₁ | 0.55 | 0.55 |
| **oddity-blind knn_mean** | **0.12** | **0.43** |

On shapenet a modest blind relationship survives, comparable to the incumbent. The honest
Act-2 sentence is: *a model-free metric matches the incumbent on shapenet; on shapegen the
apparent win was the triangle problem.* Everything from figure 06 onward uses the blind
metric and is unaffected.

### 03 — every trial × every model
`03_every_trial_x_every_model.png`

All 8,472 observations (706 trials × 12 fine-tunes), raw axes, nothing centred. Point-level
r = −0.223 (p = 4×10⁻⁹⁶); in quantile bins the conditional mean margin falls
monotonically across the whole range — binned r = −0.84 (10 bins), −0.78 (30), −0.77
(50), −0.76 (70). This is the figure that first showed the relationship was there.

### 04 — a model-free measure predicts which trials benefit from fine-tuning
`04_margin_validation.png`

Three panels from the original `make_figures.py`: the statistics without binning, why a
single-model analysis cannot settle the question, and the Fig 1 reproduction against a
model-based vs a model-free target.

### 05 — what is one data point, and each category separately
`05_what_is_one_point_and_per_category.png`

The explainer. *"Every MOCHI trial is run through 12 separate fine-tuned models, one per
ShapeNet training category. So one trial contributes 12 points: the x is how far that
trial's objects sit from that category's training objects, and the y is that model's oddity
margin on that trial. The trial's images, its two objects and its difficulty are identical
in all 12 — only the training set changes."* Then the same plot restricted to each of the 12
ShapeNet categories in turn, so you can see the relationship isn't carried by one or two
of them.

---

## Part 2 — Trial difficulty removed by design

### 06 — the oddity margin recovers distance-to-training, with trial difficulty removed by design
`06_difficulty_removed_by_design.png`

*"Each of 706 trials appears 12 times — once per ShapeNet training category. Each category
was used to fine-tune one model, so training set ↔ model is 1:1. x is a property of the
STIMULI and the training set only (no encoder in the geometric panel); y is that model's
behaviour. d(A,B) has zero within-trial variance, so trial difficulty cannot contribute."*

### 07 — models trained further from a trial's objects show a smaller margin
`07_margin_by_rank.png`

*"Points are ordered by rank within each trial (1 = its closest training set … 12 = its
furthest), so all 12 contain the same 706 trials — identical objects, identical d(A,B),
identical difficulty. Both axes raw."* Twelve points, every one an average over the same
706 trials. Nothing on the x axis can be a trial property.

### 08 — the two distribution-shift metrics in final form
`08_two_metrics_final_form.png`

*"Columns 1–2: within-trial rank design — 12 points, each containing the same 706 trials.
Raw axes, nothing demeaned. Column 3: trial level. A distribution-shift measure should not
be tracking these."* Geometric and incumbent side by side in the clean design.

### 09 — accuracy, in the within-trial design
`09_accuracy_within_trial.png`  *(new — the panel fig 25 was missing)*

Fine-tuned accuracy falls from **81.2 % at the nearest training set to 55.7 % at the
farthest** (25.5 points). The pretrained model, which never saw any of these training sets,
sits at 50.6 % in every rank — flat because every rank holds the same trials. The fall is
attributable to the training set alone. (Chance is 33 %.)

### 10 — moving the training data moves the margin
`10_moving_training_causal.png`

The formal version, at the honest grain. One OLS slope per trial (12 points each):
mean −0.171, **84.1 % of 706 trials negative**. Permutation null from 10,000 within-trial
shuffles of x: p = 0.0001. Cluster bootstrap over the 12 training categories (the true
number of independent units): 95 % CI [−0.219, −0.028]. The base-DINOv2 control on the
same axis is 4×10⁻¹⁸ — zero by construction, since it is trial-constant.

Effect sizes: on-category vs off-category margin advantage +0.134 vs +0.019, Cohen's
d = 1.41; accuracy 81.6 % vs 60.1 %. The 12×12 transfer matrix (panel D) has readable
structure: bench ↔ chair, cabinet ↔ table and loudspeaker ↔ cabinet transfer positively;
airplane is the one category hurt by nearly every other training set.

---

## Part 3 — It doesn't depend on how you compute it

### 11, 12, 13 — centring and binning
`11_centring_variants.png`, `12_binning_and_centring.png`, `13_centring_x_only.png`

Seven ways to centre the axes (11): *"Methods 1–3 leave between-trial variation in play, so
trial difficulty can contribute. Methods 4–7 remove it by different routes and should
agree."* They do. Rank bins vs value bins (12): *"fig 9 uses RANK bins, so there can only
ever be 12, and every bin holds the same 706 trials: d(A,B) is identical across the axis.
Binning on the raw VALUE allows any number of bins, but each bin then holds different
trials, so d(A,B) varies and the difficulty confound returns."* Centring x only (13):
*"Within a trial the centred x sums to zero, so its covariance with any trial-constant is
zero — the SLOPE is the same whether or not y is centred too; only r differs."*

### 14 — the same relationship under eight model-free object representations
`14_eight_representations.png`

*"The margin falls with distance under every representation — including raw voxel grids
with no designed features — and is flat when objects are randomly re-assigned."*
Recomputed this session for the four core descriptors, within-trial: 57-d rotation-invariant
r = −0.270 (82 % of slopes negative); voxels 8³ / 16³ / 32³: −0.300 / −0.329 / −0.338
(82–84 %). Resolution above 8³ adds nothing.

### 15 — nine estimators, and the pretrained model as control
`15_nine_estimators_and_control.png`

Top: *"pooled across trials, the PRETRAINED margin tracks geometric shift MORE strongly than
the fine-tuned one, though it cannot possibly depend on these training sets. The pooled
correlation is therefore a stimulus property, not a distributional effect."* Bottom:
*"within trial the pretrained margin is a constant (variance ~1e-35), so the design returns
a flat line; only the fine-tuned margin responds."* Then nine estimators — single NN, k-NN,
density-corrected, PCA reconstruction, Mahalanobis — same features, same bank, same curve.

A separate hill climb over **1,895 metric candidates** (8 representations × ~20
estimators × 12 trial-level metrics, 50 × 5-fold cross-validated over trials) could not
beat the simplest fixed metric out of sample. The result is not a descriptor choice.

### 16 — how simple can the measure be?
`16_how_simple.png`  *(rebuilt; each panel on its own y-axis)*

Left: the full pipeline. Middle and right: **one scalar per object** — the std of
centroid-to-surface distance, or log bounding-box volume straight from
`model_normalized.json` — as |z| against that category's training objects. No neighbours,
no cosine, no covariance. Within-trial r = −0.270 / −0.176 / −0.175; 82 / 75 / 74 % of
trials negative. Unsquashed, the two one-number measures show a graded decline over the
full rank range rather than a single step.

### 17 — the margin itself needs a trained encoder (manuscript §5)
`17_randinit_control_sec5.png`

Across 11 architectures with random-init twins, the trained margin tracks human accuracy at
r = −0.295 and RT at +0.323 (n = 2,019 MOCHI trials); random-init gives −0.014 and −0.006.
§5 passes its control.

---

## The optimistic read

This is a first pass, and what it has already established is a lot:

1. **A distance with no learned parameters recovers the margin about as well as the
   encoder-based one on shapenet** (02, A3). The shapegen "win" in 01 was the entangled
   shift tracking d(A,B) and is withdrawn.
2. **The relationship survives removing trial difficulty by design** (06–10). Every rank,
   every centred bin, holds the same trials; the control is flat by construction; the
   fine-tuned accuracy still falls by 25 points.
3. **It is causal in the ordinary sense**: the training set is assigned, fully crossed
   with trial, and the margin moves in 84 % of trials with a permutation p of 10⁻⁴ and a
   cluster CI that excludes zero at the category grain.
4. **It does not depend on the descriptor, the estimator, the centring, or the binning**
   (11–16), and a 1,895-candidate search confirmed there is no better single metric hiding.
5. **Even one scalar per object gets most of the way** (16).

What it hasn't established is a graded dose–response law *within* a category's training
set — see `WHY_WE_PIVOTED.md` for what that question is, why it made us pessimistic, and
why it is a separate, narrower question from the one the paper needs.

---

## Added after the packet: coverage instead of distance
`A4_coverage.png`

Count the category's training objects within a fixed cosine radius ε of the test object,
take −log(1 + count), average over the trial's images. Across ε = 0.03–0.20 and on both
voxel16 and d57 it beats knn_mean on every criterion: within-trial r −0.40 to −0.42
(vs −0.33), 85 % of trials negative, and — new — the **pooled** row passes the base-DINOv2
control (effect −0.25 to −0.27, control −0.02 to −0.04; knn_mean: −0.22 / −0.14). Split-half
(ε chosen on one half, scored on the other, 20 splits): coverage wins 20/20, held-out
−0.414 vs −0.328, held-out pooled control −0.041 vs −0.143. The partial correlation beyond
the on-category binary nearly doubles (0.07 → 0.12). Soft kernel mass does none of this;
the hard cut-off is what stops atypicality leaking in.

## Not included

The human RT / accuracy relationships with geometric distance are left out: MOCHI's ShapeNet
trials were selected adversarially, so a trial-geometry ↔ human-performance correlation may
be a property of trial construction.
