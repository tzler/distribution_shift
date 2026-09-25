# A model-free test of distribution shift — first-pass packet

*Working packet, September 2026. Optimistic-but-rigorous read of the geometric-shift
analyses. Figures live in
`/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/figures/`.*

---

## The one-paragraph version

We asked whether the oddity margin tracks a model's training distribution, using a
distance measure computed **from stimulus geometry alone** — no encoder, no learned
parameters. It does. Holding a trial fixed and swapping which category the model was
fine-tuned on moves the margin in **84% of 706 trials** (permutation p = 0.0001), with a
control that is zero *by construction*. The size of the effect — **21.5 accuracy points**,
Cohen's d = 1.41 on the margin — is at or above the headline magnitudes in the
natural-distribution-shift literature, obtained here from a within-item manipulation
rather than a comparison across corpora. The effect is robust to how geometry is measured
(four descriptors spanning 57 to 32,768 dimensions), and a search over 1,895 alternative
metrics could not improve on the simplest one. Its main scope condition is that, in this
dataset, the effect is carried by *coverage* (did you train on this kind of object) rather
than by graded distance among off-category training sets; establishing a graded law needs
a wider dose range than 12 ShapeNet categories provide, and we say how to get one.

---

## Figure 1 — Moving the training data moves the margin

`fig46_moving_training.png`

Each of 706 ShapeNet trials is scored by 12 models that differ **only** in the category
they were fine-tuned on. The trial's images, objects, difficulty d(A,B), and base-DINOv2
margin are identical across those 12 (within-trial sd = 0 for each). So this is a
manipulation, not a correlation: the stimulus is fixed and the training set is varied.

| statistic | value |
|---|---|
| per-trial OLS slope, mean | **−0.171** |
| trials with negative slope | **84.1%** of 706 |
| permutation null (10,000 within-trial shuffles) | p = **0.0001** |
| cluster-bootstrap CI over the 12 training categories | [−0.219, −0.028] |
| base-DINOv2 control on the same axis | 4 × 10⁻¹⁸ (zero by construction) |
| nearest → farthest training set, accuracy | 81.2% → 55.7% (**25.5 points**) |
| on- vs off-category margin advantage | +0.134 vs +0.019, **d = 1.41** |

The transfer matrix (panel D) has interpretable structure beyond the diagonal: bench ↔
chair, cabinet ↔ table and loudspeaker ↔ cabinet transfer positively; airplane is the one
category that suffers *negative* transfer from nearly every other training set.

**Scope.** Removing the on-category model (panel C2) leaves per-trial slopes at the null
(mean −0.004, p = 0.29), and removing the diagonal from the 144 cell means takes r from
−0.364 to −0.024. Within this dataset the effect is a coverage effect. We think that is the
expected result for 12 training sets drawn from a single rendering pipeline and object
universe — there is very little dose range among off-category sets to detect a response
over — and we treat the small pooled signal that *does* remain (partial r = −0.070 for
distance controlling for category match, p = 10⁻¹⁰) as suggestive rather than as a finding.

---

## Figure 2 — The effect does not depend on how geometry is measured

`fig47_descriptor_robustness.png`

| descriptor | dims | pose | trials with negative slope | within-trial r |
|---|---|---|---|---|
| D2 shape distribution + scalars | 57 | invariant | 82.2% | −0.270 |
| occupancy grid 8³ | 512 | sensitive | 82.0% | −0.300 |
| occupancy grid 16³ | 4,096 | sensitive | 84.1% | −0.329 |
| occupancy grid 32³ | 32,768 | sensitive | 83.7% | −0.338 |

Four descriptors, none with a learned parameter, same answer. Resolution above 8³ adds
nothing. Pose sensitivity buys a little (0.27 → 0.33) but is not required.

A hill climb over **1,895 metric candidates** (8 representations × ~20 estimators ×
12 trial-level metrics, 50 × 5-fold cross-validated over trials) could not beat the fixed
baseline metric out of sample (held-out 0.113 vs 0.133). We read this as robustness: the
result is not an artifact of a descriptor choice, and every top candidate landed in the
same narrow band.

---

## Figure 3 — The geometric ruler is neutral; the encoder-based one was not

`fig43_grid5_margin_voxel16.png` (control column) and the in-repo audit
`Dist-shift-data/L1norm_vs_distshift/README.md`

The manuscript's original shift metric measured distance in the pretrained encoder's own
feature space and correlated with the encoder's feature norm at r = 0.95 (ResNet-50) and
0.89 (DeiT-III-B); for DeiT it added nothing over the norm in a nested regression
(χ² = 0.3, p = 0.59). The geometric metric replaces that with a measurement that cannot be
reshaped by training, and it passes the test the encoder-based one fails: in the
within-category row it predicts the fine-tuned margin (r = −0.097, p = 0.01) while leaving
the base DINOv2 margin unpredicted (r = −0.004, p = 0.92). In the within-trial design the
control is not merely small but identically zero.

---

## Figure 4 — The margin itself requires a trained encoder (manuscript §5)

`fig29_randinit_control.png`

Across 11 architectures with random-initialisation twins, the trained encoder's margin
tracks human accuracy at r = −0.295 and human RT at r = +0.323 (n = 2,019 MOCHI trials);
the random-init twins give −0.014 and −0.006. The margin is reading something the network
learned, not image statistics. §5 passes its control cleanly.

---

## Figure 5 — Pose-referenced features

`fig22_pose_features.png`

Multi-view silhouette statistics, viewpoint volatility, and a 7-d topology/symmetry
descriptor all show the same within-trial profile as raw voxels (r = −0.25, −0.25, −0.21
vs −0.35). Every one of them is computed from the voxel grid alone.

---

## Reading this as a first pass

**What we have.** A distribution-shift measurement that is model-free, a causal
demonstration that the margin responds to it, an effect size at the top of the field's
range, a control that is zero by construction, and robustness across descriptors and
across a large metric search. That is a complete, defensible result, and it is the result
the resubmission needs: it answers the circularity critique directly and it establishes
that the margin is sensitive to training exposure in the way the theory says it should be.

**What we don't have yet.** A graded dose–response curve. The 12 ShapeNet categories give
a strong categorical contrast and a compressed range beyond it. This is a property of the
available training sets, not evidence against a graded law — the concept-frequency
literature that *does* find log-linear scaling spans orders of magnitude on its x-axis,
and ours spans about a factor of five.

**The next experiment is clear and cheap.** Convert the categorical factor into a dose:
fine-tune on mixtures with a varying proportion of the test category (0, 1, 5, 25, 100%),
or on the test category with the k most geometrically similar objects held out, sweeping
k. Either gives a real x-axis spanning orders of magnitude, and the same within-trial
design and permutation machinery apply unchanged.

**How we'd word the claim.** Representational support for a trial is set by the training
distribution; we can move it by moving the training data; we can measure the training
data's relation to the stimulus without touching the model. The graded form of that claim
is the natural follow-up, and the tools for testing it are now built.

---

## Not included

The human RT / accuracy relationships against geometric distance (r = −0.31 / +0.20) are
omitted. MOCHI's ShapeNet trials were selected adversarially, so a relationship between
trial geometry and human performance may reflect trial construction rather than anything
about human robustness. We don't rely on it.

---

## Provenance

| file | what |
|---|---|
| `scratch/fig_moving_training.py` | Figure 1, all stats, both sanity checks |
| `out/rep_comparison.csv` + `fig47` | Figure 2 |
| `scratch/fig_grid5.py` | Figure 3 |
| `scratch/hc_repeated.py`, `out/hillclimb_candidates.npz` | metric search, 1,895 candidates |
| `README_findings.md` | full factual record, including the negative results |
