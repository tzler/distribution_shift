# A model-free geometric distribution-shift metric

Run overnight 2026-09-08. Everything here is reproducible from this directory with
`/vast/home/b/bonnen/.conda/envs/dev/bin/python`. No GPU, no torch, no new installs.

---

## TL;DR

**Purpose: replace the model-based target in the Sec 4.2 / Fig 1 validation.** The
manuscript validates the oddity-margin proxy `ŝ` against an "empirical" distribution
shift — but that target is itself a nearest-neighbour ℓ1 distance in a *pretrained
DINOv2 feature space*, so a model-based proxy is being validated against a model-based
ground truth. This directory provides a **purely geometric target**, computed from the
3D meshes (and silhouettes), with **no encoder anywhere**, and re-runs that validation.

**Headline (after a full audit and a proper causal design): the model-free geometric
measure of distance-to-training is REAL, and it corroborates the manuscript's empirical
metric rather than replacing it.**

Getting here took three passes; the first two were misleading and are recorded below so
the reasoning can be checked.

**Pass 1 (naive).** Margin vs Delta_geom, 30 quantile bins: shapenet +0.726, shapegen
+0.536. Looks great.

**Pass 2 (over-corrected).** Delta_geom is r=+0.997 with within-trial dissimilarity
d(A,B); partialling d(A,B) kills it (-0.019 n.s.). Same for the manuscript's own l1
target on all 4 subsets. BUT this test is invalid: knn and d(A,B) are collinear at
r~0.85 *by construction*, so a null partial is expected under both hypotheses. Stratified
within-d(A,B) tests were merely underpowered (Stouffer Z=+1.73, p=.083), not null.
MOCHI cannot settle it: each object is the oddity in exactly 1 trial, so the pairwise
term cannot be marginalised.

**Pass 3 (the design that works).** Hold the TRIAL fixed and vary the TRAINING SET, using
the 12 per-category DINOv2-L fine-tunes in
`Dist-shift/HIDA/hida-tune/ShapeNet_OOD_Analyses/<category>/`. d(A,B) is a stimulus
property, so it is absorbed *exactly* by trial fixed effects (verified: its within-trial
variance is 0). Panel = 706 trials x 12 category-models = 8,472 observations.

| predictor (DV = fine-tuning advantage) | pooled | **two-way within (trial + category FE)** |
|---|---|---|
| GEOMETRIC shift (model-free) | -0.034 | **-0.0911**  t=-8.06  p=9e-16 |
| GEOMETRIC knn (model-free) | -0.070 | **-0.1204**  t=-10.68  p=1e-28 |
| GEOMETRIC knn, DV = fine-tuned margin | | **-0.3510**  p=3e-244 |
| DINOv2 empirical l1 | -0.243 | **-0.1816**  p=1e-63 |

Sign is the manuscript's prediction: fine-tuning helps most on trials geometrically
CLOSE to that category's training set. The effect *strengthens* under the fixed effects
(-0.034 -> -0.091), so d(A,B) was masking it, not creating it. The kNN density version
(not bounded by d_AB) beats the min version, as a genuine distributional measure should.

**Permutation placebo (200 perms of which category's geometry pairs with which model):**

| effect | real r | placebo null | z |
|---|---|---|---|
| geom_shift -> advantage | -0.0911 | -0.0008 +/- 0.0118 | **-7.7** |
| geom_knn -> advantage | -0.1204 | +0.0004 +/- 0.0118 | **-10.2** |
| geom_knn -> margin | -0.3510 | +0.0002 +/- 0.0110 | **-31.9** |

The effect requires the correspondence between a training set's geometry and the model
actually trained on it.

**Head-to-head.** Within-trial, the two metrics agree at r=+0.647, but DINOv2 is stronger
(-0.182 vs -0.120) and subsumes geometry in a joint model (beta_DINOv2 t=-12.73;
beta_GEOM t=-0.36, n.s.). So geometry is a *weaker* proxy for the same underlying
quantity -- it does not add incremental predictive power.

**Why this is still worth having.** The concern with the manuscript's validation is
circularity: a DINOv2-based proxy validated against a DINOv2-based target. A model-free
measure computed from 3D meshes, with no encoder in the loop, independently predicts
fine-tuning benefit in the predicted direction, in a design where inter-object similarity
is controlled by construction, and agrees with the empirical metric at r=0.647. That is
evidence the empirical shift tracks a real property of the data, not an artifact of the
encoder's feature space. It corroborates Sec 4, it does not replace it.

Run: `within_trial_design.py` (the causal design), `validate_fig1.py`, `validate_margin.py`.

### Properties of the geometric metric itself

1. **It is not a restatement of anything trivial.** The audit in
   `../L1norm_vs_distshift/` showed the incumbent is ~95% collinear with ‖φ‖₁. The 3D
   geometric metric correlates **+0.045 with `feat_l1`**, +0.101 with `pix_l1`, and
   **+0.074 with the incumbent metric itself**. It is close to orthogonal to all of it.
2. **But the shared-neighbour form does not isolate distance-from-training.** It
   correlates **r = +0.997** with `d(A,B)`, the within-trial dissimilarity of the two
   objects — the same structural issue the audit found in the incumbent. Use
   `geom_knn200` (k-NN density, not bounded below by d_AB) if you want the distributional
   component alone. See "The honest reading".

### Secondary (not the first analysis — recorded for later)

Behavioural correlations: the geometric metric predicts human RT better than the
incumbent (shapenet 3D −0.261 vs −0.132; shapegen −0.235 vs −0.181), but see the
caveats — most of that is carried by `d(A,B)` and, for shapenet, by between-category
variance.

---

## What was built

| File | What it is |
|---|---|
| `geom3d.py` | binvox parser + rotation/translation/scale-invariant 3D shape descriptor (57-d: D2 shape distribution, shell histogram, inertia eigenvalues, convexity, compactness) |
| `geom2d.py` | silhouette extraction + 20-d scale-invariant 2D shape descriptor |
| `build_bank_3d.py` / `build_bank_2d.py` | descriptor extraction, multiprocessing |
| `compute_shift.py` | the metric itself (paper Eqs. 1–3 with φ swapped) |
| **`validate_fig1.py`** | **PRIMARY: exact Fig 1 reproduction with the geometric target substituted** |
| **`validate_margin.py`** | **THE PRIMARY ANALYSIS: oddity margin vs the model-free geometric target (Fig 1 replacement)** |
| `analyse.py` | secondary battery: confounds, partials, behaviour |
| `fig_qualitative.py` | trials next to their retrieved nearest training object |

Outputs, all under `/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/`:

```
out/shift3d_shapenet.csv     706 trials   <- the headline metric
out/shift2d_shapenet.csv     865 trials
out/shift2d_shapegen.csv     548 trials
out/fig1_margin_vs_geometric.csv  <- PRIMARY: fine-tuned margin vs geometric target
out/fig1_repro_shapenet.png, out/fig1_repro_shapegen.png  <- Fig 1, both targets side by side
out/margin_vs_geometric.csv  <- same test across 8 pretrained encoders (margin vs geometric target, per encoder)
out/fig_margin_vs_geometric_shapegen.png   <- Fig 1 replacement, model-free target
out/fig_margin_vs_geometric_shapenet.png
out/stats_all.csv            every number below, machine-readable
out/fig_shapenet_3d.png      binned curves vs human accuracy + RT, and the d_AB decomposition
out/fig_shapenet_3d_confounds.png   scatter vs feat_l1 / pix_l1 / coverage / contrast / incumbent
out/fig_shapegen_2d.png, out/fig_shapegen_2d_confounds.png
out/fig_shapenet_2d.png, out/fig_shapenet_2d_confounds.png
out/fig_qualitative_nn.png   sanity: trials next to their retrieved nearest training object
bank/bank3d_shapenet.npz     36,448 training objects x 57
bank/test3d_shapenet.npz     897 MOCHI objects x 57
bank/bank2d_shapenet.npz     20,754 objects (311,370 renders)
bank/bank2d_shapegen.npz     19,200 objects (307,200 renders)
bank/mochi2d.npz             4,241 MOCHI test images
```

Join key is `trial`, matching `MOCHI/mochi_trials.csv` and
`../L1norm_vs_distshift/trials_*.csv` 1:1.

---

## Method

Identical to the paper's Eqs. 1–3; **only φ changes**, from a DINOv2 feature to a
hand-computed geometric descriptor:

```
D(trial, C_i) = ½[ d(φ(A), φ(C_i)) + d(φ(B), φ(C_i)) ]
Δ_geom        = min_i D(trial, C_i)        over the 36,448 training objects
```

- **3D (shapenet)** — φ is a shape descriptor of the *object*, computed from ShapeNet's
  `model_normalized.solid.binvox` 128³ voxelization. View-independent, so lighting,
  background, camera and resolution cannot enter. Descriptors are z-scored using the
  training bank's mean/sd, then ℓ1.
- **2D (shapegen, and shapenet as a control)** — φ is a silhouette descriptor.
  Train-side silhouettes come free from the RGBA alpha channel; MOCHI test silhouettes
  come from a border-connected-component flood fill (not thresholding — objects reach
  gray 255 against white, so a threshold punches holes in the mask).

**Train/test separation is verified, not assumed.** `shapenet_training` (36,488 objects)
has **zero id overlap** with the 1,054 MOCHI object ids; asserted in code before
anything is computed. Note the directory named `shapenet_mochi_excluded` contains the
MOCHI objects *themselves* (excluded **from** training) — the opposite of what the name
suggests, and an easy way to invalidate the whole analysis.

---

## Results

### shapenet, 3D voxel metric (n = 706)

| predictor | human acc | human RT | DINOv2-G acc |
|---|---|---|---|
| **geometric shift** | **+0.200*** | **−0.261*** | +0.055 |
| incumbent L1 shift | +0.183*** | −0.132*** | +0.073 |
| `feat_l1` | +0.148*** | −0.051 | +0.290*** |

Collinearity with the quantities that sink the incumbent — this is the metric's
strongest selling point:

| | `feat_l1` | `pix_l1` | `coverage` | `contrast` | incumbent |
|---|---|---|---|---|---|
| geometric shift | +0.045 (n.s.) | +0.101 | −0.104 | −0.102 | +0.074 |

Incremental value, partialling out the incumbent *and* all four low-level confounds:

| DV | geometric \| (incumbent + confounds) | incumbent \| (geometric + confounds) |
|---|---|---|
| human accuracy | +0.218 (p=5e-09) | +0.205 (p=4e-08) |
| **human RT** | **−0.304 (p=2e-16)** | −0.138 (p=2e-04) |

### shapegen, 2D silhouette metric (n = 548)

| predictor | human acc | human RT |
|---|---|---|
| **geometric shift** | **+0.192*** | **−0.235*** |
| incumbent L1 shift | +0.191*** | −0.181*** |

Confound collinearity is clean here too (`feat_l1` +0.076, `pix_l1` +0.046,
`coverage` −0.051, `contrast` −0.048, incumbent +0.045).

### shapenet, 2D silhouette metric (n = 865) — **this one fails**

Reported so it is not silently omitted. The 2D metric on shapenet is badly contaminated:
`pix_l1` **+0.562**, `coverage` **−0.562**, `contrast` **−0.573**, and it barely predicts
behaviour (acc +0.085, RT −0.075). Silhouettes work for shapegen's abstract shapes but
not for shapenet.

Diagnosed cause: **view-sampling noise**, not uninformative silhouettes. A bank object is
averaged over 15 views but a test trial supplies only 2–3. A 2-view mean recovers its own
15-view identity at only 0.350 (chance 0.010), and the 2-view/15-view discrepancy is
~40% of the mean between-object distance. **This asymmetry exists in the paper's own
metric too** — training objects are averaged over 50 viewpoints while a MOCHI trial
contributes 2. Worth checking there.

---

## The honest reading

Three caveats, each of which I think matters more than the headline correlation.

**1. This is not distance-from-training; it is within-trial dissimilarity.**
`Δ_geom` correlates **+0.997** with `d(A,B)` (shapenet 3D; +0.993 shapegen). The
nearest-neighbour search adds essentially nothing — which is *exactly* the pathology the
audit found in the incumbent, arising the same way: the shared-neighbour min is bounded
below by ½·d_AB. Controlling for `d(A,B)`:

| | raw | partial \| d(A,B) |
|---|---|---|
| shift → human accuracy | +0.200 | −0.048 (n.s.) |
| shift → human RT | −0.261 | +0.128 (sign flips) |

The genuinely distributional components — each object's own distance to its nearest
training object — are weak (`dB_nn`: acc +0.153, RT −0.157; `dA_nn`: +0.087, −0.099),
and do not survive the same control.

**1b. A proper k-NN density measure separates the two — and the answer differs by dataset.**
Because the shared-neighbour min is degenerate, the output CSVs also carry
`geom_knn{10,50,200}` = mean distance to the k nearest training objects (a standard OOD
score, *not* bounded below by d_AB). Within-category:

| | shapenet 3D | shapegen 2D |
|---|---|---|
| `knn200` → human accuracy | +0.026 (n.s.) | **+0.131** (p=2e-03) |
| `knn200` → human RT | −0.039 (n.s.) | **−0.133** (p=2e-03) |
| r with d(A,B) | 0.848 | 0.637 |

So for **shapegen** there is a genuine, model-free distance-to-training effect that
survives both the d_AB and the category controls — modest (|r| ≈ 0.13) but real, and
clean on every low-level confound (`feat_l1` +0.132, `pix_l1` −0.028, `coverage` +0.048).
For **shapenet** the distance-to-training component is **null** within category; only
the d_AB-driven part predicts anything.

Note that even k-NN density stays substantially correlated with d(A,B) (0.64–0.85). That
is intrinsic, not a bug: an object far from the training manifold also tends to be far
from other objects generally.

**2. The sign is opposite to the paper's narrative.** Objects geometrically *far* from
training are *easier*, not harder: accuracy rises and RT falls with shift. This is
sensible for an odd-one-out task — a distinctive, unusual object is easy to spot — but
it is the reverse of "OOD ⇒ harder". The qualitative figure shows it plainly: the
lowest-shift trials are flat slab-like telephones/displays (acc 0.38, RT 5069 ms) and
the highest-shift are spindly lamps (acc 1.00, RT ~2200 ms).

**3. Half the shapenet effect is between-category.** 49.7% of the geometric-shift
variance is between ShapeNet categories. Centering within category:

| | acc raw → within | RT raw → within |
|---|---|---|
| geometric shift | +0.200 → **+0.079*** | −0.261 → **−0.135*** |
| incumbent | +0.183 → +0.159*** | −0.132 → −0.127*** |

So within category the incumbent is *better* on accuracy; the geometric metric keeps an
edge only on RT. Per-category, the effect is carried by airplane (acc +0.427, RT −0.421),
watercraft (−0.393), telephone (−0.351) and car (−0.282), and is null for cabinet, chair,
table, sofa, display, loudspeaker.

shapegen has no such problem — only 2.6% between-condition variance, and the within-
condition effects hold nearly full strength (acc +0.163***, RT −0.212***, vs the
incumbent's +0.160/−0.157).

---

## Coverage gap

706 of 867 shapenet trials have the 3D metric. The missing 161 trials involve **157 MOCHI
car objects (synset 02958343) whose meshes are absent from this filesystem** — the
directories under `shapenet_mochi_excluded/02958343/` are empty. Not recoverable without
re-downloading ShapeNet. All 867 are covered by the (weaker) 2D metric.

---

## Suggested next steps

1. **Use `dA_nn`/`dB_nn`, not the shared-neighbour min.** The shared min is structurally
   dominated by d_AB. Per-object distance-to-training is the quantity that actually means
   "distribution shift", and it is already in the output CSVs.
2. **Report d_AB alongside any shift metric.** Given both the incumbent and this metric
   collapse onto it, the paper's shift results may be substantially an inter-object
   similarity effect. §4.2 already argues this is "a feature, not a bug" — this gives a
   model-free way to actually measure how much.
3. **Check the 2-view vs 50-view asymmetry in the paper's own φ(A).**
4. Recover the 157 car meshes to close the coverage gap.
5. shapegen meshes exist only as 48,000 `.blend` files, and MOCHI's shapegen objects
   (`batch0_scene_*`) are not among them, so a 3D shapegen metric would need Blender
   extraction plus locating the MOCHI source geometry.

---

## Verification

`verify.py` (exits non-zero on any failure) — all checks pass as of this run:

```
1. train/test disjointness   training bank shares no object with MOCHI (|overlap|=0, |bank|=36448)
                             all 897 test descriptors are MOCHI objects
2. output tables             all three CSVs join 1:1 with mochi_trials.csv, no NaN
3. 3D descriptor invariance  128^3 vs 64^3 voxel grid: r=0.9965
4. 2D descriptor invariance  128 / 256 / 512 px: r=0.9981, r=0.9992
5. regression test           solidity tracks shapegen K |r|=0.590; compactness |r|=0.500
```

One honest caveat on check 3: although r=0.9965, the mean absolute change between the
128³ and 64³ descriptors is 0.533 sd units, so the descriptor is not perfectly
resolution-free. It does not affect any result here because every object uses the same
128³ binvox, but do not mix voxel resolutions in a single bank.

## Reproducing

```bash
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
D=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
sbatch $D/run_bank3d.sbatch     # ~1 min   (36k objects)
sbatch $D/run_bank2d.sbatch     # ~3 min   (620k renders)
$PY $D/build_bank_2d.py --spec mochi --out $D/bank/mochi2d.npz --workers 48
$PY $D/compute_shift.py --mode 3d --dataset shapenet \
    --bank $D/bank/bank3d_shapenet.npz --test $D/bank/test3d_shapenet.npz \
    --out $D/out/shift3d_shapenet.csv
$PY $D/analyse.py && $PY $D/fig_qualitative.py
```

Jobs ran on `genoa-std-mem`, 64 CPUs, ~352 GB. Total compute was under 10 minutes —
the cost is NFS latency, not arithmetic, so everything is parallel over objects.

---

## The oddity-blind estimator (current best version)

`blind_shift.py` — the same model-free geometric descriptors, but with an estimator
that closes the d(A,B) leak structurally:

    shift(t) = (1/|t|) sum_{x in t} (1/k) sum_{i in kNN_k(x)} ( 1 - g(x) . C_i )

oddity-blind (average a per-image quantity over ALL trial images, never a difference),
hub-robust (mean over k=50 nearest, not argmin), norm-free (cosine on L2-normalised
descriptors). Outputs: `out/blindshift_{shapenet,shapegen}.csv`,
`out/blindshift_shapenet_percat.csv`.

**It works, for shapenet/3D (n=706).**

| check | shared-neighbour min (old) | oddity-blind k=50 |
|---|---|---|
| r with d(A,B) | **0.996** | **0.455** |
| r with feat_l1 | +0.045 | **-0.011** |
| r with pix_l1 | +0.101 | +0.082 |
| r with coverage | -0.104 | -0.073 |
| r with contrast | -0.102 | -0.025 |
| r with DINOv2 empirical l1 | +0.074 | **+0.016** |

The leak is closed and the measure is essentially orthogonal to feature magnitude,
pixel statistics AND to the incumbent metric.

**The human relationships survive the fix** — they were not the d(A,B) artifact:

| DV | oddity-blind k=50 | old shared-min |
|---|---|---|
| human accuracy | +0.198*** | +0.204*** |
| human RT | **-0.265*** | -0.269*** |

**Within-trial x 12 category-models (8,472 obs), d(A,B) absorbed by construction:**

| DV | pooled | within-trial | permutation z | categories negative |
|---|---|---|---|---|
| fine-tuning advantage | -0.093 | **-0.110**  p=3e-24 | **-9.2** | 11/12 |
| fine-tuned margin | -0.104 | **-0.316**  p=5e-196 | **-30.2** | — |

Sign is the manuscript's prediction. Note the pooled trial-level advantage correlation
is +0.082 (wrong sign) and flips negative within-trial, so between-trial variance
remains confounded by trial difficulty; the within-trial design is still required.

**shapegen/2D FAILS this estimator and should not be used.** The oddity-blind form
exposes a per-image size confound that the shared-neighbour difference partly
cancelled: r with coverage +0.253, pix_l1 -0.225, contrast +0.188, and it predicts
nothing (human accuracy -0.021, RT +0.096). Silhouettes are not a substitute for
meshes here.

**On the neutral-ruler problem.** For an encoder-based shift the ruler is a free
choice, and measuring shift in the evaluated model's own space inverts the sign,
because contrastive fine-tuning expands the region it specialises in. A geometric
ruler cannot be reshaped by training: it is neutral by construction, for every model
and every training stage, so that failure mode cannot arise here.

---

## Geometric A–B similarity (a trial property, not a shift measure)

`ab_similarity.py` → `out/absim_{shapenet,shapegen}.csv`.
sim(A,B) = cosine between the L2-normalised geometric descriptors of the two trial
objects. Never touches the training bank. This is the quantity the shared-neighbour
shift was leaking, so it is worth characterising directly.

**shapenet (n=706), trial level, unbinned:**

| DV | r with A–B similarity |
|---|---|
| human accuracy | **−0.231*** |
| **human RT** | **+0.337*** |
| fine-tuned oddity margin | −0.236*** |
| fine-tuned accuracy | −0.203*** |
| pretrained oddity margin | −0.049 (n.s.) |
| pretrained accuracy | −0.062 (n.s.) |

More geometrically similar → harder, exactly as a difficulty variable should behave.
+0.337 with human RT is the strongest human relationship anywhere in this directory
(the shift measure gives −0.265).

**A dissociation worth noting:** humans and the FINE-TUNED model track geometric A–B
similarity; the PRETRAINED encoder does not (−0.049, −0.062, both n.s.). Fine-tuning
appears to install a 3D-shape sensitivity that pretrained DINOv2 lacks and humans have
natively. See `out/figures/fig5_ab_similarity.png`, top row vs bottom row.

r(A–B similarity, oddity-blind shift) = −0.534, so the two are related but separable —
which is the point of the oddity-blind estimator.

shapegen/2D is much weaker here too (human accuracy −0.092, RT +0.141).

## Figures

```
out/figures/fig4_blind_shift.png        oddity-blind shift vs margin / advantage / human behaviour
out/figures/fig5_ab_similarity.png      A–B geometric similarity vs the same DVs
out/figures/fig6_within_trial_blind.png within-trial x 12 models, d(A,B) absorbed
```
All 300 dpi. Bins are display only; every quoted statistic is trial-level.
