# Distribution shift and the oddity margin
*Last meaningful update: 2026-09-19 · Lead: TB · Status: active*

## Goal
Find a good estimate of distribution shift — how far a test trial sits from what a model was trained on — and establish whether that estimate tracks the model's oddity margin. If it does, the margin is a readout of representational support, and "support depends on training data" becomes a claim we can test by changing the training data rather than just correlate.

## Current status

**Where we started.** The manuscript's shift estimate is computed inside the encoder whose margin it explains: nearest-neighbour distance in DINOv2 feature space, in the form ½[d(A,C)+d(B,C)] over training images C. That form has a floor — by the triangle inequality it can never be smaller than ½·d(A,B), the very thing the oddity task is decided by — and in raw feature space it is mostly the size of the feature vector (r = 0.95 with ‖φ‖₁). On our data it predicts the *pretrained* model's margin (+0.38 pooled, +0.61 on-category) better than any fine-tuned model's, with the sign backwards. It measures how easy the trial is for the encoder, not shift [D06].

![why the original metric is wrong](evidence/fig60_why_wrong.png)
![the triangle inequality, and that cosine/L2 do not escape it](evidence/fig61_triangle.png)

**What we did about it.** Two changes. (1) A new estimate: never difference A and B — average a per-image quantity over the trial's images — and count *coverage*, the training mass within a radius of each image, rather than distance to the nearest neighbour. (2) A model-free space: the descriptors come from the stimuli's known 3D geometry (ShapeNet voxels), so no encoder is anywhere in the ruler [D02, D05].

**Where it works.** With 12 category-specific fine-tunes and every trial run through all 12, the stimulus can be held fixed while the training set varies. The margin follows the training set in 84% of trials (permutation p = 10⁻⁴), accuracy falls 81 → 56% from the nearest to the farthest training set while the pretrained model stays flat at 51%, and the base-model control — the thing the original metric fails — is zero by construction. Coverage is the only estimate whose *pooled* relationship also passes that control (−0.27 vs a control of −0.03) [D02, D05].

![moving the training data moves the margin](evidence/fig46_moving_training.png)
![coverage vs distance vs the original, pooled, control on its own axis](evidence/fig56_pooled_three_metrics.png)

**Where it fails.** Within a condition. Hold the category fixed and ask whether a chair that is farther from the chair training set gets a lower margin from the chair model: r ≈ 0 for every estimator we have, after category centring. The one curve that looked graded turned out to sort trials by category, not by anything within one [D03, D07]. This matters: without a within-condition result, what we have shown is *category* shift — did you train on this kind of object — not distribution shift as a continuous quantity.

![the on-category curve is category identity](evidence/fig57_oncat_anatomy.png)

**Why it fails, and what we are doing now.** With one training set per category, "far from the training set" and "an unusual object" are the same variable; no estimator can separate them. So the current work is not a better metric but a finer-grained assay: fine-tune models on training sets that vary *within* a category — knock out a trial's nearest training objects, or knock in a small targeted subset — and ask whether the margin follows coverage with the trial fixed [D09]. Alongside: an encoder-space upper bound (does the chair model's own representation show a within-category signal that geometry misses?), and a test of whether support is view-conditioned (it is not, within the training view grid) [D08].

## Current approach
- **Knockout** (30 fine-tunes, chair/airplane/table): remove each test group's k nearest training objects, k ∈ {10, 50}, plus size-matched random removals. Each trial gets 11 training sets, stimulus fixed. Pilot running (job 8519673).
- **Knock-in** (20 fine-tunes from pretrained, chairs): 8 random subsets of 100, 6 targeted (nearest 100 to a low-margin trial), 6 cross-category controls. The random runs make the metric comparison free: any candidate estimate is recomputed on the same runs, within trial.
- **Encoder-space bound**: features of all 311k training renders under pretrained and three fine-tuned models (job 8519397).
- Everything reuses the collaborator's training pipeline unchanged; per-epoch cost from the pilot sets the run budget.

## Next steps
- [ ] Pilot epoch time → run budget; random knock-in subsets first. (Claude)
- [ ] Encoder-space within-category test from the extracted features. (Claude)
- [ ] Submit knockout + knock-in; evaluate each checkpoint exactly as the 12 category models were. (Claude)
- [ ] Metric search on the random runs; targeted runs with the winner and runner-up. (Claude)
- [ ] Single-trial figure: the trial, the training chairs the metric chose, pretrained vs targeted vs random margins. (Claude)
- [ ] Lab meeting Monday: present this page and the arc D02 → D05 → D06 → D07 → D09. (TB)
- [ ] Decide whether the resubmission claims category coverage now or waits for the within-category result. (TB)

## Open questions
1. Is the within-category relationship graded once the training set varies? (The experiments above.)
2. Which estimate best predicts margin *change* under intervention — coverage radius, kNN k, descriptor, object vs image level?
3. Does the encoder's own space carry a within-category signal that geometry misses?
4. What is the margin's noise floor? One model per category, no repeat seeds.
5. Does view-invariance hold beyond ~25° from the training grid? (Needs new renders.)
6. Epochs for the small knock-in subsets — the one training variable worth an ablation.

Set aside: human RT/accuracy vs geometric distance (adversarial trial selection) [D04]; pseudo-depth for non-ShapeNet MOCHI [D08].

## Pointers
- This repo: estimator `blind_shift.py`; coverage `scratch/coverage_sweep.py`; viewpoint `scratch/viewdepth_pipeline.py`; cited figures in `evidence/` with provenance. Full factual record `README_findings.md`. Walkthrough artifact https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r.
- Experiments: `../knockout/` (`design.json`, `design_knockin.json`, `scripts/`).
- Upstream: collaborator pipeline `../../Dist-shift/HIDA/hida-tune/` (read-only); category results `ShapeNet_OOD_Analyses/<cat>/ood_analysis_results.csv`; metric audit `../L1norm_vs_distshift/README.md`.
- Background: `background/` — narrative back-fill, citation list, slot for the MOCHI project's STATE.md.

## TODOs (back-fill)
- D01–D03 dates approximate (recalled); what was tried before this session's record is not yet captured — needs TB.
- The manuscript's published shapenet r = .91 (Fig. 1) was never reproduced (got +0.34); unresolved.
- `background/` citation list is a first draft.
