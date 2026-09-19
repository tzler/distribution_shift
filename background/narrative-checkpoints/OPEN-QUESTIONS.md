# Open questions (19 September 2026)

Ordered by how much they would change the paper.

## 1. Is the within-category relationship graded, once the training set varies?

The claim the paper wants ("the further from the training data, the worse the support")
has been tested only in a design where it is not identifiable. The knockout and knock-in
runs vary the training set *within* a category with the trial fixed. Prediction: margin
falls for own-group knockouts in proportion to coverage lost and not for size-matched
random removals; rises for targeted knock-ins more than for random subsets of the same
size. If neither holds, the honest claim is coverage (binary), not distance.

## 2. Which metric best predicts margin change under intervention?

The random knock-in subsets make this a table, not a training job: every candidate x is
recomputed on the same runs, within trial. Candidates: knn k ∈ {1, 10, 50}; coverage ε;
d57 / voxel16 / depth maps; object- vs image-level. The winner and runner-up get targeted
runs for a head-to-head.

## 3. Does the encoder's own space show a within-category signal?

Track A. If even the chair model's representation shows nothing within category, the
descriptor search is over; if it does, geometry is too coarse and finer descriptors are
worth building.

## 4. What is the noise floor of the margin?

One model per category, no repeat seeds: the margin's reliability is unknown, so the
0.13 ceiling of the within-category search could be signal or noise. One second seed per
category would settle it.

## 5. Does view-invariance hold beyond the training grid?

No MOCHI camera is more than ~25° from a training camera. A test at 60° or 90° needs new
renders of MOCHI objects and cannot be done with the existing images.

## 6. Nuisance variables in the interventions

LoRA rank, epochs, learning rate are pinned to the original recipe. The one worth an
ablation is epochs for the small knock-in subsets (10 vs 30), because it sets the cost of
everything after.

## Set aside

- Human RT / accuracy vs geometric distance (−0.31 / +0.20): MOCHI's ShapeNet trials were
  selected adversarially, so this may be trial construction.
- Pseudo-depth for the non-ShapeNet MOCHI datasets: puts a model back into the model-free
  metric, and there are no per-category fine-tunes to test against there.
