# Checkpoint 1 — A model-free ruler, and results that looked finished

*Written from the vantage point of ~mid-September 2026, before any control was run.*

## Where we started

The manuscript's distribution-shift metric (§3.5) is model-based twice: trial-to-training
distance is a nearest-neighbour search in the pretrained encoder's own feature space, and
the test-time proxy `ŝ = 1 + εm` is that encoder's oddity margin. The in-repo audit
(`Dist-shift-data/L1norm_vs_distshift/README.md`) showed r(shift, ‖φ‖₁) = 0.95 (ResNet-50),
0.89 (DeiT): for DeiT the shift adds nothing to the feature norm in a nested regression.

MOCHI's ShapeNet and ShapeGen trials come from known 3D assets, so geometry is *known*.
Goal: a shift measure with **zero learned parameters**, tested head-to-head against the
incumbent.

## What we built

- Descriptors from `model_normalized.solid.binvox` (128³): `d57` (32 D2 bins + 16 shell
  bins + 9 scalars, rotation-invariant), raw occupancy grids `voxel8/16/32`, `bbox` (7).
  2D silhouette descriptors (20-d) for ShapeGen, where only `.blend` files exist.
- Training bank: 20,885 ShapeNet objects in the 12 fine-tuned categories, MOCHI objects
  excluded.
- **The shift, first version:** the manuscript's Eq. 1–3 with φ swapped —
  `min_C ½[d(φ(A),C) + d(φ(B),C)]`, L1 on z-scored descriptors. (`compute_shift.py`)
- The design we had and did not yet appreciate: 12 category-specific DINOv2-L fine-tunes
  (LoRA r=16, multi-similarity loss, 30 epochs), every MOCHI trial run through all 12.

## Results that looked finished

**Figure-1 reproduction with φ swapped** (`out/fig_margin_vs_geometric_shapegen.png`),
30-bin r against three pretrained encoders' `ŝ = 1 − m`:

| | dinov2-L | dinov2-g | clip-g14 |
|---|---|---|---|
| shapegen, geometric | −0.739 | −0.805 | −0.800 |
| shapegen, incumbent | −0.547 | −0.635 | −0.526 |
| shapenet, geometric | −0.496 | −0.403 | −0.562 |
| shapenet, incumbent | −0.548 | −0.565 | −0.277 |

"A metric with no learned parameters beats the incumbent on shapegen for every encoder."

**Every trial × every model** (`fig40`): 8,472 observations, r = −0.223, binned r −0.84 at
10 bins and −0.76 at 70. A clean monotone decline.

## What we believed

That the circularity critique was answered and the metric was a drop-in replacement.

## What we had not yet checked

Whether the pretrained model — which never saw any fine-tuning set — was also predicted
by the metric. And how tight the ½·d(A,B) lower bound on the Eq. 1–3 form actually is.
Both turn out to matter (Checkpoints 2 and 4).
