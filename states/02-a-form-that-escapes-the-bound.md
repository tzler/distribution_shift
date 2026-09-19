# State 2 — A form that escapes the bound, and what it does in the encoder's space
*Snapshot after the estimator changed* · ← [State 1](01-what-the-metric-measures.md) · [index](README.md) · [resources](../RESOURCES.md) · next → [State 3](03-a-model-free-ruler.md)

## Goal
A distance to the training data that is not bounded by the trial's own difficulty, and that does not predict the margin of a model which never saw the training set.

## Status
**The oddity-blind form.** For each image in the trial, the mean distance to its 50 nearest training items; then the mean over the trial's images. No A/B roles, nothing differenced, so no ½·d(A,B) floor by construction. Cosine on unit-normalised features, so vector length cannot enter either [../REASONING.md#D02].

**In the encoder's own space it helps, and it is not enough.** Giving each object its own nearest training image takes r(shift, d(A,B)) from 0.78 to 0.66 — no longer a theorem, but still high, because in that space both quantities scale with how the encoder spreads images out. And it still predicts the pretrained margin (+0.17) [../REASONING.md#D06]. The form was one of the problems; the space is the other.

![what rescues it: not the distance — the form, then the space](../evidence/fig61_triangle.png)

**Which distance metric?** L1, unit-normalised L1, cosine — all obey the bound; none is the fix. The fix is the form, and then the space.

## Strategy
Leave the encoder's space. MOCHI's ShapeNet and ShapeGen trials come from known 3D assets, so distance to the training data can be measured on the *stimuli* — descriptors computed from the objects' geometry, with no encoder and no learned parameter anywhere in the ruler. The encoder then appears only where it should: as the thing being measured.

## Resources used (cumulative, as of this state)
Agent time ≈ 5 h (8 Sep, to ~00:30). Compute: audit + the first shift computations (`clean_shift`, `cosshift`) ≈ **$4.80 / $1.50**. No new models.

## Next steps
- [ ] Voxel-based descriptors for every ShapeNet object (rotation-invariant and raw-grid variants); silhouette descriptors for ShapeGen.
- [ ] A training bank of the 20,885 objects the category models actually saw, MOCHI objects excluded.
- [ ] The oddity-blind estimator on those descriptors; the base-DINOv2 control on every row.

## Open questions
- Will a geometric ruler predict anything about a model trained on images?
- How to separate "far from this training set" from "an unusual object" — both are large distances.
