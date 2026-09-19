# State 0 — The manuscript, and why we opened it up
*Snapshot as of the submission's rejection · Lead: TB* · [index](README.md) · next → [State 1](01-what-the-metric-measures.md)

## Goal
Characterise human visual perception through distribution shift — the distance between what a system was trained on and what it is tested on. In a setting where both are known (3D object datasets, fine-tuned vision encoders, the MOCHI benchmark), show that a distance between train and test data predicts model performance, build a proxy for that distance computable from test images alone, and use the proxy to ask when humans outperform models and why.

## Status
**What the manuscript claimed** (*Humans adapt to distribution shift with increasing test-time compute*, NeurIPS 2026 submission):
1. An empirical distribution shift — nearest-neighbour ℓ₁ distance from a MOCHI trial to the fine-tuning set, in the pretrained encoder's feature space — predicts the benefit of fine-tuning: large close to the training data, none far away (Fig. 1 left).
2. A model-internal quantity, the fine-tuned model's oddity margin, recovers that empirical shift (r = .91 shapenet, .62 shapegen; Fig. 1 right) — so distribution shift can be estimated from test images alone.
3. Applied to large pretrained encoders, the proxy predicts when humans beat models and how long humans take; restricting viewing to a glance collapses humans to model level.

![Figure 1 of the submitted manuscript](../evidence/manuscript_fig1.png)

**The concerns.** [Reviewer text to be added by TB.] Internally, the audit in `L1norm_vs_distshift/` had already found that the empirical shift correlates with the size of the test images' feature vectors at r = 0.95 (ResNet-50) and 0.89 (DeiT), and that for DeiT it adds nothing to the norm in a nested regression. If the "distance to training data" is mostly a property of the encoder's response to the test image, then claim 2 is validating one encoder quantity against another, and claim 1 is not about the training data.

## Strategy
Roll up our sleeves: take the metric apart and find out what it measures. Work in the tractable setting first — ShapeNet trials, twelve category-specific fine-tunes — where the training data and the stimulus geometry are both known, so any confound can be named.

## Next steps
- [ ] Reproduce Fig. 1 right from the stored margins and distances.
- [ ] Ask what the metric is bounded by, and what it correlates with besides the training set.
- [ ] Look for a way to measure distance to the training data that does not go through the encoder.

## Open questions
- Does the empirical shift measure the training set, or the encoder?
- Is there any distance that could be computed without the encoder at all?
- If the metric is wrong, does the human result (claim 3) survive?
