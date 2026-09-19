# State 1 — What the manuscript's metric actually measures
*Snapshot after taking the metric apart* · ← [State 0](00-the-manuscript.md) · [index](README.md) · next → [State 2](02-a-form-that-escapes-the-bound.md)

## Goal
Same as State 0: a distance between train and test data that predicts the margin. Now with a specific worry — that the manuscript's version is a measure of the encoder's ease on the trial, wearing the wrong name.

## Status
The metric is ½[d(A,C) + d(B,C)] over training images C, with A the matched object and B the oddity, d the ℓ₁ distance in the encoder's raw features, C the training image minimising the sum. Three things are wrong with that number, and each lets it be large without the trial being far from the training data.

**1 · It contains the trial's own difficulty.** For any distance, the direct route is never longer than a detour: d(A,B) ≤ d(A,C) + d(C,B). Halve it: the shift can never be smaller than ½·d(A,B) — the difference between the two objects, which is what the oddity task is decided by. The floor is reached only if a training image lies exactly on the path from A to B, so shift = ½·d(A,B) + excess, and only the excess is about the training set. For L2 the level sets are ellipses with A and B as foci; move the objects apart and every ellipse grows, training images included. In data: the floor holds in 100% of trials, r(shift, d(A,B)) = 0.73, and half the shift's variance is the floor [../REASONING.md#D06].

**2 · It is mostly the size of the feature vector.** r(shift, ‖φ‖₁) = 0.95 for ResNet-50 [../REASONING.md#D01].

**3 · It is validated against itself.** The proxy is the encoder's margin; the target is computed in the same encoder's features; and the encoder's own d(A,B) — the floor — *is* the margin (r = +0.57 across all 2,019 trials). Decomposed: the floor predicts the pretrained margin at +0.57 (wrong sign), the excess at −0.17 (right sign), and their sum lands at +0.32 [../REASONING.md#D06].

![why the metric is wrong](../evidence/fig60_why_wrong.png)
![the triangle inequality spelled out; cosine and L2 do not escape it](../evidence/fig61_triangle.png)

**In hindsight, one review reads differently.** SCwg's "the margin is inter-class distance, and every result survives the renaming" may be exactly this: the floor is the inter-object distance, and it is the part that carries the correlation. At the time that sentence read as a naming quibble. It was not.

**Does cosine or L2 get around it?** No — the bound is a property of every metric. Recomputed under three distances the floor holds in 100% of trials for all three and cosine is slightly worse (r with d(A,B) 0.78). Normalising the features fixes the norm problem and leaves the floor exactly where it was.

## Strategy
Fix the *form* first: never difference A and B. Average a per-image quantity over the trial's images, so the task's decision variable cannot enter. Test whether that alone is enough, in the encoder's own space.

## Next steps
- [ ] Build the oddity-blind estimator (mean over images of distance to the k nearest training items).
- [ ] Check its correlation with d(A,B) and with the pretrained margin in encoder space.
- [ ] If the encoder's space still leaks, leave it.

## Open questions
- Is the form the whole problem, or does the space contribute too?
- Where does the genuine excess signal (−0.17) go once the floor is removed?
