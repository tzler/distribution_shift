# Counting nearby training data instead of measuring distance to it
A score that counts how many training objects sit near the test objects is the first one that behaves across all trials at once — the model that never saw the training data stays flat — and the viewpoint from which the training images were rendered turns out not to matter.

*Snapshot: after the coverage result · Lead: TB* · ← [previous](03-a-model-free-ruler.md) · [index](README.md) · [resources](../RESOURCES.md) · next → [next](05-the-within-category-limit.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status
**Where we were.** We had a distance measured on the objects' 3-D shape, with no neural network in it, and a way of showing that moving a model's training data moves its oddity margin: compare the twelve category-trained models on the very same trial, so that nothing about the trial can differ between them. But that comparison only works within a trial. Plotted the ordinary way — every trial against every model, the plot a paper would show — every distance we had tried still failed the same check: the pretrained model, which never saw any of the training sets, had smaller margins on trials that were far from them. The distance was still partly reporting how unusual the objects are, and unusual objects are hard for every model.

**A different kind of score.** All of those distances asked *how far away* the nearest training objects are. The alternative is to ask *how many* training objects are nearby: draw a fixed radius around each of the trial's objects (in the same 3-D-shape description as before), count the training objects of that category inside it, and turn the count into a score that grows as the count shrinks. We call this coverage. The important difference is at the far end. A distance keeps growing for objects that are far from everything, so the most unusual objects get the most extreme scores. A count stops at zero — "no training objects nearby" is the most shift there is, and an object that is a little farther still gets the same score. That is what keeps how-unusual-the-object-is from spreading along the axis [[D05](../REASONING.md#D05)].

**What it does.** The figure below puts three scores side by side on the plot a paper would show — all 706 trials, all twelve models, nothing subtracted. In each column the top panel is the fine-tuned model, which should have a smaller margin as shift grows, and the bottom panel is the pretrained model on its own axis, which should be flat if the score is about the training set. Counting nearby training objects (left) is the first score whose bottom panel is flat. Distance to the nearest training objects (middle) still falls in the bottom panel. The manuscript's own estimate (right) does something worse: its bottom panel *rises* — the pretrained model is more confident on trials the estimate calls farther, because that estimate was computed from the same network's features [[D06](../REASONING.md#D06)]. The radius was chosen on one half of the trials and tested on the other, twenty times; the count won every time, so this is not a tuned coincidence.

![Three scores on the plot a paper would show: only counting nearby training objects leaves the pretrained model flat](../evidence/fig56_pooled_three_metrics.png)

**Does the viewpoint matter?** A network is trained on images, not on objects, and each training object was rendered from fifteen fixed camera positions. So perhaps what a model learns is view-specific, and coverage should count training *images* that look like the test image, from the test's own viewpoint, rather than training objects with a similar shape. We built this without a renderer or a network: each object's 3-D shape is projected to a depth map from a chosen camera, and the test images' cameras were recovered by matching silhouettes (they match the true silhouette 86% on average). Below, coverage is counted three ways, comparing the twelve models within each trial as before. Counting from the object's shape and counting from training images at the nearest training camera give the same curve. Counting from the test's actual camera — which sits between the training cameras, typically about 20° off — is weaker, and it stays weaker on the trials where the camera was recovered well, so this is not a pose-estimation error. Within the roughly 25° spacing of the training views, viewpoint does not matter [[D08](../REASONING.md#D08)].

![Coverage counted from the object's shape, from training images at the nearest training viewpoint, and from training images at the test's actual viewpoint](../evidence/fig58_viewdepth_ladder.png)

Reviewer r64w had asked what an object-based measure averaged over all views would do. This is that measure, and it does at least as well as the view-specific one: the reviewer's question turned out to be the right test.

**A correction carried forward.** In the first state we reported that the manuscript's estimate agreed with a shape-based distance on the ShapeGen objects (|r| = 0.74). Once the estimate is computed without comparing the two objects to each other, that agreement drops to 0.12: what had agreed was the difference between the two objects, which both estimates contained. The earlier claim is withdrawn [[D06](../REASONING.md#D06)].

**What is still open.** Restricting to one point per trial — only the model trained on the trial's own category — the count seems to track the margin in a graded way (a binned correlation of −0.78). If that holds up, it would mean the margin responds to *how much* of the category the model saw near this object, not just to whether the category was in the training set at all. But a correlation across trials from twelve different categories can come from the categories differing from each other, not from anything inside a category. That has to be tested before it is claimed.

## Strategy
Take the one-model-per-trial result apart: subtract each category's own average from both axes, and look inside each category separately. Only if a relationship survives inside categories is it a within-category result.

## Resources used
| | this state | cumulative |
|---|---|---|
| agent time | 9 h | 27 h |
| lead time (guess) | 3 h | 13 h |
| compute, unsub / sub | $0.20 / $0.10 | $16.50 / $4.60 |

Compute: coverage sweeps on the login node; the depth-map pipeline, six `genoa-std-mem` jobs. Still no new models.

## Next steps
- [ ] Category-centred and per-category versions of the on-category coverage curve.
- [ ] The strongest within-category test the data allow: pretrained margin partialled out of both sides.

## Open questions
- Is the on-category curve a within-category relationship, or category identity?
