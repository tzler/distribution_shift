# Measuring distribution shift on the objects themselves, without an encoder
We stopped measuring distance inside a neural network's feature space and measured it on the 3-D shapes of the objects instead. With that ruler, and a way of comparing models on the very same trial, the oddity margin does track how far a model's training data is from the test objects.

*Snapshot: after the within-trial result · Lead: TB* · ← [previous](02-a-form-that-escapes-the-bound.md) · [index](README.md) · [resources](../RESOURCES.md) · next → [Coverage](04-coverage.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status
**Where we were.** The previous step left us with two problems in the manuscript's shift estimate. One was its form — it was bounded by how different the two objects in a trial are, so it partly measured the trial rather than the training set — and we had a fix for that. The other was the space it was computed in: a neural network's features. Any distance measured there is a property of that network, and there was a persistent worry that "far from the training set" and "looks unusual to the network" were the same thing wearing two names. The only way to be sure a distance is about the training data is to compute it from something that is not a network.

**What we built.** The test stimuli are ShapeNet objects, and so is the training data, so we have the 3-D shape of every object involved. We turned each shape into a numerical description with no learned parameters in it — a grid of which cells of a 16 × 16 × 16 box the object occupies, plus a 57-number summary of the shape that does not depend on how it is rotated [[D01](../REASONING.md#D01)]. Distance from a test trial to a training set is then: for each of the trial's objects, how far is it (by these numbers) from the nearest fifty objects in that training set; average over the trial's objects [[D02](../REASONING.md#D02)]. Nothing in that ruler knows which object is the odd one out, and nothing in it has seen a network.

**What we saw first, and why it was not enough.** Plotting every trial against every model — 706 trials, each scored by twelve models fine-tuned on a different ShapeNet category — the fine-tuned margin fell cleanly as the distance grew. But so did the margin of the pretrained model, which never saw any of the twelve training sets (left panel below). That cannot be a training effect. It means some objects are simply far from everything — unusual shapes — and unusual shapes are hard for every model. The distance was still partly reporting which trials are hard [[D02](../REASONING.md#D02)]. This is, we now think, what reviewer r64w was describing: a correlation that appears because both quantities respond to how unusual the example is. Their suggested check — two random halves of the same data should give no difference — is a version of the check we were applying.

**The comparison that settles it.** Because every trial was scored by all twelve models, we can compare the twelve models *on the same trial*. Within one trial nothing about the stimulus changes from model to model — same images, same objects, same pretrained margin, same human data; we checked that these are literally constant. The only thing that differs between the twelve scores is which training set the model saw. So subtract each trial's own average and look at what is left (right panel below): the pretrained model is now flat by construction, and the fine-tuned margin still falls. In 84% of trials the nearer training set gives the bigger margin; shuffling training sets within trials 10,000 times never produces an effect this large; a bootstrap over the twelve training categories keeps the effect away from zero [[D02](../REASONING.md#D02)]. Moving the training data moved the margin.

![Moving the training data moves the margin: left, all trials pooled — the pretrained model falls too; right, each trial's own average subtracted — the pretrained model is flat and the fine-tuned model still falls](../evidence/fig46_moving_training.png)

**The same thing seen two other ways.** Rank each trial's twelve training sets from nearest to farthest and the margin steps down along the ranking; the model trained on the trial's own category sits far above the other eleven. Accuracy tells the same story: 81% correct when the nearest training set was used, 56% when the farthest, with the pretrained model at 51% throughout.

![Rank each trial's twelve training sets from nearest to farthest: the margin falls along the ranking](../evidence/fig9_margin_rank.png)
![Accuracy from nearest to farthest training set; the pretrained model is flat](../evidence/fig48_accuracy_within_trial.png)

**Does the choice of ruler matter?** We tried seven ways of centring, eight shape descriptions (finer and coarser grids, the rotation-invariant summary alone, single numbers such as volume), and nine ways of turning distances into a score. The curve is the same every time. A cross-validated search over 1,895 candidate rulers could not beat the simplest one [[D03](../REASONING.md#D03)]. We take that as good news: the result is not about a particular descriptor.

**What is still open.** Two things. First, the left panel — the one a paper would naturally show, one model at a time — still fails the pretrained-model check; the within-trial comparison rescues the conclusion but not that plot. Second, the ranking looks mostly like a step: the model trained on the trial's own category is far above the rest, and beyond that the curve is fairly flat. Is there anything graded in the margin beyond "was this category in the training set", or is that all there is?

## Strategy
Look for a way of scoring distance whose pooled plot passes the pretrained-model check on its own — a score that counts how much training data sits near the test objects, rather than how far the nearest training objects are, is the obvious candidate. And look directly at the on-category row: one point per trial, only the model trained on that trial's category, to see whether the margin varies with distance inside a category at all.

## Resources used
| | this state | cumulative |
|---|---|---|
| agent time | 13 h | 18 h |
| lead time (guess) | 2 h | 10 h |
| compute, unsub / sub | $3.70 / $1.20 | $16.30 / $4.50 |

Compute: descriptor banks (`hida_bank` 114 core-h), representation variants, pose features; the 10k-permutation analysis on the login node. All results from the 12 inherited fine-tunes.

## Next steps
- [ ] An estimate whose pooled relationship passes the base-DINOv2 control.
- [ ] The on-category row: one point per trial, only the model trained on the trial's own category.

## Open questions
- Is the within-trial effect "which category" or a graded function of distance?
- Why does every pooled metric leak object atypicality, and can a different estimator stop it?
