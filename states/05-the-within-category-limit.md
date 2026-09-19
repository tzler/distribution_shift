# Testing within a category means changing the training data, not the score
With one training set per category, no score computed on this data can tell "far from the chair training set" from "an unusual chair" — they are the same fact. So we are training new models on chosen subsets of a category and measuring whether the margin moves.

*Snapshot: 2026-09-19 — the live state; the maintained version is [STATE.md](../STATE.md) · Lead: TB* · ← [previous](04-coverage.md) · [index](README.md) · [resources](../RESOURCES.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status
**Where we were.** Counting nearby training data gave us a score that behaves across all trials at once, and within a trial we can show that moving a model's training data moves its margin. What was left was the finer question: inside one category, does the margin respond to *how much* of the category the model saw near this object — a graded relationship — or only to whether the category was in the training set at all? The plot that seemed to say "graded" (one point per trial, only the model trained on that trial's category) had a strong-looking fall along the count.

**That relationship is between categories, not within them.** The figure below shows the same plot, then the same plot with each category's own average subtracted from both axes. As measured, the margin rises with the count of nearby training objects. But the trials with few nearby training objects are chairs, tables, sofas and cabinets, and the trials with many are airplanes, cars, benches and telephones. Compare each trial only with its own category and the relationship is gone: inside every one of the twelve categories, count and margin are unrelated (r = +0.04; four of twelve categories go the other way). The twelve category averages, taken on their own, give r = +0.30 with p = 0.34 — twelve points, no conclusion. Seven variants of the count, including one calibrated to each category's own training set, all come out the same [[D07](../REASONING.md#D07)].

![One point per trial, before and after subtracting each category's own average: the within-category relationship is flat](../evidence/fig57_oncat_anatomy.png)

**Why no score can fix this.** The check we relied on so far — the pretrained model, which never saw the training set, must stay flat — rules out "this object is hard for every model". It cannot rule out "this category's model is good at this category". Comparing the twelve models on the same trial worked because the training set *varied*: twelve different training sets, one trial. Inside a category there is exactly one training set. "Far from the chair training set" and "an unusual chair" are then the same fact about the same object, and no score computed on these models can separate them. That is why two exhaustive searches for a better score found nothing: the question was not answerable with this data, however it was scored [[D03](../REASONING.md#D03), [D07](../REASONING.md#D07)].

**What is established and what is not.** Established: the margin responds to the training data (in 84% of trials the nearer training set gives the bigger margin, with the pretrained model flat within each trial); counting nearby training data is the score to use, and it needs no network; the manuscript's estimate was measuring the network's ease with the object, not its distance from the training set; within the ~25° spacing of the training views, viewpoint does not matter. Not established: any graded relationship inside a category. What we have shown so far is that the margin tracks *which category* a model was trained on. The encoder-space results of the second state remain a concern, not a kill: distance measured in a network's own features tracks the pretrained model's margin as well as the fine-tuned one on exactly these own-category trials, which is consistent with the same explanation [[D16](../REASONING.md#D16)].

**A later look back at the encoder's own space.** Once features of every training object under the fine-tuned models were extracted, we could ask the within-category question inside each model's own feature space, where the model has every advantage. There *is* a within-category signal there: after subtracting category averages, r ≈ −0.10 to −0.15 in the pretrained space, and −0.42 in the airplane model's own space on airplane trials. So the fine-tuned models do see structure inside a category that our 3-D-shape description does not — which says our description is too coarse, not that the question is closed [[D15](../REASONING.md#D15)].

## Strategy
Not a better score — a different experiment, in which the training set varies *within* a category while the trial stays fixed. Three parts, run on the cluster:
- **Knockout.** Take a group of test trials; remove from the chair (or airplane, or table) training set the training objects nearest to them; fine-tune a new model on what is left. Removing the ten nearest raises that group's own distance-to-training three to four times more than other groups', and five times more than removing the same number of random objects. Thirty fine-tunes across three categories; each trial then has eleven training sets that differ only in what was removed near it.
- **Knock-in.** Starting from the pretrained model, fine-tune on one hundred chairs: eight random subsets, six chosen to be the hundred nearest to a trial the model does badly on, six chosen from airplanes as a control. The random runs let any candidate score be compared on the same models; the targeted runs ask whether the score is *actionable* — can you pick training data that raises the margin on a chosen trial.
- **The ceiling.** Score distance inside the fine-tuned models' own feature spaces as well. If a model's own space shows a within-category effect that our shape description misses, the description needs to be finer [[D09](../REASONING.md#D09), [D15](../REASONING.md#D15)].

## Resources used
| | this state | cumulative |
|---|---|---|
| agent time | 7 h | 34 h |
| lead time (guess) | 4 h | 17 h |
| compute, unsub / sub | $8.50 / $1.90 | $25 / $6.50 |

Compute: the 1,895-candidate search ($0.09), the encoder-space extraction (1.7 GPU-h), the pilot fine-tune (1.7 GPU-h so far), the evaluation test. **Committed**: batch 1, 22 fine-tunes ≈ 44 GPU-h ≈ $200 / $44. **Held**: 12 targeted/cross knock-ins + 15 k = 50 knockouts ≈ 54 GPU-h ≈ $245 / $54. First new models of the project.

## Next steps
- [ ] Pilot fine-tune's epoch time → run budget; random knock-in subsets first.
- [ ] Encoder-space within-category test.
- [ ] Submit the runs; evaluate every checkpoint exactly as the 12 category models were.
- [ ] Single-trial figure: the trial, the training chairs the metric chose, pretrained vs targeted vs random margins.

## Open questions
- Is the within-category relationship graded once the training set varies?
- Which estimate best predicts margin *change* under intervention?
- What is the margin's noise floor, with no repeat seeds?
