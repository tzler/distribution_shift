# The within-category question needs an experiment, not another metric
Every estimator is flat within a category once category identity is removed, for a structural reason: with one training set per category, distance-to-training and object atypicality are the same variable. So we vary the training set within a category and measure the margin.

*Snapshot: 2026-09-19 — the live state; the maintained version is [STATE.md](../STATE.md) · Lead: TB* · ← [previous](04-coverage.md) · [index](README.md) · [resources](../RESOURCES.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status
**The on-category curve is category identity.** Its bins sort by category — ">100 neighbours" is airplane/bench/car/lamp/telephone/watercraft, "0" is chair/table/sofa/cabinet/display/loudspeaker. Category-centred r = +0.013 (p 0.72); inside each category, mean r = +0.04; the twelve category points give r = +0.30 (p 0.34) with chair the counterexample. Seven coverage variants, including one calibrated to each category's own bank: all ≈ 0 [[D07](../REASONING.md#D07)].

![anatomy of the on-category curve](../evidence/fig57_oncat_anatomy.png)

**Why.** With one training set per category, "far from the chair training set" and "an unusual chair" are the same fact. The pretrained control rules out "hard for every model"; it cannot rule out category identity. The within-trial design separates training exposure from the stimulus by varying the training set — and that is exactly what the on-category row cannot do. Every estimator tried on it, including two exhaustive searches, gives the same answer, because the question is not identifiable there [[D03](../REASONING.md#D03), [D07](../REASONING.md#D07)].

**What we have, and what we don't.** Established: the margin responds to the training set (84% of trials, control zero by arithmetic); coverage is the right model-free ruler; the original metric measured the encoder's ease; support is view-invariant within the training grid. Not established: anything graded within a category. Without it, what we have shown is *category* shift.

## Strategy
Not a better metric — a finer-grained assay. Vary the training set *within* a category with the trial fixed, and ask whether the margin follows coverage:
- **Knockout**: for each test group, remove its k nearest training objects (k = 10 removes ~10–16% of the bank and raises the group's own kNN distance 3–4× more than other groups', 5× more than a size-matched random removal). 30 fine-tunes across chair/airplane/table; each trial ends up with 11 training sets.
- **Knock-in** from pretrained: 8 random subsets of 100 chairs, 6 targeted (the 100 nearest to a low-margin trial), 6 cross-category controls. The random runs make the metric comparison free — any candidate estimate is recomputed on the same runs, within trial — and the targeted runs test whether the metric is *actionable*.
- **Encoder-space upper bound**: features of all training renders under the fine-tuned models; if the chair model's own space shows nothing within category, no descriptor will [[D09](../REASONING.md#D09)].

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
