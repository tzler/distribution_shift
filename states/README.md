# states/ — the project as a sequence of snapshots

Each file is STATE.md as it would have read at that moment: goal, status with the evidence
we had, strategy, next steps, open questions. Written retroactively on 2026-09-19 for the
lab walkthrough; the reasoning entries they cite carry the actual dates. Click forward.

| | state | the turn |
|---|---|---|
| 0 | [Re-evaluating our distribution-shift measure after the NeurIPS reviews](00-the-manuscript.md) | submitted; three reviewers doubt the word "distribution shift"; the metric lives inside the encoder |
| 1 | [What the manuscript's shift metric actually measures](01-what-the-metric-measures.md) | bounded by ½·d(A,B); feature norm; validated against itself |
| 2 | [A shift estimate that is not bounded by the trial's difficulty](02-a-form-that-escapes-the-bound.md) | oddity-blind; helps but the encoder's space still leaks |
| 3 | [Measuring distribution shift on the objects themselves, without an encoder](03-a-model-free-ruler.md) | distance on 3-D shape; comparing the twelve models on the same trial; moving the training data moves the margin |
| 4 | [Counting nearby training data instead of measuring distance to it](04-coverage.md) | a count leaves the pretrained model flat across all trials; the training images' viewpoint does not matter |
| 5 | [Testing within a category means changing the training data, not the score](05-the-within-category-limit.md) | the graded-looking curve is categories differing; new models trained on chosen subsets |

Each snapshot ends with the resources used up to that point (agent time, compute, dollars). What was available throughout is in [RESOURCES.md](../RESOURCES.md).

The live page is [STATE.md](../STATE.md); the log is [REASONING.md](../REASONING.md).
