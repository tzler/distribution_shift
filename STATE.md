# Distribution shift and the oddity margin
*Last meaningful update: 2026-09-21 · Lead: TB · Status: active* · history: [states/](states/README.md) · log: [REASONING.md](REASONING.md) · what we have: [RESOURCES.md](RESOURCES.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status
The short walk from the manuscript to now; each step has its own page in [states/](states/README.md) with the evidence at the time.

**1 · The manuscript's shift estimate measured the trial and the encoder, not the training set.** It could never be smaller than half the difference between the trial's two objects (the triangle inequality), and it was read off the same network whose margin it explained — so it predicted the *pretrained* model's margin with the sign backwards [[D06](REASONING.md#D06)]. [figure](evidence/step1_manuscript_metric.png)

**2 · Fix the way it is computed.** Average a per-image distance over all of a trial's images and never compare the two objects to each other. On the same descriptor, the manuscript's form *is* the trial's difficulty (r = +0.99); the new form is not [[D02](REASONING.md#D02)]. [figure](evidence/step2_fix_the_form.png)

**3 · Measure on the objects, not in a network.** Inside a network's own features the fix passes across all trials but not on a trial's own category, where the pretrained model's margin falls with distance just as the fine-tuned one does — a concern that never went away [[D15](REASONING.md#D15), [D16](REASONING.md#D16)]. So the distance moved to the objects' 3-D shape. [figure](evidence/fig62_encoder_space.png)

**4 · It works across categories; within a category the models we had could not tell.** Twelve category models on the same trial: the margin follows distance to the training set, the pretrained model flat because it is the same model on all twelve points (84 % of trials, permutation p = 10⁻⁴). Inside a category, with one training set per category, "far from the training set" and "an unusual object" are the same fact [[D02](REASONING.md#D02), [D07](REASONING.md#D07)]. [figure](evidence/step4_across_not_within.png)

**5 · Change the training data.** The agent's first design — remove ten training objects near a trial, test on MOCHI — showed nothing (a dose the training procedure cannot register; a test set selected to be hard). The lead's design — train on 25 objects of one kind, test on hundreds of trials built from held-out objects — showed the effect, and the distance predicts which model wins a trial without being told the kinds [[D18](REASONING.md#D18), [D21](REASONING.md#D21), [D22](REASONING.md#D22)]. In all twelve categories, the analysis of step 4 gives the same three pictures [[D29](REASONING.md#D29)]. [figure](evidence/fig76_round3_moving_training.png)

**6 · The margin is a relative measure, and "relative" comes from the training regime.** 92 % of a trial's margin is where pretraining left it; distance to the training data moves it from there, by about 0.02 across the whole range. Overwrite more of the prior — a larger learning rate, a full fine-tune, no pretraining at all — and the level's share falls from 85 % to 45 % to a correlation of 0.28 with the prior, while the effect of which objects the model saw more than doubles. Nothing absolute replaces the level: with no prior the margin follows trial difficulty and noise, not distance [[D31](REASONING.md#D31)–[D35](REASONING.md#D35)]. [figure](evidence/fig78_absolute_vs_relative.png)

**7 · The distance measure, searched for rather than assumed.** Sixty-nine candidates scored on held-out trials: within a category everything reasonable ties at about −0.23 and a seven-number bounding box reaches it; across categories only a frozen pretrained network's features order the training sets; comparing with the nearest few training objects predicts better than any comparison with the set as a whole at 25 objects. On the manuscript's own MOCHI images the frozen-network distance transfers best and the bounding box holds. With that distance, three models trained on different subsets fall along one continuum, and fine-tuning on distant data can make a trial worse than no fine-tuning [[D36](REASONING.md#D36)–[D38](REASONING.md#D38)]. [figure](evidence/fig81_gradient.png)

**Pending.** Trials built from each model's own training objects — the anchor at distance zero — are being scored (jobs in `../knockout/logs/batch3_jobs.txt`); every measure must put a model's own objects first.

## Strategy
Score the 69 measures on the training-object trials; a measure that fails the anchor is out whatever its held-out score. Write the paper's within-category figure from the continuum plot (full fine-tune, frozen-network distance, bounding box in the supplement). State the finding at two scales — graded within a category, a step across categories on geometry, one axis in a learned space — with the caveat that the fine-tuned models were initialised from that network. Then the human side: a human margin has the same structure, a level plus a small shift term, and the claim has to be within-trial or averaged.

## Resources used, and what the plan will cost
**Used.** Agent ≈ 68 h; lead ≈ 29–30 h (his estimate, revised as we go). Compute: 100 GPU-hours ≈ **$449 unsubsidised / $101 subsidised** across 89 jobs — 62 fine-tunes and the evaluations; the analysis phase before any training cost under $25. Ledger: `background/compute_ledger.csv`; per-state totals in each [states/](states/README.md) page.
**Committed.** Nothing beyond the queued evaluations (≈ $5).
**Rates.** B200: $4.51/h unsubsidised, $1.00/h subsidised; a fine-tune ≈ 1.5 GPU-h; the account cap is far away (≈ 10 M of 60 M billing-minutes used).

## Next steps
- [ ] Training-object anchor: score all 69 measures; a model's own objects must come first. (Claude)
- [ ] Paper figures: the continuum plot (D38), the twelve-category reproduction (D29), the ladder (D34), the search (D36). (Claude drafts; TB chooses)
- [ ] The two states not yet reviewed by the lead (7 and 8), then the demo repo and viewer rebuilt from the confirmed states. (TB, then Claude)
- [ ] Push the three repositories once created on GitHub. (TB creates; Claude pushes)
- [ ] The human side: the same level-plus-shift structure, stated and tested. (TB)

## Open questions
- How much of the frozen network's cross-category ordering is the prior the fine-tuned models inherited from it?
- Does the negative transfer at far distances appear in people?
- Why does our pipeline give 0.155 where the collaborator's gave 0.241 on the same chairs? (Only matters if old and new models are ever compared directly.)

## Pointers
- This repo: estimator `blind_shift.py`; coverage `scratch/coverage_sweep.py`; viewpoint `scratch/viewdepth_pipeline.py`; cited figures in `evidence/` with provenance; full factual record `README_findings.md`; walkthrough artifact https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r.
- Experiments: `../knockout/` (`design.json`, `design_knockin.json`, `scripts/`).
- Upstream: collaborator pipeline `../../Dist-shift/HIDA/hida-tune/` (read-only); category results `ShapeNet_OOD_Analyses/<cat>/ood_analysis_results.csv`; metric audit `../L1norm_vs_distshift/README.md`; the manuscript `Human-3D-generalization-copy/paper-to-follow-*/`.
- Background: `background/` — narrative back-fill, citation list, slot for the MOCHI project's STATE.md.
