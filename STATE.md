# Distribution shift and the oddity margin
*Last meaningful update: 2026-09-19 · Lead: TB · Status: active* · history: [states/](states/README.md) · log: [REASONING.md](REASONING.md) · what we have: [RESOURCES.md](RESOURCES.md)

## Goal
Find a way to use the oddity margin as a proxy for distribution shift — the distance between what a model was trained on and what it is tested on — and establish that it really is one: a shift estimate the margin tracks, computed in a way that cannot be gamed. The NeurIPS reviews questioned whether the manuscript's estimate measures shift at all. We are working out how much of that to take on board and how much to set aside, by testing rather than arguing.
## Status

**Where we started.** The manuscript claimed that a model's oddity margin — how confidently it picks the odd object out of three — tracks how far the test objects are from the model's training data, and that this holds for humans too. Its measure of "how far" was computed inside the same network whose margin it was explaining: a nearest-neighbour distance in the network's features, in a form that compares the trial's two objects to each training image. Three reviewers doubted that this measured distribution shift at all. Taking it apart, they were right: that form can never be smaller than half the difference between the two objects — the thing the oddity task is decided by — so it partly measures the trial's difficulty; in raw features it is mostly the size of the feature vector; and on our data it predicts the *pretrained* model's margin better than any fine-tuned model's, with the sign backwards. It measures how easy the trial is for the network, not distance from training [[D06](REASONING.md#D06)]. The figure below takes the estimate apart into those two pieces.

![What the manuscript's estimate is made of: the part that is the two objects' difference, and the part that is left](evidence/fig60_why_wrong.png)

**What we changed.** Two things. The estimate is now computed without ever comparing the trial's two objects to each other: for each of the trial's images, how far it is from the nearest training objects, averaged over the images. And it is computed on the objects' 3-D shape rather than in any network's features, so nothing in the measurement has learned anything [[D02](REASONING.md#D02)]. A network's own features remain a comparison, not the measurement: measured there, the estimate leaves the pretrained model flat across all trials, but on the trials that matter most — each trial scored by the model trained on its own category — the pretrained model's margin falls with the distance just as the fine-tuned one does. We record that as a concern that has not gone away, not as a kill [[D15](REASONING.md#D15), [D16](REASONING.md#D16)].

**Where it works.** Every one of 706 ShapeNet trials was scored by twelve models that differ only in which category they were fine-tuned on. Compare the twelve on the same trial and nothing about the trial can differ between them; only the training set does. The nearer training set gives the bigger margin in 84% of trials; shuffling training sets within trials 10,000 times never produces an effect this large; accuracy falls from 81% with the nearest training set to 56% with the farthest while the pretrained model stays at 51%. Moving the training data moves the margin [[D02](REASONING.md#D02)]. Plotted the ordinary way — all trials, one model at a time — every *distance* still fails the check that the pretrained model should be flat, because objects far from everything are hard for every model. *Counting* the training objects near the trial's objects, rather than measuring how far the nearest are, is the first score that passes it [[D05](REASONING.md#D05)]. In both figures below, the pretrained model should be flat if the score is about the training set.

![Moving the training data moves the margin: all trials pooled, then each trial's own average subtracted](evidence/fig46_moving_training.png)
![Three scores on the plot a paper would show: only counting nearby training objects leaves the pretrained model flat](evidence/fig56_pooled_three_metrics.png)

**Where it stops.** Inside a category. Ask whether a chair with fewer training chairs nearby gets a lower margin from the chair model, and the answer is no, for every score we tried, once each category's own average is subtracted. The one curve that looked graded was sorting trials by category [[D03](REASONING.md#D03), [D07](REASONING.md#D07)]. So far, then, we have shown that the margin tracks *which category* a model was trained on, not distance from training data as a continuous quantity. Below: the same plot before and after subtracting each category's own average — if the margin were graded within a category, the right panel would still rise.

![One point per trial, before and after subtracting each category's average: flat inside categories](evidence/fig57_oncat_anatomy.png)

**Why, and what we are doing about it.** With one training set per category, "far from the chair training set" and "an unusual chair" are the same fact about the same object; no score computed on these models can separate them. The comparison that worked above worked because the training set varied. So the current work is to make it vary *within* a category: train new models on chosen subsets of a category, with the trial fixed, and measure whether the margin moves [[D09](REASONING.md#D09)]. The fine-tuned models' own feature spaces do show within-category structure that our 3-D-shape description misses, so a finer description is also on the list [[D15](REASONING.md#D15)].

## The argument in six figures
1. [The manuscript's estimate measures the trial and the encoder](evidence/step1_manuscript_metric.png) · 2. [Never compare the trial's two objects to each other](evidence/step2_fix_the_form.png) · 3. [Measure on the objects, not in a network](evidence/fig62_encoder_space.png) · 4. [Works across categories, not within](evidence/step4_across_not_within.png) · 5. [Change the training data](evidence/fig76_round3_moving_training.png) · 6. [The margin is relative](evidence/fig78_absolute_vs_relative.png) · 7. Remove pretraining — running (D33).

## Strategy

Comparing the twelve models on the same trial worked because it varied the training set with the trial fixed. The within-category question needs the same move one level down: vary the training set *within a category*, with the trial fixed, and ask whether the margin follows coverage. Three tracks, all reusing the collaborator's fine-tuning pipeline unchanged (same backbone, LoRA, loss, epochs), so the new models are comparable to the twelve we have.

**1 · Support knockout** — does removing a trial's support lower its margin? For chair, airplane and table: split each category's test objects into four random groups; for each group and k ∈ {10, 50}, fine-tune on the category minus every group member's k nearest training objects. k = 10 removes ~10–16% of the bank and raises the group's own distance to its nearest training objects 3–4× more than other groups' and 5× more than a size-matched random removal (two random-removal controls per category). 30 fine-tunes; each trial ends up scored under 11 training sets spanning Δdistance 0 to +0.2, stimulus fixed. Prediction: the margin drops for own-group knockouts in proportion to coverage lost, and not for other-group or random removals of the same size. Batch 1 running (pilot 8519673 + 21 jobs, k = 10 and random controls; `../knockout/logs/batch1_jobs.txt`); k = 50 held [[D14](REASONING.md#D14)].

**2 · Knock-in** — does adding the right data raise it, and can the metric choose the data? From pretrained DINOv2-L, fine-tune on 8 random subsets of 100 chairs, 6 targeted subsets (the 100 nearest chairs to each of six low-margin chair trials; 2–3× closer than any random subset), and 6 cross-category controls (the 100 nearest airplanes). The random runs make the metric comparison free: every candidate estimate — the counting radius, how many nearest objects to use, the shape description, objects vs images — is recomputed on the same runs, within trial, and the one that best predicts margin *change* wins. The targeted runs then test whether that metric is actionable. 20 fine-tunes, designed in `../knockout/design_knockin.json`; the single-trial figure (the trial, the chairs the metric chose, pretrained vs targeted vs random margins) is the intended headline.

**3 · Encoder-space upper bound** — is the within-category signal there at all? Features of all 311k training renders and the MOCHI images under pretrained DINOv2-L and the chair/airplane/table fine-tunes (job 8519397). If the chair model's own representation shows no within-category distance→margin relation, no descriptor will and the descriptor search is over.

**Budget and order.** One fine-tune is ~2 GPU-hours including evaluation (~3 min/epoch): $9 unsubsidised / $2 subsidised on a full B200; all 50 runs ≈ $450 / $100 [[D12](REASONING.md#D12), [D14](REASONING.md#D14)]. Order: random knock-in subsets → knockout k = 10 → targeted and cross knock-ins → knockout k = 50. Training hyperparameters stay pinned; each condition gets its own similarity table so mining fills every epoch; evaluation is chained into each job.

## Resources used, and what the plan will cost
**Used so far.** Agent ≈ 34 h; lead ≈ 12.5 h (a guess: reviewing, dictating direction, reading the artifact and these files). Compute: 4 GPU-h + 250 CPU-core-h ≈ **$25 unsubsidised / $6.50 subsidised** — the entire analysis phase, Acts 1–8, cost under $10; the rest is today's extraction and pilot. Ledger: `background/compute_ledger.csv`.

**The plan as designed** (one fine-tune ≈ 2 GPU-h incl. evaluation; rates in [RESOURCES.md](RESOURCES.md)):

| | runs | GPU-h | $ unsub | $ sub |
|---|---|---|---|---|
| batch 1 — knockout k = 10 + random controls + random knock-ins (submitted) | 22 | 44 | 200 | 44 |
| batch 2 — targeted and cross-category knock-ins | 12 | 24 | 108 | 24 |
| batch 3 — knockout k = 50 (held pending batch 1) | 15 | 30 | 135 | 30 |
| **core plan** | **49** | **98** | **≈ 440** | **≈ 100** |
| optional: second knock-in dose N = 25 | 8 | 16 | 72 | 16 |
| optional: epochs ablation (10 vs 30) | 2 | 4 | 18 | 4 |
| optional: targeted runs with the runner-up metric | 6 | 12 | 54 | 12 |
| optional: second seed for the 12 category models (noise floor) | 12 | 24 | 108 | 24 |
| **everything** | **77** | **154** | **≈ 700** | **≈ 155** |

The core plan is ~6M billing-minutes, 10 % of the account's annual cap (17 % used before it); everything is ~16 %. Time to finish the core plan: ≈ 10–20 h wall at 6–8 jobs in parallel, plus ≈ 10–15 agent-hours of analysis and figures and ≈ 3–5 lead-hours of review.

## Next steps
- [x] Pilot epoch time → run budget: ~3 min/epoch, all 50 runs affordable [D12]. (Claude)
- [x] Encoder-space within-category test: signal present (−0.10 to −0.15 category-centred; airplane −0.42) [D15]. (Claude)
- [ ] Finer geometric descriptors aimed at the within-category signal; encoder-space x's in the knock-in metric comparison. (Claude)
- [x] Build subset dirs and per-condition similarity tables; verify chained evaluation; submit batch 1 [D14]. (Claude)
- [ ] Route targeted/cross knock-ins by the mig45 timing; decide on k = 50 from the k = 10 results. (Claude)
- [ ] Analysis script for knockout and knock-in results (within-trial regressions of Δmargin on Δcoverage; own vs other vs random). (Claude)
- [ ] Metric comparison on the random knock-in runs; targeted runs with the winner and runner-up. (Claude)
- [ ] Single-trial figure. (Claude)
- [ ] Lab meeting Monday: walk [states/](states/README.md) 0 → 5. (TB)
- [ ] Decide the resubmission's framing — claim category coverage now, or wait for the within-category result; r64w's reframing advice is the live constraint [[D11](REASONING.md#D11)]. (TB)

## Open questions
1. Is the within-category relationship graded once the training set varies?
2. Which estimate best predicts margin *change* under intervention?
3. ~~Does the encoder's own space carry a within-category signal that geometry misses?~~ Yes — category-centred −0.10 to −0.15, and −0.42 in the airplane model's own space on airplane trials [[D15](REASONING.md#D15)]. So: which finer geometric descriptors recover it?
4. What is the margin's noise floor? One model per category, no repeat seeds; the 0.13 within-category ceiling could be noise.
5. Does view-invariance hold beyond ~25° from the training grid? (Needs new renders.)

## Pointers
- This repo: estimator `blind_shift.py`; coverage `scratch/coverage_sweep.py`; viewpoint `scratch/viewdepth_pipeline.py`; cited figures in `evidence/` with provenance; full factual record `README_findings.md`; walkthrough artifact https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r.
- Experiments: `../knockout/` (`design.json`, `design_knockin.json`, `scripts/`).
- Upstream: collaborator pipeline `../../Dist-shift/HIDA/hida-tune/` (read-only); category results `ShapeNet_OOD_Analyses/<cat>/ood_analysis_results.csv`; metric audit `../L1norm_vs_distshift/README.md`; the manuscript `Human-3D-generalization-copy/paper-to-follow-*/`.
- Background: `background/` — narrative back-fill, citation list, slot for the MOCHI project's STATE.md.
