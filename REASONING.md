# REASONING — Distribution shift and the oddity margin
Append-only; newest entry at the top. Entries are true at their date; supersede, never edit.
Cited from STATE.md as [Dxx]. Entries before 2026-09-18 are back-filled and marked (recalled).

Template — one line each:
**State:** the general situation at the time · **Observation:** the specific thing being engaged with ·
**Decision:** what we concluded or chose · **Because:** the reasoning · **Rejected:** alternatives and why not ·
**Implication:** what changes / next steps ·
**Steering:** who drove it — lead / agent / joint (note when the lead is deferring) ·
**Confidence:** one line per contributor, by role and identity — `lead (TB)`, `agent (Claude Opus 5)` — low / medium / high, and what they are unsure of.

<a id="D40"></a>
## D40 — 2026-09-24 — The anchor passes: a model's own training objects get its biggest margins, every measure ranks them nearest, and the frozen-network distance tracks the margin best on them
**State:** The 42 training-object evaluations queued on 21 Sep completed; this was item 1 of the handoff list.
**Observation:** 2,700 oddity trials built from the 25 training objects of each of 35 models, every model scored on all of them (`data/anchor_long.csv.gz`, evidence/fig82). The model that trained on a trial's objects gives margin +0.136; another model of the same category +0.103; a model of another category +0.076; the pretrained model +0.063. The owner gives the biggest margin on 34 % of trials against a 3 % chance rate, and sits +0.057 above its trial's average. All three measures place the owner first by distance (mean rank 1.00 of 35) — as they must. Within-trial correlation between distance and margin, with the anchor included: frozen pretrained network −0.333, 16³ voxels −0.202, bounding box −0.151; excluding the owner: −0.238, −0.108, −0.104.
**Decision:** The anchor is passed by every measure, so nothing is eliminated by it; but it separates them by degree, and it separates them the same way the MOCHI transfer did — the frozen-network distance first, the bounding box last. Report the two rankings together: on held-out trials the bounding box ties the best (D36); on training objects and on a different render pipeline it does not (D37, D40).
**Because:** The anchor is the one point where the true distance is known (zero), so a measure's behaviour there is a fact rather than a fit; and the memorisation effect is large enough to be visible per trial, unlike the graded effect.
**Rejected:** Treating the anchor as a filter (it excludes nothing); reading the bounding box's weaker showing here as its failure — it was never asked to order objects a model had literally seen.
**Implication:** The continuum plot (D38) can now be drawn with its left end: distance 0, the memorised objects. Remaining: fold rung 3b into the ladder table; the lead's review of states 7–8.
**Steering:** agent ran the queued analysis; the design (training-set trials, "there is no reason to treat train and test as different for our purposes") was the lead's.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D39"></a>
## D39 — 2026-09-22 — Packaged for a collaborator: the experiments are now in the repo, the analysis runs from a fresh clone, and the viewer had two real bugs
**State:** The lead wants to share the repositories with a collaborator and test whether the framework is actually usable by someone else.
**Observation:** The project repo held the analysis (`scratch/`, 53 scripts) but not the experiments — the 31 condition-builder / evaluator / SLURM scripts lived in a sibling directory in no repository, so nobody could rerun anything. Analysis scripts also hard-coded our share's absolute path and read bulk tables from `out/`, which is git-ignored, so a clone could not run them. The viewer contained two genuine bugs found by reading it: in demo mode the commit list is reversed twice (the nine states play backwards), and if the CDN copy of `marked` is unavailable every pane stays blank — which is what the lead saw at the lab meeting.
**Decision:** Add `experiments/` (scripts, designs, trial sets, job lists, and *patches* rather than copies of the collaborator's read-only pipeline) and `data/` (the tables the analysis consumes, 9 MB incl. the 465k-row trial × model table); make every `scratch/` script resolve the repository from its own location, fall back to `data/` when `out/` is absent, and create its output directory. Rewrite the root README for a first-time reader, add `REPRODUCE.md` (figure → script → data for all 35 cited figures) and READMEs for `experiments/` and `data/`. Fix both viewer bugs and add `bootstrap.sh` to the template so the system can be adopted in one command.
**Because:** "Reproducible" is a claim about someone else's machine. The test was a fresh clone with no access to the share: three advertised scripts now run there and produce their figures.
**Rejected:** Shipping the collaborator's training and evaluation code (patches instead — 28 and 8 lines); committing the 74 MB analysis table (slimmed and gzipped to 6.9 MB); moving the first-phase scripts out of the repository root (`scratch/` imports several of them — documented the layout instead).
**Implication:** The framework's first real test is a collaborator using it. What is still untested: the viewer in a browser we do not control, and whether the trace is legible to someone who was not here — the thing the whole system claims.
**Steering:** lead asked for the packaging and the test; agent did both.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high that the analysis reproduces from a clone (tested); medium on the viewer (two bugs fixed by reading, not yet seen working in a browser).

<a id="D38"></a>
## D38 — 2026-09-21 — The continuum plot: with the frozen-network distance, three models trained on different subsets lie on one curve; the effect is largest and cleanest under full fine-tuning; far training data can cost
**State:** The lead asked for the direct plot — models trained on different subsets, best measure on x, margin on y — to see whether items at the same distance get the same margin whichever model they come from. (He had also mis-read D36 as scored on the training set; corrected in D37.)
**Observation:** 882 held-out chair trials × the three 25-chair cluster models, y = fine-tuned − pretrained on the same trial (evidence/fig81). DINOv2 / nearest: under gentle LoRA the three curves overlap and fall from ~+0.10 to +0.02–0.04 (within-trial r −0.36); under full fine-tuning they fall from ~+0.18 to ~0 along one continuum (within-trial r −0.23, pooled −0.21), and the tall-narrow-backed model goes *below zero* on the farthest trials — fine-tuning on distant data made those trials worse than the pretrained model. Bounding box / nearest: each model's curve falls, but the models' distance ranges differ and the curves do not align on a shared axis.
**Decision:** This is the paper's within-category figure: full-fine-tune rung, DINOv2-nearest on x, gain on y, three models on one continuum, with the bounding-box version in the supplement as the fully model-free confirmation of the slope. Note the negative-transfer tail explicitly.
**Because:** It is the lead's question answered in the form he asked for; the continuum is what "the margin tracks distance" should look like; and it is the frozen-network measure that makes the models commensurable, which is the D36/D37 finding restated visually.
**Rejected:** The gentle rung as the headline (smaller effect, and the nearest bin dips for two models — the D26 confound at the near end); the bounding box as the headline (falls, but no shared axis).
**Implication:** With the anchor trials (training objects, distance ≈ 0) the continuum gets its left end; the negative tail under full fine-tuning is a testable prediction for the human side (learning far data should cost on near trials).
**Steering:** lead specified the plot; agent built it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D37"></a>
## D37 — 2026-09-21 — Transfer test: the search's measures applied to MOCHI (unseen images, different render pipeline) — the frozen-network distance transfers best, the bounding box holds, voxels degrade
**State:** The lead asked that measures chosen in the toy setting be tested on images the models never trained on; he had read D36's ranking as scored on the training set (it was scored on held-out trials — the training-object trials are still being evaluated).
**Observation:** 706 MOCHI single-category trials × 34 cluster models, within-trial score as in D36 (`out/transfer_mochi.csv`). Within a category: DINOv2 / nearest −0.259 (bank held-out: −0.236), bounding box / nearest −0.173 (−0.232), voxel16 / nearest −0.091 (−0.212), structure −0.066 (−0.186). Across categories: DINOv2 −0.16 (unchanged), voxel16 −0.11, structure −0.11, bbox −0.06. All models: DINOv2 −0.35, voxel16 −0.22, structure −0.19, bbox −0.15. DINOv2 on MOCHI uses the test image's own features (image level), which carries the pipeline.
**Decision:** Report both levels. The within-category result is robust to the render pipeline with the crudest geometry (bbox) and with the frozen-network distance; the finer geometric descriptors were partly fitting the bank's rendering. Recommend the paper's within-category figure use bbox/nearest (model-free, transfers) with DINOv2/nearest beside it (transfers best, with the initialisation caveat of D36).
**Because:** A measure's job is to predict margins on images the model has not seen from a pipeline it was not trained on; the ranking on the bank alone over-credited voxels and structure.
**Rejected:** Treating MOCHI as the last word (76 chair trials per category; the bank held-out set is ten times larger — the two rankings together are the evidence).
**Implication:** The level the lead expects to fail — an internet-scale pretrained encoder with no measurable training set — cannot be tested model-free; that is the manuscript's original wall and where the human claim sits. Next: the training-object anchor (queued), then the one-axis figure with DINOv2 distance on x.
**Steering:** lead asked for the transfer; agent corrected the premise (held-out, not training) and ran it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D37"></a>
## D37 — 2026-09-21 — Transfer of the search's winners to MOCHI: the geometric measures degrade on the harder, selected trials; the frozen-network distance holds
**State:** The lead worried the search had been scored on the training set (it had not — the 11,634 trials are held-out halves; the training-object trials are still queued) and that measures tuned on a toy setting need not transfer. Test: the 34 cluster models on the MOCHI single-category trials — never trained on, a different render pipeline, the manuscript's selection of hard trials.
**Observation:** Within category, nearest-neighbour: DINOv2 pretrained −0.259 (bank −0.236), bounding box −0.173 (bank −0.232), 16³ voxels −0.091 (bank −0.212), structure −0.066 (bank −0.186). Across categories: DINOv2 −0.162 (bank −0.155), voxels −0.110, structure −0.111, bounding box −0.058. All models pooled: DINOv2 −0.346, voxels −0.220, bounding box −0.146. The object descriptors are identical across the two test sets (same assets); the trials differ — MOCHI's pairs are near-identical in shape (`out/transfer_mochi.csv`).
**Decision:** Report both levels. On unselected trials within a category, geometry suffices and the bounding box is the model-free statement. On the manuscript's trials and across categories, the frozen-network distance is the measure that transfers; carry its caveat (the fine-tuned models inherit that space).
**Because:** A measure that survives a change of trial set and pipeline is the one to trust one level further out; the geometric ones lose resolution exactly where the trials become hard, which is where the manuscript lives.
**Rejected:** Declaring the geometric measure sufficient on the strength of the bank-trial search alone.
**Implication:** The training-object trials (queued) give the anchor for both; the D28 one-axis figure should be redrawn with the DINOv2 distance on x to see whether within- and across-category points now share a line. The lead's deeper worry — none of this need hold for a large model pretrained on the internet — remains open and is not testable with these tools; say so in the paper.
**Steering:** lead — the transfer requirement and the scepticism; agent — the test.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the numbers; medium on the "resolution" explanation for the geometric drop.

<a id="D36"></a>
## D36 — 2026-09-21 — The distance-measure search: within a category everything reasonable ties near −0.23 and a 7-number bounding box is as good as any; across categories only a learned space orders the training sets; nearest-few beats every set-level comparison
**State:** The lead: we have the right margins now but only an assumed distance measure — search for it. 69 candidates (11 feature sets × up to 8 comparison rules, incl. view-specific depth maps at the test image's own view), each scored as the within-trial correlation between distance-to-the-model's-training-objects and the model's margin, fitted on odd trials and reported on even (evidence/fig80, `out/distance_search.csv`).
**Observation:** Within a category (own-category models): pretrained DINOv2 features / nearest −0.236, bounding box / nearest −0.232, DINOv2 / centroid −0.231, bounding box / 5-nearest −0.230, 16³ voxels / nearest −0.212, structure −0.207, view-specific depth −0.201, view-averaged depth −0.197, 57-d invariant −0.187, multiview −0.157, volatility −0.141. Across categories (other-category models): DINOv2 −0.155 / −0.148 / −0.139; every geometric feature ≤ −0.10 (most −0.05 to −0.08). Comparison rule, same order for every feature: nearest ≈ 5-nearest > 10-nearest ≈ centroid > mean-of-all > coverage at any radius (coverage −0.07 to −0.17 within, ≤ −0.03 across). Fit and held-out scores agree to ±0.01 everywhere: no selection overfitting. View-specificity does not help (−0.201 vs −0.197), consistent with D08.
**Decision:** (1) Within a category the measure is not the bottleneck: the attainable r with these margins is ≈ −0.23 whatever the descriptor, and a bounding box gets there — so the within-category claim can be made with the crudest model-free description, which is the strongest possible form of "model-free". (2) Across categories the geometric descriptors do not order training sets and the frozen network's features do; report that as the descriptor's limit, and offer the frozen-network distance as the cross-category ruler with its caveat (the fine-tuned models were initialised from it, so part of what it orders is the prior they inherited — not the manuscript's circularity, but not nothing). (3) Comparison: nearest-few, not coverage, at N = 25; say why (saturation, D24).
**Because:** The search was blind to labels and held out; the ranking is flat where the margin's noise sets the ceiling and steep only where the geometry gives up (between categories).
**Rejected:** Tuning further on the same trials (fit ≈ test says there is nothing to tune); declaring DINOv2 "the" measure (its within-category advantage over a bounding box is 0.004).
**Implication:** Score the same 69 on the training-object trials when they land (D37): every candidate must put a model's own objects at the top; that is the anchor the lead asked for and a second, independent ranking. The one-axis figure (D28) may be recoverable with the DINOv2 distance on x — check.
**Steering:** lead — the search and the two levels (features × comparisons); agent — construction and reading.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high; the held-out design makes these numbers safe to quote.

<a id="D35"></a>
## D35 — 2026-09-21 — From scratch: the prior-linked level goes (0.79 → 0.28), but nothing absolute replaces it; the model is weak and the single-training-set wall returns
**State:** The from-scratch ViT-small (random init, 2,000 chairs, 60 epochs) scored on the 11,634 bank trials; the lead's first interest.
**Observation:** Accuracy 0.51 on chair trials (chance 0.33) against 0.63 for the *pretrained* ViT-small that never saw a chair; margin +0.045 vs +0.064. r(its margin, pretrained ViT-S margin) = +0.28 on chair trials, +0.32 over all — the end of the ladder's 0.79 → 0.44 → 0.34. Against distance to its own 2,000 training chairs: r = +0.13 raw, +0.05 with pair distance held fixed, the same as the pretrained models that never saw the data (+0.17 / −0.02).
**Decision:** Report the ladder as a dose-response with the from-scratch rung as its end point: the level tied to pretraining vanishes without pretraining; what is left is trial difficulty and noise, not distance. Do not pursue from-scratch models on subsets (this run shows 30,000 images barely suffice; 25 objects cannot).
**Because:** A single from-scratch model has one training set, which is the D07 wall again — distance-to-training and unusualness are one variable — and the model is too weak for the residual to be read as anything.
**Rejected:** Longer from-scratch training (the trend from epoch 18 to 60 was 0.53 → 0.70 in training accuracy; the evaluation is still below the untrained-on-chairs prior); ViT-L from scratch.
**Implication:** The paper's statement: the margin reads distance relative to what the model already had; how much of the margin is "already had" is set by the training regime (85 % → 45 % across the ladder, ~0.28 correlation with the prior at the from-scratch end); an absolute reading is not available at any rung we could train. Rung 3b (ViT-S pretrained, full fine-tune) pending: it separates the small architecture from the missing prior.
**Steering:** lead asked for from-scratch first; agent ran and read it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the numbers; medium that a stronger from-scratch model would say the same (the wall argument says it cannot say more on one training set).

<a id="D34"></a>
## D34 — 2026-09-21 — The ladder's first two rungs: the trial's level shrinks as more of the prior is overwritten, and the effect of which objects grows
**State:** All eight ladder runs trained; the six ViT-L runs evaluated; the two ViT-small evaluations failed on a run-dir glob (fixed, re-queued 8568995–96).
**Observation:** Chair trials, three cluster models at N = 25 per rung. LoRA lr 1e-6: margin 0.146, 85 % of its variance is the trial's level, r(level, pretrained margin) +0.79, own-kind advantage +0.015. LoRA lr 1e-5: 0.208, 54 %, +0.44, +0.039. Full fine-tune lr 1e-5: 0.170, 45 %, +0.34, +0.037. In every rung r(level, distance to the training data) stays at +0.03 to +0.09 — the level that remains is trial difficulty, not distance. The from-scratch ViT-small reached training accuracy ~0.7 at epoch 60 (LoRA runs: 0.85–0.90 by epoch 30).
**Decision:** State the D31 result as a property of the regime: "relative" because the fine-tuning is a small perturbation of a large prior; overwrite more of the prior and the level's share falls while the effect of which data grows. Report the ladder as a dose-response, with the from-scratch rung as its end point once scored.
**Because:** Three rungs, one direction, on the same trials with the same analysis; and the prediction was written down before the runs (D33).
**Rejected:** Reading the higher margins at lr 1e-5 as "better models" (a different rung, not a better one; the relevant quantity is the split).
**Implication:** The paper's framing: the margin reads distance relative to what the model already had; how much of the margin is "already had" is set by the training regime; humans are at the far-prior end. The from-scratch rung tests whether, with no prior, the level itself becomes distance to the training data.
**Steering:** lead asked for the from-scratch test; agent designed the ladder and ran it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on rungs 1–2; the from-scratch rung pending.

<a id="D33"></a>
## D33 — 2026-09-20 — "Relative" may be a property of the fine-tuning regime: a ladder of how much of the prior is overwritten, ending in training from scratch
**State:** D31–D32 established the margin as relative — 92 % of its variance is where pretraining left the trial. The lead asked how that plays with the pretraining-then-fine-tuning regime, and called the from-scratch test "the single highest-powered thing we can do".
**Observation:** By pretrained-margin quintile: the gain from any fine-tuning is largest on the trials pretraining handled worst (+0.059 hardest fifth → +0.016 easiest, own-category models; +0.016 → +0.005 other-category), while the within-trial distance slope is flat across quintiles (−0.011, −0.011, −0.009, −0.008, −0.012): level and shift are additive. The manuscript's "random_init" feature sets are untrained encoders (its §5 control), and the from-scratch checkpoints on disk are ViT-small runs on co3d/primigen — no from-scratch ShapeNet category models exist, so this is training, not evaluation; and from scratch cannot be done at N = 25.
**Decision:** A ladder on chairs, each rung overwriting more of the prior: LoRA lr 1e-6 (have) → LoRA lr 1e-5 (3 clusters × 25) → full fine-tune lr 1e-5 (3 × 25) → from scratch, ViT-small, full 2,000-chair bank (pilot, 60 epochs, lr 3e-4) with a ViT-small pretrained full-fine-tune on the same bank as the architecture control. Eight runs, jobs 8557001–47, ≈ 12–20 GPU-h ≈ $60–90 / $15–20. All scored on MOCHI and on the 11,634 bank trials; the evaluator detects LoRA vs full state dicts.
**Because:** The prediction is sharp and cheap to test at the first two rungs: the share of margin variance that is "the trial's level, predicted by the pretrained margin" should fall rung by rung, and by the from-scratch rung a trial's level should correlate with distance to its training data rather than with a pretrained margin that no longer applies. If so, "relative" is a fact about small perturbations of a large prior, not about the margin — the answer the reviewers' question needs. If the from-scratch pilot does not learn (a small ViT on 30k renders), that is the finding for that rung, and rungs 1–2 still carry the dose-response.
**Rejected:** From scratch on 25 objects (cannot learn); ViT-L from scratch (will not converge on this data); evaluating the untrained "random_init" encoders as if they were trained models.
**Implication:** If the ladder works, extend rung 3 to airplane and table (the state-3 analysis without pretraining, ≈ $60 more). Analysis: the D31 variance split per rung, plus the between-trial r(level, distance) per rung.
**Steering:** lead — the question and the push for from-scratch; agent — the ladder design and the correction about what "random init" was.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on rungs 1–2 running; medium on rung 3 learning at all; medium-high on the prediction's direction.

<a id="D32"></a>
## D32 — 2026-09-20 — The direct version: hold one trial fixed, retrain at different distances, watch its margin — the effect is a population effect, invisible in one trial
**State:** The lead asked for the most direct test of absolute vs relative: one trial, models trained on data at varying distance from it, that trial's margin.
**Observation:** Each held-out trial has 34 models at distances 0.1–1.4 (evidence/fig79). In a single trial the 34 margins scatter by sd 0.02–0.03 and the distance trend is inside that scatter: Spearman ρ per trial ranges from −0.43 to +0.19 across the eight shown. Overlaying 300 trials relative to their pretrained margin, the mean runs from +0.023 at the nearest training sets to +0.006 at the farthest — and stays above zero: a model fine-tuned on anything raises a trial's margin a little, and nearby data raises it ~0.02 more. The designed version on MOCHI: shapenet119, pretrained −0.069 → nearest 25 chairs +0.032, random −0.002, farthest −0.044 (monotone, 0.08 across the range); shapenet262, pretrained −0.105 → nearest −0.074, random −0.102, farthest −0.032 (not monotone; the farthest set did best).
**Decision:** Say plainly: the margin is a relative measure, and a noisy one. One trial under one model does not give a distance; one trial under many models gives a trend that is inside the model-to-model scatter; many trials give the trend cleanly. The claim in the paper is about the population of trials, and should be stated as such — the single-trial "banger" is one of two targeted trials, and the other goes the wrong way.
**Because:** The honest scale of the effect is ~0.02 of margin over the full range of distance against a per-trial, per-model scatter of ~0.02–0.03; no amount of design changes that ratio, only averaging does.
**Rejected:** Presenting shapenet119 alone (a 1-in-2 result); increasing the number of targeted trials as the next spend (the population result already carries the claim; the targeted design's value is illustrative).
**Implication:** For the human side this is the constraint that matters: a human margin on a trial has the same structure (level + small shift term + noise), so human claims need the same within-trial or averaged design. Also worth noting in the text: even far-away training data lifts the margin above the pretrained level (+0.006) — the domain effect never fully disappears at these distances.
**Steering:** lead asked; agent built.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D31"></a>
## D31 — 2026-09-20 — The margin is a relative measure of distance to the training set, not an absolute one
**State:** The lead, looking at the un-normalised own-category panel (fig77, "up and down and up and down"), asked whether the margin gives an absolute or a relative measure of distance — a question the reviewers had circled.
**Observation:** 11,634 trials × the three models of each trial's own category. 92 % of the margin's variance is between trials (the trial's level), 8 % within (which training set). The level is the pretrained margin on the same trial (r = +0.82; +0.96 over all 34 models) and is not the trial's distance to the training sets (r = +0.07). Within a trial, distance predicts the margin (r = −0.21, −0.027 per unit distance; same sign in 11 of 12 categories, with the slope varying from −0.08 airplane to −0.01 telephone). A single function margin = g(distance) explains 0.0 % of the raw margin; level(trial) + g(distance) is the model that fits. Splitting trials into five bands by pretrained margin gives five stacked curves, each at the height the pretrained model set, not one curve (evidence/fig78).
**Decision:** State it this way in the paper: the margin reads distance to the training set *relative to the trial's own level*; comparisons are meaningful within a trial (same images, different training sets) or after subtracting the level (the pretrained margin, or the mean over models). A single margin from a single model on a single trial is not a distance. This is the within-trial design's justification, stated as a property of the measure rather than a convenience.
**Because:** It answers the reviewers' question directly, explains why every raw plot in this project wobbled and every within-trial plot did not, and is exactly what "the margin reveals representational support" should mean: support is relative to what the model already had.
**Rejected:** Treating the wobble in fig77's right panel as noise to be smoothed (it is the trial levels, structured and 92 % of the variance).
**Implication:** The manuscript's human claim inherits this: a human margin on a trial is a level plus a shift term, and the level has to be estimated (the same participants on matched trials, or the pretrained-like baseline). The within-category slope differing by category is a second question — is the "exchange rate" between distance and margin category-specific? — for the text, not the headline.
**Steering:** lead asked; agent tested.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D30"></a>
## D30 — 2026-09-20 — Walkthrough artifact updated for the collaborator (Act 9); an un-normalised view of the round-3 result
**State:** Round 3 in (34/36 cluster models, 6 targeted); the lead asked for the shared artifact to carry the weekend's take-homes, and for a version of the result with nothing normalised.
**Observation:** Un-normalised: fine-tuned and pretrained margins on the same trials against distance to the model's 25 training objects — the gap is what training added: all models +0.029 in the nearest bin, +0.004 in the farthest; the trial's own-category models only, +0.048 → +0.017 (evidence/fig77, 30 bins at the lead's request). Read the gap, not the slope: the slope carries the pair-distance confound (D26).
**Decision:** Artifact version 8 adds Act 9, "Change the training data": round one's null and why (dose; MOCHI's selection; pipeline ≠ reference), the lead's sub-category design and its result, the distance predicting the winner without labels and with random subsets, the twelve-category reproduction of the Act 4 analysis, the un-normalised view; two limits (small effect; graded within / step across on this descriptor) and the weekend's cost from the ledger ($355 / $80; project $380 / $85). Not included: the search, the seed mechanics, the descriptor question.
**Because:** The collaborator needs the take-homes and the attribution, not the trace; the trace is the repo.
**Rejected:** Embedding every figure (the page has a 16 MB limit and a reader's patience).
**Implication:** URL unchanged: https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r. The demo repo and viewer still need rebuilding from the states (pending the lead's confirmation of the new state).
**Steering:** lead asked; agent wrote.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D29"></a>
## D29 — 2026-09-20 — The state-3 analysis reproduces on the round-3 models: 34 models trained on 25 objects of one kind, 11,634 trials, moving the training data moves the margin
**State:** The lead asked to keep it simple — run the analyses that already worked (every trial × every model; distance to that model's training set; trial held fixed; the rank curve) on the new models, rather than the relationship-coloured figure of D28.
**Observation:** 34 of the 36 cluster models evaluated (telephone c0 and watercraft c1 still scoring), 395,556 trial × model points. As measured: fine-tuned r = −0.10, pretrained r = −0.06 (grey falls too, as always). Trial held fixed: within-trial r = −0.13, 70 % of trials slope down, permutation null sd 0.0015; pretrained flat by construction. Rank curve: nearest training set +0.027 above the trial's average, farthest −0.008, steepest over the first few ranks — the same shape as with the twelve category models (evidence/fig76). Spend for round 3 to date ≈ $250 / $55.
**Decision:** This is the figure format for the paper's within-and-across result: the three panels of state 3, with models trained on 25 objects of one kind. D28's relationship split stays as the analysis behind it (the step across categories and the graded slope within), not as the headline.
**Because:** The lead's point: the simple analysis is what worked and what a reader already understands; the new models are just more training sets, and the same plot carries them. The finer decomposition is for the text.
**Rejected:** Leading with the coloured one-axis figure (D28).
**Implication:** Re-run when the last two evaluations land; then the targeted-trial figure (6 models, in). Then the descriptor sweep (no training) to see whether cross-category distance can be made informative.
**Steering:** lead redirected to the simple analysis; agent ran it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D28"></a>
## D28 — 2026-09-20 — First cut of the final figure (8 models): graded within a category, a step across categories; the voxel distance does not put the two on one line
**State:** The lead asked to design the final visualisation now, from the finished models, so the rest drop into it. Eight N = 25 cluster models (chair, airplane, bench ×2) scored on all 11,634 bank trials.
**Observation:** What training adds (fine-tuned − pretrained, same trial): +0.068 on the kind the model saw, +0.050 on other kinds of the same category, +0.017 on other categories — the right order. But "fine-tuned − pretrained" does not cancel trial difficulty (fine-tuning amplifies the pair-distance effect), and even with pair distance regressed out, at matched distance from the training set (0.45–0.80) a same-category trial gains 0.066–0.077 and an other-category trial 0.018 (evidence/fig75). Within trial: among a trial's own-category models, distance predicts the winner (r = −0.19, slope −0.049 per unit); among other-category models, r = +0.06 — the voxel distance from an airplane to 25 chairs vs 25 benches predicts nothing.
**Decision:** The final figure is the within-trial one, in two parts that say different things: (A) the graded within-category effect — each trial against the models of its own category, all twelve categories pooled, x = distance to the model's training set, y = margin, both relative to the trial's mean over those models; (B) the category step — the same trial against every model, showing same-category models as a separate cloud at low distance and high margin, and no slope inside the other-category cloud. Plus (C) the matched-distance bars as the honesty check. The raw curves with the pretrained control go in the supplement, with the pair-distance confound named.
**Because:** The lead's hoped-for "one trend line" is a claim about the descriptor as much as about the margin, and the data with this descriptor do not support it: the 16³ voxel distance is informative inside a category and uninformative between categories (cross-category distances are compressed into 0.8–1.3 and carry no ordering). Saying so is a result; forcing one axis would hide it.
**Rejected:** y = fine-tuned − pretrained as the headline (does not remove difficulty); one pooled trend line (would be dominated by the category step, with the within-category slope invisible in it).
**Implication:** A descriptor question, answerable with no new training once round 3 is in: does a finer description (32³ voxels, the 57-d invariant descriptor, or a learned but frozen space) make cross-category distance predictive? If one does, the one-axis figure returns; if none does, "graded within, step across" is the finding and the paper's claim is stated at category and sub-category scale separately.
**Steering:** lead asked for the format; agent built the first cut and found the step.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): medium — eight of thirty-six models, three of twelve categories; the within/other split could narrow with the full set, the sign of the other-category slope is unlikely to flip.

<a id="D27"></a>
## D27 — 2026-09-20 — Round 3 submitted: the sub-category design in all twelve categories at N = 25, scored on every category; and the targeted-trial experiment
**State:** Lead's read: results so far are promising and point to one visualisation — within-category and across-category points on one trend line, the distance measure working for all of it; the clustered version is "fine, kick it off now"; plus: pick a test trial, build a training set for it, show it and its neighbours improve with distance to it.
**Observation:** Every category has renders, shape descriptors and similarity tables. Clustered as for chairs (k-means 8, keep the 3 most separated; cluster sizes 86–681), split in half, N = 25 from each training half. Bank trials from every held-out half: 11,634 trials over 12 categories, pair distance recorded per trial (D26). Two MOCHI chair trials with the lowest pretrained margin (shapenet262, shapenet119) each get three 25-chair training sets — the 25 nearest by our distance (mean 0.52–0.67), 25 random (0.91), the 25 farthest (1.10–1.13).
**Decision:** Submit 33 new cluster models + 6 targeted models (jobs in `knockout/logs/batch3_jobs.txt`), each evaluated on MOCHI and on all 11,634 bank trials; the three existing chair N = 25 models are re-scored on the combined file. ≈ 39 × 1.5 GPU-h training + ~40 min evaluation each ≈ 70 GPU-h ≈ $320 / $70.
**Because:** The target figure needs every model scored on every category so that within-category distances (other clusters, same category) and across-category distances (other categories) sit on one axis, with the within-trial comparison across 36 models; N = 25 is where composition matters most (D24) and the seed floor is 0.0035 (D26). The targeted experiment is the actionable form of the same claim, sized at ~+0.02 (D25) with the bank neighbourhood as the readout.
**Rejected:** Waiting for the pair-distance-matched trial builder (pair distance is recorded; matching can be done in analysis); N = 50 as well (doubles cost for a smaller effect).
**Implication:** Analysis to write while it runs: the one-trend-line figure (x = distance from trial to the model's training set; y = within-trial margin; points coloured by same-cluster / other-cluster-same-category / other-category) and the targeted figure (margin change vs distance of each MOCHI and bank trial to the targeted training set).
**Steering:** lead — the design and the "kick it off"; agent — construction and submission.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high that the figure can be made from these runs; medium on some categories' clusters being distinct enough (lamp: one cluster of 681).

<a id="D26"></a>
## D26 — 2026-09-20 — The rise of the raw margin with distance is how the bank trials were built; seed floor 0.0035; the cluster-free random-subset test passes
**State:** The lead found the left panel of fig72 unintuitive — why would the margin *rise* as the trial gets farther from the training data? All remaining evaluations landed (seeds, eight random-100, five knockouts on bank trials).
**Observation:** (1) Distance from a trial to the training set correlates 0.81 with the distance between the trial's own two objects: the bank trials pair each held-out object with one of its ten nearest held-out neighbours, and an object far from the training set sits in a sparse region, so its nearest neighbour is also far, so the pair differs more and the trial is easier for any model (pretrained r with pair distance +0.21; fine-tuned +0.24). Hold the pair distance fixed and the pretrained model's rise with distance-to-training goes from +0.18 to +0.02 (evidence/fig73). (2) Seed floor: three models on the same 104 chairs (seeds 42/43/44) differ per trial by sd 0.0035, means within 0.002; the own-vs-other numbers reproduce to three decimals. So the per-trial spread between models trained on *different* random 100-chair sets (0.0165) is almost entirely data content, and the cluster effect (0.009–0.015) is three to four times the seed floor. (3) The cluster-free test: eight random-100-chair models on the 882 bank trials, within trial, distance to the model's own 100 chairs vs margin: r = −0.077, permutation null sd 0.012 (p = 0.001), 57 % of per-trial slopes negative (p = 8e-8). Smaller than with clusters (−0.26) because the contrast is smaller (distance sd 0.038 vs ~0.25); the slope per unit distance is of the same order (−0.033 vs −0.057).
**Decision:** Treat (1) as a property of the test set to control in round 3: build trials with matched pair distance (or partial it out), so the raw plot is readable as well as the within-trial one. Treat (3) as the answer to "does clustering matter": no — random composition moves the margin in proportion to the distance contrast it creates.
**Because:** The pair-distance confound is the same object as the ½·d(A,B) floor of the manuscript's metric (D06), reappearing on the test-set side; naming it once more and designing it out is cheaper than explaining it every time.
**Rejected:** Reading the raw rise as evidence against the measure; treating the between-subset spread as noise (it is signal — different data, different model).
**Implication:** Round 3 = random subsets at N = 25 per category on pair-distance-matched bank trials; and the targeted-subset test (D25). Both now have a seed floor to be judged against.
**Steering:** lead asked the question that exposed the confound; agent tested it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on all three.

<a id="D25"></a>
## D25 — 2026-09-20 — Distance vs margin by training-set size: as measured, size and difficulty are entangled with it; within trial, every size gives the same falling line
**State:** The lead re-asked the question precisely: distance on x, margin on y, lines by training-set size — are distance and size so entangled that the relationship moves with N?
**Observation:** Bank-built chair trials, 12 models with evaluations (25 / 50 / 100 / 2,000 chairs). As measured: larger training sets are nearer to every trial (the lines shift left: mean distance 0.80 → 0.74 → 0.65 → 0.27), and inside any one size the curve is non-monotone — margin *rises* with distance up to ~0.6 and falls beyond; the pretrained model rises with distance too (against distance to all 2,000 chairs), because inside a category the unusual objects make the easy trials; the fall at the far right is a model scored on a kind of chair it never saw. Point-level r ≈ −0.05 at every N as measured. Within trial (same trial, different training sets of the same size): r = −0.27 / −0.29 / −0.26 at 25 / 50 / 100, the same line at every size, about 0.04 of margin across the range of composition contrast (evidence/fig72).
**Decision:** State the relationship as a within-trial one and give the raw plot beside it with the confound named. The slope of margin on distance does not depend on training-set size in these units; what depends on size is the range of distances a design can create (D23).
**Because:** The raw curve mixes three things with different signs (size shifts distance; unusual objects are easy; unseen kinds are hard); the within-trial comparison removes the first two by construction.
**Rejected:** Reporting the raw r ≈ −0.05 as the effect (it is the sum of opposed effects); centring on the pretrained margin instead of within trial (leaves the size shift in).
**Implication:** For the lead's next aim — choosing training subsets by distance to raise the margin on chosen MOCHI trials — the relevant slope is the within-trial one: ~0.04 margin per 0.7 of distance contrast, and at N = 25 the achievable contrast is ~0.4 (D23). So a targeted 25-object subset should move a trial's margin by roughly 0.02 against random — small, but with hundreds of neighbouring bank trials as the readout, measurable.
**Steering:** lead specified the plot; agent built it and added the pretrained control.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high; the 100-chair line will firm up as the remaining random-100 evaluations land.

<a id="D24"></a>
## D24 — 2026-09-20 — The margin has two doses: any fine-tuning at all (saturates by 25 objects), and which objects (0.01–0.02, largest at small N)
**State:** The lead asked how the margin itself behaves across training-set size, to choose the regime where an effect on the fine-tuned models can be expected — "if you train on one object and the margin does not move, that is informative; if it does, that is informative".
**Observation:** Every chair model placed by its training-set size (evidence/fig71). Bank trials: pretrained 0.079; 25 chairs of one kind → 0.167 on that kind, 0.143 on other kinds; 50 → 0.156 / 0.137; ~100 → 0.145 / 0.132; all 2,000 → 0.138; a random 100 → 0.152. MOCHI: the same shape (pretrained 0.081; 25 → 0.157; 100 → 0.145–0.166; 1,800–2,000 → 0.155; the collaborator's 2,000-chair model 0.241, not reproduced). So: the first effect — fine-tuning at all — is +0.07 to +0.09 and is complete by 25 objects; it does not grow with more data and slightly shrinks. The second effect — which objects — is +0.015 at 25, +0.012 at 50, +0.009 at ~100 (own kind minus other kinds, model level and trial difficulty removed), against a per-trial between-model sd of 0.019.
**Decision:** Within-category experiments live in the second effect and therefore at small N, with many trials per model. The pipeline's fixed 20,000 triplets per epoch is why the first effect saturates; within-category shift is only visible once the training set is small enough that its composition changes what those triplets contain.
**Because:** The design quantity is not the margin's level but the part of it that depends on composition, and that part is largest where the composition contrast (D23, right panel) is largest.
**Rejected:** Reading the falling own-kind line as "smaller is always better" — at N = 1–10 the model may not learn the kind at all; that end has not been measured.
**Implication:** Worth measuring the low end directly: N = 1, 5, 10 objects of one kind (3 fine-tunes, ≈ $27 / $6) — whether the margin moves with one object is exactly the lead's question. Combined with D23: random subsets at N ≈ 25, many per category.
**Steering:** lead asked the question; agent assembled the figure.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the shape; the seed floor (still training) will set the error bar on the second effect.

<a id="D23"></a>
## D23 — 2026-09-20 — What the next experiment should be: random subsets per category, not clusters; the estimate's behaviour with training-set size
**State:** New state drafted (the agent's round failed, the lead's worked); the lead asks what to run next and whether the cluster design is "searching for the answer"; the version he wants is the one where clustering does not matter.
**Observation:** With no training: the distance from a held-out trial to a random training set of N chairs falls smoothly with N (0.93 at N = 10 → 0.33 at N = 1,800); cluster training sets sit far below the random line on their own cluster (0.28–0.46 at N = 25–100) and above it on others (0.85–1.03). Counts within a fixed radius are zero for 97 % of trials at N = 100 and still 66 % at N = 1,800 — the count-based estimate has no resolution at the sizes the recipe registers. The spread of the estimate across random training sets of the same size, for a fixed trial — the contrast a design can create — is largest at small N (sd 0.044 at 25, 0.035 at 100, 0.017 at 800) for the distance form and the reverse for the count form (evidence/fig70). The eight random-100-chair models from round 1 already exist and are being scored on the bank trials (8528212–19): the cluster-free test costs $3.
**Decision:** The general experiment is: per category, k random training subsets of N objects (no clustering), every subset's model scored on bank-built trials of that category, and the within-trial analysis of D22 — does the model whose subset is nearer give the bigger margin? Clusters become a special case (a subset with a large, known contrast) rather than the design. Run the free version first (the eight random-100 models on bank trials); if it shows the D22 effect, scale to N = 25 with more subsets and more categories; if not, clusters remain the way to guarantee contrast and the question becomes why random contrast does not register.
**Because:** Random subsets need no assumption about clustering or the descriptor's geometry; the contrast they create is measurable in advance from the right-hand panel; and the effect at N = 25 clusters grew as N shrank, so small random subsets should carry the most signal per model.
**Rejected:** Varying the number of clusters as the next step (answers a question about the clustering, not about the measure); clustering every category first (the version that "searches for the answer").
**Implication:** If the random-subset test passes: 12 categories × 6 subsets × N = 25 ≈ 72 fine-tunes ≈ $650 / $145, or 4 categories × 6 ≈ $215 / $48 as the staged first cut. Bank-built trials need building per category (the chair set took one script).
**Steering:** lead framed the choice (random over clusters; the estimate vs set size); agent computed and recommended the order.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the no-training numbers; medium that random subsets at N = 100 will show the effect on bank trials (MOCHI showed r = −0.10; the contrast at N = 100 is smaller than at 25).

<a id="D22"></a>
## D22 — 2026-09-20 — The model-free distance predicts which cluster model wins a trial, better than the cluster labels and beyond them; coverage saturates at small N; graded-within-one-training-set still not identifiable
**State:** D21 in hand (sub-category membership moves the margin); the lead asked how that relates to the shift measure.
**Observation:** Same nine models and 882 trials, labels discarded. For each trial × model, distance from the trial's objects to that model's actual training chairs (3-D shape; mean distance to the 10 nearest). With model level and trial removed: r = −0.21 / −0.26 / −0.24 at N ≈ 100 / 50 / 25, against the cluster label's +0.17 / +0.23 / +0.23; distance with the label held fixed −0.15 / −0.12 / −0.09 (evidence/fig69). Coverage (count within a fixed radius) does worse here (+0.14 to +0.21): with 25–100 training objects most counts are zero, so the count saturates where the distance still varies. Inside a single cluster with its single model, distance vs margin is r ≈ +0.05 to +0.09 and the pretrained model shows the same pattern on the same trials — the one-training-set confound, as in D07.
**Decision:** This is the answer to "how does it relate": the measure predicts, without labels, which of three differently-trained models gives a trial the bigger margin — the state-3 result reproduced one level down with new models, bank-built trials and a continuous ruler. Report the distance form (nearest-neighbour) for small training sets and coverage for large banks, and say why.
**Because:** The label is a coarse proxy for the distance (they correlate at −0.95); the distance carries what the label carries and more. Predicting beyond the label is what "model-free shift estimate" was supposed to mean.
**Rejected:** Claiming a graded within-cluster result (not identifiable with one training set per cluster — needs nearest / random / farthest subsets around held-out objects, the round-3 design).
**Implication:** The manuscript's claim survives in a form that can be stated precisely: distance from the test objects to the training objects, measured on the stimuli, predicts the fine-tuned margin within trial, at category and sub-category scale. Still owed: the seed floor; the round-3 graded test; the human side (no new human data touched by any of this).
**Steering:** lead asked the question; agent computed and framed it.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the numbers (three independent N, 2,646 trial × model rows each); medium-high on "better than the label" (partial r shrinks as N shrinks: −0.15 → −0.09).

<a id="D21"></a>
## D21 — 2026-09-20 — The sub-category calibration works: models specialise to the cluster they were trained on, more so with fewer training chairs; the bank-built test set carries the category effect
**State:** Round 2 (D19/D20) complete except the seed replicates, which failed on a bad Hydra override and were resubmitted (8528180–81).
**Observation:** (a) Test-set check: the original twelve models on 882 bank-built chair trials — chair model 0.231, then bench 0.161, display 0.119, the rest 0.08–0.11, pretrained 0.079 (accuracy 65 % → 92 % for the chair model). The ordering of the other eleven cannot be memorisation (evidence/fig68). (b) Cluster models: with each model's overall level and each cluster's difficulty removed, the own-cluster advantage is +0.009 (N ≈ 90–104), +0.012 (N = 50), +0.015 (N = 25), the largest of the six possible model–cluster pairings each time (next best +0.005 / +0.003 / +0.007); per-trial ± 0.004. Two models specialise clearly (tall narrow-backed +0.010–0.017; wide armchair-like +0.014–0.022); the round-backed/office model barely (+0.002–0.007) — the most "typical chair" cluster, useful everywhere (evidence/fig67). The naive "own − others" statistic gave +0.013–0.023 at p < 1e-14 but was inflated by one model being better overall; the interaction is the honest number.
**Decision:** The assay's operating point is: training sets of ~25–100 objects, hundreds of bank-built hard trials, per-trial statistics with model level and trial difficulty removed, seed replicates for the floor. The graded questions are now asked at this point.
**Because:** This is the category effect one level down, in the predicted direction at every N, and it grows as the training set shrinks — the recipe registers *which* objects it saw when there are few of them, and not at all when 10 of 2,000 are removed (D18).
**Rejected:** Reporting the naive per-trial contrast (row effect); calling the effect large (it is ~0.01–0.015 against a per-trial sd of ~0.02 — real, and small).
**Implication:** Next: seed floor (pending) → then the graded test at N ≈ 50 on bank trials: for held-out objects, train on the 50 nearest vs 50 random vs 50 farthest of the same category, and relate margin change to coverage/distance — the targeted knock-in, done where it can be seen. Also worth one figure: the cluster-3 model's generality vs the descriptors.
**Steering:** lead designed the calibration; agent ran and analysed; the row-effect correction was the agent's own catch after over-claiming in a first footer.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on (a); high that the interaction is real (consistent sign and monotone growth across three independent N); medium on its size until the seed floor is in.

<a id="D20"></a>
## D20 — 2026-09-20 — Round 2 trimmed to the calibration: knock-ins cancelled, seed replicates added
**State:** 34 jobs pending, none running; the lead asked what we were waiting on and how best to spend.
**Observation:** The 12 targeted / cross knock-ins (D19) read out on six MOCHI trials, expect ~0.01–0.03 against ±0.02 per-trial noise, and test on the set the lead had just said not to test on; a cluster model already is a targeted knock-in for a whole sub-population. The cluster matrix cannot be sized without a run-to-run noise floor from identical data.
**Decision:** Cancel the 12 knock-ins while pending (no compute used; −$110 / −$24). Add two seed replicates of chair_c0_all (seeds 43, 44 on all three seed settings; jobs 8526306–07; +$18 / +$4). Round 2 is now: bank-eval test (C) → reference twelve on bank trials (D) → 9 cluster models (B) → 2 seeds. ≈ $100 / $22 total.
**Because:** Every remaining job serves the calibration the lead designed, in the order the information is needed; the targeted question is asked once, later, at the N the matrix says works and on bank-built targets.
**Rejected:** Keeping half the knock-ins (still six trials on MOCHI); cancelling D (cheap, and it validates the test set).
**Implication:** `scripts/run_train_seed.sbatch` takes a seed as third argument; `data/chair_c0_all_s43/44` are symlinks to the same objects.
**Steering:** agent proposed; lead: "Go."
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high.

<a id="D19"></a>
## D19 — 2026-09-20 — Round 2: calibrate the assay on sub-categories we construct, and test on trials built from the bank, not on MOCHI
**State:** Batch 1 analysed (D18): knockout flat inside the noise band; a weak graded effect in the random 100-chair arm; MOCHI's 76 chair trials small and built to be hard for the pretrained model.
**Observation:** The lead's design: cluster a category's objects into distinct sub-populations; split each in half; fine-tune one model per cluster on its training half; build oddity trials from every cluster's held-out half; score every model on every cluster. We know category membership moves the margin, so sub-category membership should too — and the sizes at which it does (objects per model, trials per cluster, cluster separation) are the assay's operating point. Chairs: k-means (k = 8) on the 16³ shape descriptors, keep the three most separated clusters (n = 208 / 175 / 203; between-cluster cosine distance 0.89–1.12 vs within 0.63–0.81); the montage (evidence/fig66) shows tall narrow-backed, wide low armchair-like, and round-backed/office chairs. Trials: 3 per held-out object, distractor from its 10 nearest held-out neighbours in the same cluster, 2 views + 1 view, 882 trials.
**Decision:** Submit 9 models (3 clusters × N ∈ {all ≈ 90–104, 50, 25}; jobs 8526137–45), each evaluated on MOCHI and on the bank trials, plus the pretrained model and chair_full on the bank trials (8526136). Also submit the held 12 targeted / cross-category knock-ins (8526115–27), as the lead asked. ≈ 21 fine-tunes ≈ 42 GPU-h ≈ $190 / $42.
**Because:** Testing on trials built from the bank gives thousands of triplets with controlled difficulty and no MOCHI selection effect; a whole sub-population is a dose the training recipe can register, where 10 objects of 2,000 was not; and a 3 × 3 transfer matrix with the pretrained model flat by construction is the state-3 logic one level down, so a null would implicate the recipe, not the metric.
**Rejected:** k = 50 knockouts on the full bank (2.5 % of the bank; held); evaluating only on MOCHI; forcing all 2,000 chairs into three clusters (silhouette 0.19 — the discarded middle would blur the contrast).
**Implication:** The lead then asked whether the existing twelve category models plus bank-built trials could replace training: they can replicate the *category-level* result on hundreds of controlled trials (evaluation only; the twelve checkpoints are in `Dist-shift/logs/`; the 400 held-out objects per category are not recoverable, so own-category models have seen the objects — the ordering of the other eleven is the memorisation-proof part), and that is submitted as a calibration of the bank-trial set (8526249–61, ≈ $3); they cannot answer the within-category question, because each saw its whole category. Once the matrix shows the expected diagonal at some N, the graded questions (targeted knock-in, coverage vs margin change) are asked at that N with bank-built trials. Analysis to write: `scratch/analyze_clusters.py` — 3 × 3 margins, diagonal minus off-diagonal per N, pretrained control.
**Steering:** lead — the design is his ("find categories within the category"); agent — clustering choices, trial construction, submission.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high that this is the right calibration; medium that these three clusters are separated enough for the recipe to register at N = 25.

<a id="D18"></a>
## D18 — 2026-09-20 — Batch 1 complete: knockout is flat inside the noise band; the random knock-ins show the first graded within-category effect; our pipeline does not reproduce the reference models
**State:** All 22 batch-1 fine-tunes evaluated at their final checkpoint, plus the full-bank replication (chair_full) and the chair g1 re-evaluation.
**Observation:** (1) chair_full, all 2,000 chairs through our pipeline, gives a mean margin of 0.155 on chair trials; the collaborator's reference gives 0.241 on the same chairs (their run used a 1,600/400 split; anything else lives on /datasets/hida, not visible here). (2) The four knockout models, the random-removal model and chair_full are almost the same model: per-trial differences ~0.001; airplane and table likewise. Removing a trial's own support changes its margin by −0.001 in chair and airplane, +0.000 in table, against run-to-run noise of ±0.019 per trial (evidence/fig64). (3) The pipeline draws a fixed 20,000 triplets per epoch whatever the bank size, so 100 chairs (0.16–0.17) train as much as 2,000 (0.155). (4) Across the eight random 100-chair models, the model whose chairs sit closer to a trial gives it the bigger margin: within-trial r −0.10, permutation p = 0.011, 66 % of trials sloping down (p = 0.003), total swing ~0.01 (evidence/fig65). As measured, inside chairs the margin *rises* with distance for pretrained and fine-tuned alike (+0.32 / +0.23): an unusual chair is an easier trial.
**Decision:** Treat batch 1 as a calibration of the assay, not a test of the hypothesis. The knockout dose (10 of ~2,000, ~10 % of the bank) is far below what this training procedure can register; the 100-object regime is where the training set's content moves the margin. Do not compare any new model to the reference twelve. Hold the k = 50 knockouts; the targeted and cross knock-ins (N = 100) are the right next runs.
**Because:** A uniform offset across every condition including the random control is pipeline, not support; near-identical models across 10 % removals mean the LoRA fine-tune at lr 1e-6 encodes the category and the render domain, not which objects were present, unless the bank is small enough that each object carries weight. The 100-chair effect is the one place in batch 1 where the training set differed enough to matter, and there the predicted relationship appears — weakly.
**Rejected:** Reading the knockout null as "the margin does not depend on nearby objects" (the dose was too small to say); chasing the reference offset further today (the design no longer needs the reference as a baseline).
**Implication:** Round 2 works at N ≈ 100 and needs a noise floor from seed replicates; see STATE Strategy. The within-category question remains open but is now answerable with this assay at that scale.
**Steering:** agent — analysis and diagnosis; lead asked for the state of play and for round-2 thinking, and had said he was not surprised the first batch did not work since no high-level strategy feedback had been given.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on (1)–(3); medium on (4) — one permutation p of 0.011 on 76 trials; medium that N ≈ 100 is the right regime rather than a different training recipe.

<a id="D17"></a>
## D17 — 2026-09-19 — First look at batch 1: no own-support effect yet, and our retrained models sit 0.09 below the reference
**State:** 19 of 22 batch-1 fine-tunes complete with chained evaluations (three table knockouts finishing); the lead asked when the first new results would arrive — they had, unread, while we documented.
**Observation:** Chair and airplane knockouts (remove the 10 nearest training objects of a group of test objects): the margin change on a trial is the same whether or not that trial's support was removed (within-trial contrast −0.002 chair, −0.0006 airplane; 49 % / 55 % negative) and the same as removing 10 random objects. Trials with *both* objects' support removed drop 0.02–0.03 more (n = 12 / 27, wide error) — a hint, not a result. Every retrained chair model, full bank minus ~10 %, reaches a mean margin of 0.155 on these trials against the collaborator's reference model's 0.241 on the same 2,000-chair bank; and 100 random chairs reach 0.16–0.17. Run-to-run noise among the eight 100-chair models: sd of means 0.006, per-trial sd 0.019, trial ranking agreement r = 0.97. The chair g1 evaluation on disk was the epoch-10 test, not the final checkpoint — re-submitted (job 8522613).
**Decision:** Do not read the knockout result until a same-bank replication says whether our pipeline reproduces the reference: `chair_full` submitted (job 8522617, ≈ 2 GPU-h, $9 / $2). Also: the design grouped test *objects*, so most trials have one object's support removed per condition — analyse by objects hit (0 / 1 / 2), not by trial group.
**Because:** A uniform 0.09 offset across every condition including the random control cannot be a support effect; either the patched pipeline differs from the collaborator's or 10 % random removal costs 0.09, and 100 chairs ≈ 1,800 chairs makes the second implausible. Until that is settled, "no effect of removing the 10 nearest" and "the margin does not respond within a category" are not distinguishable from "the retraining is not comparable".
**Rejected:** Reporting the null now (premature); comparing to the reference model at all until the replication is in.
**Implication:** If chair_full reproduces 0.24, the knockouts are interpretable as they stand and k = 10 was too small a dose → run k = 50. If it lands at ~0.155, all batch-1 comparisons are within-pipeline only (still valid among themselves) and the reference twelve are not a baseline for the new runs. Either way, 100 chairs saturating the margin is itself a within-category fact worth a figure.
**Steering:** agent — found it while answering the lead's timing question; lead not yet consulted on the replication spend.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the numbers; medium that the offset is pipeline rather than data; low on the "both objects" hint.

<a id="D16"></a>
## D16 — 2026-09-19 — The encoder-space own-category result is a concern, not a kill; how a state figure should be built
**State:** State 2 under review with the lead; the replacement encoder-space figure (evidence/fig62: one plot type × 4, expectation stated, verdict boxes).
**Observation:** On the own-category rows the pretrained margin falls with the encoder-space distance about as steeply as the fine-tuned margin (−0.24 vs −0.26). Lead's read: top row looks good (a small upturn at the far right, let go); bottom row "makes me pause" — a subtle, persistent concern, not a conclusive result.
**Decision:** Record it at that weight in State 2. Grade results as behaves-as-expected / concern / conclusive, not pass/fail. "Fine-tuned r > pretrained r" is not the evidence we are after; what we want is to understand what the distance measures. The figure recipe (one plot type repeated; say the expectation before the data; graded verdict box off the data; nothing anachronistic) goes into agent/FEEDBACK.md and CHECKS.md.
**Because:** The trace should carry the concerns that did not go away, at the weight they had. Calling this a kill, or hiding it behind a coefficient comparison, both misrepresent the process.
**Rejected:** Wording the bottom row as a failure of the encoder-space estimate (not established); dropping it (it is why leaving the encoder's space felt necessary).
**Implication:** The concern carries into States 3–5; the knock-in runs are the test that can settle it. Figure recipe applies to every state figure from here.
**Steering:** lead — read the result and set its weight; agent — built the figure and drafted the text.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high the pattern is real; medium on "ruler and object are one" as the explanation.

<a id="D15"></a>
## D15 — 2026-09-19 — The oddity-blind estimate inside the encoders' own spaces: leaving the encoder was a choice, and the encoder sees within-category structure geometry does not
**State:** Track A features extracted (pretrained DINOv2-L and the chair/airplane/table fine-tunes, 311k renders + MOCHI); lead reviewing the states and asking, for State 2, what actually showed the encoder space to be insufficient.
**Observation:** Same form, same 20,750-object bank, in the encoders' own features (image-level knn): within-trial r −0.42 to −0.48 (geometric coverage −0.42); pooled −0.23 to −0.32 with the pretrained control at 0.00 to +0.08 (passes); on-category −0.17 to −0.28 with control −0.10 to −0.28 (fails); category-centred on-category −0.09 to −0.15, p ≤ .01 (geometric: +0.01). In the airplane model's own space on airplane trials: r −0.42, control −0.22; chair −0.24, control 0.00; table null.
**Decision:** State 2 is rewritten: the model-free space was a *choice* — no learned parameters, reviewer-proof, and the only ruler that applies to humans — not a necessity forced by failure of the form fix. Open question 3 is answered yes: the encoder's own representation carries a within-category signal our descriptors lack, so finer geometric descriptors are worth building, and encoder-space x's join the knock-in metric comparison.
**Because:** Fixing the form is enough for the pooled control even in encoder space; the earlier "level 1 alone fails" was measured in DINO ViT-B against ImageNet, not this bank. Encoder-space distance and encoder-space atypicality are more entangled (on-category control), which is why the within-trial design matters even more there.
**Rejected:** Keeping State 2's "the encoder space still leaks, so we left it" framing — it was true for the wrong reason.
**Implication:** Level-3 diagnosis revised: not only the design — the descriptors too. evidence/fig62.
**Steering:** lead — asked the question that exposed it; agent — ran the battery and corrected its own earlier claim.
**Confidence:** lead (TB): — (to fill) · agent (Claude Opus 5): high on the numbers; medium on how much of the encoder-space within-category signal is training-specific (the on-category controls say roughly half).

<a id="D14"></a>
## D14 — 2026-09-19 — Batch 1 submitted; cost made explicit; k = 50 held; documentation instructions revised
**State:** Pilot at ~3 min/epoch (D12); eval step verified on its epoch-10 checkpoint (same columns, pretrained margins bit-identical to the reference); all 50 subset directories and per-condition similarity tables built; lead offline for the afternoon, asked for resource-rational compute and for the cost in money.
**Observation:** PARCC rates: full B200 $4.51/h unsubsidised, $1.00 subsidised; mig45 exactly a quarter of that, so a quarter slice saves money only if the job runs faster than quarter speed. A run is ~2 GPU-hours (train + eval): ~$9 / ~$2. Account cap 60M billing-min, 17% used; the whole design would add ~12%.
**Decision:** Submit batch 1 to dgx-b200 (21 runs: knockout k = 10 for all groups + random controls, and the 7 remaining random knock-ins; job IDs in `knockout/logs/batch1_jobs.txt`); hold the 15 k = 50 knockouts until k = 10 results say they are needed; route the 12 targeted/cross knock-ins by the mig45 timing run (8520577); 4.5 h time limits so jobs backfill. Write HANDOFF v2 recording how the documentation system changed in first use.
**Because:** Batch 1 ≈ $200 / $44, all 50 ≈ $450 / $100 — small either way, but the k = 50 dose is the least informative third and can wait a day; done-this-weekend is met either way.
**Rejected:** Submitting everything at once (30 % of the budget on the least specific manipulation before seeing the specific one).
**Implication:** Results arrive in `knockout/eval/<cat>_<cond>/ood_analysis_results.csv` over the next ~10 h; analysis script to be written while they run.
**Steering:** agent, under the lead's cost constraint.
**Confidence:** lead (TB): medium — deferring to Claude on staging; I want the money stated *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high on the cost; medium on holding k = 50.

<a id="D13"></a>
## D13 — 2026-09-19 — D11's framing corrected: the reviews were not the source of the findings
**State:** Reviews ingested (D11) with a table mapping each concern to a later result, as if the work had been a response to them; lead reviewing the trace.
**Observation:** Lead: at the time the reviewers' language was hard to interpret and the honest read was "we were right and miscommunicated"; the sense that they were right came only as our own results pushed back. The trace should show that state faithfully, not a retrofitted clarity.
**Decision:** State 0 now records what the reviewers said, what was unclear to us, and what we believed; the recognitions ("this may be what they meant") move to the states where they occurred — State 1 (SCwg / the floor), State 3 (r64w / the control), State 4 (r64w / the object-based measure). The mapping table in background/ is replaced by a note on what was unclear and pointers to those states.
**Because:** Retroactively aligning the work to the reviews overstates both their clarity and our foresight; the point of the trace is how understanding actually changed.
**Rejected:** Editing D11 in place (append-only); dropping the recognitions entirely (they are real, just later).
**Implication:** Supersedes D11's third sentence. Pattern for future external feedback: log what was said and what we made of it then; log recognitions when they happen.
**Steering:** lead.
**Confidence:** lead (TB): high — this correction was mine *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high.

<a id="D12"></a>
## D12 — 2026-09-19 — Pilot fine-tune timed; the full 50-run design is affordable and goes ahead
**State:** Interventions designed (D09) but unsubmitted, waiting on the cost of one fine-tune; the pilot (chair, group 1, k = 10) reached training on its fifth attempt after four missing-dependency / hard-coded-path failures.
**Observation:** ~3 min per epoch on a dedicated B200 once the file cache is warm (epochs 5–10: 15 min); 30 epochs ≈ 1.5–2 h. The subset pipeline behaves: 1,831 chairs discovered, 20,000 similarity-binned triplets per epoch, checkpoints saving on val-loss improvement. All 50 runs ≈ 90 GPU-hours.
**Decision:** Run the whole design as independent 1-GPU jobs, in the order random knock-in → knockout k = 10 → targeted and cross knock-in → knockout k = 50; keep the recipe pinned (20k triplets, 30 epochs) even for 100-object subsets; give each condition its own similarity table filtered to its objects so binned mining fills every epoch; chain the MOCHI evaluation into each job.
**Because:** Half the estimated cost; small subsets would otherwise silently under-fill the epoch (partners missing from the subset are rejected by the miner); a bare checkpoint without its evaluation is not a result.
**Rejected:** Reducing triplets per epoch for small subsets (changes the recipe; comparability matters more than 40 GPU-hours). Whole-node requests (wait far longer than 1-GPU jobs).
**Implication:** ~a day of wall time at 6–8 jobs in parallel; each run ends with `eval/<cat>_<cond>/ood_analysis_results.csv` in the same format as the 12 category models.
**Steering:** agent — the lead asked to be resource-rational; the agent chose the staging.
**Confidence:** lead (TB): medium — letting Claude steer the run order; I care about cost and this weekend *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high on the timing; medium on the run order.

<a id="D11"></a>
## D11 — 2026-09-19 — The NeurIPS reviews ingested into the trace (external feedback)
**State:** Documentation system live (D10); State 0 had a placeholder for the reviewer concerns.
**Observation:** Three reviews (June 2026) and a post-rebuttal follow-up: the human results are real and the statistics sound; the word "distribution shift" is doubted by all three — inter-class distance (SCwg); within-set weirdness with no shift required, test it on two random halves of one set (r64w); compare against existing shift metrics, scope, overclaims (HVBU). r64w recommends resubmitting after a large reframing or dropping the framing.
**Decision:** Record the reviews paraphrased in background/ and map each concern to what we later found — the ½·d(A,B) floor is SCwg's inter-class distance [D06]; the pretrained control is r64w's two-halves test in a stronger form [D02]; the object-based view-averaged measure r64w asked for is what we built and it wins [D08].
**Because:** The trace should show that the reviewers were pointing at the same defect we found, in different words — that is the argument for the resubmission's framing.
**Rejected:** Quoting review text verbatim in the repo.
**Implication:** State 0 now opens with the reviews; the framing decision (claim category coverage vs wait for within-category) is TB's, listed in STATE next steps.
**Steering:** agent.
**Confidence:** lead (TB): low — I was confused by the reviewers' language at the time and did not want the trace to pretend otherwise *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): medium: I mapped the reviews onto findings too confidently.

<a id="D10"></a>
## D10 — 2026-09-19 — Adopt the STATE/REASONING documentation system
**State:** Q1 answered and written up as an artifact; Q2 handed to experiments; a RECAP + five narrative checkpoints drafted that afternoon.
**Observation:** Lead supplied the lab HANDOFF: STATE = current truth, REASONING = why, evidence = only what is cited, git = history; wants the process presentable and projects citable.
**Decision:** Initialise a git repo on branch `docs-system`; replace the draft with STATE / REASONING / meetings / evidence / background; keep the narrative draft under background/ as source.
**Because:** Two "current truth" documents drift; the HANDOFF form is what the lab will read.
**Rejected:** Keeping RECAP + checkpoints as the primary record.
**Implication:** Every substantive STATE change now ships with a Dxx in the same commit; add the MOCHI project's STATE.md to background/ when it exists.
**Steering:** lead.
**Confidence:** lead (TB): high *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high.

<a id="D09"></a>
## D09 — 2026-09-19 — Run interventions instead of searching for another metric
**State:** Coverage is the best ruler for Q1; every estimator gives r ≈ 0 within category after category centring (D07); the level-3 test on the training-specific margin is −0.02 (object) / −0.07, p 0.06 (image).
**Observation:** With one training set per category, "far from the training set" and "unusual object" are one variable — no metric can separate them; only varying the training set within a category can.
**Decision:** Launch (a) encoder-space upper bound — features of 311k renders + MOCHI under pretrained and chair/airplane/table fine-tunes (job 8519397); (b) support knockout — chair/airplane/table, 4 random groups × k∈{10,50} nearest-neighbour removal + 2 size-matched random controls, 30 fine-tunes, pilot 8519673; (c) knock-in from pretrained — 8 random + 6 targeted + 6 cross-category subsets of 100 chairs, designed.
**Because:** A knockout raises the own group's knn_mean 3–4× more than other groups' and 5× more than random removal, so each trial gets 11 training sets with Δknn 0–0.2, stimulus fixed; random knock-in runs make metric comparison free (any x recomputed on the same runs).
**Rejected:** Hyperparameter search (LoRA rank, lr) — pinned to the original recipe; only epochs earns an ablation. Radius-based knockout — did not transfer across categories (chairs have ~3 neighbours at ε, airplanes ~280).
**Implication:** Collaborator pipeline reused unchanged via a patched `train.py` copy; five pilot attempts fixed missing deps and hard-coded paths; run budget waits on the pilot's epoch time.
**Steering:** joint — the agent proposed interventions repeatedly; the lead came round and then owned the design.
**Confidence:** lead (TB): high now — Claude said this several times before I heard it; the experiment is the right move *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high that the design is the fix; low on whether the graded effect exists.

<a id="D08"></a>
## D08 — 2026-09-19 — View-conditioned (image-level) coverage tested; not supported
**State:** Object-level coverage established (D05); lead's hypothesis that encoders learn view-dependent appearance, so the training distribution that matters is over images (shape × viewpoint).
**Observation:** Built model-free depth-from-voxels descriptors (projector IoU 0.77–0.95 vs real renders; MOCHI poses recovered by silhouette matching, IoU 0.856; bank = 20,885 objects × 15 fixed training views). Within-trial r: object voxels −0.424; depth maps at all 15 training views −0.436, one random view −0.426, nearest view −0.422, the actual MOCHI view −0.266 (best-posed tertile −0.29 vs object −0.41 on the same trials). Image-space pooled control fails (+0.06 to +0.16).
**Decision:** Fine-tuned support behaves as view-invariant within the training grid's ~25° gaps; use object-level coverage.
**Because:** Any on-grid view recovers the full effect, so the descriptor is fine and single-view noise is not the cause; only the off-grid viewpoint loses it — consistent with the multi-view contrastive objective. Robust to silhouette/depth/pooled variants.
**Rejected:** Pseudo-depth for non-ShapeNet MOCHI (reintroduces a model; no per-category fine-tunes there).
**Implication:** Untested beyond ~25° (needs new renders); Act 8 of the artifact.
**Steering:** lead — the hypothesis was the lead's; the agent built the test.
**Confidence:** lead (TB): medium — my hypothesis; the ladder is convincing but 25° of viewpoint range is not much *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high on the ladder of controls; medium on the interpretation (only ~25° of viewpoint range was tested).

<a id="D07"></a>
## D07 — 2026-09-19 — The on-category coverage curve is category identity; Q2 is a design problem
**State:** Coverage adopted (D05); lead read its on-category row (binned r −0.78, clean control) as the within-category result the paper wanted.
**Observation:** The bins sort by category: ">100 neighbours" is airplane/bench/car/lamp/telephone/watercraft (homogeneous shapes), "0" is chair/table/sofa/cabinet/display/loudspeaker (diverse, biggest banks). Category-centred r +0.013 (p 0.72); within each category mean r +0.04, four of twelve negative; twelve category points r +0.30 (p 0.34), chair the counterexample. Seven coverage variants incl. category-calibrated percentile: all ≈ 0.
**Decision:** The within-category question is not identifiable with one training set per category; stop searching estimators for it.
**Because:** The pretrained control rules out "hard for every model", not "this category's model is good at its category"; category centring is the test, and everything fails it.
**Rejected:** More estimator variants on the existing 12 models (two exhaustive searches, same answer).
**Implication:** → D09. Anatomy figure fig57 added to the artifact so the reasoning is visible.
**Steering:** agent — the lead read the curve as the result; the agent dissected it.
**Confidence:** lead (TB): medium — I was unconvinced until the anatomy figure; I still find it hard to give up the on-category curve *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high after category-centring; the lead was initially unconvinced and asked to be shown.

<a id="D06"></a>
## D06 — 2026-09-19 — D01's shapegen headline withdrawn: the entangled shift was d(A,B)
**State:** Packet and artifact drafted with "model-free beats the incumbent on shapegen" as a lead result; lead asked whether the A/B entanglement had been removed everywhere.
**Observation:** The Eq. 1–3 shift correlates with d(A,B) at r 0.993 (shapegen); d(A,B) alone predicts the encoder proxy better (0.79) than the shift (0.74); the oddity-blind form gives |r| 0.12. The incumbent's r with d(A,B) is 0.07 — it was never riding this. Every distance obeys the bound (L1 / unit-L1 / cosine: floor holds 100%, r with d(A,B) 0.73 / 0.75 / 0.78). Decomposed: the floor predicts the pretrained margin +0.57 (wrong sign), the excess −0.17 (right sign).
**Decision:** Withdraw the shapegen claim; state the triangle inequality explicitly; the honest Act-2 sentence is "matches the incumbent on shapenet".
**Because:** ½[d(A,C)+d(B,C)] ≥ ½·d(A,B) for any metric; changing the form (own neighbour) takes r to 0.66 in encoder space; leaving the encoder's space finishes it (+0.45 / −0.36).
**Rejected:** Rescuing the form with cosine or L2 (tested; cosine is worst).
**Implication:** Prologue + fig60/fig61 added to the artifact; README and packet corrected.
**Steering:** joint — the lead asked whether the entanglement had been removed everywhere.
**Confidence:** lead (TB): high once shown; I had assumed the entanglement was gone everywhere and asked to be sure *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high: r = 0.993 is not subtle; I should have checked this before the packet.

<a id="D05"></a>
## D05 — 2026-09-19 — Coverage (training mass within ε) adopted as the primary ruler
**State:** Within-trial result solid but pooled rows fail the base-DINOv2 control (D02); lead asked for density-based rather than nearest-neighbour estimates.
**Observation:** coverage = −log(1 + #category objects within cosine ε), mean over trial images. voxel16: within-trial r −0.424 (kNN −0.329); pooled −0.268 with control −0.030 (kNN −0.223 / −0.137). Split-half over ε: wins 20/20, held-out −0.414 vs −0.328, control −0.041 vs −0.143. Holds ε 0.03–0.20 on d57 and voxel16.
**Decision:** Coverage replaces kNN mean as the primary estimate.
**Because:** The hard cut-off saturates atypicality — "no neighbours" is the most shift there is — so it cannot spread into the pooled row; soft kernel mass, Gaussian KDE and percentile variants do not reproduce this.
**Rejected:** Gaussian KDE with per-category bandwidth (r −0.08, bandwidths differ 20×).
**Implication:** First single-model pooled measure that passes the control; fig50/52/56.
**Steering:** joint — the lead asked for density-based estimates; the agent built and validated them.
**Confidence:** lead (TB): high — coverage was my ask and the within-trial figures convinced me; I over-read the on-category curve (see D07) *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high: split-half 20/20 across ε and descriptors; medium on why the hard cut-off matters.

<a id="D04"></a>
## D04 — 2026-09-18 — Human RT / accuracy results set aside (external feedback)
**State:** Optimistic packet drafted with human accuracy (+0.20) and RT (−0.31) vs geometric distance — opposite in sign to the model effect — proposed as the lead result.
**Observation:** Lead: MOCHI's ShapeNet trials were selected adversarially, so a trial-geometry ↔ human-performance relationship may be a property of trial construction.
**Decision:** Remove it from the packet and artifact.
**Because:** Human measures are trial-constant across the 12 models, so no within-trial version exists to separate construction from robustness.
**Rejected:** Keeping it as a secondary result.
**Implication:** Listed under "set aside" in STATE.
**Steering:** lead.
**Confidence:** lead (TB): high — I know how MOCHI was built *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high: this was the lead's knowledge of how MOCHI was built, not mine.

<a id="D03"></a>
## D03 — 2026-09 (recalled) — Within-category relationship weak; exhaustive search finds a ceiling
**State:** Within-trial design established (D02); the rank curve is mostly a step (on-category model far above the other eleven).
**Observation:** On-category row, one point per trial, n = 706: voxel16 r −0.097, category-centred +0.014. Hill climb of 1,895 candidates (8 representations × ~20 estimators × 12 trial-level metrics, 50 × 5-fold CV over trials, permutation null on the max): held-out 0.113 vs fixed baseline 0.133 — nothing wins.
**Decision:** Read at the time as "the metric is weak".
**Because:** In that row "which model" and "which category" are one variable; in-sample winners (0.184, corrected p 0.001) did not generalise.
**Rejected:** Fitted-weight combinations of descriptors (excluded by design); further single-combination search.
**Implication:** The pessimism conflated a weak answer to Q2 with the strong answer to Q1 — separated at D05–D07.
**Steering:** agent.
**Confidence:** lead (TB): low — this is where I stopped believing the project; I did not yet see that Q1 and Q2 were different questions *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high that the search found nothing; medium about what that implied — I read a weak Q2 as a weak project.

<a id="D02"></a>
## D02 — 2026-09 (recalled) — The pretrained control fails pooled; adopt the within-trial design and the oddity-blind estimator
**State:** First results looked finished (D01); no control had been run.
**Observation:** Base DINOv2, never fine-tuned on any set, tracks the pooled geometric shift as strongly as the fine-tuned models (r −0.121 vs −0.135). Within a trial the pretrained margin, d(A,B) and the human data have sd = 0.
**Decision:** Rank or centre within trial, so every trial-level property — including the control — is flat by construction; switch the estimator to mean-over-images kNN (never differencing A and B).
**Because:** The pooled relation is a stimulus property (atypical objects are hard for every model); the shared-neighbour form is bounded by ½·d(A,B). Result: 84.1% of 706 slopes negative, permutation p 1e-4, category-cluster CI [−0.22, −0.03], accuracy 81 → 56% with the pretrained model flat.
**Rejected:** Keeping the pooled row as the headline.
**Implication:** Q1 answered; fig7/9/46/48. The D01 encoder comparison still used the old form — not revisited until D06.
**Steering:** joint.
**Confidence:** lead (TB): high — the design argument is arithmetic and the figures showed it *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high: the within-trial invariants are arithmetic.

<a id="D01"></a>
## D01 — 2026-09 (recalled) — Build a model-free shift ruler; first results look finished
**State:** The manuscript's shift is computed in the encoder's own feature space; the in-repo audit shows r(shift, ‖φ‖₁) 0.95 (ResNet-50), 0.89 (DeiT); MOCHI ShapeNet/ShapeGen geometry is known.
**Observation:** Replacing φ in Eq. 1–3 with geometric descriptors (d57, voxel grids from ShapeNet voxels; 20-d silhouettes for ShapeGen) gives shapegen |r| 0.74–0.80 vs incumbent 0.53–0.64 across three encoders; the pooled 8,472-point curve has binned r −0.84.
**Decision:** Treat the circularity critique as answered; the geometric ruler as a drop-in replacement.
**Because:** No learned parameter anywhere in the ruler; the numbers beat the incumbent.
**Rejected:** Mesh-based descriptors (no mesh library on the cluster); pixel-space metrics (the audit shows pix_l1 = a coverage confound).
**Implication:** Superseded by D02 (control) and D06 (the shapegen result was d(A,B)).
**Steering:** agent.
**Confidence:** lead (TB): high at the time — the numbers beat the incumbent; in hindsight, misplaced *(drafted by the agent from the record; TB to correct)* · agent (Claude Opus 5): high: the numbers were clean; low, in hindsight, about what they meant.
