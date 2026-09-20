# REASONING — Distribution shift and the oddity margin
Append-only; newest entry at the top. Entries are true at their date; supersede, never edit.
Cited from STATE.md as [Dxx]. Entries before 2026-09-18 are back-filled and marked (recalled).

Template — one line each:
**State:** the general situation at the time · **Observation:** the specific thing being engaged with ·
**Decision:** what we concluded or chose · **Because:** the reasoning · **Rejected:** alternatives and why not ·
**Implication:** what changes / next steps ·
**Steering:** who drove it — lead / agent / joint (note when the lead is deferring) ·
**Confidence:** one line per contributor, by role and identity — `lead (TB)`, `agent (Claude Opus 5)` — low / medium / high, and what they are unsure of.

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
