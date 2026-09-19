# REASONING — Distribution shift and the oddity margin
Append-only; newest entry at the top. Entries are true at their date; supersede, never edit.
Cited from STATE.md as [Dxx]. Entries before 2026-09-18 are back-filled and marked (recalled).

Template — one line each:
**State:** the general situation at the time · **Observation:** the specific thing being engaged with ·
**Decision:** what we concluded or chose · **Because:** the reasoning · **Rejected:** alternatives and why not ·
**Implication:** what changes / next steps.

<a id="D14"></a>
## D14 — 2026-09-19 — Batch 1 submitted; cost made explicit; k = 50 held; documentation instructions revised
**State:** Pilot at ~3 min/epoch (D12); eval step verified on its epoch-10 checkpoint (same columns, pretrained margins bit-identical to the reference); all 50 subset directories and per-condition similarity tables built; lead offline for the afternoon, asked for resource-rational compute and for the cost in money.
**Observation:** PARCC rates: full B200 $4.51/h unsubsidised, $1.00 subsidised; mig45 exactly a quarter of that, so a quarter slice saves money only if the job runs faster than quarter speed. A run is ~2 GPU-hours (train + eval): ~$9 / ~$2. Account cap 60M billing-min, 17% used; the whole design would add ~12%.
**Decision:** Submit batch 1 to dgx-b200 (21 runs: knockout k = 10 for all groups + random controls, and the 7 remaining random knock-ins; job IDs in `knockout/logs/batch1_jobs.txt`); hold the 15 k = 50 knockouts until k = 10 results say they are needed; route the 12 targeted/cross knock-ins by the mig45 timing run (8520577); 4.5 h time limits so jobs backfill. Write HANDOFF v2 recording how the documentation system changed in first use.
**Because:** Batch 1 ≈ $200 / $44, all 50 ≈ $450 / $100 — small either way, but the k = 50 dose is the least informative third and can wait a day; done-this-weekend is met either way.
**Rejected:** Submitting everything at once (30 % of the budget on the least specific manipulation before seeing the specific one).
**Implication:** Results arrive in `knockout/eval/<cat>_<cond>/ood_analysis_results.csv` over the next ~10 h; analysis script to be written while they run.

<a id="D13"></a>
## D13 — 2026-09-19 — D11's framing corrected: the reviews were not the source of the findings
**State:** Reviews ingested (D11) with a table mapping each concern to a later result, as if the work had been a response to them; lead reviewing the trace.
**Observation:** Lead: at the time the reviewers' language was hard to interpret and the honest read was "we were right and miscommunicated"; the sense that they were right came only as our own results pushed back. The trace should show that state faithfully, not a retrofitted clarity.
**Decision:** State 0 now records what the reviewers said, what was unclear to us, and what we believed; the recognitions ("this may be what they meant") move to the states where they occurred — State 1 (SCwg / the floor), State 3 (r64w / the control), State 4 (r64w / the object-based measure). The mapping table in background/ is replaced by a note on what was unclear and pointers to those states.
**Because:** Retroactively aligning the work to the reviews overstates both their clarity and our foresight; the point of the trace is how understanding actually changed.
**Rejected:** Editing D11 in place (append-only); dropping the recognitions entirely (they are real, just later).
**Implication:** Supersedes D11's third sentence. Pattern for future external feedback: log what was said and what we made of it then; log recognitions when they happen.

<a id="D12"></a>
## D12 — 2026-09-19 — Pilot fine-tune timed; the full 50-run design is affordable and goes ahead
**State:** Interventions designed (D09) but unsubmitted, waiting on the cost of one fine-tune; the pilot (chair, group 1, k = 10) reached training on its fifth attempt after four missing-dependency / hard-coded-path failures.
**Observation:** ~3 min per epoch on a dedicated B200 once the file cache is warm (epochs 5–10: 15 min); 30 epochs ≈ 1.5–2 h. The subset pipeline behaves: 1,831 chairs discovered, 20,000 similarity-binned triplets per epoch, checkpoints saving on val-loss improvement. All 50 runs ≈ 90 GPU-hours.
**Decision:** Run the whole design as independent 1-GPU jobs, in the order random knock-in → knockout k = 10 → targeted and cross knock-in → knockout k = 50; keep the recipe pinned (20k triplets, 30 epochs) even for 100-object subsets; give each condition its own similarity table filtered to its objects so binned mining fills every epoch; chain the MOCHI evaluation into each job.
**Because:** Half the estimated cost; small subsets would otherwise silently under-fill the epoch (partners missing from the subset are rejected by the miner); a bare checkpoint without its evaluation is not a result.
**Rejected:** Reducing triplets per epoch for small subsets (changes the recipe; comparability matters more than 40 GPU-hours). Whole-node requests (wait far longer than 1-GPU jobs).
**Implication:** ~a day of wall time at 6–8 jobs in parallel; each run ends with `eval/<cat>_<cond>/ood_analysis_results.csv` in the same format as the 12 category models.

<a id="D11"></a>
## D11 — 2026-09-19 — The NeurIPS reviews ingested into the trace (external feedback)
**State:** Documentation system live (D10); State 0 had a placeholder for the reviewer concerns.
**Observation:** Three reviews (June 2026) and a post-rebuttal follow-up: the human results are real and the statistics sound; the word "distribution shift" is doubted by all three — inter-class distance (SCwg); within-set weirdness with no shift required, test it on two random halves of one set (r64w); compare against existing shift metrics, scope, overclaims (HVBU). r64w recommends resubmitting after a large reframing or dropping the framing.
**Decision:** Record the reviews paraphrased in background/ and map each concern to what we later found — the ½·d(A,B) floor is SCwg's inter-class distance [D06]; the pretrained control is r64w's two-halves test in a stronger form [D02]; the object-based view-averaged measure r64w asked for is what we built and it wins [D08].
**Because:** The trace should show that the reviewers were pointing at the same defect we found, in different words — that is the argument for the resubmission's framing.
**Rejected:** Quoting review text verbatim in the repo.
**Implication:** State 0 now opens with the reviews; the framing decision (claim category coverage vs wait for within-category) is TB's, listed in STATE next steps.

<a id="D10"></a>
## D10 — 2026-09-19 — Adopt the STATE/REASONING documentation system
**State:** Q1 answered and written up as an artifact; Q2 handed to experiments; a RECAP + five narrative checkpoints drafted that afternoon.
**Observation:** Lead supplied the lab HANDOFF: STATE = current truth, REASONING = why, evidence = only what is cited, git = history; wants the process presentable and projects citable.
**Decision:** Initialise a git repo on branch `docs-system`; replace the draft with STATE / REASONING / meetings / evidence / background; keep the narrative draft under background/ as source.
**Because:** Two "current truth" documents drift; the HANDOFF form is what the lab will read.
**Rejected:** Keeping RECAP + checkpoints as the primary record.
**Implication:** Every substantive STATE change now ships with a Dxx in the same commit; add the MOCHI project's STATE.md to background/ when it exists.

<a id="D09"></a>
## D09 — 2026-09-19 — Run interventions instead of searching for another metric
**State:** Coverage is the best ruler for Q1; every estimator gives r ≈ 0 within category after category centring (D07); the level-3 test on the training-specific margin is −0.02 (object) / −0.07, p 0.06 (image).
**Observation:** With one training set per category, "far from the training set" and "unusual object" are one variable — no metric can separate them; only varying the training set within a category can.
**Decision:** Launch (a) encoder-space upper bound — features of 311k renders + MOCHI under pretrained and chair/airplane/table fine-tunes (job 8519397); (b) support knockout — chair/airplane/table, 4 random groups × k∈{10,50} nearest-neighbour removal + 2 size-matched random controls, 30 fine-tunes, pilot 8519673; (c) knock-in from pretrained — 8 random + 6 targeted + 6 cross-category subsets of 100 chairs, designed.
**Because:** A knockout raises the own group's knn_mean 3–4× more than other groups' and 5× more than random removal, so each trial gets 11 training sets with Δknn 0–0.2, stimulus fixed; random knock-in runs make metric comparison free (any x recomputed on the same runs).
**Rejected:** Hyperparameter search (LoRA rank, lr) — pinned to the original recipe; only epochs earns an ablation. Radius-based knockout — did not transfer across categories (chairs have ~3 neighbours at ε, airplanes ~280).
**Implication:** Collaborator pipeline reused unchanged via a patched `train.py` copy; five pilot attempts fixed missing deps and hard-coded paths; run budget waits on the pilot's epoch time.

<a id="D08"></a>
## D08 — 2026-09-19 — View-conditioned (image-level) coverage tested; not supported
**State:** Object-level coverage established (D05); lead's hypothesis that encoders learn view-dependent appearance, so the training distribution that matters is over images (shape × viewpoint).
**Observation:** Built model-free depth-from-voxels descriptors (projector IoU 0.77–0.95 vs real renders; MOCHI poses recovered by silhouette matching, IoU 0.856; bank = 20,885 objects × 15 fixed training views). Within-trial r: object voxels −0.424; depth maps at all 15 training views −0.436, one random view −0.426, nearest view −0.422, the actual MOCHI view −0.266 (best-posed tertile −0.29 vs object −0.41 on the same trials). Image-space pooled control fails (+0.06 to +0.16).
**Decision:** Fine-tuned support behaves as view-invariant within the training grid's ~25° gaps; use object-level coverage.
**Because:** Any on-grid view recovers the full effect, so the descriptor is fine and single-view noise is not the cause; only the off-grid viewpoint loses it — consistent with the multi-view contrastive objective. Robust to silhouette/depth/pooled variants.
**Rejected:** Pseudo-depth for non-ShapeNet MOCHI (reintroduces a model; no per-category fine-tunes there).
**Implication:** Untested beyond ~25° (needs new renders); Act 8 of the artifact.

<a id="D07"></a>
## D07 — 2026-09-19 — The on-category coverage curve is category identity; Q2 is a design problem
**State:** Coverage adopted (D05); lead read its on-category row (binned r −0.78, clean control) as the within-category result the paper wanted.
**Observation:** The bins sort by category: ">100 neighbours" is airplane/bench/car/lamp/telephone/watercraft (homogeneous shapes), "0" is chair/table/sofa/cabinet/display/loudspeaker (diverse, biggest banks). Category-centred r +0.013 (p 0.72); within each category mean r +0.04, four of twelve negative; twelve category points r +0.30 (p 0.34), chair the counterexample. Seven coverage variants incl. category-calibrated percentile: all ≈ 0.
**Decision:** The within-category question is not identifiable with one training set per category; stop searching estimators for it.
**Because:** The pretrained control rules out "hard for every model", not "this category's model is good at its category"; category centring is the test, and everything fails it.
**Rejected:** More estimator variants on the existing 12 models (two exhaustive searches, same answer).
**Implication:** → D09. Anatomy figure fig57 added to the artifact so the reasoning is visible.

<a id="D06"></a>
## D06 — 2026-09-19 — D01's shapegen headline withdrawn: the entangled shift was d(A,B)
**State:** Packet and artifact drafted with "model-free beats the incumbent on shapegen" as a lead result; lead asked whether the A/B entanglement had been removed everywhere.
**Observation:** The Eq. 1–3 shift correlates with d(A,B) at r 0.993 (shapegen); d(A,B) alone predicts the encoder proxy better (0.79) than the shift (0.74); the oddity-blind form gives |r| 0.12. The incumbent's r with d(A,B) is 0.07 — it was never riding this. Every distance obeys the bound (L1 / unit-L1 / cosine: floor holds 100%, r with d(A,B) 0.73 / 0.75 / 0.78). Decomposed: the floor predicts the pretrained margin +0.57 (wrong sign), the excess −0.17 (right sign).
**Decision:** Withdraw the shapegen claim; state the triangle inequality explicitly; the honest Act-2 sentence is "matches the incumbent on shapenet".
**Because:** ½[d(A,C)+d(B,C)] ≥ ½·d(A,B) for any metric; changing the form (own neighbour) takes r to 0.66 in encoder space; leaving the encoder's space finishes it (+0.45 / −0.36).
**Rejected:** Rescuing the form with cosine or L2 (tested; cosine is worst).
**Implication:** Prologue + fig60/fig61 added to the artifact; README and packet corrected.

<a id="D05"></a>
## D05 — 2026-09-19 — Coverage (training mass within ε) adopted as the primary ruler
**State:** Within-trial result solid but pooled rows fail the base-DINOv2 control (D02); lead asked for density-based rather than nearest-neighbour estimates.
**Observation:** coverage = −log(1 + #category objects within cosine ε), mean over trial images. voxel16: within-trial r −0.424 (kNN −0.329); pooled −0.268 with control −0.030 (kNN −0.223 / −0.137). Split-half over ε: wins 20/20, held-out −0.414 vs −0.328, control −0.041 vs −0.143. Holds ε 0.03–0.20 on d57 and voxel16.
**Decision:** Coverage replaces kNN mean as the primary estimate.
**Because:** The hard cut-off saturates atypicality — "no neighbours" is the most shift there is — so it cannot spread into the pooled row; soft kernel mass, Gaussian KDE and percentile variants do not reproduce this.
**Rejected:** Gaussian KDE with per-category bandwidth (r −0.08, bandwidths differ 20×).
**Implication:** First single-model pooled measure that passes the control; fig50/52/56.

<a id="D04"></a>
## D04 — 2026-09-18 — Human RT / accuracy results set aside (external feedback)
**State:** Optimistic packet drafted with human accuracy (+0.20) and RT (−0.31) vs geometric distance — opposite in sign to the model effect — proposed as the lead result.
**Observation:** Lead: MOCHI's ShapeNet trials were selected adversarially, so a trial-geometry ↔ human-performance relationship may be a property of trial construction.
**Decision:** Remove it from the packet and artifact.
**Because:** Human measures are trial-constant across the 12 models, so no within-trial version exists to separate construction from robustness.
**Rejected:** Keeping it as a secondary result.
**Implication:** Listed under "set aside" in STATE.

<a id="D03"></a>
## D03 — 2026-09 (recalled) — Within-category relationship weak; exhaustive search finds a ceiling
**State:** Within-trial design established (D02); the rank curve is mostly a step (on-category model far above the other eleven).
**Observation:** On-category row, one point per trial, n = 706: voxel16 r −0.097, category-centred +0.014. Hill climb of 1,895 candidates (8 representations × ~20 estimators × 12 trial-level metrics, 50 × 5-fold CV over trials, permutation null on the max): held-out 0.113 vs fixed baseline 0.133 — nothing wins.
**Decision:** Read at the time as "the metric is weak".
**Because:** In that row "which model" and "which category" are one variable; in-sample winners (0.184, corrected p 0.001) did not generalise.
**Rejected:** Fitted-weight combinations of descriptors (excluded by design); further single-combination search.
**Implication:** The pessimism conflated a weak answer to Q2 with the strong answer to Q1 — separated at D05–D07.

<a id="D02"></a>
## D02 — 2026-09 (recalled) — The pretrained control fails pooled; adopt the within-trial design and the oddity-blind estimator
**State:** First results looked finished (D01); no control had been run.
**Observation:** Base DINOv2, never fine-tuned on any set, tracks the pooled geometric shift as strongly as the fine-tuned models (r −0.121 vs −0.135). Within a trial the pretrained margin, d(A,B) and the human data have sd = 0.
**Decision:** Rank or centre within trial, so every trial-level property — including the control — is flat by construction; switch the estimator to mean-over-images kNN (never differencing A and B).
**Because:** The pooled relation is a stimulus property (atypical objects are hard for every model); the shared-neighbour form is bounded by ½·d(A,B). Result: 84.1% of 706 slopes negative, permutation p 1e-4, category-cluster CI [−0.22, −0.03], accuracy 81 → 56% with the pretrained model flat.
**Rejected:** Keeping the pooled row as the headline.
**Implication:** Q1 answered; fig7/9/46/48. The D01 encoder comparison still used the old form — not revisited until D06.

<a id="D01"></a>
## D01 — 2026-09 (recalled) — Build a model-free shift ruler; first results look finished
**State:** The manuscript's shift is computed in the encoder's own feature space; the in-repo audit shows r(shift, ‖φ‖₁) 0.95 (ResNet-50), 0.89 (DeiT); MOCHI ShapeNet/ShapeGen geometry is known.
**Observation:** Replacing φ in Eq. 1–3 with geometric descriptors (d57, voxel grids from ShapeNet voxels; 20-d silhouettes for ShapeGen) gives shapegen |r| 0.74–0.80 vs incumbent 0.53–0.64 across three encoders; the pooled 8,472-point curve has binned r −0.84.
**Decision:** Treat the circularity critique as answered; the geometric ruler as a drop-in replacement.
**Because:** No learned parameter anywhere in the ruler; the numbers beat the incumbent.
**Rejected:** Mesh-based descriptors (no mesh library on the cluster); pixel-space metrics (the audit shows pix_l1 = a coverage confound).
**Implication:** Superseded by D02 (control) and D06 (the shapegen result was d(A,B)).
