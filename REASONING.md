# REASONING — geometric_shift
Append-only. Entries are true at their date; supersede, never edit. Cited from STATE.md as [Dxx].
Entries before 2026-09-18 are back-filled from the session record and marked (recalled).

## D01 — 2026-09 (recalled) — Build a model-free shift ruler; first results look finished
**Decided/Observed:** Replaced φ in the manuscript's Eq. 1–3 shift with geometric descriptors from ShapeNet voxels (d57, voxel grids); shapegen Fig-1 reproduction gave |r| 0.74–0.80 vs incumbent 0.53–0.64; pooled 8,472-point curve binned r −0.84.
**Because:** Audit showed the incumbent tracks feature norm (r 0.95 ResNet-50); MOCHI ShapeNet/ShapeGen geometry is known, so no encoder is needed.
**Rejected:** Mesh-based descriptors (no mesh library on cluster); image-pixel metrics (audit shows pix_l1 = coverage confound).
**Implication:** Believed the circularity critique answered. Superseded by D02, D06.

## D02 — 2026-09 (recalled) — Pretrained control fails pooled; adopt within-trial design and oddity-blind estimator
**Decided/Observed:** Base DINOv2 (never fine-tuned on any set) tracks the pooled geometric shift as strongly as fine-tuned models (r −0.121 vs −0.135). Within a trial, pretrained margin, d(A,B), human data have sd = 0, so rank/centre within trial: control becomes flat by construction; fine-tuned margin still falls (84.1% of 706 slopes negative, perm p 1e-4, category-cluster CI [−0.22,−0.03]; accuracy 81→56%).
**Because:** Pooled relation is a stimulus property (atypical objects hard for every model). Shared-neighbour form is bounded by ½·d(A,B), so estimator changed to mean-over-images kNN (never differencing A,B).
**Rejected:** Keeping the pooled row as the headline.
**Implication:** H1 supported. evidence/fig46, fig48, fig7, fig9.

## D03 — 2026-09 (recalled) — Within-category relationship weak; 1,895-candidate search finds a ceiling; pessimism
**Decided/Observed:** On-category row (one point/trial, n=706): voxel16 r −0.097, category-centred +0.014. Hill climb (8 reps × ~20 estimators × 12 metrics, 50×5-fold CV over trials, perm null on max): held-out 0.113 vs fixed baseline 0.133; no metric beats it.
**Because:** In that row "which model" and "which category" are one variable.
**Rejected:** Further single-combination metric search on the existing 12 models (exhaustive already). Fitted-weight combinations excluded by design.
**Implication:** Read at the time as "the metric is weak"; later split into Q1 (strong) vs Q2 (not identifiable) — see D07.

## D04 — 2026-09-18 — Human RT/accuracy results set aside (external feedback)
**Decided/Observed:** Lead pointed out MOCHI ShapeNet trials were selected adversarially, so r(geometric distance, human RT) −0.31 / accuracy +0.20 may reflect trial construction.
**Because:** Human measures are trial-constant across the 12 models; no within-trial version exists to separate construction from robustness.
**Rejected:** Leading the packet with the human/model divergence.
**Implication:** Removed from packet and artifact; listed under "set aside" in STATE.

## D05 — 2026-09-19 — Coverage (training mass within ε) adopted as the primary ruler
**Decided/Observed:** coverage = −log(1+#category objects within cosine ε), mean over trial images. voxel16: within-trial r −0.424 (kNN −0.329); pooled −0.268 with control −0.030 (kNN −0.223/−0.137) — first pooled measure to pass the control. Split-half over ε: wins 20/20, held-out −0.414 vs −0.328. Holds ε 0.03–0.20, d57 and voxel16.
**Because:** Hard cut-off saturates atypicality ("no neighbours" is maximal shift) so it cannot leak into the pooled row; soft kernel-mass, KDE, percentile variants do not reproduce it.
**Rejected:** Gaussian KDE with per-category bandwidth (poorly calibrated, r −0.08).
**Implication:** H5 supported. evidence/fig50, fig52, fig56.

## D06 — 2026-09-19 — D01's shapegen headline withdrawn: the entangled shift was d(A,B)
**Decided/Observed:** Eq. 1–3 shift correlates with d(A,B) at r 0.993 (shapegen); d(A,B) alone predicts the encoder proxy better (0.79) than the shift (0.74); oddity-blind form → |r| 0.12. Incumbent's r with d(A,B) is 0.07 — it was never riding this. Every distance obeys the bound (L1/unit-L1/cosine: floor 100%, r 0.73/0.75/0.78); decomposition: floor → pretrained margin +0.57, excess −0.17.
**Because:** Triangle inequality: ½[d(A,C)+d(B,C)] ≥ ½d(A,B) for any metric; only changing the form, then the space, escapes it.
**Rejected:** Rescuing the form with cosine/L2 (tested; cosine is worst).
**Implication:** Prologue + Act 3 of artifact; H4 supported. evidence/fig49, fig60, fig61.

## D07 — 2026-09-19 — On-category coverage curve is between-category; Q2 not identifiable with this design
**Decided/Observed:** Coverage on-category binned r −0.78 looked graded; bins sort by category (>100 neighbours = airplane/bench/car/lamp/telephone/watercraft; 0 = chair/table/sofa/…). Category-centred r +0.013 (p 0.72); within each category mean r +0.04; 12 category points r +0.30 (p 0.34, chair the counterexample). Seven coverage variants incl. category-calibrated percentile: all ≈ 0.
**Because:** One training set per category ⇒ "far from training set" ≡ "unusual object" within a category. Pretrained control cannot catch category identity.
**Rejected:** More estimator variants on the existing models.
**Implication:** Q2 needs within-category training-set variation → D09. evidence/fig54, fig57.

## D08 — 2026-09-19 — View-conditioned (image-level) coverage tested; not supported
**Decided/Observed:** Depth-from-voxels descriptors (projector IoU 0.77–0.95 vs real renders; MOCHI poses recovered, IoU 0.856); bank 20,885 objects × 15 training views. Within-trial r: object voxels −0.424; depth maps at 15 views −0.436, at one random view −0.426, at nearest view −0.422, at the actual MOCHI view −0.266 (also in best-posed tertile: −0.29 vs −0.41). Image-space pooled control fails (+0.06–0.16).
**Because:** Fine-tuned support behaves as view-invariant within the ~25° grid gaps — consistent with the multi-view contrastive objective. Not pose error, not descriptor resolution (silhouette/depth/pooled variants agree).
**Rejected:** Pseudo-depth for non-ShapeNet MOCHI (reintroduces a model; no per-category fine-tunes there).
**Implication:** H3 killed within the training grid; untested beyond ~25°. evidence/fig58.

## D09 — 2026-09-19 — Run interventions instead of searching metrics
**Decided/Observed:** Launched (a) encoder-space upper bound: features of 311k renders + MOCHI under pretrained/chair/airplane/table fine-tunes (job 8519397); (b) support knockout: chair/airplane/table, 4 random groups × k∈{10,50} nearest-neighbour removal + 2 random controls = 30 fine-tunes, pilot job 8519673; (c) knock-in from pretrained: 8 random + 6 targeted + 6 cross-category subsets of 100 chairs (designed, pending pilot timing).
**Because:** D07 — the within-category question is a design problem; random knock-in subsets also make metric comparison free (any x recomputed on the same runs, within trial).
**Rejected:** Hyperparameter search (LoRA rank, lr) — pinned to the original recipe; only epochs earns one ablation.
**Implication:** Collaborator pipeline reused untouched via patched train.py copy; five pilot attempts fixed deps/paths.

## D10 — 2026-09-19 — Adopt the STATE/REASONING documentation system (this file)
**Decided/Observed:** Replaced the RECAP + narrative-checkpoints draft with STATE.md / REASONING.md / meetings / evidence / background per the lab HANDOFF; repo initialised on branch `docs-system`.
**Because:** Lead wants the process — what changed our confidence and why — presentable to the lab; projects as citable objects.
**Rejected:** Keeping two "current truth" documents.
**Implication:** narrative-checkpoints kept under background/ as source; every STATE change now needs a Dxx.
