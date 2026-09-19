# geometric_shift — model-free distribution shift for the oddity margin
*Last meaningful update: 2026-09-19 · Lead: TB · Status: active*

## Goal
The manuscript (*Human perception under distribution shift*) measures train–test distribution shift inside the same encoder whose oddity margin it explains, with a form that is bounded below by the trial's own difficulty. We want a shift measure with **no learned parameters** — from the stimuli's known 3D geometry and the training set alone — that predicts the margin, passes a control the original cannot, and makes "representational support depends on training data" a causal claim rather than a correlation.

## Hypotheses
- **H1** The oddity margin responds to which training set the model saw, measured model-free. *Status: supported* [D02].
- **H2** Within a category, the margin falls gradedly with distance from the training set. *Status: live — not identifiable with one training set per category* [D03, D07]; being tested by intervention [D09].
- **H3** Fine-tuned support is view-conditioned (training *images*, not objects, are what matter). *Status: killed within the training view grid* [D08].
- **H4** The manuscript's metric measures the encoder's ease on the trial, not distribution shift. *Status: supported* [D06].
- **H5** Coverage (training mass within ε) is the right model-free ruler. *Status: supported* [D05]; wins over kNN distance on every criterion.

## Current evidence
- **Within-trial manipulation** (706 trials × 12 category fine-tunes; images, d(A,B), base margin identical across the 12): slope negative in 84.1% of trials, perm p 1e-4, category-cluster CI [−0.22, −0.03]; accuracy 81.2 → 55.7% nearest→farthest with the pretrained model flat at 50.6%; on/off-category margin advantage d = 1.41. `evidence/fig46_moving_training.png`, `fig48`, `fig7`, `fig9` [D02].
- **Coverage vs kNN** (voxel16): within-trial −0.424 vs −0.329; pooled −0.268 / control −0.030 vs −0.223 / −0.137; split-half 20/20. `evidence/fig50`, `fig52`, `fig56` [D05].
- **Original metric**: bounded by ½·d(A,B) under every distance (r with d(A,B) 0.73–0.78); on our data predicts the *pretrained* margin (+0.38 pooled, +0.61 on-category) better than the fine-tuned one, wrong sign; works only after two-way centring. `evidence/fig49`, `fig60`, `fig61` [D06].
- **On-category row**: apparent graded curve (binned −0.78) is category identity; category-centred ≈ 0 for every estimator. `evidence/fig54`, `fig57` [D07].
- **Viewpoint**: image-level coverage matches object-level at any training view (−0.42 to −0.44) and loses it only at the actual off-grid view (−0.27). `evidence/fig58` [D08].
- Robustness: descriptor (57-d invariant → 32³ voxels), 9 estimators, 7 centring schemes, binning, 1,895-candidate CV search — all agree [D02, D03].

## Current status
Q1 (H1, H4, H5) is answered and written up (artifact Prologue + Acts 1–8; `README_findings.md`). Q2 (H2) cannot be answered by any analysis of the existing 12 models, so three interventions were launched 19 Sep: encoder-space upper bound (extraction running), support knockout (pilot on its 5th attempt after dependency/path fixes; 29 conditions designed), knock-in from pretrained (20 subsets designed, waiting on the pilot's epoch time). Blocked on: per-epoch cost of one fine-tune, which sets how many of the 50 designed runs are affordable.

## Next steps
- [ ] Pilot epoch time → decide run budget; prioritise knock-in random subsets (metric comparison is free on them) over k=50 knockouts. (Claude)
- [ ] Encoder-space battery from job 8519397: does the chair model's own space show a within-category signal? (Claude)
- [ ] Build remaining subset dirs; submit knockout + knock-in as 1-GPU jobs; evaluate each checkpoint with `ood_distance_analysis.py` exactly as the 12 category models were. (Claude)
- [ ] Metric search on the random knock-in runs; targeted runs with winner + runner-up. (Claude)
- [ ] Single-trial figure: trial images, selected training chairs, pretrained vs targeted vs random margins. (Claude)
- [ ] Lab meeting Monday: present STATE + D02/D05/D06/D07 as the arc. (TB)
- [ ] Decide whether the resubmission claims coverage (binary) or waits for the graded result. (TB)

## Open questions
1. Is the within-category relationship graded once the training set varies? (H2; knockout/knock-in.)
2. Which metric best predicts margin *change* under intervention — kNN k, coverage ε, descriptor, object vs image level?
3. Does the encoder's own representation carry a within-category signal that geometry misses?
4. What is the margin's noise floor? One model per category, no repeat seeds; the 0.13 within-category ceiling could be noise.
5. Does view-invariance hold beyond ~25° from the training grid? (Needs new renders.)
6. Epochs for small knock-in subsets: 10 vs 30 — the one nuisance variable worth an ablation.

Set aside: human RT/accuracy vs geometric distance [D04]; pseudo-depth for non-ShapeNet MOCHI [D08].

## Pointers
- Code: this repo (`blind_shift.py` estimator; `scratch/coverage_sweep.py`, `scratch/viewdepth_pipeline.py`, `scratch/fig_moving_training.py`). Data banks in `bank/` (ignored), figures in `out/figures/` (ignored; cited ones in `evidence/`).
- Experiments: `../knockout/` — `design.json`, `design_knockin.json`, `scripts/`, logs.
- Write-ups: `README_findings.md` (full factual record); artifact https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r; `packet/` (figure packet).
- Upstream: collaborator pipeline `../../Dist-shift/HIDA/hida-tune/` (read-only); category results `ShapeNet_OOD_Analyses/<cat>/ood_analysis_results.csv`; audit `../L1norm_vs_distshift/README.md`.
- Background: `background/narrative-checkpoints/` (long-form back-fill source); `background/` awaits the MOCHI project's STATE.md.

## TODOs (back-fill)
- D01–D03 dates are approximate (recalled); exact dates recoverable from `out/figures/` mtimes.
- `evidence/` provenance lists script paths; commit hashes to be added after the first commit.
- Verify the manuscript's published shapenet r = .91 (Fig. 1) — never reproduced (got +0.34); unresolved.
- `background/` is empty apart from the narrative draft; add the audit README and the MOCHI STATE.md when available.
