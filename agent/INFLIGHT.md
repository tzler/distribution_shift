# INFLIGHT — what is running or waiting, right now
*Updated 2026-09-21, end of the weekend session. Paths relative to `../knockout/` unless stated.*

## Running / queued
- **Training-object ("anchor") evaluations — DONE and analysed** (D40). 42 result files in `eval_train/<cond>/`; the long table is `data/anchor_long.csv.gz`. Trials built from each cluster model's own 25 training objects at their training views (`banktrials/traintrials_all.csv`, 2,700 trials, 36 conditions; built by `scripts/make_train_trials.py`). Every one of the 36 cluster models plus the 7 ladder models is being scored on them → `eval_train/<cond>/ood_analysis_results.csv`. Jobs: `evaltrain1..4` (9 models each) and `evaltrainL` (ladder) on mig45, with full-GPU twins `evaltrain4B`, `evaltrainLB` (8580291, 8580293) — whichever twin starts, cancel the other while pending. IDs in `logs/batch3_jobs.txt`.
- Nothing else is training. The ViT-small control (`chair_full_smallft`) is evaluated on MOCHI and the all-category bank trials but its numbers have not been folded into the ladder table (D34) yet.

## First things for the next session
1. ~~Anchor trials~~ DONE 24 Sep — [D40](../REASONING.md#D40), `scratch/analyze_anchor.py`, evidence/fig82. Original instruction kept for reference: when `eval_train/*/ood_analysis_results.csv` exist for all 36 cluster models: run the D36 scorer on them (`scratch/distance_search_fast.py` reads `out/all_categories_long.csv`; build the equivalent long table for the training trials first — the loader in `scratch/analyze_all_categories.py` shows the pattern). The test: for each model, its own 25 objects (distance ≈ 0 in every measure) must get the highest margin; then the within-trial ranking of all 69 measures as before. Log as D39.
2. Fold rung 3b (`eval_bank_all/chair_full_smallft`) into `scratch/analyze_ladder.py` and D34's table: it separates the small architecture from the missing prior.
3. Lead review of states 7 and 8 (`states/07-*`, `states/08-*`); then rebuild `../../lab-trace-workspace/distribution-shift-trace` from the confirmed states (pedagogical order, one synthetic commit per state; builder `demo-src/build_demo.py`) and the viewer bundle (`lab-trace-viewer/make_bundle.py`).
4. Pushed 21 Sep: this repo → github.com/tzler/distribution_shift (branch `docs-system`; the repo has no main — the lead may set docs-system as default or merge); template → tzler/state; viewer → tzler/state_view; demo → tzler/state_distributionshift (still the chronological build — rebuild from the confirmed states, then push).

## Viewer: two bugs fixed 22 Sep, still unverified in a browser
Fixed by reading the source: (1) demo mode reversed the commit list twice, so the nine states played backwards; (2) no fallback when the CDN copy of `marked` fails to load — every pane stays blank, which matches what the lead saw. Added a small built-in markdown renderer as a fallback. Republished (artifact v4) and pushed to `tzler/state_view`. **Someone still has to open it in a browser and confirm.**

## Viewer: the original report (21 Sep lab meeting)
The viewer artifact (https://claude.ai/artifact/B4SwzRTvVVdmRVHiW46qvs, v3, bundle inlined) showed nothing for the lead in his browser; never verified in a real browser by the agent. Debug with a browser console against `lab-trace-workspace/lab-trace-viewer/index.html` opened locally (file://) with `demo-bundle.js` beside it; likely candidates: `marked` CDN blocked, the DEMO `api()` path matching, or the artifact wrapper's outer `<head>`. The GitHub repo `tzler/state_distributionshift` works as the fallback walkthrough (nine commits, open STATE.md at each).

## Held / decided against
- k = 50 knockouts on the full bank (D20); from-scratch models on subsets (D35); further tuning of measures on the bank trials (D36: fit ≈ held-out, nothing to tune).

## Where the numbers live
`out/all_categories_long.csv` (round-3 margins × voxel16 distance, 395k rows) · `out/distance_search.csv` (69 measures) · `out/transfer_mochi.csv` · `out/cluster_summary.csv` · `out/margin_vs_dose.csv` · `background/compute_ledger.csv` (run `scratch/compute_ledger.py`).
