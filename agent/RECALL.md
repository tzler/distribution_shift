# RECALL — the numbers that get quoted, with their source
Check here before repeating a number. Each line: value · where it came from · entry.

## Design
- 706 usable ShapeNet trials × 12 category fine-tunes = 8,472 observations · `coverage_shapenet_voxel16_percat.csv` · D02
- within-trial sd of pretrained margin, d_AB, human acc, human RT = 0.000e+00 · `scratch/fig_moving_training.py` sanity block · D02

## Within-trial (Q1)
- slopes negative 84.1 % of 706; mean −0.171; permutation p = 0.0001; category-cluster CI [−0.219, −0.028] · `fig46` · D02
- accuracy 81.2 % → 55.7 % rank 1→12; pretrained 50.6 % flat · `fig48` · D02
- on/off-category margin advantage +0.134 / +0.019, d = 1.41 · `fig46` D panel · D02
- two-way centred trial-level r: coverage −0.436, knn −0.350, original −0.448 · `fig55` · D05, D06

## Coverage vs distance (voxel16)
- within-trial r −0.424 vs −0.329; pooled −0.268/ctrl −0.030 vs −0.223/−0.137 · `coverage_sweep_voxel16.csv` · D05
- split-half 20/20; held-out −0.414 vs −0.328; ctrl −0.041 vs −0.143; ε chosen 0.10–0.15 · session 19 Sep · D05
- zero-coverage share at ε = 0.12: 85 % off-category pairs, 13 % on-category · `fig51` · D05

## The original metric
- r(entangled shift, d_AB): 0.993 shapegen, 0.996 shapenet (geometric); 0.73/0.75/0.78 under L1/unit-L1/cosine in DINO ViT-B space · `fig49`, `fig61` · D06
- floor → pretrained margin +0.57; excess → −0.17; sum +0.32 (DINO ViT-B, n = 2,019) · `fig61` C · D06
- r(shift, ‖φ‖₁) = 0.949 ResNet-50, 0.893 DeiT · `../L1norm_vs_distshift/trials_*.csv` · D01
- on our data: pooled r −0.013 (ft) / +0.381 (pre); on-category +0.331 / +0.608 · `fig56`, `fig54` · D06
- shapegen encoder comparison with blind metric: |r| 0.74 → 0.12 · `fig49` · D06

## Within-category (Q2)
- on-category voxel16 r −0.097 (p 0.01), control −0.004; category-centred +0.014 · `fig41` · D03
- hill climb: 1,895 candidates; held-out 0.113 vs baseline 0.133; wins 28 %; in-sample max 0.184, corrected p 0.001 · `hc_repeated.py` · D03
- coverage on-category binned −0.78 → category-centred +0.013 (p 0.72); per-category mean r +0.04; 12 category points r +0.30 (p 0.34) · `fig57` · D07
- level-3 residualised: obj coverage −0.018; img coverage −0.070 (p 0.06) · session 19 Sep · D09
- power: n = 706 sees |r| ≥ 0.105; n ≈ 59 sees ≥ 0.36 · D09

## Viewpoint
- projector IoU vs real renders 0.77–0.95 (identity axes; 0.2–0.5 otherwise); MOCHI pose IoU 0.856; view gap median 20.9° · `viewdepth_pipeline.py` · D08
- within-trial r: object −0.424; depth 15 views −0.436; random view −0.426; nearest view −0.422; actual view −0.266; best-posed tertile −0.29 vs −0.41 · `fig58` · D08

## Cost
- one fine-tune ≈ 2 GPU-h (3 min/epoch × 30 + 12 min eval) ≈ $9 unsub / $2 sub; project to 19 Sep ≈ $25 / $6.50; core plan ≈ $440 / $100 · `compute_ledger.csv`, RESOURCES.md · D12, D14

## Interventions (19–21 Sep; models we trained)
- Round 1 knockout (remove 10 nearest of ~2,000): within-trial contrast −0.001 (chair, airplane), +0.000 (table); run-to-run per-trial sd 0.019 between random subsets; seed floor 0.0035 · `analyze_knockout.py` · D18, D26
- Our full-bank chair model 0.155 vs the collaborator's 0.241 on the same chairs · D18
- Chair clusters (N = all/50/25): own-kind advantage +0.009 / +0.012 / +0.015, largest of the 6 pairings each time · `analyze_clusters.py` · D21
- Distance predicts the winner within trial: r −0.21 / −0.26 / −0.24; label alone +0.17/+0.23; partial −0.15 → −0.09 · `analyze_clusters_shift.py` · D22
- Random 100-chair subsets on bank trials: within-trial r −0.077, perm p 0.001 · D26
- Round 3: 34 models × 11,634 trials; within-trial r −0.13, 70 % of trials slope down, null sd 0.0015; rank curve +0.027 → −0.008 · `analyze_round3_simple.py` · D29
- Relative measure: 92 % of variance is the trial's level; r(level, pretrained) +0.82; r(level, distance) +0.07; within-trial r −0.21 · `fig_absolute_vs_relative.py` · D31
- Ladder (chair, 3 models at N = 25): level share 85 / 54 / 45 %; r(level, pretrained) 0.79 / 0.44 / 0.34; own-kind +0.015 / +0.039 / +0.037; from scratch r 0.28, accuracy 0.51 (pretrained ViT-S 0.63) · `analyze_ladder.py` · D34, D35
- Search (held-out): DINOv2/nearest −0.236, bbox/nearest −0.232, voxel16/nearest −0.212; other-category DINOv2 −0.155, geometry ≤ −0.10; coverage worst · `out/distance_search.csv` · D36
- Transfer to MOCHI: DINOv2 −0.26 within / −0.16 across; bbox −0.17; voxel16 −0.09 · `out/transfer_mochi.csv` · D37
- Pair-distance confound on bank trials: r(distance to training, pair distance) = 0.81; pretrained rise +0.18 → +0.02 held fixed · D26
- Costs: project ≈ $449 / $101, 100 GPU-h, 89 jobs (ledger 21 Sep)
