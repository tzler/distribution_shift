# Checkpoint 5 — Viewpoint, the three levels, and the experiments we launched

*Written 19 September 2026, evening.*

## Images, not objects (Act 8)

Hypothesis: encoders learn view-conditioned appearance, so coverage should be counted
over training *images* (shape × viewpoint), not objects. Built model-free: each object's
voxels projected from a camera to a 32×32 depth + silhouette map (projector calibrated
against real renders, IoU 0.77–0.95); MOCHI poses recovered by silhouette matching over
800 candidate cameras (mean IoU 0.856); bank = 20,885 objects × 15 fixed training views.
(`scratch/voxdepth.py`, `scratch/viewdepth_pipeline.py`)

| test descriptor | within-trial r |
|---|---|
| object voxels | −0.424 |
| depth maps, 15 training views averaged | −0.436 |
| depth maps, one random training view | −0.426 |
| depth maps, nearest training view | −0.422 |
| **depth maps, the actual MOCHI view** | **−0.266** |

Any on-grid view recovers the full effect; only the off-grid view loses it, and not from
pose error (in the 395 best-posed trials, image −0.29 vs object −0.41 on the same trials).
Robust to silhouette/depth/pooling variants. **Not supported**: within the ~25° gaps of the
training grid the fine-tuned support behaves as view-invariant, consistent with the
multi-view contrastive objective. Image-space coverage fails the pooled control everywhere
(+0.06 to +0.16). (`fig58`)

## The three levels

1. A better *form* (oddity-blind, coverage) — done. Inside the encoder's space alone it
   does **not** pass the control (sep-cosine vs ImageNet bank: r with d(A,B) 0.66, pretrained
   margin +0.17): level 2 is necessary.
2. A model-free *space* — done; this is what makes the control pass.
3. *Within-condition* variance — the weakest, and for a structural reason: within a
   category with one training set, "far from the training set" and "an unusual object" are
   the same fact. The strongest level-3 test the data allow (on-category, category-centred,
   pretrained margin partialled out of both sides): object coverage −0.018; image coverage
   −0.070, p = 0.06. Power: n = 706 sees |r| ≥ 0.10; per category |r| ≥ 0.36.

## The experiments (launched 19 Sep)

Not a better metric — a different design, so that "which training set" and "which
category" come apart. All in `Dist-shift-data/knockout/`; the collaborator's pipeline is
reused untouched via a private `train.py` copy with two hard-coded paths redirected.

- **Track A, encoder-space upper bound** (job 8519397): per-image features of 311k training
  renders + 2,601 MOCHI images under pretrained DINOv2-L and the chair / airplane / table
  fine-tunes. If the chair model's own space shows no within-category distance→margin
  relation, no descriptor will.
- **Track B, support knockout** (`design.json`, 30 fine-tunes): chair / airplane / table;
  4 random balanced groups of test objects; for each group and k ∈ {10, 50}, remove each
  member's k nearest training objects (k = 10 removes ~10–16 % of the bank and raises the
  own group's knn_mean 3–4× more than other groups', 5× more than a size-matched random
  removal); plus 2 random controls per category. Each trial ends up with 11 training sets,
  Δknn 0 to +0.2, stimulus fixed. Pilot: chair g1 k10 (job 8519606, fourth attempt after
  three missing-dependency / hard-coded-path failures).
- **Knock-in** (`design_knockin.json`, 20 fine-tunes from pretrained): 8 random subsets of
  100 chairs, 6 targeted (the 100 nearest chairs to each of six low-pretrained-margin chair
  trials; 2–3× closer than any random subset), 6 cross-category controls (nearest
  airplanes). The random runs make a metric search free: any candidate x is recomputed on
  the same runs, within trial. Then targeted runs with the winner and runner-up.

## Revisions to Checkpoint 4

None. This checkpoint adds the viewpoint result and the reason level 3 needs an experiment.

## What we believe now

The margin is a function of the training distribution, and coverage of the shape —
independent of viewpoint within the training grid — is the right model-free ruler for it.
Whether that relationship is graded within a category is an open question that the
knockout and knock-in runs are designed to answer, and no analysis of the existing 12
models can.
