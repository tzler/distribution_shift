# geometric_shift — where we are (19 September 2026)

One-page recap for Monday. The checkpoints in `checkpoints/` preserve how the story looked
at each stage, including what turned out to be wrong. The shareable walkthrough is
https://claude.ai/artifact/6AhaQJwZjQoZnYEXsBWT9r (Prologue + Acts 1–8, recipes, build spec).

## The question

The manuscript's distribution-shift metric is computed inside the encoder whose margin it
explains, and it is bounded below by the trial's own difficulty. **Can a shift measure with
no learned parameters — computed from the stimuli's known 3D geometry and the training
set alone — predict the oddity margin, with a control that cannot be gamed?**

## Numbers to keep in your head

| | value | note |
|---|---|---|
| design | 706 trials × 12 category fine-tunes | trial fixed, training set varied |
| within-trial slopes negative | **84.1 %** of 706 | permutation p = 10⁻⁴, category-cluster CI [−0.22, −0.03] |
| accuracy, nearest → farthest training set | **81.2 % → 55.7 %** | pretrained pinned at 50.6 % |
| on- vs off-category margin advantage | +0.134 vs +0.019, **d = 1.41** | |
| coverage, within-trial r | **−0.424** | knn_mean −0.329 |
| coverage, pooled r / control | **−0.268 / −0.030** | knn_mean −0.223 / −0.137; original metric −0.01 / **+0.38** |
| within-category, category-centred | **≈ 0** | every estimator; the design cannot answer it |
| entangled shift vs d(A,B), shapegen | r = **0.993** | the Act-2 headline was this |
| image-level coverage at the actual view | −0.266 | vs −0.42 at any training view |

## What we established (in order of confidence)

1. **The margin responds to the training distribution.** Same images, twelve training
   sets: the margin moves in five trials out of six, and the base-DINOv2 control is zero
   by arithmetic. Robust to descriptor (57-d invariant to 32³ voxels), estimator, centring,
   binning, and a 1,895-candidate search.
2. **Coverage — training mass within ε — is the right ruler.** Only estimator whose pooled
   relationship passes the control; split-half 20/20.
3. **The original metric measures the encoder's ease, not distribution shift.** Bounded by
   ½·d(A,B) (every distance, cosine included); mostly feature norm; validated against
   itself; on our data predicts the pretrained margin (+0.38 / +0.61) better than the
   fine-tuned one, wrong sign.
4. **Support is view-invariant within the training grid.** Image-level coverage at the
   actual viewpoint predicts worse than at any training view.
5. **Within a category, the graded relationship is not identifiable with one training set
   per category.** "Far from the training set" and "unusual object" are one variable there.

## What we got wrong along the way (and fixed)

- "Model-free beats the incumbent on shapegen" → the entangled shift was d(A,B) at
  r = 0.99; withdrawn.
- "The pooled curve shows a distributional effect" → pretrained control fell with it;
  fixed by the within-trial design.
- "The rotation-invariant descriptor is null" → only in the within-category row.
- "Coverage rescues the on-category row" (binned r −0.78) → a between-category difference
  that vanishes under category centring.
- The pessimism of Checkpoint 3 conflated a weak answer to Q2 with the strong answer to Q1.

## What is running (19 Sep)

- Track A: encoder-space upper bound — features of 311k training renders under pretrained
  and chair/airplane/table fine-tunes (job 8519397).
- Track B: support-knockout pilot, chair g1 k10 (job 8519606); 29 more conditions designed.
- Knock-in: 8 random + 6 targeted + 6 cross subsets of 100 chairs, designed, waiting on
  the pilot's epoch time.

## Where things live

`Dist-shift-data/geometric_shift/` — descriptors, estimator, figures (`out/figures/`),
`README_findings.md` (full factual record), `packet/` (figure packet), `web/` (artifact
source). `Dist-shift-data/knockout/` — the experiments. Nothing in the collaborator's
`hida-tune` was modified.
