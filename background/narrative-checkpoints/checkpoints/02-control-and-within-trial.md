# Checkpoint 2 — The control tracks it; the within-trial design

*Written from the vantage point of the first control failure and its fix.*

## The control

Base DINOv2 never saw any of the 12 fine-tuning sets, so a valid shift measure should
leave its margin unpredicted. Pooled over the 8,472 observations it did not:
fine-tuned r = −0.135, **pretrained r = −0.121** (accuracy; `fig25` A). The two lines fall
in parallel. Caption written at the time (`fig16`): *"the pooled correlation is therefore
a stimulus property, not a distributional effect."* Objects geometrically far from every
training set are also hard for every model.

## The fix: remove trial difficulty by design

Within a trial the pretrained margin, d(A,B), human accuracy and RT are all constants
(verified sd = 0.000e+00). So centre or rank within trial, and every trial-level property
— including the control — becomes a flat line by construction.

- `fig7` — two-way centred (within trial and within training set), 30 bins: geometric
  binned r −0.845 / trial-level −0.316; incumbent −0.968 / −0.448.
- `fig9` — rank each trial's 12 training sets nearest→farthest; every point holds the
  same 706 trials. Margin 0.174 (rank 1) → 0.053 (rank 12).
- `fig11, 12, 14, 16, 17, 32, 33, 35` — the same relationship under seven centring
  schemes, eight representations, nine estimators, one-scalar descriptors, and per
  category. It holds under all of them and is flat when objects are randomly reassigned.

Along the way the estimator changed to the **oddity-blind** form (`blind_shift.py`):
mean over all a trial's images of the mean cosine distance to the 50 nearest training
objects, never differencing A and B. Reason: the shared-neighbour form is bounded below
by ½·d(A,B).

## The formal version (added later, same design)

`fig46_moving_training.png`: one OLS slope per trial, mean −0.171, **84.1 % negative**;
permutation null from 10,000 within-trial shuffles p = 0.0001; cluster bootstrap over the
12 training categories 95 % CI [−0.219, −0.028]; control r = 4×10⁻¹⁸. Accuracy 81.2 % →
55.7 % nearest→farthest with the pretrained model pinned at 50.6 % (`fig48`). On- vs
off-category margin advantage +0.134 vs +0.019, d = 1.41.

## Revisions to Checkpoint 1

- "The pooled figure shows a distributional effect" → it shows training exposure *plus*
  object atypicality, and only the within-trial design separates them.
- The entangled Eq. 1–3 shift was replaced by the oddity-blind form for everything from
  here on. The Checkpoint-1 encoder comparison still used the old form — not revisited
  until Checkpoint 4, where it does not survive.

## What we believed

That the margin responds to the training distribution, measured without an encoder, with
a control that is zero by arithmetic. This belief has held.
