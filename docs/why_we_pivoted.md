# Why we stopped believing it — reconstructed from the record

The sequence, taken from the figure captions and outputs in the order they were made.

## 1. It looked great
`03_every_trial_x_every_model.png`, `01/02_*_geometric_vs_incumbent.png`

Pooled over all 8,472 observations: r = −0.22, binned r = −0.84, a clean monotone decline.
On shapegen the model-free metric beat the incumbent for every encoder.

## 1b. And the shapegen result was the triangle problem  *(found later)*
`A3_entanglement.png`

The Act-2 shift is Eq. 1–3 with φ swapped, bounded below by ½·d(A,B). On shapegen it
correlates with d(A,B) at r = 0.993. The oddity-blind metric takes the shapegen encoder
relationship from |r| = 0.74 to 0.12. This one was not a reason we pivoted — we had not
measured it — but it should have been.

## 2. The control tracked it too
`15_nine_estimators_and_control.png` (top row), `A1_where_we_got_pessimistic.png` (panel A)

The pretrained DINOv2 margin — from a model that never saw any of the 12 fine-tuning sets —
tracked the pooled geometric shift as strongly as the fine-tuned margin did (fig 25 A:
fine-tuned r = −0.135, pretrained r = −0.121; the two lines fall in parallel). Its caption
says exactly what that means: *"the pooled correlation is therefore a stimulus property,
not a distributional effect."* Objects that are geometrically far from every training set
are also hard for every model. The pooled figure has two things in it — training exposure
and object atypicality — and cannot separate them.

## 3. The within-trial design fixed it
`06`–`08`, `11`–`16`

Because the pretrained margin is a constant within a trial (variance ~10⁻³⁵), centring
within trial — or equivalently ranking within trial — makes the control **a flat line by
construction**. The fine-tuned margin still falls. This is the point at which the analysis
became rigorous, and it is the design behind every figure you singled out. It answers the
question the paper needs answered: *does the margin respond to the training distribution,
measured model-free, with a control that cannot be gamed?* Yes.

## 4. We then asked a second, harder question
`A2_on_category_only.png`, `A1` panel B, fig 20 / fig 21

The within-trial fall is mostly a step: the on-category model (rank 1) is far above the
other eleven, which are roughly flat. So we asked: *holding category membership fixed,
does graded geometric distance still predict the margin?* That is the **within-category**
row: keep only the model trained on each trial's own category, one point per trial, n = 706.

Answer: barely. r = −0.097 (p = 0.01); the accuracy version (fig 25 B) is r = −0.031. The
binned version in `A2` gets noisier as the bins get finer (−0.56 → −0.30), the opposite
of what a real graded relationship does.

## 5. We searched for a metric that would work there, and none did
`out/hillclimb_candidates.npz`, `scratch/hc_repeated.py`

**This was the exhaustive within-category search.** The hill climb's primary objective was
within-category margin advantage: 1,895 candidates (8 representations × ~20 estimators
× 12 trial-level metrics), 50 repeats of 5-fold cross-validation over trials, permutation
null on the max statistic. In-sample the best candidate reached |r| = 0.184 with a
permutation-corrected p of 0.001 — and out of sample it scored 0.113 against 0.133 for the
fixed baseline, losing in 72 % of repeats. There is no single model-free metric that makes
the within-category relationship strong. |r| ≈ 0.13 is the ceiling for that question in
these data.

What was *not* searched, by design: fitted linear combinations of descriptors, and
image-based (render-level) references — one attempt at the latter failed its control.

## 6. The causal analysis confirmed the shape
`10_moving_training_causal.png`, panels C2 and D

Remove the on-category model from each trial and the per-trial slopes go to the null (mean
−0.004, p = 0.29). Remove the diagonal from the 12×12 transfer matrix and r goes from
−0.364 to −0.024. Within this dataset the effect is **coverage** — did you train on this
kind of object — with at most a small graded component beyond it (partial r = −0.070 for
distance controlling for category match; real, p = 10⁻¹⁰, small).

## So: was the pessimism warranted?

**Half of it.** Steps 4–6 are a correct answer to a real question, and the answer is that
the graded within-category relationship is weak and no metric rescues it. That should be
stated plainly in the paper as a scope condition.

**The other half was a conflation.** The weak answer to question 2 got read as a weak
result overall, when question 1 — the one in steps 1–3, the one every figure you liked
addresses — had already been answered strongly. Those are different claims:

| | question | answer | strength |
|---|---|---|---|
| Q1 | Does the margin respond to the training distribution, measured model-free, with a clean control? | yes | 84 % of trials, p = 10⁻⁴, 25 accuracy points, d = 1.4, control = 0 by construction |
| Q2 | Beyond category membership, does graded distance within a category's training set predict the margin? | barely | r ≈ −0.10, ceiling ≈ 0.13 after a 1,895-metric search |

The paper needs Q1. Q2 is a finer-grained follow-up whose honest answer is "not resolvable
with 12 categorical training sets — the dose range is too narrow." The pooled figure (03)
is also not wrong; it just contains both a training effect and an atypicality effect, and
the within-trial design is how you show which part is which.

## If you want to push on Q2 anyway

Two things would change the answer, neither of which is a better metric:

- **A real dose axis.** Fine-tune on mixtures with a varying proportion of the test
  category (0, 1, 5, 25, 100 %), or on the test category with the k most similar objects
  held out, sweeping k. Either spans orders of magnitude of distance instead of a factor
  of ~5, which is what the concept-frequency literature needed to see log-linear scaling.
- **Repeat fine-tunes.** There is one model per category, so the margin's reliability is
  unknown and |r| ≈ 0.13 could be the noise floor. A second seed per category would tell
  you whether there is anything left to find.
