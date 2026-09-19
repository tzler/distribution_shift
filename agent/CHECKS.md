# CHECKS — run these before claiming a result
Each caught a real error in this project. The number in brackets is the entry where it mattered.

## For any shift metric
- **Entanglement:** r(shift, d_AB) across trials. The manuscript's Eq. 1–3 form gives 0.99; anything near 1 is measuring the trial, not the training set. [D06]
- **Pretrained control:** r(shift, pretrained margin) on the same rows. A model that never saw the training set must not be predicted. Pooled rows fail this for every distance metric; coverage passes. [D02, D05]
- **Within-trial invariants:** groupby(trial).std() of pretrained margin, d_AB, human accuracy, human RT must be exactly 0 before centring; assert it. [D02]

## For any on-category / within-category claim
- **Category-centre it.** Subtract each category's mean from x and y; a binned r of −0.78 went to +0.01. Then look inside each category separately and at the 12 category means. [D07]
- **Never trust binned r on <15 bins** as evidence; report the point-level r with n, and the per-category sign count.
- **Power:** n = 706 category-centred sees |r| ≥ 0.10; per category (n ≈ 59) only ≥ 0.36. Say what is invisible.

## For any tuned parameter (ε, k, descriptor)
- **Split-half:** choose on one half of trials, score on the other, ≥20 splits; report wins and held-out values. The 1,895-candidate in-sample winner (p = 0.001 corrected) lost out of sample. [D03, D05]
- **Permutation null on the max statistic** when reporting a search's best.

## For any figure
- **A state figure states its expectation** (what the lines should do if the claim is true; what the control should do) before showing the data, one plot type repeated, verdict boxes graded green / orange / red and worded as concern vs kill honestly. Recipe in FEEDBACK.md. Only measures that existed at that state.
- **Own axis for the control** when it compresses the effect; **independent y-limits** when panels are compared on shape, not level (fig 17).
- **Coverage-type measures saturate:** bin on the raw count with log-spaced edges, not on quantiles of the score (85 % of pairs sit at zero).
- Check the rendered PNG for title/caption collisions before sending; the lead reads on a phone.

## For any within-trial "causal" statistic
- Per-trial OLS slope; permutation by shuffling x within trial (Σxc² is invariant, only the cross term moves); cluster bootstrap over the 12 training categories, not the 706 trials. [D02]

## For the intervention experiments
- Compare each new model's csv against the reference: identical columns; pretrained margins bit-identical (they must be — same pretrained model). [D12]
- Never compare an epoch-10 checkpoint to an epoch-30 reference as a result; only the chained end-of-job evaluation counts.
- Own-group vs other-group vs size-matched random removal is the contrast; report all three.
