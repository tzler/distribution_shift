# Reviews of the NeurIPS 2026 submission — paraphrased summary

*Humans adapt to distribution shift with increasing test-time compute* (Knapp & Bonnen). Submitted May 2026;
three reviews June 2026; author responses 1 Aug; reviewer follow-up 3 Aug. Paraphrased from the OpenReview
page; no verbatim text. Source: OpenReview record, held by TB.

## The reviews

**SCwg** (quality 2 / clarity 2 / significance 2 / originality 2; confidence 2). Statistics look sound and the
result holds across 8 encoders. The central claim — that the margin m is a proxy for distribution shift — is an
overclaim: what has been shown is that m correlates with the trial distance, RT, accuracy and the human–model gap.
A more accurate name for m would be **inter-class distance**, and every result would survive the renaming, which
removes the novelty. The manuscript's framing that "similar to the model" and "out of distribution" are the same
property is wrong — a model can distinguish two objects while representing them as similar. "Distribution shift
estimate" and "empirical distribution shift" are never clearly defined in the main text. Fig. 1 axes labelled only
small/large. Asks: why is φ(C) view-averaged while φ(A) and φ(B) are not; consider a simpler shift measure such as
the variance of an object's embedding across viewpoints; show the highest- and lowest-shift trials and whether
they look out-of-distribution.

**r64w** (quality 3 / clarity 2 / significance 3 / originality 3; confidence 3). Solid, well written; the human
validation and the feedforward-vs-recurrent angle are exciting; the paper is clearly measuring something real about
image difficulty. Main worry: the results are consistent with **no distribution shift at all**. If train and test
come from the same distribution, the margin and the trial distance will still correlate, because both measure how
unusual an example is — one within the test set, one across train and test. Conversely, under a genuine shift the
margin can be high for the few test images that are in fact closest to training. Proposes a control: apply both
metrics to two random halves of one dataset; if the relationship survives, the metric measures within-set
weirdness, not shift. Asks what an **object-based measure averaged over all views** would do, and whether it would
still predict training advantage and human accuracy. Follow-up after rebuttal: still uncertain what "distribution
shift" means here; not addressable in a minor revision; recommends resubmitting after a large reframing — either
present the idea far more clearly or drop the distribution-shift framing.

**HVBU** (quality 3 / clarity 3 / significance 3 / originality 2). Well-controlled framework; broadly applicable
proxy; compelling human evidence; well written; the compensation-by-test-time-compute hypothesis is plausible.
Overclaims: that difficulty is not an intrinsic stimulus property (not disproven); that the quantity can be evaluated
in biological systems (not demonstrated). Viewing-time results are not new (Mayo et al. 2023). Many shift metrics
already exist (e.g. OOD-Bench); the paper should compare against them and justify a new one. Scope limited to the
oddity task. Model choice is inconsistent across figure panels. Notes the proxy is only interpretable in relative
terms.

## What was unclear to us at the time

Whether the three reviews were one objection or three. Whether "inter-class distance" (SCwg) and "within-set
weirdness" (r64w) named the same thing. What experiment would satisfy r64w's construction, and whether the
two-random-halves control was fair. The prevailing read was that the work was right and miscommunicated.

## Where recognitions were logged later

Recognitions of what a reviewer may have meant are recorded at the state where they happened, not here:
State 1 (SCwg's inter-class distance and the ½·d(A,B) floor), State 3 (r64w's no-shift construction and the
pretrained control), State 4 (r64w's object-based view-averaged measure). Each is phrased as "this may be what
they meant" — the reviews were not the source of those analyses, and the trace should not imply they were.
