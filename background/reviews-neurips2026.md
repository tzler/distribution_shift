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

## What they were pointing at, in our terms

| reviewer's concern | what we found |
|---|---|
| SCwg: m is inter-class distance, not shift | The metric is bounded by ½·d(A,B); the floor predicts the pretrained margin at +0.57. The "shift" was the inter-object distance [D06]. |
| r64w: both metrics measure within-set weirdness; no shift needed | Pooled, the pretrained model — which saw none of the training sets — tracks the geometric shift as strongly as fine-tuned models: a stimulus property [D02]. |
| r64w: apply the metrics to two random halves of one set | Our base-DINOv2 control is the same test in a stronger form; the within-trial design makes it exact [D02]; coverage is the first estimate to pass it pooled [D05]. |
| r64w: what would an object-based, view-averaged measure do? | Built it. Object-level coverage (view-free) predicts the margin as well as image-level coverage at any training view, and better than at the actual view — support is view-invariant within the training grid [D08]. |
| SCwg: why view-average φ(C) but not φ(A), φ(B)? | Same asymmetry we removed by making the estimator oddity-blind (per image, never differenced) [D02]. |
| HVBU: compare against existing shift metrics | kNN (Sun et al. 2022), Mahalanobis (Lee et al. 2018), PCA-reconstruction, energy, density-ratio and percentile estimators in the 1,895-candidate search and the coverage variants [D03, D05]. |
| SCwg / r64w: define "distribution shift" | The within-trial design defines it operationally: the change in margin when the training set changes and nothing else does [D02]. What it cannot yet define is a graded within-category quantity [D07, D09]. |
| r64w: reframe, or make "distribution shift" much clearer | Open decision for TB: claim category coverage now, or wait for the within-category interventions. |
