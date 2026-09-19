# FEEDBACK — how the lead wants work done here
Typed like Claude Code `feedback` memories: the correction, why, how to apply. Newest first.

## A state figure: one plot type, the expectation stated, the verdict graded (2026-09-19)
"With these figures I have to spend a really long time to understand what I should expect." Then, of the replacement: "fantastic".
**Why:** a state figure is read on a phone by someone who has not been in the analysis for hours; if the reader has to work out what a flat line would mean, the figure has failed regardless of what it shows.
**Apply — the recipe:**
- One plot type, repeated. Every panel has the same x, the same y, the same two lines. Rows/columns vary one thing each and the panel title says which (`every trial × every model` / `one model per trial: its own category`).
- Say the expectation before the data: in the subtitle, one sentence per line — what should happen if the claim is true (blue falls), and what the control should do (grey flat). The reader then checks, not decodes.
- A verdict box in each panel, in the empty corner, never on the data: the two numbers, then one clause in plain words. Colour the box by grade — green = behaves as expected; orange = a concern, not settled; red is reserved for a conclusive failure and has not yet been needed.
- Grade honestly. "Grey falls with blue" is a concern that makes the lead pause, not a kill; do not write ✗/fails when the honest state is "we have not separated these yet". A larger r for the fine-tuned than the pretrained model is not the kind of evidence we want; say what the pattern is consistent with instead.
- Nothing anachronistic: a state figure may only use measures that existed at that state (geometric coverage does not belong in State 2).
- Footer: two lines that say what each row showed, in the same words as the verdicts. Then check the rendered PNG for collisions and clipping before sending.

## Never retrofit external feedback onto later findings (2026-09-19)
Reviews were confusing at the time and the honest belief was "right but miscommunicated". Log what was said and how it read then; log recognitions where they happen, as recognitions.
**Why:** retrofitting overstates both the reviewers' clarity and our foresight; the trace is about how understanding actually changed.
**Apply:** State 0 style. A new entry supersedes a wrongly-framed one; never edit.

## Be resource-rational, and say the cost in money (2026-09-19)
**Why:** the lead wants results this weekend but not by burning the most expensive GPUs; internal billing units meant nothing to him until converted at the published rates.
**Apply:** every resource decision gets GPU-hours and dollars (unsub / sub) in its Dxx; stage large designs (submit the informative third first, hold the rest); measure a cheaper partition before committing; tight time limits so jobs backfill.

## Show the figure, not the statistic (long-standing; reaffirmed 2026-09-19)
"I can't tell looking at what you've shown me here" — after a sweep plot instead of the pictures.
**Why:** several results looked convincing as numbers and fell apart when plotted, and vice versa.
**Apply:** put new metrics through the figure templates the lead already liked (fig 7 / 9 / 25 / 40 / 41 / 54 styles); draw the control on its own axis when it compresses the effect; send the PNG.

## Explain, don't assert (2026-09-19)
"You're still skipping the part that is unclear to me — by the triangle inequality."
**Why:** an argument the lead can't reconstruct is not evidence to him.
**Apply:** when a claim rests on a step (an inequality, a decomposition, a control), spell the step out with a picture and test the obvious escape routes (cosine? L2?) with data before moving on.

## Do the experiment, not another metric, when the design is the limit (2026-09-19)
The lead came round to this after several rounds; it was said before it was heard.
**Why:** within a category with one training set, distance-to-training and atypicality are one variable; no estimator separates them.
**Apply:** when two exhaustive searches agree there is nothing, stop searching and change what varies.

## Interpretability over effect size for the headline (2026-09-19)
The single-trial demo — the trial, the training data the metric chose, the margin before and after — is "the banger". 
**Apply:** design experiments so one trial's story can be shown end to end; keep the random-subset runs, they make the metric comparison free.

## Optimistic and pessimistic reads are both required (2026-09-18)
After a pessimistic run of results the lead asked for "the most favorable version… acknowledging we want to be rigorous". Then rejected a packet that led with the wrong figures.
**Why:** the lead's sense of the project depends on which figures are in front of him; my choice of figures is a judgement he wants to make.
**Apply:** when asked for a packet, use the figures the lead named, in the order he named them; when unsure which, ask by listing them.

## Full absolute paths for anything generated (long-standing)
**Apply:** every file mentioned gets its absolute path; send files rather than describing them when the lead is on a phone.

## Documentation is part of the work (2026-09-19)
**Apply:** at the end of any session with work, ask whether a trigger fired (decision / result incl. null / reflection / feedback / literature / resource); log the Dxx and STATE diff together; add a states/ snapshot when the story turns; update INFLIGHT.md and write the session summary.
