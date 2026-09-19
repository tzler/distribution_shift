Read STATE.md first (current truth), then REASONING.md (why), then agent/ (FEEDBACK,
CHECKS, PITFALLS, INFLIGHT, RECALL — model-facing; five minutes). Weight meetings/
"Notes (human)" over "Summary (machine)".
- REASONING.md is append-only; NEWEST ENTRY AT THE TOP; next sequential ID; one line per
  field: State · Observation · Decision · Because · Rejected · Implication · Steering ·
  Confidence (one line per contributor, named by role and identity).
- Human oversight, default ON: a STATE rewrite or a new states/ snapshot is PROPOSED by the
  agent and CONFIRMED by the lead before it is committed. An entry's lead-confidence line is
  the lead's to fill; leave "— (to fill)" rather than guess it. A project may switch this
  off explicitly here.
- STATE.md rewritten in place; ≤5-min read; cite [Dxx].
- Invariant: every substantive STATE change is justified by a REASONING
  entry in the same commit/PR.
- Log triggers: decision · results (incl. null/failed) · reflection ·
  external feedback · literature event.
- The log entry is the mandatory act; STATE can lag.
- No transcripts or verbatim quotes of lab members, ever.
- Changes via PR (or commits on a branch the lead reviews); never to main directly.

Project-specific:
- states/ holds retroactive STATE snapshots for the lab walkthrough (prev/next links, index in
  states/README.md); add a new snapshot only when the story turns, never edit old ones.
- Cite reasoning entries as [Dxx](REASONING.md#Dxx); each entry heading carries an <a id> anchor.
- The intervention experiments live in ../knockout/ (same conventions apply; cite from here).
- The collaborator's ../../Dist-shift/HIDA/hida-tune is read-only; use the patched copy
  in ../knockout/scripts/.
- background/narrative-checkpoints/ is the long-form back-fill source, not maintained.
- Figures: regenerate from the script named in evidence/README.md; do not hand-edit PNGs.
