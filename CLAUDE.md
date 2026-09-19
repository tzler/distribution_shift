WRITING A STATE (STATE.md or states/*.md) — the thing this project has drifted on most:
- The reader was not here. Title = what we did or found, in a sentence a colleague from
  another lab would understand. No state numbers anywhere. Unpack every term of art the
  first time it appears in that file. Status is prose from where we were to where we are.
- Each figure is introduced by a sentence saying what to expect and what it shows, and is
  built to the recipe in agent/FEEDBACK.md (one plot type, expectation stated, graded
  verdict, nothing anachronistic).
- Before committing: `python scripts/state_lint.py <file>` must be clean (pre-commit runs
  it; `bash scripts/install_hooks.sh` once per clone). Silence a hit only with
  `<!-- lint: ok term -->` after the term is defined in that file. Then re-read the whole
  file once as the year-later reader.
- When the lead says a state reads as "Claude speak", add the offending phrase to JARGON
  in scripts/state_lint.py in the same commit as the fix.

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
