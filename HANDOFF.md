# HANDOFF v2 — lab reasoning-trace system (for a Claude agent)

You are setting up and then maintaining this lab's project-documentation system in this
repo. Work in three phases: **bootstrap → back-fill → ongoing operation.** All edits via
commits on a branch the lead reviews (or PRs); never directly to main.

The system in one line: **STATE.md is what we currently believe; REASONING.md is why;
states/ is how STATE looked at earlier turns; evidence/ holds only what is cited; git is
the history.**

v2 changes from v1 are marked ▲ and explained at the end.

---

## Phase 1 — Bootstrap (skip any file that already exists)

```
STATE.md  REASONING.md  CLAUDE.md  HANDOFF.md  .gitignore
states/   meetings/   evidence/   background/
```

**If the project has no git repo, initialise one** on a branch (e.g. `docs-system`) and
leave `main` for the lead to create by merging. `.gitignore` bulk data and regenerable
outputs (banks, checkpoints, figure dumps); the cited figures live in `evidence/`. ▲

**STATE.md** — rewritten in place; claims true *now*; ≤5-minute read; cite reasoning
entries inline as `[Dxx](REASONING.md#Dxx)` (the entries carry anchors). ▲ Sections, in
this order:

```markdown
# <project name — the question, not the method>          ▲ name it by what is being asked
*Last meaningful update: YYYY-MM-DD · Lead: <name> · Status: active* · [states/] · [REASONING.md]

## Goal      ← one paragraph, HIGH level: what are we trying to find out and why it matters.
              Not the method, not the current sub-question.                              ▲
## Status    ← a short walkthrough from where we started to where we are, in 4–6 beats
              (where we started · what we did · where it works · where it fails · why ·
              what we are doing now), each beat with the figure that carried it and its
              [Dxx]. This is what a reader opens first.                                   ▲
## Strategy  ← the current approach in enough detail to be judged: each track, its
              prediction, what it costs, in what order and why.                          ▲
## Next steps      ← concrete, assigned checkboxes
## Open questions  ← unknowns shaping the plan (drop items that are merely old)          ▲
## Pointers        ← code / data / experiments / draft / upstream / background/
```

Hypotheses may appear as a short list if they help; they are not required. ▲ Images are
relative links into `evidence/`; verify every link resolves (script it) before committing.

**REASONING.md** — append-only; never edit or delete an entry, supersede with a new one;
entries true *at their date*; stable sequential IDs; **newest entry at the top** ▲ so the
latest is always visible; each heading preceded by `<a id="Dxx"></a>` so STATE can link
to it. ▲ One line per field, six fields: ▲

```markdown
<a id="D07"></a>
## D07 — YYYY-MM-DD — <short title>
**State:** the general situation at the time — what we believed, one sentence.
**Observation:** the specific thing being engaged with (a result, feedback, a reading).
**Decision:** what we concluded or chose; our interpretation.
**Because:** the evidence or argument (link evidence/).
**Rejected:** alternatives considered and why not.
**Implication:** what this changes about the plan / next steps.
```

Null and failed results are first-class entries. Decisions that spend resources (GPU
hours, money) record the cost and the budget reasoning. ▲

**states/** ▲ — retroactive snapshots of STATE.md at each turn of the project, for
walking a lab through the history: `NN-<slug>.md`, same sections as STATE, written from
that moment's vantage point with the evidence available then, prev/next links, and an
index `states/README.md`. Add a snapshot when the story turns; never edit old ones. The
first snapshot is the starting point — for a resubmission, the submitted manuscript, its
claims, and the reviews.

**RESOURCES.md** ▲ — what is available, stated once: data (sizes, working subsets, known
quirks), models (inherited and new, with their cost), compute (partitions, rates, account
cap, unit cost of the project's typical job), environments, people. Linked from STATE and
every snapshot. And every state — snapshots and the live STATE — ends with a **"Resources
used (cumulative)"** section: agent time, compute (GPU-h, CPU-core-h), dollars at the
published rates, what is committed and what is held. Build the numbers from the
scheduler's accounting (a script in the repo), not from memory; place them in time with
file timestamps.

**agent/** ▲ — model-facing files, modelled on Claude Code's own memory layout (typed
entries with a recall line, why + how-to-apply, verify-before-use, mutable with an index):
`FEEDBACK.md` (standing corrections from the lead: what, why, how to apply), `CHECKS.md`
(sanity checks that caught real errors), `PITFALLS.md` (environment and pipeline traps with
fixes), `INFLIGHT.md` (live jobs, logs, expected outputs, resume commands — refreshed every
session), `RECALL.md` (every number that gets quoted, with its source), `sessions/` (one
machine summary per session: done / failed / logged / open). These are what let the next
agent — or the same one after a context summary — behave correctly on day one; humans
rarely need them. Mutable; the append-only record stays REASONING.md.

**meetings/** — one file per meeting, `YYYY-MM-DD.md`, typed sections in order:
`## Summary (machine)` / `## Decisions & results` / `## Notes (human)`.
**Raw transcripts and verbatim quotes of lab members NEVER enter this repo.**

**evidence/** — only results *cited* from a REASONING entry or a STATE/states claim
(admission rule). `evidence/README.md` is a provenance table: file · script that regenerates
it · data it reads · SLURM job IDs where relevant · admission commit · cited by. ▲

**background/** — papers and ideas drawn on; other lab projects' STATE.md files welcome;
a **paraphrased** summary of external reviews (see the feedback rule below). First draft of
the citation list.

**.gitignore** — `notes/`, `*.local.md` (personal scratch, by design), plus bulk data.

**CLAUDE.md** — the standing rules (this is what makes future sessions maintenance-free):

```markdown
Read STATE.md first (current truth), then REASONING.md (why). Weight
meetings/ "Notes (human)" over "Summary (machine)".
- REASONING.md is append-only; NEWEST ENTRY AT THE TOP; next sequential ID; one line
  per field: State · Observation · Decision · Because · Rejected · Implication.
- STATE.md rewritten in place; ≤5-min read; Goal / Status / Strategy / Next / Open /
  Pointers; cite [Dxx](REASONING.md#Dxx).
- Invariant: every substantive STATE change is justified by a REASONING entry in
  the same commit/PR. Presentation-only edits need no entry.
- Log triggers: decision · results (incl. null/failed) · reflection · external
  feedback · literature event · resource decisions.
- The log entry is the mandatory act; STATE can lag.
- External feedback: log what was said and how it read to us at the time; log any
  later recognition at the entry where it happened. Never retrofit.
- states/ snapshots are added when the story turns and never edited.
- No transcripts or verbatim quotes of lab members, ever.
- Changes via PR or a reviewed branch; never to main directly.
```

Add project-specific rules under it (read-only upstream directories, where experiments
live, how figures are regenerated).

## Phase 2 — Back-fill (one-time, with the project lead)

Retroactive population has two sources; use both:

**(a) Ingest what exists.** Ask the lead to point you at existing materials — figures,
notes, slides, analysis code, prior write-ups, the manuscript and its reviews, chat
exports, an earlier session's record. Draft STATE.md and the first states/ snapshot from
them. For each result figure, propose an evidence/ admission with provenance.

**(b) Interview for the trace.** One question at a time, short questions:
1. Draft STATE first; show for correction before proceeding. Expect corrections on the
   *level* of the goal (too methodological) and on what Status should walk through. ▲
2. Walk backward from today: "What was the most recent decision or result that changed
   the project's direction?" For each: state, observed, decided, because, rejected.
   Draft the entry; confirm; repeat.
3. Ask at least once: **"What did you try that DIDN'T work — including quiet
   redirections that never felt like decisions?"** Null/failed attempts are first-class
   entries, the highest-value trace content.
4. Ask about the period *before* your own record. Entries reconstructed from a session
   record rather than the lead's memory are dated approximately and marked "(recalled)". ▲
5. Stop at ~5–12 entries, numbered oldest-first even though displayed newest-first.

**External feedback (reviews, a collaborator's critique)** ▲: record what was said,
paraphrased; record how it read to the team *at the time* — including confusion, and a
belief that the work was right but miscommunicated, if that was the belief; do not map
concerns onto later findings. When a later result makes a reviewer's remark legible,
log the recognition *there* ("this may be what they meant"), phrased as a recognition,
not as the source of the analysis. The trace must not overstate either the reviewers'
clarity or the team's foresight.

End phase 2 by listing remaining TODOs (unverified claims, missing evidence files, empty
background/, the lead's pre-history) at the bottom of STATE.md.

## Phase 3 — Ongoing operation

On any session in this repo where work happened, before closing out ask yourself: **did a
trigger fire?** (decision / result incl. null / reflection / external feedback / literature
event / resource decision). If yes, propose the REASONING entry and any STATE diff
together in one commit. If the story turned, add a states/ snapshot. If the lead pastes
meeting notes, produce `meetings/YYYY-MM-DD.md` in the typed format (leave "Notes
(human)" empty) plus proposed entries — never invent decisions from mere discussion;
genuinely-open threads go to STATE "Open questions".

**Working with a lead on a phone** ▲: they may not be able to open the repo. Send the
changed files as attachments after each round of edits, in reading order; say which
images won't render inline; keep each file self-contained enough to read alone.

**When the lead corrects the trace** ▲: apply the correction, and if it changes what an
existing entry asserts, add a new entry that supersedes it rather than editing — the
correction is itself part of the history.

Style discipline: terse everywhere; machine-drafted entries obey the same caps as
hand-written ones; no hedging prose; no meeting narrative inside STATE.md.

---

## What changed from v1, and why ▲

From the first live use (geometric_shift, 19 Sep 2026):
- **Newest-first REASONING with State and Decision fields.** The lead wanted the latest
  entry visible without scrolling, and each entry readable on its own: a sentence on the
  general state, then the specific observation, then the decision, then why.
- **STATE = Goal / Status / Strategy / Next / Open.** The first draft named the project
  by its method and led with a hypothesis list; the lead wanted the goal at the highest
  level and Status as a short walkthrough with figures. Strategy needs enough detail to
  be judged.
- **states/ snapshots.** The lead presents the *process* to the lab by clicking forward
  through the project's earlier states. Snapshots make that possible without touching
  the live STATE.
- **The external-feedback rule.** The first draft mapped every reviewer concern onto a
  later finding, which read as if the work had been a response to the reviews. It had
  not; at the time the reviews were confusing and the team believed it was right. The
  trace must preserve that.
- **Cost in the log, and a resources inventory.** Resource decisions are decisions; the
  entry carries GPU-hours and dollars. RESOURCES.md states what is available once; every
  state carries what had been used by then, so the lab can see what each turn cost.
- **Repo bootstrap, link verification, phone workflow, evidence provenance table** —
  practicalities that cost time when missing.
- **agent/ — the model-facing layer.** The human-facing files had no place for "how the
  lead wants work done", the checks that caught errors, the traps, or the live job state;
  those are exactly what Claude Code's memory system keeps per user, and a project needs
  them per repo so any agent — not only one account — inherits them.
