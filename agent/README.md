# agent/ — model-facing files

Written for whichever Claude session works on this project next (including this one after a
context summary). Humans are welcome but these are not written for them: no narrative, no
hedging, one fact per entry, each with why it matters and how to act on it. Modelled on Claude
Code's own memory layout (typed entries, a recall line, why + how-to-apply, verify-before-use).

| file | what | when to read |
|---|---|---|
| `FEEDBACK.md` | standing corrections from the lead — how work should be done here | every session, after STATE and REASONING |
| `CHECKS.md` | analysis sanity checks that caught real errors in this project | before claiming any result |
| `PITFALLS.md` | environment and pipeline traps, each with the fix | before running anything |
| `INFLIGHT.md` | live jobs: IDs, logs, expected outputs, how to resume | on resume, and before closing out |
| `RECALL.md` | the numbers that get quoted, each with its source | when writing or checking a claim |
| `sessions/` | one machine summary per agent session: done / failed / logged / open | on resume |

Rules: update rather than duplicate; delete what turns out wrong (these files are mutable —
the append-only record is REASONING.md); verify a named path or job still exists before
acting on it; keep entries short enough that the whole directory is a five-minute read.
