# legacy/ — code kept for provenance, not for use

| | |
|---|---|
| `phase1/` | the first phase (8–18 Sep): descriptor builders, bank builders, one-off analyses and their SLURM scripts. Several figures cited in `evidence/` were made here; the shared modules they import now live in `../lib/`, so run them with `PYTHONPATH=../lib` |
| `superseded/` | three earlier passes of the distance-measure search, replaced by `scratch/distance_search_fast.py` (same scores, minutes instead of hours) |

Nothing here is needed to reproduce the current results — see [`../REPRODUCE.md`](../REPRODUCE.md).
