"""Refresh agent/INFLIGHT.md from the live queue and the eval directory."""
import subprocess, os, datetime, glob
import os as _os
_here=_os.path.dirname(_os.path.abspath(__file__)); _root=_os.path.dirname(_here)
_RESOLVED_G=_root if _os.path.exists(_os.path.join(_root,'STATE.md')) else '/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift'
_os.makedirs(_os.path.join(_RESOLVED_G,'out','figures'),exist_ok=True)

K='/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout'; G=_RESOLVED_G
q=[l.split('|') for l in subprocess.run(['squeue','-u','bonnen','-h','-o','%i|%j|%T|%M|%P|%R'],capture_output=True,text=True).stdout.strip().splitlines() if l]
now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'); rows=[]
for jid,name,state,t,part,reason in q:
    kind='extract' if name=='enc_extract' else ('eval-test' if name=='evaltest' else 'fine-tune')
    log={'fine-tune':f'logs/train_{name}_{jid}.log','extract':f'logs/extract_{jid}.log','eval-test':f'logs/evaltest_{jid}.log'}[kind]
    out={'fine-tune':f'eval/{name}/ood_analysis_results.csv','extract':'eval/encoder_features/*.npz','eval-test':''}[kind]
    ep=''
    if kind=='fine-tune' and os.path.exists(f'{K}/{log}'):
        ep=str(sum(1 for l in open(f'{K}/{log}',errors='ignore') if 'Recalculated train_ratio' in l))+' ep'
    rows.append(f'| {jid} | {name} | {kind} | {part} | {state} {reason if state=="PENDING" else ""} | {t} {ep} | `{log}` | `{out}` |')
done=[os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob(f'{K}/eval/*/ood_analysis_results.csv'))]
b1=[l.split()[1] for l in open(f'{K}/logs/batch1_jobs.txt')] if os.path.exists(f'{K}/logs/batch1_jobs.txt') else []
s=f'''# INFLIGHT — live jobs and how to resume
*Machine-maintained by `scratch/inflight.py`; last refreshed {now}. Paths relative to `../knockout/`.*

## Queue right now ({len(q)} jobs)
| job | name | kind | partition | state | elapsed | log | expected output |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Completed evaluations: {", ".join(done) if done else "none"}. (`chair_g1_k10`'s csv from the eval test is an epoch-10 checkpoint; the job's own chained eval overwrites it.)

## Batches
- **Batch 1** (submitted 19 Sep ~13:40, dgx-b200, 4.5 h limit): pilot `chair_g1_k10` + {len(b1)} jobs in `logs/batch1_jobs.txt` — knockout k = 10 for every group, `random_k10` per category, `chair_random_100_1..7`. Each trains 30 epochs (~3 min/epoch) then runs `scripts/eval_one.sh` → `eval/<cat>_<cond>/ood_analysis_results.csv`.
- **mig45 timing run** `chair_random_100_0` (`b200-mig45`, `--qos=normal`, 12 h limit): compare epoch time with ~3 min; ≤ 2.5× slower ⇒ mig45 is cheaper per run ⇒ batch 2 goes there.
- **Batch 2** (not submitted): `chair_target_100_<trial>` × 6, `airplane_cross_100_<trial>` × 6; subsets and sims built. `sbatch --job-name=<cat>_<cond> scripts/run_train[_mig45].sbatch <cat> <cond>`.
- **Batch 3** (held): `<cat>_g<1-4>_k50`, `<cat>_random_k50` (15); decide from batch-1 results.
- **Track A** `enc_extract` → `eval/encoder_features/{{pretrained,ft_chair,ft_airplane,ft_table}}.npz`; then write and run `scratch/encoder_space_battery.py` (within-category, category-centred, in each model's space).

## How to check / resume
```
squeue -u bonnen -o "%i %j %T %M %P %R"
python {G}/scratch/inflight.py                    # refresh this file
ls {K}/eval/*/ood_analysis_results.csv
bash {K}/scripts/eval_one.sh <cat> <cond>          # re-run only the evaluation (GPU job)
python {G}/scratch/compute_ledger.py               # refresh the cost ledger
```
A job that died after training leaves checkpoints in `logs/<cat>_<cond>/*/checkpoints/` — re-run only the eval. A job that died in training: just resubmit (the csv dir is keyed by a random exp tag; nothing to clean).

## Analysis to write while jobs run
`scratch/analyze_knockout.py` — per category, per trial: x = knn_mean / coverage of the trial to each model's actual training set (recompute from `design.json` removals against `bank_voxel16`); y = fine-tuned margin from each `eval/` csv; within-trial regression of Δmargin on Δx; own-group vs other-group vs size-matched-random contrast.
`scratch/analyze_knockin.py` — Δmargin from pretrained vs x over the 8 random subsets, per trial; candidate-metric comparison on those runs; targeted vs random vs cross for the 6 target trials; the single-trial figure.
'''
open(f'{G}/agent/INFLIGHT.md','w').write(s); print(f'INFLIGHT refreshed: {len(q)} jobs, {len(done)} evals done')
