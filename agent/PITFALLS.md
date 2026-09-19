# PITFALLS — traps with fixes
Environment, pipeline and code traps that cost time here. Cluster-wide ones are also in the account memory `parcc-slurm-quirks`.

## Collaborator pipeline (`Dist-shift/HIDA/hida-tune`, owned by knappv, read-only)
- **Hard-coded `/datasets/hida/current/…`** in `train.py` (csv output dir, lines ~155/353), config roots, `evaluation/ood_distance_analysis.py` (`_object_name_to_white_bg_dir`). Fix: run the private copies in `../knockout/scripts/` (`train_knockout.py`, `ood_eval_knockout.py`) with paths redirected; pass Hydra overrides for roots, sim_dirs, `training.log_dir`, `augmentation.save_dir`, `splits.test_dataset.csv_path`.
- **Relative writes**: `augmentation.save_dir: ./augmented_samples` → PermissionError in their dir. `cd` to a writable dir and override.
- **Hydra key names**: the MOCHI csv is `splits.test_dataset.csv_path`, not `dataset.test_dataset…` ("Could not override").
- **Subprocess calls `"python"`** — patched to `sys.executable`; otherwise the wrong interpreter.
- **Subsets under-fill epochs**: the triplet miner draws similarity-binned partners from the full category table and rejects ones missing from the subset. Give each condition its own filtered `object_similarities_precomputed.pkl` (`scripts/make_sims.py`).
- **Object discovery is `listdir`** → a training subset is a directory of symlinks (`scripts/make_subset.py`); ~300k symlinks take minutes on /vast.
- **Best checkpoint** = newest `.pth` by mtime (their convention, `ls -t | head -1`).

## Environments
- `envs/torchfeat` (torch 2.8 cu128) needed `hydra-core wandb loguru matplotlib opencv-python-headless scipy scikit-learn` added — the repo's `requirements.txt` was incomplete. Test the full import chain (`from datasets.builder import build_loader; import training; from evaluation.ood_distance_analysis import get_pretrained_model`) before submitting.
- `~/.conda/envs/dev` has no torch; system `python3` has no pandas. Use dev for analysis, torchfeat for GPU.
- `WANDB_MODE=disabled`, `HF_HUB_OFFLINE=1`, `HF_HOME=…/hf_cache`.

## SLURM
- `b200-mig45` needs `--qos=normal` (default QOS `dgx` maps to `mig`, which is refused).
- Omit `--mem`; the CLI filter derives it from CPUs (28 on dgx-b200, 6 on mig45).
- `sbatch` output is wrapped in a banner; grep `Submitted batch job` explicitly or you will double-submit (happened once; the duplicate was cancelled).
- Job names are how the ledger script finds this project's jobs — keep `<cat>_<cond>`, `viewdepth*`, `enc_extract`, `evaltest`, `geom_*`.

## Code
- **pandas attribute access collides with method names**: `df.sem`, `df.cov`, `df.shift`, `r.cov` on itertuples → use `df['sem']` or rename. Bit three times.
- **`python - <<'PY'` consumes stdin** — you cannot also pipe data into it. Read via `subprocess` inside the script.
- `read_binvox` axis order: raw is x, z, y → transpose (0, 2, 1); validated against real renders (IoU 0.77–0.95 with identity perm; 0.2–0.5 for every other).
- Login node: fine for pandas/matplotlib; a ViT-L forward pass or a 313k×20k distance matrix belongs in a job.
- `fig17` and other early figures had no saved script; every figure now has one under `scratch/` — keep it that way.
