#!/bin/bash
#SBATCH --job-name=viewdepth
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/viewdepth_%j.log
set -u
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
echo "host $(hostname) start $(date)"
for s in 1 2 3 4; do
  echo "=== stage $s  $(date) ==="
  /usr/bin/time -f "  peak %M KB, wall %E" $PY scratch/viewdepth_pipeline.py --stage $s --workers 60 || { echo "FAILED stage $s"; exit 1; }
done
echo "done $(date)"
