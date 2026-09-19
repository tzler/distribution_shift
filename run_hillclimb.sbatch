#!/bin/bash
#SBATCH --job-name=hillclimb
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/hillclimb_%j.log

set -u
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  start $(date)"
for r in d57 bbox structure volatility multiview voxel8 voxel16 voxel32; do
  echo "=== $r ==="
  /usr/bin/time -f "  peak %M KB, wall %E" $PY scratch/hillclimb.py --rep $r || echo "  FAILED $r"
done

echo "=== merging shards ==="
$PY scratch/merge_shards.py

echo "=== scoring: margin advantage ==="
$PY scratch/hillclimb_eval.py --y margin_adv
echo "=== scoring: human RT (secondary) ==="
$PY scratch/hillclimb_eval.py --y rt
echo "done $(date)"
