#!/bin/bash
#SBATCH --job-name=viewdepth1v
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/viewdepth1v_%j.log
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
echo "############ SINGLE FIXED VIEW: nearest"; $PY scratch/viewdepth_pipeline.py --stage 4 --variant full --fixed15 --single nearest 2>&1 | grep -v "^\s*$"
echo "############ ACTUAL VIEW, by pose quality"; $PY scratch/viewdepth_pipeline.py --stage 4 --variant full 2>&1 | grep -v "^\s*$"
