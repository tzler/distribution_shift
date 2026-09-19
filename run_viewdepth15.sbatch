#!/bin/bash
#SBATCH --job-name=viewdepth15
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/viewdepth15_%j.log
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
$PY scratch/viewdepth_pipeline.py --stage 3b --workers 16
for v in full pool2; do echo "############ FIXED-15  VARIANT $v"; $PY scratch/viewdepth_pipeline.py --stage 4 --variant $v --fixed15 2>&1 | grep -v "^\s*$"; done
