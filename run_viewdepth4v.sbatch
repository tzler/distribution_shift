#!/bin/bash
#SBATCH --job-name=viewdepth4v
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/viewdepth4v_%j.log
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
for v in sil depth pool2 pool4 pool2sil pool4sil; do echo "############ VARIANT $v"; /vast/home/b/bonnen/.conda/envs/dev/bin/python scratch/viewdepth_pipeline.py --stage 4 --variant $v 2>&1 | grep -v "^\s*$"; done
