#!/bin/bash
#SBATCH --job-name=viewdepth4
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/out/viewdepth4_%j.log
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
/vast/home/b/bonnen/.conda/envs/dev/bin/python scratch/viewdepth_pipeline.py --stage 4
