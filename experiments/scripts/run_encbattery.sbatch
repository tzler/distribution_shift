#!/bin/bash
#SBATCH --job-name=encbattery
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/encbattery_%j.log
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
/vast/home/b/bonnen/.conda/envs/dev/bin/python /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/scratch/encoder_space_battery.py
