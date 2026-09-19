#!/usr/bin/env bash
#SBATCH --job-name=pose_feats
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/logs/pose_%j.log
set -uo pipefail
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
export OMP_NUM_THREADS=1
/vast/home/b/bonnen/.conda/envs/dev/bin/python \
  /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/pose_features.py --workers 64
echo ALL_DONE
