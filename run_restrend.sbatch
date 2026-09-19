#!/usr/bin/env bash
#SBATCH --job-name=res_trend
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/logs/restrend_%j.log
set -uo pipefail
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
/vast/home/b/bonnen/.conda/envs/dev/bin/python \
  /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/res_trend.py
echo ALL_DONE
