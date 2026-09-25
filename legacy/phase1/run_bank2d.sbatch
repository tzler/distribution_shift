#!/usr/bin/env bash
#SBATCH --job-name=geom_bank2d
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/logs/bank2d_%j.log

set -uo pipefail
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
D=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift

echo "=== shapenet render bank (20698 objects x 15 views, MOCHI ids excluded) ==="
$PY $D/build_bank_2d.py --spec shapenet_bank --out $D/bank/bank2d_shapenet.npz --workers 64

echo "=== shapegen render bank (800 objects/stratum x 16 views) ==="
$PY $D/build_bank_2d.py --spec shapegen_bank --out $D/bank/bank2d_shapegen.npz \
    --workers 64 --objs 800 --views 16

echo "ALL_DONE"
