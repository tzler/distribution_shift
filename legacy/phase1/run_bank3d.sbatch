#!/usr/bin/env bash
#SBATCH --job-name=geom_bank3d
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/logs/bank3d_%j.log

set -uo pipefail
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/vast/home/b/bonnen/.conda/envs/dev/bin/python
D=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
B=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/blend-data

echo "=== MOCHI test objects (shapenet_mochi_excluded) ==="
$PY $D/build_bank_3d.py --root $B/shapenet_mochi_excluded \
    --out $D/bank/test3d_shapenet.npz --workers 64

echo "=== training bank (shapenet_training) ==="
$PY $D/build_bank_3d.py --root $B/shapenet_training \
    --out $D/bank/bank3d_shapenet.npz --workers 64

echo "ALL_DONE"
