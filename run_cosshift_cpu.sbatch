#!/usr/bin/env bash
#SBATCH --job-name=cosshift
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift/logs/cosshift_cpu_%j.log

set -uo pipefail
export HF_HOME=/vast/projects/bonnen/naturalistic-navig/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64
export TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp

PY=/vast/projects/bonnen/naturalistic-navig/envs/torchfeat/bin/python
D=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/geometric_shift
CKPT="/vast/projects/bonnen/naturalistic-navig/Dist-shift/logs/vit_large_patch14_reg4_dinov2_bs32x1_lr2e-06_ep30_multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_|45|---most-results-ood/checkpoints/checkpoint_epoch_020.pth"

echo "=== ARM 1: NEUTRAL RULER (fixed pretrained DINOv2-L/14 reg4) ==="
$PY $D/cos_shift_extract.py --tag neutral --ckpt "" --n-obj 200 --n-view 6 --n-sg-obj 1000 --bs 32 || exit 1

echo "=== ARM 2: OWN SPACE (fine-tuned |45| epoch 20) ==="
$PY $D/cos_shift_extract.py --tag ownspace --ckpt "$CKPT" --lora-r 16 --lora-alpha 8 --n-obj 200 --n-view 6 --n-sg-obj 1000 --bs 32 || exit 1

echo "=== compute shift(t) for both arms ==="
$PY $D/cos_shift_compute.py --tag neutral
$PY $D/cos_shift_compute.py --tag ownspace

echo "ALL_DONE"
