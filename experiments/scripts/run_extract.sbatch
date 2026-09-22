#!/bin/bash
#SBATCH --job-name=enc_extract
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=06:00:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/extract_%j.log
export HF_HOME=/vast/projects/bonnen/naturalistic-navig/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
cd /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
echo "host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/vast/projects/bonnen/naturalistic-navig/envs/torchfeat/bin/python scripts/extract_encoder_features.py
echo "done $(date)"
