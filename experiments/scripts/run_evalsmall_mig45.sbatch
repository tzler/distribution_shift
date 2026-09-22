#!/bin/bash
#SBATCH --job-name=evalsmall
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=b200-mig45
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/eval_%x_%j.out
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout; export EVAL_BACKBONE=vit_small_patch14_reg4_dinov2
echo "host $(hostname) $(date) eval $1 $2 ($EVAL_BACKBONE)"; bash $K/scripts/eval_one.sh $1 $2; bash $K/scripts/eval_bank_all.sh $1 $2; echo "done $(date)"
