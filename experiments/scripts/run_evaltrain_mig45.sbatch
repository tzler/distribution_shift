#!/bin/bash
#SBATCH --job-name=evaltrain
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=b200-mig45
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/eval_%x_%j.out
# usage: sbatch run_evaltrain_mig45.sbatch "<cond1> <cond2> ..."   (all chair-cat conds given as cat_cond)
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
for c in $1; do cat=${c%%_*}; cond=${c#*_}; export EVAL_BACKBONE=vit_large_patch14_reg4_dinov2; case $cond in *scratch_s|*smallft) export EVAL_BACKBONE=vit_small_patch14_reg4_dinov2;; esac
  echo "== $(date) eval $cat $cond ($EVAL_BACKBONE)"; bash $K/scripts/eval_train.sh $cat $cond; done; echo "done $(date)"
