#!/bin/bash
#SBATCH --job-name=evalref
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=b200-mig45
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=0:45:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/eval_%x_%j.out
# usage: sbatch run_eval_mig45.sbatch <cat> <cond>
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
echo "host $(hostname) $(date) eval $1 $2"; bash $K/scripts/eval_ref_bank.sh $1 $2; echo "done $(date)"
