#!/bin/bash
#SBATCH --job-name=evaltest
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=01:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/evaltest_%j.log
bash /vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/scripts/eval_one.sh chair g1_k10
