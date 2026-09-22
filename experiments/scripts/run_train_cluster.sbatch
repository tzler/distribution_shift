#!/bin/bash
#SBATCH --job-name=knockout
#SBATCH --account=bonnen-naturalistic-navig
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=04:30:00
#SBATCH --output=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout/logs/train_%x_%j.log
# usage: sbatch --job-name=<cat>_<cond> run_train.sbatch <cat> <cond>
CAT=$1; COND=$2
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
H=/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune
R=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/shapenet_rendered
PY=/vast/projects/bonnen/naturalistic-navig/envs/torchfeat/bin/python
export HF_HOME=/vast/projects/bonnen/naturalistic-navig/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/vast/projects/bonnen/naturalistic-navig/.tmp
export WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTHONPATH=$H
mkdir -p $K/logs/${CAT}_${COND}/aug && cd $K/logs/${CAT}_${COND}     # writable cwd; config path is absolute
echo "host $(hostname) $(date)  cat=$CAT cond=$COND"; nvidia-smi --query-gpu=name --format=csv,noheader
$PY $K/scripts/train_knockout.py --config-path $H --config-name=config_shapenet \
  dataset_generation.shapenet.categories=[$CAT] \
  dataset_generation.shapenet.roots.white=$K/data/${CAT}_${COND}/white \
  dataset_generation.shapenet.roots.black=$K/data/${CAT}_${COND}/black \
  dataset_generation.shapenet.roots.random=$K/data/${CAT}_${COND}/random \
  dataset_generation.shapenet.sim_dirs.white=$K/data/${CAT}_${COND}/sim/white \
  dataset_generation.shapenet.sim_dirs.black=$K/data/${CAT}_${COND}/sim/black \
  dataset_generation.shapenet.sim_dirs.random=$K/data/${CAT}_${COND}/sim/random \
  training.log_dir=$K/logs/${CAT}_${COND} \
  augmentation.save_dir=$K/logs/${CAT}_${COND}/aug \
  splits.test_dataset.csv_path=/vast/projects/bonnen/naturalistic-navig/MOCHI/mochi_benchmark.csv \
  hydra.run.dir=$K/logs/${CAT}_${COND}/hydra
echo "train done $(date)"; bash $K/scripts/eval_one.sh $CAT $COND; bash $K/scripts/eval_bank.sh $CAT $COND; echo "done $(date)"
