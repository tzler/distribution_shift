#!/bin/bash
# usage: eval_one.sh <cat> <cond>   — evaluates the newest checkpoint of that run on all MOCHI trials
CAT=$1; COND=$2
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
H=/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune
PY=/vast/projects/bonnen/naturalistic-navig/envs/torchfeat/bin/python
export HF_HOME=/vast/projects/bonnen/naturalistic-navig/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$H:$H/evaluation
EXP=$(ls -d $K/logs/${CAT}_${COND}/vit_*/ 2>/dev/null | head -1); EXP=$(basename "$EXP")
CKPT=$(ls -t $K/logs/${CAT}_${COND}/$EXP/checkpoints/*.pth | head -1)
CSV=$K/csv/$EXP/shapenet_dataset.csv
OUT=$K/eval/${CAT}_${COND}; mkdir -p $OUT
echo "eval ${CAT}_${COND}: ckpt $(basename $CKPT)  csv $CSV"
cd $H && $PY $K/scripts/ood_eval_knockout.py \
  --hida-csv "$CSV" \
  --mochi-csv /vast/projects/bonnen/naturalistic-navig/MOCHI/mochi_benchmark.csv \
  --mochi-root /vast/projects/bonnen/naturalistic-navig/MOCHI \
  --backbone ${EVAL_BACKBONE:-vit_large_patch14_reg4_dinov2} \
  --checkpoints pretrained "fine_tuned:${CKPT}" \
  --lora-r 16 --lora-alpha 8 --lora-dropout 0.1 --bg-match white \
  --output-dir "$OUT" --device cuda:0
echo "eval done: $(ls $OUT/ood_analysis_results.csv 2>/dev/null)"
