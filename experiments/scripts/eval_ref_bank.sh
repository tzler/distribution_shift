#!/bin/bash
# usage: eval_ref_bank.sh <cat> <runid>  — evaluates one of the collaborator's 12 category models on the bank-built chair trials
CAT=$1; RID=$2
K=/vast/projects/bonnen/naturalistic-navig/Dist-shift-data/knockout
H=/vast/projects/bonnen/naturalistic-navig/Dist-shift/HIDA/hida-tune
L=/vast/projects/bonnen/naturalistic-navig/Dist-shift/logs
PY=/vast/projects/bonnen/naturalistic-navig/envs/torchfeat/bin/python
export HF_HOME=/vast/projects/bonnen/naturalistic-navig/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$H:$H/evaluation
D=$(ls -d "$L"/vit_large_patch14_reg4_dinov2_bs32x1_lr1e-06_ep30_multi_similarity_seed42_train:_val:_lora_r16_alpha8_dropout0.1_\|${RID}\|* | head -1)
CKPT=$(ls -t "$D"/checkpoints/*.pth | head -1)
CSV=$(ls -d $K/csv/*/ | head -1)shapenet_dataset.csv     # any training csv; only used for the NN-distance columns
OUT=$K/eval_bank/ref_${CAT}; mkdir -p $OUT
echo "eval ref ${CAT} (|${RID}|): ckpt $(basename $CKPT)"
cd $H && $PY $K/scripts/ood_eval_knockout.py --hida-csv "$CSV" --mochi-csv $K/banktrials/banktrials_chair.csv --mochi-root $K/banktrials \
  --backbone vit_large_patch14_reg4_dinov2 --checkpoints pretrained "fine_tuned:${CKPT}" --lora-r 16 --lora-alpha 8 --lora-dropout 0.1 --bg-match white --output-dir "$OUT" --device cuda:0
echo "eval done: $(ls $OUT/ood_analysis_results.csv 2>/dev/null)"
