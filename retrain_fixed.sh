#!/bin/bash
set -u
OPT=/data1/kurumi/opt
cd "$OPT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export FIX_GRAPH=1

VENV=/data1/kurumi/.conda/envs/ai_exp/bin/python
RESULTS_DIR=$OPT/saved_models_fixed
LOGDIR=$OPT/logs_retrain_fixed
DATA=./dataset/docred
BERT=./bert-base-cased
SEEDS=(42 123 456 789 2024)
CONFIGS="base full"
GPUS=(1 2)
LOCKFILE=$OPT/.retrain_fixed.lock
mkdir -p "$LOGDIR" "$RESULTS_DIR"

BASE_EXTRA="--num_class 97 --num_labels 4 --adam_epsilon 1e-6 --max_grad_norm 1.0"

acquire_lock() {
  if [ -f "$LOCKFILE" ]; then
    oldpid=$(cat "$LOCKFILE" 2>/dev/null || true)
    if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
      echo "ERROR: another run is active (pid $oldpid). Aborting."; exit 1
    fi
  fi
  echo $$ > "$LOCKFILE"
  trap 'rm -f "$LOCKFILE"' EXIT
}

cfg_args() {
  case "$1" in
    base) echo "--disable_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2" ;;
    full) echo "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3" ;;
    *)    echo "" ;;
  esac
}

run_bert() {
  local gpu="$1" name="$2" seed="$3" extra="$4"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu (FIX_GRAPH=1)"
  {
    echo "===== run start ====="; date '+%F %T'; echo "name=$name seed=$seed gpu=$gpu FIX_GRAPH=1"
  } > "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$DATA" --model_name_or_path "$BERT" --transformer_type bert \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size 2 --test_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 30 --lr_transformer 5e-5 --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $extra \
    >> "$LOGDIR/${name}.log" 2>&1
}

acquire_lock
echo "=== FIXED retrain: $CONFIGS x seeds ${SEEDS[*]} (GPUs ${GPUS[*]}, FIX_GRAPH=1) ==="
for seed in "${SEEDS[@]}"; do
  idx=0
  for cfg in $CONFIGS; do
    gpu=${GPUS[$(( idx % 2 ))]}
    run_bert "$gpu" "bert_${cfg}_s${seed}" "$seed" "$(cfg_args "$cfg")" &
    idx=$(( idx + 1 ))
  done
  wait
  echo "[$(date '+%F %T')] seed $seed round done"
done
echo "=== FIXED RETRAIN DONE ==="
