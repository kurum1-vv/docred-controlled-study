#!/bin/bash
set -u
OPT=/data1/kurumi/opt
cd "$OPT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export FIX_GRAPH=1

VENV=/data1/kurumi/.conda/envs/ai_exp/bin/python
RESULTS_DIR=$OPT/saved_models_fixed
LOGDIR=$OPT/logs_encoder_fixed
DATA=./dataset/docred
SEEDS=(42 123 456 789 2024)
GPUS=(1 2)
LOCKFILE=$OPT/.retrain_encoder.lock
mkdir -p "$LOGDIR" "$RESULTS_DIR"

# 解析模型路径
RB=""
for cand in /data1/hcw/models/roberta-base ./roberta-base; do
  if [ -f "$cand/config.json" ]; then RB="$cand"; break; fi
done
RL="./roberta-large"
if [ -z "$RB" ]; then echo "ERROR: roberta-base 未找到(已查 /data1/hcw/models/roberta-base 与 ./roberta-base)"; exit 1; fi
if [ ! -f "$RL/config.json" ]; then echo "ERROR: ./roberta-large 缺失"; exit 1; fi
echo "roberta-base  = $RB"
echo "roberta-large = $RL"

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

run_rob() {
  local gpu="$1" name="$2" seed="$3" mpath="$4" lr="$5" bs="$6" acc="$7" extra="$8"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu (FIX_GRAPH=1)"
  {
    echo "===== run start ====="; date '+%F %T'; echo "name=$name seed=$seed gpu=$gpu FIX_GRAPH=1"
  } > "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$DATA" --model_name_or_path "$mpath" --transformer_type roberta \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size "$bs" --test_batch_size 8 --gradient_accumulation_steps "$acc" \
    --num_train_epochs 30 --lr_transformer "$lr" --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $extra \
    >> "$LOGDIR/${name}.log" 2>&1
}

acquire_lock
echo "=== FIXED encoder re-run: rob/robl x base/full x 5 seeds (GPUs ${GPUS[*]}, FIX_GRAPH=1) ==="
for seed in "${SEEDS[@]}"; do
  run_rob "${GPUS[0]}" "rob_base_s${seed}"  "$seed" "$RB" 5e-5 2 2 "$(cfg_args base)" &
  run_rob "${GPUS[1]}" "rob_full_s${seed}"  "$seed" "$RB" 5e-5 2 2 "$(cfg_args full)" &
  wait
  run_rob "${GPUS[0]}" "robl_base_s${seed}" "$seed" "$RL" 3e-5 1 4 "$(cfg_args base)" &
  run_rob "${GPUS[1]}" "robl_full_s${seed}" "$seed" "$RL" 3e-5 1 4 "$(cfg_args full)" &
  wait
  echo "[$(date '+%F %T')] seed $seed round done"
done
echo "=== ENCODER FIXED DONE ==="
