#!/bin/bash
# paper_original_v3 experiments runner for kurumi account (GPUs 1 and 2 only)
# usage: bash run_v3_experiments.sh check | e1e2 | e4 | e5 | e7 | test
set -u

OPT=/data1/kurumi/opt
cd "$OPT" || { echo "cd $OPT failed"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com

VENV=/data1/kurumi/.conda/envs/ai_exp/bin/python
RESULTS_DIR=$OPT/saved_models
LOGDIR=$OPT/logs_v3
DATA=./dataset/docred
BERT=./bert-base-cased
ROB_BASE=/data1/hcw/models/roberta-base
ROB_LARGE=./roberta-large
SEEDS=(42 123 456 789 2024)
GPUS=(1 2)
LOCKFILE=$OPT/.train.lock
mkdir -p "$LOGDIR" "$RESULTS_DIR" "$OPT/graph_dump"

BASE_EXTRA="--num_class 97 --num_labels 4 --adam_epsilon 1e-6 --max_grad_norm 1.0"

acquire_lock() {
  if [ -f "$LOCKFILE" ]; then
    oldpid=$(cat "$LOCKFILE" 2>/dev/null || true)
    if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
      echo "ERROR: another run is already active (pid $oldpid). Aborting to avoid duplicate jobs."
      exit 1
    fi
  fi
  echo $$ > "$LOCKFILE"
  trap 'rm -f "$LOCKFILE"' EXIT
}

cfg_args() {
  case "$1" in
    base) echo "--disable_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2" ;;
    ms)   echo "--use_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2" ;;
    gr)   echo "--disable_ms_ecc --use_gated_graph --disable_focal_loss --num_graph_layers 3" ;;
    cb)   echo "--disable_ms_ecc --disable_gated_graph --use_focal_loss --num_graph_layers 2" ;;
    msgr) echo "--use_ms_ecc --use_gated_graph --disable_focal_loss --num_graph_layers 3" ;;
    mscb) echo "--use_ms_ecc --disable_gated_graph --use_focal_loss --num_graph_layers 2" ;;
    grcb) echo "--disable_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3" ;;
    full) echo "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3" ;;
    *)    echo "" ;;
  esac
}

run_bert() {
  local gpu="$1" name="$2" seed="$3" extra="$4" data="${5:-$DATA}"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu"
  {
    echo "===== run start ====="; date '+%F %T'; echo "name=$name seed=$seed gpu=$gpu"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  } > "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$data" --model_name_or_path "$BERT" --transformer_type bert \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size 2 --test_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 30 --lr_transformer 5e-5 --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $extra \
    >> "$LOGDIR/${name}.log" 2>&1
}

run_rob() {
  local gpu="$1" name="$2" seed="$3" mpath="$4" lr="$5" bs="$6" acc="$7" extra="$8" data="${9:-$DATA}"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu"
  {
    echo "===== run start ====="; date '+%F %T'; echo "name=$name seed=$seed gpu=$gpu"
  } > "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$data" --model_name_or_path "$mpath" --transformer_type roberta \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size "$bs" --test_batch_size 8 --gradient_accumulation_steps "$acc" \
    --num_train_epochs 30 --lr_transformer "$lr" --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $extra \
    >> "$LOGDIR/${name}.log" 2>&1
}

run_check() {
  echo "=== setup check ==="
  $VENV -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpus', torch.cuda.device_count())"
  for f in "$DATA/train_annotated.json" "$DATA/dev.json" "$DATA/test.json" "$OPT/dataset/meta/rel2id.json" "$BERT/config.json"; do
    [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
  done
  [ -f "$ROB_BASE/config.json" ] && echo "OK   $ROB_BASE" || echo "MISS $ROB_BASE"
  [ -f "$ROB_LARGE/config.json" ] && echo "OK   $ROB_LARGE" || echo "MISS $ROB_LARGE (run: bash run_v3_experiments.sh dl_large)"
  echo "--- GPU usage ---"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
  if [ -f "$LOCKFILE" ]; then
    p=$(cat "$LOCKFILE" 2>/dev/null || true)
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then echo "training active (pid $p)"; else echo "no active training"; fi
  else
    echo "no active training"
  fi
}

run_dl_large() {
  echo "downloading roberta-large to $ROB_LARGE ..."
  $VENV -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='roberta-large', local_dir='$ROB_LARGE')"
  echo "done"
}

run_e1e2() {
  acquire_lock
  echo "=== E1/E2: BERT-base 8 configs x 5 seeds (40 runs, GPUs ${GPUS[*]}) ==="
  for seed in "${SEEDS[@]}"; do
    n=0
    for cfg in base ms gr cb msgr mscb grcb full; do
      gpu=${GPUS[$(( n % 2 ))]}
      run_bert "$gpu" "bert_${cfg}_s${seed}" "$seed" "$(cfg_args "$cfg")" &
      n=$(( n + 1 ))
      if [ $(( n % 2 )) -eq 0 ]; then wait; fi
    done
    wait
  done
  echo "=== E1/E2 DONE ==="
}

run_e4() {
  acquire_lock
  echo "=== E4: encoder comparison (20 runs) ==="
  if [ ! -f "$ROB_LARGE/config.json" ]; then echo "roberta-large missing; run dl_large first"; return 1; fi
  for seed in "${SEEDS[@]}"; do
    run_rob "${GPUS[0]}" "rob_base_s${seed}" "$seed" "$ROB_BASE" 5e-5 2 2 "$(cfg_args base)" &
    run_rob "${GPUS[1]}" "rob_full_s${seed}" "$seed" "$ROB_BASE" 5e-5 2 2 "$(cfg_args full)" &
    wait
    run_rob "${GPUS[0]}" "robl_base_s${seed}" "$seed" "$ROB_LARGE" 3e-5 1 4 "$(cfg_args base)" &
    run_rob "${GPUS[1]}" "robl_full_s${seed}" "$seed" "$ROB_LARGE" 3e-5 1 4 "$(cfg_args full)" &
    wait
  done
  echo "=== E4 DONE ==="
}

run_e5() {
  acquire_lock
  local rd="./dataset/redocred"
  [ -d "$rd" ] || { echo "Re-DocRED not found at $rd"; return 1; }
  echo "=== E5: Re-DocRED base/full x 5 seeds ==="
  for seed in "${SEEDS[@]}"; do
    run_bert "${GPUS[0]}" "redoc_base_s${seed}" "$seed" "$(cfg_args base)" "$rd" &
    run_bert "${GPUS[1]}" "redoc_full_s${seed}" "$seed" "$(cfg_args full)" "$rd" &
    wait
  done
  echo "=== E5 DONE ==="
}

run_e7() {
  acquire_lock
  echo "=== E7: hyperparameter sensitivity (full, seed 42) ==="
  for v in 1.0 2.0 3.0; do
    run_bert "${GPUS[0]}" "sens_gamma_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 --focal_gamma $v"
  done
  for v in 0.3 0.5 0.7; do
    run_bert "${GPUS[0]}" "sens_lambda_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 --focal_lambda $v"
  done
  for v in 2 3 4; do
    run_bert "${GPUS[0]}" "sens_layers_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers $v"
  done
  echo "=== E7 DONE ==="
}

run_test() {
  local load="$1" testfile="${2:-test.json}"
  [ -d "$load" ] || { echo "load dir not found: $load"; return 1; }
  if [ -f "$load/best.ckpt" ]; then cp "$load/best.ckpt" "$load/last.ckpt"; fi
  echo "[$(date '+%F %T')] test inference $load"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $VENV -u run_optimized.py \
    --data_dir "$DATA" --model_name_or_path "$BERT" --transformer_type bert \
    --num_class 97 --num_labels 4 --test_batch_size 8 \
    --max_seq_length 1024 --max_sent_num 25 \
    --use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 \
    --load_path "$load" --test_file "$testfile" --pred_file results.json \
    > "$load/test_infer.log" 2>&1
  echo "predictions -> $load/results.json"
}

case "${1:-}" in
  check)    run_check ;;
  dl_large) run_dl_large ;;
  e1e2)     run_e1e2 ;;
  e4)       run_e4 ;;
  e5)       run_e5 ;;
  e7)       run_e7 ;;
  test)     run_test "$2" "${3:-test.json}" ;;
  *) echo "usage: bash run_v3_experiments.sh check|dl_large|e1e2|e4|e5|e7|test <dir>"; exit 1 ;;
esac
echo "ALL DONE"
