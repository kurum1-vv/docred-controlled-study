#!/bin/bash
# paper_original_v3 experiments runner (optimized)
# usage: bash run_v3_experiments.sh check | e1e2 | e4 | e5 | e7 | test
# recommend: nohup bash run_v3_experiments.sh e1e2 > run_e1e2.out 2>&1 &
set -u

cd ~/docred_project/optimized || { echo "cd ~/docred_project/optimized failed"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

VENV=~/paper_code/chapter4_work/.venv/bin/python
RESULTS_DIR=~/docred_project/optimized/saved_models
LOGDIR=~/docred_project/optimized/logs_v3
DATA=./dataset/docred
BERT=./bert-base-cased
ROB_BASE=/data1/hcw/models/roberta-base
ROB_LARGE=/data1/hcw/models/roberta-large
SEEDS=(42 123 456 789 2024)
mkdir -p "$LOGDIR" "$RESULTS_DIR"

BASE_EXTRA="--num_class 97 --num_labels 4 --adam_epsilon 1e-6 --max_grad_norm 1.0"

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

header() {
  {
    echo "===== run start ====="
    date '+%F %T'
    echo "seed=$2 cfg=$1 gpu=$3"
    git -C ~/docred_project/optimized rev-parse HEAD 2>/dev/null || echo "no-git"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null
    echo "====================="
  } > "$LOGDIR/${4}.log"
}

run_bert() {
  local gpu="$1" name="$2" seed="$3" extra="$4" data="${5:-$DATA}"
  header "$name" "$seed" "$gpu" "$name"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu" >> "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$data" --model_name_or_path "$BERT" --transformer_type bert \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size 4 --test_batch_size 8 --gradient_accumulation_steps 1 \
    --num_train_epochs 30 --lr_transformer 5e-5 --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $extra \
    >> "$LOGDIR/${name}.log" 2>&1
}

run_rob() {
  local gpu="$1" name="$2" seed="$3" mpath="$4" lr="$5" bs="$6" acc="$7" extra="$8" data="${9:-$DATA}"
  header "$name" "$seed" "$gpu" "$name"
  echo "[$(date '+%F %T')] $name seed=$seed GPU=$gpu" >> "$LOGDIR/${name}.log"
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
  for f in "$DATA/train_annotated.json" "$DATA/dev.json" "$DATA/test.json" "$BERT/config.json"; do
    [ -f "$f" ] && echo "OK   $f" || echo "MISS $f"
  done
  [ -f "$ROB_BASE/config.json" ] && echo "OK   $ROB_BASE" || echo "MISS $ROB_BASE"
  [ -f "$ROB_LARGE/config.json" ] && echo "OK   $ROB_LARGE" || echo "MISS $ROB_LARGE"
}

run_e1e2() {
  echo "=== E1/E2: BERT-base 8 configs x 5 seeds (40 runs) ==="
  for seed in "${SEEDS[@]}"; do
    n=0
    for cfg in base ms gr cb msgr mscb grcb full; do
      gpu=$(( n % 2 + 1 ))
      run_bert "$gpu" "bert_${cfg}_s${seed}" "$seed" "$(cfg_args "$cfg")" &
      n=$(( n + 1 ))
      [ $(( n % 2 )) -eq 0 ] && wait
    done
    wait
  done
  echo "=== E1/E2 DONE ==="
}

run_e4() {
  echo "=== E4: encoder comparison (20 runs) ==="
  for seed in "${SEEDS[@]}"; do
    run_rob 1 "rob_base_s${seed}" "$seed" "$ROB_BASE" 5e-5 4 1 "$(cfg_args base)" &
    run_rob 2 "rob_full_s${seed}" "$seed" "$ROB_BASE" 5e-5 4 1 "$(cfg_args full)" &
    wait
    run_rob 1 "robl_base_s${seed}" "$seed" "$ROB_LARGE" 3e-5 2 2 "$(cfg_args base)" &
    run_rob 2 "robl_full_s${seed}" "$seed" "$ROB_LARGE" 3e-5 2 2 "$(cfg_args full)" &
    wait
  done
  echo "=== E4 DONE ==="
}

run_e5() {
  local rd="./dataset/redocred"
  [ -d "$rd" ] || { echo "Re-DocRED not found at $rd"; return 1; }
  echo "=== E5: Re-DocRED base/full x 5 seeds ==="
  for seed in "${SEEDS[@]}"; do
    run_bert 1 "redoc_base_s${seed}" "$seed" "$(cfg_args base)" "$rd" &
    run_bert 2 "redoc_full_s${seed}" "$seed" "$(cfg_args full)" "$rd" &
    wait
  done
  echo "=== E5 DONE ==="
}

run_e7() {
  echo "=== E7: hyperparameter sensitivity (full model, seed 42) ==="
  for v in 1.0 2.0 3.0; do
    run_bert 1 "sens_gamma_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 --focal_gamma $v"
    wait
  done
  for v in 0.3 0.5 0.7; do
    run_bert 1 "sens_lambda_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 --focal_lambda $v"
    wait
  done
  for v in 2 3 4; do
    run_bert 1 "sens_layers_${v}" 42 "--use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers $v"
    wait
  done
  echo "=== E7 DONE ==="
}

run_test() {
  local load="$1" testfile="${2:-test.json}"
  [ -d "$load" ] || { echo "load dir not found: $load"; return 1; }
  if [ -f "$load/best.ckpt" ]; then
    cp "$load/best.ckpt" "$load/last.ckpt"
    echo "copied best.ckpt -> last.ckpt in $load"
  fi
  echo "[$(date '+%F %T')] test inference $load"
  CUDA_VISIBLE_DEVICES=0 $VENV -u run_optimized.py \
    --data_dir "$DATA" --model_name_or_path "$BERT" --transformer_type bert \
    --num_class 97 --num_labels 4 --test_batch_size 8 \
    --max_seq_length 1024 --max_sent_num 25 \
    --use_ms_ecc --use_gated_graph --use_focal_loss --num_graph_layers 3 \
    --load_path "$load" --test_file "$testfile" --pred_file results.json \
    > "$load/test_infer.log" 2>&1
  echo "predictions -> $load/results.json ; submit this file to CodaLab"
}

case "${1:-}" in
  check) run_check ;;
  e1e2)  run_e1e2 ;;
  e4)    run_e4 ;;
  e5)    run_e5 ;;
  e7)    run_e7 ;;
  test)  run_test "$2" "${3:-test.json}" ;;
  *) echo "usage: bash run_v3_experiments.sh check | e1e2 | e4 | e5 | e7 | test <saved_dir> [test_file]"; exit 1 ;;
esac
echo "ALL DONE"
