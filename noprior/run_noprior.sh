#!/bin/bash
set -u
OPT=/data1/kurumi/opt; cd "$OPT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VENV=/data1/kurumi/.conda/envs/ai_exp/bin/python
RESULTS_DIR=$OPT/saved_models; LOGDIR=$OPT/logs_v3; DATA=./dataset/docred; BERT=./bert-base-cased
BASE_EXTRA="--num_class 97 --num_labels 4 --adam_epsilon 1e-6 --max_grad_norm 1.0"
BASE_ARGS="--disable_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2"
run_bert(){
  local gpu="$1" name="$2" seed="$3"
  { echo "===== run start ====="; date '+%F %T'; echo "name=$name seed=$seed gpu=$gpu"; } > "$LOGDIR/${name}.log"
  CUDA_VISIBLE_DEVICES=$gpu $VENV -u run_optimized.py --do_train \
    --data_dir "$DATA" --model_name_or_path "$BERT" --transformer_type bert \
    --train_file train_annotated.json --dev_file dev.json \
    --train_batch_size 2 --test_batch_size 8 --gradient_accumulation_steps 2 \
    --num_train_epochs 30 --lr_transformer 5e-5 --lr_added 1e-4 --warmup_ratio 0.06 \
    --max_seq_length 1024 --max_sent_num 25 --no_doc_prior \
    --evi_thresh 0.2 --evi_lambda 0.1 --doc_lambda 0.2 \
    --model_name "$name" --save_path "$RESULTS_DIR" --seed "$seed" $BASE_EXTRA $BASE_ARGS \
    >> "$LOGDIR/${name}.log" 2>&1
}
seeds=(42 123 456 789 2024)
i=0
while [ $i -lt 5 ]; do
  run_bert 1 "nop_base_s${seeds[$i]}" "${seeds[$i]}" &
  p1=$!
  if [ $((i+1)) -lt 5 ]; then run_bert 2 "nop_base_s${seeds[$i+1]}" "${seeds[$i+1]}" & p2=$!; fi
  wait $p1 $p2 2>/dev/null
  i=$((i+2))
done
echo "=== NOPRIOR DONE ==="
