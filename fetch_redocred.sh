#!/bin/bash
# Fetch and prepare Re-DocRED for the e5 experiment.
# Run on the server:  bash fetch_redocred.sh
set -euo pipefail

OPT=/data1/kurumi/opt
HF=/data1/kurumi/.conda/envs/ai_exp/bin/huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com

echo "[1/3] download Re-DocRED"
$HF download tonytan48/Re-DocRED --repo-type dataset --local-dir "$OPT/hf_redocred"

echo "[2/3] convert to DocRED-style names"
mkdir -p "$OPT/dataset/redocred"
cp "$OPT/hf_redocred/train_revised.json" "$OPT/dataset/redocred/train_annotated.json"
cp "$OPT/hf_redocred/dev_revised.json"   "$OPT/dataset/redocred/dev.json"
cp "$OPT/hf_redocred/test_revised.json"  "$OPT/dataset/redocred/test.json"
cp "$OPT/dataset/docred/train_distant.json" "$OPT/dataset/redocred/train_distant.json"

echo "[3/3] verify"
ls -l "$OPT/dataset/redocred"
echo "Re-DocRED ready. Now run:  bash run_v3_experiments.sh e5"
