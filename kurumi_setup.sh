#!/bin/bash
set -euo pipefail

OPT=/data1/kurumi/opt
PY=/data1/kurumi/.conda/envs/ai_exp/bin/python
PIP=/data1/kurumi/.conda/envs/ai_exp/bin/pip
export HF_ENDPOINT=https://hf-mirror.com

echo "[1/7] create dirs"
mkdir -p "$OPT/dataset/docred" "$OPT/dataset/meta" "$OPT/graph_dump" "$OPT/logs_v3" "$OPT/saved_models" "$OPT/hf_tmp"

echo "[2/7] install packages (pypi.org)"
$PIP install -i https://pypi.org/simple "numpy==1.26.4" "transformers==4.30.0" opt-einsum ujson huggingface_hub
$PY -c "import transformers, huggingface_hub, numpy; print('pkgs ok', transformers.__version__, numpy.__version__)"

echo "[3/7] download DocRED (hf-mirror)"
$PY - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="thunlp/docred", repo_type="dataset",
                      allow_patterns=["data/dev.json.gz", "data/test.json.gz",
                                      "data/train_annotated.json.gz", "data/rel_info.json.gz"],
                      local_dir="$OPT/hf_tmp")
print("downloaded to", p)
PY
ls -l "$OPT/hf_tmp/data"

echo "[4/7] unzip"
for n in dev test train_annotated; do
  gunzip -f -c "$OPT/hf_tmp/data/$n.json.gz" > "$OPT/dataset/docred/$n.json"
done
gunzip -f -c "$OPT/hf_tmp/data/rel_info.json.gz" > "$OPT/dataset/meta/rel_info.json"
ls -l "$OPT/dataset/docred" "$OPT/dataset/meta"

echo "[5/7] build rel2id.json"
$PY "$OPT/build_rel2id.py" "$OPT/dataset/meta/rel_info.json" "$OPT/dataset/meta/rel2id.json"

echo "[6/7] download bert-base-cased (hf-mirror)"
$PY - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="bert-base-cased", local_dir="$OPT/bert-base-cased")
print("bert done")
PY

echo "[7/7] sanity check"
$PY "$OPT/fix_rel2id.py" "$OPT/dataset/meta/rel2id.json"
$PY -c "import torch, transformers, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__, 'numpy', numpy.__version__)"
echo "SETUP DONE"
