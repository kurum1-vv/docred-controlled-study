# Reproducibility Package — README

**Paper:** *Where Do Gains in Document-Level Relation Extraction Come From? A Reproducible Study of Architectural Modules versus Encoder Scale*

This package lets you reproduce every table in the paper from scratch. All numbers
are means over five random seeds (42, 123, 456, 789, 2024).

> **Anonymous review note.** This repository accompanies a manuscript under
> double-blind review. Author-identifying information has been removed for the
> review process; please evaluate the work without attempting to identify the
> authors. This link will be replaced by the final (attributed) repository upon
> acceptance.

> **Per-seed logs.** All training/evaluation logs are bundled in `logs.tgz`
> (extract with `tar xzf logs.tgz`); it contains `logs_v3/` (encoder comparison,
> Re-DocRED, sensitivity) and `logs_v3_final/` (the 8-configuration BERT ablation).
> The no-document-prior logs are in `noprior/logs_v3/` (see Sect. 5.1 below).

---

## 1. Environment

- Python 3.10, PyTorch 2.5.1+cu121
- `transformers==4.30.0`, `numpy<2`, `opt-einsum`, `ujson`, `scikit-learn`, `pandas`, `tqdm`
- One or more NVIDIA GPUs (results reported on RTX 4090)

```bash
pip install -r requirements.txt
```

## 2. Data

We use the public **DocRED** dataset (`thunlp/docred`).

```bash
export HF_ENDPOINT=https://hf-mirror.com     # optional mirror
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="thunlp/docred", repo_type="dataset",
                  allow_patterns=["data/*.json.gz"], local_dir="hf_tmp")
PY
mkdir -p dataset/docred dataset/meta
for n in dev test train_annotated train_distant; do
  gunzip -c hf_tmp/data/${n}.json.gz > dataset/docred/${n}.json
done
gunzip -c hf_tmp/data/rel_info.json.gz > dataset/meta/rel_info.json
python build_rel2id.py dataset/meta/rel_info.json dataset/meta/rel2id.json
```

Final layout:
```
dataset/docred/{train_annotated,dev,test,train_distant}.json
dataset/meta/rel2id.json          # {Na:0, Pxxxx:1..96}  (num_class = 97)
```

## 3. Code layout

```
run_optimized.py     # training/evaluation entry point
model_optimized.py   # DocREModel (MS-ECC / GR-GR / CB-FL switches)
prepro.py  evaluation.py  utils.py  long_seq.py  losses.py  graph.py
run_v3_experiments.sh   # launches all configurations
collect_v3.py           # aggregates logs into the paper's tables
build_rel2id.py  fix_rel2id.py
```

## 4. Reproducing the tables

Table 1 (factorial ablation) and Table 2 are produced by:

```bash
cd <project root>
# 8 configurations x 5 seeds = 40 runs
GPUS="1 2" SEEDS="42 123 456 789 2024" bash run_v3_experiments.sh e1e2
python collect_v3.py            # prints per-configuration mean ± std, dF1, t, p
```

Configuration switches:
| Configuration | flags |
|---|---|
| backbone | `--disable_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2` |
| +MS-ECC | `--use_ms_ecc --disable_gated_graph --disable_focal_loss --num_graph_layers 2` |
| +GR-GR | `--disable_ms_ecc --use_gated_graph --disable_focal_loss --num_graph_layers 3` |
| +CB-FL | `--disable_ms_ecc --disable_gated_graph --use_focal_loss --num_graph_layers 2` |
| and the three 2-way / 3-way combinations | see `cfg_args` in `run_v3_experiments.sh` |

Encoder-scale comparison (Table 5, Sect. 4.7):
```bash
bash run_v3_experiments.sh dl_large   # downloads roberta-large
bash run_v3_experiments.sh e4
```

## 5. Expected results (5 seeds)

| Configuration | F1 | IgnF1 | dF1 | p |
|---|---|---|---|---|
| EEGRNet backbone | 66.42 ± 0.17 | 64.98 ± 0.18 | – | – |
| + MS-ECC | 66.36 ± 0.13 | 64.97 ± 0.15 | −0.06 | 0.547 |
| + GR-GR | 66.16 ± 0.20 | 64.72 ± 0.20 | −0.26 | 0.088 |
| + CB-FL | 66.47 ± 0.31 | 65.05 ± 0.31 | +0.05 | 0.736 |
| + MS-ECC + GR-GR | 66.09 ± 0.13 | 64.73 ± 0.12 | −0.33 | 0.008 |
| + MS-ECC + CB-FL | 66.27 ± 0.35 | 64.87 ± 0.34 | −0.15 | 0.448 |
| + GR-GR + CB-FL | 66.26 ± 0.16 | 64.82 ± 0.16 | −0.16 | 0.294 |
| full (all three) | 65.91 ± 0.24 | 64.51 ± 0.21 | −0.51 | 0.030 |

Per-seed logs are under `logs_v3/`; each `.log` contains the dev scores printed as
`dev_rel : [P, R, F1]` and `dev_rel_ign : [...]`.

### 5.1 No-document-prior control (Sect. 4.5.2)

The paper reports a backbone retrained without the document-level relation
prior; its dev F1 is statistically indistinguishable from the with-prior
backbone, showing the prior does not explain the dev margin.

| Configuration | F1 |
|---|---|
| backbone (with prior) | 66.42 ± 0.17 |
| backbone (no prior) | 66.44 ± 0.20 |

Artifacts are in `noprior/`: the runner (`run_noprior.sh`), the two source
patches (`model_optimized.no_doc_prior.diff`, `run_optimized.no_doc_prior.diff`),
the environment record (`NOPRIOR_INFO.txt`), and the five per-seed logs
(`noprior/logs_v3/nop_base_s{42,123,456,789,2024}.log`). To reproduce: apply the
two patches and run `bash run_noprior.sh` (it passes `--no_doc_prior` with the
same BERT-base settings as the backbone runs).

## 6. Notes
- `run_optimized.py` defaults to `--num_class 96`; always pass `--num_class 97`.
- The best checkpoint is selected by dev IgnF1; test-time code loads `last.ckpt`.
- `evaluation.py` requires `dataset/docred/train_distant.json` for IgnF1.
