#!/usr/bin/env python
import argparse, glob, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from utils import set_seed, collate_fn
from prepro import read_docred
from model_optimized import DocREModel
from run_optimized import load_input

DATA_DIR     = "./dataset/docred"
BERT_PATH    = "./bert-base-cased"
SAVE_DIR     = "/data1/kurumi/opt/saved_models"
DEV_FILE     = "dev.json"
MAX_SEQ_LEN  = 1024
MAX_SENT_NUM = 25
EVI_THRESH   = 0.2
NUM_LABELS   = 4

CFG_FLAGS = {
    "ms":   (True, False, False, 2),
    "msgr": (True, True,  False, 3),
    "mscb": (True, False, True,  2),
    "full": (True, True,  True,  3),
}

def find_ckpt(cfg, seed):
    cands = sorted(glob.glob(os.path.join(SAVE_DIR, f"bert_{cfg}_s{seed}_*", "best.ckpt")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(SAVE_DIR, f"bert_{cfg}_s{seed}_*", "last.ckpt")))
    return cands[-1] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=list(CFG_FLAGS.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ms, use_gated, use_focal, n_layers = CFG_FLAGS[args.config]

    config = AutoConfig.from_pretrained(BERT_PATH, num_labels=97)
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    config.transformer_type = "bert"
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    print("Loading dev data ...", flush=True)
    dev_features = read_docred(os.path.join(DATA_DIR, DEV_FILE), tokenizer,
                               transformer_type="bert", max_seq_length=MAX_SEQ_LEN)
    print(f"dev docs: {len(dev_features)}", flush=True)

    rows = []
    for seed in args.seeds:
        ckpt = find_ckpt(args.config, seed)
        if ckpt is None:
            print(f"[seed {seed}] no checkpoint found, skip", flush=True)
            continue
        print(f"[seed {seed}] loading {ckpt}", flush=True)

        set_seed(argparse.Namespace(seed=seed))
        transformer = AutoModel.from_pretrained(BERT_PATH, config=config)
        model = DocREModel(config, transformer, tokenizer,
                           num_labels=NUM_LABELS,
                           max_sent_num=MAX_SENT_NUM,
                           evi_thresh=EVI_THRESH,
                           use_ms_ecc=use_ms,
                           use_gated_graph=use_gated,
                           use_focal_loss=use_focal,
                           num_graph_layers=n_layers)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        taus = []
        def hook(module, inp, out):
            taus.append(out.detach().float().cpu().view(-1))
        h = model.learnable_thresh.register_forward_hook(hook)

        loader = DataLoader(dev_features, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)
        with torch.no_grad():
            for batch in loader:
                model(**load_input(batch, device, tag="dev"))
        h.remove()

        tau = torch.cat(taus).numpy()
        q = np.percentile(tau, [1, 5, 25, 50, 75, 95, 99]).round(4)
        row = dict(seed=seed, pairs=int(tau.size),
                   mean=round(float(tau.mean()), 4), std=round(float(tau.std()), 4),
                   min=round(float(tau.min()), 4), max=round(float(tau.max()), 4),
                   q01=q[0], q05=q[1], q25=q[2], q50=q[3], q75=q[4], q95=q[5], q99=q[6],
                   frac_below_0p2=round(float((tau < 0.2).mean()), 4),
                   frac_in_0p1_0p5=round(float(((tau >= 0.1) & (tau <= 0.5)).mean()), 4))
        rows.append(row)
        print(f"[seed {seed}] pairs={row['pairs']} tau mean={row['mean']} std={row['std']} "
              f"min={row['min']} max={row['max']}", flush=True)
        print(f"           quantiles(1/5/25/50/75/95/99)={list(q)}", flush=True)
        print(f"           frac(tau<0.2)={row['frac_below_0p2']}  "
              f"frac(0.1<=tau<=0.5)={row['frac_in_0p1_0p5']}", flush=True)

        del model
        torch.cuda.empty_cache()

    if rows:
        import csv
        out = args.out_csv or f"tau_stats_bert_{args.config}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out}")
        means = [r["mean"] for r in rows]
        mins = [r["min"] for r in rows]
        maxs = [r["max"] for r in rows]
        print(f"OVERALL ({args.config}): seed-mean of tau mean = {np.mean(means):.4f}; "
              f"min across seeds in [{min(mins)}, {max(mins)}]; "
              f"max across seeds in [{min(maxs)}, {max(maxs)}]")

if __name__ == "__main__":
    main()
