import argparse
import os
import datetime
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import json

from model_optimized import DocREModel
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred
from evaluation import to_official, official_evaluate, merge_results
from sklearn.metrics import f1_score, precision_score, recall_score


def load_input(batch, device, tag="dev"):
    input = {
        'input_ids': batch[0].to(device),
        'attention_mask': batch[1].to(device),
        'labels': batch[2].to(device),
        'entity_pos': batch[3],
        'hts': batch[4],
        'sent_pos': batch[5],
        'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
        'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
        'tag': tag,
        'doc_rel': batch[8],
        'hts_graph': batch[9]
    }
    return input


def find_best_threshold(probs, labels):
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.9, 0.01)
    doc_precision = 0.0
    doc_recall = 0.0
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, preds, average='micro')
        precision = precision_score(labels, preds, average='micro')
        recall = recall_score(labels, preds, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            doc_precision = precision
            doc_recall = recall
    return best_threshold, best_f1, doc_precision, doc_recall


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, evi_preds = [], []
    scores, topks = [], []
    doc_preds, doc_rel = [], []
    best_threshold, doc_f1, doc_precision, doc_recall = 0, 0, 0, 0
    alpha_all = []

    for batch in tqdm(dataloader, desc=f"Evaluating"):
        model.eval()
        inputs = load_input(batch, args.device, tag)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            if "alpha" in outputs:
                alpha = outputs["alpha"].cpu().numpy()
                alpha_all.extend(alpha.tolist())
            if "doc_prob" in outputs:
                doc_pred = outputs["doc_prob"].cpu().numpy()
                doc_pred[np.isnan(doc_pred)] = 0
                doc_preds.append(doc_pred)
                doc_rel.extend(batch[8])
            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())
                topks.append(outputs["topks"].cpu().numpy())
            if "evi_pred" in outputs:
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)

    preds = np.concatenate(preds, axis=0)
    if doc_rel != []:
        doc_preds = np.concatenate(doc_preds, axis=0).astype(np.float32)
        doc_labels = np.array(doc_rel, dtype=np.float32)
        best_threshold, doc_f1, doc_precision, doc_recall = find_best_threshold(doc_preds, doc_labels > 0)
    if scores != []:
        scores = np.concatenate(scores, axis=0)
        topks = np.concatenate(topks, axis=0)
    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)

    official_results, results = to_official(preds, features, evi_preds=evi_preds, scores=scores, topks=topks)

    if len(official_results) > 0:
        if tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.dev_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.test_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]

    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign],
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores_dict = {
        "dev_F1": best_re[-1] * 100,
        "dev_evi_F1": best_evi[-1] * 100,
        "dev_F1_ign": best_re_ign[-1] * 100,
        "dev_doc_F1": doc_f1,
        "dev_doc_precision": doc_precision,
        "dev_doc_recall": doc_recall,
        "dev_threshold": best_threshold,
        "mean_alpha": float(np.mean(alpha_all)) if alpha_all else 0.0,
    }
    return scores_dict, output, official_results, results


def train(args, model, train_features, dev_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler()
        print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

        for epoch in tqdm(train_iterator, desc='Train epoch'):
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                optimizer.zero_grad()
                model.train()
                inputs = load_input(batch, args.device)
                outputs = model(**inputs)
                loss = [outputs["loss"]["rel_loss"]]

                if inputs["sent_labels"] is not None:
                    loss.append(outputs["loss"]["evi_loss"] * args.evi_lambda)
                if inputs["teacher_attns"] is not None:
                    loss.append(outputs["loss"]["attn_loss"] * args.attn_lambda)
                if inputs["doc_rel"] is not None:
                    loss.append(outputs["loss"]["doc_loss"] * args.doc_lambda)

                loss = sum(loss) / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if (step + 1) % 50 == 0:
                    print(f"Epoch {epoch}, Step {step+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

                is_last_step = (epoch == train_iterator[-1]) and ((step + 1) == len(train_dataloader))
                should_eval = args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0
                if is_last_step or should_eval:
                    dev_scores, dev_output, official_results, results = evaluate(args, model, dev_features, tag="dev")
                    print(f"Epoch {epoch}: {dev_output}")
                    print(f"Scores: {dev_scores}")

                    if dev_scores["dev_F1_ign"] > best_score:
                        best_score = dev_scores["dev_F1_ign"]
                        best_offi_results = official_results
                        best_output = dev_output
                        best_results = results
                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        print(f"Saving best model to {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)

                    if is_last_step:
                        ckpt_file = os.path.join(args.save_path, "last.ckpt")
                        torch.save(model.state_dict(), ckpt_file)
                        pred_file = os.path.join(args.save_path, args.pred_file)
                        score_file = os.path.join(args.save_path, "scores.csv")
                        results_file = os.path.join(args.save_path, f"topk_{args.pred_file}")
                        dump_to_file(best_offi_results, pred_file, best_output, score_file, best_results, results_file)

        return num_steps

    new_layer = ["extractor", "bilinear", "graph", "learnable", "token_evi", "evi_fusion", "edge_emb", "gate", "focal"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def dump_to_file(offi, offi_path, scores, score_path, results=[], res_path=""):
    print(f"Saving predictions to {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))
    print(f"Saving scores to {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')
    if len(results) != 0:
        assert res_path != ""
        print(f"Saving topk results to {res_path} ...")
        json.dump(results, open(res_path, "w"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="", type=str)
    parser.add_argument("--pred_file", default="results.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--max_sent_num", default=25, type=int)
    parser.add_argument("--evi_thresh", default=0.2, type=float)
    parser.add_argument("--evi_lambda", default=0.1, type=float)
    parser.add_argument("--attn_lambda", default=1.0, type=float)
    parser.add_argument("--doc_lambda", default=0.2, type=float)
    parser.add_argument("--lr_transformer", default=5e-5, type=float)
    parser.add_argument("--lr_added", default=1e-4, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--evaluation_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_class", type=int, default=96)
    # Optimized model args
    parser.add_argument("--use_ms_ecc", action="store_true", default=True)
    parser.add_argument("--use_gated_graph", action="store_true", default=True)
    parser.add_argument("--use_focal_loss", action="store_true", default=True)
    parser.add_argument("--num_graph_layers", default=3, type=int)
    parser.add_argument("--focal_gamma", default=2.0, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--focal_lambda", default=0.5, type=float)
    parser.add_argument("--model_name", default="EEGRNet_Optimized", type=str)
    # Ablation control
    parser.add_argument("--disable_ms_ecc", action="store_true")
    parser.add_argument("--disable_gated_graph", action="store_true")
    parser.add_argument("--disable_focal_loss", action="store_true")
    args = parser.parse_args()

    if args.disable_ms_ecc:
        args.use_ms_ecc = False
    if args.disable_gated_graph:
        args.use_gated_graph = False
    if args.disable_focal_loss:
        args.use_focal_loss = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    config.transformer_type = args.transformer_type
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    set_seed(args)
    read = read_docred

    model = DocREModel(config, model, tokenizer,
                    num_labels=args.num_labels,
                    max_sent_num=args.max_sent_num,
                    evi_thresh=args.evi_thresh,
                    use_ms_ecc=args.use_ms_ecc,
                    use_gated_graph=args.use_gated_graph,
                    use_focal_loss=args.use_focal_loss,
                    num_graph_layers=args.num_graph_layers,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    focal_lambda=args.focal_lambda)
    model.to(args.device)

    if args.load_path != "":
        model_path = os.path.join(args.load_path, "last.ckpt")
        model.load_state_dict(torch.load(model_path, map_location=device))

    if args.do_train:
        save_path = os.path.join(args.save_path, f"{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        create_directory(save_path)
        args.save_path = save_path

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        print(f"Loading training data from {train_file}...")
        train_features = read(train_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)
        print(f"Loading dev data from {dev_file}...")
        dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)
        print(f"Train: {len(train_features)} docs, Dev: {len(dev_features)} docs")

        train(args, model, train_features, dev_features)
    else:
        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)
        test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test")
        print(f"Test Results: {test_output}")
        print(f"Test Scores: {test_scores}")
        offi_path = os.path.join(args.load_path, args.pred_file)
        score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
        res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")
        dump_to_file(official_results, offi_path, test_output, score_path, results, res_path)


if __name__ == "__main__":
    main()
