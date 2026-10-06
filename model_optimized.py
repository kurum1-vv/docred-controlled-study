import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from graph import AttentionGCNLayer


class LearnableThreshold(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, hs, ts):
        combined = torch.cat([hs, ts], dim=-1)
        threshold = self.mlp(combined).squeeze(-1)
        return threshold


class TokenLevelEvidenceRefiner(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, entity_repr, seq_output, doc_attn):
        weighted_seq = contract("ld,rl->rd", seq_output, doc_attn)
        refined = self.proj(entity_repr + weighted_seq)
        return refined


class GatedResidualLayer(nn.Module):
    def __init__(self, hidden_size, edges, nhead=2, iters=2, dropout=0.1):
        super().__init__()
        self.graph_attention = AttentionGCNLayer(edges, hidden_size, nhead=nhead, iters=iters)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.edge_emb = nn.Embedding(len(edges) + 1, hidden_size)

    def forward(self, nodes_embed, node_adj):
        residual = nodes_embed
        graph_out, attn = self.graph_attention(nodes_embed, node_adj)
        gate = torch.sigmoid(self.gate(torch.cat([graph_out, residual], dim=-1)))
        output = gate * graph_out + (1 - gate) * residual
        output = self.norm(output)
        return output, attn


class FocalATLoss(nn.Module):
    def __init__(self, num_labels, gamma=2.0, alpha=0.25, focal_lambda=0.5):
        super().__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.focal_lambda = focal_lambda
        self.at_loss = ATLoss()

    def get_label(self, logits, num_labels=-1):
        return self.at_loss.get_label(logits, num_labels)

    def get_score(self, logits, num_labels=-1):
        return self.at_loss.get_score(logits, num_labels)

    def forward(self, logits, labels):
        if logits.size(-1) != labels.size(-1):
            pad = torch.zeros(labels.size(0), logits.size(-1) - labels.size(-1), device=labels.device, dtype=labels.dtype)
            labels = torch.cat([labels, pad], dim=-1)
        at_loss = self.at_loss(logits.float(), labels.float())
        probs = torch.sigmoid(logits[:, 1:].float())
        labels_f = labels[:, 1:].float()
        p_t = probs * labels_f + (1 - probs) * (1 - labels_f)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits[:, 1:].float(), labels_f, reduction='none')
        focal_loss = (self.alpha * focal_weight * bce).mean()
        return at_loss + self.focal_lambda * focal_loss


class DocREModel(nn.Module):

    def __init__(self, config, model, tokenizer,
                emb_size=768, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2,
                hts_graph_mode="full",
                hts_edges=None,
                use_ms_ecc=True,
                use_gated_graph=True,
                use_focal_loss=True,
                num_graph_layers=3,
                focal_gamma=2.0,
                focal_alpha=0.25,
                focal_lambda=0.5):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.use_ms_ecc = use_ms_ecc
        self.use_gated_graph = use_gated_graph
        self.use_focal_loss = use_focal_loss

        if use_focal_loss:
            self.loss_fnt = FocalATLoss(config.num_labels, gamma=focal_gamma, alpha=focal_alpha, focal_lambda=focal_lambda)
        else:
            self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.head_extractor2 = nn.Linear(self.hidden_size * 3, emb_size)
        self.tail_extractor2 = nn.Linear(self.hidden_size * 3, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.bilinear2 = nn.Linear(emb_size * block_size, self.hidden_size)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.rs_gate = nn.Linear(self.hidden_size * 2, 1)
        self.hts_graph_mode = hts_graph_mode
        self.hts_edges = hts_edges if hts_edges is not None else ['chain', 'share_head', 'share_tail', 'share_any', 'evi_overlap']

        if use_ms_ecc:
            self.learnable_thresh = LearnableThreshold(self.hidden_size)
            self.token_evi_refiner = TokenLevelEvidenceRefiner(self.hidden_size)
            self.evi_fusion_gate = nn.Linear(self.hidden_size * 3, self.hidden_size)

        if use_gated_graph:
            self.hts_graph_layers = nn.ModuleList(
                GatedResidualLayer(self.hidden_size, self.hts_edges, nhead=2, iters=2)
                for _ in range(num_graph_layers)
            )
        else:
            self.hts_graph_layers = nn.ModuleList(
                AttentionGCNLayer(self.hts_edges, self.hidden_size, nhead=2, iters=2) for _ in range(2)
            )
        self.num_graph_layers = num_graph_layers

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for eid, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e):
                        if start + offset < c:
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        ht_atts = torch.cat(ht_atts, dim=0)
        return hss, rss, tss, ht_atts, rels_per_batch

    def graph_hts(self, hts, hts_graph, rels_per_batch):
        if self.hts_graph_mode != "full" or hts_graph is None:
            return hts
        max_node = max([graph.shape[0] for graph in hts_graph])
        n = len(rels_per_batch)
        graph_fea = torch.zeros(n, max_node, self.hidden_size, device=hts.device)
        graph_adj = torch.zeros(n, max_node, max_node, device=hts.device)
        new_hts = torch.zeros([hts.shape[0], hts.shape[1]], device=hts.device)
        for i, graph in enumerate(hts_graph):
            nodes_num = graph.shape[0]
            graph_adj[i, :nodes_num, :nodes_num] = torch.from_numpy(graph)
        for i in range(n):
            graph_fea[i, :rels_per_batch[i], :] = hts[:rels_per_batch[i]]
            graph_fea[i, rels_per_batch[i]:, :] = torch.zeros(self.hidden_size).to(hts.device)

        for graph_layer in self.hts_graph_layers:
            if self.use_gated_graph:
                graph_fea, _ = graph_layer(graph_fea, graph_adj)
            else:
                graph_fea, _ = graph_layer(graph_fea, graph_adj)

        for i in range(n):
            new_hts[:rels_per_batch[i]] = graph_fea[i][:rels_per_batch[i]]
        return new_hts

    def forward_rel(self, hs, ts, rs, hts_graph, batch_rel):
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        hts = self.bilinear2(bl)
        hts = self.graph_hts(hts, hts_graph, batch_rel)
        hs = torch.tanh(self.head_extractor2(torch.cat([hs, rs, hts], dim=-1)))
        ts = torch.tanh(self.tail_extractor2(torch.cat([ts, rs, hts], dim=-1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        return logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]
            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1)
            rel_sent_attn.append(sum_attn)
        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                sent_pos=None,
                sent_labels=None,
                teacher_attns=None,
                tag="train",
                doc_rel=None,
                hts_graph=None):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)
        doc_cls = sequence_output[:,0,:]
        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)
        s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)

        if self.use_ms_ecc:
            dyn_thresh = self.learnable_thresh(hs, ts)
            evi_mask = s_attn > dyn_thresh.unsqueeze(1)
        else:
            evi_mask = s_attn > self.evi_thresh

        if self.use_ms_ecc:
            token_evi_list = []
        r_evi_list, r_non_list = [], []
        start_idx = 0
        for i in range(len(sent_pos)):
            end_idx = start_idx + batch_rel[i]
            curr_attn = doc_attn[start_idx:end_idx]
            curr_mask = evi_mask[start_idx:end_idx]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]
            att_evi = curr_attn.clone()
            att_non = curr_attn.clone()
            for k, idxs in enumerate(curr_sent_pos):
                w = curr_mask[:, k].unsqueeze(1).float()
                att_evi[:, idxs] = att_evi[:, idxs] * w
                att_non[:, idxs] = att_non[:, idxs] * (1.0 - w)
            att_evi = att_evi / (att_evi.sum(1, keepdim=True) + 1e-30)
            att_non = att_non / (att_non.sum(1, keepdim=True) + 1e-30)
            r_evi_list.append(contract("ld,rl->rd", sequence_output[i], att_evi))
            r_non_list.append(contract("ld,rl->rd", sequence_output[i], att_non))
            if self.use_ms_ecc:
                token_evi_list.append(self.token_evi_refiner(hs[start_idx:end_idx], sequence_output[i], doc_attn[start_idx:end_idx]))
            start_idx = end_idx
        r_evi = torch.cat(r_evi_list, dim=0)
        r_non = torch.cat(r_non_list, dim=0)

        if self.use_ms_ecc:
            token_evi = torch.cat(token_evi_list, dim=0)
            fusion_input = torch.cat([r_evi, r_non, token_evi], dim=-1)
            fusion_gate = torch.sigmoid(self.evi_fusion_gate(fusion_input))
            rs = fusion_gate * r_evi + (1 - fusion_gate) * r_non
        else:
            alpha = torch.sigmoid(self.rs_gate(torch.cat([hs, ts], dim=-1)))
            rs = alpha * r_evi + (1.0 - alpha) * r_non

        logits = self.forward_rel(hs, ts, rs, hts_graph, batch_rel)
        doc_logits = self.classifier(doc_cls)
        doc_prob = torch.sigmoid(doc_logits)
        logits = doc_prob.repeat_interleave(torch.tensor(batch_rel,dtype=torch.int32).to(doc_logits.device), dim=0) + logits
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if self.use_ms_ecc:
            output["alpha"] = fusion_gate.mean(dim=-1)
        else:
            output["alpha"] = alpha.squeeze(-1)

        if doc_rel != None:
            output["doc_prob"] = doc_prob.to(sequence_output)
        if sent_labels != None:
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]:
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

        if tag == "infer":
            output["attns"] = doc_attn.split(batch_rel)
        else:
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
            if doc_rel != None:
                doc_rel = torch.tensor(doc_rel).to(logits)
                if doc_rel.size(-1) != doc_logits.size(-1):
                    pad_size = doc_logits.size(-1) - doc_rel.size(-1)
                    if pad_size > 0:
                        doc_rel = torch.cat([doc_rel, torch.zeros(doc_rel.size(0), pad_size, device=doc_rel.device, dtype=doc_rel.dtype)], dim=-1)
                doc_loss = nn.BCEWithLogitsLoss()(doc_logits.float(),doc_rel.float())
                output["loss"]["doc_loss"] = doc_loss.to(sequence_output)
            if sent_labels != None:
                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                s_attn_u = s_attn[idx_used]
                sent_labels_u = sent_labels[idx_used]
                norm_s_labels = sent_labels_u/(sent_labels_u.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn_u = s_attn_u.clone()
                s_attn_u[s_attn_u == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn_u.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            elif teacher_attns != None:
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

        return output
