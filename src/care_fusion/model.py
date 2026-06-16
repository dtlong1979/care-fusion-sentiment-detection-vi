"""CARE-Fusion model (Protocol Part C).

Forward pass:
  1. H = PhoBERT(text); p_text = softmax(W_t . attn_pool(H))
  2. per marker j: e_j = W_e[l_j || c_j]; delta_j = JSD(p_text || q_j)
  3. g_j = LayerNorm(e_j + GatedCrossAttention(e_j, H))
  4. r_j = softmax(W_r[g_j || e_j || delta_j || c_j])  in Delta^2
  5. z_j = r_j^rho F_rho + r_j^kappa F_kappa + r_j^chi F_chi ; z = sum_j c_j z_j / sum_j c_j
  6. z_tilde = GCN_enrich(z, A); y_hat = softmax(W_o z_tilde + b_o)

Where the protocol/Methods leave a detail open (the exact fusion operators and
how the PMI graph enriches z), we pick a concrete, documented choice and keep it
configurable so ablations can toggle each component.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

REGIME_NAMES = ["redundancy", "complementarity", "conflict"]


def jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Jensen-Shannon divergence along the last dim. p,q broadcastable -> [...]."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(-1, keepdim=True)
    q = q / q.sum(-1, keepdim=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class AttentionPool(nn.Module):
    """Mask-aware additive attention pooling over the token sequence."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(H).squeeze(-1)                       # [B, T]
        s = s.masked_fill(mask == 0, float("-inf"))
        a = s.softmax(-1).unsqueeze(-1)                      # [B, T, 1]
        return (a * H).sum(1)                                # [B, d]


class TextEncoder(nn.Module):
    def __init__(self, name: str, num_classes: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        d_t = self.bert.config.hidden_size
        self.pool = AttentionPool(d_t)
        self.cls = nn.Linear(d_t, num_classes)
        self.d_t = d_t

    def forward(self, input_ids, attention_mask):
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.pool(H, attention_mask)
        logits = self.cls(pooled)                            # text-only logits
        return H, pooled, logits


class MarkerEncoder(nn.Module):
    """e_j = W_e[l_j || c_j], where l_j = proj(q_j) or a learned embedding (low-freq)."""

    def __init__(self, vocab_size: int, num_classes: int, d_e: int):
        super().__init__()
        self.q_proj = nn.Linear(num_classes, d_e)
        self.emb = nn.Embedding(vocab_size, d_e, padding_idx=0)
        self.W_e = nn.Sequential(nn.Linear(d_e + 1, d_e), nn.GELU(), nn.LayerNorm(d_e))

    def forward(self, marker_ids, marker_q, marker_c, marker_lowfreq):
        l_q = self.q_proj(marker_q)                          # [B, M, d_e]
        l_emb = self.emb(marker_ids)                         # [B, M, d_e]
        lf = marker_lowfreq.unsqueeze(-1).float()
        l = lf * l_emb + (1 - lf) * l_q                      # low-freq -> learned branch
        e = self.W_e(torch.cat([l, marker_c.unsqueeze(-1)], dim=-1))
        return e


class GatedCrossAttention(nn.Module):
    def __init__(self, d_e: int, d_t: int, d: int, heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_e, d)
        self.kv_proj = nn.Linear(d_t, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, d_e)
        self.gate = nn.Linear(2 * d_e, d_e)
        self.norm = nn.LayerNorm(d_e)

    def forward(self, e, H, attn_mask):
        q = self.q_proj(e)                                   # [B, M, d]
        kv = self.kv_proj(H)                                 # [B, T, d]
        key_padding = attn_mask == 0                         # True = ignore
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=key_padding)
        ctx = self.out(ctx)                                  # [B, M, d_e]
        gate = torch.sigmoid(self.gate(torch.cat([e, ctx], dim=-1)))
        return self.norm(e + gate * ctx)


class RegimeFusion(nn.Module):
    """Three regime-specific fusion operators, mixed by the router weights."""

    def __init__(self, d_e: int, d_t: int, d: int):
        super().__init__()
        def op():
            return nn.Sequential(nn.Linear(d_e + d_t, d), nn.GELU(), nn.Linear(d, d))
        self.F_rho = op()    # redundancy
        self.F_kappa = op()  # complementarity
        self.F_chi = op()    # conflict

    def forward(self, g, pooled_text, r):
        # g: [B,M,d_e]; pooled_text: [B,d_t] -> broadcast to markers
        B, M, _ = g.shape
        ctx = pooled_text.unsqueeze(1).expand(B, M, -1)
        x = torch.cat([g, ctx], dim=-1)
        f = torch.stack([self.F_rho(x), self.F_kappa(x), self.F_chi(x)], dim=-2)  # [B,M,3,d]
        z_j = (r.unsqueeze(-1) * f).sum(-2)                  # [B, M, d]
        return z_j


class GCNEnrich(nn.Module):
    """2-layer GCN over the static PPMI graph; enriches z with the sample's marker
    node embeddings. Node init: marker -> proj(q_m); emotion group -> learned."""

    def __init__(self, pmi_graph: dict, q_table: dict, marker2id: dict,
                 num_classes: int, d: int):
        super().__init__()
        nodes = pmi_graph["nodes"]
        types = pmi_graph["node_types"]
        N = len(nodes)
        self.N = N

        # normalized adjacency  hat A = D^-1/2 (A + I) D^-1/2
        A = torch.eye(N)
        for i, j, w in pmi_graph["edges"]:
            A[i, j] += w
            A[j, i] += w
        deg = A.sum(1)
        dinv = deg.clamp_min(1e-8).pow(-0.5)
        self.register_buffer("A_hat", dinv.unsqueeze(1) * A * dinv.unsqueeze(0))

        # node init features
        q = q_table["q"]
        global_dist = q_table["global_dist"]
        q_feat = torch.zeros(N, num_classes)
        is_marker = torch.zeros(N, dtype=torch.bool)
        emo_idx = torch.full((N,), -1, dtype=torch.long)
        node_marker_str = []
        for n, (name, t) in enumerate(zip(nodes, types)):
            if t == "marker":
                is_marker[n] = True
                q_feat[n] = torch.tensor(q.get(name, global_dist))
                node_marker_str.append((name, n))
            else:
                emo = name.replace("__EMO__", "")
                emo_idx[n] = q_table["labels"].index(emo)
        self.register_buffer("q_feat", q_feat)
        self.register_buffer("is_marker", is_marker)
        self.register_buffer("emo_idx", emo_idx)

        self.q_proj = nn.Linear(num_classes, d)
        self.emo_emb = nn.Embedding(num_classes, d)
        self.gcn1 = nn.Linear(d, d)
        self.gcn2 = nn.Linear(d, d)

        # marker-vocab-id -> graph-node-index (for gathering per sample); 0 -> -1
        name2node = {name: n for name, n in node_marker_str}
        id2node = torch.full((max(marker2id.values()) + 1,), -1, dtype=torch.long)
        for m, mid in marker2id.items():
            if m in name2node:
                id2node[mid] = name2node[m]
        self.register_buffer("id2node", id2node)
        self.enrich = nn.Linear(d, d)

    def node_embeddings(self) -> torch.Tensor:
        # Blend marker- and emotion-node init without in-place ops so it stays
        # autocast/autograd-safe (Embedding output is fp32, q_proj may be fp16).
        x_marker = self.q_proj(self.q_feat)                  # [N, d]
        x_emo = self.emo_emb(self.emo_idx.clamp_min(0))      # [N, d]
        is_m = self.is_marker.unsqueeze(-1).to(x_marker.dtype)
        x = is_m * x_marker + (1 - is_m) * x_emo.to(x_marker.dtype)
        h = F.gelu(self.A_hat @ self.gcn1(x))
        h = self.A_hat @ self.gcn2(h)                        # [N, d]
        return h

    def forward(self, z, marker_ids, marker_mask):
        node_emb = self.node_embeddings()                    # [N, d]
        nodes = self.id2node[marker_ids]                     # [B, M] (-1 if none)
        valid = (nodes >= 0).float() * marker_mask           # [B, M]
        gathered = node_emb[nodes.clamp_min(0)]              # [B, M, d]
        denom = valid.sum(1, keepdim=True).clamp_min(1.0)
        pooled = (gathered * valid.unsqueeze(-1)).sum(1) / denom   # [B, d]
        return z + self.enrich(pooled)


def _marker_summary(marker_q, marker_mask):
    """Unweighted mean of marker q-vectors per sample -> [B, C] (zeros if none)."""
    m = marker_mask.unsqueeze(-1)
    denom = marker_mask.sum(1, keepdim=True).clamp_min(1e-8)
    return (marker_q * m).sum(1) / denom


class MarkerConcatModel(nn.Module):
    """B2: PhoBERT pooled text concatenated with an unweighted marker-q summary."""

    def __init__(self, cfg: dict):
        super().__init__()
        C = len(cfg["labels"])
        self.text = TextEncoder(cfg["preprocess"]["phobert_name"], C)
        self.cls = nn.Sequential(nn.Linear(self.text.d_t + C, 256), nn.GELU(), nn.Linear(256, C))

    def forward(self, batch, drop_markers: bool = False):
        H, pooled, text_logits = self.text(batch["input_ids"], batch["attention_mask"])
        mask = torch.zeros_like(batch["marker_mask"]) if drop_markers else batch["marker_mask"]
        summary = _marker_summary(batch["marker_q"], mask)
        logits = self.cls(torch.cat([pooled, summary], dim=-1))
        return {"logits": logits, "p_text": text_logits.softmax(-1), "marker_mask": mask}


class ScalarGatedModel(nn.Module):
    """B3: scalar gated fusion z = a * text + (1-a) * marker, a a per-sample scalar."""

    def __init__(self, cfg: dict):
        super().__init__()
        C = len(cfg["labels"])
        d = cfg["model"]["d"]
        self.text = TextEncoder(cfg["preprocess"]["phobert_name"], C)
        self.t_proj = nn.Linear(self.text.d_t, d)
        self.m_proj = nn.Linear(C, d)
        self.gate = nn.Linear(self.text.d_t + C, 1)
        self.cls = nn.Linear(d, C)

    def forward(self, batch, drop_markers: bool = False):
        H, pooled, text_logits = self.text(batch["input_ids"], batch["attention_mask"])
        mask = torch.zeros_like(batch["marker_mask"]) if drop_markers else batch["marker_mask"]
        summary = _marker_summary(batch["marker_q"], mask)
        a = torch.sigmoid(self.gate(torch.cat([pooled, summary], dim=-1)))   # [B,1] scalar
        z = a * self.t_proj(pooled) + (1 - a) * self.m_proj(summary)
        return {"logits": self.cls(z), "p_text": text_logits.softmax(-1), "marker_mask": mask}


class CAREFusion(nn.Module):
    """Full model, with ablation switches (Part E):
      use_routing  : 3-way regime fusion (False -> single operator = baseline B4)
      use_delta    : feed congruence delta_j to the router
      use_intensity: weight markers by c_j = log(1+count) (False -> uniform)
      use_gcn      : enrich z with the PPMI-graph GCN
    """

    def __init__(self, cfg: dict, marker_vocab, pmi_graph: dict, q_table: dict,
                 use_routing: bool = True, use_delta: bool = True,
                 use_intensity: bool = True, use_gcn: bool = True):
        super().__init__()
        m = cfg["model"]
        C = len(cfg["labels"])
        self.C = C
        d_e, d = m["d_e"], m["d"]
        self.use_routing = use_routing
        self.use_delta = use_delta
        self.use_intensity = use_intensity
        self.use_gcn = use_gcn

        self.text = TextEncoder(cfg["preprocess"]["phobert_name"], C)
        d_t = self.text.d_t
        self.markers = MarkerEncoder(marker_vocab.size, C, d_e)
        self.cross = GatedCrossAttention(d_e, d_t, d, m["cross_attn_heads"])
        self.router = nn.Linear(2 * d_e + 2, 3)
        self.fusion = RegimeFusion(d_e, d_t, d)
        # single fusion operator used when routing is ablated
        self.fusion_single = nn.Sequential(nn.Linear(d_e + d_t, d), nn.GELU(), nn.Linear(d, d))
        self.z_empty = nn.Parameter(torch.zeros(d))          # z_emptyset (no markers)
        self.gcn = GCNEnrich(pmi_graph, q_table, marker_vocab.marker2id, C, d)
        self.classifier = nn.Linear(d, C)

    def forward(self, batch: Dict[str, torch.Tensor], drop_markers: bool = False):
        H, pooled, text_logits = self.text(batch["input_ids"], batch["attention_mask"])
        p_text = text_logits.softmax(-1)                     # [B, C]

        mask = batch["marker_mask"]                          # [B, M]
        if drop_markers:
            mask = torch.zeros_like(mask)                    # counterfactual: no markers

        # intensity: real c_j, or 1 for every present marker when ablated
        c_feat = batch["marker_c"] if self.use_intensity else mask.clone()

        e = self.markers(batch["marker_ids"], batch["marker_q"],
                         c_feat, batch["marker_lowfreq"])
        g = self.cross(e, H, batch["attention_mask"])

        delta = jsd(p_text.unsqueeze(1), batch["marker_q"])  # [B, M]
        delta_in = delta if self.use_delta else torch.zeros_like(delta)
        router_in = torch.cat([g, e, delta_in.unsqueeze(-1), c_feat.unsqueeze(-1)], dim=-1)
        r_logits = self.router(router_in)                    # [B, M, 3]
        r = r_logits.softmax(-1)

        if self.use_routing:
            z_j = self.fusion(g, pooled, r)                  # [B, M, d]
        else:
            ctx = pooled.unsqueeze(1).expand(-1, g.shape[1], -1)
            z_j = self.fusion_single(torch.cat([g, ctx], dim=-1))
            r_logits = None                                  # no routing supervision

        c = c_feat * mask                                    # fusion weights
        denom = c.sum(1, keepdim=True)                       # [B, 1]
        z = (c.unsqueeze(-1) * z_j).sum(1) / denom.clamp_min(1e-8)
        has_marker = (denom > 0).float()                     # [B, 1]
        z = has_marker * z + (1 - has_marker) * self.z_empty  # z_emptyset fallback

        if self.use_gcn:
            z = self.gcn(z, batch["marker_ids"], mask)

        logits = self.classifier(z)
        return {
            "logits": logits,
            "p_text": p_text,
            "text_logits": text_logits,
            "r": r,
            "r_logits": r_logits,
            "delta": delta,
            "marker_mask": mask,
        }
