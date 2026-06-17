# -*- coding: utf-8 -*-
"""Emoji-amplified PhoBERT (simple, no fusion/routing): keep emoji in the text and
repeat each marker K times, then a plain PhoBERT classifier. Tests whether the
simple idea matches/beats CARE-Fusion with a strong backbone.

    python scripts/run_emoji_amp_phobert.py --Ks 0,3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.baselines import TextOnlyModel, make_loss_fn          # noqa: E402
from care_fusion.data import class_counts, load_jsonl                   # noqa: E402
from care_fusion.engine import macro_f1, move, predict, set_seed, train_model  # noqa: E402

POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}
LABELS = ["positive", "sadness", "anger", "fear", "interest", "neutral"]


def amp(rec, K):
    extra = " ".join(" ".join([m] * K) for m in rec["marker_seq"]) if K > 0 else ""
    return (rec["text_segmented"] + " " + extra).strip()


class DS(Dataset):
    def __init__(self, recs, K): self.r, self.K = recs, K
    def __len__(self): return len(self.r)
    def __getitem__(self, i): return {"text": amp(self.r[i], self.K), "label": self.r[i]["label_id"]}


class Coll:
    def __init__(self, tok, ml): self.tok, self.ml = tok, ml
    def __call__(self, b):
        enc = self.tok([x["text"] for x in b], padding=True, truncation=True,
                       max_length=self.ml, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor([x["label"] for x in b])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks", default="0,3")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    C = len(cfg["labels"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = load_jsonl(ROOT / "data/processed/train.jsonl")
    va = load_jsonl(ROOT / "data/processed/val.jsonl")
    te = load_jsonl(ROOT / "data/processed/test.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    coll = Coll(tok, 128)
    counts = class_counts(tr, C)

    # slices
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    qmap, gdist = q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(LABELS) if l in POS]; neg_i = [i for i, l in enumerate(LABELS) if l in NEG]
    pol = lambda lab: "POS" if lab in POS else "NEG" if lab in NEG else "NEU"
    def epol(r):
        v = np.zeros(C); w = 0.0
        for mk in dict.fromkeys(r["marker_seq"]):
            ww = np.log1p(r["marker_counts"].get(mk, 1)); v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9); return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"
    gp = [pol(r["label"]) for r in te]; ep = [epol(r) for r in te]
    cong = np.array([ep[i] == gp[i] and ep[i] in {"POS", "NEG"} for i in range(len(te))])
    conf = np.array([ep[i] != gp[i] and ep[i] in {"POS", "NEG"} and gp[i] in {"POS", "NEG"} for i in range(len(te))])
    accm = lambda y, p, m: float((p[m] == y[m]).mean())

    print(f"{'K':>3} {'test macroF1':>13} {'acc đồng thuận':>15} {'acc mỉa mai':>12}")
    for K in [int(x) for x in args.Ks.split(",")]:
        set_seed(42)
        model = TextOnlyModel(cfg)
        train_model(model, DataLoader(DS(tr, K), batch_size=8, shuffle=True, collate_fn=coll),
                    DataLoader(DS(va, K), batch_size=8, collate_fn=coll),
                    cfg, device, make_loss_fn(counts, cfg), max_epochs=6, patience=2, fp16=True)
        ev = predict(model, DataLoader(DS(te, K), batch_size=8, collate_fn=coll), device)
        y, p = ev["labels"], ev["preds"]
        print(f"{K:>3} {macro_f1(y, p):13.4f} {accm(y, p, cong):15.3f} {accm(y, p, conf):12.3f}")


if __name__ == "__main__":
    main()
