"""Baselines (Protocol Part E) and the out-of-fold p_text generator for B2.

Implemented here:
  B0  majority class                              (lower bound)
  B1  PhoBERT text-only                           (marker contribution)
  --emit-oof : 5-fold OOF p_text from B1 -> artifacts/p_text_oof.json (feeds B2)

B2 (marker concat), B3 (scalar gated), B4 (cross-attn no routing) are variant
fusions; they reuse this training harness and are added alongside the CARE-Fusion
ablations.

Usage:
    python -m care_fusion.baselines --config configs/default.yaml --model B0
    python -m care_fusion.baselines --config configs/default.yaml --model B1
    python -m care_fusion.baselines --config configs/default.yaml --emit-oof
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import CAREDataset, Collator, MarkerVocab, class_counts, load_jsonl
from .engine import macro_f1, predict, set_seed, train_model
from .losses import class_balanced_focal_loss
from .model import TextEncoder


class TextOnlyModel(nn.Module):
    """B1: PhoBERT + attention pool + linear classifier (ignores markers)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.enc = TextEncoder(cfg["preprocess"]["phobert_name"], len(cfg["labels"]))

    def forward(self, batch):
        _, _, logits = self.enc(batch["input_ids"], batch["attention_mask"])
        return {"logits": logits}


# --------------------------------------------------------------------------- #
def build_loader(records, vocab, collate, batch_size, shuffle, weak=None):
    ds = CAREDataset(records, vocab, weak_labels=weak)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def make_loss_fn(class_counts_t, cfg):
    def loss_fn(model, batch):
        out = model(batch)
        l = class_balanced_focal_loss(out["logits"], batch["labels"], class_counts_t,
                                      beta=cfg["train"]["cb_beta"], gamma=cfg["train"]["focal_gamma"])
        return {"total": l}
    return loss_fn


def _setup(cfg, profile):
    C = len(cfg["labels"])
    pdir = cfg["paths"]["processed_dir"]
    art = Path(cfg["paths"]["artifacts_dir"]); art.mkdir(parents=True, exist_ok=True)
    q_table = json.loads((art / "q_table.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q_table)
    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    train = load_jsonl(Path(pdir) / "train.jsonl")
    val = load_jsonl(Path(pdir) / "val.jsonl")
    test = load_jsonl(Path(pdir) / "test.jsonl")
    if profile.get("subset"):
        n = profile["subset"]
        train, val, test = train[:n], val[: n // 4 or 1], test[: n // 4 or 1]
    return C, vocab, collate, train, val, test


def run_majority(cfg, train, val, test):
    maj = Counter(r["label_id"] for r in train).most_common(1)[0][0]
    for name, split in [("val", val), ("test", test)]:
        labels = [r["label_id"] for r in split]
        preds = [maj] * len(labels)
        print(f"[B0] {name} macro-F1 = {macro_f1(labels, preds):.4f} "
              f"(majority class = {cfg['labels'][maj]})")


def run_b1(cfg, device, profile):
    C, vocab, collate, train, val, test = _setup(cfg, profile)
    bs = profile.get("batch_size", cfg["train"]["batch_size"])
    counts = class_counts(train, C)
    set_seed(cfg["train"]["seeds"][0])
    model = TextOnlyModel(cfg)
    res = train_model(
        model,
        build_loader(train, vocab, collate, bs, True),
        build_loader(val, vocab, collate, bs, False),
        cfg, device, make_loss_fn(counts, cfg),
        max_epochs=profile.get("max_epochs", cfg["train"]["max_epochs"]),
        patience=cfg["train"]["patience"],
        fp16=profile.get("fp16", cfg["train"]["fp16"]),
    )
    ev = predict(model, build_loader(test, vocab, collate, bs, False), device)
    print(f"[B1] best val macro-F1={res['best_f1']:.4f} | test macro-F1={macro_f1(ev['labels'], ev['preds']):.4f}")


def emit_oof(cfg, device, profile):
    """5-fold OOF: train B1 on K-1 folds, predict the held-out fold -> p_text."""
    C, vocab, collate, train, val, test = _setup(cfg, profile)
    bs = profile.get("batch_size", cfg["train"]["batch_size"])
    n_folds = profile.get("oof_folds", cfg["resources"]["oof_folds"])
    y = np.array([r["label_id"] for r in train])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["train"]["seeds"][0])

    p_text: Dict[str, list] = {}
    for fold, (tr_idx, ho_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f"[OOF] fold {fold}/{n_folds}: train={len(tr_idx)} holdout={len(ho_idx)}")
        set_seed(cfg["train"]["seeds"][0] + fold)
        tr = [train[i] for i in tr_idx]
        ho = [train[i] for i in ho_idx]
        counts = class_counts(tr, C)
        model = TextOnlyModel(cfg)
        train_model(
            model,
            build_loader(tr, vocab, collate, bs, True),
            build_loader(ho, vocab, collate, bs, False),
            cfg, device, make_loss_fn(counts, cfg),
            max_epochs=profile.get("max_epochs", cfg["train"]["max_epochs"]),
            patience=cfg["train"]["patience"],
            fp16=profile.get("fp16", cfg["train"]["fp16"]),
        )
        ev = predict(model, build_loader(ho, vocab, collate, bs, False), device)
        for r, prob in zip(ho, ev["probs"]):
            p_text[str(r["id"])] = [float(x) for x in prob]

    out = Path(cfg["paths"]["artifacts_dir"]) / "p_text_oof.json"
    out.write_text(json.dumps(p_text), encoding="utf-8")
    print(f"[OOF] wrote {len(p_text)} p_text vectors -> {out}")


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description="Baselines + OOF p_text (Part E / B2)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["B0", "B1"], default=None)
    ap.add_argument("--emit-oof", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use the tiny smoke profile")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    profile = cfg.get("smoke", {}) if args.smoke else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baselines] device={device} smoke={args.smoke}")

    if args.model == "B0":
        _, _, _, train, val, test = _setup(cfg, profile)
        run_majority(cfg, train, val, test)
    elif args.model == "B1":
        run_b1(cfg, device, profile)
    if args.emit_oof:
        emit_oof(cfg, device, profile)


if __name__ == "__main__":
    main()
