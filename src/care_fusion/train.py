"""CARE-Fusion training (Protocol Part D2-D4): multi-seed, discriminative LR,
counterfactual regularization, early stop on val macro-F1, best-checkpoint save.

    python -m care_fusion.train --config configs/default.yaml [--smoke] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from .baselines import build_loader
from .data import CAREDataset, Collator, MarkerVocab, class_counts, load_jsonl
from .engine import macro_f1, predict, set_seed, train_model
from .losses import compute_loss
from .model import CAREFusion


def make_care_loss_fn(counts, cfg):
    use_cf = cfg["train"]["lambda2"] > 0

    def loss_fn(model, batch):
        out = model(batch)
        out_cf = model(batch, drop_markers=True) if use_cf else None
        return compute_loss(out, batch, counts, cfg, out_cf=out_cf)

    return loss_fn


def run(cfg, device, profile, out_dir: Path):
    C = len(cfg["labels"])
    art = Path(cfg["paths"]["artifacts_dir"])
    q_table = json.loads((art / "q_table.json").read_text(encoding="utf-8"))
    pmi_graph = json.loads((art / "pmi_graph.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q_table)

    weak_path = art / "weak_labels.json"
    weak = json.loads(weak_path.read_text(encoding="utf-8"))["weak_labels"] if weak_path.exists() else None
    if weak is None:
        print("[train] WARNING: no weak_labels.json -> L_route=0 "
              "(run baselines --emit-oof then resources --steps weak first)")

    pdir = Path(cfg["paths"]["processed_dir"])
    train = load_jsonl(pdir / "train.jsonl")
    val = load_jsonl(pdir / "val.jsonl")
    test = load_jsonl(pdir / "test.jsonl")
    if profile.get("subset"):
        n = profile["subset"]
        train, val, test = train[:n], val[: max(1, n // 4)], test[: max(1, n // 4)]

    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    bs = profile.get("batch_size", cfg["train"]["batch_size"])
    counts = class_counts(train, C)
    seeds = profile.get("seeds", cfg["train"]["seeds"])

    test_f1s = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"=== seed {seed} ===")
        set_seed(seed)
        model = CAREFusion(cfg, vocab, pmi_graph, q_table)
        res = train_model(
            model,
            build_loader(train, vocab, collate, bs, True, weak=weak),
            build_loader(val, vocab, collate, bs, False, weak=weak),
            cfg, device, make_care_loss_fn(counts, cfg),
            max_epochs=profile.get("max_epochs", cfg["train"]["max_epochs"]),
            patience=cfg["train"]["patience"],
            fp16=profile.get("fp16", cfg["train"]["fp16"]),
        )
        ev = predict(model, build_loader(test, vocab, collate, bs, False), device)
        tf1 = macro_f1(ev["labels"], ev["preds"])
        test_f1s.append(tf1)
        ckpt = out_dir / f"care_fusion_seed{seed}.pt"
        torch.save({"state_dict": res["state"], "cfg": cfg, "seed": seed,
                    "val_f1": res["best_f1"], "test_f1": tf1}, ckpt)
        print(f"  seed {seed}: val={res['best_f1']:.4f} test={tf1:.4f} -> {ckpt.name}")

    arr = np.array(test_f1s)
    print(f"\n=== CARE-Fusion test macro-F1 over {len(seeds)} seed(s): "
          f"{arr.mean():.4f} ± {arr.std():.4f} ===")


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description="Train CARE-Fusion (Part D)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/checkpoints")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed/1-epoch run on CPU")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    profile = cfg.get("smoke", {}) if args.smoke else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} smoke={args.smoke}")
    run(cfg, device, profile, Path(args.out))


if __name__ == "__main__":
    main()
