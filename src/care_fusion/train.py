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
from .engine import apply_profile, macro_f1, predict, set_seed, train_model
from .losses import compute_loss
from .model import CAREFusion


def make_care_loss_fn(counts, cfg):
    tcfg = cfg["train"]
    use_cf = tcfg["lambda2"] > 0
    detach = tcfg.get("cf_detach", False)

    def loss_fn(model, batch):
        out = model(batch)
        out_cf = None
        if use_cf:
            if detach:                      # counterfactual as a fixed target (low VRAM)
                with torch.no_grad():
                    out_cf = model(batch, drop_markers=True)
            else:
                out_cf = model(batch, drop_markers=True)
        return compute_loss(out, batch, counts, cfg, out_cf=out_cf)

    return loss_fn


def run(cfg, device, profile, out_dir: Path):
    cfg, subset = apply_profile(cfg, profile)
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
    if subset:
        train, val, test = train[:subset], val[: max(1, subset // 4)], test[: max(1, subset // 4)]

    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    tcfg = cfg["train"]
    bs = tcfg["batch_size"]
    counts = class_counts(train, C)
    seeds = tcfg["seeds"]

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
            max_epochs=tcfg["max_epochs"],
            patience=tcfg["patience"],
            fp16=tcfg["fp16"],
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
    ap.add_argument("--profile", choices=["full", "smoke", "pilot"], default="full",
                    help="run profile: full (default), smoke (tiny CPU), pilot (full data, few epochs)")
    ap.add_argument("--smoke", action="store_true", help="alias for --profile smoke")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = "smoke" if args.smoke else args.profile
    profile = cfg.get(name, {}) if name != "full" else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} profile={name}")
    run(cfg, device, profile, Path(args.out))


if __name__ == "__main__":
    main()
