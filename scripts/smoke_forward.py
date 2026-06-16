"""Forward/backward smoke test for CARE-Fusion on a tiny real batch (CPU).

Downloads vinai/phobert-base on first run, then checks output shapes and that a
full loss (cls + route + counterfactual) backpropagates. No training.

    python scripts/smoke_forward.py
"""
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from care_fusion.data import CAREDataset, Collator, MarkerVocab, class_counts, load_jsonl
from care_fusion.losses import compute_loss
from care_fusion.model import CAREFusion


def main():
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    C = len(cfg["labels"])

    q_table = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    pmi_graph = json.loads((ROOT / "artifacts/pmi_graph.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q_table)

    records = load_jsonl(ROOT / "data/processed/train.jsonl")[:12]
    ds = CAREDataset(records, vocab, weak_labels=None)

    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate)

    model = CAREFusion(cfg, vocab, pmi_graph, q_table)
    model.train()
    counts = class_counts(records, C)

    batch = next(iter(loader))
    out = model(batch)
    out_cf = model(batch, drop_markers=True)   # counterfactual: markers removed

    print("=== shapes ===")
    print("input_ids      ", tuple(batch["input_ids"].shape))
    print("marker_mask    ", tuple(batch["marker_mask"].shape),
          "n_markers/sample:", batch["marker_mask"].sum(1).tolist())
    print("logits         ", tuple(out["logits"].shape))
    print("p_text         ", tuple(out["p_text"].shape))
    print("router r       ", tuple(out["r"].shape))
    print("delta (JSD)    ", tuple(out["delta"].shape))

    losses = compute_loss(out, batch, counts, cfg, out_cf=out_cf)
    print("=== losses ===")
    for k, v in losses.items():
        print(f"{k:6}: {v.item():.4f}")

    losses["total"].backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    n_params = sum(1 for _ in model.parameters())
    print(f"=== backward OK: {n_grad}/{n_params} param tensors received gradient ===")

    assert out["logits"].shape == (4, C)
    assert out["r"].shape[-1] == 3
    assert torch.isfinite(losses["total"]), "loss is not finite"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
