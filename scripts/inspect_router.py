"""Diagnostic: is the regime router actually routing, or has it collapsed onto a
single fusion operator (which would make CARE-Fusion ~= the no-routing B4)?

Loads a CARE_full checkpoint, runs the test set, and reports the distribution of
router weights r_j (redundancy/complementarity/conflict), their entropy, and how
the chosen regime relates to the congruence delta_j.

    python scripts/inspect_router.py --care-ckpt artifacts/checkpoints_local/CARE_full_seed42.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
import sys; sys.path.insert(0, str(ROOT / "src"))
from care_fusion.data import CAREDataset, Collator, MarkerVocab, load_jsonl  # noqa: E402
from care_fusion.model import CAREFusion                                      # noqa: E402

REGIMES = ["redundancy", "complementarity", "conflict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--care-ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    C = len(cfg["labels"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    pmi = json.loads((ROOT / "artifacts/pmi_graph.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q)

    ckpt = torch.load(ROOT / args.care_ckpt, map_location=device, weights_only=False)
    model = CAREFusion(ckpt["cfg"], vocab, pmi, q, **ckpt.get("flags", {}))
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()

    test = load_jsonl(ROOT / "data/processed/test.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    loader = DataLoader(CAREDataset(test, vocab), batch_size=32, shuffle=False, collate_fn=collate)

    R, D = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b)
            m = b["marker_mask"].bool()
            R.append(out["r"][m].cpu().numpy())          # [n_markers, 3]
            D.append(out["delta"][m].cpu().numpy())       # [n_markers]
    R = np.concatenate(R); D = np.concatenate(D)
    n = len(R)

    argmax = R.argmax(1)
    print(f"Markers analyzed: {n}")
    print("\n--- Router argmax distribution (which operator wins) ---")
    for i, name in enumerate(REGIMES):
        print(f"  {name:16}: {100*(argmax==i).mean():5.1f}%   mean weight={R[:,i].mean():.3f}")
    ent = -(R * np.log(R + 1e-9)).sum(1)
    print(f"\n  mean max-weight = {R.max(1).mean():.3f}  (1.0 = hard pick, ~0.33 = uniform)")
    print(f"  mean entropy    = {ent.mean():.3f}  (0 = collapsed, {np.log(3):.3f} = uniform)")
    print("\n--- Mean congruence delta_j by chosen operator (sanity: conflict-op should "
          "get higher-delta markers) ---")
    for i, name in enumerate(REGIMES):
        sel = argmax == i
        if sel.any():
            print(f"  {name:16}: mean delta={D[sel].mean():.3f}  (n={int(sel.sum())})")
    print(f"\n  overall delta: mean={D.mean():.3f} min={D.min():.3f} max={D.max():.3f}")
    if R.max(1).mean() > 0.9 or ent.mean() < 0.2:
        print("\n  >>> WARNING: router looks COLLAPSED -> CARE ≈ single-operator (B4). "
              "Routing not differentiating. <<<")
    else:
        print("\n  >>> Router is differentiating across operators. <<<")


if __name__ == "__main__":
    main()
