# -*- coding: utf-8 -*-
"""Build an OBJECTIVE conflict (sarcasm-prone) subset, ranked by clarity.

Definition (model-free, reproducible):
  - emoji polarity from the train-derived empirical q_j (intensity-weighted over a
    sample's markers): margin = |pos_mass - neg_mass|;
  - the emoji must be CONFIDENT (margin >= MARGIN_MIN, so it clearly leans one way);
  - its polarity must OPPOSE the human gold-label polarity.
Both signals (empirical q_j on train, human gold) are independent of any model we
evaluate. Ranked by margin (clarity); top-N exported for analysis / human check.

    python scripts/build_conflict_set.py --n 400 --margin 0.30
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="target size (300-500)")
    ap.add_argument("--margin", type=float, default=0.30, help="min emoji polarity margin")
    args = ap.parse_args()

    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]
    pol = lambda lab: "POS" if lab in POS else "NEG" if lab in NEG else "NEU"

    cands = []
    for split in ["train", "val", "test"]:
        for rec in (json.loads(l) for l in open(ROOT / f"data/processed/{split}.jsonl", encoding="utf-8")):
            ms = list(dict.fromkeys(rec["marker_seq"]))
            if not ms:
                continue
            v = np.zeros(len(labels)); w = 0.0
            for mk in ms:
                ww = np.log1p(rec["marker_counts"].get(mk, 1))
                v += ww * np.array(qmap.get(mk, gdist)); w += ww
            v /= max(w, 1e-9)
            pos_m, neg_m = float(v[pos_i].sum()), float(v[neg_i].sum())
            emoji_pol = "POS" if pos_m > neg_m else "NEG"
            margin = abs(pos_m - neg_m)
            gold_pol = pol(rec["label"])
            if margin >= args.margin and emoji_pol in {"POS", "NEG"} \
                    and gold_pol in {"POS", "NEG"} and emoji_pol != gold_pol:
                cands.append({
                    "split": split, "id": rec["id"], "margin": round(margin, 4),
                    "emoji_pol": emoji_pol, "gold": rec["label"], "gold_pol": gold_pol,
                    "markers": " ".join(rec["marker_seq"]), "text": rec["text_raw"],
                    "direction": f"emoji_{emoji_pol}->gold_{gold_pol}",
                })

    cands.sort(key=lambda c: -c["margin"])
    sel = cands[: args.n]

    out = ROOT / "data" / "conflict_set.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["split", "id", "margin", "direction",
                                            "emoji_pol", "gold", "gold_pol", "markers", "text"])
        wri.writeheader()
        wri.writerows(sel)

    print(f"Ứng viên conflict (margin>={args.margin}): {len(cands)}; chọn top {len(sel)} -> {out}")
    print(f"Hướng: {dict(Counter(c['direction'] for c in sel))}")
    print(f"Theo split: {dict(Counter(c['split'] for c in sel))}")
    print(f"Nhãn vàng: {dict(Counter(c['gold'] for c in sel))}")
    print(f"Margin: min={sel[-1]['margin']:.3f} max={sel[0]['margin']:.3f} "
          f"median={sel[len(sel)//2]['margin']:.3f}")
    print("\nVí dụ (rõ nhất):")
    for c in sel[:8]:
        print(f"  [{c['margin']:.2f}] {c['direction']} | {c['markers']} -> gold={c['gold']}")
        print(f"      {c['text'][:80]}")


if __name__ == "__main__":
    main()
