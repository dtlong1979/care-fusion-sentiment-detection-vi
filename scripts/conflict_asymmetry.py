"""Test the polarity-asymmetry hypothesis on the conflict cases.

Uses the text-only baseline (B1) prediction as a proxy for the TEXT polarity, the
marker q-table for the EMOJI polarity, and the gold label. Among samples where
text and emoji polarities DISAGREE, we ask: which way does the gold label go, and
is it asymmetric between the two conflict directions?

  Dir A: text POS, emoji NEG  -> does gold follow emoji (NEG) or text (POS)?
  Dir B: text NEG, emoji POS  -> does gold follow text (NEG) or emoji (POS)?

    python scripts/conflict_asymmetry.py
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
PREDS = ROOT / "artifacts/checkpoints_local/preds/B1_text__seed42.npz"
POS, NEG, NEU = {"positive", "interest"}, {"sadness", "anger", "fear"}, {"neutral"}


def pol_of(group):
    return "POS" if group in POS else "NEG" if group in NEG else "NEU"


def main():
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]

    test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl", encoding="utf-8")]
    b1 = np.load(PREDS)
    text_pred = b1["preds"]                      # text-only predicted class (proxy: text polarity)
    assert len(text_pred) == len(test), f"{len(text_pred)} vs {len(test)}"

    def emoji_pol(rec):
        markers = list(dict.fromkeys(rec["marker_seq"]))
        if not markers:
            return None
        v = np.zeros(len(labels)); w = 0.0
        for mk in markers:
            ww = np.log1p(rec["marker_counts"].get(mk, 1))
            v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9)
        return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"

    cells = collections.defaultdict(collections.Counter)   # (text_pol,emoji_pol) -> gold pol counts
    n_disagree = 0
    for rec, tp_idx in zip(test, text_pred):
        ep = emoji_pol(rec)
        if ep is None:
            continue
        tp = pol_of(labels[tp_idx])
        gp = pol_of(rec["label"])
        if tp in {"POS", "NEG"} and ep in {"POS", "NEG"} and tp != ep:
            n_disagree += 1
            cells[(tp, ep)][gp] += 1

    print(f"Samples where TEXT(B1) polarity != EMOJI polarity: {n_disagree}\n")
    print(f"{'text':>5} {'emoji':>6} | {'n':>4} | gold-> {'POS':>5} {'NEG':>5} {'NEU':>5} | nhận xét")
    for (tp, ep), c in sorted(cells.items()):
        n = sum(c.values())
        follow = "EMOJI" if c[ep] > c[tp] else "TEXT" if c[tp] > c[ep] else "tie"
        print(f"{tp:>5} {ep:>6} | {n:4d} | gold-> {c['POS']:5d} {c['NEG']:5d} {c['NEU']:5d} "
              f"| nhãn nghiêng về {follow}")
    print("\nĐọc: 'text POS / emoji NEG' -> nếu gold đa số NEG = emoji áp đảo (đảo nội dung).")
    print("     'text NEG / emoji POS' -> nếu gold đa số NEG = text thắng (emoji là cờ mỉa mai).")


if __name__ == "__main__":
    main()
