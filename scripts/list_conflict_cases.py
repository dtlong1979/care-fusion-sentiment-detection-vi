"""List specific conflict cases for manual label inspection.

Default: text NEG (B1 predicts negative) + emoji POS (marker q leans positive)
but gold = POS. Lets us eyeball whether the gold labels look right.

    python scripts/list_conflict_cases.py [--text NEG --emoji POS --gold POS] [--limit 40]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
PREDS = ROOT / "artifacts/checkpoints_local/preds/B1_text__seed42.npz"
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def pol(g):
    return "POS" if g in POS else "NEG" if g in NEG else "NEU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="NEG")
    ap.add_argument("--emoji", default="POS")
    ap.add_argument("--gold", default="POS")
    ap.add_argument("--limit", type=int, default=80)
    a = ap.parse_args()

    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]

    test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl", encoding="utf-8")]
    b1 = np.load(PREDS)["preds"]

    def emoji_pol(rec):
        ms = list(dict.fromkeys(rec["marker_seq"]))
        if not ms:
            return None
        v = np.zeros(len(labels)); w = 0.0
        for mk in ms:
            ww = np.log1p(rec["marker_counts"].get(mk, 1))
            v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9)
        return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"

    hits = []
    for rec, tp in zip(test, b1):
        ep = emoji_pol(rec)
        if ep is None:
            continue
        if pol(labels[tp]) == a.text and ep == a.emoji and pol(rec["label"]) == a.gold:
            hits.append((rec, labels[tp]))

    print(f"Tiêu chí: text(B1)={a.text}, emoji={a.emoji}, gold={a.gold}  ->  {len(hits)} mẫu\n")
    for i, (rec, b1lab) in enumerate(hits[:a.limit], 1):
        print(f"[{i}] gold={rec['label']:9} | B1_pred={b1lab:9} | markers={rec['marker_seq']}")
        print(f"     {rec['text_raw']}")
    if len(hits) > a.limit:
        print(f"\n... và {len(hits)-a.limit} mẫu nữa.")


if __name__ == "__main__":
    main()
