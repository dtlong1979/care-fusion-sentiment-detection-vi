# -*- coding: utf-8 -*-
"""Evaluate already-saved model predictions on the objective SARCASM test set.

Sarcasm test set (model-free, no leakage — test split only):
  emoji polarity POSITIVE (q_j weighted, margin >= M) AND gold polarity NEGATIVE.
For each variant we report, across seeds (mean±std):
  - accuracy: exact emotion correct;
  - polarity-accuracy: predicted a NEGATIVE class (not fooled by the positive emoji)
    = sarcasm robustness.

    python scripts/eval_conflict_set.py --preds-dir artifacts/checkpoints_a100/preds --margin 0.2
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--margin", type=float, default=0.2)
    args = ap.parse_args()

    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]
    neg_set = set(neg_i)
    pol = lambda lab: "POS" if lab in POS else "NEG" if lab in NEG else "NEU"

    test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl", encoding="utf-8")]

    def emoji_posmargin(rec):
        ms = list(dict.fromkeys(rec["marker_seq"]))
        if not ms:
            return None
        v = np.zeros(len(labels)); w = 0.0
        for mk in ms:
            ww = np.log1p(rec["marker_counts"].get(mk, 1))
            v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9)
        return float(v[pos_i].sum() - v[neg_i].sum())   # >0 => emoji positive

    # sarcasm mask: emoji clearly positive, gold negative
    mask = np.array([
        (emoji_posmargin(r) is not None and emoji_posmargin(r) >= args.margin
         and pol(r["label"]) == "NEG")
        for r in test
    ])
    idx = np.where(mask)[0]
    print(f"Sarcasm test set (emoji POS margin>={args.margin}, gold NEG): {len(idx)} mẫu\n")

    pdir = Path(args.preds_dir)
    if not pdir.is_absolute():
        pdir = ROOT / pdir
    per = defaultdict(lambda: {"acc": [], "pol": []})
    for f in sorted(glob.glob(str(pdir / "*__seed*.npz"))):
        name = re.match(r"(.+)__seed", Path(f).name).group(1)
        d = np.load(f)
        preds, y = d["preds"][idx], d["labels"][idx]
        per[name]["acc"].append(float((preds == y).mean()))
        per[name]["pol"].append(float(np.isin(preds, list(neg_set)).mean()))

    print(f"{'variant':18} {'accuracy':>16} {'polarity-acc (kháng mỉa mai)':>30} {'seeds':>6}")
    for name in sorted(per):
        a = np.array(per[name]["acc"]); p = np.array(per[name]["pol"])
        print(f"{name:18} {a.mean():.3f} ± {a.std():.3f}     "
              f"{p.mean():.3f} ± {p.std():.3f}            {len(a):4d}")


if __name__ == "__main__":
    main()
