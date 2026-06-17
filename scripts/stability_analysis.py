# -*- coding: utf-8 -*-
"""Objectively test 'is CARE-Fusion more STABLE across cases?'

Two notions of stability, from saved 5-seed predictions:
  (1) seed-stability  = std of overall macro-F1 across seeds (lower = more stable);
  (2) cross-condition balance = accuracy on CONGRUENT vs CONFLICT slices; a balanced
      model has a small gap and a high worst-case (the weaker slice).

    python scripts/stability_analysis.py --preds-dir artifacts/checkpoints_a100/preds
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    args = ap.parse_args()
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]
    pol = lambda lab: "POS" if lab in POS else "NEG" if lab in NEG else "NEU"

    test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl", encoding="utf-8")]

    def epol(rec):
        ms = list(dict.fromkeys(rec["marker_seq"]))
        if not ms:
            return None
        v = np.zeros(len(labels)); w = 0.0
        for mk in ms:
            ww = np.log1p(rec["marker_counts"].get(mk, 1))
            v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9)
        return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"

    gp = [pol(r["label"]) for r in test]
    ep = [epol(r) for r in test]
    congruent = np.array([ep[i] in {"POS", "NEG"} and gp[i] in {"POS", "NEG"} and ep[i] == gp[i]
                          for i in range(len(test))])
    conflict = np.array([ep[i] in {"POS", "NEG"} and gp[i] in {"POS", "NEG"} and ep[i] != gp[i]
                         for i in range(len(test))])
    print(f"congruent={int(congruent.sum())}  conflict={int(conflict.sum())}  "
          f"(của {len(test)} test)\n")

    acc = lambda y, p, m: float((p[m] == y[m]).mean())
    mf1 = lambda y, p: f1_score(y, p, average="macro", zero_division=0)

    per = defaultdict(lambda: {"o": [], "c": [], "k": [], "of1": []})
    for f in sorted(glob.glob(str(Path(ROOT / args.preds_dir) / "*__seed*.npz"))):
        name = re.match(r"(.+)__seed", Path(f).name).group(1)
        d = np.load(f); preds, y = d["preds"], d["labels"]
        per[name]["o"].append(acc(y, preds, np.ones(len(y), bool)))
        per[name]["c"].append(acc(y, preds, congruent))
        per[name]["k"].append(acc(y, preds, conflict))
        per[name]["of1"].append(mf1(y, preds))

    print(f"{'variant':16} {'seed-std(F1)':>12} | {'acc đồng thuận':>15} {'acc mỉa mai':>13} "
          f"{'gap':>7} {'worst':>7}")
    rows = []
    for name in sorted(per):
        f1std = np.std(per[name]["of1"])
        cong = np.mean(per[name]["c"]); conf = np.mean(per[name]["k"])
        gap = cong - conf; worst = min(cong, conf)
        rows.append((name, f1std, cong, conf, gap, worst))
        print(f"{name:16} {f1std:12.4f} | {cong:15.3f} {conf:13.3f} {gap:7.3f} {worst:7.3f}")

    print("\nXếp hạng ỔN ĐỊNH:")
    print("  Ít dao động seed nhất (std F1 nhỏ):",
          ", ".join(f"{n}({s:.3f})" for n, s, *_ in sorted(rows, key=lambda r: r[1])[:3]))
    print("  Cân bằng nhất (gap đồng thuận↔mỉa mai nhỏ):",
          ", ".join(f"{n}({g:.3f})" for n, *_, g, _ in sorted(rows, key=lambda r: abs(r[4]))[:3]))
    print("  Worst-case cao nhất (mỉa mai tốt nhất):",
          ", ".join(f"{n}({w:.3f})" for n, *_, w in sorted(rows, key=lambda r: -r[5])[:3]))


if __name__ == "__main__":
    main()
