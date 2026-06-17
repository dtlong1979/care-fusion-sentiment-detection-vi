"""Aggregate multi-seed predictions: overall vs CONFLICT-slice macro-F1 (mean+/-std)
per variant, plus Wilcoxon CARE_full vs others. Computed from saved preds npz only
(the conflict slice = marker polarity opposes gold polarity), no checkpoints needed.

    python scripts/slice_summary.py --preds-dir artifacts/checkpoints_5seed/preds
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    q = json.loads((ROOT / cfg["paths"]["artifacts_dir"] / "q_table.json").read_text(encoding="utf-8"))
    labels, qmap, gdist = q["labels"], q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(labels) if l in POS]
    neg_i = [i for i, l in enumerate(labels) if l in NEG]
    pol = lambda i: "POS" if labels[i] in POS else "NEG" if labels[i] in NEG else "NEU"

    test = [json.loads(l) for l in open(ROOT / cfg["paths"]["processed_dir"] / "test.jsonl", encoding="utf-8")]

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

    gold = np.array([r["label_id"] for r in test])
    conflict = np.array([
        emoji_pol(r) is not None and emoji_pol(r) in {"POS", "NEG"}
        and pol(r["label_id"]) in {"POS", "NEG"} and emoji_pol(r) != pol(r["label_id"])
        for r in test
    ])
    print(f"Conflict slice (sarcasm-prone): {int(conflict.sum())}/{len(test)} test samples\n")

    pdir = Path(args.preds_dir)
    if not pdir.is_absolute():
        pdir = ROOT / pdir
    macro = lambda y, p: f1_score(y, p, average="macro", zero_division=0)

    per_variant = defaultdict(lambda: {"overall": [], "slice": []})
    for f in sorted(glob.glob(str(pdir / "*__seed*.npz"))):
        name = re.match(r"(.+)__seed", Path(f).name).group(1)
        d = np.load(f)
        preds, y = d["preds"], d["labels"]
        per_variant[name]["overall"].append(macro(y, preds))
        per_variant[name]["slice"].append(macro(y[conflict], preds[conflict]))

    print(f"{'variant':18} {'overall (mean±std)':>22} {'sarcasm-slice (mean±std)':>26} {'seeds':>6}")
    for name in sorted(per_variant):
        o = np.array(per_variant[name]["overall"]); s = np.array(per_variant[name]["slice"])
        print(f"{name:18} {o.mean():.4f} ± {o.std():.4f}        "
              f"{s.mean():.4f} ± {s.std():.4f}      {len(o):4d}")

    # Wilcoxon CARE_full vs each other (paired across seeds), on overall + slice
    if "CARE_full" in per_variant:
        print("\nWilcoxon (CARE_full vs X), paired across seeds:")
        a_o = per_variant["CARE_full"]["overall"]; a_s = per_variant["CARE_full"]["slice"]
        for name in sorted(per_variant):
            if name == "CARE_full":
                continue
            b_o, b_s = per_variant[name]["overall"], per_variant[name]["slice"]
            if len(a_o) == len(b_o) and len(a_o) >= 2:
                try:
                    po = stats.wilcoxon(a_o, b_o).pvalue
                    ps = stats.wilcoxon(a_s, b_s).pvalue
                    print(f"  vs {name:16}: overall p={po:.3f} | sarcasm-slice p={ps:.3f}")
                except ValueError as e:
                    print(f"  vs {name:16}: {e}")


if __name__ == "__main__":
    main()
