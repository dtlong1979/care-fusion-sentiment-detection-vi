"""Audit two hypotheses about the derived dataset, model-free:

H1 (conflict scarcity): how many samples have the MARKER-implied polarity
   disagree with the GOLD-label polarity (i.e. sarcasm-/conflict-prone)?
H2 (mapping/priority label noise): how many samples are "polarity-overridden" by
   the priority rule, e.g. a POSITIVE-majority label set ends up labeled NEGATIVE?

    python scripts/data_audit.py
"""
import ast
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
ORIG = Path(r"C:\Users\SingPC\Downloads")
POS, NEG, NEU = {"positive", "interest"}, {"sadness", "anger", "fear"}, {"neutral"}
PRIORITY = ["anger", "sadness", "fear", "positive", "interest", "neutral"]


def polarity(group):
    return "POS" if group in POS else "NEG" if group in NEG else "NEU"


def main():
    der = pd.read_csv(ROOT / "data/raw/vigoemotions_text_emoji_6groups.csv", encoding="utf-8-sig")
    der.columns = [c.strip() for c in der.columns]
    frames = []
    for s in ["train", "val", "test"]:
        d = pd.read_csv(ORIG / f"{s}.csv", encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]; d["split"] = s
        frames.append(d)
    orig = pd.concat(frames, ignore_index=True)
    orig["labels"] = orig["labels"].apply(ast.literal_eval)
    m = der.merge(orig[["id", "split", "labels"]],
                  left_on=["source_id", "source_split"], right_on=["id", "split"], how="left")

    # recover idx -> group
    idx2group = {}
    for _, r in m.iterrows():
        if isinstance(r["labels"], list) and len(r["labels"]) == 1:
            idx2group[r["labels"][0]] = r["emotion_group"]

    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels6 = q["labels"]

    # ---- H2: priority-rule polarity override ----
    n = len(m)
    flip_pos2neg = flip_neg2pos = multipolar = 0
    for _, r in m.iterrows():
        groups = [idx2group[i] for i in r["labels"] if i in idx2group]
        pols = [polarity(g) for g in groups]
        cnt = collections.Counter(pols)
        final_pol = polarity(r["emotion_group"])
        if cnt["POS"] and cnt["NEG"]:
            multipolar += 1
        # final polarity is a strict minority vs the other pole
        if final_pol == "NEG" and cnt["POS"] > cnt["NEG"]:
            flip_pos2neg += 1
        if final_pol == "POS" and cnt["NEG"] > cnt["POS"]:
            flip_neg2pos += 1

    print("=== H2: mapping/priority label noise ===")
    print(f"  multipolar (POS & NEG present): {multipolar}/{n} = {100*multipolar/n:.1f}%")
    print(f"  POSITIVE-majority -> labeled NEGATIVE: {flip_pos2neg} ({100*flip_pos2neg/n:.1f}%)")
    print(f"  NEGATIVE-majority -> labeled POSITIVE: {flip_neg2pos} ({100*flip_neg2pos/n:.1f}%)")
    print(f"  total polarity-overridden: {flip_pos2neg+flip_neg2pos} "
          f"({100*(flip_pos2neg+flip_neg2pos)/n:.1f}%)")

    # ---- H1: marker-implied vs gold polarity (conflict prevalence) ----
    qmap = q["q"]; gdist = q["global_dist"]
    lab2pol_vec = np.array([1 if l in POS else -1 if l in NEG else 0 for l in labels6])
    by_split = collections.defaultdict(lambda: [0, 0])  # split -> [conflict, total_with_marker]
    examples = []
    for split in ["train", "val", "test"]:
        recs = [json.loads(l) for l in open(ROOT / f"data/processed/{split}.jsonl", encoding="utf-8")]
        for rec in recs:
            markers = list(dict.fromkeys(rec["marker_seq"]))
            if not markers:
                continue
            vec = np.zeros(len(labels6))
            wsum = 0.0
            for mk in markers:
                w = np.log1p(rec["marker_counts"].get(mk, 1))
                vec += w * np.array(qmap.get(mk, gdist)); wsum += w
            vec /= max(wsum, 1e-9)
            marker_pol = "POS" if (vec * (lab2pol_vec == 1)).sum() > (vec * (lab2pol_vec == -1)).sum() else "NEG"
            gold_pol = polarity(rec["label"])
            by_split[split][1] += 1
            if {marker_pol, gold_pol} == {"POS", "NEG"}:   # strict opposite-polarity
                by_split[split][0] += 1
                if len(examples) < 6 and split == "train":
                    examples.append((rec["text_raw"][:60], rec["marker_seq"], rec["label"]))

    print("\n=== H1: conflict prevalence (marker polarity vs gold polarity) ===")
    tot_c = tot_n = 0
    for s in ["train", "val", "test"]:
        c, t = by_split[s]; tot_c += c; tot_n += t
        print(f"  {s:5}: {c}/{t} conflict ({100*c/max(t,1):.1f}% of marker-bearing)")
    print(f"  TOTAL: {tot_c}/{tot_n} = {100*tot_c/max(tot_n,1):.1f}% strict POS<->NEG conflict")
    print("\n  ví dụ conflict (text | markers | nhãn):")
    for tx, mk, lb in examples:
        print(f"   - {tx!r}  {mk}  -> {lb}")


if __name__ == "__main__":
    main()
