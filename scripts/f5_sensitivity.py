"""Protocol F5 — preprocessing tie-break sensitivity (evaluation-time, no retrain).

Using the recovered ViGoEmotions index->group mapping, we re-derive the TEST
labels under alternative rules and recompute macro-F1 from the ALREADY-SAVED
model predictions. If the CARE-Fusion vs baseline ranking holds under all rules,
the conclusion is robust to the (consequential, ~32% of samples) tie-break.

Rules compared:
  ORIG : priority anger>sadness>fear>positive>interest>neutral  (paper default)
  ALT  : priority fear>interest>anger>sadness>positive>neutral  (rare-class-first)
  DROP : drop cross-polarity (POS<->NEG) multi-polar samples, evaluate the rest

    python scripts/f5_sensitivity.py --preds-dir artifacts/checkpoints/preds \
        --models CARE_full,B1_text,B4_crossattn
"""
import argparse
import ast
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
ORIG = Path(r"C:\Users\SingPC\Downloads")
PRIORITY = ["anger", "sadness", "fear", "positive", "interest", "neutral"]
PRIORITY_ALT = ["fear", "interest", "anger", "sadness", "positive", "neutral"]
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}


def recover_idx2group(der):
    frames = []
    for split in ["train", "val", "test"]:
        d = pd.read_csv(ORIG / f"{split}.csv", encoding="utf-8-sig")
        d.columns = [c.strip() for c in d.columns]
        d["split"] = split
        frames.append(d)
    orig = pd.concat(frames, ignore_index=True)
    orig["labels"] = orig["labels"].apply(ast.literal_eval)
    m = der.merge(orig[["id", "split", "labels"]],
                  left_on=["source_id", "source_split"], right_on=["id", "split"], how="left")
    idx2group = {}
    for _, r in m.iterrows():
        if isinstance(r["labels"], list) and len(r["labels"]) == 1:
            idx2group[r["labels"][0]] = r["emotion_group"]
    return idx2group, m


def resolve(groups, priority):
    for g in priority:
        if g in groups:
            return g
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="artifacts/checkpoints/preds")
    ap.add_argument("--models", default="CARE_full,B1_text")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    labels = cfg["labels"]
    lab2id = {l: i for i, l in enumerate(labels)}

    der = pd.read_csv(ROOT / "data/raw/vigoemotions_text_emoji_6groups.csv", encoding="utf-8-sig")
    der.columns = [c.strip() for c in der.columns]
    idx2group, merged = recover_idx2group(der)

    # build per-id alternative labels for the TEST split, in test.jsonl order
    test = [json.loads(l) for l in open(ROOT / "data/processed/test.jsonl", encoding="utf-8")]
    by_id = {(str(r["source_id"]), r["source_split"]): r["labels"]
             for _, r in merged.iterrows() if isinstance(r["labels"], list)}
    orig_lab, alt_lab, keep_drop = [], [], []
    for rec in test:
        groups = {idx2group[i] for i in by_id.get((str(rec["id"]), "test"), []) if i in idx2group}
        orig_lab.append(lab2id[resolve(groups, PRIORITY)])
        alt_lab.append(lab2id[resolve(groups, PRIORITY_ALT)])
        cross = bool(groups & POS) and bool(groups & NEG)
        keep_drop.append(not cross)            # True = keep (single-polarity)
    orig_lab = np.array(orig_lab); alt_lab = np.array(alt_lab); keep = np.array(keep_drop)

    agree = (orig_lab == alt_lab).mean()
    print(f"Test samples: {len(test)} | ORIG vs ALT label agreement: {100*agree:.2f}%")
    print(f"Cross-polarity (dropped under DROP): {int((~keep).sum())} "
          f"| kept: {int(keep.sum())}")

    def macro(y, p, m=None):
        idx = np.arange(len(y)) if m is None else np.where(m)[0]
        return f1_score(y[idx], p[idx], average="macro", zero_division=0)

    pdir = Path(args.preds_dir)
    if not pdir.is_absolute():
        pdir = ROOT / pdir
    print(f"\n{'model':16} {'ORIG':>8} {'ALT':>8} {'DROP-subset':>12}")
    for name in args.models.split(","):
        cand = sorted(pdir.glob(f"{name}__seed*.npz"))
        if not cand:
            print(f"  {name}: no preds found in {pdir}")
            continue
        # average over seeds
        f_orig, f_alt, f_drop = [], [], []
        for c in cand:
            preds = np.load(c)["preds"]
            n = len(preds)                      # align to first n test rows
            f_orig.append(macro(orig_lab[:n], preds))
            f_alt.append(macro(alt_lab[:n], preds))
            f_drop.append(macro(orig_lab[:n], preds, keep[:n]))
        print(f"  {name:14} {np.mean(f_orig):8.4f} {np.mean(f_alt):8.4f} {np.mean(f_drop):12.4f}")

    print("\nKết luận F5: nếu thứ hạng (CARE_full > baseline) giữ nguyên ở cả 3 cột "
          "→ kết luận KHÔNG phụ thuộc quy tắc tie-break.")


if __name__ == "__main__":
    main()
