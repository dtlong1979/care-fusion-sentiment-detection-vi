"""Recover the index->group mapping ViGoEmotions used, verify it reproduces the
derived emotion_group, and report the % of samples affected by the priority rule
(i.e. samples that map to >=2 distinct high-level groups). Enables Protocol F5.

    python scripts/compute_multigroup.py
"""
import ast
import collections
from pathlib import Path

import pandas as pd

ORIG = Path(r"C:\Users\SingPC\Downloads")
DERIVED = Path(r"C:\Dev\care-fusion-sentiment-detection-vi\data\raw\vigoemotions_text_emoji_6groups.csv")
PRIORITY = ["anger", "sadness", "fear", "positive", "interest", "neutral"]


def load_orig():
    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(ORIG / f"{split}.csv", encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        df["split"] = "val" if split == "val" else split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["labels"] = out["labels"].apply(ast.literal_eval)
    return out


def main():
    orig = load_orig()
    der = pd.read_csv(DERIVED, encoding="utf-8-sig")
    der.columns = [c.strip() for c in der.columns]

    # join derived (the 6813 filtered) back to original label-index lists
    der = der.merge(
        orig[["id", "split", "labels"]],
        left_on=["source_id", "source_split"], right_on=["id", "split"], how="left",
    )
    missing = der["labels"].isna().sum()
    print(f"Joined {len(der)} derived rows; unmatched={missing}")

    # 1) recover index -> group from SINGLE-label samples
    idx2group = {}
    conflicts = collections.defaultdict(collections.Counter)
    for _, r in der.iterrows():
        if isinstance(r["labels"], list) and len(r["labels"]) == 1:
            conflicts[r["labels"][0]][r["emotion_group"]] += 1
    for idx, c in sorted(conflicts.items()):
        idx2group[idx] = c.most_common(1)[0][0]
        if len(c) > 1:
            print(f"  [warn] index {idx} ambiguous from single-label: {dict(c)}")
    print(f"Recovered mapping for {len(idx2group)} indices: "
          f"{ {i: idx2group[i] for i in sorted(idx2group)} }")

    # 2) verify: apply mapping + priority rule -> should reproduce emotion_group
    def resolve(label_idxs):
        groups = {idx2group[i] for i in label_idxs if i in idx2group}
        for g in PRIORITY:
            if g in groups:
                return g
        return None

    ok, bad = 0, 0
    n_multi = 0
    dist_ngroups = collections.Counter()
    for _, r in der.iterrows():
        if not isinstance(r["labels"], list):
            continue
        groups = {idx2group[i] for i in r["labels"] if i in idx2group}
        dist_ngroups[len(groups)] += 1
        if len(groups) >= 2:
            n_multi += 1
        if resolve(r["labels"]) == r["emotion_group"]:
            ok += 1
        else:
            bad += 1

    total = ok + bad
    print(f"\nVerification: reproduced {ok}/{total} emotion_group labels "
          f"({100*ok/total:.2f}%), mismatches={bad}")
    print(f"Distribution of #distinct groups/sample: {dict(sorted(dist_ngroups.items()))}")
    print(f"\n>>> MULTI-GROUP samples (>=2 groups, affected by priority rule): "
          f"{n_multi}/{total} = {100*n_multi/total:.2f}% <<<")


if __name__ == "__main__":
    main()
