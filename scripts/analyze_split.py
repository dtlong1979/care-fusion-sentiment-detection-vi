"""Quick report: how is the dataset split into train/val/test?"""
import collections
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "data/raw/vigoemotions_text_emoji_6groups.csv",
                                encoding="utf-8-sig")))
N = len(rows)
splits = collections.Counter(r["source_split"] for r in rows)

print("Tong so mau:", N)
print("\n=== Ty le chia ===")
for s in ["train", "val", "test"]:
    print(f"{s:5}: {splits[s]:5}  ({splits[s]/N*100:.1f}%)")

labels = ["positive", "sadness", "anger", "fear", "interest", "neutral"]
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[r["source_split"]][r["emotion_group"]] += 1

print("\n=== Phan bo nhom cam xuc theo bo (xem co stratified khong) ===")
print("lop".ljust(10) + "".join(f"{s:>18}" for s in ["train", "val", "test"]))
for l in labels:
    cells = ""
    for s in ["train", "val", "test"]:
        c = by[s][l]
        cells += f"{c:6} ({c/splits[s]*100:4.1f}%) "
    print(l.ljust(10) + cells)

print("\n=== emoji_type theo bo ===")
et = collections.defaultdict(collections.Counter)
for r in rows:
    et[r["source_split"]][r["emoji_type"]] += 1
for s in ["train", "val", "test"]:
    print(s, dict(et[s]))

ids = collections.defaultdict(set)
for r in rows:
    ids[r["source_split"]].add(r["source_id"])
print("\nID trung train&test:", len(ids["train"] & ids["test"]),
      "| train&val:", len(ids["train"] & ids["val"]),
      "| val&test:", len(ids["val"] & ids["test"]))
