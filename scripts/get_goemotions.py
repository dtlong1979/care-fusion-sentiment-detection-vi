# -*- coding: utf-8 -*-
"""Download GoEmotions full dataset (3 CSVs), inspect format, and count how many
UNIQUE comments contain an emoji/emoticon (using our marker extractor).
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.markers import extract_markers  # noqa: E402

DST = ROOT / "data" / "goemotions_raw"
BASE = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset"
FILES = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

DST.mkdir(parents=True, exist_ok=True)
for fn in FILES:
    dst = DST / fn
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {fn} ({dst.stat().st_size/1e6:.1f} MB)")
        continue
    print(f"[download] {fn} ...")
    urllib.request.urlretrieve(f"{BASE}/{fn}", dst)
    print(f"  done ({dst.stat().st_size/1e6:.1f} MB)")

df = pd.concat([pd.read_csv(DST / fn) for fn in FILES], ignore_index=True)
print(f"\nTổng dòng (rater-level): {len(df)}")
print(f"Cột: {list(df.columns)[:12]} ...")
print(f"Số comment duy nhất (theo id): {df['id'].nunique()}")

# unique comments (text per id is identical)
uniq = df.drop_duplicates(subset="id")[["id", "text"]].reset_index(drop=True)
print(f"Unique comments: {len(uniq)}")

# count emoji-bearing
has_marker = uniq["text"].apply(lambda t: extract_markers(str(t)).n_marker > 0)
n_emoji = int(has_marker.sum())
print(f"\n>>> Comment có emoji/emoticon: {n_emoji} / {len(uniq)} = {100*n_emoji/len(uniq):.1f}% <<<")
print("\nVí dụ comment có emoji:")
for t in uniq.loc[has_marker, "text"].head(8):
    print("  -", str(t)[:90])
