# -*- coding: utf-8 -*-
"""Download UIT-VSMEC (train/valid/test CSV), inspect format, count emoji-bearing rows."""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.markers import extract_markers  # noqa: E402

DST = ROOT / "data" / "vsmec_raw"
BASE = "https://raw.githubusercontent.com/dinhthienan33/RecognitionEmotion-for-Vietnamese-Social-Media/main/UIT-VSMEC"
FILES = ["train.csv", "valid.csv", "test.csv"]
DST.mkdir(parents=True, exist_ok=True)

for fn in FILES:
    dst = DST / fn
    if not (dst.exists() and dst.stat().st_size > 0):
        print(f"[download] {fn}")
        urllib.request.urlretrieve(f"{BASE}/{fn}", dst)

frames = {fn: pd.read_csv(DST / fn) for fn in FILES}
df = pd.concat(frames.values(), ignore_index=True)
print(f"\nCột: {list(df.columns)}")
print(f"Tổng dòng: {len(df)}  | theo file: {{ {', '.join(f'{k}:{len(v)}' for k,v in frames.items())} }}")
# find text + label columns
text_col = next((c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.len().mean() > 15), df.columns[-1])
lab_col = next((c for c in df.columns if "emot" in c.lower() or "label" in c.lower()), None)
print(f"Cột text đoán: '{text_col}' | cột nhãn đoán: '{lab_col}'")
if lab_col:
    print("Phân phối nhãn:", df[lab_col].value_counts().to_dict())

has = df[text_col].apply(lambda t: extract_markers(str(t)).n_marker > 0)
n = int(has.sum())
print(f"\n>>> Dòng có emoji/emoticon: {n}/{len(df)} = {100*n/len(df):.1f}% <<<")
print("\nVí dụ có emoji:")
for t in df.loc[has, text_col].head(8):
    print("  -", str(t)[:90])
