# -*- coding: utf-8 -*-
"""Convert UIT-VSMEC and GoEmotions into the ViGoEmotions unified schema
(source_split, source_id, text, emoji_type, emotion_group), keeping only
emoji-bearing rows, mapped to the same 6 groups. The existing preprocess/
resources/experiments pipeline then ingests them via per-dataset configs.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.markers import extract_markers  # noqa: E402

RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def emoji_type(markers):
    has_uni = any(any(ord(c) > 0x2190 for c in m) for m in markers)
    has_emo = any(re.search(r"[:;=^<>T]", m) for m in markers)
    return "unicode+emoticon" if (has_uni and has_emo) else "unicode" if has_uni else "emoticon"


def keep_row(text):
    ext = extract_markers(str(text))
    if ext.n_marker == 0:
        return None
    if not re.search(r"[A-Za-zÀ-ỹ0-9]", ext.text_clean):
        return None
    return emoji_type(ext.marker_seq)


# ---------------- UIT-VSMEC (7 -> 6) ----------------
VSMEC_MAP = {"Enjoyment": "positive", "Sadness": "sadness", "Anger": "anger",
             "Disgust": "anger", "Fear": "fear", "Surprise": "interest", "Other": "neutral"}
vrows = []
for fn, split in [("train.csv", "train"), ("valid.csv", "val"), ("test.csv", "test")]:
    df = pd.read_csv(ROOT / "data" / "vsmec_raw" / fn)
    for i, r in df.iterrows():
        et = keep_row(r["Sentence"])
        if et is None:
            continue
        vrows.append({"source_split": split, "source_id": f"vsmec_{split}_{i}",
                      "text": r["Sentence"], "emoji_type": et,
                      "emotion_group": VSMEC_MAP[r["Emotion"]]})
vdf = pd.DataFrame(vrows)
vdf.to_csv(RAW / "vsmec.csv", index=False, encoding="utf-8-sig")
print(f"VSMEC -> {len(vdf)} mẫu | {vdf['emotion_group'].value_counts().to_dict()}")
print(f"  split: {vdf['source_split'].value_counts().to_dict()}")

# ---------------- GoEmotions (27 -> 6) ----------------
NAME2GROUP = {}
for g, names in {
    "positive": "amusement excitement joy love optimism caring pride admiration gratitude relief approval",
    "interest": "desire curiosity realization surprise confusion",
    "sadness": "disappointment sadness grief remorse embarrassment",
    "anger": "disgust anger annoyance disapproval", "fear": "fear nervousness", "neutral": "neutral",
}.items():
    for nm in names.split():
        NAME2GROUP[nm] = g
PRI = ["anger", "sadness", "fear", "positive", "interest", "neutral"]
EMO = list(NAME2GROUP.keys())
ge = pd.concat([pd.read_csv(ROOT / "data" / "goemotions_raw" / f"goemotions_{i}.csv") for i in (1, 2, 3)],
               ignore_index=True)
agg = ge.groupby("id")[EMO].sum(); nr = ge.groupby("id").size(); txt = ge.drop_duplicates("id").set_index("id")["text"]
grows = []
for cid in agg.index:
    v = agg.loc[cid]; n = nr[cid]
    present = [c for c in EMO if v[c] >= max(1, (n + 1) // 2)] or ([v.idxmax()] if v.max() > 0 else [])
    groups = {NAME2GROUP[c] for c in present}
    label = next((g for g in PRI if g in groups), None)
    if label is None:
        continue
    et = keep_row(txt[cid])
    if et is None:
        continue
    grows.append({"source_id": cid, "text": str(txt[cid]), "emoji_type": et, "emotion_group": label})
gdf = pd.DataFrame(grows)
tr, tmp = train_test_split(gdf, test_size=0.2, random_state=42, stratify=gdf["emotion_group"])
va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["emotion_group"])
gdf.loc[tr.index, "source_split"] = "train"; gdf.loc[va.index, "source_split"] = "val"; gdf.loc[te.index, "source_split"] = "test"
gdf = gdf[["source_split", "source_id", "text", "emoji_type", "emotion_group"]]
gdf.to_csv(RAW / "goemotions.csv", index=False, encoding="utf-8-sig")
print(f"GoEmotions -> {len(gdf)} mẫu | {gdf['emotion_group'].value_counts().to_dict()}")
print(f"  split: {gdf['source_split'].value_counts().to_dict()}")
