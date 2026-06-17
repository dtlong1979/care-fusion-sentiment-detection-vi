# -*- coding: utf-8 -*-
"""Build a GoEmotions-derived 6-group EMOJI subset (same 27->6 mapping + priority as
ViGoEmotions), then a quick cross-lingual check: does amplifying emoji help (EN)?
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.markers import extract_markers  # noqa: E402

RAW = ROOT / "data" / "goemotions_raw"
NAME2GROUP = {}
for g, names in {
    "positive": "amusement excitement joy love optimism caring pride admiration gratitude relief approval",
    "interest": "desire curiosity realization surprise confusion",
    "sadness": "disappointment sadness grief remorse embarrassment",
    "anger": "disgust anger annoyance disapproval",
    "fear": "fear nervousness",
    "neutral": "neutral",
}.items():
    for nm in names.split():
        NAME2GROUP[nm] = g
PRIORITY = ["anger", "sadness", "fear", "positive", "interest", "neutral"]
EMO_COLS = list(NAME2GROUP.keys())

df = pd.concat([pd.read_csv(RAW / f"goemotions_{i}.csv") for i in (1, 2, 3)], ignore_index=True)
# aggregate rater-level -> comment-level: emotion present if majority of raters marked it
agg = df.groupby("id").agg({c: "sum" for c in EMO_COLS})
nrater = df.groupby("id").size()
text = df.drop_duplicates("id").set_index("id")["text"]

rows = []
for cid in agg.index:
    votes = agg.loc[cid]
    n = nrater[cid]
    present = [c for c in EMO_COLS if votes[c] >= max(1, (n + 1) // 2)]   # majority
    if not present:
        present = [votes.idxmax()] if votes.max() > 0 else []
    groups = {NAME2GROUP[c] for c in present}
    label = next((g for g in PRIORITY if g in groups), None)
    if label is None:
        continue
    t = str(text[cid])
    ext = extract_markers(t)
    if ext.n_marker == 0:                       # keep only emoji-bearing
        continue
    if not re.search(r"[A-Za-z0-9]", ext.text_clean):  # keep only with language content
        continue
    etype = ("unicode+emoticon" if any(ord(ch) > 0x2000 for m in ext.marker_seq for ch in m)
             and any(re.search(r"[:;=]", m) for m in ext.marker_seq)
             else "unicode" if any(ord(ch) > 0x2000 for m in ext.marker_seq for ch in m)
             else "emoticon")
    rows.append({"source_id": cid, "text": t, "emotion_group": label,
                 "n_marker": ext.n_marker})

ds = pd.DataFrame(rows)
# stratified 80/10/10 split
tr, tmp = train_test_split(ds, test_size=0.2, random_state=42, stratify=ds["emotion_group"])
va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=tmp["emotion_group"])
for name, part in [("train", tr), ("val", va), ("test", te)]:
    ds.loc[part.index, "source_split"] = name
out = ROOT / "data" / "goemotions_6groups.csv"
ds.to_csv(out, index=False, encoding="utf-8-sig")

print(f"GoEmotions-6group emoji subset: {len(ds)} mẫu -> {out}")
print("Phân phối nhãn:", ds["emotion_group"].value_counts().to_dict())
print("Split:", ds["source_split"].value_counts().to_dict())

# ---- quick cross-lingual check: does amplifying emoji help (overall macro-F1)? ----
def amp(t, K):
    ext = extract_markers(str(t))
    extra = " ".join(" ".join([m] * K) for m in ext.marker_seq) if K > 0 else ""
    return (ext.text_clean + " " + extra).strip()

lab2id = {l: i for i, l in enumerate(PRIORITY)}
tr_d = ds[ds.source_split == "train"]; te_d = ds[ds.source_split == "test"]
ytr = tr_d["emotion_group"].map(lab2id).values; yte = te_d["emotion_group"].map(lab2id).values
print(f"\n{'K':>3} {'GoEmotions test macro-F1':>26}")
for K in [0, 3]:
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000)
    Xtr = hstack([wv.fit_transform([amp(t, K) for t in tr_d.text]),
                  cv.fit_transform([amp(t, K) for t in tr_d.text])])
    Xte = hstack([wv.transform([amp(t, K) for t in te_d.text]),
                  cv.transform([amp(t, K) for t in te_d.text])])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(Xtr, ytr)
    f1 = f1_score(yte, clf.predict(Xte), average="macro", zero_division=0)
    print(f"{K:>3} {f1:26.4f}")
