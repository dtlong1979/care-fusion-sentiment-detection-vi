# -*- coding: utf-8 -*-
"""Replicate the key findings on a 2nd Vietnamese dataset (UIT-VSMEC, Facebook):
(1) emoji-amplification helps; (2) emoji is double-edged (helps congruent, hurts conflict).
Uses UIT-VSMEC's own 7 emotions and splits, emoji-bearing rows only."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
sys.path.insert(0, str(ROOT / "src"))
from care_fusion.markers import extract_markers  # noqa: E402

RAW = ROOT / "data" / "vsmec_raw"
POS, NEG = {"Enjoyment"}, {"Sadness", "Anger", "Disgust", "Fear"}   # Other/Surprise = neutral/ambiguous

def load(fn):
    df = pd.read_csv(RAW / fn)
    df["ext"] = df["Sentence"].apply(lambda t: extract_markers(str(t)))
    df["nmark"] = df["ext"].apply(lambda e: e.n_marker)
    return df[df["nmark"] > 0].reset_index(drop=True)

tr, va, te = load("train.csv"), load("valid.csv"), load("test.csv")
emos = sorted(pd.concat([tr, va, te])["Emotion"].unique())
lab2id = {e: i for i, e in enumerate(emos)}
print(f"Emoji-bearing: train {len(tr)} / val {len(va)} / test {len(te)} | 7 lớp: {emos}")

# build emoji q over 7 classes from train (emoji-bearing)
cnt = defaultdict(lambda: np.zeros(len(emos)))
for _, r in tr.iterrows():
    for m in dict.fromkeys(r["ext"].marker_seq):
        cnt[m][lab2id[r["Emotion"]]] += 1
glob = np.ones(len(emos)); glob /= glob.sum()
def qvec(m):
    if m in cnt and cnt[m].sum() >= 3:
        return (cnt[m] + 1) / (cnt[m].sum() + len(emos))
    return glob
pos_i = [i for i, e in enumerate(emos) if e in POS]; neg_i = [i for i, e in enumerate(emos) if e in NEG]
pol = lambda e: "POS" if e in POS else "NEG" if e in NEG else "NEU"

def epol(ext):
    ms = list(dict.fromkeys(ext.marker_seq))
    v = np.zeros(len(emos)); w = 0.0
    for m in ms:
        ww = np.log1p(1); v += ww * qvec(m); w += ww
    v /= max(w, 1e-9)
    return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"

gp = [pol(e) for e in te["Emotion"]]; ep = [epol(x) for x in te["ext"]]
cong = np.array([ep[i] == gp[i] and ep[i] in {"POS", "NEG"} for i in range(len(te))])
conf = np.array([ep[i] != gp[i] and ep[i] in {"POS", "NEG"} and gp[i] in {"POS", "NEG"} for i in range(len(te))])
print(f"test: đồng thuận {int(cong.sum())} | mỉa mai {int(conf.sum())}")

def amp(ext, K):
    extra = " ".join(" ".join([m] * K) for m in ext.marker_seq) if K > 0 else ""
    return (ext.text_clean + " " + extra).strip()

ytr = tr["Emotion"].map(lab2id).values; yte = te["Emotion"].map(lab2id).values
macro = lambda y, p, m=None: f1_score(y if m is None else y[m], p if m is None else p[m], average="macro", zero_division=0)
accm = lambda y, p, m: float((p[m] == y[m]).mean())

print(f"\n{'K':>3} {'test macroF1(7)':>16} {'acc đồng thuận':>15} {'acc mỉa mai':>12}")
for K in [0, 1, 3, 5]:
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000)
    Xtr = hstack([wv.fit_transform([amp(x, K) for x in tr["ext"]]), cv.fit_transform([amp(x, K) for x in tr["ext"]])])
    Xte = hstack([wv.transform([amp(x, K) for x in te["ext"]]), cv.transform([amp(x, K) for x in te["ext"]])])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(Xtr, ytr)
    p = clf.predict(Xte)
    print(f"{K:>3} {macro(yte, p):16.4f} {accm(yte, p, cong):15.3f} {accm(yte, p, conf):12.3f}")
