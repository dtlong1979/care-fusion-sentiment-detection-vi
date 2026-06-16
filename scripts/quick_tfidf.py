"""Tier-0 estimate (seconds, CPU): classical TF-IDF + Logistic Regression on the
marker-free text, plus a 'text + marker-bag' variant. Gives a cheap reference
floor and a first hint of whether markers add signal — before any GPU spend.

NOT a reported number: PhoBERT will do better. Use for relative sanity only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["positive", "sadness", "anger", "fear", "interest", "neutral"]


def load(split):
    return [json.loads(l) for l in open(ROOT / f"data/processed/{split}.jsonl", encoding="utf-8")]


def macro(y, p, k=None):
    if k:  # macro over the k most frequent classes (exclude near-empty neutral)
        return f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0)
    return f1_score(y, p, average="macro", zero_division=0)


def evaluate(name, Xtr, ytr, splits):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(Xtr, ytr)
    print(f"\n[{name}]")
    for s, (X, y) in splits.items():
        pred = clf.predict(X)
        print(f"  {s:5}: macro-F1(6)={macro(y, pred):.4f}  macro-F1(5 lop dong)={macro(y, pred, 5):.4f}")


def main():
    tr, va, te = load("train"), load("val"), load("test")
    ytr = np.array([r["label_id"] for r in tr])
    yva = np.array([r["label_id"] for r in va])
    yte = np.array([r["label_id"] for r in te])

    # --- text-only (word + char n-grams) ---
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000)

    def text_feats(recs, fit=False):
        txt = [r["text_clean"] for r in recs]
        if fit:
            return hstack([wv.fit_transform(txt), cv.fit_transform(txt)])
        return hstack([wv.transform(txt), cv.transform(txt)])

    Xtr_t = text_feats(tr, fit=True)
    splits_t = {"val": (text_feats(va), yva), "test": (text_feats(te), yte)}
    evaluate("TEXT-only (TF-IDF word+char)", Xtr_t, ytr, splits_t)

    # --- text + marker-bag (does the marker signal help at all?) ---
    mv = TfidfVectorizer(tokenizer=lambda s: s.split(), preprocessor=lambda s: s,
                         token_pattern=None, min_df=1)

    def marker_str(recs):
        return [" ".join(r["marker_seq"]) if r["marker_seq"] else "__none__" for r in recs]

    Mtr = mv.fit_transform(marker_str(tr))
    Xtr_tm = hstack([Xtr_t, Mtr])
    splits_tm = {
        "val": (hstack([text_feats(va), mv.transform(marker_str(va))]), yva),
        "test": (hstack([text_feats(te), mv.transform(marker_str(te))]), yte),
    }
    evaluate("TEXT + MARKER-bag", Xtr_tm, ytr, splits_tm)

    # reference floors
    maj = np.bincount(ytr).argmax()
    print(f"\n[reference] majority-class test macro-F1(6) = "
          f"{macro(yte, np.full_like(yte, maj)):.4f}  (class={LABELS[maj]})")


if __name__ == "__main__":
    main()
