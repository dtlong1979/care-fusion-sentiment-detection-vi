# -*- coding: utf-8 -*-
"""Test the 'amplify emoji as text' idea (simple, no fusion/routing).

Build text = text_clean + each marker repeated K times, then a simple TF-IDF
(word + char n-gram, char captures emoji) + Logistic Regression. Compare overall
and on congruent vs sarcasm slices, across K. Also check how PhoBERT tokenizes
emoji (does it become <unk>?).

    python scripts/emoji_amplify.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")
POS, NEG = {"positive", "interest"}, {"sadness", "anger", "fear"}
LABELS = ["positive", "sadness", "anger", "fear", "interest", "neutral"]


def load(split):
    return [json.loads(l) for l in open(ROOT / f"data/processed/{split}.jsonl", encoding="utf-8")]


def amplified(rec, K):
    extra = " ".join(" ".join([m] * K) for m in rec["marker_seq"]) if K > 0 else ""
    return (rec["text_clean"] + " " + extra).strip()


def main():
    # ---- PhoBERT tokenization check ----
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
        unk = tok.unk_token
        print("=== PhoBERT tokenize thử (emoji/emoticon) ===")
        for s in ["😂", "❤️", "🤣", ":))", ":v", "buồn 😢 quá"]:
            toks = tok.tokenize(s)
            print(f"  {s!r:14} -> {toks}  {'(<unk>!)' if unk in toks else ''}")
    except Exception as e:
        print("PhoBERT check skipped:", e)

    tr, va, te = load("train"), load("val"), load("test")
    ytr = np.array([r["label_id"] for r in tr])
    yte = np.array([r["label_id"] for r in te])

    # slices on test (marker polarity vs gold)
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    qmap, gdist = q["q"], q["global_dist"]
    pos_i = [i for i, l in enumerate(LABELS) if l in POS]
    neg_i = [i for i, l in enumerate(LABELS) if l in NEG]
    pol = lambda lab: "POS" if lab in POS else "NEG" if lab in NEG else "NEU"

    def epol(rec):
        ms = list(dict.fromkeys(rec["marker_seq"]))
        v = np.zeros(len(LABELS)); w = 0.0
        for mk in ms:
            ww = np.log1p(rec["marker_counts"].get(mk, 1)); v += ww * np.array(qmap.get(mk, gdist)); w += ww
        v /= max(w, 1e-9)
        return "POS" if v[pos_i].sum() > v[neg_i].sum() else "NEG"

    gp = [pol(r["label"]) for r in te]; ep = [epol(r) for r in te]
    cong = np.array([ep[i] == gp[i] and ep[i] in {"POS", "NEG"} for i in range(len(te))])
    conf = np.array([ep[i] != gp[i] and ep[i] in {"POS", "NEG"} and gp[i] in {"POS", "NEG"}
                     for i in range(len(te))])

    macro = lambda y, p, m=None: f1_score(y if m is None else y[m], p if m is None else p[m],
                                           average="macro", zero_division=0)
    acc = lambda y, p, m: float((p[m] == y[m]).mean())

    print(f"\n=== Emoji-amplified TF-IDF (char+word) — K = số lần nhân bản mỗi emoji ===")
    print(f"{'K':>3} {'test macroF1':>13} {'acc đồng thuận':>15} {'acc mỉa mai':>12}")
    for K in [0, 1, 3, 5, 10]:
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000)
        Xtr = hstack([wv.fit_transform([amplified(r, K) for r in tr]),
                      cv.fit_transform([amplified(r, K) for r in tr])])
        Xte = hstack([wv.transform([amplified(r, K) for r in te]),
                      cv.transform([amplified(r, K) for r in te])])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0).fit(Xtr, ytr)
        pte = clf.predict(Xte)
        print(f"{K:>3} {macro(yte, pte):13.4f} {acc(yte, pte, cong):15.3f} {acc(yte, pte, conf):12.3f}")


if __name__ == "__main__":
    main()
