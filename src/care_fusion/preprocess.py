"""Preprocessing pipeline (Protocol Part A).

A1 Unicode/whitespace normalization (incl. fixing space-separated diacritics)
A2 marker extraction            -> see markers.py
A3 word segmentation (RDRSegmenter via py_vncorenlp) — REQUIRED for PhoBERT
A4 tokenization / truncation    -> done lazily at training time
A5 label encoding
A6 cleanliness check (flag empty text_clean; never silently drop test rows)

Run as a script to materialize data/processed/{train,val,test}.jsonl:
    python -m care_fusion.preprocess --config configs/default.yaml [--no-segment] [--subset N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .markers import extract_markers

# Combining diacritical marks block — Vietnamese tone/modifier marks live here.
_COMBINING = r"̀-ͯ"
_SPACE_BEFORE_COMBINING = re.compile(rf"\s+(?=[{_COMBINING}])")


def normalize_text(text: str, apply_nfc: bool = True) -> str:
    """A1: glue diacritics split by whitespace, NFC-normalize, collapse spaces."""
    if not isinstance(text, str):
        return ""
    # Fix patterns like "thâ ̣ thi ̀" where a combining mark is detached by a space.
    text = _SPACE_BEFORE_COMBINING.sub("", text)
    if apply_nfc:
        text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# A3: word segmentation (lazy global so import stays cheap and test-friendly)  #
# --------------------------------------------------------------------------- #
_SEGMENTER = None

# Only the jar + word-segmenter model are needed for `wseg`. We download them
# ourselves because py_vncorenlp.download_model shells out to `wget`, which is
# absent on Windows cmd.exe and leaves a half-written model dir behind.
_VNCORE_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"
_VNCORE_FILES = {
    "VnCoreNLP-1.2.jar": "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab": "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": "models/wordsegmenter/wordsegmenter.rdr",
}


def ensure_vncorenlp(save_dir: str) -> str:
    """Download the VnCoreNLP jar + word-segmenter model if missing (cross-platform)."""
    import urllib.request

    save_dir = os.path.abspath(save_dir)
    # VnCoreNLP's constructor only checks that a models/ dir and the jar exist.
    for sub in ("models", "models/wordsegmenter", "models/dep",
                "models/ner", "models/postagger"):
        os.makedirs(os.path.join(save_dir, sub), exist_ok=True)
    for rel, url_rel in _VNCORE_FILES.items():
        dest = os.path.join(save_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{_VNCORE_BASE}/{url_rel}"
        print(f"[vncorenlp] downloading {url_rel} ...")
        urllib.request.urlretrieve(url, dest)
    return save_dir


def get_segmenter(save_dir: str):
    """Load RDRSegmenter once. Requires Java (>= 1.8).

    py_vncorenlp changes the process cwd internally, so we hand it an absolute
    path and let `ensure_vncorenlp` guarantee the model files are present first.
    """
    global _SEGMENTER
    if _SEGMENTER is None:
        import py_vncorenlp

        save_dir = ensure_vncorenlp(save_dir)
        cwd = os.getcwd()
        try:
            _SEGMENTER = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        finally:
            # VnCoreNLP.__init__ does os.chdir(save_dir) and never restores it;
            # put the caller's cwd back so relative output paths still resolve.
            os.chdir(cwd)
    return _SEGMENTER


def segment_text(text: str, save_dir: str) -> str:
    """A3: return space-joined, word-segmented text (PhoBERT input form)."""
    if not text:
        return text
    seg = get_segmenter(save_dir)
    out = seg.word_segment(text)
    return " ".join(out) if isinstance(out, list) else str(out)


def build_label_maps(labels: List[str]):
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def process_dataframe(
    df: pd.DataFrame,
    cfg: dict,
    do_segment: bool = True,
) -> List[dict]:
    """Run A1-A3, A5-A6 over a dataframe -> list of processed records."""
    pcfg, dcfg = cfg["preprocess"], cfg["data"]
    label2id, _ = build_label_maps(cfg["labels"])
    save_dir = cfg["paths"]["vncorenlp_dir"]

    records: List[dict] = []
    for _, row in df.iterrows():
        raw = row[dcfg["text_col"]]
        norm = normalize_text(raw, apply_nfc=pcfg["apply_nfc"])           # A1
        ext = extract_markers(norm, collapse_elongation=pcfg["collapse_elongation"])  # A2

        text_for_bert = ext.text_clean
        if do_segment and pcfg.get("segment", True):
            text_for_bert = segment_text(ext.text_clean, save_dir)         # A3

        label = row[dcfg["label_col"]]
        records.append(
            {
                "id": row[dcfg["id_col"]],
                "split": row[dcfg["split_col"]],
                "emoji_type": row.get(dcfg["type_col"]),
                "text_raw": raw,
                "text_norm": norm,
                "text_clean": ext.text_clean,
                "text_segmented": text_for_bert,
                "marker_seq": ext.marker_seq,
                "marker_counts": ext.marker_counts,
                "n_marker": ext.n_marker,
                "label": label,
                "label_id": label2id[label],
                "text_empty": len(ext.text_clean.strip()) == 0,  # A6 flag, not a drop
            }
        )
    return records


def load_raw(cfg: dict) -> pd.DataFrame:
    dcfg = cfg["data"]
    df = pd.read_csv(cfg["paths"]["raw_csv"], encoding=dcfg["csv_encoding"])
    df.columns = [c.strip() for c in df.columns]  # defensive: strip stray BOM/space
    return df


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="CARE-Fusion preprocessing (Part A)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-segment", action="store_true",
                    help="skip A3 word segmentation (debugging only; hurts PhoBERT)")
    ap.add_argument("--subset", type=int, default=0,
                    help="process only the first N rows per split (smoke-test)")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = load_raw(cfg)
    print(f"[preprocess] loaded {len(df)} rows; splits = "
          f"{df[cfg['data']['split_col']].value_counts().to_dict()}")

    out_dir = Path(cfg["paths"]["processed_dir"]).resolve()  # absolute: survives chdir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_col = cfg["data"]["split_col"]

    for split in df[split_col].unique():
        sub = df[df[split_col] == split]
        if args.subset:
            sub = sub.head(args.subset)
        recs = process_dataframe(sub, cfg, do_segment=not args.no_segment)
        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_empty = sum(r["text_empty"] for r in recs)
        print(f"[preprocess] {split}: wrote {len(recs)} -> {out_path} "
              f"({n_empty} text_empty flagged)")


if __name__ == "__main__":
    main()
