"""Dataset + collation for CARE-Fusion.

Turns the processed JSONL records (Part A) plus the train-only q-table (B1) and
optional weak regime labels (B2) into padded tensors:

  input_ids, attention_mask        : PhoBERT inputs (segmented text)
  marker_ids   [B, M]              : marker vocab id (0 = PAD/UNK)
  marker_q     [B, M, C]           : empirical q_j (or global back-off)
  marker_c     [B, M]              : intensity c_j = log(1 + count)
  marker_lowfreq [B, M]            : bool, use learned-embedding branch
  marker_mask  [B, M]              : 1 for real markers
  regime_labels [B, M]             : weak label id (0/1/2) or -100 (ignore)
  labels       [B]                 : gold emotion id

Markers are deduplicated per sample (set semantics) and ordered by first
appearance; c_j comes from the repeat count in `marker_counts`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

REGIME2ID = {"redundancy": 0, "complementarity": 1, "conflict": 2}
ID2REGIME = {v: k for k, v in REGIME2ID.items()}
IGNORE_INDEX = -100


def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class MarkerVocab:
    """Maps marker strings to ids (0 reserved for PAD/UNK)."""

    def __init__(self, q_table: dict):
        self.q = q_table["q"]
        self.low_freq = q_table["low_freq"]
        self.global_dist = q_table["global_dist"]
        self.C = len(q_table["labels"])
        markers = sorted(self.q.keys())
        self.marker2id = {m: i + 1 for i, m in enumerate(markers)}  # 0 = PAD/UNK
        self.size = len(markers) + 1

    def lookup(self, marker: str):
        """Return (id, q_vector, low_freq_flag) with global back-off for OOV."""
        mid = self.marker2id.get(marker, 0)
        q = self.q.get(marker, self.global_dist)
        lf = self.low_freq.get(marker, True)  # unseen markers back off
        return mid, q, lf


class CAREDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        vocab: MarkerVocab,
        weak_labels: Optional[Dict[str, list]] = None,
    ):
        self.records = records
        self.vocab = vocab
        self.weak = weak_labels or {}

    def __len__(self):
        return len(self.records)

    def _unique_markers(self, rec: dict):
        seen, out = set(), []
        for m in rec["marker_seq"]:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        counts = rec["marker_counts"]
        sid = str(rec["id"])
        regime_map = {d["marker"]: d["regime"] for d in self.weak.get(sid, [])}

        ids, qs, cs, lfs, regimes = [], [], [], [], []
        for m in self._unique_markers(rec):
            mid, q, lf = self.vocab.lookup(m)
            ids.append(mid)
            qs.append(q)
            cs.append(math.log1p(counts.get(m, 1)))
            lfs.append(bool(lf))
            regimes.append(REGIME2ID.get(regime_map.get(m), IGNORE_INDEX))

        return {
            "text": rec["text_segmented"],
            "label": rec["label_id"],
            "marker_ids": ids,
            "marker_q": qs,
            "marker_c": cs,
            "marker_lowfreq": lfs,
            "regime_labels": regimes,
        }


class Collator:
    """Tokenizes text and pads the variable-length marker lists."""

    def __init__(self, tokenizer, max_length: int, num_classes: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.C = num_classes

    def __call__(self, batch: List[dict]) -> dict:
        texts = [b["text"] for b in batch]
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        B = len(batch)
        M = max((len(b["marker_ids"]) for b in batch), default=0)
        M = max(M, 1)  # keep a marker axis even for marker-free batches

        marker_ids = torch.zeros(B, M, dtype=torch.long)
        marker_q = torch.zeros(B, M, self.C, dtype=torch.float)
        marker_c = torch.zeros(B, M, dtype=torch.float)
        marker_lowfreq = torch.zeros(B, M, dtype=torch.bool)
        marker_mask = torch.zeros(B, M, dtype=torch.float)
        regime_labels = torch.full((B, M), IGNORE_INDEX, dtype=torch.long)

        for i, b in enumerate(batch):
            for j, mid in enumerate(b["marker_ids"]):
                marker_ids[i, j] = mid
                marker_q[i, j] = torch.tensor(b["marker_q"][j], dtype=torch.float)
                marker_c[i, j] = b["marker_c"][j]
                marker_lowfreq[i, j] = b["marker_lowfreq"][j]
                marker_mask[i, j] = 1.0
                regime_labels[i, j] = b["regime_labels"][j]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "marker_ids": marker_ids,
            "marker_q": marker_q,
            "marker_c": marker_c,
            "marker_lowfreq": marker_lowfreq,
            "marker_mask": marker_mask,
            "regime_labels": regime_labels,
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }


def class_counts(records: List[dict], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for r in records:
        counts[r["label_id"]] += 1
    return counts
