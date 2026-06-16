"""Affective-marker extraction (Protocol A2).

Splits the non-linguistic signal (emoji + text emoticons) out of a sentence
while preserving order and repeat counts. Kept dependency-light (only the
`emoji` lib) so it can be unit-tested without torch / Java.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import emoji as emoji_lib

# Ordered (regex, canonical) pairs. Order matters: Python's alternation is
# leftmost-then-first-alternative, NOT longest-match, so more specific / longer
# patterns must come first (e.g. ":))" before ":)") to avoid ":))" being sliced
# into ":)". The `+` quantifier folds elongated variants (":)))", ":))))") onto
# the same base token.
_EMOTICON_SPECS: List[Tuple[str, str]] = [
    (r"=\)\)+", "=))"),
    (r":\)\)+", ":))"),
    (r":\(\(+", ":(("),
    (r":'\(", ":'("),
    (r">:\(", ">:("),
    (r"T_T", "T_T"),
    (r"\^\^", "^^"),
    (r"<3", "<3"),
    (r":v", ":v"),
    (r":3", ":3"),
    (r":D", ":D"),
    (r":P", ":P"),
    (r";\)", ";)"),
    (r":\|", ":|"),
    (r":\)", ":)"),
    (r":\(", ":("),
]

_EMOTICON_RE = re.compile("|".join(f"(?:{pat})" for pat, _ in _EMOTICON_SPECS))
_FULLMATCH = [(re.compile(f"^(?:{pat})$"), canon) for pat, canon in _EMOTICON_SPECS]


def _canonical_emoticon(surface: str, collapse_elongation: bool) -> str:
    """Map a matched emoticon surface form to its canonical token."""
    if not collapse_elongation:
        return surface
    for rx, canon in _FULLMATCH:
        if rx.match(surface):
            return canon
    return surface  # should not happen, but stay safe


@dataclass
class MarkerExtraction:
    text_clean: str              # text with all markers removed (feeds PhoBERT)
    marker_seq: List[str]        # canonical markers in order of appearance
    marker_counts: dict          # {marker: count}; intensity c_j = log(1+count)
    n_marker: int


def _remove_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """Drop the given [start, end) character ranges from `text`."""
    if not spans:
        return text
    out, cursor = [], 0
    for start, end in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    cleaned = "".join(out)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_markers(text: str, collapse_elongation: bool = True) -> MarkerExtraction:
    """Extract emoji + emoticons from `text` (A2).

    Returns the marker-free text plus the ordered marker sequence and counts.
    Emoji (non-ASCII) and emoticons (ASCII) do not overlap, but we still guard
    against overlapping spans defensively.
    """
    spans: List[Tuple[int, int, str]] = []

    for em in emoji_lib.emoji_list(text):
        spans.append((em["match_start"], em["match_end"], em["emoji"]))

    for m in _EMOTICON_RE.finditer(text):
        canon = _canonical_emoticon(m.group(0), collapse_elongation)
        spans.append((m.start(), m.end(), canon))

    spans.sort(key=lambda s: s[0])

    # Drop overlaps, keeping the earlier-starting span.
    deduped: List[Tuple[int, int, str]] = []
    last_end = -1
    for start, end, canon in spans:
        if start < last_end:
            continue
        deduped.append((start, end, canon))
        last_end = end

    marker_seq = [canon for _, _, canon in deduped]
    text_clean = _remove_spans(text, [(s, e) for s, e, _ in deduped])
    counts = dict(Counter(marker_seq))
    return MarkerExtraction(text_clean, marker_seq, counts, len(marker_seq))
