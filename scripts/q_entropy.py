"""Show the empirical emotion distribution q_j and its entropy per marker.
High entropy = the emoji is intrinsically AMBIVALENT (data can't pin one emotion).

    python scripts/q_entropy.py [--top 30]
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(r"C:\Dev\care-fusion-sentiment-detection-vi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    q = json.loads((ROOT / "artifacts/q_table.json").read_text(encoding="utf-8"))
    labels, qmap, freq, low = q["labels"], q["q"], q["freq"], q["low_freq"]
    maxent = math.log(len(labels))

    def ent(p):
        return -sum(x * math.log(x + 1e-12) for x in p)

    rows = [(m, freq[m], ent(qmap[m]), qmap[m]) for m in qmap if not low[m]]
    rows.sort(key=lambda r: -r[1])   # by frequency

    print(f"{'marker':8} {'freq':>5} {'entropy':>8} (max={maxent:.2f}) | top groups")
    print("-" * 78)
    for m, f, e, p in rows[: a.top]:
        top = sorted(zip(labels, p), key=lambda x: -x[1])[:3]
        amb = "  <== AMBIVALENT" if e > 0.9 * maxent else ""
        print(f"{m:8} {f:5d} {e:8.3f}          | "
              + ", ".join(f"{l}:{v:.2f}" for l, v in top) + amb)

    hi = [m for m, f, e, p in rows if e > 0.9 * maxent]
    print(f"\nSố marker (freq>=tau) 'ambivalent' (entropy > 0.9*max): {len(hi)}/{len(rows)}")
    print("Ambivalent:", hi[:25])


if __name__ == "__main__":
    main()
