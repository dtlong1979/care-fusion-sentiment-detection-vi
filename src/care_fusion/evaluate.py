"""Evaluation & statistics (Protocol Part F-G).

F1  macro / weighted / per-class F1, confusion matrix, 5-class macro (drop neutral)
F2  regime-stratified macro-F1 (redundancy/complementarity/conflict) — headline table
F3  causal probing: prediction shift under marker removal (Causal Sensitivity Score)
F4  faithfulness (comprehensiveness): predicted-class prob drop when markers removed
G   bootstrap 95% CI for macro-F1; McNemar (single run) + Wilcoxon (across seeds)

F5 (preprocessing tie-break sensitivity) needs the ORIGINAL 27-label ViGoEmotions
annotations to recompute the 27->6 priority mapping; the provided 6-group CSV does
not carry them, so F5 is reported as a documented limitation (see README).

    python -m care_fusion.evaluate --config configs/default.yaml \
        --care-ckpt artifacts/checkpoints/CARE_full_seed42.pt --baseline B1_text
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from scipy import stats
from transformers import AutoTokenizer

from .data import Collator, MarkerVocab, class_counts, load_jsonl
from .engine import move
from .model import CAREFusion


# --------------------------------------------------------------------------- #
def macro(y, p, labels=None):
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def bootstrap_ci(labels, preds, n=1000, seed=0, k5=False):
    """95% CI of macro-F1 via test resampling."""
    rng = np.random.default_rng(seed)
    N = len(labels)
    lab = list(range(5)) if k5 else None
    stats_ = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        stats_.append(macro(labels[idx], preds[idx], lab))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(np.mean(stats_)), float(lo), float(hi)


def mcnemar(labels, preds_a, preds_b):
    """McNemar test on paired correctness (model A vs B), one fixed test set."""
    a_ok, b_ok = preds_a == labels, preds_b == labels
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if b + c == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_chi2": 1.0, "p_exact": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)               # continuity-corrected
    p_chi2 = float(stats.chi2.sf(stat, df=1))
    p_exact = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"b": b, "c": c, "stat": float(stat), "p_chi2": p_chi2, "p_exact": p_exact}


# --------------------------------------------------------------------------- #
def load_care(ckpt_path, vocab, pmi, q_table, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = CAREFusion(cfg, vocab, pmi, q_table, **ckpt.get("flags", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def run_care(model, loader, device):
    """Forward + counterfactual (marker removal) over test; collect per-sample stats."""
    keys = ["preds", "probs", "labels", "delta_max", "n_marker", "cf_preds", "kl_shift"]
    acc = {k: [] for k in keys}
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        cf = model(batch, drop_markers=True)
        p = out["logits"].softmax(-1)
        pcf = cf["logits"].softmax(-1)
        mask = batch["marker_mask"]
        dmax = out["delta"].masked_fill(mask == 0, float("nan"))
        dmax = torch.nan_to_num(dmax, nan=-1.0).max(1).values     # -1 if no marker
        kl = (p * (p.clamp_min(1e-8).log() - pcf.clamp_min(1e-8).log())).sum(-1)
        acc["preds"].append(p.argmax(-1).cpu())
        acc["probs"].append(p.cpu())
        acc["labels"].append(batch["labels"].cpu())
        acc["delta_max"].append(dmax.cpu())
        acc["n_marker"].append(mask.sum(1).cpu())
        acc["cf_preds"].append(pcf.argmax(-1).cpu())
        acc["kl_shift"].append(kl.cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


# --------------------------------------------------------------------------- #
def report_f1(name, labels, preds, cfg):
    L = cfg["labels"]
    print(f"\n--- F1 [{name}] ---")
    print(f"  macro-F1(6)      = {macro(labels, preds):.4f}")
    print(f"  macro-F1(5 dong) = {macro(labels, preds, list(range(5))):.4f}")
    print(f"  weighted-F1      = {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")
    per = f1_score(labels, preds, average=None, labels=list(range(len(L))), zero_division=0)
    print("  per-class F1: " + ", ".join(f"{l}={v:.3f}" for l, v in zip(L, per)))
    cm = confusion_matrix(labels, preds, labels=list(range(len(L))))
    print("  confusion (rows=true):")
    for l, row in zip(L, cm):
        print(f"    {l:9} " + " ".join(f"{x:4d}" for x in row))


def report_f2(stats_c, baseline, cfg, d_low, d_high):
    """Regime-stratified macro-F1: redundancy / complementarity / conflict."""
    d = stats_c["delta_max"]
    regime = np.full(len(d), "no_marker", dtype=object)
    has = d >= 0
    regime[has & (d < d_low)] = "redundancy"
    regime[has & (d >= d_low) & (d < d_high)] = "complementarity"
    regime[has & (d >= d_high)] = "conflict"
    print(f"\n--- F2 regime-stratified macro-F1 (delta_low={d_low:.3f}, delta_high={d_high:.3f}) ---")
    print(f"  {'stratum':16} {'n':>5}  {'CARE':>7}  {'baseline':>9}  {'Δ':>7}")
    y = stats_c["labels"]
    for strat in ["redundancy", "complementarity", "conflict", "no_marker"]:
        m = regime == strat
        if m.sum() == 0:
            continue
        fc = macro(y[m], stats_c["preds"][m])
        fb = macro(y[m], baseline["preds"][m]) if baseline is not None else float("nan")
        print(f"  {strat:16} {int(m.sum()):5d}  {fc:7.4f}  {fb:9.4f}  {fc-fb:+7.4f}")


def report_f3_f4(stats_c, cfg):
    print("\n--- F3 causal probing (marker removal) ---")
    has = stats_c["n_marker"] > 0
    change = (stats_c["preds"][has] != stats_c["cf_preds"][has]).mean() if has.any() else 0.0
    print(f"  samples with markers: {int(has.sum())}")
    print(f"  Causal Sensitivity Score (pred-change rate) = {change:.4f}")
    print(f"  mean KL(y || y_no-marker)                    = {stats_c['kl_shift'][has].mean():.4f}")
    print("\n--- F4 faithfulness (comprehensiveness) ---")
    # comprehensiveness = prob mass moved off the predicted class when markers removed
    print(f"  mean prediction shift (KL) when markers removed = {stats_c['kl_shift'][has].mean():.4f}")
    print(f"  (higher => marker weights are actually used by the model)")


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description="Evaluate CARE-Fusion (Part F-G)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--care-ckpt", required=True, help="CARE_full checkpoint (.pt)")
    ap.add_argument("--baseline", default="B1_text", help="variant name for comparison preds")
    ap.add_argument("--preds-dir", default="artifacts/checkpoints/preds")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--profile", choices=["full", "smoke", "pilot"], default="full")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from .engine import apply_profile
    pname = "smoke" if args.smoke else args.profile
    cfg, subset = apply_profile(cfg, cfg.get(pname, {}) if pname != "full" else {})
    C = len(cfg["labels"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    art = Path(cfg["paths"]["artifacts_dir"])
    q_table = json.loads((art / "q_table.json").read_text(encoding="utf-8"))
    pmi = json.loads((art / "pmi_graph.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q_table)

    weak = json.loads((art / "weak_labels.json").read_text(encoding="utf-8"))
    d_low, d_high = weak["delta_low"], weak["delta_high"]

    test = load_jsonl(Path(cfg["paths"]["processed_dir"]) / "test.jsonl")
    if subset:
        test = test[: max(1, subset // 4)]
    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    from torch.utils.data import DataLoader
    from .data import CAREDataset
    loader = DataLoader(CAREDataset(test, vocab), batch_size=cfg["train"]["batch_size"],
                        shuffle=False, collate_fn=collate)

    model, ckpt_cfg = load_care(args.care_ckpt, vocab, pmi, q_table, device)
    sc = run_care(model, loader, device)

    # baseline predictions (same test order) for F2 / G
    base = None
    bp = Path(args.preds_dir)
    cand = sorted(bp.glob(f"{args.baseline}__seed*.npz"))
    if cand:
        base = dict(np.load(cand[0]))

    report_f1("CARE_full", sc["labels"], sc["preds"], cfg)
    if base is not None:
        report_f1(args.baseline, base["labels"], base["preds"], cfg)
    report_f2(sc, base, cfg, d_low, d_high)
    report_f3_f4(sc, cfg)

    # Part G
    print("\n--- G statistical tests ---")
    m, lo, hi = bootstrap_ci(sc["labels"], sc["preds"])
    print(f"  CARE macro-F1(6) 95% CI = {m:.4f} [{lo:.4f}, {hi:.4f}]")
    if base is not None:
        mb, lob, hib = bootstrap_ci(base["labels"], base["preds"])
        print(f"  {args.baseline} macro-F1(6) 95% CI = {mb:.4f} [{lob:.4f}, {hib:.4f}]")
        mc = mcnemar(sc["labels"], sc["preds"], base["preds"])
        print(f"  McNemar CARE vs {args.baseline}: b={mc['b']} c={mc['c']} "
              f"stat={mc['stat']:.3f} p_chi2={mc['p_chi2']:.4g} p_exact={mc['p_exact']:.4g}")

    # Wilcoxon across seeds (needs >=2 seeds in results.json)
    res_path = art / "results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        a = res.get("CARE_full", {}).get("test_macro_f1", [])
        b = res.get(args.baseline, {}).get("test_macro_f1", [])
        if len(a) >= 2 and len(a) == len(b):
            w = stats.wilcoxon(a, b)
            print(f"  Wilcoxon CARE vs {args.baseline} over {len(a)} seeds: "
                  f"stat={w.statistic:.3f} p={w.pvalue:.4g}")
        else:
            print("  Wilcoxon: need >=2 matched seeds (run experiments with full seeds).")

    print("\n  F5 (tie-break sensitivity): requires original 27-label ViGoEmotions "
          "annotations (not in this 6-group CSV) -> see README Limitations.")


if __name__ == "__main__":
    main()
