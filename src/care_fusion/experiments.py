"""Run the full experiment matrix (Protocol Part E) in one pass.

Baselines B0-B4, CARE-Fusion (full), and its ablations are each trained over the
configured seeds; per-seed test predictions are saved (for Part-G significance)
and a mean +/- std results table is written to artifacts/results.json.

    python -m care_fusion.experiments --config configs/default.yaml [--profile pilot] \
        [--variants CARE_full,B1_text] --out artifacts/checkpoints

Run baselines --emit-oof + resources --steps weak FIRST so weak labels exist.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from .baselines import TextOnlyModel, build_loader, make_loss_fn
from .data import Collator, MarkerVocab, class_counts, load_jsonl
from .engine import apply_profile, macro_f1, predict, set_seed, train_model
from .model import CAREFusion, MarkerConcatModel, ScalarGatedModel
from .train import make_care_loss_fn

# variant -> how to build it. kind in {majority,text,concat,gated,care}.
# flags go to CAREFusion; overrides patch cfg (e.g. ablate L_cf via lambda2=0).
VARIANTS: Dict[str, dict] = {
    "B0_majority":     {"kind": "majority"},
    "B1_text":         {"kind": "text"},
    "B2_concat":       {"kind": "concat"},
    "B3_gated":        {"kind": "gated"},
    "B4_crossattn":    {"kind": "care", "flags": {"use_routing": False}},
    "CARE_full":       {"kind": "care", "flags": {}},
    "CARE_-routing":   {"kind": "care", "flags": {"use_routing": False}},
    "CARE_-delta":     {"kind": "care", "flags": {"use_delta": False}},
    "CARE_-cf":        {"kind": "care", "flags": {}, "overrides": {"train": {"lambda2": 0.0}}},
    "CARE_-intensity": {"kind": "care", "flags": {"use_intensity": False}},
    "CARE_-gcn":       {"kind": "care", "flags": {"use_gcn": False}},
    "CARE_neutralbeta": {"kind": "care", "flags": {"beta_init": [0.0, 0.0, 0.0]}},  # F5-style control
    "CARE_-confidence": {"kind": "care", "flags": {"use_confidence": False}},  # ablate entropy gating
}


def build_model(kind, cfg, vocab, pmi, q_table, flags):
    if kind == "text":
        return TextOnlyModel(cfg)
    if kind == "concat":
        return MarkerConcatModel(cfg)
    if kind == "gated":
        return ScalarGatedModel(cfg)
    if kind == "care":
        return CAREFusion(cfg, vocab, pmi, q_table, **flags)
    raise ValueError(kind)


def run(cfg, device, profile, variants: List[str], out_dir: Path):
    cfg, subset = apply_profile(cfg, profile)
    C = len(cfg["labels"])
    art = Path(cfg["paths"]["artifacts_dir"])
    q_table = json.loads((art / "q_table.json").read_text(encoding="utf-8"))
    pmi = json.loads((art / "pmi_graph.json").read_text(encoding="utf-8"))
    vocab = MarkerVocab(q_table)
    weak_path = art / "weak_labels.json"
    weak = json.loads(weak_path.read_text(encoding="utf-8"))["weak_labels"] if weak_path.exists() else None

    pdir = Path(cfg["paths"]["processed_dir"])
    train = load_jsonl(pdir / "train.jsonl")
    val = load_jsonl(pdir / "val.jsonl")
    test = load_jsonl(pdir / "test.jsonl")
    if subset:
        train, val, test = train[:subset], val[: max(1, subset // 4)], test[: max(1, subset // 4)]

    tok = AutoTokenizer.from_pretrained(cfg["preprocess"]["phobert_name"])
    collate = Collator(tok, cfg["preprocess"]["max_length"], C)
    tcfg = cfg["train"]
    bs, seeds = tcfg["batch_size"], tcfg["seeds"]
    counts = class_counts(train, C)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # results.json lives in out_dir (e.g. Google Drive) so it survives a Colab
    # disconnect, alongside the preds/checkpoints. Merge with any previous run so
    # staged invocations (main models first, ablations later, or a re-run after a
    # disconnect) accumulate rather than overwrite.
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    for name in variants:
        spec = VARIANTS[name]
        kind = spec["kind"]
        print(f"\n########## {name} ({kind}) ##########")

        if kind == "majority":
            maj = Counter(r["label_id"] for r in train).most_common(1)[0][0]
            yte = np.array([r["label_id"] for r in test])
            f1 = macro_f1(yte, np.full_like(yte, maj))
            results[name] = {"test_macro_f1": [f1], "mean": f1, "std": 0.0}
            res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  test macro-F1 = {f1:.4f}")
            continue

        cfg_v = apply_profile(cfg, spec.get("overrides"))[0] if spec.get("overrides") else cfg
        is_care = kind == "care"
        loss_maker = make_care_loss_fn if is_care else make_loss_fn

        f1s = []
        for seed in seeds:
            set_seed(seed)
            model = build_model(kind, cfg_v, vocab, pmi, q_table, spec.get("flags", {}))
            w = weak if is_care else None
            res = train_model(
                model,
                build_loader(train, vocab, collate, bs, True, weak=w),
                build_loader(val, vocab, collate, bs, False, weak=w),
                cfg_v, device, loss_maker(counts, cfg_v),
                max_epochs=tcfg["max_epochs"], patience=tcfg["patience"], fp16=tcfg["fp16"],
            )
            ev = predict(model, build_loader(test, vocab, collate, bs, False), device)
            f1 = macro_f1(ev["labels"], ev["preds"])
            f1s.append(f1)
            np.savez(preds_dir / f"{name}__seed{seed}.npz",
                     logits=ev["logits"], probs=ev["probs"],
                     preds=ev["preds"], labels=ev["labels"])
            if name == "CARE_full":   # save only the headline model (disk-limited)
                torch.save({"state_dict": res["state"], "cfg": cfg_v, "seed": seed,
                            "flags": spec.get("flags", {})}, out_dir / f"{name}_seed{seed}.pt")
            print(f"  seed {seed}: val={res['best_f1']:.4f} test={f1:.4f}")
            if hasattr(model, "fusion") and hasattr(model.fusion, "marker_weights"):
                bw = [round(float(x), 3) for x in model.fusion.marker_weights().tolist()]
                print(f"    learned beta [redundancy, complementarity, conflict] = {bw}")

        arr = np.array(f1s)
        results[name] = {"test_macro_f1": f1s, "mean": float(arr.mean()), "std": float(arr.std())}
        print(f"  >>> {name}: {arr.mean():.4f} ± {arr.std():.4f}")

        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")  # save after each variant

    (art / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")  # convenience copy
    print("\n================ RESULTS (test macro-F1, all accumulated) ================")
    for name, r in results.items():
        print(f"  {name:18} {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"\nSaved -> {res_path} (+ {art/'results.json'}) ; predictions -> {preds_dir}")


def main(argv: List[str] = None):
    ap = argparse.ArgumentParser(description="Full experiment matrix (Part E)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="artifacts/checkpoints")
    ap.add_argument("--profile", choices=["full", "smoke", "pilot", "pilot3", "pilot5"], default="full")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", default=None,
                    help="comma list; default = all. e.g. CARE_full,B1_text")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    name = "smoke" if args.smoke else args.profile
    profile = cfg.get(name, {}) if name != "full" else {}
    variants = args.variants.split(",") if args.variants else list(VARIANTS.keys())
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}\navailable: {list(VARIANTS)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[experiments] device={device} profile={name} variants={variants}")
    run(cfg, device, profile, variants, Path(args.out))


if __name__ == "__main__":
    main()
