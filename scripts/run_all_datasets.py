# -*- coding: utf-8 -*-
"""Run the FULL transformer pipeline for one dataset config (for the A100 multi-
dataset run). Steps: preprocess -> q/PMI -> OOF p_text -> weak labels ->
experiments matrix -> emoji-amplified (B1/B6) -> slice summary.

    python scripts/run_all_datasets.py --config configs/vsmec.yaml \
        --out /content/drive/MyDrive/care-fusion/vsmec \
        --variants B0_majority,B1_text,B2_concat,B3_gated,B4_crossattn,CARE_full
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants",
                    default="B0_majority,B1_text,B2_concat,B3_gated,B4_crossattn,CARE_full")
    ap.add_argument("--skip-preprocess", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    art = cfg["paths"]["artifacts_dir"]
    ptext = f"{art}/p_text_oof.json"
    C = args.config

    if not args.skip_preprocess:
        run([PY, "-m", "care_fusion.preprocess", "--config", C])
    run([PY, "-m", "care_fusion.resources", "--config", C, "--steps", "q,pmi"])
    run([PY, "-m", "care_fusion.baselines", "--config", C, "--emit-oof"])
    run([PY, "-m", "care_fusion.resources", "--config", C, "--steps", "weak", "--ptext", ptext])
    run([PY, "-m", "care_fusion.experiments", "--config", C, "--out", args.out,
         "--variants", args.variants])
    run([PY, "scripts/run_emoji_amp_phobert.py", "--config", C, "--Ks", "0,3"])
    run([PY, "scripts/slice_summary.py", "--preds-dir", f"{args.out}/preds"])
    print(f"\n=== DONE: {C} -> {args.out} ===")


if __name__ == "__main__":
    main()
