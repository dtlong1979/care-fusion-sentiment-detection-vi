import json, sys
from pathlib import Path
from scipy import stats
import numpy as np
ds = sys.argv[1]
r = json.loads(Path(f"artifacts_a100/{ds}/results.json").read_text(encoding="utf-8"))
care = r["CARE_full"]["test_macro_f1"]
print(f"=== {ds} (n_seed={len(care)}) | CARE_full mean={np.mean(care):.4f} ===")
for x in ["B1_text","B2_concat","B3_gated","B4_crossattn"]:
    if x not in r: continue
    b = r[x]["test_macro_f1"]
    if len(b)!=len(care): print(f"  vs {x}: seed mismatch ({len(b)} vs {len(care)})"); continue
    d = np.mean(care)-np.mean(b)
    try: p = stats.wilcoxon(care,b).pvalue
    except ValueError as e: p=float('nan')
    print(f"  CARE_full vs {x:12}: Δ={d:+.4f}  Wilcoxon p={p:.4f}  {'*' if p<0.05 else ''}")
