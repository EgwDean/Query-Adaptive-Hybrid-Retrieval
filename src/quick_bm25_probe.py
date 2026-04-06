"""
quick_bm25_probe.py

Evaluates only the BM25 parameter combinations listed under bm25_probe in
config.yaml, plus the current bm25 baseline, across all configured datasets.
Reuses every cache that optimize_bm25.py has already built -- no redundant work.

Delete this script and the bm25_probe config section once the best params
have been picked and written into the bm25 section of config.yaml.

Usage:
    python src/quick_bm25_probe.py
"""

import itertools
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.optimize_bm25 import _run_or_load_results, _dataset_ndcg_at_k
import src.utils as u
from src.utils import load_config, get_config_path, ensure_dir


def main():
    cfg = load_config()
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise SystemExit("No datasets in config.")

    ndcg_k = int(cfg.get("benchmark", {}).get("ndcg_k", 10))

    # Current baseline from bm25 section
    baseline = u.get_bm25_params(cfg)

    # Probe grid from config
    probe = cfg.get("bm25_probe", {})
    k1_vals   = [float(v) for v in probe.get("k1_values",   [baseline["k1"]])]
    b_vals    = [float(v) for v in probe.get("b_values",    [baseline["b"]])]
    stem_vals = [bool(v)  for v in probe.get("use_stemming_values", [baseline["use_stemming"]])]

    # Build the full combo set, always including the baseline
    combos = set(itertools.product(k1_vals, b_vals, stem_vals))
    combos.add((baseline["k1"], baseline["b"], baseline["use_stemming"]))
    combos = sorted(combos)

    print("=" * 60)
    print(f"BM25 probe  --  {len(combos)} combinations, {len(datasets)} datasets")
    print(f"Baseline : k1={baseline['k1']}, b={baseline['b']}, "
          f"stemming={baseline['use_stemming']}")
    print("=" * 60)

    rows = []
    for k1, b, use_stemming in combos:
        tag = f"k1={k1:.3f}  b={b:.3f}  stem={int(use_stemming)}"
        ds_ndcgs = []
        for ds in datasets:
            bm25_results, qrels = _run_or_load_results(ds, cfg, k1, b, use_stemming)
            score_map = {qid: dict(pairs) for qid, pairs in bm25_results.items()}
            ds_ndcgs.append(_dataset_ndcg_at_k(score_map, qrels, ndcg_k))
        macro = float(np.mean(ds_ndcgs))
        is_base = (k1 == baseline["k1"] and b == baseline["b"]
                   and use_stemming == baseline["use_stemming"])
        rows.append((macro, k1, b, use_stemming, ds_ndcgs, is_base))
        marker = "  <- baseline" if is_base else ""
        print(f"  {tag}  |  macro NDCG@{ndcg_k} = {macro:.4f}{marker}")

    rows.sort(reverse=True)
    best_macro, best_k1, best_b, best_stem, best_per_ds, _ = rows[0]

    print("\n" + "=" * 60)
    print(f"Best: k1={best_k1}  b={best_b}  stemming={best_stem}  "
          f"macro NDCG@{ndcg_k}={best_macro:.4f}")
    print(f"Per dataset: " +
          "  ".join(f"{ds}={v:.4f}" for ds, v in zip(datasets, best_per_ds)))

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    out = os.path.join(results_folder, "bm25_probe_best.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "k1": float(best_k1),
            "b": float(best_b),
            "use_stemming": bool(best_stem),
            f"macro_ndcg@{ndcg_k}": best_macro,
            "per_dataset": {ds: v for ds, v in zip(datasets, best_per_ds)},
        }, f, indent=2)
    print(f"\nSaved best config to: {out}")
    print("Next: copy k1/b/use_stemming into config.yaml bm25 section, "
          "then delete this script and the bm25_probe config section.")


if __name__ == "__main__":
    main()
