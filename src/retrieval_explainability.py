"""
retrieval_explainability.py

For each dataset, trains an XGBoost router with the best per-dataset
hyperparameters found in the grid search, evaluates on the held-out 15% test
set, and compares against three baselines:

  BM25-only       : pure sparse retrieval (no fusion)
  Dense-only      : pure dense retrieval (no fusion)
  Static RRF      : wRRF with α = 0.5 for every query (no routing)
  wRRF (XGBoost)  : query-adaptive fusion with per-dataset best model

The 85/15 train/test split is identical to weak_signal_params_grid_search.py.
Feature set: 15 features (query_length excluded per ablation study).

Outputs:
  data/results/retrieval_comparison.csv   -- per-dataset + macro NDCG@10
  data/results/retrieval_comparison.png   -- grouped bar chart
  data/results/shap_<dataset>.png         -- SHAP beeswarm plot per dataset

Usage:
    python src/retrieval_explainability.py
"""

import csv
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import shap
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from src.utils import ensure_dir, get_config_path, load_config
from src.weak_signal_model_grid_search import (
    FEATURE_NAMES,
    dataset_seed_offset,
    load_dataset_for_grid_search,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
)

# ── Feature selection (must match weak_signal_params_grid_search.py) ──────────

REMOVED_FEATURES   = ["query_length"]
ACTIVE_COLS        = [i for i, n in enumerate(FEATURE_NAMES) if n not in set(REMOVED_FEATURES)]
ACTIVE_FEATURE_NAMES = [FEATURE_NAMES[i] for i in ACTIVE_COLS]


# ── Retrieval scoring helpers ─────────────────────────────────────────────────

def _bm25_ndcg(qids, bm25_res, qrels, ndcg_k):
    """Evaluate BM25-only ranked results."""
    ndcgs = [
        query_ndcg_at_k(bm25_res.get(qid, []), qrels.get(qid, {}), ndcg_k)
        for qid in qids
    ]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _dense_ndcg(qids, dense_res, qrels, ndcg_k):
    """Evaluate dense-only ranked results."""
    ndcgs = [
        query_ndcg_at_k(dense_res.get(qid, []), qrels.get(qid, {}), ndcg_k)
        for qid in qids
    ]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _wrrf_ndcg(alphas, qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
    """Evaluate wRRF fusion with given per-query alphas."""
    ndcgs = []
    for qid, alpha in zip(qids, alphas):
        alpha    = float(alpha)
        bm_pairs = bm25_res.get(qid, [])
        de_pairs = dense_res.get(qid, [])
        bm_rank  = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
        de_rank  = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
        bm_miss  = len(bm_pairs) + 1
        de_miss  = len(de_pairs) + 1
        fused = {
            d: alpha / (rrf_k + bm_rank.get(d, bm_miss))
               + (1.0 - alpha) / (rrf_k + de_rank.get(d, de_miss))
            for d in set(bm_rank) | set(de_rank)
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ndcgs.append(query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ── SHAP beeswarm plot ────────────────────────────────────────────────────────

def _save_shap_plot(mdl, X_all_z, dataset_name, out_path):
    """Compute SHAP values and save beeswarm/summary plot.

    Uses all queries (z-scored with traindev stats) to maximise the number of
    samples in the plot.  Features sorted by mean |SHAP value|; dots coloured
    by feature value (blue=low, red=high).
    """
    explainer   = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X_all_z)

    shap.summary_plot(
        shap_values,
        X_all_z,
        feature_names=ACTIVE_FEATURE_NAMES,
        show=False,
        max_display=len(ACTIVE_FEATURE_NAMES),
    )
    # Get figure reference explicitly — plt.title() would target the colorbar
    # axes (last axes created by SHAP); suptitle targets the whole figure.
    fig = plt.gcf()
    fig.suptitle(f"SHAP Feature Importance — {dataset_name}", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP plot saved: {out_path}")


# ── Comparison bar chart ──────────────────────────────────────────────────────

def _save_comparison_plot(rows, dataset_names, ndcg_k, out_path):
    """Grouped bar chart: per-dataset + macro NDCG@10 for all four methods."""
    methods = ["bm25", "dense", "static_rrf", "wrrf"]
    labels  = ["BM25", "Dense", "Static RRF (α=0.5)", "wRRF (XGBoost)"]
    colors  = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F"]

    groups  = list(dataset_names) + ["MACRO"]
    x       = np.arange(len(groups))
    width   = 0.18
    offsets = np.linspace(-(len(methods) - 1) / 2,
                           (len(methods) - 1) / 2,
                           len(methods)) * width

    by_method = {m: [] for m in methods}
    for grp in groups:
        row = next(r for r in rows if r["group"] == grp)
        for m in methods:
            by_method[m].append(row[m])

    fig, ax = plt.subplots(figsize=(14, 6))

    for method, label, color, offset in zip(methods, labels, colors, offsets):
        scores = by_method[method]
        bars   = ax.bar(x + offset, scores, width,
                        label=label, color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom",
                fontsize=6.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(f"Retrieval Method Comparison — NDCG@{ndcg_k} per Dataset",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    all_scores = [s for ss in by_method.values() for s in ss]
    ax.set_ylim(0, min(1.0, max(all_scores) + 0.12))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_names  = cfg["datasets"]
    ndcg_k         = int(cfg["benchmark"]["ndcg_k"])
    rrf_k          = int(cfg["benchmark"]["rrf"]["k"])
    test_frac      = float(cfg.get("xgboost_params_grid", {}).get("test_fraction", 0.15))
    per_ds_params  = cfg.get("xgboost_per_dataset", {})
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)

    print(f"\nActive features : {len(ACTIVE_COLS)} / {len(FEATURE_NAMES)}  "
          f"(removed: {REMOVED_FEATURES})")

    # ── Load all datasets ──────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_dataset_for_grid_search(ds, cfg, device)
        n_q = len(datasets_data[ds]["qids"])
        print(f"  {n_q} queries loaded")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Per-dataset train / evaluate / explain ─────────────────────────────────
    comparison_rows = []

    for ds_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")

        ds        = datasets_data[ds_name]
        X_all     = ds["X"][:, ACTIVE_COLS]
        y_all     = ds["y"]
        qids_all  = ds["qids"]
        bm25_res  = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels     = ds["qrels"]
        n_q       = len(qids_all)

        # ── Identical 85/15 split to weak_signal_params_grid_search.py ────────
        split_seed   = seed + dataset_seed_offset(ds_name)
        rng_split    = np.random.RandomState(split_seed)
        perm         = rng_split.permutation(n_q)
        n_test       = max(1, int(test_frac * n_q))
        n_traindev   = n_q - n_test
        traindev_idx = perm[:n_traindev]
        test_idx     = perm[n_traindev:]

        X_traindev   = X_all[traindev_idx]
        y_traindev   = y_all[traindev_idx]
        te_qids      = [qids_all[i] for i in test_idx]

        # Z-score: fit on train+dev, apply to test and full dataset
        mu, sigma    = zscore_stats(X_traindev)
        X_traindev_z = (X_traindev        - mu) / sigma
        X_test_z     = (X_all[test_idx]   - mu) / sigma
        X_all_z      = (X_all             - mu) / sigma  # used for SHAP

        # ── Train XGBoost with per-dataset best params ─────────────────────────
        if ds_name in per_ds_params:
            xgb_params = dict(per_ds_params[ds_name])
        else:
            print(f"  [WARN] No per-dataset params found for {ds_name}; using global xgboost_best.")
            xgb_params = dict(cfg["xgboost_best"])
        print(f"  XGBoost params: {xgb_params}")

        mdl = xgb.XGBRegressor(
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
            n_jobs=-1,
            **xgb_params,
        )
        mdl.fit(X_traindev_z, y_traindev)
        test_alphas = np.clip(mdl.predict(X_test_z).astype(np.float32), 0.0, 1.0)

        # ── Evaluate all four methods on the held-out test set ─────────────────
        bm25_score  = _bm25_ndcg(te_qids, bm25_res, qrels, ndcg_k)
        dense_score = _dense_ndcg(te_qids, dense_res, qrels, ndcg_k)
        srrf_alphas = np.full(len(te_qids), 0.5, dtype=np.float32)
        srrf_score  = _wrrf_ndcg(srrf_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        wrrf_score  = _wrrf_ndcg(test_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)

        print(f"  BM25-only      NDCG@{ndcg_k} = {bm25_score:.4f}")
        print(f"  Dense-only     NDCG@{ndcg_k} = {dense_score:.4f}")
        print(f"  Static RRF     NDCG@{ndcg_k} = {srrf_score:.4f}")
        print(f"  wRRF (XGBoost) NDCG@{ndcg_k} = {wrrf_score:.4f}")

        comparison_rows.append({
            "group":      ds_name,
            "bm25":       bm25_score,
            "dense":      dense_score,
            "static_rrf": srrf_score,
            "wrrf":       wrrf_score,
        })

        # ── SHAP beeswarm plot (all queries, z-scored with traindev stats) ─────
        shap_path = os.path.join(results_folder, f"shap_{ds_name}.png")
        _save_shap_plot(mdl, X_all_z, ds_name, shap_path)

    # ── Macro averages ─────────────────────────────────────────────────────────
    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["bm25", "dense", "static_rrf", "wrrf"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print(f"Macro averages across {len(dataset_names)} datasets:")
    print(f"  BM25-only      NDCG@{ndcg_k} = {macro['bm25']:.4f}")
    print(f"  Dense-only     NDCG@{ndcg_k} = {macro['dense']:.4f}")
    print(f"  Static RRF     NDCG@{ndcg_k} = {macro['static_rrf']:.4f}")
    print(f"  wRRF (XGBoost) NDCG@{ndcg_k} = {macro['wrrf']:.4f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_folder, "retrieval_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            f"bm25_ndcg@{ndcg_k}",
            f"dense_ndcg@{ndcg_k}",
            f"static_rrf_ndcg@{ndcg_k}",
            f"wrrf_ndcg@{ndcg_k}",
        ])
        for r in comparison_rows:
            writer.writerow([
                r["group"],
                f"{r['bm25']:.6f}",
                f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}",
                f"{r['wrrf']:.6f}",
            ])
    print(f"\nCSV saved: {csv_path}")

    # ── Save comparison bar chart ──────────────────────────────────────────────
    png_path = os.path.join(results_folder, "retrieval_comparison.png")
    _save_comparison_plot(comparison_rows, dataset_names, ndcg_k, png_path)


if __name__ == "__main__":
    main()
