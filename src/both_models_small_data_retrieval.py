"""
both_models_small_data_retrieval.py

Controlled equal-data comparison of weak-signal and strong-signal XGBoost
routers under identical small-data training conditions.

All datasets are truncated to n_queries (default 300 = scifact, the smallest
dataset) using a fixed random seed.  This eliminates training-data size as a
confounding variable and allows a direct comparison of the two input
representations under matched conditions.

Both routers use the same shared XGBoost hyperparameters
(small_data_experiment.xgboost in config.yaml), so the comparison is purely
about the input representation, not model configuration.

Routing alpha predictions are produced by a 10-fold CV ensemble: 10 XGBoost
models are trained on independent random 80/20 splits of the 85% traindev set;
their test-set predictions are averaged for a more stable final alpha.

Methods compared on the 15% held-out test set:
  BM25-only            : sparse retrieval, no fusion
  Dense-only           : dense retrieval, no fusion
  Static RRF           : wRRF with α = 0.5 for every query
  wRRF (weak signal)   : 15 hand-crafted features → XGBoost CV ensemble
  wRRF (strong signal) : ~1024-dim embeddings     → XGBoost CV ensemble

Outputs:
  data/results/small_data_retrieval_comparison.csv
  data/results/small_data_retrieval_comparison.png

Usage:
    python src/both_models_small_data_retrieval.py
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
import xgboost as xgb

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
from src.strong_signal_model_grid_search import load_embeddings_for_dataset

# Weak-signal feature selection — identical to retrieval_explainability.py.
REMOVED_FEATURES = ["query_length"]
ACTIVE_COLS      = [i for i, n in enumerate(FEATURE_NAMES)
                    if n not in set(REMOVED_FEATURES)]


# ── CV-ensemble router ────────────────────────────────────────────────────────

def _cv_ensemble_alphas(X_traindev, y_traindev, X_test,
                        fold_indices, xgb_params, seed, xgb_device):
    """
    Train one XGBoost per fold on that fold's 80 % training portion.
    Return the mean of all 10 models' predictions on X_test.

    Each fold fits its own z-score normaliser on the fold training data
    and applies those stats to X_test, so there is no leakage from the
    validation or test portions into normalisation.
    """
    test_preds = []
    for tr_idx, _ in fold_indices:
        X_tr, y_tr = X_traindev[tr_idx], y_traindev[tr_idx]
        mu, sigma  = zscore_stats(X_tr)
        X_tr_z     = (X_tr   - mu) / sigma
        X_te_z     = (X_test - mu) / sigma

        mdl = xgb.XGBRegressor(
            objective    = "binary:logistic",
            eval_metric  = "logloss",
            tree_method  = "hist",
            device       = xgb_device,
            verbosity    = 0,
            random_state = seed,
            n_jobs       = 1 if xgb_device == "cuda" else -1,
            **xgb_params,
        )
        mdl.fit(X_tr_z, y_tr)
        test_preds.append(mdl.predict(X_te_z).astype(np.float32))

    return np.clip(np.mean(test_preds, axis=0), 0.0, 1.0)


# ── Retrieval scoring helpers ─────────────────────────────────────────────────

def _bm25_ndcg(qids, bm25_res, qrels, ndcg_k):
    ndcgs = [
        query_ndcg_at_k(bm25_res.get(qid, []), qrels.get(qid, {}), ndcg_k)
        for qid in qids
    ]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _dense_ndcg(qids, dense_res, qrels, ndcg_k):
    ndcgs = [
        query_ndcg_at_k(dense_res.get(qid, []), qrels.get(qid, {}), ndcg_k)
        for qid in qids
    ]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _wrrf_ndcg(alphas, qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
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


# ── Comparison bar chart (5 methods) ─────────────────────────────────────────

def _save_comparison_plot(rows, ndcg_k, n_queries, out_path):
    """Grouped bar chart: per-dataset + macro NDCG@k for all five methods."""
    methods = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong"]
    labels  = ["BM25", "Dense", "Static RRF (α=0.5)",
               "wRRF (weak)", "wRRF (strong)"]
    colors  = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#B47CC7"]

    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    width   = 0.15
    offsets = np.linspace(-(len(methods) - 1) / 2,
                           (len(methods) - 1) / 2,
                           len(methods)) * width

    by_method = {m: [r[m] for r in rows] for m in methods}

    fig, ax = plt.subplots(figsize=(16, 6))

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
                fontsize=6.0, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(
        f"Equal-Data Retrieval Comparison (n={n_queries} queries per dataset)"
        f" — NDCG@{ndcg_k}",
        fontsize=12,
    )
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
    warnings.filterwarnings("ignore")

    cfg  = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    cuda_available = torch.cuda.is_available()
    device         = torch.device("cuda" if cuda_available else "cpu")
    xgb_device     = "cuda" if cuda_available else "cpu"
    print(f"Device: {device}")

    dataset_names = cfg["datasets"]
    ndcg_k        = int(cfg["benchmark"]["ndcg_k"])
    rrf_k         = int(cfg["benchmark"]["rrf"]["k"])

    exp_cfg    = cfg["small_data_experiment"]
    n_queries  = int(exp_cfg["n_queries"])
    trunc_seed = int(exp_cfg["truncation_seed"])
    n_folds    = int(exp_cfg["n_folds"])
    test_frac  = float(exp_cfg["test_fraction"])
    xgb_params = dict(exp_cfg["xgboost"])

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)

    print(f"Truncating all datasets to {n_queries} queries")
    print(f"Shared XGBoost params: {xgb_params}")

    # ── Load all datasets ──────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    weak_data   = {}
    strong_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        weak_data[ds]   = load_dataset_for_grid_search(ds, cfg, device)
        strong_data[ds] = load_embeddings_for_dataset(ds, cfg, device)
        n_q = len(weak_data[ds]["qids"])
        print(f"  {n_q} queries  (truncating to {min(n_queries, n_q)})")

    if cuda_available:
        torch.cuda.empty_cache()

    # ── Per-dataset evaluation ─────────────────────────────────────────────────
    comparison_rows = []

    for ds_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")

        wd       = weak_data[ds_name]
        sd       = strong_data[ds_name]
        n_q_full = len(wd["qids"])

        # ── Truncation ─────────────────────────────────────────────────────────
        # Use a per-dataset seed so each dataset gets a different subsample
        # while remaining reproducible.
        n_use     = min(n_queries, n_q_full)
        rng_trunc = np.random.RandomState(trunc_seed + dataset_seed_offset(ds_name))
        trunc_idx = np.sort(rng_trunc.choice(n_q_full, size=n_use, replace=False))

        X_weak_full   = wd["X"][:, ACTIVE_COLS][trunc_idx]
        X_strong_full = sd["X"][trunc_idx]
        y_full        = wd["y"][trunc_idx]
        qids_full     = [wd["qids"][i] for i in trunc_idx]
        bm25_res      = wd["bm25_results"]
        dense_res     = wd["dense_results"]
        qrels         = wd["qrels"]

        print(f"  Truncated : {n_use} / {n_q_full} queries")
        print(f"  Weak dim  : {X_weak_full.shape[1]}   "
              f"Strong dim: {X_strong_full.shape[1]}")

        # ── 85 / 15 split on the truncated set ─────────────────────────────────
        split_seed   = seed + dataset_seed_offset(ds_name)
        rng_split    = np.random.RandomState(split_seed)
        perm         = rng_split.permutation(n_use)
        n_test       = max(1, int(test_frac * n_use))
        n_traindev   = n_use - n_test
        traindev_idx = perm[:n_traindev]
        test_idx     = perm[n_traindev:]

        X_weak_traindev   = X_weak_full[traindev_idx]
        X_strong_traindev = X_strong_full[traindev_idx]
        y_traindev        = y_full[traindev_idx]
        te_qids           = [qids_full[i] for i in test_idx]
        X_weak_test       = X_weak_full[test_idx]
        X_strong_test     = X_strong_full[test_idx]

        print(f"  Train+dev : {n_traindev}  |  Test: {n_test}")

        # ── 10-fold CV splits (same formula as all other scripts) ──────────────
        fold_indices = []
        for fi in range(n_folds):
            rng  = np.random.RandomState(
                seed + fi * 1000 + dataset_seed_offset(ds_name)
            )
            p    = rng.permutation(n_traindev)
            n_tr = max(1, min(n_traindev - 1, int(0.8 * n_traindev)))
            fold_indices.append((p[:n_tr], p[n_tr:]))

        # ── CV-ensemble routing ────────────────────────────────────────────────
        print(f"  Training weak-signal ensemble  ({n_folds} folds) ...")
        weak_alphas = _cv_ensemble_alphas(
            X_weak_traindev, y_traindev, X_weak_test,
            fold_indices, xgb_params, seed, xgb_device,
        )

        print(f"  Training strong-signal ensemble ({n_folds} folds) ...")
        strong_alphas = _cv_ensemble_alphas(
            X_strong_traindev, y_traindev, X_strong_test,
            fold_indices, xgb_params, seed, xgb_device,
        )

        # ── Evaluate all 5 methods on the held-out test set ────────────────────
        bm25_score   = _bm25_ndcg(te_qids, bm25_res, qrels, ndcg_k)
        dense_score  = _dense_ndcg(te_qids, dense_res, qrels, ndcg_k)
        srrf_alphas  = np.full(len(te_qids), 0.5, dtype=np.float32)
        srrf_score   = _wrrf_ndcg(srrf_alphas,  te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        wrrf_w_score = _wrrf_ndcg(weak_alphas,   te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        wrrf_s_score = _wrrf_ndcg(strong_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)

        print(f"  BM25-only          NDCG@{ndcg_k} = {bm25_score:.4f}")
        print(f"  Dense-only         NDCG@{ndcg_k} = {dense_score:.4f}")
        print(f"  Static RRF         NDCG@{ndcg_k} = {srrf_score:.4f}")
        print(f"  wRRF (weak)        NDCG@{ndcg_k} = {wrrf_w_score:.4f}")
        print(f"  wRRF (strong)      NDCG@{ndcg_k} = {wrrf_s_score:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "bm25":        bm25_score,
            "dense":       dense_score,
            "static_rrf":  srrf_score,
            "wrrf_weak":   wrrf_w_score,
            "wrrf_strong": wrrf_s_score,
        })

    # ── Macro averages ─────────────────────────────────────────────────────────
    n_evaluated = len(comparison_rows)
    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print(f"Macro averages across {n_evaluated} datasets "
          f"(n={n_queries} queries each):")
    print(f"  BM25-only          NDCG@{ndcg_k} = {macro['bm25']:.4f}")
    print(f"  Dense-only         NDCG@{ndcg_k} = {macro['dense']:.4f}")
    print(f"  Static RRF         NDCG@{ndcg_k} = {macro['static_rrf']:.4f}")
    print(f"  wRRF (weak)        NDCG@{ndcg_k} = {macro['wrrf_weak']:.4f}")
    print(f"  wRRF (strong)      NDCG@{ndcg_k} = {macro['wrrf_strong']:.4f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(
        results_folder, "small_data_retrieval_comparison.csv"
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            f"bm25_ndcg@{ndcg_k}",
            f"dense_ndcg@{ndcg_k}",
            f"static_rrf_ndcg@{ndcg_k}",
            f"wrrf_weak_ndcg@{ndcg_k}",
            f"wrrf_strong_ndcg@{ndcg_k}",
        ])
        for r in comparison_rows:
            writer.writerow([
                r["group"],
                f"{r['bm25']:.6f}",
                f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}",
                f"{r['wrrf_weak']:.6f}",
                f"{r['wrrf_strong']:.6f}",
            ])
    print(f"\nCSV saved: {csv_path}")

    # ── Save comparison bar chart ──────────────────────────────────────────────
    png_path = os.path.join(
        results_folder, "small_data_retrieval_comparison.png"
    )
    _save_comparison_plot(comparison_rows, ndcg_k, n_queries, png_path)


if __name__ == "__main__":
    main()
