"""
full_vs_smaller_model_ablation_study.py

Compares the full 16-feature XGBoost wRRF router against a reduced
14-feature variant that drops `query_length` and `average_idf` --
the two features whose removal *improved* macro NDCG@10 in the
leave-one-out ablation study.

Both models are evaluated with the same protocol used in ablation_study.py:
  - 10-fold Monte Carlo CV (80/20 splits, same seeds)
  - Best XGBoost hyperparameters from config (xgboost_best)
  - Macro NDCG@10 across all configured datasets

Outputs:
  data/results/full_vs_smaller_ablation.csv  -- per-fold and per-dataset scores
  data/results/full_vs_smaller_ablation.png  -- grouped bar chart

Usage:
    python src/full_vs_smaller_model_ablation_study.py
"""

import csv
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import ensure_dir, get_config_path, load_config
from src.weak_signal_model_grid_search import (
    FEATURE_NAMES,
    dataset_seed_offset,
    load_dataset_for_grid_search,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
)

import torch

# ── Features to remove in the smaller model ───────────────────────────────────

REMOVED_FEATURES = ["query_length", "average_idf"]

FULL_COLS    = list(range(len(FEATURE_NAMES)))
REDUCED_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n not in set(REMOVED_FEATURES)]


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name,
    feature_cols,
    datasets_data,
    fold_indices,
    xgb_params,
    seed,
    rrf_k,
    ndcg_k,
):
    """Evaluate one model variant (full or reduced) across all datasets and folds.

    Returns a dict with:
      macro_ndcg10  : float  -- average across all datasets
      per_dataset   : dict   -- {ds_name: mean_ndcg_across_folds}
      fold_scores   : dict   -- {ds_name: [ndcg_fold_0, ..., ndcg_fold_n]}
    """
    per_ds = {}
    fold_scores = {}

    for ds_name, ds in datasets_data.items():
        X_full = ds["X"]
        y      = ds["y"]
        qids   = ds["qids"]
        bm25_res   = ds["bm25_results"]
        dense_res  = ds["dense_results"]
        qrels      = ds["qrels"]

        X = X_full[:, feature_cols]

        ds_fold_ndcgs = []
        for train_idx, test_idx in fold_indices[ds_name]:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]
            test_qids = [qids[i] for i in test_idx]

            mu, sigma = zscore_stats(X_tr)
            X_tr_z = (X_tr - mu) / sigma
            X_te_z = (X_te - mu) / sigma

            mdl = xgb.XGBRegressor(
                objective="binary:logistic",
                eval_metric="logloss",
                verbosity=0,
                random_state=seed,
                n_jobs=1,
                **xgb_params,
            )
            mdl.fit(X_tr_z, y_tr)
            alphas = np.clip(mdl.predict(X_te_z).astype(np.float32), 0.0, 1.0)

            ndcgs = []
            for qid, alpha in zip(test_qids, alphas):
                alpha = float(alpha)
                bm_pairs = bm25_res.get(qid, [])
                de_pairs = dense_res.get(qid, [])
                bm_rank = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
                de_rank = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
                bm_miss = len(bm_pairs) + 1
                de_miss = len(de_pairs) + 1
                fused = {}
                for d in set(bm_rank) | set(de_rank):
                    fused[d] = (
                        alpha / (rrf_k + bm_rank.get(d, bm_miss))
                        + (1.0 - alpha) / (rrf_k + de_rank.get(d, de_miss))
                    )
                ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
                ndcgs.append(query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k))

            ds_fold_ndcgs.append(float(np.mean(ndcgs)) if ndcgs else 0.0)

        per_ds[ds_name] = float(np.mean(ds_fold_ndcgs))
        fold_scores[ds_name] = ds_fold_ndcgs

    macro = float(np.mean(list(per_ds.values())))
    return {
        "model_name":   model_name,
        "macro_ndcg10": macro,
        "per_dataset":  per_ds,
        "fold_scores":  fold_scores,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def _make_plot(results, dataset_names, out_path):
    """Grouped bar chart: per-dataset NDCG@10 and macro average.

    One bar group per dataset (+ macro), two bars per group (full vs reduced).
    """
    full    = next(r for r in results if r["model_name"] == "full")
    reduced = next(r for r in results if r["model_name"] == "reduced")

    labels     = list(dataset_names) + ["MACRO"]
    full_scores    = [full["per_dataset"][ds]    for ds in dataset_names] + [full["macro_ndcg10"]]
    reduced_scores = [reduced["per_dataset"][ds] for ds in dataset_names] + [reduced["macro_ndcg10"]]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_full    = ax.bar(x - width / 2, full_scores,    width, label="Full (16 features)",    color="#4C72B0", alpha=0.85)
    bars_reduced = ax.bar(x + width / 2, reduced_scores, width, label="Reduced (14 features)", color="#DD8452", alpha=0.85)

    # Annotate bars with their values
    for bar in list(bars_full) + list(bars_reduced):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom",
            fontsize=7.5,
        )

    # Highlight delta on macro bar
    macro_delta = reduced["macro_ndcg10"] - full["macro_ndcg10"]
    sign = "+" if macro_delta >= 0 else ""
    ax.annotate(
        f"{sign}{macro_delta:.4f}",
        xy=(x[-1] + width / 2, reduced["macro_ndcg10"]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="red" if macro_delta < 0 else "green",
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("NDCG@10", fontsize=10)
    ax.set_title(
        f"Full (16 features) vs Reduced (14 features, no query_length / average_idf)\n"
        f"10-fold Monte Carlo CV · XGBoost best params · wRRF",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, max(full_scores + reduced_scores) + 0.06))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_names = cfg["datasets"]
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])

    abl_cfg    = cfg.get("ablation_study", {})
    n_folds    = int(abl_cfg.get("n_folds", 10))
    train_frac = float(abl_cfg.get("train_fraction", 0.8))
    n_jobs     = int(abl_cfg.get("n_jobs", -1))

    xgb_params = dict(cfg["xgboost_best"])

    print(f"\nFeatures removed in reduced model: {REMOVED_FEATURES}")
    print(f"Full model   : {len(FULL_COLS)} features")
    print(f"Reduced model: {len(REDUCED_COLS)} features")

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_dataset_for_grid_search(ds, cfg, device)
        n_q = len(datasets_data[ds]["qids"])
        print(f"  {n_q} queries, {datasets_data[ds]['X'].shape[1]} features")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Precompute fold indices (identical to ablation_study.py) ──────────────
    fold_indices = {}
    for ds in dataset_names:
        n_q = len(datasets_data[ds]["qids"])
        folds = []
        for fi in range(n_folds):
            rng = np.random.RandomState(seed + fi * 1000 + dataset_seed_offset(ds))
            perm = rng.permutation(n_q)
            n_train = max(1, min(n_q - 1, int(train_frac * n_q)))
            folds.append((perm[:n_train], perm[n_train:]))
        fold_indices[ds] = folds

    # ── Run both model variants in parallel ───────────────────────────────────
    variants = [
        ("full",    FULL_COLS),
        ("reduced", REDUCED_COLS),
    ]

    print("\n=== Evaluating full vs reduced model ===")
    results = Parallel(n_jobs=min(2, n_jobs if n_jobs > 0 else 2), prefer="threads")(
        delayed(evaluate_model)(
            name, cols, datasets_data, fold_indices,
            xgb_params, seed, rrf_k, ndcg_k,
        )
        for name, cols in variants
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    full_r    = next(r for r in results if r["model_name"] == "full")
    reduced_r = next(r for r in results if r["model_name"] == "reduced")

    print(f"\n{'Model':<12}  {'Macro NDCG@10':>14}", end="")
    for ds in dataset_names:
        print(f"  {ds:>12}", end="")
    print()

    for r in [full_r, reduced_r]:
        label = "full (16)" if r["model_name"] == "full" else "reduced (14)"
        print(f"{label:<12}  {r['macro_ndcg10']:>14.6f}", end="")
        for ds in dataset_names:
            print(f"  {r['per_dataset'][ds]:>12.6f}", end="")
        print()

    delta = reduced_r["macro_ndcg10"] - full_r["macro_ndcg10"]
    sign  = "+" if delta >= 0 else ""
    print(f"\nDelta (reduced − full): {sign}{delta:.6f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path = os.path.join(results_folder, "full_vs_smaller_ablation.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            ["model", "n_features", "removed_features", "macro_ndcg10"]
            + list(dataset_names)
            + [f"fold_{i}_{ds}" for ds in dataset_names for i in range(n_folds)]
        )
        for r in [full_r, reduced_r]:
            is_full = r["model_name"] == "full"
            removed = "—" if is_full else ", ".join(REMOVED_FEATURES)
            n_feat  = len(FULL_COLS) if is_full else len(REDUCED_COLS)
            per_ds_vals = [f"{r['per_dataset'][ds]:.6f}" for ds in dataset_names]
            fold_vals   = [
                f"{score:.6f}"
                for ds in dataset_names
                for score in r["fold_scores"][ds]
            ]
            writer.writerow(
                [r["model_name"], n_feat, removed, f"{r['macro_ndcg10']:.6f}"]
                + per_ds_vals
                + fold_vals
            )

    print(f"\nCSV saved to: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    png_path = os.path.join(results_folder, "full_vs_smaller_ablation.png")
    _make_plot(results, dataset_names, png_path)


if __name__ == "__main__":
    main()
