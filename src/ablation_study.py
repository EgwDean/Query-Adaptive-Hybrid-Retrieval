"""
ablation_study.py

Feature ablation for the XGBoost wRRF router, using the best hyperparameters
found during model selection (config: xgboost_best).

Two ablation modes:
  1. Leave-one-feature-out  -- 16 configurations, one per feature.
  2. Leave-one-group-out    -- 5 configurations, one per feature group.

Each configuration is evaluated with 10-fold Monte Carlo CV (80/20 split)
across all configured datasets. Macro NDCG@10 is the comparison metric.

Outputs:
  data/results/ablation_study.csv   -- all configurations and scores
  data/results/ablation_study.png   -- two-panel horizontal bar chart

Usage:
    python src/ablation_study.py
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

# ── Feature groups (must match FEATURE_NAMES order) ───────────────────────────

FEATURE_GROUPS = {
    "A: Query Surface": [
        "query_length",
        "stopword_ratio",
        "has_question_word",
    ],
    "B: Vocabulary Match": [
        "average_idf",
        "max_idf",
        "rare_term_ratio",
        "cross_entropy",
    ],
    "C: Retriever Confidence": [
        "top_dense_score",
        "top_sparse_score",
        "dense_confidence",
        "sparse_confidence",
    ],
    "D: Retriever Agreement": [
        "overlap_at_k",
        "first_shared_doc_rank",
        "spearman_topk",
    ],
    "E: Distribution Shape": [
        "dense_entropy_topk",
        "sparse_entropy_topk",
    ],
}

def _cols_without(excluded_names):
    """Return column indices for all features except the excluded ones."""
    excluded = set(excluded_names)
    return [i for i, name in enumerate(FEATURE_NAMES) if name not in excluded]


def _build_ablation_configs():
    """
    Return a list of (config_name, ablation_type, removed_label, feature_cols).

    config_name   : unique string identifier written to the CSV
    ablation_type : "full" | "leave_one_feature" | "leave_one_group"
    removed_label : human-readable description of what was removed
    feature_cols  : list of column indices to use from X
    """
    configs = []

    # Baseline: all features
    configs.append((
        "full",
        "full",
        "—",
        list(range(len(FEATURE_NAMES))),
    ))

    # Leave-one-feature-out
    for name in FEATURE_NAMES:
        configs.append((
            f"no_{name}",
            "leave_one_feature",
            name,
            _cols_without([name]),
        ))

    # Leave-one-group-out
    for group_name, group_features in FEATURE_GROUPS.items():
        configs.append((
            f"no_group_{group_name.split(':')[0].strip()}",
            "leave_one_group",
            group_name,
            _cols_without(group_features),
        ))

    return configs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_config(
    config_name,
    feature_cols,
    datasets_data,
    fold_indices,
    xgb_params,
    seed,
    rrf_k,
    ndcg_k,
):
    """Evaluate one ablation configuration across all datasets and folds."""
    per_ds = {}

    for ds_name, ds in datasets_data.items():
        X_full = ds["X"]
        y = ds["y"]
        qids = ds["qids"]
        bm25_res = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels = ds["qrels"]

        # Select only the active feature columns
        X = X_full[:, feature_cols]

        fold_ndcgs = []
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

            fold_ndcgs.append(float(np.mean(ndcgs)) if ndcgs else 0.0)

        per_ds[ds_name] = float(np.mean(fold_ndcgs))

    macro = float(np.mean(list(per_ds.values())))
    return {
        "config_name": config_name,
        "macro_ndcg10": macro,
        "per_dataset": per_ds,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def _make_plot(full_score, lof_results, log_results, out_path):
    """
    Two-panel horizontal bar chart.

    Top panel   : leave-one-feature-out, sorted by score descending.
    Bottom panel: leave-one-group-out,   sorted by score descending.

    Both panels share the same x-axis range and show a vertical reference
    line at the full-model score.
    """
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 14),
        gridspec_kw={"height_ratios": [len(lof_results), len(log_results)]},
    )
    fig.suptitle("Feature Ablation Study — wRRF XGBoost Router", fontsize=14, y=1.01)

    # Determine shared x range with a small margin
    all_scores = [r["macro_ndcg10"] for r in lof_results + log_results] + [full_score]
    x_min = max(0.0, min(all_scores) - 0.01)
    x_max = min(1.0, max(all_scores) + 0.005)

    colors = {
        "leave_one_feature": "#4C72B0",
        "leave_one_group":   "#DD8452",
    }

    panels = [
        (axes[0], lof_results, "leave_one_feature", "Leave-one-feature-out"),
        (axes[1], log_results, "leave_one_group",   "Leave-one-group-out"),
    ]

    for ax, results, atype, title in panels:
        # Sort ascending; barh plots first item at bottom, so highest score ends up at top
        sorted_res = sorted(results, key=lambda r: r["macro_ndcg10"])
        labels = [r["removed_label"] for r in sorted_res]
        scores = [r["macro_ndcg10"] for r in sorted_res]
        deltas = [s - full_score for s in scores]

        bars = ax.barh(
            labels, scores,
            color=colors[atype],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
        )

        # Annotate each bar with delta vs full model
        for bar, delta in zip(bars, deltas):
            sign = "+" if delta >= 0 else ""
            ax.text(
                bar.get_width() + 0.0003,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.4f}",
                va="center", ha="left",
                fontsize=7.5,
                color="black",
            )

        # Full-model reference line
        ax.axvline(full_score, color="green", linewidth=1.4,
                   linestyle="--", label=f"Full model ({full_score:.4f})")

        ax.set_xlim(x_min, x_max + 0.012)
        ax.set_xlabel("Macro NDCG@10", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.legend(loc="lower right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

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
    rrf_k = int(cfg["benchmark"]["rrf"]["k"])

    abl_cfg = cfg.get("ablation_study", {})
    n_folds = int(abl_cfg.get("n_folds", 10))
    train_frac = float(abl_cfg.get("train_fraction", 0.8))
    n_jobs = int(abl_cfg.get("n_jobs", -1))

    xgb_params = dict(cfg["xgboost_best"])

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

    # ── Precompute fold indices ────────────────────────────────────────────────
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

    # ── Build ablation configurations ─────────────────────────────────────────
    ablation_configs = _build_ablation_configs()
    print(f"\n=== Ablation: {len(ablation_configs)} configurations ===")

    # ── Run in parallel ────────────────────────────────────────────────────────
    results_raw = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(evaluate_config)(
            config_name, feature_cols, datasets_data, fold_indices,
            xgb_params, seed, rrf_k, ndcg_k,
        )
        for config_name, _, _, feature_cols in ablation_configs
    )

    # Attach metadata from ablation_configs to results
    results = []
    for (config_name, atype, removed, _), res in zip(ablation_configs, results_raw):
        results.append({
            "config_name":   config_name,
            "ablation_type": atype,
            "removed_label": removed,
            "macro_ndcg10":  res["macro_ndcg10"],
            "per_dataset":   res["per_dataset"],
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path = os.path.join(results_folder, "ablation_study.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["config_name", "ablation_type", "removed", "macro_ndcg10"]
            + list(dataset_names)
        )
        for r in sorted(results, key=lambda x: x["macro_ndcg10"], reverse=True):
            writer.writerow([
                r["config_name"],
                r["ablation_type"],
                r["removed_label"],
                f"{r['macro_ndcg10']:.6f}",
                *[f"{r['per_dataset'].get(ds, 0.0):.6f}" for ds in dataset_names],
            ])

    print(f"\nCSV saved to: {csv_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    full_row = next((r for r in results if r["ablation_type"] == "full"), None)
    if full_row is None:
        raise RuntimeError("Full-model result missing from ablation output.")
    full_score = full_row["macro_ndcg10"]

    print(f"\nFull model macro NDCG@10: {full_score:.4f}")

    lof = [r for r in results if r["ablation_type"] == "leave_one_feature"]
    log_ = [r for r in results if r["ablation_type"] == "leave_one_group"]

    print("\nLeave-one-feature-out (sorted by drop):")
    for r in sorted(lof, key=lambda x: x["macro_ndcg10"]):
        delta = r["macro_ndcg10"] - full_score
        print(f"  {r['removed_label']:<28}  {r['macro_ndcg10']:.4f}  ({delta:+.4f})")

    print("\nLeave-one-group-out (sorted by drop):")
    for r in sorted(log_, key=lambda x: x["macro_ndcg10"]):
        delta = r["macro_ndcg10"] - full_score
        print(f"  {r['removed_label']:<35}  {r['macro_ndcg10']:.4f}  ({delta:+.4f})")

    # ── Plot ───────────────────────────────────────────────────────────────────
    png_path = os.path.join(results_folder, "ablation_study.png")
    _make_plot(full_score, lof, log_, png_path)


if __name__ == "__main__":
    main()
