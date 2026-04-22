"""
both_models_small_data_retrieval.py

Cross-dataset generalisation comparison of weak-signal and strong-signal
XGBoost routers trained on a merged pool of all datasets.

All datasets are truncated to n_queries (default 300 = scifact, the smallest
dataset).  The traindev portions are concatenated into one merged pool
(5 × ~255 ≈ 1275 queries) and a per-representation hyperparameter grid
search is run on that merged pool.  Each dataset's 15% test set is then
evaluated independently with the best parameters found.

Grid searches:
  Weak-signal  : same grid as xgboost_params_grid      (6 480 combos)
  Strong-signal: same grid as strong_signal_params_grid (   96 combos)

Best params are selected by mean NDCG@10 over 10-fold 80/20 CV on the
merged traindev.  NDCG is computed using actual BM25 / dense retrieval
results for each validation query, not proxy losses.

A 10-fold CV ensemble (trained with the best params) produces the final
alpha predictions on each dataset's test set.

Methods compared on each dataset's 15% held-out test set:
  BM25-only            : sparse retrieval, no fusion
  Dense-only           : dense retrieval, no fusion
  Static RRF           : wRRF with α = 0.5 for every query
  wRRF (weak signal)   : 15 hand-crafted features → merged CV ensemble
  wRRF (strong signal) : ~1024-dim embeddings     → merged CV ensemble

Outputs:
  data/results/small_data_best_params.csv         (best params per model)
  data/results/small_data_retrieval_comparison.csv
  data/results/small_data_retrieval_comparison.png

Usage:
    python src/both_models_small_data_retrieval.py
"""

import csv
import itertools
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
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

REMOVED_FEATURES = ["query_length"]
ACTIVE_COLS      = [i for i, n in enumerate(FEATURE_NAMES)
                    if n not in set(REMOVED_FEATURES)]

PARAM_KEYS = [
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "min_child_weight", "gamma",
]


# ── wRRF fusion (single query) ────────────────────────────────────────────────

def _wrrf_single(alpha, qid, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
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
    return query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k)


# ── Grid search ───────────────────────────────────────────────────────────────

def _make_combos(grid_cfg):
    """Cartesian product of all grid axes → list of param dicts."""
    lists = [grid_cfg[k] for k in PARAM_KEYS]
    return [dict(zip(PARAM_KEYS, vals)) for vals in itertools.product(*lists)]


def _eval_combo(combo, X, y, meta, fold_indices,
                retrieval_data, rrf_k, ndcg_k, seed):
    """
    10-fold CV NDCG@k for one param combination.
    Always uses CPU (called from joblib workers; CUDA cannot be shared).
    meta[i] = (ds_name, qid) for merged-traindev row i.
    """
    all_ndcgs = []
    for tr_idx, va_idx in fold_indices:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va       = X[va_idx]
        mu, sigma  = zscore_stats(X_tr)
        X_tr_z     = (X_tr - mu) / sigma
        X_va_z     = (X_va - mu) / sigma

        mdl = xgb.XGBRegressor(
            objective    = "binary:logistic",
            eval_metric  = "logloss",
            tree_method  = "hist",
            device       = "cpu",
            verbosity    = 0,
            random_state = seed,
            n_jobs       = 1,
            **combo,
        )
        mdl.fit(X_tr_z, y_tr)
        preds = np.clip(mdl.predict(X_va_z), 0.0, 1.0)

        for alpha, i in zip(preds, va_idx):
            ds_name, qid = meta[i]
            rd = retrieval_data[ds_name]
            all_ndcgs.append(_wrrf_single(
                alpha, qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))

    return float(np.mean(all_ndcgs)) if all_ndcgs else 0.0


def _grid_search(X, y, meta, fold_indices, grid_cfg,
                 retrieval_data, rrf_k, ndcg_k, n_jobs, seed, label):
    """Run full grid search; return (best_params_dict, best_cv_ndcg)."""
    combos  = _make_combos(grid_cfg)
    n_total = len(combos) * len(fold_indices)
    print(f"  {label}: {len(combos)} combos × {len(fold_indices)} folds = {n_total} fits")

    scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_eval_combo)(
            combo, X, y, meta, fold_indices,
            retrieval_data, rrf_k, ndcg_k, seed,
        )
        for combo in combos
    )

    best_idx   = int(np.argmax(scores))
    best_combo = combos[best_idx]
    best_score = float(scores[best_idx])
    print(f"  Best CV NDCG@{ndcg_k} : {best_score:.4f}")
    print(f"  Best params          : {best_combo}")
    return best_combo, best_score


# ── CV-ensemble: train on merged pool, apply per-dataset ─────────────────────

def _train_cv_ensemble(X, y, fold_indices, xgb_params, seed, xgb_device):
    """
    Train one XGBoost per fold on 80% of the merged traindev pool.
    Returns list of (model, mu, sigma) — one entry per fold.
    """
    fold_models = []
    for tr_idx, _ in fold_indices:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        mu, sigma  = zscore_stats(X_tr)
        X_tr_z     = (X_tr - mu) / sigma

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
        fold_models.append((mdl, mu, sigma))

    return fold_models


def _apply_ensemble(fold_models, X_test):
    """Apply a trained ensemble to X_test; returns averaged, clipped predictions."""
    test_preds = []
    for mdl, mu, sigma in fold_models:
        X_te_z = (X_test - mu) / sigma
        test_preds.append(mdl.predict(X_te_z).astype(np.float32))
    return np.clip(np.mean(test_preds, axis=0), 0.0, 1.0)


# ── Retrieval scoring helpers ─────────────────────────────────────────────────

def _bm25_ndcg(qids, bm25_res, qrels, ndcg_k):
    scores = [query_ndcg_at_k(bm25_res.get(q, []), qrels.get(q, {}), ndcg_k)
              for q in qids]
    return float(np.mean(scores)) if scores else 0.0


def _dense_ndcg(qids, dense_res, qrels, ndcg_k):
    scores = [query_ndcg_at_k(dense_res.get(q, []), qrels.get(q, {}), ndcg_k)
              for q in qids]
    return float(np.mean(scores)) if scores else 0.0


def _wrrf_ndcg(alphas, qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
    scores = [_wrrf_single(a, q, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
              for a, q in zip(alphas, qids)]
    return float(np.mean(scores)) if scores else 0.0


# ── Comparison bar chart (5 methods) ─────────────────────────────────────────

def _save_comparison_plot(rows, ndcg_k, n_queries, out_path):
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
        bars = ax.bar(x + offset, by_method[method], width,
                      label=label, color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=6.0, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(
        f"Cross-Dataset Retrieval Comparison — merged training"
        f" (n={n_queries} per dataset) — NDCG@{ndcg_k}",
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
    # Grid search always uses CPU workers (joblib + CUDA don't mix);
    # n_jobs=1 when CUDA present to avoid spawning CPU-heavy forks alongside GPU.
    n_jobs = 1 if cuda_available else int(
        cfg.get("xgboost_params_grid", {}).get("n_jobs", -1)
    )
    print(f"Device: {device}  |  grid-search n_jobs: {n_jobs}")

    dataset_names   = cfg["datasets"]
    ndcg_k          = int(cfg["benchmark"]["ndcg_k"])
    rrf_k           = int(cfg["benchmark"]["rrf"]["k"])

    exp_cfg         = cfg["small_data_experiment"]
    n_queries       = int(exp_cfg["n_queries"])
    trunc_seed      = int(exp_cfg["truncation_seed"])
    n_folds         = int(exp_cfg["n_folds"])
    test_frac       = float(exp_cfg["test_fraction"])

    weak_grid_cfg   = cfg["xgboost_params_grid"]
    strong_grid_cfg = cfg["strong_signal_params_grid"]

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)

    print(f"Truncating all datasets to {n_queries} queries each")

    # ── Load, truncate, split ─────────────────────────────────────────────────
    print("\n=== Loading and splitting datasets ===")

    per_ds         = {}   # per_ds[ds] holds test arrays + retrieval data
    traindev_meta  = []   # (ds_name, qid) for each row in the merged traindev pool
    retrieval_data = {}   # ds_name → {bm25_results, dense_results, qrels}

    X_weak_td_parts   = []
    X_strong_td_parts = []
    y_td_parts        = []

    for ds_name in dataset_names:
        print(f"\n--- {ds_name} ---")
        wd = load_dataset_for_grid_search(ds_name, cfg, device)
        sd = load_embeddings_for_dataset(ds_name, cfg, device)

        n_q_full = len(wd["qids"])
        n_use    = min(n_queries, n_q_full)
        print(f"  {n_q_full} queries → truncating to {n_use}")

        rng_trunc = np.random.RandomState(trunc_seed + dataset_seed_offset(ds_name))
        trunc_idx = np.sort(rng_trunc.choice(n_q_full, size=n_use, replace=False))

        X_weak_full   = wd["X"][:, ACTIVE_COLS][trunc_idx]
        X_strong_full = sd["X"][trunc_idx]
        y_full        = wd["y"][trunc_idx]
        qids_full     = [wd["qids"][i] for i in trunc_idx]

        # 85 / 15 split — same seed formula as all other scripts
        split_seed = seed + dataset_seed_offset(ds_name)
        rng_split  = np.random.RandomState(split_seed)
        perm       = rng_split.permutation(n_use)
        n_test     = max(1, int(test_frac * n_use))
        n_traindev = n_use - n_test
        td_idx     = perm[:n_traindev]
        test_idx   = perm[n_traindev:]
        td_qids    = [qids_full[i] for i in td_idx]

        print(f"  Train+dev: {n_traindev}  |  Test: {n_test}")

        X_weak_td_parts.append(X_weak_full[td_idx])
        X_strong_td_parts.append(X_strong_full[td_idx])
        y_td_parts.append(y_full[td_idx])
        traindev_meta.extend((ds_name, qid) for qid in td_qids)

        retrieval_data[ds_name] = {
            "bm25_results":  wd["bm25_results"],
            "dense_results": wd["dense_results"],
            "qrels":         wd["qrels"],
        }
        per_ds[ds_name] = {
            "X_weak_test":   X_weak_full[test_idx],
            "X_strong_test": X_strong_full[test_idx],
            "te_qids":       [qids_full[i] for i in test_idx],
            "bm25_res":      wd["bm25_results"],
            "dense_res":     wd["dense_results"],
            "qrels":         wd["qrels"],
        }

    if cuda_available:
        torch.cuda.empty_cache()

    # ── Merge traindev pools ──────────────────────────────────────────────────
    X_weak_merged   = np.vstack(X_weak_td_parts)
    X_strong_merged = np.vstack(X_strong_td_parts)
    y_merged        = np.concatenate(y_td_parts)
    n_merged        = len(y_merged)

    print(f"\n=== Merged training pool ===")
    print(f"  Total traindev queries : {n_merged}")
    print(f"  Weak dim               : {X_weak_merged.shape[1]}")
    print(f"  Strong dim             : {X_strong_merged.shape[1]}")

    # ── 10-fold CV splits on merged pool ──────────────────────────────────────
    fold_indices = []
    for fi in range(n_folds):
        rng  = np.random.RandomState(seed + fi * 1000)
        p    = rng.permutation(n_merged)
        n_tr = max(1, min(n_merged - 1, int(0.8 * n_merged)))
        fold_indices.append((p[:n_tr], p[n_tr:]))

    # ── Grid searches ─────────────────────────────────────────────────────────
    print("\n=== Grid search: weak-signal model ===")
    weak_best, weak_cv = _grid_search(
        X_weak_merged, y_merged, traindev_meta, fold_indices,
        weak_grid_cfg, retrieval_data, rrf_k, ndcg_k, n_jobs, seed,
        "Weak signal",
    )

    print("\n=== Grid search: strong-signal model ===")
    strong_best, strong_cv = _grid_search(
        X_strong_merged, y_merged, traindev_meta, fold_indices,
        strong_grid_cfg, retrieval_data, rrf_k, ndcg_k, n_jobs, seed,
        "Strong signal",
    )

    # ── Save best params ──────────────────────────────────────────────────────
    params_csv = os.path.join(results_folder, "small_data_best_params.csv")
    with open(params_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + PARAM_KEYS + [f"cv_ndcg@{ndcg_k}"])
        w.writerow(["weak_signal"]
                   + [weak_best[k]   for k in PARAM_KEYS]
                   + [f"{weak_cv:.6f}"])
        w.writerow(["strong_signal"]
                   + [strong_best[k] for k in PARAM_KEYS]
                   + [f"{strong_cv:.6f}"])
    print(f"\nBest params saved: {params_csv}")

    # ── Train final CV ensembles with best params ─────────────────────────────
    print("\n=== Training final ensembles ===")

    print(f"  Weak-signal  ({n_folds} folds) ...")
    weak_models = _train_cv_ensemble(
        X_weak_merged, y_merged, fold_indices, weak_best, seed, xgb_device,
    )

    print(f"  Strong-signal ({n_folds} folds) ...")
    strong_models = _train_cv_ensemble(
        X_strong_merged, y_merged, fold_indices, strong_best, seed, xgb_device,
    )

    # ── Per-dataset evaluation ────────────────────────────────────────────────
    print("\n=== Per-dataset evaluation ===")
    comparison_rows = []

    for ds_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        ds = per_ds[ds_name]

        weak_alphas   = _apply_ensemble(weak_models,   ds["X_weak_test"])
        strong_alphas = _apply_ensemble(strong_models, ds["X_strong_test"])

        bm25_res  = ds["bm25_res"]
        dense_res = ds["dense_res"]
        qrels     = ds["qrels"]
        te_qids   = ds["te_qids"]

        bm25_score   = _bm25_ndcg(te_qids, bm25_res, qrels, ndcg_k)
        dense_score  = _dense_ndcg(te_qids, dense_res, qrels, ndcg_k)
        srrf_alphas  = np.full(len(te_qids), 0.5, dtype=np.float32)
        srrf_score   = _wrrf_ndcg(srrf_alphas,   te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        wrrf_w_score = _wrrf_ndcg(weak_alphas,   te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        wrrf_s_score = _wrrf_ndcg(strong_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)

        print(f"  BM25-only      NDCG@{ndcg_k} = {bm25_score:.4f}")
        print(f"  Dense-only     NDCG@{ndcg_k} = {dense_score:.4f}")
        print(f"  Static RRF     NDCG@{ndcg_k} = {srrf_score:.4f}")
        print(f"  wRRF (weak)    NDCG@{ndcg_k} = {wrrf_w_score:.4f}")
        print(f"  wRRF (strong)  NDCG@{ndcg_k} = {wrrf_s_score:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "bm25":        bm25_score,
            "dense":       dense_score,
            "static_rrf":  srrf_score,
            "wrrf_weak":   wrrf_w_score,
            "wrrf_strong": wrrf_s_score,
        })

    # ── Macro averages ────────────────────────────────────────────────────────
    n_eval = len(comparison_rows)
    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print(f"Macro averages ({n_eval} datasets, n={n_queries} each, merged training):")
    for lbl, key in [
        ("BM25",         "bm25"),
        ("Dense",        "dense"),
        ("Static RRF",   "static_rrf"),
        ("wRRF (weak)",  "wrrf_weak"),
        ("wRRF (strong)","wrrf_strong"),
    ]:
        print(f"  {lbl:<16} NDCG@{ndcg_k} = {macro[key]:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_folder, "small_data_retrieval_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            f"bm25_ndcg@{ndcg_k}",
            f"dense_ndcg@{ndcg_k}",
            f"static_rrf_ndcg@{ndcg_k}",
            f"wrrf_weak_ndcg@{ndcg_k}",
            f"wrrf_strong_ndcg@{ndcg_k}",
        ])
        for r in comparison_rows:
            w.writerow([
                r["group"],
                f"{r['bm25']:.6f}",
                f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}",
                f"{r['wrrf_weak']:.6f}",
                f"{r['wrrf_strong']:.6f}",
            ])
    print(f"CSV saved: {csv_path}")

    # ── Save plot ─────────────────────────────────────────────────────────────
    png_path = os.path.join(results_folder, "small_data_retrieval_comparison.png")
    _save_comparison_plot(comparison_rows, ndcg_k, n_queries, png_path)


if __name__ == "__main__":
    main()
