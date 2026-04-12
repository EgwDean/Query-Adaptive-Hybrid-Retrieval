"""
strong_signal_params_grid_search.py

Per-dataset XGBoost hyperparameter search using query embedding vectors
(~1024 dimensions from BAAI/bge-m3) as input features.

Identical method to weak_signal_params_grid_search.py:
  - 85 % train+dev  (used for 10-fold CV to select hyperparameters)
  - 15 % test       (held out; evaluated exactly once with the best params)
  - Same expanded XGBoost grid (xgboost_params_grid in config.yaml)
  - Same 85/15 split seeds as the weak-signal script for a fair comparison

Output:
  data/results/strong_signal_per_dataset_best_params.csv

Usage:
    python src/strong_signal_params_grid_search.py
"""

import csv
import itertools
import os
import sys
import warnings

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from src.utils import ensure_dir, get_config_path, load_config
from src.weak_signal_model_grid_search import (
    dataset_seed_offset,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
)
from src.strong_signal_model_grid_search import load_embeddings_for_dataset


# ── wRRF scoring ──────────────────────────────────────────────────────────────

def _wrrf_ndcg(alphas, qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
    ndcgs = []
    for qid, alpha in zip(qids, alphas):
        alpha = float(alpha)
        bm_pairs = bm25_res.get(qid, [])
        de_pairs = dense_res.get(qid, [])
        bm_rank = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
        de_rank = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
        bm_miss = len(bm_pairs) + 1
        de_miss = len(de_pairs) + 1
        fused = {
            d: alpha / (rrf_k + bm_rank.get(d, bm_miss))
               + (1.0 - alpha) / (rrf_k + de_rank.get(d, de_miss))
            for d in set(bm_rank) | set(de_rank)
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ndcgs.append(query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ── Single-combo CV evaluation (called in parallel) ───────────────────────────

def _eval_combo(params, X_traindev, y_traindev, traindev_qids,
                fold_indices, bm25_res, dense_res, qrels, seed, rrf_k, ndcg_k):
    """10-fold CV on train+dev; return mean NDCG@k across folds."""
    fold_scores = []
    for tr_idx, va_idx in fold_indices:
        X_tr, y_tr = X_traindev[tr_idx], y_traindev[tr_idx]
        X_va = X_traindev[va_idx]
        va_qids = [traindev_qids[i] for i in va_idx]

        mu, sigma = zscore_stats(X_tr)
        X_tr_z = (X_tr - mu) / sigma
        X_va_z = (X_va - mu) / sigma

        mdl = xgb.XGBRegressor(
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
            n_jobs=1,
            **params,
        )
        mdl.fit(X_tr_z, y_tr)
        alphas = np.clip(mdl.predict(X_va_z).astype(np.float32), 0.0, 1.0)
        fold_scores.append(_wrrf_ndcg(alphas, va_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k))

    return float(np.mean(fold_scores))


# ── Grid construction ─────────────────────────────────────────────────────────

def _build_grid(grid_cfg):
    keys = sorted(grid_cfg.keys())
    values = [grid_cfg[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    cfg = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_names = cfg["datasets"]
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])

    grid_cfg  = dict(cfg["xgboost_params_grid"])
    n_jobs    = int(grid_cfg.pop("n_jobs", -1))
    n_folds   = int(grid_cfg.pop("n_folds", 10))
    test_frac = float(grid_cfg.pop("test_fraction", 0.15))

    combos     = _build_grid(grid_cfg)
    param_keys = sorted(grid_cfg.keys())

    print(f"Input     : query embedding vectors (full dimensionality)")
    print(f"Grid size : {len(combos)} combinations per dataset")
    print(f"CV folds  : {n_folds}")
    print(f"Test hold : {int(test_frac * 100)} %")

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_embeddings_for_dataset(ds, cfg, device)
        n_q    = len(datasets_data[ds]["qids"])
        emb_dim = datasets_data[ds]["X"].shape[1]
        print(f"  {n_q} queries, embedding dim = {emb_dim}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Per-dataset search ─────────────────────────────────────────────────────
    results = []

    for ds_name in dataset_names:
        ds        = datasets_data[ds_name]
        X         = ds["X"]          # shape [N, emb_dim] — full embedding matrix
        y         = ds["y"]
        qids      = ds["qids"]
        bm25_res  = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels     = ds["qrels"]
        n_q       = len(qids)

        # Fixed 85/15 traindev/test split — identical seed to weak-signal script
        # so both experiments evaluate on the same held-out test queries.
        split_seed   = seed + dataset_seed_offset(ds_name)
        rng_split    = np.random.RandomState(split_seed)
        perm         = rng_split.permutation(n_q)
        n_test       = max(1, int(test_frac * n_q))
        n_traindev   = n_q - n_test
        traindev_idx = perm[:n_traindev]
        test_idx     = perm[n_traindev:]

        # Precompute 10 CV fold splits on the traindev portion
        fold_indices = []
        for fi in range(n_folds):
            rng  = np.random.RandomState(seed + fi * 1000 + dataset_seed_offset(ds_name))
            p    = rng.permutation(n_traindev)
            n_tr = max(1, min(n_traindev - 1, int(0.8 * n_traindev)))
            fold_indices.append((p[:n_tr], p[n_tr:]))

        X_traindev    = X[traindev_idx]
        y_traindev    = y[traindev_idx]
        traindev_qids = [qids[i] for i in traindev_idx]
        te_qids       = [qids[i] for i in test_idx]

        print(f"\n{'=' * 60}")
        print(f"Dataset    : {ds_name}")
        print(f"Train+dev  : {n_traindev}  |  Test: {n_test}")
        print(f"CV folds   : {n_folds} × 80/20 splits of train+dev")
        print(f"Grid       : {len(combos)} combinations")
        print(f"{'=' * 60}")

        # ── Grid search (parallel across combos, each does 10-fold CV) ────────
        cv_scores = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                delayed(_eval_combo)(
                    p, X_traindev, y_traindev, traindev_qids,
                    fold_indices, bm25_res, dense_res, qrels,
                    seed, rrf_k, ndcg_k,
                )
                for p in combos
            ),
            total=len(combos),
            desc=f"  [{ds_name}] grid search",
            dynamic_ncols=True,
        ))

        best_idx    = int(np.argmax(cv_scores))
        best_params = combos[best_idx]
        best_cv     = cv_scores[best_idx]

        print(f"  Best CV NDCG@{ndcg_k} = {best_cv:.4f}  |  params: {best_params}")

        # ── Final model: train on full train+dev, evaluate on test ────────────
        mu, sigma    = zscore_stats(X_traindev)
        X_traindev_z = (X_traindev     - mu) / sigma
        X_te_z       = (X[test_idx]    - mu) / sigma

        final_mdl = xgb.XGBRegressor(
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            random_state=seed,
            n_jobs=-1,
            **best_params,
        )
        final_mdl.fit(X_traindev_z, y_traindev)
        test_alphas = np.clip(final_mdl.predict(X_te_z).astype(np.float32), 0.0, 1.0)
        test_ndcg   = _wrrf_ndcg(test_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k)

        print(f"  Test NDCG@{ndcg_k}  = {test_ndcg:.4f}")

        results.append({
            "dataset":     ds_name,
            "params":      best_params,
            "cv_ndcg10":   best_cv,
            "test_ndcg10": test_ndcg,
        })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path = os.path.join(results_folder, "strong_signal_per_dataset_best_params.csv")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset"] + param_keys + [f"cv_ndcg@{ndcg_k}", f"test_ndcg@{ndcg_k}"]
        )
        for r in results:
            writer.writerow(
                [r["dataset"]]
                + [r["params"][k] for k in param_keys]
                + [f"{r['cv_ndcg10']:.6f}", f"{r['test_ndcg10']:.6f}"]
            )

    print(f"\nSaved: {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'Dataset':<15} {'CV NDCG@10':>11} {'Test NDCG@10':>13}")
    print("-" * 42)
    for r in results:
        print(f"  {r['dataset']:<13} {r['cv_ndcg10']:>11.4f} {r['test_ndcg10']:>13.4f}")
    macro_test = float(np.mean([r["test_ndcg10"] for r in results]))
    print("-" * 42)
    print(f"  {'macro':<13} {'':>11} {macro_test:>13.4f}")


if __name__ == "__main__":
    main()
