"""
strong_signal_model_grid_search.py

Grid search across 8 model families to identify the best strong-signal
router for query-adaptive hybrid retrieval.

Input: raw query embedding vectors from the pre-trained dense encoder
(BAAI/bge-m3, ~1024 dimensions per query), as opposed to the 15
hand-crafted weak-signal features used in weak_signal_model_grid_search.py.

Models tested:
  Ridge Regression, ElasticNet, SVR (RBF kernel), KNN,
  MLP, XGBoost, Random Forest, Extra Trees

For each (model, hyperparameter) combination the script runs 10 repeated
random 80/20 train/test splits on every configured dataset, computes
wRRF NDCG@10 on each test fold, and reports the macro average across all
datasets.

Fold indices are identical to those in weak_signal_model_grid_search.py
(same seed formula) so the two experiments are directly comparable.

The top-N results are written to:
  data/results/strong_signal_grid_search_top100.csv

Usage:
    python src/strong_signal_model_grid_search.py
"""

import csv
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import (
    ensure_dir,
    get_config_path,
    load_config,
    load_pickle,
    model_short_name,
)
from src.weak_signal_model_grid_search import (
    dataset_seed_offset,
    load_dataset_for_grid_search,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_embeddings_for_dataset(dataset_name, cfg, device):
    """Return a dataset dict whose X is the query embedding matrix.

    Reuses load_dataset_for_grid_search to obtain:
      - y         : soft labels (same formula as weak-signal)
      - qids      : sorted query IDs (same ordering)
      - bm25_results, dense_results, qrels : for wRRF evaluation

    Then replaces X with the query vectors loaded from query_vectors.pt,
    aligned to the sorted qids order.
    """
    # ── Reuse existing pipeline for y, qids, retrieval results ───────────────
    ds = load_dataset_for_grid_search(dataset_name, cfg, device)

    # ── Load query embedding vectors ──────────────────────────────────────────
    short_model  = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir = os.path.join(processed_root, short_model, dataset_name)

    query_vecs_pt  = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl  = os.path.join(ds_dir, "query_ids.pkl")

    # Load to CPU regardless of where the tensor was saved.
    q_vecs = torch.load(query_vecs_pt, weights_only=True).cpu()
    q_ids  = load_pickle(query_ids_pkl)           # list[str], encoded order

    if len(q_ids) != q_vecs.shape[0]:
        raise RuntimeError(
            f"[{dataset_name}] query_ids length {len(q_ids)} != "
            f"query_vectors rows {q_vecs.shape[0]}"
        )

    # Build lookup and align to sorted qids used throughout the pipeline.
    qid_to_idx = {qid: i for i, qid in enumerate(q_ids)}
    missing = [qid for qid in ds["qids"] if qid not in qid_to_idx]
    if missing:
        raise RuntimeError(
            f"[{dataset_name}] {len(missing)} query IDs in qids not found "
            f"in query_vectors.pt: {missing[:5]}"
        )

    # Vectorised reordering: build index list then use tensor fancy indexing
    # rather than a Python loop that calls .numpy() on each row individually.
    indices = [qid_to_idx[qid] for qid in ds["qids"]]
    X_emb = q_vecs[indices].numpy().astype(np.float32)

    return {
        "X":            X_emb,
        "y":            ds["y"],
        "qids":         ds["qids"],
        "bm25_results": ds["bm25_results"],
        "dense_results": ds["dense_results"],
        "qrels":        ds["qrels"],
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def create_model(model_name, params, seed):
    """Instantiate one model.

    All models use n_jobs=1 (where supported) to avoid nested parallelism —
    the outer Parallel loop already saturates available cores.
    SVR uses cache_size=500 MB to speed up the kernel matrix computation.
    """

    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=params["alpha"], max_iter=3000)

    if model_name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=3000,
            random_state=seed,
        )

    if model_name == "svr":
        from sklearn.svm import SVR
        return SVR(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            epsilon=params["epsilon"],
            cache_size=500,
        )

    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric="euclidean",  # z-scored euclidean is the appropriate distance
            algorithm="auto",
            n_jobs=1,
        )

    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation="relu",
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            verbosity=0,
            random_state=seed,
            n_jobs=1,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=1,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_alpha(model, X):
    """Return alpha predictions clipped to [0, 1].

    All models in this grid are regressors; no predict_proba path needed.
    """
    return np.clip(model.predict(X).astype(np.float32), 0.0, 1.0)


# ── Single-combination evaluation ─────────────────────────────────────────────

def evaluate_combination(
    model_name, params, datasets_data, fold_indices, seed, rrf_k, ndcg_k,
):
    """Train and evaluate one (model, params) tuple across all datasets
    and folds. Returns a result dict with macro and per-dataset NDCG@10."""
    per_ds = {}

    for ds_name, ds in datasets_data.items():
        X    = ds["X"]          # shape [N, emb_dim]
        y    = ds["y"]
        qids = ds["qids"]
        bm25_res  = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels     = ds["qrels"]

        fold_ndcgs = []
        for train_idx, test_idx in fold_indices[ds_name]:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te       = X[test_idx]
            test_qids  = [qids[i] for i in test_idx]

            # Z-score normalisation: fit on train split, apply to test.
            mu, sigma = zscore_stats(X_tr)
            X_tr_z = (X_tr - mu) / sigma
            X_te_z = (X_te - mu) / sigma

            try:
                mdl = create_model(model_name, params, seed)
                mdl.fit(X_tr_z, y_tr)
                alphas = predict_alpha(mdl, X_te_z)
            except (ValueError, np.linalg.LinAlgError):
                # Degenerate fold; fall back to equal weighting (static RRF).
                alphas = np.full(len(test_qids), 0.5, dtype=np.float32)

            # wRRF NDCG@10 on test queries
            ndcgs = []
            for qid, alpha in zip(test_qids, alphas):
                alpha    = float(alpha)
                bm_pairs = bm25_res.get(qid, [])
                de_pairs = dense_res.get(qid, [])
                bm_rank  = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
                de_rank  = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
                bm_miss  = len(bm_pairs) + 1
                de_miss  = len(de_pairs) + 1
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
        "model":        model_name,
        "params":       params,
        "macro_ndcg10": macro,
        "per_dataset":  per_ds,
    }


# ── Grid construction ─────────────────────────────────────────────────────────

def build_param_grid(grid_cfg):
    """Return all hyperparameter dicts for one model (full Cartesian product)."""
    keys   = sorted(grid_cfg.keys())
    values = [grid_cfg[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from sklearn.exceptions import ConvergenceWarning
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

    gs_cfg    = cfg["strong_signal_model_grid_search"]
    n_folds   = int(gs_cfg["n_folds"])
    train_frac = float(gs_cfg["train_fraction"])
    top_n     = int(gs_cfg["top_n"])
    n_jobs    = int(gs_cfg.get("n_jobs", -1))

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    emb_dim = None
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_embeddings_for_dataset(ds, cfg, device)
        n_q = len(datasets_data[ds]["qids"])
        emb_dim = datasets_data[ds]["X"].shape[1]
        print(f"  {n_q} queries, embedding dim = {emb_dim}")

    print(f"\nEmbedding dimension: {emb_dim}")

    # Free GPU memory after retrieval (reused from weak signal loading).
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Precompute fold indices (identical formula to weak_signal script) ──────
    # Using the same seeds ensures folds match across the two experiments,
    # making weak-signal vs strong-signal scores directly comparable.
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

    # ── Build grids ────────────────────────────────────────────────────────────
    model_cfgs = gs_cfg["models"]
    model_order = [
        "ridge", "elasticnet", "svr", "knn",
        "mlp", "xgboost", "random_forest", "extra_trees",
    ]
    model_order = [m for m in model_order if m in model_cfgs]

    # Precompute all grids once — used for the count summary and evaluation.
    all_grids    = {m: build_param_grid(model_cfgs[m]) for m in model_order}
    total_combos = sum(len(g) for g in all_grids.values())
    print(f"\n=== Grid search: {total_combos} total combinations ===")
    for m in model_order:
        print(f"  {m:<20}: {len(all_grids[m])} combinations")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    all_results = []
    t0_total = time.time()

    for model_name in model_order:
        combos   = all_grids[model_name]
        n_combos = len(combos)
        print(f"\n[{model_name}] {n_combos} combinations ...", flush=True)
        t0 = time.time()

        batch = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                delayed(evaluate_combination)(
                    model_name, p, datasets_data, fold_indices, seed, rrf_k, ndcg_k,
                )
                for p in combos
            ),
            total=n_combos,
            desc=f"  [{model_name}]",
            dynamic_ncols=True,
        ))

        all_results.extend(batch)
        best    = max(batch, key=lambda r: r["macro_ndcg10"])
        elapsed = time.time() - t0
        print(
            f"  Done in {elapsed:.1f}s | "
            f"Best macro NDCG@10 = {best['macro_ndcg10']:.4f}"
        )

    total_time = time.time() - t0_total
    print(f"\nTotal grid search time: {total_time:.1f}s")

    # ── Sort and save ──────────────────────────────────────────────────────────
    all_results.sort(key=lambda r: r["macro_ndcg10"], reverse=True)
    top = all_results[:top_n]

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path = os.path.join(results_folder, "strong_signal_grid_search_top100.csv")

    ds_cols = list(dataset_names)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "params", "macro_ndcg10"] + ds_cols)
        for rank, r in enumerate(top, 1):
            writer.writerow([
                rank,
                r["model"],
                json.dumps(r["params"], sort_keys=True),
                f"{r['macro_ndcg10']:.6f}",
                *[f"{r['per_dataset'].get(ds, 0.0):.6f}" for ds in ds_cols],
            ])

    print(f"\nSaved top {len(top)} results to {csv_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'Model':<20} {'Macro NDCG@10':>14}")
    print("-" * 41)
    for rank, r in enumerate(top[:20], 1):
        print(f"{rank:<5} {r['model']:<20} {r['macro_ndcg10']:>14.4f}")
    if len(top) > 20:
        print(f"  ... ({len(top) - 20} more rows in CSV)")


if __name__ == "__main__":
    main()
