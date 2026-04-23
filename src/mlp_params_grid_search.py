"""
mlp_params_grid_search.py

Per-dataset MLP hyperparameter search using query embedding vectors
(~1024 dimensions from BAAI/bge-m3) as input features.

Method mirrors strong_signal_params_grid_search.py:
  - 85 % train+dev  (10-fold 80/20 CV to select hyperparameters)
  - 15 % test       (held out; evaluated exactly once with best params)
  - Same 85/15 split seeds -> same test queries as all other scripts

The MLP is implemented in PyTorch with:
  - GELU activations
  - Optional BatchNorm1d after each hidden layer
  - Dropout after each activation
  - Adam optimizer with cosine LR annealing
  - MSE loss on soft routing targets in [0, 1]
  - Sigmoid output layer to constrain predictions to (0, 1)

Parallelism strategy:
  - CUDA available  : sequential outer loop (n_jobs=1) — one model on GPU at a time
  - CPU only        : parallel outer loop (n_jobs from config)

Grid: 96 combinations per dataset.
Note: torch.manual_seed is NOT called inside parallel workers to avoid race
conditions on the global RNG. Model weights are randomly initialised; the 10-fold
CV averages out initialisation variance.  The final model is seeded explicitly.

Output:
  data/results/mlp_per_dataset_best_params.csv

Usage:
    python src/mlp_params_grid_search.py
"""

import csv
import itertools
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import ensure_dir, get_config_path, load_config
from src.weak_signal_model_grid_search import (
    dataset_seed_offset,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
)
from src.strong_signal_model_grid_search import load_embeddings_for_dataset


# ── MLP model ─────────────────────────────────────────────────────────────────

class AlphaMLP(nn.Module):
    """
    Feedforward network mapping a query embedding to α ∈ (0, 1).

    Architecture per hidden layer:
        Linear  →  [BatchNorm1d]  →  GELU  →  Dropout
    Final layer:
        Linear  →  Sigmoid
    """

    def __init__(self, input_dim: int, hidden_sizes: list,
                 dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_mlp(model: AlphaMLP,
               X_tr: np.ndarray, y_tr: np.ndarray,
               lr: float, weight_decay: float,
               n_epochs: int, batch_size: int,
               device: torch.device) -> None:
    """Train model in-place with Adam + cosine LR annealing and MSE loss."""
    X_t = torch.from_numpy(X_tr).float().to(device)
    y_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    # Cap batch size so it never exceeds the dataset size.
    bs = max(1, min(batch_size, len(X_t)))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()


# ── wRRF scoring ──────────────────────────────────────────────────────────────

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


# ── Single-combo CV evaluation ────────────────────────────────────────────────

def _eval_combo(params, X_traindev, y_traindev, traindev_qids,
                fold_indices, bm25_res, dense_res, qrels,
                seed, rrf_k, ndcg_k, device_str, n_epochs, batch_size):
    """10-fold CV on train+dev; return mean wRRF NDCG@k across folds."""
    device    = torch.device(device_str)
    input_dim = X_traindev.shape[1]
    fold_scores = []

    for fi, (tr_idx, va_idx) in enumerate(fold_indices):
        X_tr, y_tr = X_traindev[tr_idx], y_traindev[tr_idx]
        X_va       = X_traindev[va_idx]
        va_qids    = [traindev_qids[i] for i in va_idx]

        mu, sigma = zscore_stats(X_tr)
        X_tr_z    = (X_tr - mu) / sigma
        X_va_z    = (X_va - mu) / sigma

        mdl = AlphaMLP(
            input_dim    = input_dim,
            hidden_sizes = params["hidden_sizes"],
            dropout      = params["dropout"],
            use_batchnorm= params["use_batchnorm"],
        ).to(device)

        _train_mlp(mdl, X_tr_z, y_tr,
                   lr           = params["learning_rate"],
                   weight_decay = params["weight_decay"],
                   n_epochs     = n_epochs,
                   batch_size   = batch_size,
                   device       = device)

        mdl.eval()
        with torch.no_grad():
            alphas = mdl(
                torch.from_numpy(X_va_z).float().to(device)
            ).cpu().numpy()

        fold_scores.append(
            _wrrf_ndcg(np.clip(alphas, 0.0, 1.0), va_qids,
                       bm25_res, dense_res, qrels, rrf_k, ndcg_k)
        )

    return float(np.mean(fold_scores))


# ── Grid construction ─────────────────────────────────────────────────────────

def _build_grid(grid_cfg):
    keys   = sorted(grid_cfg.keys())
    values = [grid_cfg[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in itertools.product(*values)]


# ── CSV serialisation helper for list-valued params ───────────────────────────

def _fmt_param(key, value):
    """Serialise a param value; lists (hidden_sizes) become '512-256' strings."""
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    return value


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg  = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    cuda_available = torch.cuda.is_available()
    device_str     = "cuda" if cuda_available else "cpu"
    print(f"Device     : {device_str}")

    dataset_names = cfg["datasets"]
    ndcg_k        = int(cfg["benchmark"]["ndcg_k"])
    rrf_k         = int(cfg["benchmark"]["rrf"]["k"])

    grid_cfg   = dict(cfg["mlp_params_grid"])
    n_jobs     = int(grid_cfg.pop("n_jobs",          -1))
    n_folds    = int(grid_cfg.pop("n_folds",         10))
    test_frac  = float(grid_cfg.pop("test_fraction", 0.15))
    n_epochs   = int(grid_cfg.pop("n_epochs",       200))
    batch_size = int(grid_cfg.pop("batch_size",      64))

    # Sequential on CUDA (one model on device at a time); parallel on CPU.
    n_outer = 1 if cuda_available else n_jobs

    combos     = _build_grid(grid_cfg)
    param_keys = sorted(grid_cfg.keys())

    print(f"Input      : query embedding vectors (full dimensionality)")
    print(f"Grid size  : {len(combos)} combinations per dataset")
    print(f"CV folds   : {n_folds}")
    print(f"Test hold  : {int(test_frac * 100)} %")
    print(f"Epochs     : {n_epochs}  |  Batch size: {batch_size}")
    print(f"n_jobs     : {n_outer}  "
          f"({'CUDA sequential' if cuda_available else 'CPU parallel'})")

    # ── Load datasets ──────────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_embeddings_for_dataset(
            ds, cfg, torch.device(device_str)
        )
        n_q     = len(datasets_data[ds]["qids"])
        emb_dim = datasets_data[ds]["X"].shape[1]
        print(f"  {n_q} queries, embedding dim = {emb_dim}")

    if cuda_available:
        torch.cuda.empty_cache()

    # ── CSV setup (incremental — one row appended per completed dataset) ───────
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path   = os.path.join(results_folder, "mlp_per_dataset_best_params.csv")
    csv_header = (["dataset"] + param_keys
                  + [f"cv_ndcg@{ndcg_k}", f"test_ndcg@{ndcg_k}"])
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8", newline="") as _f:
            csv.writer(_f).writerow(csv_header)

    # ── Per-dataset search ─────────────────────────────────────────────────────
    results = []

    for ds_name in dataset_names:
        ds        = datasets_data[ds_name]
        X         = ds["X"]
        y         = ds["y"]
        qids      = ds["qids"]
        bm25_res  = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels     = ds["qrels"]
        n_q       = len(qids)

        # Identical 85/15 split to every other script in this project.
        split_seed   = seed + dataset_seed_offset(ds_name)
        rng_split    = np.random.RandomState(split_seed)
        perm         = rng_split.permutation(n_q)
        n_test       = max(1, int(test_frac * n_q))
        n_traindev   = n_q - n_test
        traindev_idx = perm[:n_traindev]
        test_idx     = perm[n_traindev:]

        # Precompute 10 CV fold splits on the traindev portion.
        fold_indices = []
        for fi in range(n_folds):
            rng  = np.random.RandomState(
                seed + fi * 1000 + dataset_seed_offset(ds_name)
            )
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
        print(f"CV folds   : {n_folds} x 80/20 splits of train+dev")
        print(f"Grid       : {len(combos)} combinations")
        print(f"{'=' * 60}")

        # ── Grid search (parallel across combos) ───────────────────────────────
        cv_scores = list(tqdm(
            Parallel(n_jobs=n_outer, prefer="threads", return_as="generator")(
                delayed(_eval_combo)(
                    p, X_traindev, y_traindev, traindev_qids,
                    fold_indices, bm25_res, dense_res, qrels,
                    seed, rrf_k, ndcg_k, device_str, n_epochs, batch_size,
                )
                for p in combos
            ),
            total=len(combos),
            desc=f"  [{ds_name}] MLP grid",
            dynamic_ncols=True,
        ))

        best_idx    = int(np.argmax(cv_scores))
        best_params = combos[best_idx]
        best_cv     = cv_scores[best_idx]

        print(f"  Best CV NDCG@{ndcg_k} = {best_cv:.4f}  |  params: {best_params}")

        # ── Final model: full train+dev → test (evaluated once) ────────────────
        mu, sigma    = zscore_stats(X_traindev)
        X_traindev_z = (X_traindev      - mu) / sigma
        X_te_z       = (X[test_idx]     - mu) / sigma

        torch.manual_seed(seed)   # seed the final model for reproducibility
        final_mdl = AlphaMLP(
            input_dim    = X_traindev.shape[1],
            hidden_sizes = best_params["hidden_sizes"],
            dropout      = best_params["dropout"],
            use_batchnorm= best_params["use_batchnorm"],
        ).to(torch.device(device_str))

        _train_mlp(final_mdl, X_traindev_z, y_traindev,
                   lr           = best_params["learning_rate"],
                   weight_decay = best_params["weight_decay"],
                   n_epochs     = n_epochs,
                   batch_size   = batch_size,
                   device       = torch.device(device_str))

        final_mdl.eval()
        with torch.no_grad():
            test_alphas = np.clip(
                final_mdl(
                    torch.from_numpy(X_te_z).float().to(torch.device(device_str))
                ).cpu().numpy(),
                0.0, 1.0,
            )

        test_ndcg = _wrrf_ndcg(
            test_alphas, te_qids, bm25_res, dense_res, qrels, rrf_k, ndcg_k
        )
        print(f"  Test NDCG@{ndcg_k}  = {test_ndcg:.4f}")

        results.append({
            "dataset":     ds_name,
            "params":      best_params,
            "cv_ndcg10":   best_cv,
            "test_ndcg10": test_ndcg,
        })

        # Append row immediately so progress survives a job kill.
        with open(csv_path, "a", encoding="utf-8", newline="") as _f:
            csv.writer(_f).writerow(
                [ds_name]
                + [_fmt_param(k, best_params[k]) for k in param_keys]
                + [f"{best_cv:.6f}", f"{test_ndcg:.6f}"]
            )
        print(f"  Appended row -> {csv_path}")

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
