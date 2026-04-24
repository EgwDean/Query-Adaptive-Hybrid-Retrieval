"""
meta_learner_moe_grid_search.py

Stacking MoE meta-learner that combines weak-signal and strong-signal
XGBoost router predictions into a single ensemble alpha.

Zero-leakage meta-dataset construction
---------------------------------------
  Traindev queries (85 %):
    Standard 10-fold CV on base models — each traindev query's
    alpha_weak / alpha_strong is produced by a model that never saw it.
  Test queries (15 %):
    Both base models retrained on the full traindev, then applied to test.
    No traindev query ever contaminates test predictions.

Meta-learner grid search
--------------------------
  10-round Monte Carlo CV on the traindev meta-features.
  Each round: random split where dev ≈ 15 % of total (= 15/85 of traindev).
  Metric: mean NDCG@10 on dev queries using actual BM25 / dense results.
  4 model families, 48 hyperparameter combinations total:
    Ridge       (6)  — features: [aw, as, |aw−as|]
    ElasticNet (12)  — features: [aw, as, |aw−as|]
    XGBoost    (18)  — features: [aw, as]          (tree can infer difference)
    SVR        (12)  — features: [aw, as, |aw−as|]

Final model: best combo retrained on full traindev meta-features,
evaluated once on test meta-features.

Outputs
--------
  data/results/meta_learner_dataset.csv          (cached meta-features)
  data/results/meta_learner_best_params.csv      (best model + params)
  data/results/meta_learner_retrieval_comparison.csv
  data/results/meta_learner_retrieval_comparison.png  (6-method bar chart)
  data/results/meta_learner_decision_boundary.png     (prediction surface)

Usage
-----
  python src/meta_learner_moe_grid_search.py
"""

import csv
import itertools
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
import xgboost as xgb

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

# Model families that receive the 3-feature input [aw, as, |aw−as|].
# XGBoost uses only [aw, as] — the tree can split on the difference implicitly.
LINEAR_MODELS = {"ridge", "elasticnet", "svr"}

DS_PALETTE = {
    "scifact":  "#4878D0",
    "nfcorpus": "#EE854A",
    "arguana":  "#6ACC65",
    "fiqa":     "#D65F5F",
    "scidocs":  "#B47CC7",
}


# ── Retrieval helpers ─────────────────────────────────────────────────────────

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


# ── Meta-feature construction ─────────────────────────────────────────────────

def _meta_features(aw, as_, model_name):
    """
    Linear models: [alpha_weak, alpha_strong, |alpha_weak − alpha_strong|]
    Tree models:   [alpha_weak, alpha_strong]
    """
    aw  = np.asarray(aw,  dtype=np.float32)
    as_ = np.asarray(as_, dtype=np.float32)
    if model_name in LINEAR_MODELS:
        return np.column_stack([aw, as_, np.abs(aw - as_)])
    return np.column_stack([aw, as_])


# ── Meta-model training / prediction ─────────────────────────────────────────

def _fit_meta(model_name, params, X, y):
    if model_name == "ridge":
        mdl = Ridge(alpha=float(params["alpha"]))
    elif model_name == "elasticnet":
        mdl = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
        )
    elif model_name == "svr":
        mdl = SVR(
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            kernel="rbf",
        )
    elif model_name == "xgboost":
        mdl = xgb.XGBRegressor(
            objective       = "binary:logistic",
            eval_metric     = "logloss",
            tree_method     = "hist",
            device          = "cpu",     # always CPU — called from joblib workers
            verbosity       = 0,
            random_state    = 42,
            n_jobs          = 1,
            subsample       = 1.0,
            colsample_bytree= 1.0,
            min_child_weight= 1,
            gamma           = 0.0,
            n_estimators    = int(params["n_estimators"]),
            max_depth       = int(params["max_depth"]),
            learning_rate   = float(params["learning_rate"]),
        )
    else:
        raise ValueError(f"Unknown meta-model: {model_name}")
    mdl.fit(X, y)
    return mdl


def _pred_meta(mdl, X):
    return np.clip(mdl.predict(X), 0.0, 1.0).astype(np.float32)


# ── OOF base predictions (zero leakage on traindev) ──────────────────────────

def _build_oof_predictions(
    X_weak_td, X_strong_td, y_td,
    n_oof_folds, weak_params, strong_params, seed, xgb_device,
):
    """
    Standard k-fold OOF for both base XGBoost models.
    Every traindev query's prediction is made by a model that never saw it.
    Returns (alpha_weak_oof, alpha_strong_oof) — shape (n_td,).
    """
    n    = len(y_td)
    rng  = np.random.RandomState(seed + 7777)
    perm = rng.permutation(n)
    # array_split produces n_oof_folds chunks; last chunk may be 1 larger
    folds = np.array_split(perm, n_oof_folds)

    aw_oof = np.empty(n, dtype=np.float32)
    as_oof = np.empty(n, dtype=np.float32)

    for fi, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(n_oof_folds) if j != fi])

        # ── weak base model ──────────────────────────────────────────────────
        mu_w, sig_w = zscore_stats(X_weak_td[tr_idx])
        Xw_tr = (X_weak_td[tr_idx] - mu_w) / sig_w
        Xw_va = (X_weak_td[val_idx] - mu_w) / sig_w

        mdl_w = xgb.XGBRegressor(
            objective    = "binary:logistic",
            eval_metric  = "logloss",
            tree_method  = "hist",
            device       = xgb_device,
            verbosity    = 0,
            random_state = seed,
            n_jobs       = 1 if xgb_device == "cuda" else -1,
            **weak_params,
        )
        mdl_w.fit(Xw_tr, y_td[tr_idx])
        aw_oof[val_idx] = np.clip(mdl_w.predict(Xw_va), 0.0, 1.0)

        # ── strong base model ────────────────────────────────────────────────
        mu_s, sig_s = zscore_stats(X_strong_td[tr_idx])
        Xs_tr = (X_strong_td[tr_idx] - mu_s) / sig_s
        Xs_va = (X_strong_td[val_idx] - mu_s) / sig_s

        mdl_s = xgb.XGBRegressor(
            objective    = "binary:logistic",
            eval_metric  = "logloss",
            tree_method  = "hist",
            device       = xgb_device,
            verbosity    = 0,
            random_state = seed,
            n_jobs       = 1 if xgb_device == "cuda" else -1,
            **strong_params,
        )
        mdl_s.fit(Xs_tr, y_td[tr_idx])
        as_oof[val_idx] = np.clip(mdl_s.predict(Xs_va), 0.0, 1.0)

        print(f"  OOF fold {fi + 1}/{n_oof_folds} done "
              f"(val={len(val_idx)}, tr={len(tr_idx)})")

    return aw_oof, as_oof


def _train_base_models(
    X_weak_td, X_strong_td, y_td,
    weak_params, strong_params, seed, xgb_device,
):
    """
    Retrain both base models on the full traindev.
    Returns (mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s) for test-set prediction.
    """
    mu_w, sig_w = zscore_stats(X_weak_td)
    mdl_w = xgb.XGBRegressor(
        objective    = "binary:logistic",
        eval_metric  = "logloss",
        tree_method  = "hist",
        device       = xgb_device,
        verbosity    = 0,
        random_state = seed,
        n_jobs       = 1 if xgb_device == "cuda" else -1,
        **weak_params,
    )
    mdl_w.fit((X_weak_td - mu_w) / sig_w, y_td)

    mu_s, sig_s = zscore_stats(X_strong_td)
    mdl_s = xgb.XGBRegressor(
        objective    = "binary:logistic",
        eval_metric  = "logloss",
        tree_method  = "hist",
        device       = xgb_device,
        verbosity    = 0,
        random_state = seed,
        n_jobs       = 1 if xgb_device == "cuda" else -1,
        **strong_params,
    )
    mdl_s.fit((X_strong_td - mu_s) / sig_s, y_td)

    return mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def _save_meta_dataset(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ds_name", "qid", "split",
                        "alpha_weak", "alpha_strong", "alpha_gt"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "ds_name":      r["ds_name"],
                "qid":          r["qid"],
                "split":        r["split"],
                "alpha_weak":   f"{r['alpha_weak']:.6f}",
                "alpha_strong": f"{r['alpha_strong']:.6f}",
                "alpha_gt":     f"{r['alpha_gt']:.6f}",
            })


def _load_meta_dataset(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "ds_name":      r["ds_name"],
                "qid":          r["qid"],
                "split":        r["split"],
                "alpha_weak":   float(r["alpha_weak"]),
                "alpha_strong": float(r["alpha_strong"]),
                "alpha_gt":     float(r["alpha_gt"]),
            })
    return rows


# ── Monte Carlo CV fold indices ───────────────────────────────────────────────

def _mc_fold_indices(n, dev_frac_of_n, n_rounds, seed):
    """
    n_rounds random splits of n items.
    Each split: dev = first dev_frac_of_n fraction, train = remainder.
    Returns list of (train_idx, dev_idx) arrays.
    """
    n_dev = max(1, round(dev_frac_of_n * n))
    folds = []
    for fi in range(n_rounds):
        rng  = np.random.RandomState(seed + fi * 1000)
        perm = rng.permutation(n)
        folds.append((perm[n_dev:], perm[:n_dev]))   # (train_idx, dev_idx)
    return folds


# ── Grid search: single combo evaluation ─────────────────────────────────────

def _eval_combo(
    model_name, params,
    td_aw, td_as, td_gt,
    td_qids, td_ds_names,
    mc_folds, retrieval_data, rrf_k, ndcg_k,
):
    """
    10-round MC CV for one (model_name, params) combination.
    Returns the mean dev NDCG@k across all rounds.
    """
    round_scores = []
    for tr_idx, dv_idx in mc_folds:
        X_tr = _meta_features(td_aw[tr_idx], td_as[tr_idx], model_name)
        X_dv = _meta_features(td_aw[dv_idx], td_as[dv_idx], model_name)
        y_tr = td_gt[tr_idx]

        try:
            mdl   = _fit_meta(model_name, params, X_tr, y_tr)
            preds = _pred_meta(mdl, X_dv)
        except Exception:
            round_scores.append(0.0)
            continue

        ndcgs = []
        for alpha, i in zip(preds, dv_idx):
            qid = td_qids[i]
            ds  = td_ds_names[i]
            rd  = retrieval_data[ds]
            ndcgs.append(_wrrf_single(
                alpha, qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))
        round_scores.append(float(np.mean(ndcgs)) if ndcgs else 0.0)

    return float(np.mean(round_scores))


# ── Build combo list from grid config ─────────────────────────────────────────

def _build_combos(grid_cfg):
    """Returns list of (model_name, param_dict) for all grid combinations."""
    combos = []

    for alpha in grid_cfg["ridge"]["alpha"]:
        combos.append(("ridge", {"alpha": alpha}))

    for alpha, l1_ratio in itertools.product(
        grid_cfg["elasticnet"]["alpha"],
        grid_cfg["elasticnet"]["l1_ratio"],
    ):
        combos.append(("elasticnet", {"alpha": alpha, "l1_ratio": l1_ratio}))

    for n_est, depth, lr in itertools.product(
        grid_cfg["xgboost"]["n_estimators"],
        grid_cfg["xgboost"]["max_depth"],
        grid_cfg["xgboost"]["learning_rate"],
    ):
        combos.append(("xgboost", {
            "n_estimators": n_est,
            "max_depth":    depth,
            "learning_rate": lr,
        }))

    for C, epsilon in itertools.product(
        grid_cfg["svr"]["C"],
        grid_cfg["svr"]["epsilon"],
    ):
        combos.append(("svr", {"C": C, "epsilon": epsilon}))

    return combos


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_comparison_plot(rows, ndcg_k, out_path):
    methods = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    labels  = ["BM25", "Dense", "Static RRF (α=0.5)",
               "wRRF (weak)", "wRRF (strong)", "MoE Meta-Learner"]
    colors  = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#B47CC7", "#956CB4"]

    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    width   = 0.12
    offsets = np.linspace(-(len(methods) - 1) / 2,
                           (len(methods) - 1) / 2,
                           len(methods)) * width
    by_method = {m: [r[m] for r in rows] for m in methods}

    fig, ax = plt.subplots(figsize=(18, 6))
    for method, label, color, offset in zip(methods, labels, colors, offsets):
        bars = ax.bar(x + offset, by_method[method], width,
                      label=label, color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(f"MoE Meta-Learner Retrieval Comparison — NDCG@{ndcg_k}", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    all_scores = [s for ss in by_method.values() for s in ss]
    ax.set_ylim(0, min(1.0, max(all_scores) + 0.12))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {out_path}")


def _save_decision_boundary_plot(
    final_mdl, model_name,
    td_aw, td_as, td_ds_names,
    te_aw, te_as, te_ds_names,
    out_path,
):
    """
    Smooth contourf of the meta-learner's predicted alpha over the
    (alpha_strong, alpha_weak) unit square, with actual queries overlaid.
    Large markers = test queries; small = traindev queries.
    Colours encode dataset identity.
    """
    grid_size = 100
    aw_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    as_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    AW, AS  = np.meshgrid(aw_vals, as_vals)   # both (grid_size, grid_size)

    X_grid = _meta_features(AW.ravel(), AS.ravel(), model_name)
    Z      = np.clip(final_mdl.predict(X_grid), 0.0, 1.0).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(AS, AW, Z, levels=30, cmap="RdYlBu_r", alpha=0.85)
    plt.colorbar(cf, ax=ax,
                 label="Predicted α   (1 = prefer BM25 / sparse,  0 = prefer Dense)")

    # Scatter queries, coloured by dataset
    for ds_name, color in DS_PALETTE.items():
        mask_td = np.array([d == ds_name for d in td_ds_names])
        mask_te = np.array([d == ds_name for d in te_ds_names])
        if mask_td.any():
            ax.scatter(
                td_as[mask_td], td_aw[mask_td],
                c=color, s=12, alpha=0.30, edgecolors="none", zorder=3,
            )
        if mask_te.any():
            ax.scatter(
                te_as[mask_te], te_aw[mask_te],
                c=color, s=40, alpha=0.85,
                edgecolors="k", linewidths=0.4,
                label=ds_name, zorder=4,
            )

    ax.set_xlabel("α_strong  (strong-signal XGBoost prediction)", fontsize=10)
    ax.set_ylabel("α_weak   (weak-signal XGBoost prediction)",  fontsize=10)
    ax.set_title(
        f"Meta-Learner Prediction Surface — {model_name}\n"
        "(large markers = test queries;  small = traindev)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right", title="dataset")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Decision boundary plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg  = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    cuda_available = torch.cuda.is_available()
    device         = torch.device("cuda" if cuda_available else "cpu")
    xgb_device     = "cuda" if cuda_available else "cpu"
    # Grid search always uses CPU workers (joblib loky + CUDA = unsafe).
    # When CUDA is present, use n_jobs=1 to avoid forking alongside the GPU.
    n_jobs = 1 if cuda_available else int(
        cfg.get("meta_learner_moe", {}).get("n_jobs", -1)
    )
    print(f"Device: {device}  |  grid-search n_jobs: {n_jobs}")

    dataset_names = cfg["datasets"]
    ndcg_k        = int(cfg["benchmark"]["ndcg_k"])
    rrf_k         = int(cfg["benchmark"]["rrf"]["k"])

    exp_cfg          = cfg["meta_learner_moe"]
    n_queries        = int(exp_cfg["n_queries"])
    trunc_seed       = int(exp_cfg["truncation_seed"])
    test_frac        = float(exp_cfg["test_fraction"])
    dev_frac_of_td   = float(exp_cfg["dev_fraction_of_traindev"])
    n_cv_rounds      = int(exp_cfg["n_cv_rounds"])
    n_oof_folds      = int(exp_cfg["n_oof_folds"])
    grid_cfg         = exp_cfg["grid"]

    # Base model best params (from small_data_experiment grid search)
    bp            = cfg["small_data_experiment"]["best_params"]
    weak_params   = {k: v for k, v in bp["weak_signal"].items()
                     if k != "cv_ndcg_at_10"}
    strong_params = {k: v for k, v in bp["strong_signal"].items()
                     if k != "cv_ndcg_at_10"}

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    cache_path = os.path.join(results_folder, "meta_learner_dataset.csv")

    # ── Load all datasets ─────────────────────────────────────────────────────
    print("\n=== Loading datasets ===")
    all_raw        = {}
    retrieval_data = {}

    for ds_name in dataset_names:
        print(f"  {ds_name} ...")
        wd = load_dataset_for_grid_search(ds_name, cfg, device)
        sd = load_embeddings_for_dataset(ds_name, cfg, device)
        all_raw[ds_name] = {"wd": wd, "sd": sd}
        retrieval_data[ds_name] = {
            "bm25_results":  wd["bm25_results"],
            "dense_results": wd["dense_results"],
            "qrels":         wd["qrels"],
        }

    if cuda_available:
        torch.cuda.empty_cache()

    # ── Truncate and split ────────────────────────────────────────────────────
    print("\n=== Truncating and splitting datasets ===")

    X_weak_td_parts,   X_strong_td_parts   = [], []
    y_td_parts,        qids_td,   ds_td    = [], [], []
    X_weak_te_parts,   X_strong_te_parts   = [], []
    y_te_parts,        qids_te,   ds_te    = [], [], []

    for ds_name in dataset_names:
        wd = all_raw[ds_name]["wd"]
        sd = all_raw[ds_name]["sd"]

        n_q_full = len(wd["qids"])
        n_use    = min(n_queries, n_q_full)

        rng_trunc = np.random.RandomState(trunc_seed + dataset_seed_offset(ds_name))
        trunc_idx = np.sort(rng_trunc.choice(n_q_full, size=n_use, replace=False))

        X_weak_full   = wd["X"][:, ACTIVE_COLS][trunc_idx]
        X_strong_full = sd["X"][trunc_idx]
        y_full        = wd["y"][trunc_idx]
        qids_full     = [wd["qids"][i] for i in trunc_idx]

        # Same split seed formula as all other scripts
        split_seed = seed + dataset_seed_offset(ds_name)
        rng_split  = np.random.RandomState(split_seed)
        perm       = rng_split.permutation(n_use)
        n_test     = max(1, int(test_frac * n_use))
        n_td_ds    = n_use - n_test
        td_idx, te_idx = perm[:n_td_ds], perm[n_td_ds:]

        X_weak_td_parts.append(X_weak_full[td_idx])
        X_strong_td_parts.append(X_strong_full[td_idx])
        y_td_parts.append(y_full[td_idx])
        qids_td.extend(qids_full[i] for i in td_idx)
        ds_td.extend([ds_name] * n_td_ds)

        X_weak_te_parts.append(X_weak_full[te_idx])
        X_strong_te_parts.append(X_strong_full[te_idx])
        y_te_parts.append(y_full[te_idx])
        qids_te.extend(qids_full[i] for i in te_idx)
        ds_te.extend([ds_name] * n_test)

        print(f"  {ds_name}: {n_td_ds} traindev | {n_test} test")

    X_weak_td   = np.vstack(X_weak_td_parts)
    X_strong_td = np.vstack(X_strong_td_parts)
    y_td        = np.concatenate(y_td_parts).astype(np.float32)
    X_weak_te   = np.vstack(X_weak_te_parts)
    X_strong_te = np.vstack(X_strong_te_parts)
    y_te        = np.concatenate(y_te_parts).astype(np.float32)
    n_td        = len(y_td)
    n_te        = len(y_te)

    print(f"\n  Merged traindev : {n_td}  |  Merged test : {n_te}")
    print(f"  Weak dim        : {X_weak_td.shape[1]}")
    print(f"  Strong dim      : {X_strong_td.shape[1]}")

    # ── Load or build meta-dataset ────────────────────────────────────────────
    if os.path.exists(cache_path):
        print(f"\n=== Loading meta-dataset from cache ===")
        print(f"  {cache_path}")
        cached = _load_meta_dataset(cache_path)

        td_by_key = {(r["ds_name"], r["qid"]): r
                     for r in cached if r["split"] == "traindev"}
        te_by_key = {(r["ds_name"], r["qid"]): r
                     for r in cached if r["split"] == "test"}

        try:
            td_aw = np.array([td_by_key[(d, q)]["alpha_weak"]
                              for d, q in zip(ds_td, qids_td)], dtype=np.float32)
            td_as = np.array([td_by_key[(d, q)]["alpha_strong"]
                              for d, q in zip(ds_td, qids_td)], dtype=np.float32)
            td_gt = np.array([td_by_key[(d, q)]["alpha_gt"]
                              for d, q in zip(ds_td, qids_td)], dtype=np.float32)
            te_aw = np.array([te_by_key[(d, q)]["alpha_weak"]
                              for d, q in zip(ds_te, qids_te)], dtype=np.float32)
            te_as = np.array([te_by_key[(d, q)]["alpha_strong"]
                              for d, q in zip(ds_te, qids_te)], dtype=np.float32)
        except KeyError as e:
            raise RuntimeError(
                f"Cache at '{cache_path}' does not match the current query split "
                f"(missing key {e}). Delete the cache file and re-run to rebuild it."
            ) from None

    else:
        print("\n=== Building meta-dataset (OOF + final base models) ===")

        # OOF predictions for traindev — zero leakage
        print("  Step 1: OOF predictions for traindev ...")
        td_aw, td_as = _build_oof_predictions(
            X_weak_td, X_strong_td, y_td,
            n_oof_folds, weak_params, strong_params, seed, xgb_device,
        )
        td_gt = y_td.copy()

        # Final base models on full traindev → test predictions
        print("  Step 2: Training final base models on full traindev ...")
        mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s = _train_base_models(
            X_weak_td, X_strong_td, y_td,
            weak_params, strong_params, seed, xgb_device,
        )
        te_aw = np.clip(
            mdl_w.predict((X_weak_te   - mu_w) / sig_w), 0.0, 1.0
        ).astype(np.float32)
        te_as = np.clip(
            mdl_s.predict((X_strong_te - mu_s) / sig_s), 0.0, 1.0
        ).astype(np.float32)

        # Cache to CSV
        rows = []
        for i, (d, q) in enumerate(zip(ds_td, qids_td)):
            rows.append({"ds_name": d, "qid": q, "split": "traindev",
                         "alpha_weak": float(td_aw[i]),
                         "alpha_strong": float(td_as[i]),
                         "alpha_gt": float(td_gt[i])})
        for i, (d, q) in enumerate(zip(ds_te, qids_te)):
            rows.append({"ds_name": d, "qid": q, "split": "test",
                         "alpha_weak": float(te_aw[i]),
                         "alpha_strong": float(te_as[i]),
                         "alpha_gt": float(y_te[i])})
        _save_meta_dataset(rows, cache_path)
        print(f"  Meta-dataset cached: {cache_path}")

    # ── MC CV fold indices for meta-learner ───────────────────────────────────
    mc_folds = _mc_fold_indices(n_td, dev_frac_of_td, n_cv_rounds, seed)
    n_train_per_fold = len(mc_folds[0][0])
    n_dev_per_fold   = len(mc_folds[0][1])
    print(f"\n  MC CV: {n_cv_rounds} rounds  "
          f"(train≈{n_train_per_fold}, dev≈{n_dev_per_fold} per round)")

    # ── Grid search ───────────────────────────────────────────────────────────
    print("\n=== Meta-learner grid search ===")
    combos = _build_combos(grid_cfg)
    print(f"  {len(combos)} combos × {n_cv_rounds} rounds = "
          f"{len(combos) * n_cv_rounds} evaluations")

    scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_eval_combo)(
            model_name, params,
            td_aw, td_as, td_gt,
            qids_td, ds_td,
            mc_folds, retrieval_data, rrf_k, ndcg_k,
        )
        for model_name, params in combos
    )

    best_idx        = int(np.argmax(scores))
    best_model_name = combos[best_idx][0]
    best_params     = combos[best_idx][1]
    best_cv_score   = float(scores[best_idx])

    print(f"\n  Best model       : {best_model_name}")
    print(f"  Best params      : {best_params}")
    print(f"  Best CV NDCG@{ndcg_k} : {best_cv_score:.4f}")

    # ── Save best params ──────────────────────────────────────────────────────
    params_csv = os.path.join(results_folder, "meta_learner_best_params.csv")
    with open(params_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params_json", f"cv_ndcg@{ndcg_k}"])
        writer.writerow([
            best_model_name,
            json.dumps(best_params),
            f"{best_cv_score:.6f}",
        ])
    print(f"  Best params saved: {params_csv}")

    # ── Retrain best model on full traindev ───────────────────────────────────
    print("\n=== Training final meta-learner on full traindev ===")
    X_td_meta = _meta_features(td_aw, td_as, best_model_name)
    X_te_meta = _meta_features(te_aw, te_as, best_model_name)
    final_mdl = _fit_meta(best_model_name, best_params, X_td_meta, td_gt)
    te_moe    = _pred_meta(final_mdl, X_te_meta)

    # ── Per-dataset test evaluation ───────────────────────────────────────────
    print("\n=== Per-dataset evaluation (test set) ===")
    comparison_rows = []

    for ds_name in dataset_names:
        mask = np.array([d == ds_name for d in ds_te])
        if not mask.any():
            print(f"  {ds_name}: no test queries — skipping")
            continue

        te_qids_ds = [qids_te[i] for i, m in enumerate(mask) if m]
        aw_ds      = te_aw[mask]
        as_ds      = te_as[mask]
        moe_ds     = te_moe[mask]
        srrf_ds    = np.full(mask.sum(), 0.5, dtype=np.float32)

        rd = retrieval_data[ds_name]
        bm25_s   = _bm25_ndcg(te_qids_ds, rd["bm25_results"], rd["qrels"], ndcg_k)
        dense_s  = _dense_ndcg(te_qids_ds, rd["dense_results"], rd["qrels"], ndcg_k)
        srrf_s   = _wrrf_ndcg(srrf_ds, te_qids_ds, rd["bm25_results"], rd["dense_results"], rd["qrels"], rrf_k, ndcg_k)
        wrrf_w_s = _wrrf_ndcg(aw_ds,   te_qids_ds, rd["bm25_results"], rd["dense_results"], rd["qrels"], rrf_k, ndcg_k)
        wrrf_s_s = _wrrf_ndcg(as_ds,   te_qids_ds, rd["bm25_results"], rd["dense_results"], rd["qrels"], rrf_k, ndcg_k)
        moe_s    = _wrrf_ndcg(moe_ds,  te_qids_ds, rd["bm25_results"], rd["dense_results"], rd["qrels"], rrf_k, ndcg_k)

        print(f"\n  {ds_name}  ({mask.sum()} test queries):")
        print(f"    BM25           : {bm25_s:.4f}")
        print(f"    Dense          : {dense_s:.4f}")
        print(f"    Static RRF     : {srrf_s:.4f}")
        print(f"    wRRF (weak)    : {wrrf_w_s:.4f}")
        print(f"    wRRF (strong)  : {wrrf_s_s:.4f}")
        print(f"    MoE            : {moe_s:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "bm25":        bm25_s,
            "dense":       dense_s,
            "static_rrf":  srrf_s,
            "wrrf_weak":   wrrf_w_s,
            "wrrf_strong": wrrf_s_s,
            "moe":         moe_s,
        })

    # Macro
    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print("Macro averages:")
    for lbl, key in [
        ("BM25",          "bm25"),
        ("Dense",         "dense"),
        ("Static RRF",    "static_rrf"),
        ("wRRF (weak)",   "wrrf_weak"),
        ("wRRF (strong)", "wrrf_strong"),
        ("MoE",           "moe"),
    ]:
        print(f"  {lbl:<16} NDCG@{ndcg_k} = {macro[key]:.4f}")

    # ── Save comparison CSV ───────────────────────────────────────────────────
    comp_csv = os.path.join(results_folder, "meta_learner_retrieval_comparison.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            f"bm25_ndcg@{ndcg_k}",       f"dense_ndcg@{ndcg_k}",
            f"static_rrf_ndcg@{ndcg_k}", f"wrrf_weak_ndcg@{ndcg_k}",
            f"wrrf_strong_ndcg@{ndcg_k}", f"moe_ndcg@{ndcg_k}",
        ])
        for r in comparison_rows:
            writer.writerow([
                r["group"],
                f"{r['bm25']:.6f}",       f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}", f"{r['wrrf_weak']:.6f}",
                f"{r['wrrf_strong']:.6f}", f"{r['moe']:.6f}",
            ])
    print(f"\nCSV saved: {comp_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    comp_png = os.path.join(results_folder, "meta_learner_retrieval_comparison.png")
    _save_comparison_plot(comparison_rows, ndcg_k, comp_png)

    boundary_png = os.path.join(results_folder, "meta_learner_decision_boundary.png")
    _save_decision_boundary_plot(
        final_mdl, best_model_name,
        td_aw, td_as, ds_td,
        te_aw, te_as, ds_te,
        boundary_png,
    )


if __name__ == "__main__":
    main()
