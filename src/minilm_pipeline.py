"""
minilm_pipeline.py

Complete end-to-end routing pipeline using
sentence-transformers/all-MiniLM-L6-v2 (384-dim) as the dense encoder.

All MiniLM-specific artifacts land in  data/minilm_version/
  data/minilm_version/<dataset>/   -- MiniLM embeddings, dense results, feature cache
  data/minilm_version/results/     -- CSVs, plots, meta-dataset cache

BM25 indexes and qrels are model-agnostic and are loaded from the existing
data/processed_data/<bge-m3>/<dataset>/ tree (no recomputation needed).

Pipeline
--------
Phase 1-3  Encode queries + corpus with MiniLM (cached per dataset).
           Run dense retrieval (cosine sim, top-1000, cached).
           Compute 15 routing features using BM25 + MiniLM dense results.
Phase 4    Weak-signal XGBoost grid search on merged pool (300 q/dataset).
Phase 5    Strong-signal XGBoost grid search on merged pool (384-dim MiniLM).
Phase 6    OOF meta-dataset construction (10-fold, zero-leakage).
Phase 7    Meta-learner grid search (10 families, 145 combos, 10-round MC CV).
Phase 8    Final evaluation -- BM25 / Dense(MiniLM) / Static-RRF /
           wRRF-weak / wRRF-strong / MoE -- on 15% held-out test per dataset.
Phase 9    Comparison bar chart + decision boundary plot.

Outputs
-------
  data/minilm_version/results/minilm_weak_best_params.csv
  data/minilm_version/results/minilm_strong_best_params.csv
  data/minilm_version/results/minilm_meta_dataset.csv
  data/minilm_version/results/minilm_meta_best_params.csv
  data/minilm_version/results/minilm_retrieval_comparison.{csv,png}
  data/minilm_version/results/minilm_decision_boundary.png

Usage
-----
  python src/minilm_pipeline.py
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
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import torch
import xgboost as xgb

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    ensure_dir, get_config_path, load_config,
    load_pickle, save_pickle,
    load_queries, load_qrels,
    model_short_name,
)
from src.weak_signal_model_grid_search import (
    FEATURE_NAMES,
    compute_query_features,
    dataset_seed_offset,
    ensure_english_stopwords,
    query_ndcg_at_k,
    run_dense_retrieval,
    set_global_seed,
    zscore_stats,
)

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

# ── Constants ─────────────────────────────────────────────────────────────────

MINILM_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_BASE_DIR = os.path.join("data", "minilm_version")
RESULTS_SUBDIR  = "results"

REMOVED_FEATURES = ["query_length"]
ACTIVE_COLS      = [i for i, n in enumerate(FEATURE_NAMES)
                    if n not in set(REMOVED_FEATURES)]

WEAK_PARAM_KEYS = [
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "min_child_weight", "gamma",
]

# Models that receive 3 features [aw, as, |aw-as|].
# Tree models (xgboost, lightgbm, random_forest, extra_trees) use only [aw, as].
NON_TREE_MODELS = {"ridge", "lasso", "elasticnet", "svr", "knn", "mlp"}
LINEAR_MODELS   = NON_TREE_MODELS   # alias used by _save_decision_boundary_plot

DS_PALETTE = {
    "scifact":  "#4878D0",
    "nfcorpus": "#EE854A",
    "arguana":  "#6ACC65",
    "fiqa":     "#D65F5F",
    "scidocs":  "#B47CC7",
}


# ── Phase 1-3: MiniLM data loading ───────────────────────────────────────────

def _minilm_ds_dir(ds_name):
    return os.path.join(MINILM_BASE_DIR, ds_name)


def _read_corpus_texts(corpus_jsonl):
    """Read corpus.jsonl -> {doc_id: 'title text'}."""
    docs = {}
    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            d     = json.loads(line)
            title = (d.get("title") or "").strip()
            text  = (d.get("text")  or "").strip()
            docs[d["_id"]] = (title + " " + text).strip()
    return docs


def _load_or_compute_minilm_embeddings(ds_name, cfg, device):
    """
    Encode queries + corpus with MiniLM; cache results to
    data/minilm_version/<dataset>/.
    Returns (q_vecs, q_ids, c_vecs, c_ids) — all tensors on CPU.
    """
    ds_dir = _minilm_ds_dir(ds_name)
    ensure_dir(ds_dir)
    q_pt  = os.path.join(ds_dir, "query_vectors.pt")
    q_pkl = os.path.join(ds_dir, "query_ids.pkl")
    c_pt  = os.path.join(ds_dir, "corpus_embeddings.pt")
    c_pkl = os.path.join(ds_dir, "corpus_ids.pkl")

    if all(os.path.exists(p) for p in [q_pt, q_pkl, c_pt, c_pkl]):
        q_vecs = torch.load(q_pt, weights_only=True)
        q_ids  = load_pickle(q_pkl)
        c_vecs = torch.load(c_pt, weights_only=True)
        c_ids  = load_pickle(c_pkl)
        return q_vecs, q_ids, c_vecs, c_ids

    # Source artifacts (model-agnostic paths)
    short_model    = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    src_dir        = os.path.join(processed_root, short_model, ds_name)

    raw_queries = load_queries(os.path.join(src_dir, "queries.jsonl"))
    corpus_docs = _read_corpus_texts(os.path.join(src_dir, "corpus.jsonl"))

    print(f"  [{ds_name}] Encoding with {MINILM_MODEL} ...")
    enc = SentenceTransformer(MINILM_MODEL, device=str(device))

    q_ids   = sorted(raw_queries.keys())
    q_texts = [raw_queries[q] for q in q_ids]
    q_np    = enc.encode(q_texts, batch_size=256, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    q_vecs  = torch.tensor(q_np, dtype=torch.float32)

    c_ids   = sorted(corpus_docs.keys())
    c_texts = [corpus_docs[c] for c in c_ids]
    c_np    = enc.encode(c_texts, batch_size=256, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    c_vecs  = torch.tensor(c_np, dtype=torch.float32)

    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    torch.save(q_vecs, q_pt)
    save_pickle(q_ids,  q_pkl)
    torch.save(c_vecs, c_pt)
    save_pickle(c_ids,  c_pkl)
    print(f"  [{ds_name}] Saved  q={q_vecs.shape}  c={c_vecs.shape}")
    return q_vecs, q_ids, c_vecs, c_ids


def _load_dataset_minilm(ds_name, cfg, device):
    """
    Build and cache a dataset dict using MiniLM dense results.
    Returns the same format as load_dataset_for_grid_search:
      X, y, qids, bm25_results, dense_results, qrels
    """
    ds_dir = _minilm_ds_dir(ds_name)
    ensure_dir(ds_dir)

    top_k  = int(cfg["benchmark"]["top_k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])

    # BM25 artifacts -- shared with main bge-m3 pipeline
    short_model    = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    src_dir        = os.path.join(processed_root, short_model, ds_name)

    bm25_params = u.get_bm25_params(cfg)
    bm25_paths  = u.bm25_artifact_paths(
        src_dir, bm25_params["k1"], bm25_params["b"],
        bm25_params["use_stemming"], top_k=top_k,
    )

    raw_queries                    = load_queries(os.path.join(src_dir, "queries.jsonl"))
    qrels                          = load_qrels(os.path.join(src_dir, "qrels.tsv"))
    query_tokens                   = load_pickle(bm25_paths["query_tokens_pkl"])
    word_freq, total_corpus_tokens = load_pickle(bm25_paths["word_freq_pkl"])
    doc_freq, total_docs           = load_pickle(bm25_paths["doc_freq_pkl"])
    bm25_results                   = load_pickle(bm25_paths["bm25_results_pkl"])

    # MiniLM dense retrieval (cached per dataset)
    dense_pkl     = os.path.join(ds_dir, f"dense_results_topk_{top_k}.pkl")
    dense_results = None
    if os.path.exists(dense_pkl) and os.path.getsize(dense_pkl) > 0:
        try:
            dense_results = load_pickle(dense_pkl)
        except Exception as exc:
            print(f"  [{ds_name}] dense cache corrupt; rebuilding. ({exc})")

    if dense_results is None:
        q_vecs, q_ids, c_vecs, c_ids = _load_or_compute_minilm_embeddings(
            ds_name, cfg, device)
        if device.type == "cuda":
            try:
                c_vecs = c_vecs.to(device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        dense_results = run_dense_retrieval(q_vecs, q_ids, c_vecs, c_ids, top_k, cfg)
        save_pickle(dense_results, dense_pkl)
        print(f"  [{ds_name}] Dense results saved.")

    # Features + labels (cached)
    feat_pkl = os.path.join(ds_dir, "features_labels_minilm.pkl")
    if os.path.exists(feat_pkl) and os.path.getsize(feat_pkl) > 0:
        try:
            cached = load_pickle(feat_pkl)
            return {
                "X":            cached["X"],
                "y":            cached["y"],
                "qids":         cached["qids"],
                "bm25_results": bm25_results,
                "dense_results": dense_results,
                "qrels":        qrels,
            }
        except Exception as exc:
            print(f"  [{ds_name}] feature cache corrupt; recomputing. ({exc})")

    routing_cfg    = cfg.get("routing_features", {})
    overlap_k      = int(routing_cfg.get("overlap_k", 10))
    feature_stat_k = int(routing_cfg.get("feature_stat_k", 10))
    epsilon        = float(routing_cfg.get("epsilon", 1e-8))
    ce_alpha       = float(routing_cfg.get("ce_smoothing_alpha", 1.0))
    stemmer_lang   = cfg["preprocessing"]["stemmer_language"]
    use_stemming   = bm25_params["use_stemming"]

    english_sw = ensure_english_stopwords()
    if use_stemming:
        stemmer_obj    = SnowballStemmer(stemmer_lang)
        stopword_stems = frozenset(stemmer_obj.stem(w) for w in english_sw)
    else:
        stopword_stems = frozenset(w.lower() for w in english_sw)

    qids      = sorted(raw_queries.keys())
    feat_rows = []
    labels    = []

    for qid in qids:
        feat = compute_query_features(
            raw_queries[qid],
            query_tokens.get(qid, []),
            bm25_results.get(qid, []),
            dense_results.get(qid, []),
            word_freq, total_corpus_tokens, doc_freq, total_docs,
            stopword_stems, overlap_k, feature_stat_k, epsilon, ce_alpha,
        )
        feat_rows.append([feat[name] for name in FEATURE_NAMES])

        sp_ndcg = query_ndcg_at_k(bm25_results.get(qid, []), qrels.get(qid, {}), ndcg_k)
        de_ndcg = query_ndcg_at_k(dense_results.get(qid, []), qrels.get(qid, {}), ndcg_k)
        label   = 0.5 * ((sp_ndcg - de_ndcg) / (max(sp_ndcg, de_ndcg) + epsilon) + 1.0)
        labels.append(float(np.clip(label, 0.0, 1.0)))

    X = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(labels,    dtype=np.float32)
    save_pickle({"X": X, "y": y, "qids": qids}, feat_pkl)
    print(f"  [{ds_name}] Features + labels saved  ({len(qids)} queries).")

    return {
        "X": X, "y": y, "qids": qids,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "qrels": qrels,
    }


def _load_embeddings_minilm(ds_name, cfg, device, wd):
    """
    Strong-signal variant: replace X in wd with MiniLM query embedding vectors
    (384-dim), aligned to the same qids order as wd["qids"].
    """
    q_vecs, q_ids, _, _ = _load_or_compute_minilm_embeddings(ds_name, cfg, device)
    q_vecs_cpu = q_vecs.cpu()
    qid_to_idx = {qid: i for i, qid in enumerate(q_ids)}
    indices    = [qid_to_idx[qid] for qid in wd["qids"]]
    X_emb      = q_vecs_cpu[indices].numpy().astype(np.float32)
    return {
        "X":            X_emb,
        "y":            wd["y"],
        "qids":         wd["qids"],
        "bm25_results": wd["bm25_results"],
        "dense_results": wd["dense_results"],
        "qrels":        wd["qrels"],
    }


# ── wRRF retrieval helpers ────────────────────────────────────────────────────

def _wrrf_single(alpha, qid, bm25_res, dense_res, qrels, rrf_k, ndcg_k):
    alpha    = float(alpha)
    bm_pairs = bm25_res.get(qid, [])
    de_pairs = dense_res.get(qid, [])
    bm_rank  = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
    de_rank  = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
    bm_miss  = len(bm_pairs) + 1
    de_miss  = len(de_pairs) + 1
    fused    = {
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


# ── XGBoost grid search (weak + strong signal on merged pool) ─────────────────

def _xgb_make_combos(grid_cfg, param_keys):
    lists = [grid_cfg[k] for k in param_keys]
    return [dict(zip(param_keys, vals)) for vals in itertools.product(*lists)]


def _xgb_eval_combo(combo, X, y, meta, fold_indices, retrieval_data,
                    rrf_k, ndcg_k, seed):
    """
    Evaluate one XGBoost hyperparameter combo via MC CV on the merged traindev.
    meta[i] = (ds_name, qid) for each row.  Returns mean CV NDCG@10.
    """
    fold_scores = []
    for tr_idx, va_idx in fold_indices:
        mu, sig = zscore_stats(X[tr_idx])
        sig      = np.where(sig < 1e-8, 1.0, sig)
        Xtr_z    = (X[tr_idx] - mu) / sig
        Xva_z    = (X[va_idx] - mu) / sig

        mdl = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu",
            verbosity=0, random_state=seed, n_jobs=1,
            **combo,
        )
        try:
            mdl.fit(Xtr_z, y[tr_idx])
        except Exception:
            fold_scores.append(0.0)
            continue

        preds = np.clip(mdl.predict(Xva_z), 0.0, 1.0)
        ndcgs = []
        for alpha, i in zip(preds, va_idx):
            ds_name, qid = meta[i]
            rd = retrieval_data[ds_name]
            ndcgs.append(_wrrf_single(
                alpha, qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))
        fold_scores.append(float(np.mean(ndcgs)) if ndcgs else 0.0)

    return float(np.mean(fold_scores))


def _xgb_grid_search(X, y, meta, fold_indices, grid_cfg, param_keys,
                     retrieval_data, rrf_k, ndcg_k, n_jobs, seed, label):
    combos = _xgb_make_combos(grid_cfg, param_keys)
    print(f"  {label}: {len(combos)} combos × {len(fold_indices)} CV rounds ...")
    scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_xgb_eval_combo)(
            combo, X, y, meta, fold_indices, retrieval_data, rrf_k, ndcg_k, seed,
        )
        for combo in combos
    )
    best_idx   = int(np.argmax(scores))
    best_combo = combos[best_idx]
    best_score = float(scores[best_idx])
    print(f"  {label} best: {best_combo}  CV NDCG@10={best_score:.4f}")
    return best_combo, best_score


# ── Base model OOF + final training ──────────────────────────────────────────

def _build_oof_predictions(X_weak_td, X_strong_td, y_td,
                            n_oof_folds, weak_params, strong_params,
                            seed, xgb_device):
    n     = len(y_td)
    rng   = np.random.RandomState(seed + 7777)
    perm  = rng.permutation(n)
    folds = np.array_split(perm, n_oof_folds)

    aw_oof = np.empty(n, dtype=np.float32)
    as_oof = np.empty(n, dtype=np.float32)

    for fi, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(n_oof_folds) if j != fi])

        mu_w, sig_w = zscore_stats(X_weak_td[tr_idx])
        Xw_tr = (X_weak_td[tr_idx] - mu_w) / sig_w
        Xw_va = (X_weak_td[val_idx] - mu_w) / sig_w
        mdl_w = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device=xgb_device, verbosity=0,
            random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
            **weak_params,
        )
        mdl_w.fit(Xw_tr, y_td[tr_idx])
        aw_oof[val_idx] = np.clip(mdl_w.predict(Xw_va), 0.0, 1.0)

        mu_s, sig_s = zscore_stats(X_strong_td[tr_idx])
        Xs_tr = (X_strong_td[tr_idx] - mu_s) / sig_s
        Xs_va = (X_strong_td[val_idx] - mu_s) / sig_s
        mdl_s = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device=xgb_device, verbosity=0,
            random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
            **strong_params,
        )
        mdl_s.fit(Xs_tr, y_td[tr_idx])
        as_oof[val_idx] = np.clip(mdl_s.predict(Xs_va), 0.0, 1.0)

        print(f"  OOF fold {fi+1}/{n_oof_folds}  val={len(val_idx)} tr={len(tr_idx)}")

    return aw_oof, as_oof


def _train_base_models(X_weak_td, X_strong_td, y_td,
                       weak_params, strong_params, seed, xgb_device):
    mu_w, sig_w = zscore_stats(X_weak_td)
    mdl_w = xgb.XGBRegressor(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device=xgb_device, verbosity=0,
        random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
        **weak_params,
    )
    mdl_w.fit((X_weak_td - mu_w) / sig_w, y_td)

    mu_s, sig_s = zscore_stats(X_strong_td)
    mdl_s = xgb.XGBRegressor(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device=xgb_device, verbosity=0,
        random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
        **strong_params,
    )
    mdl_s.fit((X_strong_td - mu_s) / sig_s, y_td)

    return mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s


# ── Meta-learner helpers ──────────────────────────────────────────────────────

def _meta_features(aw, as_, model_name):
    aw  = np.asarray(aw,  dtype=np.float32)
    as_ = np.asarray(as_, dtype=np.float32)
    if model_name in NON_TREE_MODELS:
        return np.column_stack([aw, as_, np.abs(aw - as_)])
    return np.column_stack([aw, as_])


def _fit_meta(model_name, params, X, y):
    if model_name == "ridge":
        mdl = Ridge(alpha=float(params["alpha"]))
    elif model_name == "lasso":
        mdl = Lasso(alpha=float(params["alpha"]), max_iter=5000)
    elif model_name == "elasticnet":
        mdl = ElasticNet(alpha=float(params["alpha"]),
                         l1_ratio=float(params["l1_ratio"]), max_iter=5000)
    elif model_name == "svr":
        mdl = SVR(C=float(params["C"]), epsilon=float(params["epsilon"]),
                  kernel="rbf")
    elif model_name == "knn":
        mdl = KNeighborsRegressor(n_neighbors=int(params["n_neighbors"]),
                                  weights=str(params["weights"]))
    elif model_name == "random_forest":
        mdl = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42, n_jobs=1,
        )
    elif model_name == "extra_trees":
        mdl = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=42, n_jobs=1,
        )
    elif model_name == "mlp":
        mdl = MLPRegressor(
            hidden_layer_sizes=tuple(int(x) for x in params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]), max_iter=2000, random_state=42,
        )
    elif model_name == "xgboost":
        mdl = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu", verbosity=0,
            random_state=42, n_jobs=1,
            subsample=1.0, colsample_bytree=1.0, min_child_weight=1, gamma=0.0,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
        )
    elif model_name == "lightgbm":
        if not _HAS_LGB:
            raise RuntimeError("lightgbm is not installed.")
        mdl = lgb.LGBMRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=0.8, colsample_bytree=1.0, min_child_samples=5,
            random_state=42, n_jobs=1, verbose=-1,
        )
    else:
        raise ValueError(f"Unknown meta-model: {model_name}")
    mdl.fit(X, y)
    return mdl


def _pred_meta(mdl, X):
    return np.clip(mdl.predict(X), 0.0, 1.0).astype(np.float32)


def _mc_fold_indices(n, dev_frac_of_n, n_rounds, seed):
    n_dev = max(1, round(dev_frac_of_n * n))
    folds = []
    for fi in range(n_rounds):
        rng  = np.random.RandomState(seed + fi * 1000)
        perm = rng.permutation(n)
        folds.append((perm[n_dev:], perm[:n_dev]))   # (train_idx, dev_idx)
    return folds


def _meta_eval_combo(model_name, params, td_aw, td_as, td_gt,
                     td_qids, td_ds_names, mc_folds, retrieval_data, rrf_k, ndcg_k):
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


def _build_meta_combos(grid_cfg):
    combos = []
    for alpha in grid_cfg["ridge"]["alpha"]:
        combos.append(("ridge", {"alpha": alpha}))
    for alpha in grid_cfg["lasso"]["alpha"]:
        combos.append(("lasso", {"alpha": alpha}))
    for alpha, l1_ratio in itertools.product(
            grid_cfg["elasticnet"]["alpha"], grid_cfg["elasticnet"]["l1_ratio"]):
        combos.append(("elasticnet", {"alpha": alpha, "l1_ratio": l1_ratio}))
    for C, epsilon in itertools.product(
            grid_cfg["svr"]["C"], grid_cfg["svr"]["epsilon"]):
        combos.append(("svr", {"C": C, "epsilon": epsilon}))
    for n_neighbors, weights in itertools.product(
            grid_cfg["knn"]["n_neighbors"], grid_cfg["knn"]["weights"]):
        combos.append(("knn", {"n_neighbors": n_neighbors, "weights": weights}))
    for n_est, depth, msl in itertools.product(
            grid_cfg["random_forest"]["n_estimators"],
            grid_cfg["random_forest"]["max_depth"],
            grid_cfg["random_forest"]["min_samples_leaf"]):
        combos.append(("random_forest", {
            "n_estimators": n_est, "max_depth": depth, "min_samples_leaf": msl}))
    for n_est, depth, msl in itertools.product(
            grid_cfg["extra_trees"]["n_estimators"],
            grid_cfg["extra_trees"]["max_depth"],
            grid_cfg["extra_trees"]["min_samples_leaf"]):
        combos.append(("extra_trees", {
            "n_estimators": n_est, "max_depth": depth, "min_samples_leaf": msl}))
    for hls, alpha in itertools.product(
            grid_cfg["mlp"]["hidden_layer_sizes"], grid_cfg["mlp"]["alpha"]):
        combos.append(("mlp", {"hidden_layer_sizes": hls, "alpha": alpha}))
    for n_est, depth, lr in itertools.product(
            grid_cfg["xgboost"]["n_estimators"],
            grid_cfg["xgboost"]["max_depth"],
            grid_cfg["xgboost"]["learning_rate"]):
        combos.append(("xgboost", {
            "n_estimators": n_est, "max_depth": depth, "learning_rate": lr}))
    if _HAS_LGB:
        for n_est, depth, lr in itertools.product(
                grid_cfg["lightgbm"]["n_estimators"],
                grid_cfg["lightgbm"]["max_depth"],
                grid_cfg["lightgbm"]["learning_rate"]):
            combos.append(("lightgbm", {
                "n_estimators": n_est, "max_depth": depth, "learning_rate": lr}))
    else:
        print("  [warning] lightgbm not installed — skipping LightGBM combos")
    return combos


# ── Meta-dataset cache ────────────────────────────────────────────────────────

def _save_meta_dataset(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ds_name", "qid", "split",
                           "alpha_weak", "alpha_strong", "alpha_gt"])
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


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_comparison_plot(rows, ndcg_k, out_path):
    methods = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    labels  = ["BM25", "Dense (MiniLM)", "Static RRF (α=0.5)",
               "wRRF (weak)", "wRRF (strong/MiniLM)", "MoE Meta-Learner"]
    colors  = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#B47CC7", "#956CB4"]

    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    width   = 0.12
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2,
                          len(methods)) * width
    by_method = {m: [r[m] for r in rows] for m in methods}

    fig, ax = plt.subplots(figsize=(18, 6))
    for method, label, color, offset in zip(methods, labels, colors, offsets):
        bars = ax.bar(x + offset, by_method[method], width,
                      label=label, color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(f"MiniLM Pipeline — Retrieval Comparison NDCG@{ndcg_k}", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    all_scores = [s for ss in by_method.values() for s in ss]
    ax.set_ylim(0, min(1.0, max(all_scores) + 0.12))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot: {out_path}")


def _save_decision_boundary_plot(final_mdl, model_name,
                                  td_aw, td_as, td_ds_names,
                                  te_aw, te_as, te_ds_names,
                                  out_path):
    grid_size = 100
    aw_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    as_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    AW, AS  = np.meshgrid(aw_vals, as_vals)
    X_grid  = _meta_features(AW.ravel(), AS.ravel(), model_name)
    Z       = np.clip(final_mdl.predict(X_grid), 0.0, 1.0).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(AS, AW, Z, levels=30, cmap="RdYlBu_r", alpha=0.85)
    plt.colorbar(cf, ax=ax,
                 label="Predicted α   (1 = prefer BM25 / sparse,  0 = prefer Dense)")

    for ds_name, color in DS_PALETTE.items():
        m_td = np.array([d == ds_name for d in td_ds_names])
        m_te = np.array([d == ds_name for d in te_ds_names])
        if m_td.any():
            ax.scatter(td_as[m_td], td_aw[m_td], c=color, s=12,
                       alpha=0.30, edgecolors="none", zorder=3)
        if m_te.any():
            ax.scatter(te_as[m_te], te_aw[m_te], c=color, s=40,
                       alpha=0.85, edgecolors="k", linewidths=0.4,
                       label=ds_name, zorder=4)

    ax.set_xlabel("α_strong / MiniLM  (strong-signal XGBoost prediction)", fontsize=10)
    ax.set_ylabel("α_weak   (weak-signal XGBoost prediction)", fontsize=10)
    ax.set_title(
        f"MiniLM Meta-Learner Prediction Surface — {model_name}\n"
        "(large markers = test;  small = traindev)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right", title="dataset")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Decision boundary plot: {out_path}")


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

    # Grid search n_jobs — avoid forking alongside CUDA
    n_jobs_gs = 1 if cuda_available else int(
        cfg.get("meta_learner_moe", {}).get("n_jobs", -1)
    )

    exp_cfg        = cfg["meta_learner_moe"]
    n_queries      = int(exp_cfg["n_queries"])              # 300
    trunc_seed     = int(exp_cfg["truncation_seed"])        # 31415
    test_frac      = float(exp_cfg["test_fraction"])        # 0.15
    dev_frac_of_td = float(exp_cfg["dev_fraction_of_traindev"])  # 0.176
    n_cv_rounds    = int(exp_cfg["n_cv_rounds"])            # 10
    n_oof_folds    = int(exp_cfg["n_oof_folds"])            # 10
    meta_grid_cfg  = exp_cfg["grid"]

    weak_grid_cfg   = cfg["xgboost_params_grid"]
    strong_grid_cfg = cfg["strong_signal_params_grid"]
    n_cv_folds_xgb  = int(weak_grid_cfg.get("n_folds", 10))

    results_folder = os.path.join(MINILM_BASE_DIR, RESULTS_SUBDIR)
    ensure_dir(MINILM_BASE_DIR)
    ensure_dir(results_folder)

    # ── Phase 1-3: Load all datasets ─────────────────────────────────────────
    print("\n=== Phase 1-3: Loading datasets with MiniLM ===")
    all_raw        = {}
    retrieval_data = {}

    for ds_name in dataset_names:
        print(f"\n  {ds_name} ...")
        wd = _load_dataset_minilm(ds_name, cfg, device)
        sd = _load_embeddings_minilm(ds_name, cfg, device, wd)
        all_raw[ds_name] = {"wd": wd, "sd": sd}
        retrieval_data[ds_name] = {
            "bm25_results":  wd["bm25_results"],
            "dense_results": wd["dense_results"],
            "qrels":         wd["qrels"],
        }

    if cuda_available:
        torch.cuda.empty_cache()

    # ── Truncate and split (equal 300 q/dataset) ──────────────────────────────
    print("\n=== Truncating and splitting datasets (n=300 each) ===")

    X_weak_td_parts, X_strong_td_parts = [], []
    y_td_parts                          = []
    qids_td, ds_td                      = [], []
    X_weak_te_parts, X_strong_te_parts = [], []
    y_te_parts                          = []
    qids_te, ds_te                      = [], []

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

        print(f"  {ds_name}: {n_td_ds} traindev  |  {n_test} test")

    X_weak_td   = np.vstack(X_weak_td_parts)
    X_strong_td = np.vstack(X_strong_td_parts)
    y_td        = np.concatenate(y_td_parts).astype(np.float32)
    X_weak_te   = np.vstack(X_weak_te_parts)
    X_strong_te = np.vstack(X_strong_te_parts)
    y_te        = np.concatenate(y_te_parts).astype(np.float32)
    n_td        = len(y_td)

    meta_traindev = list(zip(ds_td, qids_td))   # (ds_name, qid) per row

    print(f"\n  Merged traindev : {n_td}  |  Merged test : {len(y_te)}")
    print(f"  Weak dim        : {X_weak_td.shape[1]}")
    print(f"  Strong dim      : {X_strong_td.shape[1]}")

    # MC CV fold indices for XGBoost grid searches (10-round 80/20)
    xgb_folds = _mc_fold_indices(n_td, 0.20, n_cv_folds_xgb, seed)

    # ── Phase 4: Weak XGBoost grid search ────────────────────────────────────
    print("\n=== Phase 4: Weak-signal XGBoost grid search ===")
    best_weak_params, best_weak_cv = _xgb_grid_search(
        X_weak_td, y_td, meta_traindev, xgb_folds,
        weak_grid_cfg, WEAK_PARAM_KEYS,
        retrieval_data, rrf_k, ndcg_k, n_jobs_gs, seed,
        "Weak signal (15 features, MiniLM dense)",
    )

    # ── Phase 5: Strong XGBoost grid search ──────────────────────────────────
    print("\n=== Phase 5: Strong-signal (MiniLM 384-dim) XGBoost grid search ===")
    best_strong_params, best_strong_cv = _xgb_grid_search(
        X_strong_td, y_td, meta_traindev, xgb_folds,
        strong_grid_cfg, WEAK_PARAM_KEYS,   # same key names, different grid values
        retrieval_data, rrf_k, ndcg_k, n_jobs_gs, seed,
        "Strong signal (MiniLM 384-dim)",
    )

    # Save base model best params
    for label, params, cv_score, fname in [
        ("weak",   best_weak_params,   best_weak_cv,   "minilm_weak_best_params.csv"),
        ("strong", best_strong_params, best_strong_cv, "minilm_strong_best_params.csv"),
    ]:
        p = os.path.join(results_folder, fname)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "params_json", f"cv_ndcg@{ndcg_k}"])
            w.writerow(["xgboost", json.dumps(params), f"{cv_score:.6f}"])
        print(f"  {label} params saved: {p}")

    # ── Phase 6: OOF meta-dataset ─────────────────────────────────────────────
    cache_path = os.path.join(results_folder, "minilm_meta_dataset.csv")

    if os.path.exists(cache_path):
        print(f"\n=== Phase 6: Loading OOF meta-dataset from cache ===")
        cached    = _load_meta_dataset(cache_path)
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
                f"Cache '{cache_path}' does not match the current split "
                f"(missing key {e}). Delete it and re-run to rebuild."
            ) from None
    else:
        print("\n=== Phase 6: Building OOF meta-dataset (zero-leakage) ===")
        print("  Step 1: OOF predictions for traindev ...")
        td_aw, td_as = _build_oof_predictions(
            X_weak_td, X_strong_td, y_td,
            n_oof_folds, best_weak_params, best_strong_params, seed, xgb_device,
        )
        td_gt = y_td.copy()

        print("  Step 2: Final base models on full traindev → test predictions ...")
        mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s = _train_base_models(
            X_weak_td, X_strong_td, y_td,
            best_weak_params, best_strong_params, seed, xgb_device,
        )
        te_aw = np.clip(
            mdl_w.predict((X_weak_te   - mu_w) / sig_w), 0.0, 1.0
        ).astype(np.float32)
        te_as = np.clip(
            mdl_s.predict((X_strong_te - mu_s) / sig_s), 0.0, 1.0
        ).astype(np.float32)

        cache_rows = []
        for i, (d, q) in enumerate(zip(ds_td, qids_td)):
            cache_rows.append({"ds_name": d, "qid": q, "split": "traindev",
                               "alpha_weak":   float(td_aw[i]),
                               "alpha_strong": float(td_as[i]),
                               "alpha_gt":     float(td_gt[i])})
        for i, (d, q) in enumerate(zip(ds_te, qids_te)):
            cache_rows.append({"ds_name": d, "qid": q, "split": "test",
                               "alpha_weak":   float(te_aw[i]),
                               "alpha_strong": float(te_as[i]),
                               "alpha_gt":     float(y_te[i])})
        _save_meta_dataset(cache_rows, cache_path)
        print(f"  Meta-dataset cached: {cache_path}")

    # ── Phase 7: Meta-learner grid search ─────────────────────────────────────
    print("\n=== Phase 7: Meta-learner grid search ===")
    mc_folds = _mc_fold_indices(n_td, dev_frac_of_td, n_cv_rounds, seed)
    combos   = _build_meta_combos(meta_grid_cfg)
    print(f"  {len(combos)} combos × {n_cv_rounds} rounds = "
          f"{len(combos) * n_cv_rounds} evaluations")

    meta_scores = Parallel(n_jobs=n_jobs_gs, backend="loky")(
        delayed(_meta_eval_combo)(
            model_name, params,
            td_aw, td_as, td_gt,
            qids_td, ds_td,
            mc_folds, retrieval_data, rrf_k, ndcg_k,
        )
        for model_name, params in combos
    )

    best_idx         = int(np.argmax(meta_scores))
    best_meta_name   = combos[best_idx][0]
    best_meta_params = combos[best_idx][1]
    best_meta_cv     = float(meta_scores[best_idx])

    print(f"\n  Best meta-learner  : {best_meta_name}")
    print(f"  Best params        : {best_meta_params}")
    print(f"  CV NDCG@{ndcg_k}      : {best_meta_cv:.4f}")

    meta_params_csv = os.path.join(results_folder, "minilm_meta_best_params.csv")
    with open(meta_params_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_json", f"cv_ndcg@{ndcg_k}"])
        w.writerow([best_meta_name, json.dumps(best_meta_params),
                    f"{best_meta_cv:.6f}"])
    print(f"  Meta params saved: {meta_params_csv}")

    # ── Phase 8: Final model + per-dataset evaluation ─────────────────────────
    print("\n=== Phase 8: Final meta-learner + evaluation ===")
    X_td_meta = _meta_features(td_aw, td_as, best_meta_name)
    X_te_meta = _meta_features(te_aw, te_as, best_meta_name)
    final_mdl = _fit_meta(best_meta_name, best_meta_params, X_td_meta, td_gt)
    te_moe    = _pred_meta(final_mdl, X_te_meta)

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
        rd         = retrieval_data[ds_name]

        bm25_s   = _bm25_ndcg(te_qids_ds, rd["bm25_results"],  rd["qrels"], ndcg_k)
        dense_s  = _dense_ndcg(te_qids_ds, rd["dense_results"], rd["qrels"], ndcg_k)
        srrf_s   = _wrrf_ndcg(srrf_ds, te_qids_ds,
                               rd["bm25_results"], rd["dense_results"], rd["qrels"],
                               rrf_k, ndcg_k)
        wrrf_w_s = _wrrf_ndcg(aw_ds,   te_qids_ds,
                               rd["bm25_results"], rd["dense_results"], rd["qrels"],
                               rrf_k, ndcg_k)
        wrrf_s_s = _wrrf_ndcg(as_ds,   te_qids_ds,
                               rd["bm25_results"], rd["dense_results"], rd["qrels"],
                               rrf_k, ndcg_k)
        moe_s    = _wrrf_ndcg(moe_ds,  te_qids_ds,
                               rd["bm25_results"], rd["dense_results"], rd["qrels"],
                               rrf_k, ndcg_k)

        print(f"\n  {ds_name}  ({mask.sum()} test queries):")
        for lbl, val in [
            ("BM25",            bm25_s),
            ("Dense (MiniLM)",  dense_s),
            ("Static RRF",      srrf_s),
            ("wRRF weak",       wrrf_w_s),
            ("wRRF strong",     wrrf_s_s),
            ("MoE",             moe_s),
        ]:
            print(f"    {lbl:<20}: {val:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "bm25":        bm25_s,
            "dense":       dense_s,
            "static_rrf":  srrf_s,
            "wrrf_weak":   wrrf_w_s,
            "wrrf_strong": wrrf_s_s,
            "moe":         moe_s,
        })

    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print("Macro averages (MiniLM pipeline):")
    for lbl, key in [
        ("BM25",           "bm25"),
        ("Dense (MiniLM)", "dense"),
        ("Static RRF",     "static_rrf"),
        ("wRRF (weak)",    "wrrf_weak"),
        ("wRRF (strong)",  "wrrf_strong"),
        ("MoE",            "moe"),
    ]:
        print(f"  {lbl:<22} NDCG@{ndcg_k} = {macro[key]:.4f}")

    # ── Phase 9: Save CSV + plots ─────────────────────────────────────────────
    comp_csv = os.path.join(results_folder, "minilm_retrieval_comparison.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset",
                    f"bm25_ndcg@{ndcg_k}",         f"dense_ndcg@{ndcg_k}",
                    f"static_rrf_ndcg@{ndcg_k}",   f"wrrf_weak_ndcg@{ndcg_k}",
                    f"wrrf_strong_ndcg@{ndcg_k}",  f"moe_ndcg@{ndcg_k}"])
        for r in comparison_rows:
            w.writerow([r["group"],
                        f"{r['bm25']:.6f}",        f"{r['dense']:.6f}",
                        f"{r['static_rrf']:.6f}",  f"{r['wrrf_weak']:.6f}",
                        f"{r['wrrf_strong']:.6f}", f"{r['moe']:.6f}"])
    print(f"\nCSV saved: {comp_csv}")

    comp_png = os.path.join(results_folder, "minilm_retrieval_comparison.png")
    _save_comparison_plot(comparison_rows, ndcg_k, comp_png)

    boundary_png = os.path.join(results_folder, "minilm_decision_boundary.png")
    _save_decision_boundary_plot(
        final_mdl, best_meta_name,
        td_aw, td_as, ds_td,
        te_aw, te_as, ds_te,
        boundary_png,
    )

    print("\nMiniLM pipeline complete.")
    print(f"All outputs in: {results_folder}")


if __name__ == "__main__":
    main()
