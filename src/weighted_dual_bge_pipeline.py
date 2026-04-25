"""
weighted_dual_bge_pipeline.py

Full routing pipeline using BGE-M3 in BOTH retrieval modes:
  Sparse leg  — BGE-M3 sparse head   (lexical matching in token-weight space)
  Dense leg   — BGE-M3 dense head    (semantic cosine similarity)

The two modes share the same model weights; differences in retrieval
behaviour are attributable to the retrieval paradigm, not model quality.

Prerequisites
-------------
  python src/preprocess.py           (dense artifacts)
  python src/preprocess_bge_sparse.py  (sparse artifacts)

Pipeline
--------
Phase 1-3   Load datasets.  Per dataset: load BGE-M3 sparse results and
            dense results; compute 16 routing features (adapted for sparse
            head); compute soft labels from NDCG(sparse) vs NDCG(dense).
Phase 4     Merge 5 × 300 equal-data pool (same truncation seed as all
            prior experiments).  Split 85% traindev / 15% test.
Phase 5     Weak XGBoost grid search: 10-fold KFold CV on traindev.
            Input: 15 hand-crafted routing features (query_length excluded).
            Grid: xgboost_params_grid from config (6 480 combos).
Phase 6     Strong XGBoost grid search: 10-fold KFold CV on traindev.
            Input: 1 024-dim BGE-M3 dense query embeddings (existing cache).
            Grid: strong_signal_params_grid from config (96 combos).
Phase 7     10-fold OOF meta-dataset (zero-leakage).
Phase 8     MoE meta-learner grid search: 10-round Monte Carlo CV.
            10 families, 145 combos (meta_learner_moe.grid from config).
Phase 9     Final evaluation on 15% held-out test set per dataset + macro:
              BGE-M3 Sparse only | BGE-M3 Dense only | Static wRRF (α=0.5)
              wRRF (weak) | wRRF (strong) | MoE meta-learner
Phase 10    Save CSV + comparison bar chart + decision boundary plot.

Outputs (data/results/)
-----------------------
  bge_dual_weak_best_params.csv
  bge_dual_strong_best_params.csv
  bge_dual_meta_dataset.csv
  bge_dual_meta_best_params.csv
  bge_dual_retrieval_comparison.{csv,png}
  bge_dual_decision_boundary.png

Usage
-----
  python src/weighted_dual_bge_pipeline.py
"""

import csv
import hashlib
import itertools
import json
import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from src.utils import (
    ensure_dir, file_exists, get_config_path, load_config,
    load_pickle, load_queries, load_qrels, model_short_name, save_pickle,
)
from src.weak_signal_model_grid_search import (
    dataset_seed_offset,
    normalize_scores_minmax,
    query_ndcg_at_k,
    set_global_seed,
    zscore_stats,
    QUESTION_WORDS,
    ensure_english_stopwords,
)

# ── Feature schema (same ordering as existing pipeline for comparability) ──────
FEATURE_NAMES = [
    "query_length",           # Group A — excluded from model (same as BM25 pipeline)
    "stopword_ratio",         # Group A
    "has_question_word",      # Group A
    "average_idf",            # Group B — pseudo-IDF from sparse doc-frequency index
    "max_idf",                # Group B
    "rare_term_ratio",        # Group B
    "cross_entropy",          # Group B — entropy of query sparse weight distribution
    "top_dense_score",        # Group C
    "top_sparse_score",       # Group C — from BGE-M3 sparse scores
    "dense_confidence",       # Group C
    "sparse_confidence",      # Group C — from BGE-M3 sparse scores
    "overlap_at_k",           # Group D — BGE-M3 sparse vs dense lists
    "first_shared_doc_rank",  # Group D
    "spearman_topk",          # Group D
    "dense_entropy_topk",     # Group E
    "sparse_entropy_topk",    # Group E — from BGE-M3 sparse score distribution
]

REMOVED_FEATURES = ["query_length"]
ACTIVE_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n not in set(REMOVED_FEATURES)]

NON_TREE_MODELS = {"ridge", "lasso", "elasticnet", "svr", "knn", "mlp"}
LINEAR_MODELS   = NON_TREE_MODELS   # alias

DS_PALETTE = {
    "scifact":  "#4878D0",
    "nfcorpus": "#EE854A",
    "arguana":  "#6ACC65",
    "fiqa":     "#D65F5F",
    "scidocs":  "#B47CC7",
}


# ── Feature computation ───────────────────────────────────────────────────────

def compute_query_features_bge_dual(
    raw_text,
    query_sparse,       # {token_id: float}
    sparse_pairs,       # [(doc_id, score), ...]  BGE-M3 sparse top-k
    dense_pairs,        # [(doc_id, score), ...]  BGE-M3 dense top-k
    sparse_doc_freq,    # {token_id: int}          n docs with nonzero weight
    total_docs,
    stopwords,
    overlap_k,
    feature_stat_k,
    epsilon,
):
    sparse_norm = normalize_scores_minmax(sparse_pairs, epsilon)
    dense_norm  = normalize_scores_minmax(dense_pairs, epsilon)

    # ── Group A: Query Surface ────────────────────────────────────────────────
    raw_tokens   = raw_text.lower().split()
    n_tok        = len(raw_tokens)
    query_length = float(n_tok)

    if n_tok == 0:
        stopword_ratio = 0.0
    else:
        n_stop = sum(1 for t in raw_tokens if t in stopwords)
        stopword_ratio = n_stop / n_tok

    first_word = raw_tokens[0] if raw_tokens else ""
    has_qw     = 1.0 if first_word in QUESTION_WORDS else 0.0

    # ── Group B: Vocabulary Match (pseudo-IDF from sparse index) ─────────────
    # query_sparse tokens are the subword token IDs the model considers
    # important for this query. Pseudo-IDF uses the sparse inverted-index
    # document-frequency — exactly analogous to BM25 IDF.
    q_toks = [t for t, w in query_sparse.items() if w > 0]
    if q_toks:
        pseudo_idfs = [
            math.log((total_docs + 1.0) / (sparse_doc_freq.get(t, 0) + 1.0)) + 1.0
            for t in q_toks
        ]
        average_idf     = float(np.mean(pseudo_idfs))
        max_idf         = float(np.max(pseudo_idfs))
        idf_std         = float(np.std(pseudo_idfs))
        thresh          = average_idf + idf_std
        rare_term_ratio = sum(1 for v in pseudo_idfs if v >= thresh) / len(pseudo_idfs)

        # cross_entropy: Shannon entropy of the normalised query sparse-weight
        # distribution. High entropy = many equally-weighted terms (query spread
        # across concepts). Low entropy = one dominant term (focused query).
        weights      = np.array([query_sparse[t] for t in q_toks], dtype=np.float64)
        weights      = np.clip(weights, epsilon, None)
        p            = weights / weights.sum()
        cross_entropy = float(-np.sum(p * np.log2(p)))
    else:
        average_idf = max_idf = rare_term_ratio = cross_entropy = 0.0

    # ── Group C: Retriever Confidence ────────────────────────────────────────
    def _conf(normed):
        if len(normed) >= 2:
            return normed[0][1] - normed[1][1]
        return normed[0][1] if normed else 0.0

    top_dense_score   = dense_norm[0][1]  if dense_norm  else 0.0
    top_sparse_score  = sparse_norm[0][1] if sparse_norm else 0.0
    dense_confidence  = _conf(dense_norm)
    sparse_confidence = _conf(sparse_norm)

    # ── Group D: Retriever Agreement ─────────────────────────────────────────
    top_sp = [d for d, _ in sparse_pairs[:overlap_k]]
    top_de = [d for d, _ in dense_pairs[:overlap_k]]
    overlap_at_k = len(set(top_sp) & set(top_de)) / max(1, overlap_k)

    sp_rank = {d: r for r, d in enumerate([d for d, _ in sparse_pairs[:feature_stat_k]], 1)}
    de_rank = {d: r for r, d in enumerate([d for d, _ in dense_pairs[:feature_stat_k]], 1)}
    shared  = set(sp_rank) & set(de_rank)

    if shared:
        first_shared_doc_rank = min((sp_rank[d] + de_rank[d]) / 2.0 for d in shared)
    else:
        first_shared_doc_rank = float(feature_stat_k + 1)

    if len(shared) >= 2:
        diffs = [(sp_rank[d] - de_rank[d]) ** 2 for d in shared]
        n = len(diffs)
        spearman_topk = 1.0 - (6.0 * sum(diffs)) / (n * (n ** 2 - 1.0))
    else:
        spearman_topk = 0.0

    # ── Group E: Distribution Shape ───────────────────────────────────────────
    def _entropy(normed_pairs, k):
        pairs  = normed_pairs[:k]
        if not pairs:
            return 0.0
        scores = np.array([max(s, 0.0) for _, s in pairs], dtype=np.float64)
        total  = scores.sum()
        if total <= epsilon:
            return 0.0
        p = scores / total
        return float(-np.sum(p * np.log2(np.maximum(p, epsilon))))

    dense_entropy_topk  = _entropy(dense_norm,  feature_stat_k)
    sparse_entropy_topk = _entropy(sparse_norm, feature_stat_k)

    return {
        "query_length":          query_length,
        "stopword_ratio":        stopword_ratio,
        "has_question_word":     has_qw,
        "average_idf":           average_idf,
        "max_idf":               max_idf,
        "rare_term_ratio":       rare_term_ratio,
        "cross_entropy":         cross_entropy,
        "top_dense_score":       top_dense_score,
        "top_sparse_score":      top_sparse_score,
        "dense_confidence":      dense_confidence,
        "sparse_confidence":     sparse_confidence,
        "overlap_at_k":          overlap_at_k,
        "first_shared_doc_rank": first_shared_doc_rank,
        "spearman_topk":         spearman_topk,
        "dense_entropy_topk":    dense_entropy_topk,
        "sparse_entropy_topk":   sparse_entropy_topk,
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset_bge_dual(ds_name, cfg, device):
    """
    Load all retrieval artifacts for one dataset and return:
      X            (n_queries, 16) feature matrix
      y            (n_queries,)    soft labels
      qids         [str, ...]      query IDs
      sparse_results  {qid: [(doc_id, score)]}
      dense_results   {qid: [(doc_id, score)]}
      qrels           {qid: {doc_id: int}}
    """
    short_model    = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir         = os.path.join(processed_root, short_model, ds_name)
    top_k          = int(cfg["benchmark"]["top_k"])
    ndcg_k         = int(cfg["benchmark"]["ndcg_k"])
    routing_cfg    = cfg.get("routing_features", {})
    overlap_k      = int(routing_cfg.get("overlap_k", 10))
    feature_stat_k = int(routing_cfg.get("feature_stat_k", 10))
    epsilon        = float(routing_cfg.get("epsilon", 1e-8))

    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv     = os.path.join(ds_dir, "qrels.tsv")
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vecs_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl  = os.path.join(ds_dir, "query_ids.pkl")
    dense_res_pkl  = os.path.join(ds_dir, f"dense_results_topk_{top_k}.pkl")
    sparse_ids_pkl   = os.path.join(ds_dir, "bge_sparse_corpus_ids.pkl")
    sparse_index_pkl = os.path.join(ds_dir, "bge_sparse_inverted_index.pkl")
    sparse_df_pkl    = os.path.join(ds_dir, "bge_sparse_doc_freq.pkl")
    sparse_qvec_pkl  = os.path.join(ds_dir, "bge_sparse_query_vectors.pkl")
    sparse_res_pkl   = os.path.join(ds_dir, f"bge_sparse_results_topk_{top_k}.pkl")

    raw_queries = load_queries(queries_jsonl)
    qrels       = load_qrels(qrels_tsv)

    # ── Dense retrieval results ───────────────────────────────────────────────
    dense_results = None
    if file_exists(dense_res_pkl) and os.path.getsize(dense_res_pkl) > 0:
        try:
            print(f"  Loading cached dense results  [{ds_name}]")
            dense_results = load_pickle(dense_res_pkl)
        except Exception as exc:
            print(f"  [WARN] Dense cache corrupt; rebuilding. ({exc})")
    if dense_results is None:
        from src.weak_signal_model_grid_search import run_dense_retrieval
        print(f"  Running dense retrieval [{ds_name}] ...")
        corpus_emb  = torch.load(corpus_emb_pt, weights_only=True)
        if device.type == "cuda":
            try:
                corpus_emb = corpus_emb.to(device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        c_ids  = load_pickle(corpus_ids_pkl)
        q_vecs = torch.load(query_vecs_pt, weights_only=True)
        q_ids  = load_pickle(query_ids_pkl)
        dense_results = run_dense_retrieval(q_vecs, q_ids, corpus_emb, c_ids, top_k, cfg)
        save_pickle(dense_results, dense_res_pkl)

    # ── BGE-M3 sparse artifacts ───────────────────────────────────────────────
    for pth, lbl in [
        (sparse_ids_pkl,   "corpus IDs"),
        (sparse_index_pkl, "inverted index"),
        (sparse_df_pkl,    "doc-freq index"),
        (sparse_qvec_pkl,  "query sparse vectors"),
        (sparse_res_pkl,   "sparse results"),
    ]:
        if not (file_exists(pth) and os.path.getsize(pth) > 0):
            raise FileNotFoundError(
                f"BGE-M3 sparse artifact missing for '{ds_name}': {pth}\n"
                "Run  python src/preprocess_bge_sparse.py  first."
            )

    print(f"  Loading cached sparse results [{ds_name}]")
    sparse_results    = load_pickle(sparse_res_pkl)
    sparse_doc_freq   = load_pickle(sparse_df_pkl)
    query_sparse_vecs = load_pickle(sparse_qvec_pkl)
    corpus_ids        = load_pickle(sparse_ids_pkl)
    total_docs        = len(corpus_ids)

    # ── Feature + label cache ─────────────────────────────────────────────────
    cache_key = json.dumps({
        "mode":            "bge_dual",
        "top_k":           top_k,
        "ndcg_k":          ndcg_k,
        "overlap_k":       overlap_k,
        "feature_stat_k":  feature_stat_k,
        "epsilon":         epsilon,
    }, sort_keys=True)
    cache_hash = hashlib.md5(cache_key.encode(), usedforsecurity=False).hexdigest()[:12]
    feat_cache = os.path.join(ds_dir, f"features_labels_bge_dual_{cache_hash}.pkl")

    if file_exists(feat_cache) and os.path.getsize(feat_cache) > 0:
        try:
            print(f"  Loading cached features+labels [{ds_name}]")
            cached = load_pickle(feat_cache)
            return {
                "X":               cached["X"],
                "y":               cached["y"],
                "qids":            cached["qids"],
                "sparse_results":  sparse_results,
                "dense_results":   dense_results,
                "qrels":           qrels,
            }
        except Exception as exc:
            print(f"  [WARN] Feature cache corrupt; recomputing. ({exc})")

    stopwords = ensure_english_stopwords()
    qids      = sorted(raw_queries.keys())
    feat_rows, labels = [], []

    from tqdm import tqdm
    for qid in tqdm(qids, desc=f"  Features [{ds_name}]", dynamic_ncols=True):
        feat = compute_query_features_bge_dual(
            raw_queries[qid],
            query_sparse_vecs.get(qid, {}),
            sparse_results.get(qid, []),
            dense_results.get(qid, []),
            sparse_doc_freq,
            total_docs,
            stopwords,
            overlap_k,
            feature_stat_k,
            epsilon,
        )
        feat_rows.append([feat[n] for n in FEATURE_NAMES])

        sp_ndcg = query_ndcg_at_k(sparse_results.get(qid, []), qrels.get(qid, {}), ndcg_k)
        de_ndcg = query_ndcg_at_k(dense_results.get(qid, []),  qrels.get(qid, {}), ndcg_k)
        label   = 0.5 * ((sp_ndcg - de_ndcg) / (max(sp_ndcg, de_ndcg) + epsilon) + 1.0)
        labels.append(float(np.clip(label, 0.0, 1.0)))

    X = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(labels,    dtype=np.float32)
    save_pickle({"X": X, "y": y, "qids": qids}, feat_cache)

    return {
        "X":               X,
        "y":               y,
        "qids":            qids,
        "sparse_results":  sparse_results,
        "dense_results":   dense_results,
        "qrels":           qrels,
    }


def load_strong_embeddings(ds_name, cfg, device):
    """Load existing 1024-dim BGE-M3 dense query embeddings for strong signal."""
    from src.strong_signal_model_grid_search import load_embeddings_for_dataset
    return load_embeddings_for_dataset(ds_name, cfg, device)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _wrrf_single(alpha, qid, sparse_res, dense_res, qrels, rrf_k, ndcg_k):
    alpha    = float(alpha)
    sp_pairs = sparse_res.get(qid, [])
    de_pairs = dense_res.get(qid, [])
    sp_rank  = {d: r for r, (d, _) in enumerate(sp_pairs, 1)}
    de_rank  = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
    sp_miss  = len(sp_pairs) + 1
    de_miss  = len(de_pairs) + 1
    fused = {
        d: alpha        / (rrf_k + sp_rank.get(d, sp_miss))
           + (1 - alpha) / (rrf_k + de_rank.get(d, de_miss))
        for d in set(sp_rank) | set(de_rank)
    }
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k)


def _sparse_ndcg(qids, sparse_res, qrels, ndcg_k):
    scores = [query_ndcg_at_k(sparse_res.get(q, []), qrels.get(q, {}), ndcg_k) for q in qids]
    return float(np.mean(scores)) if scores else 0.0


def _dense_ndcg(qids, dense_res, qrels, ndcg_k):
    scores = [query_ndcg_at_k(dense_res.get(q, []), qrels.get(q, {}), ndcg_k) for q in qids]
    return float(np.mean(scores)) if scores else 0.0


def _wrrf_ndcg(alphas, qids, sparse_res, dense_res, qrels, rrf_k, ndcg_k):
    scores = [
        _wrrf_single(a, q, sparse_res, dense_res, qrels, rrf_k, ndcg_k)
        for a, q in zip(alphas, qids)
    ]
    return float(np.mean(scores)) if scores else 0.0


# ── XGBoost grid search (10-fold KFold) ──────────────────────────────────────

def _xgb_eval_kfold(combo, X, y, meta, kfold_splits, retrieval_data, rrf_k, ndcg_k, seed):
    """
    Evaluate one XGBoost combo via 10-fold KFold CV.
    meta[i] = (ds_name, qid).  Returns mean val NDCG@ndcg_k across folds.
    """
    fold_scores = []
    for tr_idx, va_idx in kfold_splits:
        mu, sig = zscore_stats(X[tr_idx])
        Xtr = (X[tr_idx] - mu) / sig
        Xva = (X[va_idx] - mu) / sig

        mdl = xgb.XGBRegressor(
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",
            device           = "cpu",
            verbosity        = 0,
            random_state     = seed,
            n_jobs           = 1,
            n_estimators     = int(combo["n_estimators"]),
            max_depth        = int(combo["max_depth"]),
            learning_rate    = float(combo["learning_rate"]),
            subsample        = float(combo["subsample"]),
            colsample_bytree = float(combo["colsample_bytree"]),
            min_child_weight = int(combo["min_child_weight"]),
            gamma            = float(combo["gamma"]),
        )
        mdl.fit(Xtr, y[tr_idx])
        preds = np.clip(mdl.predict(Xva), 0.0, 1.0)

        ndcgs = []
        for alpha, i in zip(preds, va_idx):
            ds_name, qid = meta[i]
            rd = retrieval_data[ds_name]
            ndcgs.append(_wrrf_single(
                alpha, qid,
                rd["sparse_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))
        fold_scores.append(float(np.mean(ndcgs)) if ndcgs else 0.0)

    return float(np.mean(fold_scores))


def _xgb_grid_search_kfold(
    X, y, meta, grid_cfg, n_folds, retrieval_data,
    rrf_k, ndcg_k, n_jobs, seed, label,
):
    """
    Run the full XGBoost hyperparameter grid search with 10-fold KFold CV.
    Returns (best_params_dict, best_cv_score).
    """
    PARAM_KEYS = ["n_estimators", "max_depth", "learning_rate",
                  "subsample", "colsample_bytree", "min_child_weight", "gamma"]
    raw_combos = list(itertools.product(
        grid_cfg["n_estimators"],
        grid_cfg["max_depth"],
        grid_cfg["learning_rate"],
        grid_cfg["subsample"],
        grid_cfg["colsample_bytree"],
        grid_cfg["min_child_weight"],
        grid_cfg["gamma"],
    ))
    combos = [dict(zip(PARAM_KEYS, c)) for c in raw_combos]

    # Pre-compute fold indices (deterministic, passed to all workers)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    kfold_splits = [(tr, va) for tr, va in kf.split(X)]

    print(f"  {len(combos)} combos × {n_folds} folds  ({label})")

    scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_xgb_eval_kfold)(
            combo, X, y, meta, kfold_splits,
            retrieval_data, rrf_k, ndcg_k, seed,
        )
        for combo in combos
    )

    best_idx   = int(np.argmax(scores))
    best_combo = combos[best_idx]
    best_score = float(scores[best_idx])

    print(f"  Best {label}: {best_combo}")
    print(f"  Best {label} CV NDCG@{ndcg_k}: {best_score:.4f}")

    return best_combo, best_score


# ── OOF base predictions ──────────────────────────────────────────────────────

def _build_oof_predictions(
    X_weak_td, X_strong_td, y_td,
    n_oof_folds, weak_params, strong_params, seed, xgb_device,
):
    n    = len(y_td)
    rng  = np.random.RandomState(seed + 7777)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_oof_folds)

    aw_oof = np.empty(n, dtype=np.float32)
    as_oof = np.empty(n, dtype=np.float32)

    for fi, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(n_oof_folds) if j != fi])

        mu_w, sig_w = zscore_stats(X_weak_td[tr_idx])
        mdl_w = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device=xgb_device, verbosity=0,
            random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
            **weak_params,
        )
        mdl_w.fit((X_weak_td[tr_idx] - mu_w) / sig_w, y_td[tr_idx])
        aw_oof[val_idx] = np.clip(
            mdl_w.predict((X_weak_td[val_idx] - mu_w) / sig_w), 0.0, 1.0
        )

        mu_s, sig_s = zscore_stats(X_strong_td[tr_idx])
        mdl_s = xgb.XGBRegressor(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device=xgb_device, verbosity=0,
            random_state=seed, n_jobs=1 if xgb_device == "cuda" else -1,
            **strong_params,
        )
        mdl_s.fit((X_strong_td[tr_idx] - mu_s) / sig_s, y_td[tr_idx])
        as_oof[val_idx] = np.clip(
            mdl_s.predict((X_strong_td[val_idx] - mu_s) / sig_s), 0.0, 1.0
        )

        print(f"  OOF fold {fi+1}/{n_oof_folds}  "
              f"(train={len(tr_idx)}, val={len(val_idx)})")

    return aw_oof, as_oof


def _train_base_models(
    X_weak_td, X_strong_td, y_td,
    weak_params, strong_params, seed, xgb_device,
):
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


# ── Meta-dataset cache I/O ────────────────────────────────────────────────────

def _save_meta_dataset(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ds_name", "qid", "split",
                        "alpha_weak", "alpha_strong", "alpha_gt"],
        )
        w.writeheader()
        for r in rows:
            w.writerow({
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


# ── MoE meta-learner (same as meta_learner_moe_grid_search.py) ────────────────

def _meta_features(aw, as_, model_name):
    aw  = np.asarray(aw,  dtype=np.float32)
    as_ = np.asarray(as_, dtype=np.float32)
    if model_name in LINEAR_MODELS:
        return np.column_stack([aw, as_, np.abs(aw - as_)])
    return np.column_stack([aw, as_])


def _fit_meta(model_name, params, X, y):
    if model_name == "ridge":
        mdl = Ridge(alpha=float(params["alpha"]))
    elif model_name == "lasso":
        mdl = Lasso(alpha=float(params["alpha"]), max_iter=5000)
    elif model_name == "elasticnet":
        mdl = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
        )
    elif model_name == "svr":
        mdl = SVR(C=float(params["C"]), epsilon=float(params["epsilon"]), kernel="rbf")
    elif model_name == "knn":
        mdl = KNeighborsRegressor(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
        )
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
            alpha=float(params["alpha"]),
            max_iter=2000, random_state=42,
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
            subsample=0.8, colsample_bytree=1.0,
            min_child_samples=5, random_state=42, n_jobs=1, verbose=-1,
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
        folds.append((perm[n_dev:], perm[:n_dev]))
    return folds


def _eval_meta_combo(
    model_name, params,
    td_aw, td_as, td_gt,
    td_qids, td_ds_names,
    mc_folds, retrieval_data, rrf_k, ndcg_k,
):
    round_scores = []
    for tr_idx, dv_idx in mc_folds:
        X_tr = _meta_features(td_aw[tr_idx], td_as[tr_idx], model_name)
        X_dv = _meta_features(td_aw[dv_idx], td_as[dv_idx], model_name)
        try:
            mdl   = _fit_meta(model_name, params, X_tr, td_gt[tr_idx])
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
                rd["sparse_results"], rd["dense_results"], rd["qrels"],
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
        grid_cfg["elasticnet"]["alpha"],
        grid_cfg["elasticnet"]["l1_ratio"],
    ):
        combos.append(("elasticnet", {"alpha": alpha, "l1_ratio": l1_ratio}))

    for C, epsilon in itertools.product(
        grid_cfg["svr"]["C"],
        grid_cfg["svr"]["epsilon"],
    ):
        combos.append(("svr", {"C": C, "epsilon": epsilon}))

    for n_neighbors, weights in itertools.product(
        grid_cfg["knn"]["n_neighbors"],
        grid_cfg["knn"]["weights"],
    ):
        combos.append(("knn", {"n_neighbors": n_neighbors, "weights": weights}))

    for n_est, depth, msl in itertools.product(
        grid_cfg["random_forest"]["n_estimators"],
        grid_cfg["random_forest"]["max_depth"],
        grid_cfg["random_forest"]["min_samples_leaf"],
    ):
        combos.append(("random_forest",
                        {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": msl}))

    for n_est, depth, msl in itertools.product(
        grid_cfg["extra_trees"]["n_estimators"],
        grid_cfg["extra_trees"]["max_depth"],
        grid_cfg["extra_trees"]["min_samples_leaf"],
    ):
        combos.append(("extra_trees",
                        {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": msl}))

    for hls, alpha in itertools.product(
        grid_cfg["mlp"]["hidden_layer_sizes"],
        grid_cfg["mlp"]["alpha"],
    ):
        combos.append(("mlp", {"hidden_layer_sizes": hls, "alpha": alpha}))

    for n_est, depth, lr in itertools.product(
        grid_cfg["xgboost"]["n_estimators"],
        grid_cfg["xgboost"]["max_depth"],
        grid_cfg["xgboost"]["learning_rate"],
    ):
        combos.append(("xgboost",
                        {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}))

    if _HAS_LGB:
        for n_est, depth, lr in itertools.product(
            grid_cfg["lightgbm"]["n_estimators"],
            grid_cfg["lightgbm"]["max_depth"],
            grid_cfg["lightgbm"]["learning_rate"],
        ):
            combos.append(("lightgbm",
                            {"n_estimators": n_est, "max_depth": depth, "learning_rate": lr}))
    else:
        print("  [warning] lightgbm not installed — skipping LightGBM combos")

    return combos


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_comparison_plot(rows, ndcg_k, out_path):
    methods = ["sparse", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    labels  = ["BGE-M3 Sparse", "BGE-M3 Dense", "Static wRRF (α=0.5)",
               "wRRF (weak)", "wRRF (strong)", "MoE Meta-Learner"]
    colors  = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#B47CC7", "#956CB4"]

    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    width   = 0.12
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2, len(methods)) * width
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
    ax.set_title(
        f"BGE-M3 Dual-Mode Retrieval Comparison — NDCG@{ndcg_k}",
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


def _save_decision_boundary_plot(
    final_mdl, model_name,
    td_aw, td_as, td_ds_names,
    te_aw, te_as, te_ds_names,
    out_path,
):
    grid_size = 100
    aw_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    as_vals = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    AW, AS  = np.meshgrid(aw_vals, as_vals)

    X_grid = _meta_features(AW.ravel(), AS.ravel(), model_name)
    Z      = np.clip(final_mdl.predict(X_grid), 0.0, 1.0).reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(AS, AW, Z, levels=30, cmap="RdYlBu_r", alpha=0.85)
    plt.colorbar(cf, ax=ax,
                 label="Predicted α   (1 = prefer BGE-M3 Sparse,  0 = prefer Dense)")

    for ds_name, color in DS_PALETTE.items():
        mask_td = np.array([d == ds_name for d in td_ds_names])
        mask_te = np.array([d == ds_name for d in te_ds_names])
        if mask_td.any():
            ax.scatter(td_as[mask_td], td_aw[mask_td],
                       c=color, s=12, alpha=0.30, edgecolors="none", zorder=3)
        if mask_te.any():
            ax.scatter(te_as[mask_te], te_aw[mask_te],
                       c=color, s=40, alpha=0.85,
                       edgecolors="k", linewidths=0.4,
                       label=ds_name, zorder=4)

    ax.set_xlabel("α_strong  (strong-signal XGBoost prediction)", fontsize=10)
    ax.set_ylabel("α_weak   (weak-signal XGBoost prediction)",    fontsize=10)
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
    n_jobs = 1 if cuda_available else int(
        cfg.get("meta_learner_moe", {}).get("n_jobs", -1)
    )
    print(f"Device: {device}  |  grid-search n_jobs: {n_jobs}")

    dataset_names = cfg["datasets"]
    ndcg_k        = int(cfg["benchmark"]["ndcg_k"])
    rrf_k         = int(cfg["benchmark"]["rrf"]["k"])

    # Experiment settings (reuse meta_learner_moe block)
    exp_cfg      = cfg["meta_learner_moe"]
    n_queries    = int(exp_cfg["n_queries"])
    trunc_seed   = int(exp_cfg["truncation_seed"])
    test_frac    = float(exp_cfg["test_fraction"])
    dev_frac_td  = float(exp_cfg["dev_fraction_of_traindev"])
    n_cv_rounds  = int(exp_cfg["n_cv_rounds"])
    n_oof_folds  = int(exp_cfg["n_oof_folds"])
    meta_grid    = exp_cfg["grid"]

    weak_grid_cfg   = cfg["xgboost_params_grid"]
    strong_grid_cfg = cfg["strong_signal_params_grid"]
    n_folds_gs      = int(weak_grid_cfg.get("n_folds", 10))

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    meta_cache = os.path.join(results_folder, "bge_dual_meta_dataset.csv")

    # ── Phase 1-3: Load datasets ──────────────────────────────────────────────
    print("\n=== Phase 1-3: Loading datasets ===")
    all_raw        = {}
    retrieval_data = {}

    for ds_name in dataset_names:
        print(f"\n  [{ds_name}]")
        wd = load_dataset_bge_dual(ds_name, cfg, device)
        sd = load_strong_embeddings(ds_name, cfg, device)
        all_raw[ds_name] = {"wd": wd, "sd": sd}
        retrieval_data[ds_name] = {
            "sparse_results": wd["sparse_results"],
            "dense_results":  wd["dense_results"],
            "qrels":          wd["qrels"],
        }

    if cuda_available:
        torch.cuda.empty_cache()

    # ── Phase 4: Merge and split ──────────────────────────────────────────────
    print("\n=== Phase 4: Merging 5×300 pool and splitting ===")

    X_weak_td_parts, X_strong_td_parts = [], []
    y_td_parts, qids_td, ds_td          = [], [], []
    X_weak_te_parts, X_strong_te_parts = [], []
    y_te_parts, qids_te, ds_te          = [], [], []
    meta_td, meta_te                    = [], []

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
        for i in td_idx:
            qids_td.append(qids_full[i])
            ds_td.append(ds_name)
            meta_td.append((ds_name, qids_full[i]))

        X_weak_te_parts.append(X_weak_full[te_idx])
        X_strong_te_parts.append(X_strong_full[te_idx])
        y_te_parts.append(y_full[te_idx])
        for i in te_idx:
            qids_te.append(qids_full[i])
            ds_te.append(ds_name)
            meta_te.append((ds_name, qids_full[i]))

        print(f"  {ds_name}: {n_td_ds} traindev | {n_test} test")

    X_weak_td   = np.vstack(X_weak_td_parts)
    X_strong_td = np.vstack(X_strong_td_parts)
    y_td        = np.concatenate(y_td_parts).astype(np.float32)
    X_weak_te   = np.vstack(X_weak_te_parts)
    X_strong_te = np.vstack(X_strong_te_parts)
    y_te        = np.concatenate(y_te_parts).astype(np.float32)

    print(f"\n  Merged traindev: {len(y_td)}  |  test: {len(y_te)}")
    print(f"  Weak dim: {X_weak_td.shape[1]}  |  Strong dim: {X_strong_td.shape[1]}")

    # ── Phase 5: Weak XGBoost grid search ─────────────────────────────────────
    print("\n=== Phase 5: Weak XGBoost grid search (15 features) ===")
    weak_params, weak_cv = _xgb_grid_search_kfold(
        X_weak_td, y_td, meta_td,
        weak_grid_cfg, n_folds_gs,
        retrieval_data, rrf_k, ndcg_k,
        n_jobs, seed, "weak",
    )
    weak_params_out = os.path.join(results_folder, "bge_dual_weak_best_params.csv")
    with open(weak_params_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for k, v in weak_params.items():
            w.writerow([k, v])
        w.writerow([f"cv_ndcg@{ndcg_k}", f"{weak_cv:.6f}"])
    print(f"  Saved: {weak_params_out}")

    # ── Phase 6: Strong XGBoost grid search ───────────────────────────────────
    print("\n=== Phase 6: Strong XGBoost grid search (1024-dim embeddings) ===")
    strong_params, strong_cv = _xgb_grid_search_kfold(
        X_strong_td, y_td, meta_td,
        strong_grid_cfg, n_folds_gs,
        retrieval_data, rrf_k, ndcg_k,
        n_jobs, seed, "strong",
    )
    strong_params_out = os.path.join(results_folder, "bge_dual_strong_best_params.csv")
    with open(strong_params_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for k, v in strong_params.items():
            w.writerow([k, v])
        w.writerow([f"cv_ndcg@{ndcg_k}", f"{strong_cv:.6f}"])
    print(f"  Saved: {strong_params_out}")

    # ── Phase 7: OOF meta-dataset ─────────────────────────────────────────────
    if os.path.exists(meta_cache):
        print(f"\n=== Phase 7: Loading meta-dataset from cache ===")
        cached = _load_meta_dataset(meta_cache)

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
                f"Cache at '{meta_cache}' does not match the current query split "
                f"(missing key {e}). Delete it and re-run."
            ) from None
    else:
        print("\n=== Phase 7: Building OOF meta-dataset ===")
        td_aw, td_as = _build_oof_predictions(
            X_weak_td, X_strong_td, y_td,
            n_oof_folds, weak_params, strong_params, seed, xgb_device,
        )
        td_gt = y_td.copy()

        print("  Training final base models on full traindev ...")
        mdl_w, mu_w, sig_w, mdl_s, mu_s, sig_s = _train_base_models(
            X_weak_td, X_strong_td, y_td,
            weak_params, strong_params, seed, xgb_device,
        )
        te_aw = np.clip(mdl_w.predict((X_weak_te   - mu_w) / sig_w), 0.0, 1.0).astype(np.float32)
        te_as = np.clip(mdl_s.predict((X_strong_te - mu_s) / sig_s), 0.0, 1.0).astype(np.float32)

        rows = []
        for i, (d, q) in enumerate(zip(ds_td, qids_td)):
            rows.append({"ds_name": d, "qid": q, "split": "traindev",
                         "alpha_weak": float(td_aw[i]), "alpha_strong": float(td_as[i]),
                         "alpha_gt": float(td_gt[i])})
        for i, (d, q) in enumerate(zip(ds_te, qids_te)):
            rows.append({"ds_name": d, "qid": q, "split": "test",
                         "alpha_weak": float(te_aw[i]), "alpha_strong": float(te_as[i]),
                         "alpha_gt": float(y_te[i])})
        _save_meta_dataset(rows, meta_cache)
        print(f"  Meta-dataset cached: {meta_cache}")

    # ── Phase 8: MoE meta-learner grid search ─────────────────────────────────
    print("\n=== Phase 8: MoE meta-learner grid search ===")
    mc_folds = _mc_fold_indices(len(td_gt), dev_frac_td, n_cv_rounds, seed)
    print(f"  MC CV: {n_cv_rounds} rounds  "
          f"(train≈{len(mc_folds[0][0])}, dev≈{len(mc_folds[0][1])} per round)")

    meta_combos = _build_meta_combos(meta_grid)
    print(f"  {len(meta_combos)} combos × {n_cv_rounds} rounds")

    meta_scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_eval_meta_combo)(
            model_name, params,
            td_aw, td_as, td_gt,
            qids_td, ds_td,
            mc_folds, retrieval_data, rrf_k, ndcg_k,
        )
        for model_name, params in meta_combos
    )

    best_meta_idx   = int(np.argmax(meta_scores))
    best_meta_name  = meta_combos[best_meta_idx][0]
    best_meta_params = meta_combos[best_meta_idx][1]
    best_meta_cv    = float(meta_scores[best_meta_idx])

    print(f"\n  Best meta-model       : {best_meta_name}")
    print(f"  Best meta-params      : {best_meta_params}")
    print(f"  Best meta CV NDCG@{ndcg_k} : {best_meta_cv:.4f}")

    meta_params_out = os.path.join(results_folder, "bge_dual_meta_best_params.csv")
    with open(meta_params_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_json", f"cv_ndcg@{ndcg_k}"])
        w.writerow([best_meta_name, json.dumps(best_meta_params), f"{best_meta_cv:.6f}"])
    print(f"  Saved: {meta_params_out}")

    # ── Phase 9: Final evaluation ─────────────────────────────────────────────
    print("\n=== Phase 9: Training final meta-learner and evaluating ===")
    X_td_meta = _meta_features(td_aw, td_as, best_meta_name)
    X_te_meta = _meta_features(te_aw, te_as, best_meta_name)
    final_mdl = _fit_meta(best_meta_name, best_meta_params, X_td_meta, td_gt)
    te_moe    = _pred_meta(final_mdl, X_te_meta)

    print("\n=== Per-dataset evaluation (15% held-out test) ===")
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
        sp_s     = _sparse_ndcg(te_qids_ds, rd["sparse_results"], rd["qrels"], ndcg_k)
        de_s     = _dense_ndcg( te_qids_ds, rd["dense_results"],  rd["qrels"], ndcg_k)
        srrf_s   = _wrrf_ndcg(srrf_ds, te_qids_ds,
                               rd["sparse_results"], rd["dense_results"],
                               rd["qrels"], rrf_k, ndcg_k)
        wrrf_w_s = _wrrf_ndcg(aw_ds,   te_qids_ds,
                               rd["sparse_results"], rd["dense_results"],
                               rd["qrels"], rrf_k, ndcg_k)
        wrrf_s_s = _wrrf_ndcg(as_ds,   te_qids_ds,
                               rd["sparse_results"], rd["dense_results"],
                               rd["qrels"], rrf_k, ndcg_k)
        moe_s    = _wrrf_ndcg(moe_ds,  te_qids_ds,
                               rd["sparse_results"], rd["dense_results"],
                               rd["qrels"], rrf_k, ndcg_k)

        print(f"\n  {ds_name}  ({mask.sum()} test queries):")
        print(f"    BGE-M3 Sparse   : {sp_s:.4f}")
        print(f"    BGE-M3 Dense    : {de_s:.4f}")
        print(f"    Static wRRF     : {srrf_s:.4f}")
        print(f"    wRRF (weak)     : {wrrf_w_s:.4f}")
        print(f"    wRRF (strong)   : {wrrf_s_s:.4f}")
        print(f"    MoE             : {moe_s:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "sparse":      sp_s,
            "dense":       de_s,
            "static_rrf":  srrf_s,
            "wrrf_weak":   wrrf_w_s,
            "wrrf_strong": wrrf_s_s,
            "moe":         moe_s,
        })

    macro = {
        m: float(np.mean([r[m] for r in comparison_rows]))
        for m in ["sparse", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
    }
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'='*60}")
    print("Macro averages:")
    for lbl, key in [
        ("BGE-M3 Sparse",  "sparse"),
        ("BGE-M3 Dense",   "dense"),
        ("Static wRRF",    "static_rrf"),
        ("wRRF (weak)",    "wrrf_weak"),
        ("wRRF (strong)",  "wrrf_strong"),
        ("MoE",            "moe"),
    ]:
        print(f"  {lbl:<16} NDCG@{ndcg_k} = {macro[key]:.4f}")

    # ── Phase 10: Save outputs ────────────────────────────────────────────────
    comp_csv = os.path.join(results_folder, "bge_dual_retrieval_comparison.csv")
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            f"sparse_ndcg@{ndcg_k}",     f"dense_ndcg@{ndcg_k}",
            f"static_rrf_ndcg@{ndcg_k}", f"wrrf_weak_ndcg@{ndcg_k}",
            f"wrrf_strong_ndcg@{ndcg_k}", f"moe_ndcg@{ndcg_k}",
        ])
        for r in comparison_rows:
            w.writerow([
                r["group"],
                f"{r['sparse']:.6f}",     f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}", f"{r['wrrf_weak']:.6f}",
                f"{r['wrrf_strong']:.6f}", f"{r['moe']:.6f}",
            ])
    print(f"\nCSV saved: {comp_csv}")

    comp_png = os.path.join(results_folder, "bge_dual_retrieval_comparison.png")
    _save_comparison_plot(comparison_rows, ndcg_k, comp_png)

    boundary_png = os.path.join(results_folder, "bge_dual_decision_boundary.png")
    _save_decision_boundary_plot(
        final_mdl, best_meta_name,
        td_aw, td_as, ds_td,
        te_aw, te_as, ds_te,
        boundary_png,
    )

    print("\nBGE-M3 dual-mode pipeline complete.")
    print(f"All outputs in: {results_folder}")


if __name__ == "__main__":
    main()
