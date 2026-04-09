"""
weak_signal_model_grid_search.py

Grid search across 9 model families to identify the best weak-signal
router for query-adaptive hybrid retrieval.

Models tested:
  Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM,
  MLP, Gaussian Naive Bayes, AdaBoost, LDA

For each (model, hyperparameter) combination the script runs 10
repeated random 80/20 train/test splits on every dataset, computes
the wRRF NDCG@10 on each test fold, and reports the macro average
across all datasets.  The top-N results are written to a single CSV.

Usage:
    python src/weak_signal_model_grid_search.py
"""

import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import util as st_util
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    ensure_dir,
    file_exists,
    get_config_path,
    load_config,
    load_pickle,
    load_queries,
    load_qrels,
    model_short_name,
    save_pickle,
    stem_and_tokenize,
)

# ── Constants ──────────────────────────────────────────────────

FEATURE_NAMES = [
    "query_length",
    "stopword_ratio",
    "has_question_word",
    "average_idf",
    "max_idf",
    "rare_term_ratio",
    "cross_entropy",
    "top_dense_score",
    "top_sparse_score",
    "dense_confidence",
    "sparse_confidence",
    "overlap_at_k",
    "first_shared_doc_rank",
    "spearman_topk",
    "dense_entropy_topk",
    "sparse_entropy_topk",
]

QUESTION_WORDS = frozenset(
    ["who", "what", "when", "where", "why", "how", "which", "whose", "whom"]
)

# Model families that use binarised labels + predict_proba for alpha.
CLASSIFIER_MODELS = {"logistic_regression", "gaussian_nb", "lda"}


# ── Reproducibility ───────────────────────────────────────────

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_seed_offset(name):
    return int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)


# ── NLTK stopwords ────────────────────────────────────────────

def ensure_english_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


# ── Score helpers ─────────────────────────────────────────────

def normalize_scores_minmax(pairs, epsilon=1e-8):
    if not pairs:
        return []
    scores = [float(s) for _, s in pairs]
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng < epsilon:
        return [(d, 0.0) for d, _ in pairs]
    return [(d, (float(s) - lo) / (rng + epsilon)) for d, s in pairs]


def query_ndcg_at_k(ranked_pairs, rels, k):
    if not rels:
        return 0.0
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(ranked_pairs[:k], start=1):
        gain = (2.0 ** rels.get(doc_id, 0)) - 1.0
        dcg += gain / math.log2(rank + 1)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(((2.0 ** r) - 1.0) / math.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ── Feature computation (one query) ──────────────────────────

def compute_query_features(
    raw_text,
    query_tokens,
    bm25_pairs,
    dense_pairs,
    word_freq,
    total_corpus_tokens,
    doc_freq,
    total_docs,
    stopword_stems,
    overlap_k,
    feature_stat_k,
    epsilon,
    ce_alpha,
):
    bm25_norm = normalize_scores_minmax(bm25_pairs, epsilon)
    dense_norm = normalize_scores_minmax(dense_pairs, epsilon)
    n_tok = len(query_tokens)

    # ── Group A: Query Surface ────────────────────────────────
    query_length = float(n_tok)

    if n_tok == 0:
        stopword_ratio = 0.0
        clean = []
    else:
        n_stop = sum(1 for t in query_tokens if t in stopword_stems)
        stopword_ratio = n_stop / n_tok
        clean = [t for t in query_tokens if t not in stopword_stems]

    first_word = raw_text.strip().split()[0].lower() if raw_text.strip() else ""
    has_qw = 1.0 if first_word in QUESTION_WORDS else 0.0

    # ── Group B: Query-Corpus Vocabulary Match ────────────────
    vocab_size = max(1, len(word_freq))
    corpus_mass = total_corpus_tokens + ce_alpha * vocab_size

    if not clean:
        cross_entropy = average_idf = max_idf = rare_term_ratio = 0.0
    else:
        ce_sum = 0.0
        idfs = []
        for t in clean:
            prob = (word_freq.get(t, 0) + ce_alpha) / corpus_mass
            ce_sum += -math.log2(max(prob, epsilon))
            idfs.append(math.log((total_docs + 1.0) / (doc_freq.get(t, 0) + 1.0)) + 1.0)
        cross_entropy = ce_sum / len(clean)
        average_idf = sum(idfs) / len(idfs)
        max_idf = max(idfs)
        idf_std = float(np.std(idfs))
        thresh = average_idf + idf_std
        rare_term_ratio = sum(1 for v in idfs if v >= thresh) / len(idfs)

    # ── Group C: Retriever Confidence ─────────────────────────
    def _confidence(normed):
        if len(normed) >= 2:
            return normed[0][1] - normed[1][1]
        return normed[0][1] if normed else 0.0

    dense_confidence = _confidence(dense_norm)
    sparse_confidence = _confidence(bm25_norm)
    top_dense_score = dense_norm[0][1] if dense_norm else 0.0
    top_sparse_score = bm25_norm[0][1] if bm25_norm else 0.0

    # ── Group D: Retriever Agreement ──────────────────────────
    top_sp = [d for d, _ in bm25_pairs[:overlap_k]]
    top_de = [d for d, _ in dense_pairs[:overlap_k]]
    overlap_at_k = len(set(top_sp) & set(top_de)) / max(1, overlap_k)

    sp_rank = {d: r for r, d in enumerate([d for d, _ in bm25_pairs[:feature_stat_k]], 1)}
    de_rank = {d: r for r, d in enumerate([d for d, _ in dense_pairs[:feature_stat_k]], 1)}
    shared = set(sp_rank) & set(de_rank)

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

    # ── Group E: Ranking Distribution Shape ───────────────────
    def _entropy(normed_pairs, k):
        pairs = normed_pairs[:k]
        if not pairs:
            return 0.0
        scores = np.array([max(s, 0.0) for _, s in pairs], dtype=np.float64)
        total = scores.sum()
        if total <= epsilon:
            return 0.0
        p = scores / total
        return float(-np.sum(p * np.log2(np.maximum(p, epsilon))))

    dense_entropy_topk = _entropy(dense_norm, feature_stat_k)
    sparse_entropy_topk = _entropy(bm25_norm, feature_stat_k)

    return {
        "query_length": float(query_length),
        "stopword_ratio": float(stopword_ratio),
        "has_question_word": float(has_qw),
        "average_idf": float(average_idf),
        "max_idf": float(max_idf),
        "rare_term_ratio": float(rare_term_ratio),
        "cross_entropy": float(cross_entropy),
        "top_dense_score": float(top_dense_score),
        "top_sparse_score": float(top_sparse_score),
        "dense_confidence": float(dense_confidence),
        "sparse_confidence": float(sparse_confidence),
        "overlap_at_k": float(overlap_at_k),
        "first_shared_doc_rank": float(first_shared_doc_rank),
        "spearman_topk": float(spearman_topk),
        "dense_entropy_topk": float(dense_entropy_topk),
        "sparse_entropy_topk": float(sparse_entropy_topk),
    }


# ── BM25 / dense retrieval ───────────────────────────────────

def run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k, use_stemming):
    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
    results = {}
    for qid, text in tqdm(queries.items(), desc="  BM25 retrieval", dynamic_ncols=True):
        tokens = stem_and_tokenize(text, stemmer)
        scores = bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k <= 0:
            results[qid] = []
            continue
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in idx]
    return results


def run_dense_retrieval(q_vecs, q_ids, c_vecs, c_ids, top_k, cfg):
    dense_cfg = cfg.get("dense_search", {})
    q_chunk = dense_cfg.get("query_chunk_size", 100)
    c_chunk = dense_cfg.get("corpus_chunk_size", 50000)
    device = c_vecs.device
    results = {}
    for start in tqdm(range(0, len(q_ids), q_chunk), desc="  Dense retrieval", dynamic_ncols=True):
        end = min(start + q_chunk, len(q_ids))
        batch = q_vecs[start:end].to(device)
        hits = st_util.semantic_search(batch, c_vecs, top_k=top_k, corpus_chunk_size=c_chunk)
        for i, hit_list in enumerate(hits):
            qid = q_ids[start + i]
            results[qid] = [(c_ids[h["corpus_id"]], float(h["score"])) for h in hit_list]
    return results


# ── Dataset loading ───────────────────────────────────────────

def load_dataset_for_grid_search(dataset_name, cfg, device):
    """Load preprocessing artifacts, compute/cache retrieval results, build
    feature matrix X, label vector y, and auxiliary data for one dataset."""
    short_model = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir = os.path.join(processed_root, short_model, dataset_name)
    top_k = int(cfg["benchmark"]["top_k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    bm25_params = u.get_bm25_params(cfg)
    bm25_paths = u.bm25_artifact_paths(
        ds_dir, bm25_params["k1"], bm25_params["b"],
        bm25_params["use_stemming"], top_k=top_k,
    )
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]

    # ── Paths ─────────────────────────────────────────────────
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vecs_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")
    bm25_results_pkl = bm25_paths["bm25_results_pkl"]
    dense_results_pkl = os.path.join(ds_dir, f"dense_results_topk_{top_k}.pkl")

    # ── Raw queries + stemmed tokens ──────────────────────────
    raw_queries = load_queries(queries_jsonl)
    qrels = load_qrels(qrels_tsv)

    query_tokens = load_pickle(bm25_paths["query_tokens_pkl"])
    if not isinstance(query_tokens, dict):
        stemmer = SnowballStemmer(stemmer_lang) if bm25_params["use_stemming"] else None
        query_tokens = {qid: stem_and_tokenize(t, stemmer) for qid, t in raw_queries.items()}

    # ── Frequency indexes ─────────────────────────────────────
    word_freq, total_corpus_tokens = load_pickle(bm25_paths["word_freq_pkl"])
    doc_freq, total_docs = load_pickle(bm25_paths["doc_freq_pkl"])

    # ── BM25 retrieval (cached) ───────────────────────────────
    if file_exists(bm25_results_pkl):
        print(f"  Loading cached BM25 results for {dataset_name}")
        bm25_results = load_pickle(bm25_results_pkl)
    else:
        print(f"  Running BM25 retrieval for {dataset_name} ...")
        bm25 = load_pickle(bm25_paths["bm25_pkl"])
        bm25_doc_ids = load_pickle(bm25_paths["bm25_docids_pkl"])
        bm25_results = run_bm25_retrieval(
            bm25, bm25_doc_ids, raw_queries,
            stemmer_lang, top_k, bm25_params["use_stemming"],
        )
        save_pickle(bm25_results, bm25_results_pkl)

    # ── Dense retrieval (cached) ──────────────────────────────
    if file_exists(dense_results_pkl):
        print(f"  Loading cached dense results for {dataset_name}")
        dense_results = load_pickle(dense_results_pkl)
    else:
        print(f"  Running dense retrieval for {dataset_name} ...")
        corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
        if device.type == "cuda":
            try:
                corpus_embeddings = corpus_embeddings.to(device)
            except torch.cuda.OutOfMemoryError:
                print("  [WARN] CUDA OOM; falling back to CPU for dense retrieval.")
                torch.cuda.empty_cache()
        corpus_ids = load_pickle(corpus_ids_pkl)
        q_vecs = torch.load(query_vecs_pt, weights_only=True)
        q_ids = load_pickle(query_ids_pkl)
        dense_results = run_dense_retrieval(q_vecs, q_ids, corpus_embeddings, corpus_ids, top_k, cfg)
        save_pickle(dense_results, dense_results_pkl)

    # ── Features + soft labels ────────────────────────────────
    routing_cfg = cfg.get("routing_features", {})
    overlap_k = int(routing_cfg.get("overlap_k", 10))
    feature_stat_k = int(routing_cfg.get("feature_stat_k", 10))
    epsilon = float(routing_cfg.get("epsilon", 1e-8))
    ce_alpha = float(routing_cfg.get("ce_smoothing_alpha", 1.0))
    use_stemming = bm25_params["use_stemming"]

    # Feature/label cache keyed by all parameters that affect X and y.
    _cache_key_str = json.dumps({
        "bm25": bm25_paths["bm25_signature"],
        "top_k": top_k,
        "ndcg_k": ndcg_k,
        "overlap_k": overlap_k,
        "feature_stat_k": feature_stat_k,
        "epsilon": epsilon,
        "ce_alpha": ce_alpha,
    }, sort_keys=True)
    _feature_hash = hashlib.md5(_cache_key_str.encode()).hexdigest()[:12]
    features_cache_pkl = os.path.join(ds_dir, f"features_labels_{_feature_hash}.pkl")

    if file_exists(features_cache_pkl):
        print(f"  Loading cached features+labels for {dataset_name}")
        _cached = load_pickle(features_cache_pkl)
        return {
            "X": _cached["X"],
            "y": _cached["y"],
            "qids": _cached["qids"],
            "bm25_results": bm25_results,
            "dense_results": dense_results,
            "qrels": qrels,
        }

    english_sw = ensure_english_stopwords()
    if use_stemming:
        stemmer = SnowballStemmer(stemmer_lang)
        stopword_stems = frozenset(stemmer.stem(w) for w in english_sw)
    else:
        stopword_stems = frozenset(w.lower() for w in english_sw)

    qids = sorted(raw_queries.keys())
    feat_rows = []
    labels = []

    for qid in tqdm(qids, desc=f"  Features {dataset_name}", dynamic_ncols=True):
        feat = compute_query_features(
            raw_queries[qid],
            query_tokens.get(qid, []),
            bm25_results.get(qid, []),
            dense_results.get(qid, []),
            word_freq, total_corpus_tokens, doc_freq, total_docs,
            stopword_stems, overlap_k, feature_stat_k, epsilon, ce_alpha,
        )
        feat_rows.append([feat[name] for name in FEATURE_NAMES])

        # Soft label
        sp_ndcg = query_ndcg_at_k(bm25_results.get(qid, []), qrels.get(qid, {}), ndcg_k)
        de_ndcg = query_ndcg_at_k(dense_results.get(qid, []), qrels.get(qid, {}), ndcg_k)
        label = 0.5 * ((sp_ndcg - de_ndcg) / (max(sp_ndcg, de_ndcg) + epsilon) + 1.0)
        labels.append(float(np.clip(label, 0.0, 1.0)))

    X = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)

    save_pickle({"X": X, "y": y, "qids": qids}, features_cache_pkl)

    return {
        "X": X,
        "y": y,
        "qids": qids,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "qrels": qrels,
    }


# ── Z-score normalisation ────────────────────────────────────

def zscore_stats(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma <= 1e-12] = 1.0
    return mu, sigma


# ── Model factory ─────────────────────────────────────────────

def create_model(model_name, params, seed):
    """Instantiate one model. All tree-based models use n_jobs=1 to avoid
    nested parallelism (the outer loop already uses multiple threads)."""

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=params["C"], penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=seed, n_jobs=1,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed, n_jobs=1,
        )

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed, n_jobs=1,
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
            verbosity=0, random_state=seed, n_jobs=1,
        )

    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="binary",
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            subsample_freq=1,
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            verbose=-1, random_state=seed, n_jobs=1,
        )

    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation=params["activation"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            batch_size=params["batch_size"],
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    if model_name == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=params["var_smoothing"])

    if model_name == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        base = DecisionTreeRegressor(
            max_depth=params["base_max_depth"], random_state=seed,
        )
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=seed,
        )

    if model_name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        kw = {"solver": params["solver"]}
        if params["solver"] != "svd" and params.get("shrinkage") is not None:
            kw["shrinkage"] = params["shrinkage"]
        return LinearDiscriminantAnalysis(**kw)

    raise ValueError(f"Unknown model: {model_name}")


def predict_alpha(model, X, model_name):
    """Return predicted alpha values in [0, 1]."""
    if model_name in CLASSIFIER_MODELS:
        proba = model.predict_proba(X)
        # predict_proba columns are [P(class 0), P(class 1)]
        return proba[:, 1].astype(np.float32)
    preds = model.predict(X).astype(np.float32)
    return np.clip(preds, 0.0, 1.0)


# ── Single-combination evaluation ─────────────────────────────

def evaluate_combination(
    model_name, params, datasets_data, fold_indices, seed, rrf_k, ndcg_k,
):
    """Train and evaluate one (model, params) tuple across all datasets
    and folds.  Returns a result dict with macro and per-dataset NDCG@10."""
    per_ds = {}

    for ds_name, ds in datasets_data.items():
        X, y = ds["X"], ds["y"]
        qids = ds["qids"]
        bm25_res = ds["bm25_results"]
        dense_res = ds["dense_results"]
        qrels = ds["qrels"]

        fold_ndcgs = []
        for train_idx, test_idx in fold_indices[ds_name]:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]
            test_qids = [qids[i] for i in test_idx]

            # Z-score (fit on train only)
            mu, sigma = zscore_stats(X_tr)
            X_tr_z = (X_tr - mu) / sigma
            X_te_z = (X_te - mu) / sigma

            y_fit = (y_tr >= 0.5).astype(int) if model_name in CLASSIFIER_MODELS else y_tr

            try:
                mdl = create_model(model_name, params, seed)
                mdl.fit(X_tr_z, y_fit)
                alphas = predict_alpha(mdl, X_te_z, model_name)
            except (ValueError, np.linalg.LinAlgError):
                # Degenerate fold (e.g. single class for a classifier); treat as RRF
                alphas = np.full(len(test_qids), 0.5, dtype=np.float32)

            # wRRF NDCG@10 for test queries
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
        "model": model_name,
        "params": params,
        "macro_ndcg10": macro,
        "per_dataset": per_ds,
    }


# ── Grid construction ─────────────────────────────────────────

def build_param_grid(model_name, grid_cfg):
    """Generate all valid hyperparameter dicts for one model."""
    keys = sorted(grid_cfg.keys())
    values = [grid_cfg[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        params = dict(zip(keys, vals))

        # LDA: shrinkage only valid for lsqr/eigen
        if model_name == "lda":
            if params["solver"] == "svd" and params.get("shrinkage") is not None:
                continue

        combos.append(params)
    return combos


# ── Main ──────────────────────────────────────────────────────

def main():
    # Suppress non-critical warnings during mass training
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
    rrf_k = int(cfg["benchmark"]["rrf"]["k"])

    gs_cfg = cfg["model_grid_search"]
    n_folds = int(gs_cfg["n_folds"])
    train_frac = float(gs_cfg["train_fraction"])
    top_n = int(gs_cfg["top_n"])
    n_jobs = int(gs_cfg.get("n_jobs", -1))

    # ── Load datasets ─────────────────────────────────────────
    print("\n=== Loading datasets ===")
    datasets_data = {}
    for ds in dataset_names:
        print(f"\n--- {ds} ---")
        datasets_data[ds] = load_dataset_for_grid_search(ds, cfg, device)
        n_q = len(datasets_data[ds]["qids"])
        print(f"  {n_q} queries, {datasets_data[ds]['X'].shape[1]} features")

    # Free GPU memory after retrieval
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Precompute fold indices ───────────────────────────────
    fold_indices = {}
    for ds in dataset_names:
        n_q = len(datasets_data[ds]["qids"])
        folds = []
        for fi in range(n_folds):
            rng = np.random.RandomState(seed + fi * 1000 + dataset_seed_offset(ds))
            perm = rng.permutation(n_q)
            n_train = int(train_frac * n_q)
            n_train = max(1, min(n_q - 1, n_train))
            folds.append((perm[:n_train], perm[n_train:]))
        fold_indices[ds] = folds

    # ── Build grids ───────────────────────────────────────────
    model_cfgs = gs_cfg["models"]
    model_order = [
        "logistic_regression", "gaussian_nb", "lda",
        "random_forest", "extra_trees", "adaboost",
        "xgboost", "lightgbm", "mlp",
    ]
    # Only iterate over models present in config
    model_order = [m for m in model_order if m in model_cfgs]

    total_combos = 0
    for m in model_order:
        total_combos += len(build_param_grid(m, model_cfgs[m]))
    print(f"\n=== Grid search: {total_combos} total combinations ===")

    # ── Evaluate ──────────────────────────────────────────────
    all_results = []
    t0_total = time.time()

    for model_name in model_order:
        combos = build_param_grid(model_name, model_cfgs[model_name])
        n_combos = len(combos)
        print(f"\n[{model_name}] {n_combos} combinations ...", flush=True)
        t0 = time.time()

        batch = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(evaluate_combination)(
                model_name, p, datasets_data, fold_indices, seed, rrf_k, ndcg_k,
            )
            for p in combos
        )

        all_results.extend(batch)
        best = max(batch, key=lambda r: r["macro_ndcg10"])
        elapsed = time.time() - t0
        print(
            f"  Done in {elapsed:.1f}s | "
            f"Best macro NDCG@10 = {best['macro_ndcg10']:.4f}"
        )

    total_time = time.time() - t0_total
    print(f"\nTotal grid search time: {total_time:.1f}s")

    # ── Sort and save ─────────────────────────────────────────
    all_results.sort(key=lambda r: r["macro_ndcg10"], reverse=True)
    top = all_results[:top_n]

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)
    csv_path = os.path.join(results_folder, "model_grid_search_top100.csv")

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

    # Quick summary table
    print(f"\n{'Rank':<5} {'Model':<25} {'Macro NDCG@10':>14}")
    print("-" * 46)
    for rank, r in enumerate(top[:20], 1):
        print(f"{rank:<5} {r['model']:<25} {r['macro_ndcg10']:>14.4f}")
    if len(top) > 20:
        print(f"  ... ({len(top) - 20} more rows in CSV)")


if __name__ == "__main__":
    main()
