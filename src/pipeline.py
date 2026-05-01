"""
pipeline.py
===========

End-to-end orchestrator for the 22-step Query-Adaptive Hybrid Retrieval
experiment across 6 BEIR datasets.  The pipeline is implemented as a sequence of
``step_NN_xxx`` functions, each of which:

  1. Prints a clear header with the step number and title.
  2. Checks whether its outputs already exist on disk and skips
     itself if so (every step is therefore idempotent / resumable).
  3. Otherwise computes its outputs, caches them under ``data/...``
     and prints a brief summary.

Run from project root:

    python src/pipeline.py

All reproducibility-critical choices (random seeds, query split
fractions, grid sizes, model lists, etc.) live in ``config.yaml``.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure we always run from project root with src/ importable, regardless of
# launch directory.  Done before any project-internal imports.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import src.utils as u
from src.utils import (
    alpha_box_plot,
    alpha_sorted_plot,
    bm25_artifact_paths,
    dataset_seed_offset,
    download_beir_dataset,
    ensure_dir,
    ensure_english_stopwords,
    file_exists,
    get_config_path,
    grouped_bar_chart,
    init_stem_worker,
    is_nonempty_file,
    kfold_indices,
    load_beir_dataset,
    load_config,
    load_csv_dicts,
    load_full_corpus,
    load_json,
    load_pickle,
    load_qrels,
    load_queries,
    md5_of_obj,
    model_short_name,
    paired_t_test,
    query_ndcg_at_k,
    query_recall_at_k,
    run_bm25_retrieval,
    run_dense_retrieval,
    save_csv_dicts,
    save_json,
    save_pickle,
    stem_and_tokenize,
    stem_batch_worker,
    stratified_split,
    write_corpus_jsonl,
    write_qrels_tsv,
    write_queries_jsonl,
    wrrf_fuse,
    wrrf_top_k,
)


# ============================================================
# Constants
# ============================================================

# Names and order of the 16 hand-crafted weak-signal features.
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

FEATURE_GROUPS = {
    "A: Query Surface": [
        "query_length", "stopword_ratio", "has_question_word",
    ],
    "B: Vocabulary Match": [
        "average_idf", "max_idf", "rare_term_ratio", "cross_entropy",
    ],
    "C: Retriever Confidence": [
        "top_dense_score", "top_sparse_score",
        "dense_confidence", "sparse_confidence",
    ],
    "D: Retriever Agreement": [
        "overlap_at_k", "first_shared_doc_rank", "spearman_topk",
    ],
    "E: Distribution Shape": [
        "dense_entropy_topk", "sparse_entropy_topk",
    ],
}

QUESTION_WORDS = frozenset(
    ["who", "what", "when", "where", "why", "how", "which", "whose", "whom"]
)

# Models that produce probabilities for binary labels (logistic_regression /
# gaussian_nb / lda).  All other model factories return regressors that we
# clip to [0, 1].
CLASSIFIER_MODELS = {"logistic_regression", "gaussian_nb", "lda"}

# Method labels for retrieval comparisons.
METHOD_KEYS_6 = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
METHOD_LABELS_6 = [
    "BM25", "Dense", "Static RRF (α=0.5)",
    "wRRF (weak)", "wRRF (strong)", "MoE Meta-Learner",
]
METHOD_COLORS_6 = [
    "#4878D0", "#EE854A", "#6ACC65",
    "#D65F5F", "#B47CC7", "#956CB4",
]

# Dataset palette used in the MoE decision heatmap.
DS_PALETTE = {
    "scifact":    "#4878D0",
    "nfcorpus":   "#EE854A",
    "arguana":    "#6ACC65",
    "fiqa":       "#D65F5F",
    "scidocs":    "#B47CC7",
    "trec-covid": "#17BECF",
}


# ============================================================
# Common helpers (Section A)
# ============================================================

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_active_bm25_params(cfg: dict) -> dict:
    """Return BM25 params from the cached best.json (Step 3) if present,
    otherwise from cfg['bm25']."""
    results_root = get_config_path(cfg, "results_folder", "data/results")
    best_json = os.path.join(results_root, "bm25_best_params.json")
    if file_exists(best_json):
        d = load_json(best_json)
        return {"k1": float(d["k1"]), "b": float(d["b"]),
                "use_stemming": bool(d["use_stemming"])}
    raw = cfg.get("bm25", {}) or {}
    return {
        "k1":           float(raw.get("k1", 1.5)),
        "b":            float(raw.get("b",  0.75)),
        "use_stemming": bool(raw.get("use_stemming", True)),
    }


def dataset_processed_dir(cfg: dict, ds_name: str) -> str:
    short = model_short_name(cfg["embeddings"]["model_name"])
    root  = get_config_path(cfg, "processed_folder", "data/processed")
    return os.path.join(root, short, ds_name)


def zscore_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma <= 1e-12] = 1.0
    return mu, sigma


def make_model(model_name: str, params: dict, seed: int):
    """Single model factory used by every grid search.

    Tree-based models always use n_jobs=1 since the outer joblib loop
    handles concurrency.  XGBoost defers to its own device choice.
    """
    p = dict(params or {})  # local copy

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=float(p["C"]), penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=seed, n_jobs=1,
        )

    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(p["alpha"]), max_iter=5000)

    if model_name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=float(p["alpha"]), max_iter=5000)

    if model_name == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(
            alpha=float(p["alpha"]), l1_ratio=float(p["l1_ratio"]),
            max_iter=5000, random_state=seed,
        )

    if model_name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=int(p["n_neighbors"]),
            weights=str(p["weights"]),
            metric="euclidean", algorithm="auto", n_jobs=1,
        )

    if model_name == "svr":
        from sklearn.svm import SVR
        return SVR(
            kernel="rbf",
            C=float(p["C"]),
            gamma=p.get("gamma", "scale"),
            epsilon=float(p["epsilon"]),
            cache_size=500,
        )

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            min_samples_split=int(p.get("min_samples_split", 2)),
            max_features=p.get("max_features", "sqrt"),
            random_state=seed, n_jobs=1,
        )

    if model_name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=int(p["n_estimators"]),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            min_samples_split=int(p.get("min_samples_split", 2)),
            max_features=p.get("max_features", "sqrt"),
            random_state=seed, n_jobs=1,
        )

    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=tuple(int(x) for x in p["hidden_layer_sizes"]),
            activation=p.get("activation", "relu"),
            alpha=float(p.get("alpha", 1e-4)),
            learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
            batch_size=int(p.get("batch_size", 32)),
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )

    if model_name == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(var_smoothing=float(p.get("var_smoothing", 1e-9)))

    if model_name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        kw = {"solver": p.get("solver", "svd")}
        if kw["solver"] != "svd" and p.get("shrinkage") is not None:
            kw["shrinkage"] = p["shrinkage"]
        return LinearDiscriminantAnalysis(**kw)

    if model_name == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        base = DecisionTreeRegressor(
            max_depth=int(p.get("base_max_depth", 1)), random_state=seed,
        )
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            random_state=seed,
        )

    if model_name == "xgboost":
        import xgboost as xgb
        cuda = torch.cuda.is_available()
        return xgb.XGBRegressor(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda" if cuda else "cpu",
            verbosity=0,
            random_state=seed,
            n_jobs=1 if cuda else -1,
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            subsample=float(p.get("subsample", 1.0)),
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            min_child_weight=int(p.get("min_child_weight", 1)),
            gamma=float(p.get("gamma", 0.0)),
        )

    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=int(p.get("n_estimators", 100)),
            num_leaves=int(p.get("num_leaves", 31)),
            max_depth=None if p.get("max_depth") is None else int(p["max_depth"]),
            learning_rate=float(p.get("learning_rate", 0.1)),
            subsample=float(p.get("subsample", 1.0)),
            subsample_freq=1,
            colsample_bytree=float(p.get("colsample_bytree", 1.0)),
            min_child_weight=int(p.get("min_child_weight", 1)),
            random_state=seed, n_jobs=1, verbose=-1,
        )

    raise ValueError(f"Unknown model: {model_name!r}")


def predict_alpha_from_model(model, X, model_name: str) -> np.ndarray:
    """Return clipped alpha predictions for any supported model family."""
    if model_name in CLASSIFIER_MODELS:
        proba = model.predict_proba(X)
        return proba[:, 1].astype(np.float32)
    preds = model.predict(X).astype(np.float32)
    return np.clip(preds, 0.0, 1.0)


def expand_grid(grid_cfg: dict) -> List[Tuple[str, dict]]:
    """Expand a {model: {param: [v1, v2]}} grid into a list of (model, params)."""
    combos: List[Tuple[str, dict]] = []
    for model_name, axes in grid_cfg.items():
        keys   = sorted(axes.keys())
        values = [axes[k] for k in keys]
        for vals in itertools.product(*values):
            combos.append((model_name, dict(zip(keys, vals))))
    return combos


def truncate_combos(combos: List[Tuple[str, dict]], cap: int) -> List[Tuple[str, dict]]:
    """Truncate the combo list to `cap` if it exceeds it."""
    if cap and len(combos) > cap:
        print(f"[INFO] grid has {len(combos)} combos; truncating to cap={cap}.")
        return combos[:cap]
    return combos


# ============================================================
# Section B — Step implementations
# ============================================================

# ------------------------------------------------------------
# STEP 1 — Download
# ------------------------------------------------------------

def step_01_download(cfg: dict) -> None:
    print_step_header(1, "Download datasets")
    datasets         = cfg.get("datasets", []) or []
    datasets_folder  = get_config_path(cfg, "datasets_folder", "data/datasets")
    ensure_dir(datasets_folder)

    if not datasets:
        print("  No datasets configured. Skipping.")
        return

    for ds_name in datasets:
        ds_path = os.path.join(datasets_folder, ds_name)
        if os.path.isdir(ds_path):
            print(f"  [SKIP] {ds_name} already present at {ds_path}")
            continue
        print(f"  [DL]   {ds_name}")
        out = download_beir_dataset(ds_name, datasets_folder)
        if out is None or not os.path.isdir(out):
            raise RuntimeError(f"Failed to download {ds_name}")


# ------------------------------------------------------------
# STEP 2 — Preprocessing (corpus / queries, BM25, embeddings)
# ------------------------------------------------------------

def _preprocess_corpus_parallel(corpus_jsonl: str, output_jsonl: str,
                                stemmer_lang: str, use_stemming: bool,
                                batch_size: int = 512) -> None:
    """Stem-tokenise the corpus JSONL using a process pool."""
    ensure_dir(os.path.dirname(output_jsonl))
    n_workers   = max(1, (os.cpu_count() or 4) - 1)
    max_pending = n_workers * 3
    total_lines = u.count_lines(corpus_jsonl)
    total_batches = math.ceil(total_lines / batch_size)

    with open(output_jsonl, "w", encoding="utf-8") as out, \
         ProcessPoolExecutor(
             max_workers=n_workers,
             initializer=init_stem_worker,
             initargs=(stemmer_lang, use_stemming),
         ) as pool:

        pending: Dict = {}
        completed: Dict[int, list] = {}
        idx = 0
        next_to_write = 0
        pbar = tqdm(total=total_batches, desc="  Preprocessing corpus", dynamic_ncols=True)

        for batch in u.load_corpus_batch_generator(corpus_jsonl, batch_size):
            fut = pool.submit(stem_batch_worker, batch)
            pending[fut] = idx
            idx += 1
            while len(pending) >= max_pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut_done in done:
                    completed[pending.pop(fut_done)] = fut_done.result()
                while next_to_write in completed:
                    for line in completed.pop(next_to_write):
                        out.write(line + "\n")
                    pbar.update(1)
                    next_to_write += 1

        while pending:
            done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
            for fut_done in done:
                completed[pending.pop(fut_done)] = fut_done.result()
            while next_to_write in completed:
                for line in completed.pop(next_to_write):
                    out.write(line + "\n")
                pbar.update(1)
                next_to_write += 1
        pbar.close()


def _build_bm25_and_word_freq(tokenized_corpus_jsonl: str, k1: float, b: float):
    """Build BM25 index plus the global token frequency index in one pass."""
    from rank_bm25 import BM25Okapi

    doc_ids: List[str] = []
    tokenized_docs: List[List[str]] = []
    global_counts: Dict[str, int] = {}
    total_tokens = 0
    n_lines = u.count_lines(tokenized_corpus_jsonl)
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines, desc="  Loading tokenized corpus", dynamic_ncols=True):
            d = json.loads(line)
            doc_ids.append(d["_id"])
            tokens = d["tokens"]
            tokenized_docs.append(tokens)
            for t in tokens:
                global_counts[t] = global_counts.get(t, 0) + 1
            total_tokens += len(tokens)

    print(f"  Building BM25 index over {len(doc_ids):,} documents ...")
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    return bm25, doc_ids, global_counts, total_tokens


def _build_doc_freq(tokenized_corpus_jsonl: str):
    doc_freq: Dict[str, int] = {}
    total_docs = 0
    n_lines = u.count_lines(tokenized_corpus_jsonl)
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=n_lines, desc="  Document frequency", dynamic_ncols=True):
            d = json.loads(line)
            total_docs += 1
            for token in set(d.get("tokens", [])):
                doc_freq[token] = doc_freq.get(token, 0) + 1
    return doc_freq, total_docs


def _preprocess_queries(queries_jsonl: str, tokenized_queries_jsonl: str,
                        query_tokens_pkl: str, stemmer_lang: str, use_stemming: bool):
    if (is_nonempty_file(tokenized_queries_jsonl) and is_nonempty_file(query_tokens_pkl)):
        return
    from nltk.stem.snowball import SnowballStemmer
    queries = load_queries(queries_jsonl)
    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
    token_map: Dict[str, List[str]] = {}
    ensure_dir(os.path.dirname(tokenized_queries_jsonl))
    with open(tokenized_queries_jsonl, "w", encoding="utf-8") as out:
        for qid, qtext in queries.items():
            toks = stem_and_tokenize(qtext, stemmer)
            token_map[qid] = toks
            out.write(json.dumps({"_id": qid, "tokens": toks}) + "\n")
    save_pickle(token_map, query_tokens_pkl)


def _encode_with_oom_retry(model, texts, device, batch_size):
    """Embed *texts* with OOM-resilient batch shrinking and CPU fallback."""
    results = []
    start = 0
    bs = batch_size
    cur_device = device
    while start < len(texts):
        end = min(start + bs, len(texts))
        sub = texts[start:end]
        try:
            embs = model.encode(sub, convert_to_tensor=True,
                                show_progress_bar=False, device=cur_device)
            results.append(embs.cpu())
            start = end
            if cur_device == device:
                bs = batch_size
        except torch.cuda.OutOfMemoryError:
            try: torch.cuda.empty_cache()
            except Exception: pass
            bs = max(1, bs // 2)
            print(f"\n  [OOM] sub-batch -> {bs}")
            if bs == 1 and end - start == 1:
                raise
        except Exception as exc:
            if cur_device != "cpu" and "cuda" in str(exc).lower():
                print(f"\n  [CUDA ERROR] falling back to CPU: {exc}")
                cur_device = "cpu"
                bs = batch_size
            else:
                raise
    return results


def _build_corpus_embeddings(corpus_jsonl, model, batch_size, device):
    all_ids, all_embs = [], []
    total = u.count_lines(corpus_jsonl)
    for ids, texts in tqdm(
        u.load_corpus_batch_generator(corpus_jsonl, batch_size),
        total=math.ceil(total / batch_size),
        desc="  Encoding corpus", dynamic_ncols=True,
    ):
        embs_list = _encode_with_oom_retry(model, texts, device, batch_size)
        all_ids.extend(ids)
        all_embs.extend(embs_list)
    if not all_embs:
        raise ValueError("Corpus encoding produced no embeddings.")
    return torch.cat(all_embs, dim=0), all_ids


def _build_query_embeddings(queries: Dict[str, str], model, batch_size, device):
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    all_embs = []
    for i in tqdm(range(0, len(qtexts), batch_size),
                  desc="  Encoding queries", dynamic_ncols=True):
        sub = qtexts[i:i + batch_size]
        embs_list = _encode_with_oom_retry(model, sub, device, batch_size)
        all_embs.extend(embs_list)
    if not all_embs:
        raise ValueError("Query encoding produced no embeddings.")
    return torch.cat(all_embs, dim=0), qids


def step_02_preprocess(cfg: dict, device: torch.device) -> None:
    print_step_header(2, "Preprocessing")
    datasets = cfg["datasets"]
    bm25_params = get_active_bm25_params(cfg)
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    emb_batch = int(cfg["embeddings"]["batch_size"])

    print(f"  Active BM25 params: k1={bm25_params['k1']}, "
          f"b={bm25_params['b']}, use_stemming={bm25_params['use_stemming']}")

    # Load embedding model only once for all datasets.
    from sentence_transformers import SentenceTransformer
    model_name = cfg["embeddings"]["model_name"]
    print(f"  Loading embedding model {model_name} on {device} ...")
    st_model = SentenceTransformer(model_name, device=str(device))

    for ds_name in datasets:
        print(f"\n  --- {ds_name} ---")
        ds_dir = dataset_processed_dir(cfg, ds_name)
        ensure_dir(ds_dir)

        # Paths
        corpus_jsonl  = os.path.join(ds_dir, "corpus.jsonl")
        queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
        qrels_tsv     = os.path.join(ds_dir, "qrels.tsv")
        sparse = bm25_artifact_paths(ds_dir, **bm25_params)
        corpus_emb_pt   = os.path.join(ds_dir, "corpus_embeddings.pt")
        corpus_ids_pkl  = os.path.join(ds_dir, "corpus_ids.pkl")
        query_vecs_pt   = os.path.join(ds_dir, "query_vectors.pt")
        query_ids_pkl   = os.path.join(ds_dir, "query_ids.pkl")

        # 1) Corpus / queries / qrels exports
        if (is_nonempty_file(corpus_jsonl)
            and is_nonempty_file(queries_jsonl)
            and is_nonempty_file(qrels_tsv)):
            print("  [1/5] corpus/queries/qrels cached.")
        else:
            ds_path = download_beir_dataset(
                ds_name, get_config_path(cfg, "datasets_folder", "data/datasets")
            )
            corpus, queries, qrels, split = load_beir_dataset(ds_path)
            if corpus is None:
                raise RuntimeError(f"Failed to load BEIR dataset {ds_name}")
            print(f"  Split={split}  Corpus={len(corpus):,}  Queries={len(queries):,}")
            write_corpus_jsonl(corpus,  corpus_jsonl)
            write_queries_jsonl(queries, queries_jsonl)
            write_qrels_tsv(qrels, qrels_tsv)

        # 2) Tokenised corpus
        if is_nonempty_file(sparse["tokenized_corpus_jsonl"]):
            print("  [2/5] tokenised corpus cached.")
        else:
            _preprocess_corpus_parallel(
                corpus_jsonl, sparse["tokenized_corpus_jsonl"],
                stemmer_lang, bm25_params["use_stemming"],
            )

        # 3) Tokenised queries
        _preprocess_queries(
            queries_jsonl,
            sparse["tokenized_queries_jsonl"], sparse["query_tokens_pkl"],
            stemmer_lang, bm25_params["use_stemming"],
        )
        print("  [3/5] tokenised queries cached.")

        # 4) BM25 + frequency indices
        need_bm25 = not (is_nonempty_file(sparse["bm25_pkl"])
                         and is_nonempty_file(sparse["bm25_docids_pkl"])
                         and is_nonempty_file(sparse["word_freq_pkl"])
                         and is_nonempty_file(sparse["doc_freq_pkl"]))
        if not need_bm25:
            print("  [4/5] BM25 + frequency indices cached.")
        else:
            bm25, doc_ids, global_counts, total_tokens = _build_bm25_and_word_freq(
                sparse["tokenized_corpus_jsonl"],
                k1=bm25_params["k1"], b=bm25_params["b"],
            )
            doc_freq, total_docs = _build_doc_freq(sparse["tokenized_corpus_jsonl"])
            save_pickle(bm25, sparse["bm25_pkl"])
            save_pickle(doc_ids, sparse["bm25_docids_pkl"])
            save_pickle((global_counts, total_tokens), sparse["word_freq_pkl"])
            save_pickle((doc_freq, total_docs), sparse["doc_freq_pkl"])

        # 5) Embeddings
        if is_nonempty_file(corpus_emb_pt) and is_nonempty_file(corpus_ids_pkl):
            print("  [5a/5] corpus embeddings cached.")
        else:
            print("  [5a/5] encoding corpus ...")
            corpus_embeddings, corpus_ids = _build_corpus_embeddings(
                corpus_jsonl, st_model, emb_batch, str(device),
            )
            torch.save(corpus_embeddings, corpus_emb_pt)
            save_pickle(corpus_ids, corpus_ids_pkl)

        if is_nonempty_file(query_vecs_pt) and is_nonempty_file(query_ids_pkl):
            print("  [5b/5] query embeddings cached.")
        else:
            print("  [5b/5] encoding queries ...")
            queries = load_queries(queries_jsonl)
            q_vecs, q_ids = _build_query_embeddings(queries, st_model, emb_batch, str(device))
            torch.save(q_vecs, query_vecs_pt)
            save_pickle(q_ids, query_ids_pkl)


# ------------------------------------------------------------
# STEP 3 — Optimize BM25
# ------------------------------------------------------------

def _ensure_sparse_for_params(cfg: dict, ds_name: str,
                              k1: float, b: float, use_stemming: bool):
    """Build (or reuse) tokenised corpus + frequency + BM25 indices for one
    BM25 configuration.  Returns the artifact paths dict."""
    ds_dir = dataset_processed_dir(cfg, ds_name)
    ensure_dir(ds_dir)
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    paths = bm25_artifact_paths(
        ds_dir, k1, b, use_stemming,
        top_k=int(cfg["benchmark"]["top_k"]),
    )

    if not is_nonempty_file(paths["tokenized_corpus_jsonl"]):
        _preprocess_corpus_parallel(
            os.path.join(ds_dir, "corpus.jsonl"),
            paths["tokenized_corpus_jsonl"],
            stemmer_lang, use_stemming,
        )
    _preprocess_queries(
        os.path.join(ds_dir, "queries.jsonl"),
        paths["tokenized_queries_jsonl"], paths["query_tokens_pkl"],
        stemmer_lang, use_stemming,
    )

    needs_bm25 = not (is_nonempty_file(paths["bm25_pkl"])
                      and is_nonempty_file(paths["bm25_docids_pkl"]))
    needs_freq = not (is_nonempty_file(paths["word_freq_pkl"])
                      and is_nonempty_file(paths["doc_freq_pkl"]))
    if needs_bm25 or needs_freq:
        bm25, doc_ids, gc, tt = _build_bm25_and_word_freq(
            paths["tokenized_corpus_jsonl"], k1=k1, b=b,
        )
        if needs_bm25:
            save_pickle(bm25,    paths["bm25_pkl"])
            save_pickle(doc_ids, paths["bm25_docids_pkl"])
        if needs_freq:
            df, td = _build_doc_freq(paths["tokenized_corpus_jsonl"])
            save_pickle((gc, tt), paths["word_freq_pkl"])
            save_pickle((df, td), paths["doc_freq_pkl"])
    return paths


def _bm25_results_for_params(cfg: dict, ds_name: str, k1: float, b: float,
                             use_stemming: bool, queries: Dict[str, str]):
    """Return BM25 retrieval results for one (k1, b, stem) config — cached."""
    paths = _ensure_sparse_for_params(cfg, ds_name, k1, b, use_stemming)
    top_k = int(cfg["benchmark"]["top_k"])
    if is_nonempty_file(paths["bm25_results_pkl"]):
        try:
            return load_pickle(paths["bm25_results_pkl"])
        except Exception as exc:
            print(f"  [WARN] BM25 results cache corrupt for {ds_name}: {exc}; rebuilding.")
    bm25     = load_pickle(paths["bm25_pkl"])
    doc_ids  = load_pickle(paths["bm25_docids_pkl"])
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    results = run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k, use_stemming,
                                 desc=f"BM25 retrieval [{ds_name}]")
    save_pickle(results, paths["bm25_results_pkl"])
    return results


def step_03_optimize_bm25(cfg: dict, device: torch.device) -> None:
    print_step_header(3, "Optimize BM25 (k1, b, use_stemming)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_root)
    macro_csv = os.path.join(results_root, "bm25_grid_search.csv")
    best_json = os.path.join(results_root, "bm25_best_params.json")

    if file_exists(best_json) and file_exists(macro_csv):
        print(f"  [SKIP] {best_json} already exists.")
        return

    grid_cfg = cfg.get("bm25_grid_search", {}) or {}
    k1_vals  = [float(v) for v in grid_cfg.get("k1_values", [0.8, 1.2, 1.5, 1.6, 2.0])]
    b_vals   = [float(v) for v in grid_cfg.get("b_values",  [0.0, 0.25, 0.5, 0.75, 1.0])]
    stem_vals = [bool(v) for v in grid_cfg.get("use_stemming_values", [True, False])]

    datasets = cfg["datasets"]
    samp     = cfg["sampling"]
    n_per_ds = int(samp["n_queries_per_dataset"])
    trunc_seed = int(samp["truncation_seed"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])

    # Pick the same 300 queries per dataset that all later steps use.
    selected_qids = _select_merged_qids(cfg)

    combos = list(itertools.product(k1_vals, b_vals, stem_vals))
    print(f"  Grid: {len(combos)} combos (k1={len(k1_vals)} × b={len(b_vals)} × stem={len(stem_vals)})")
    print(f"  Per dataset: {n_per_ds} queries (merged 5x{n_per_ds} = {5 * n_per_ds})")

    macro_rows = []
    for (k1, b, use_stemming) in tqdm(combos, desc="BM25 grid", dynamic_ncols=True):
        per_ds_ndcg = []
        for ds_name in datasets:
            ds_dir = dataset_processed_dir(cfg, ds_name)
            queries_all = load_queries(os.path.join(ds_dir, "queries.jsonl"))
            qrels = load_qrels(os.path.join(ds_dir, "qrels.tsv"))

            # Restrict retrieval to the 300 selected queries (much faster).
            sel_set = set(selected_qids[ds_name])
            queries_sel = {q: queries_all[q] for q in selected_qids[ds_name]
                           if q in queries_all}

            results = _bm25_results_for_params(
                cfg, ds_name, k1, b, use_stemming, queries_sel,
            )
            scores = []
            for qid in queries_sel:
                ranked = results.get(qid, [])
                scores.append(query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k))
            per_ds_ndcg.append(float(np.mean(scores)) if scores else 0.0)

        macro = float(np.mean(per_ds_ndcg))
        macro_rows.append({
            "k1": k1, "b": b, "use_stemming": int(use_stemming),
            f"macro_ndcg@{ndcg_k}": macro,
            **{f"{ds}_ndcg@{ndcg_k}": v for ds, v in zip(datasets, per_ds_ndcg)},
        })

    macro_rows.sort(key=lambda r: r[f"macro_ndcg@{ndcg_k}"], reverse=True)
    best = macro_rows[0]
    print(f"\n  Best: k1={best['k1']}  b={best['b']}  use_stemming={bool(best['use_stemming'])}  "
          f"macro NDCG@{ndcg_k}={best[f'macro_ndcg@{ndcg_k}']:.4f}")

    fieldnames = ["k1", "b", "use_stemming", f"macro_ndcg@{ndcg_k}"] + \
                 [f"{ds}_ndcg@{ndcg_k}" for ds in datasets]
    save_csv_dicts(macro_rows, fieldnames, macro_csv)
    save_json({
        "k1": float(best["k1"]),
        "b":  float(best["b"]),
        "use_stemming": bool(best["use_stemming"]),
        f"macro_ndcg@{ndcg_k}": float(best[f"macro_ndcg@{ndcg_k}"]),
    }, best_json)


# ------------------------------------------------------------
# Selection of merged-dataset query IDs (300 per dataset)
# ------------------------------------------------------------

def _select_merged_qids(cfg: dict) -> Dict[str, List[str]]:
    """
    Deterministically pick `n_queries_per_dataset` query IDs per dataset.
    Cached as data/results/merged_qids.json so that every subsequent step
    works on the same 1500 queries.
    """
    results_root = get_config_path(cfg, "results_folder", "data/results")
    cache_path   = os.path.join(results_root, "merged_qids.json")
    if file_exists(cache_path):
        cached = load_json(cache_path)
        return {k: list(v) for k, v in cached.items()}

    ensure_dir(results_root)
    samp = cfg["sampling"]
    n_per_ds   = int(samp["n_queries_per_dataset"])
    trunc_seed = int(samp["truncation_seed"])
    out: Dict[str, List[str]] = {}
    for ds_name in cfg["datasets"]:
        queries = load_queries(os.path.join(dataset_processed_dir(cfg, ds_name), "queries.jsonl"))
        all_qids = sorted(queries.keys())
        n_full = len(all_qids)
        n_use  = min(n_per_ds, n_full)
        rng    = np.random.RandomState(trunc_seed + dataset_seed_offset(ds_name))
        idx    = np.sort(rng.choice(n_full, size=n_use, replace=False))
        out[ds_name] = [all_qids[i] for i in idx]
    save_json(out, cache_path)
    return out


# ------------------------------------------------------------
# Cached BM25 + Dense retrieval for the active params
# ------------------------------------------------------------

def _load_dense_results_with_cache(cfg: dict, ds_name: str, queries_jsonl: str,
                                   device: torch.device) -> Dict[str, List[Tuple[str, float]]]:
    """Run dense retrieval over ALL queries of a dataset, cached on disk."""
    ds_dir   = dataset_processed_dir(cfg, ds_name)
    top_k    = int(cfg["benchmark"]["top_k"])
    cache    = os.path.join(ds_dir, f"dense_results_topk_{top_k}.pkl")

    if is_nonempty_file(cache):
        try:
            return load_pickle(cache)
        except Exception as exc:
            print(f"  [WARN] dense cache corrupt for {ds_name}: {exc}; rebuilding.")

    corpus_emb_pt  = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vecs_pt  = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl  = os.path.join(ds_dir, "query_ids.pkl")

    print(f"  Running dense retrieval for {ds_name} ...")
    corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
    if device.type == "cuda":
        try:
            corpus_embeddings = corpus_embeddings.to(device)
        except torch.cuda.OutOfMemoryError:
            print("  [WARN] CUDA OOM; falling back to CPU for dense retrieval.")
            torch.cuda.empty_cache()
    corpus_ids = load_pickle(corpus_ids_pkl)
    q_vecs = torch.load(query_vecs_pt, weights_only=True)
    q_ids  = load_pickle(query_ids_pkl)
    results = run_dense_retrieval(q_vecs, q_ids, corpus_embeddings, corpus_ids, top_k, cfg,
                                  desc=f"Dense retrieval [{ds_name}]")
    save_pickle(results, cache)
    return results


def _load_active_retrieval(cfg: dict, ds_name: str, device: torch.device):
    """Return (bm25_results, dense_results, qrels) for the active BM25 params."""
    ds_dir = dataset_processed_dir(cfg, ds_name)
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    queries = load_queries(queries_jsonl)
    qrels   = load_qrels(os.path.join(ds_dir, "qrels.tsv"))

    bm25_params = get_active_bm25_params(cfg)
    bm25_results = _bm25_results_for_params(
        cfg, ds_name,
        bm25_params["k1"], bm25_params["b"], bm25_params["use_stemming"],
        queries,
    )
    dense_results = _load_dense_results_with_cache(cfg, ds_name, queries_jsonl, device)
    return bm25_results, dense_results, qrels


# ------------------------------------------------------------
# STEP 4 — Oracle alpha grid search
# ------------------------------------------------------------

def _oracle_alpha_for_query(qid: str, qrel: Dict[str, int],
                            bm25_pairs: list, dense_pairs: list,
                            alphas: np.ndarray, rrf_k: int,
                            ndcg_k: int, top_k: int) -> Tuple[float, float]:
    """
    For a single query, scan all alphas and return (best_alpha, best_ndcg).

    We re-use the rank-arrays trick: the wRRF score of doc d at alpha a is
       a * R_bm[d] + (1-a) * R_de[d]
    where R_bm[d] = 1/(rrf_k + rank_bm25(d)) and R_de[d] symmetrically.
    These are computed once per query, then evaluated for each alpha.
    """
    if not qrel or all(g <= 0 for g in qrel.values()):
        return 0.5, 0.0  # query has no relevant docs; alpha is arbitrary
    bm_rank = {d: r for r, (d, _) in enumerate(bm25_pairs, 1)}
    de_rank = {d: r for r, (d, _) in enumerate(dense_pairs, 1)}
    bm_miss = len(bm25_pairs) + 1
    de_miss = len(dense_pairs) + 1
    docs = list(set(bm_rank) | set(de_rank))
    if not docs:
        return 0.5, 0.0

    R_bm = np.array([1.0 / (rrf_k + bm_rank.get(d, bm_miss)) for d in docs], dtype=np.float64)
    R_de = np.array([1.0 / (rrf_k + de_rank.get(d, de_miss)) for d in docs], dtype=np.float64)
    rels = np.array([qrel.get(d, 0) for d in docs], dtype=np.float64)

    best_alpha = 0.5
    best_ndcg  = -1.0

    # Pre-compute ideal DCG using the FULL qrel set (matches query_ndcg_at_k).
    # If we used grades from the candidate pool, IDCG would be biased low and
    # the reported oracle_ndcg would be inflated.  The argmax over alphas is
    # invariant to this choice (IDCG is constant per query), so the chosen
    # oracle alpha is unchanged.
    ideal_grades = sorted([g for g in qrel.values() if g > 0], reverse=True)[:ndcg_k]
    if not ideal_grades:
        return 0.5, 0.0
    idcg = float(np.sum(
        ((2.0 ** np.asarray(ideal_grades, dtype=np.float64)) - 1.0)
        / np.log2(np.arange(2, len(ideal_grades) + 2))
    ))
    if idcg <= 0:
        return 0.5, 0.0

    for a in alphas:
        scores = a * R_bm + (1.0 - a) * R_de
        # take top top_k indices, then top ndcg_k of those
        if len(scores) > top_k:
            top_idx = np.argpartition(-scores, top_k)[:top_k]
        else:
            top_idx = np.arange(len(scores))
        top_scores = scores[top_idx]
        order = top_idx[np.argsort(-top_scores, kind="stable")]
        top_rels = rels[order[:ndcg_k]]
        gains = (2.0 ** top_rels) - 1.0
        log2_pos = np.log2(np.arange(2, len(top_rels) + 2))
        dcg = float(np.sum(gains / log2_pos))
        ndcg = dcg / idcg
        if ndcg > best_ndcg:
            best_ndcg  = ndcg
            best_alpha = float(a)
    return best_alpha, best_ndcg


def step_04_oracle_alpha(cfg: dict, device: torch.device) -> None:
    print_step_header(4, "Oracle alpha grid search (per query)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "oracle_alphas.csv")
    if file_exists(out_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    selected = _select_merged_qids(cfg)
    grid_cfg = cfg.get("oracle_alpha_search", {}) or {}
    a_min  = float(grid_cfg.get("alpha_min",  0.0))
    a_max  = float(grid_cfg.get("alpha_max",  1.0))
    a_step = float(grid_cfg.get("alpha_step", 0.01))
    alphas = np.arange(a_min, a_max + 1e-9, a_step, dtype=np.float64)
    print(f"  Alphas: {len(alphas)} values in [{a_min}, {a_max}] step {a_step}")

    rrf_k = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    top_k  = int(cfg["benchmark"]["top_k"])

    rows: List[dict] = []
    for ds_name in cfg["datasets"]:
        print(f"\n  --- {ds_name} ---")
        bm25_results, dense_results, qrels = _load_active_retrieval(cfg, ds_name, device)
        for qid in tqdm(selected[ds_name], desc=f"  oracle [{ds_name}]", dynamic_ncols=True):
            best_alpha, best_ndcg = _oracle_alpha_for_query(
                qid, qrels.get(qid, {}),
                bm25_results.get(qid, []),
                dense_results.get(qid, []),
                alphas, rrf_k, ndcg_k, top_k,
            )
            rows.append({
                "ds_name":      ds_name,
                "qid":          qid,
                "oracle_alpha": float(best_alpha),
                "oracle_ndcg":  float(best_ndcg),
            })

    save_csv_dicts(rows, ["ds_name", "qid", "oracle_alpha", "oracle_ndcg"], out_csv)
    print(f"\n  Saved {len(rows)} oracle alphas to {out_csv}")

    # Also save the per-dataset average oracle NDCG for reference.
    avg = {}
    for ds_name in cfg["datasets"]:
        sub = [r["oracle_ndcg"] for r in rows if r["ds_name"] == ds_name]
        avg[ds_name] = float(np.mean(sub)) if sub else 0.0
    avg["MACRO"] = float(np.mean(list(avg.values())))
    save_json(avg, os.path.join(results_root, "oracle_ndcg_per_dataset.json"))
    for k, v in avg.items():
        print(f"    {k:<10}  oracle NDCG@{ndcg_k} = {v:.4f}")


# ------------------------------------------------------------
# Stratified split (computed once, cached)
# ------------------------------------------------------------

def _load_split(cfg: dict) -> Dict[str, dict]:
    """Read or compute the stratified train/dev/test split."""
    results_root = get_config_path(cfg, "results_folder", "data/results")
    cache_path   = os.path.join(results_root, "merged_split.json")
    if file_exists(cache_path):
        return load_json(cache_path)

    selected = _select_merged_qids(cfg)
    samp = cfg["sampling"]
    split = stratified_split(
        selected,
        test_fraction=float(samp["test_fraction"]),
        dev_fraction=float(samp["dev_fraction"]),
        base_seed=int(samp["random_seed"]),
    )
    save_json(split, cache_path)
    return split


def _load_oracle_alphas(cfg: dict) -> Dict[Tuple[str, str], float]:
    rows = load_csv_dicts(os.path.join(
        get_config_path(cfg, "results_folder", "data/results"),
        "oracle_alphas.csv",
    ))
    return {(r["ds_name"], r["qid"]): float(r["oracle_alpha"]) for r in rows}


# ------------------------------------------------------------
# STEP 5 — Weak-model dataset (16 features + oracle alpha label)
# ------------------------------------------------------------

def _normalize_pairs_minmax(pairs: list, eps: float):
    """Min-max normalise scores in a list of (doc_id, score) pairs."""
    if not pairs:
        return []
    scores = [float(s) for _, s in pairs]
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng < eps:
        return [(d, 0.0) for d, _ in pairs]
    return [(d, (float(s) - lo) / (rng + eps)) for d, s in pairs]


def _compute_query_features(raw_text: str,
                            query_tokens: list,
                            bm25_pairs: list,
                            dense_pairs: list,
                            word_freq: dict,
                            total_corpus_tokens: int,
                            doc_freq: dict,
                            total_docs: int,
                            stopword_stems: frozenset,
                            overlap_k: int,
                            feature_stat_k: int,
                            epsilon: float,
                            ce_alpha: float) -> Dict[str, float]:
    bm25_norm  = _normalize_pairs_minmax(bm25_pairs, epsilon)
    dense_norm = _normalize_pairs_minmax(dense_pairs, epsilon)
    n_tok = len(query_tokens)

    # Group A
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

    # Group B
    vocab_size  = max(1, len(word_freq))
    corpus_mass = total_corpus_tokens + ce_alpha * vocab_size
    if not clean:
        cross_entropy = average_idf = max_idf = rare_term_ratio = 0.0
    else:
        ce_sum = 0.0; idfs = []
        for t in clean:
            prob = (word_freq.get(t, 0) + ce_alpha) / corpus_mass
            ce_sum += -math.log2(max(prob, epsilon))
            idfs.append(math.log((total_docs + 1.0) / (doc_freq.get(t, 0) + 1.0)) + 1.0)
        cross_entropy = ce_sum / len(clean)
        average_idf = sum(idfs) / len(idfs)
        max_idf     = max(idfs)
        idf_std     = float(np.std(idfs))
        thresh      = average_idf + idf_std
        rare_term_ratio = sum(1 for v in idfs if v >= thresh) / len(idfs)

    # Group C
    def _conf(normed):
        if len(normed) >= 2:
            return normed[0][1] - normed[1][1]
        return normed[0][1] if normed else 0.0
    dense_confidence  = _conf(dense_norm)
    sparse_confidence = _conf(bm25_norm)
    top_dense_score   = dense_norm[0][1] if dense_norm else 0.0
    top_sparse_score  = bm25_norm[0][1]  if bm25_norm  else 0.0

    # Group D
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
        # Must convert absolute top-k positions to relative ranks within the
        # shared set (1..n) before applying the d^2 shortcut formula, otherwise
        # the formula produces values outside [-1, 1].
        sp_rel = {d: r for r, d in enumerate(sorted(shared, key=lambda x: sp_rank[x]), 1)}
        de_rel = {d: r for r, d in enumerate(sorted(shared, key=lambda x: de_rank[x]), 1)}
        diffs = [(sp_rel[d] - de_rel[d]) ** 2 for d in shared]
        n = len(diffs)
        spearman_topk = 1.0 - (6.0 * sum(diffs)) / (n * (n ** 2 - 1.0))
    else:
        spearman_topk = 0.0

    # Group E
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
    dense_entropy_topk  = _entropy(dense_norm,  feature_stat_k)
    sparse_entropy_topk = _entropy(bm25_norm,   feature_stat_k)

    return {
        "query_length":         float(query_length),
        "stopword_ratio":       float(stopword_ratio),
        "has_question_word":    float(has_qw),
        "average_idf":          float(average_idf),
        "max_idf":              float(max_idf),
        "rare_term_ratio":      float(rare_term_ratio),
        "cross_entropy":        float(cross_entropy),
        "top_dense_score":      float(top_dense_score),
        "top_sparse_score":     float(top_sparse_score),
        "dense_confidence":     float(dense_confidence),
        "sparse_confidence":    float(sparse_confidence),
        "overlap_at_k":         float(overlap_at_k),
        "first_shared_doc_rank": float(first_shared_doc_rank),
        "spearman_topk":        float(spearman_topk),
        "dense_entropy_topk":   float(dense_entropy_topk),
        "sparse_entropy_topk":  float(sparse_entropy_topk),
    }


def step_05_weak_dataset(cfg: dict, device: torch.device) -> None:
    print_step_header(5, "Weak-model dataset (16 features + oracle alpha)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "weak_dataset.csv")
    if file_exists(out_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    selected = _select_merged_qids(cfg)
    split    = _load_split(cfg)
    oracle   = _load_oracle_alphas(cfg)

    rcfg = cfg.get("routing_features", {}) or {}
    overlap_k      = int(rcfg.get("overlap_k", 10))
    feature_stat_k = int(rcfg.get("feature_stat_k", 10))
    epsilon        = float(rcfg.get("epsilon", 1e-8))
    ce_alpha       = float(rcfg.get("ce_smoothing_alpha", 1.0))
    bm25_params    = get_active_bm25_params(cfg)
    use_stemming   = bm25_params["use_stemming"]
    stemmer_lang   = cfg["preprocessing"]["stemmer_language"]

    english_sw = ensure_english_stopwords()
    if use_stemming:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer(stemmer_lang)
        stopword_stems = frozenset(stemmer.stem(w) for w in english_sw)
    else:
        stopword_stems = frozenset(w.lower() for w in english_sw)

    rows: List[dict] = []
    for ds_name in cfg["datasets"]:
        print(f"\n  --- {ds_name} ---")
        ds_dir = dataset_processed_dir(cfg, ds_name)
        queries = load_queries(os.path.join(ds_dir, "queries.jsonl"))
        sparse  = bm25_artifact_paths(ds_dir, **bm25_params)
        word_freq, total_corpus_tokens = load_pickle(sparse["word_freq_pkl"])
        doc_freq,  total_docs          = load_pickle(sparse["doc_freq_pkl"])
        query_tokens = load_pickle(sparse["query_tokens_pkl"])

        bm25_results, dense_results, _ = _load_active_retrieval(cfg, ds_name, device)

        train_set = set(split[ds_name]["train"])
        dev_set   = set(split[ds_name]["dev"])
        test_set  = set(split[ds_name]["test"])

        for qid in tqdm(selected[ds_name], desc=f"  features [{ds_name}]", dynamic_ncols=True):
            feats = _compute_query_features(
                queries[qid], query_tokens.get(qid, []),
                bm25_results.get(qid, []), dense_results.get(qid, []),
                word_freq, total_corpus_tokens, doc_freq, total_docs,
                stopword_stems, overlap_k, feature_stat_k, epsilon, ce_alpha,
            )
            if   qid in train_set: split_label = "train"
            elif qid in dev_set:   split_label = "dev"
            elif qid in test_set:  split_label = "test"
            else:                  split_label = "drop"
            rows.append({
                "ds_name":      ds_name,
                "qid":          qid,
                "split":        split_label,
                **feats,
                "oracle_alpha": float(oracle.get((ds_name, qid), 0.5)),
            })

    fieldnames = ["ds_name", "qid", "split"] + FEATURE_NAMES + ["oracle_alpha"]
    save_csv_dicts(rows, fieldnames, out_csv)
    print(f"\n  Saved {len(rows)} weak rows to {out_csv}")


def _load_weak_dataset(cfg: dict):
    """Return X_traindev, y_traindev, X_test, y_test, qids, ds_names, splits."""
    rows = load_csv_dicts(os.path.join(
        get_config_path(cfg, "results_folder", "data/results"),
        "weak_dataset.csv",
    ))
    by_split = {"train": [], "dev": [], "test": []}
    for r in rows:
        if r["split"] in by_split:
            by_split[r["split"]].append(r)

    def _slice(split_rows):
        X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in split_rows], dtype=np.float32)
        y = np.array([float(r["oracle_alpha"]) for r in split_rows], dtype=np.float32)
        qids = [r["qid"] for r in split_rows]
        ds   = [r["ds_name"] for r in split_rows]
        return X, y, qids, ds

    X_tr, y_tr, qids_tr, ds_tr = _slice(by_split["train"])
    X_dv, y_dv, qids_dv, ds_dv = _slice(by_split["dev"])
    X_te, y_te, qids_te, ds_te = _slice(by_split["test"])

    X_td = np.vstack([X_tr, X_dv]) if len(X_tr) and len(X_dv) else (X_tr if len(X_tr) else X_dv)
    y_td = np.concatenate([y_tr, y_dv]) if len(y_tr) and len(y_dv) else (y_tr if len(y_tr) else y_dv)
    qids_td = qids_tr + qids_dv
    ds_td   = ds_tr   + ds_dv

    return {
        "X_td":    X_td,
        "y_td":    y_td,
        "qids_td": qids_td,
        "ds_td":   ds_td,
        "X_te":    X_te,
        "y_te":    y_te,
        "qids_te": qids_te,
        "ds_te":   ds_te,
    }


# ------------------------------------------------------------
# Per-query wRRF NDCG (used by every grid search and comparison)
# ------------------------------------------------------------

def _wrrf_query_ndcg(alpha: float, qid: str,
                     bm25_res: dict, dense_res: dict, qrels: dict,
                     rrf_k: int, ndcg_k: int) -> float:
    ranked = wrrf_fuse(alpha, bm25_res.get(qid, []), dense_res.get(qid, []), rrf_k)
    return query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k)


def _retrieval_data_per_dataset(cfg: dict, device: torch.device,
                                datasets: list) -> Dict[str, dict]:
    """Build {ds: {bm25_results, dense_results, qrels}}."""
    out = {}
    for ds_name in datasets:
        bm25, dense, qrels = _load_active_retrieval(cfg, ds_name, device)
        out[ds_name] = {"bm25_results": bm25, "dense_results": dense, "qrels": qrels}
    return out


# ------------------------------------------------------------
# Common: evaluate one (model_name, params, feature_cols) combo
# with 10-fold CV on the train+dev portion of the merged dataset.
# Returns mean NDCG@10 on dev queries (averaged across folds).
# ------------------------------------------------------------

def _cv_score_one_combo(model_name: str, params: dict,
                        X_td: np.ndarray, y_td: np.ndarray,
                        qids_td: list, ds_td: list,
                        feature_cols: Optional[list],
                        retrieval_data: Dict[str, dict],
                        folds: list, seed: int,
                        rrf_k: int, ndcg_k: int) -> float:
    Xc = X_td[:, feature_cols] if feature_cols is not None else X_td
    fold_means = []
    for tr_idx, va_idx in folds:
        X_tr, y_tr = Xc[tr_idx], y_td[tr_idx]
        X_va       = Xc[va_idx]
        mu, sigma  = zscore_stats(X_tr)
        X_tr_z = (X_tr - mu) / sigma
        X_va_z = (X_va - mu) / sigma

        y_fit = (y_tr >= 0.5).astype(int) if model_name in CLASSIFIER_MODELS else y_tr

        try:
            mdl = make_model(model_name, params, seed)
            mdl.fit(X_tr_z, y_fit)
            preds = predict_alpha_from_model(mdl, X_va_z, model_name)
        except Exception as exc:
            warnings.warn(f"_cv_score_one_combo {model_name} {params}: {exc}")
            preds = np.full(len(va_idx), 0.5, dtype=np.float32)

        ndcgs = []
        for i, alpha in zip(va_idx, preds):
            qid = qids_td[i]; ds = ds_td[i]
            rd  = retrieval_data[ds]
            ndcgs.append(_wrrf_query_ndcg(
                float(alpha), qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))
        fold_means.append(float(np.mean(ndcgs)) if ndcgs else 0.0)
    return float(np.mean(fold_means))


def _cv_perquery_scores(model_name: str, params: dict,
                        X_td: np.ndarray, y_td: np.ndarray,
                        qids_td: list, ds_td: list,
                        feature_cols: Optional[list],
                        retrieval_data: Dict[str, dict],
                        folds: list, seed: int,
                        rrf_k: int, ndcg_k: int) -> np.ndarray:
    """Like _cv_score_one_combo but returns a per-query NDCG array (len = len(y_td)).

    Each query appears in exactly one validation fold, so the array contains
    one NDCG value per query, enabling paired t-tests between feature configs.
    """
    Xc = X_td[:, feature_cols] if feature_cols is not None else X_td
    per_query: Dict[int, float] = {}
    for tr_idx, va_idx in folds:
        X_tr, y_tr = Xc[tr_idx], y_td[tr_idx]
        X_va = Xc[va_idx]
        mu, sigma = zscore_stats(X_tr)
        X_tr_z = (X_tr - mu) / sigma
        X_va_z = (X_va - mu) / sigma
        y_fit = (y_tr >= 0.5).astype(int) if model_name in CLASSIFIER_MODELS else y_tr
        try:
            mdl = make_model(model_name, params, seed)
            mdl.fit(X_tr_z, y_fit)
            preds = predict_alpha_from_model(mdl, X_va_z, model_name)
        except Exception as exc:
            warnings.warn(f"_cv_perquery_scores {model_name} {params}: {exc}")
            preds = np.full(len(va_idx), 0.5, dtype=np.float32)
        for i, alpha in zip(va_idx, preds):
            qid = qids_td[i]; ds = ds_td[i]
            rd = retrieval_data[ds]
            per_query[i] = _wrrf_query_ndcg(
                float(alpha), qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            )
    return np.array([per_query[i] for i in range(len(y_td))], dtype=np.float32)


# ------------------------------------------------------------
# STEP 6 — Weak-model grid search
# ------------------------------------------------------------

def step_06_weak_grid_search(cfg: dict, device: torch.device) -> None:
    print_step_header(6, "Weak-model grid search")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv      = os.path.join(results_root, "weak_grid_search_top.csv")
    out_best     = os.path.join(results_root, "weak_best_params.json")
    if file_exists(out_csv) and file_exists(out_best):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    seed   = int(cfg["sampling"]["random_seed"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    n_folds = int(cfg["sampling"]["cv_n_folds"])
    cap    = int(cfg.get("max_models_per_grid", 1500))
    grid_cfg = cfg.get("weak_model_grid_search", {}).get("models", {}) or {}

    combos = expand_grid(grid_cfg)
    combos = truncate_combos(combos, cap)
    print(f"  Total combos: {len(combos)}")

    ds = _load_weak_dataset(cfg)
    retrieval = _retrieval_data_per_dataset(cfg, device, cfg["datasets"])

    folds = kfold_indices(len(ds["y_td"]), n_folds, seed)
    n_jobs = 1 if torch.cuda.is_available() else -1   # XGBoost-on-CUDA is sequential

    t0 = time.time()
    scores = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_cv_score_one_combo)(
                model_name, params,
                ds["X_td"], ds["y_td"], ds["qids_td"], ds["ds_td"],
                None, retrieval, folds, seed, rrf_k, ndcg_k,
            )
            for model_name, params in combos
        ),
        total=len(combos), desc="  Weak grid", dynamic_ncols=True,
    ))
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    ranked = sorted(zip(combos, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:100]
    rows = [{"rank": i + 1,
             "model": mn, "params_json": json.dumps(p, sort_keys=True),
             f"cv_ndcg@{ndcg_k}": s}
            for i, ((mn, p), s) in enumerate(top)]
    save_csv_dicts(rows, ["rank", "model", "params_json", f"cv_ndcg@{ndcg_k}"], out_csv)

    (best_model, best_params), best_score = ranked[0]
    save_json({
        "model":  best_model,
        "params": best_params,
        f"cv_ndcg@{ndcg_k}": float(best_score),
    }, out_best)
    print(f"  Best: {best_model}  {best_params}  CV NDCG@{ndcg_k}={best_score:.4f}")


# ------------------------------------------------------------
# STEP 7 — Weak-model ablation study
# ------------------------------------------------------------

def step_07_weak_ablation(cfg: dict, device: torch.device) -> None:
    print_step_header(7, "Weak-model feature ablation (phase 1 + phase 2 combo)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    models_root  = get_config_path(cfg, "models_folder",  "data/models")
    p1_csv    = os.path.join(results_root, "weak_ablation.csv")
    p1_png    = os.path.join(results_root, "weak_ablation.png")
    p2_csv    = os.path.join(results_root, "weak_ablation_combo.csv")
    p2_png    = os.path.join(results_root, "weak_ablation_combo.png")
    final_pkl = os.path.join(models_root,  "weak_model.pkl")

    seed      = int(cfg["sampling"]["random_seed"])
    rrf_k     = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k    = int(cfg["benchmark"]["ndcg_k"])
    n_folds   = int(cfg["sampling"]["cv_n_folds"])
    sig_alpha = float(cfg.get("significance_test", {}).get("alpha", 0.05))

    best = load_json(os.path.join(results_root, "weak_best_params.json"))
    best_model_name, best_params = best["model"], best["params"]

    ds        = _load_weak_dataset(cfg)
    retrieval = _retrieval_data_per_dataset(cfg, device, cfg["datasets"])
    folds     = kfold_indices(len(ds["y_td"]), n_folds, seed)
    n_jobs    = 1 if torch.cuda.is_available() else -1

    # ── Phase 1: leave-one-feature-out + leave-one-group-out ─────────────
    if not (file_exists(p1_csv) and file_exists(p1_png)):
        print(f"\n  --- Phase 1: single feature / group ablation ---")
        print(f"  Using best model: {best_model_name}  {best_params}")
        configs_p1 = [("full", "full", "—", list(range(len(FEATURE_NAMES))))]
        for fname in FEATURE_NAMES:
            configs_p1.append((
                f"no_{fname}", "leave_one_feature", fname,
                [i for i, n in enumerate(FEATURE_NAMES) if n != fname],
            ))
        for gname, gfeats in FEATURE_GROUPS.items():
            configs_p1.append((
                f"no_group_{gname.split(':')[0].strip()}",
                "leave_one_group", gname,
                [i for i, n in enumerate(FEATURE_NAMES) if n not in set(gfeats)],
            ))

        t0 = time.time()
        scores_p1 = list(tqdm(
            Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                delayed(_cv_score_one_combo)(
                    best_model_name, best_params,
                    ds["X_td"], ds["y_td"], ds["qids_td"], ds["ds_td"],
                    cols, retrieval, folds, seed, rrf_k, ndcg_k,
                )
                for (_, _, _, cols) in configs_p1
            ),
            total=len(configs_p1), desc="  Phase 1", dynamic_ncols=True,
        ))
        print(f"  Phase 1 elapsed: {time.time() - t0:.1f}s")

        rows_p1: List[dict] = []
        for (cfg_name, atype, removed, cols), score in zip(configs_p1, scores_p1):
            rows_p1.append({
                "config_name":       cfg_name,
                "ablation_type":     atype,
                "removed":           removed,
                "n_features":        len(cols),
                f"cv_ndcg@{ndcg_k}": float(score),
                "feature_cols_json": json.dumps(cols),
            })
        rows_p1.sort(key=lambda r: r[f"cv_ndcg@{ndcg_k}"], reverse=True)
        save_csv_dicts(
            rows_p1,
            ["config_name", "ablation_type", "removed", "n_features",
             f"cv_ndcg@{ndcg_k}", "feature_cols_json"],
            p1_csv,
        )
        full_score_p1 = next(r[f"cv_ndcg@{ndcg_k}"] for r in rows_p1
                             if r["ablation_type"] == "full")
        _plot_ablation(rows_p1, full_score_p1, ndcg_k, p1_png)
        print(f"  Phase 1 plot saved: {p1_png}")
    else:
        print(f"  [SKIP] Phase 1 already done ({p1_csv}).")

    # ── Phase 2: all subsets of non-damaging features ────────────────────
    if not (file_exists(p2_csv) and file_exists(p2_png)):
        print(f"\n  --- Phase 2: combination ablation of non-damaging features ---")
        rows_p1    = load_csv_dicts(p1_csv)
        full_score = float(next(r[f"cv_ndcg@{ndcg_k}"] for r in rows_p1
                                if r["ablation_type"] == "full"))
        lof_rows   = [r for r in rows_p1 if r["ablation_type"] == "leave_one_feature"]
        # Features with delta >= 0 when removed (zero or positive change).
        nondamaging = [r["removed"] for r in lof_rows
                       if float(r[f"cv_ndcg@{ndcg_k}"]) >= full_score]
        print(f"  Full model CV NDCG@{ndcg_k}: {full_score:.4f}")
        print(f"  Non-damaging features (delta>=0 when removed): "
              f"{nondamaging if nondamaging else '(none)'}")

        # Per-query CV scores for the full model (needed for t-tests).
        print("  Computing per-query CV scores for full model ...")
        full_pq = _cv_perquery_scores(
            best_model_name, best_params,
            ds["X_td"], ds["y_td"], ds["qids_td"], ds["ds_td"],
            list(range(len(FEATURE_NAMES))), retrieval, folds, seed, rrf_k, ndcg_k,
        )

        rows_p2: List[dict] = []
        if not nondamaging:
            print("  No non-damaging features found — phase 2 produces no combos.")
        else:
            combos_to_remove: List[List[str]] = []
            for r in range(1, len(nondamaging) + 1):
                for subset in itertools.combinations(nondamaging, r):
                    combos_to_remove.append(list(subset))
            combo_cols_list = [
                [i for i, n in enumerate(FEATURE_NAMES) if n not in set(subset)]
                for subset in combos_to_remove
            ]
            print(f"  Combinations to test: {len(combos_to_remove)}")

            t0 = time.time()
            pq_list = list(tqdm(
                Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                    delayed(_cv_perquery_scores)(
                        best_model_name, best_params,
                        ds["X_td"], ds["y_td"], ds["qids_td"], ds["ds_td"],
                        cols, retrieval, folds, seed, rrf_k, ndcg_k,
                    )
                    for cols in combo_cols_list
                ),
                total=len(combo_cols_list), desc="  Phase 2", dynamic_ncols=True,
            ))
            print(f"  Phase 2 elapsed: {time.time() - t0:.1f}s")

            for subset, cols, pq in zip(combos_to_remove, combo_cols_list, pq_list):
                mean_score = float(np.mean(pq))
                ttest      = paired_t_test(pq.tolist(), full_pq.tolist())
                sig_better = bool((ttest["p"] <= sig_alpha) and (ttest["mean_diff"] > 0))
                rows_p2.append({
                    "removed":           ",".join(subset),
                    "n_removed":         len(subset),
                    "n_features":        len(cols),
                    f"cv_ndcg@{ndcg_k}": mean_score,
                    "delta":             mean_score - full_score,
                    "t":                 ttest["t"],
                    "p_value":           ttest["p"],
                    "sig_better":        sig_better,
                    "feature_cols_json": json.dumps(cols),
                })
            rows_p2.sort(key=lambda r: float(r[f"cv_ndcg@{ndcg_k}"]), reverse=True)

        save_csv_dicts(
            rows_p2,
            ["removed", "n_removed", "n_features", f"cv_ndcg@{ndcg_k}",
             "delta", "t", "p_value", "sig_better", "feature_cols_json"],
            p2_csv,
        )
        _plot_ablation_combos(rows_p2, full_score, ndcg_k, sig_alpha, p2_png)
        print(f"  Phase 2 plot saved: {p2_png}")
        for r in rows_p2:
            print(f"    remove=[{r['removed']}]  "
                  f"NDCG@{ndcg_k}={float(r[f'cv_ndcg@{ndcg_k}']):.4f}  "
                  f"delta={float(r['delta']):+.4f}  "
                  f"p={float(r['p_value']):.4f}  sig={r['sig_better']}")
    else:
        print(f"  [SKIP] Phase 2 already done ({p2_csv}).")

    # ── Final model: select feature set from phase 2 result ──────────────
    if file_exists(final_pkl):
        print(f"  [SKIP] Final model already built ({final_pkl}).")
        return

    rows_p1    = load_csv_dicts(p1_csv)
    full_score = float(next(r[f"cv_ndcg@{ndcg_k}"] for r in rows_p1
                            if r["ablation_type"] == "full"))
    rows_p2    = load_csv_dicts(p2_csv)
    sig_rows   = [r for r in rows_p2 if str(r.get("sig_better")).lower() == "true"]

    if sig_rows:
        best_row     = max(sig_rows, key=lambda r: float(r[f"cv_ndcg@{ndcg_k}"]))
        feature_cols = json.loads(best_row["feature_cols_json"])
        print(f"  Selected combo: remove [{best_row['removed']}]  "
              f"NDCG@{ndcg_k}={float(best_row[f'cv_ndcg@{ndcg_k}']):.4f}  "
              f"(p={float(best_row['p_value']):.4f} < {sig_alpha})")
    else:
        feature_cols = list(range(len(FEATURE_NAMES)))
        print(f"  No combo significantly beats the full model — using all {len(feature_cols)} features.")

    X_train    = ds["X_td"][:, feature_cols]
    mu, sigma  = zscore_stats(X_train)
    X_train_z  = (X_train - mu) / sigma
    y_fit      = (ds["y_td"] >= 0.5).astype(int) if best_model_name in CLASSIFIER_MODELS else ds["y_td"]
    final_mdl  = make_model(best_model_name, best_params, seed)
    final_mdl.fit(X_train_z, y_fit)

    save_pickle({
        "model_name":    best_model_name,
        "params":        best_params,
        "feature_cols":  feature_cols,
        "feature_names": [FEATURE_NAMES[i] for i in feature_cols],
        "scaler_mu":     mu,
        "scaler_sigma":  sigma,
        "model":         final_mdl,
    }, final_pkl)
    print(f"  Final weak model saved: {final_pkl}  ({len(feature_cols)} features)")


def _plot_ablation(rows: List[dict], full_score: float,
                   ndcg_k: int, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    lof = sorted(
        [r for r in rows if r["ablation_type"] == "leave_one_feature"],
        key=lambda r: r[f"cv_ndcg@{ndcg_k}"],
    )
    log_ = sorted(
        [r for r in rows if r["ablation_type"] == "leave_one_group"],
        key=lambda r: r[f"cv_ndcg@{ndcg_k}"],
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 14),
        gridspec_kw={"height_ratios": [max(1, len(lof)), max(1, len(log_))]},
    )
    fig.suptitle(f"Feature Ablation Study (CV NDCG@{ndcg_k}) — Weak Router",
                 fontsize=14, y=1.01)

    all_scores = [r[f"cv_ndcg@{ndcg_k}"] for r in lof + log_] + [full_score]
    x_min = max(0.0, min(all_scores) - 0.01)
    x_max = min(1.0, max(all_scores) + 0.005)

    panels = [
        (axes[0], lof,  "#4C72B0", "Leave-one-feature-out"),
        (axes[1], log_, "#DD8452", "Leave-one-group-out"),
    ]
    for ax, results, color, title in panels:
        labels = [r["removed"] for r in results]
        scores = [r[f"cv_ndcg@{ndcg_k}"] for r in results]
        deltas = [s - full_score for s in scores]
        bars = ax.barh(labels, scores, color=color, alpha=0.80,
                       edgecolor="white", linewidth=0.5)
        for bar, delta in zip(bars, deltas):
            sign = "+" if delta >= 0 else ""
            ax.text(bar.get_width() + 0.0003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{delta:.4f}", va="center", ha="left",
                    fontsize=7.5, color="black")
        ax.axvline(full_score, color="green", linewidth=1.4, linestyle="--",
                   label=f"Full ({full_score:.4f})")
        ax.set_xlim(x_min, x_max + 0.012)
        ax.set_xlabel(f"CV NDCG@{ndcg_k}", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.legend(loc="lower right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ablation_combos(rows: List[dict], full_score: float,
                          ndcg_k: int, sig_alpha: float,
                          out_path: str) -> None:
    """Bar chart for phase-2 combo ablation results.

    Green bars = significantly better than full model; grey = not significant.
    Asterisk (*) appended to delta label for significant entries.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5,
                "No non-damaging features found.\nFull model is used as-is.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    rows_sorted = sorted(rows, key=lambda r: float(r[f"cv_ndcg@{ndcg_k}"]))
    labels  = [r["removed"] for r in rows_sorted]
    scores  = [float(r[f"cv_ndcg@{ndcg_k}"]) for r in rows_sorted]
    sig     = [str(r.get("sig_better")).lower() == "true" for r in rows_sorted]
    deltas  = [s - full_score for s in scores]
    colors  = ["#2ca02c" if s else "#aec7e8" for s in sig]

    all_scores = scores + [full_score]
    x_min = max(0.0, min(all_scores) - 0.01)
    x_max = min(1.0, max(all_scores) + 0.005)

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows_sorted) * 0.55 + 1.5)))
    bars = ax.barh(labels, scores, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for bar, delta, s in zip(bars, deltas, sig):
        sign   = "+" if delta >= 0 else ""
        marker = " *" if s else ""
        ax.text(bar.get_width() + 0.0003,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.4f}{marker}", va="center", ha="left",
                fontsize=7.5, color="black")
    ax.axvline(full_score, color="green", linewidth=1.4, linestyle="--",
               label=f"Full model ({full_score:.4f})")
    ax.set_xlim(x_min, x_max + 0.020)
    ax.set_xlabel(f"CV NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(
        f"Phase 2 — Combo Ablation (CV NDCG@{ndcg_k})  |  Weak Router\n"
        f"Green = significantly better than full (p < {sig_alpha}),  * marks significant combos",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# STEP 8 — Weak retrieval comparison (BM25 / Dense / Static / wRRF)
# ------------------------------------------------------------

def _per_dataset_test_evaluate(cfg: dict, device: torch.device,
                               alphas_per_query: Dict[Tuple[str, str], float],
                               method_keys: List[str],
                               method_alphas: Dict[str, Optional[Dict[Tuple[str, str], float]]]) -> List[dict]:
    """
    Common helper for steps 8 / 13 / 18: evaluate the listed methods on the
    test split of every dataset and return rows ready for CSV / bar plot.

    `method_alphas[method] = None` means "use the method's intrinsic ranking"
    (BM25 / Dense), `0.5` means "static RRF", or a {(ds, qid): alpha} dict.
    """
    split = _load_split(cfg)
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])

    rows: List[dict] = []
    for ds_name in cfg["datasets"]:
        test_qids = split[ds_name]["test"]
        bm25_res, dense_res, qrels = _load_active_retrieval(cfg, ds_name, device)
        row = {"group": ds_name}
        for method in method_keys:
            ndcgs = []
            for qid in test_qids:
                if method == "bm25":
                    ranked = bm25_res.get(qid, [])
                elif method == "dense":
                    ranked = dense_res.get(qid, [])
                elif method == "static_rrf":
                    ranked = wrrf_fuse(0.5, bm25_res.get(qid, []), dense_res.get(qid, []), rrf_k)
                else:
                    a_map = method_alphas[method]
                    a = float(a_map.get((ds_name, qid), 0.5)) if a_map is not None else 0.5
                    ranked = wrrf_fuse(a, bm25_res.get(qid, []), dense_res.get(qid, []), rrf_k)
                ndcgs.append(query_ndcg_at_k(ranked, qrels.get(qid, {}), ndcg_k))
            row[method] = float(np.mean(ndcgs)) if ndcgs else 0.0
        rows.append(row)

    macro_row = {"group": "MACRO"}
    for method in method_keys:
        macro_row[method] = float(np.mean([r[method] for r in rows]))
    rows.append(macro_row)
    return rows


def _predict_weak(cfg: dict) -> Dict[Tuple[str, str], float]:
    """Apply the saved weak model to every test-split query."""
    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "models_folder", "data/models"), "weak_model.pkl"))
    cols   = bundle["feature_cols"]
    mu     = bundle["scaler_mu"]; sigma = bundle["scaler_sigma"]
    mn     = bundle["model_name"]
    mdl    = bundle["model"]

    rows = load_csv_dicts(os.path.join(
        get_config_path(cfg, "results_folder", "data/results"), "weak_dataset.csv"))
    test = [r for r in rows if r["split"] == "test"]
    X = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in test], dtype=np.float32)
    Xc = X[:, cols]
    Xz = (Xc - mu) / sigma
    preds = predict_alpha_from_model(mdl, Xz, mn)
    return {(r["ds_name"], r["qid"]): float(p) for r, p in zip(test, preds)}


def step_08_weak_retrieval_comparison(cfg: dict, device: torch.device) -> None:
    print_step_header(8, "Weak retrieval comparison (test set)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "weak_retrieval_comparison.csv")
    out_png = os.path.join(results_root, "weak_retrieval_comparison.png")
    if file_exists(out_csv) and file_exists(out_png):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    weak_alphas = _predict_weak(cfg)
    methods  = ["bm25", "dense", "static_rrf", "wrrf_weak"]
    labels   = ["BM25", "Dense", "Static RRF (α=0.5)", "wRRF (weak)"]
    colors   = METHOD_COLORS_6[:4]
    rows = _per_dataset_test_evaluate(
        cfg, device, weak_alphas, methods,
        {"wrrf_weak": weak_alphas},
    )

    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    fieldnames = ["group"] + methods
    save_csv_dicts(rows, fieldnames, out_csv)
    grouped_bar_chart(rows, methods, labels, colors,
                      ylabel=f"NDCG@{ndcg_k}",
                      title=f"Weak Router Retrieval Comparison — NDCG@{ndcg_k}",
                      out_path=out_png)
    for r in rows:
        print(f"    {r['group']:<10} " + " ".join(f"{m}={r[m]:.4f}" for m in methods))


# ------------------------------------------------------------
# STEP 9 — Plot weak alphas (box + sorted)
# ------------------------------------------------------------

def step_09_plot_weak_alphas(cfg: dict) -> None:
    print_step_header(9, "Plot weak alphas (box + sorted)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    box_png    = os.path.join(results_root, "weak_alphas_boxplot.png")
    sorted_png = os.path.join(results_root, "weak_alphas_sorted.png")
    if file_exists(box_png) and file_exists(sorted_png):
        print(f"  [SKIP] alpha plots already exist.")
        return

    weak_alphas = _predict_weak(cfg)
    oracle      = _load_oracle_alphas(cfg)

    rows = load_csv_dicts(os.path.join(results_root, "weak_dataset.csv"))
    test_rows = [r for r in rows if r["split"] == "test"]

    by_ds: Dict[str, List[float]] = {ds: [] for ds in cfg["datasets"]}
    for r in test_rows:
        by_ds[r["ds_name"]].append(weak_alphas[(r["ds_name"], r["qid"])])

    alpha_box_plot(
        by_ds,
        title="Weak Router — Predicted α distribution per dataset (test set)",
        out_path=box_png,
    )

    oracle_list = [oracle[(r["ds_name"], r["qid"])] for r in test_rows]
    pred_list   = [weak_alphas[(r["ds_name"], r["qid"])] for r in test_rows]
    alpha_sorted_plot(
        oracle_list, pred_list,
        title="Weak Router — Oracle α (sorted) vs Predicted α (test set)",
        out_path=sorted_png,
    )


# ------------------------------------------------------------
# STEP 10 — Weak SHAP (single plot for merged dataset)
# ------------------------------------------------------------

def step_10_weak_shap(cfg: dict) -> None:
    print_step_header(10, "Weak SHAP (merged dataset)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_png      = os.path.join(results_root, "weak_shap.png")
    if file_exists(out_png):
        print(f"  [SKIP] {out_png} already exists.")
        return

    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "models_folder", "data/models"), "weak_model.pkl"))
    mdl   = bundle["model"]
    mn    = bundle["model_name"]
    cols  = bundle["feature_cols"]
    mu    = bundle["scaler_mu"]; sigma = bundle["scaler_sigma"]
    feature_names = bundle["feature_names"]

    rows = load_csv_dicts(os.path.join(results_root, "weak_dataset.csv"))
    X    = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    Xc   = X[:, cols]
    Xz   = (Xc - mu) / sigma

    # Use a generic explainer (works for tree and non-tree models)
    import shap
    if mn in {"xgboost", "lightgbm", "random_forest", "extra_trees"}:
        explainer = shap.TreeExplainer(mdl)
        shap_values = explainer.shap_values(Xz)
    else:
        # KernelExplainer is slow on full data; sample background.
        background = shap.sample(Xz, min(100, len(Xz)), random_state=42)
        explainer  = shap.KernelExplainer(
            lambda x: predict_alpha_from_model(mdl, x, mn), background,
        )
        shap_values = explainer.shap_values(Xz, nsamples=200)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(
        shap_values, Xz,
        feature_names=feature_names, show=False,
        max_display=len(feature_names),
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP — Weak Router  ({mn})", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP saved: {out_png}")


# ------------------------------------------------------------
# STEP 11 — Strong-model dataset (1024-dim embedding + oracle alpha)
# ------------------------------------------------------------

def step_11_strong_dataset(cfg: dict, device: torch.device) -> None:
    print_step_header(11, "Strong-model dataset (BGE-M3 query embeddings)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_pkl = os.path.join(results_root, "strong_dataset.pkl")
    if file_exists(out_pkl):
        print(f"  [SKIP] {out_pkl} already exists.")
        return

    selected = _select_merged_qids(cfg)
    split    = _load_split(cfg)
    oracle   = _load_oracle_alphas(cfg)

    rows: List[dict] = []
    X_parts: List[np.ndarray] = []

    for ds_name in cfg["datasets"]:
        ds_dir = dataset_processed_dir(cfg, ds_name)
        q_vecs = torch.load(os.path.join(ds_dir, "query_vectors.pt"), weights_only=True).cpu().numpy()
        q_ids  = load_pickle(os.path.join(ds_dir, "query_ids.pkl"))
        qid_to_idx = {q: i for i, q in enumerate(q_ids)}

        train_set = set(split[ds_name]["train"])
        dev_set   = set(split[ds_name]["dev"])
        test_set  = set(split[ds_name]["test"])

        for qid in selected[ds_name]:
            idx = qid_to_idx.get(qid)
            if idx is None:
                raise RuntimeError(f"qid {qid} missing from {ds_name} query_vectors")
            X_parts.append(q_vecs[idx:idx + 1].astype(np.float32))
            if   qid in train_set: split_label = "train"
            elif qid in dev_set:   split_label = "dev"
            elif qid in test_set:  split_label = "test"
            else:                  split_label = "drop"
            rows.append({
                "ds_name":      ds_name,
                "qid":          qid,
                "split":        split_label,
                "oracle_alpha": float(oracle.get((ds_name, qid), 0.5)),
            })

    X_all = np.vstack(X_parts).astype(np.float32)
    save_pickle({
        "rows":  rows,
        "X":     X_all,
        "dim":   X_all.shape[1],
    }, out_pkl)
    print(f"  Saved {X_all.shape[0]} rows × {X_all.shape[1]} dims to {out_pkl}")


def _load_strong_dataset(cfg: dict):
    """Same fields as `_load_weak_dataset` but with a (N, 1024) X."""
    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "results_folder", "data/results"), "strong_dataset.pkl"))
    rows = bundle["rows"]
    X    = bundle["X"]

    indices_by_split: Dict[str, list] = {"train": [], "dev": [], "test": []}
    for i, r in enumerate(rows):
        if r["split"] in indices_by_split:
            indices_by_split[r["split"]].append(i)

    def _slice(idxs):
        Xs = X[idxs]
        ys = np.array([rows[i]["oracle_alpha"] for i in idxs], dtype=np.float32)
        qids = [rows[i]["qid"] for i in idxs]
        ds   = [rows[i]["ds_name"] for i in idxs]
        return Xs, ys, qids, ds

    X_tr, y_tr, qids_tr, ds_tr = _slice(indices_by_split["train"])
    X_dv, y_dv, qids_dv, ds_dv = _slice(indices_by_split["dev"])
    X_te, y_te, qids_te, ds_te = _slice(indices_by_split["test"])
    X_td = np.vstack([X_tr, X_dv]) if len(X_tr) and len(X_dv) else (X_tr if len(X_tr) else X_dv)
    y_td = np.concatenate([y_tr, y_dv]) if len(y_tr) and len(y_dv) else (y_tr if len(y_tr) else y_dv)
    qids_td = qids_tr + qids_dv
    ds_td   = ds_tr   + ds_dv

    return {
        "X_td":    X_td,
        "y_td":    y_td,
        "qids_td": qids_td,
        "ds_td":   ds_td,
        "X_te":    X_te,
        "y_te":    y_te,
        "qids_te": qids_te,
        "ds_te":   ds_te,
    }


# ------------------------------------------------------------
# STEP 12 — Strong-model grid search
# ------------------------------------------------------------

def step_12_strong_grid_search(cfg: dict, device: torch.device) -> None:
    print_step_header(12, "Strong-model grid search")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    models_root  = get_config_path(cfg, "models_folder",  "data/models")
    out_csv  = os.path.join(results_root, "strong_grid_search_top.csv")
    out_best = os.path.join(results_root, "strong_best_params.json")
    out_pkl  = os.path.join(models_root,  "strong_model.pkl")
    if file_exists(out_csv) and file_exists(out_best) and file_exists(out_pkl):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    seed   = int(cfg["sampling"]["random_seed"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    n_folds = int(cfg["sampling"]["cv_n_folds"])
    cap    = int(cfg.get("max_models_per_grid", 1500))
    grid_cfg = cfg.get("strong_model_grid_search", {}).get("models", {}) or {}

    combos = expand_grid(grid_cfg)
    combos = truncate_combos(combos, cap)
    print(f"  Total combos: {len(combos)}")

    ds = _load_strong_dataset(cfg)
    retrieval = _retrieval_data_per_dataset(cfg, device, cfg["datasets"])
    folds = kfold_indices(len(ds["y_td"]), n_folds, seed)

    n_jobs = 1 if torch.cuda.is_available() else -1

    t0 = time.time()
    scores = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_cv_score_one_combo)(
                model_name, params,
                ds["X_td"], ds["y_td"], ds["qids_td"], ds["ds_td"],
                None, retrieval, folds, seed, rrf_k, ndcg_k,
            )
            for model_name, params in combos
        ),
        total=len(combos), desc="  Strong grid", dynamic_ncols=True,
    ))
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    ranked = sorted(zip(combos, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:100]
    rows = [{"rank": i + 1,
             "model": mn, "params_json": json.dumps(p, sort_keys=True),
             f"cv_ndcg@{ndcg_k}": s}
            for i, ((mn, p), s) in enumerate(top)]
    save_csv_dicts(rows, ["rank", "model", "params_json", f"cv_ndcg@{ndcg_k}"], out_csv)

    (best_model, best_params), best_score = ranked[0]
    save_json({
        "model":  best_model,
        "params": best_params,
        f"cv_ndcg@{ndcg_k}": float(best_score),
    }, out_best)
    print(f"  Best: {best_model}  {best_params}  CV NDCG@{ndcg_k}={best_score:.4f}")

    # Train the final strong model on full train+dev.
    mu, sigma = zscore_stats(ds["X_td"])
    X_z = (ds["X_td"] - mu) / sigma
    y_fit = (ds["y_td"] >= 0.5).astype(int) if best_model in CLASSIFIER_MODELS else ds["y_td"]
    final_mdl = make_model(best_model, best_params, seed)
    final_mdl.fit(X_z, y_fit)
    save_pickle({
        "model_name":   best_model,
        "params":       best_params,
        "scaler_mu":    mu,
        "scaler_sigma": sigma,
        "model":        final_mdl,
    }, out_pkl)
    print(f"  Final strong model saved to {out_pkl}")


def _predict_strong(cfg: dict) -> Dict[Tuple[str, str], float]:
    """Apply the saved strong model to every test-split query."""
    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "models_folder", "data/models"), "strong_model.pkl"))
    mu, sigma = bundle["scaler_mu"], bundle["scaler_sigma"]
    mn   = bundle["model_name"]
    mdl  = bundle["model"]

    ds = _load_strong_dataset(cfg)
    Xz = (ds["X_te"] - mu) / sigma
    preds = predict_alpha_from_model(mdl, Xz, mn)
    return {(d, q): float(p) for d, q, p in zip(ds["ds_te"], ds["qids_te"], preds)}


# ------------------------------------------------------------
# STEP 13 — Strong retrieval comparison
# ------------------------------------------------------------

def step_13_strong_retrieval_comparison(cfg: dict, device: torch.device) -> None:
    print_step_header(13, "Strong retrieval comparison (test set)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "strong_retrieval_comparison.csv")
    out_png = os.path.join(results_root, "strong_retrieval_comparison.png")
    if file_exists(out_csv) and file_exists(out_png):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    weak_alphas   = _predict_weak(cfg)
    strong_alphas = _predict_strong(cfg)

    methods = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong"]
    labels  = ["BM25", "Dense", "Static RRF (α=0.5)",
               "wRRF (weak)", "wRRF (strong)"]
    colors  = METHOD_COLORS_6[:5]
    rows = _per_dataset_test_evaluate(
        cfg, device, strong_alphas, methods,
        {"wrrf_weak": weak_alphas, "wrrf_strong": strong_alphas},
    )

    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    save_csv_dicts(rows, ["group"] + methods, out_csv)
    grouped_bar_chart(rows, methods, labels, colors,
                      ylabel=f"NDCG@{ndcg_k}",
                      title=f"Strong Router Retrieval Comparison — NDCG@{ndcg_k}",
                      out_path=out_png)
    for r in rows:
        print(f"    {r['group']:<10} " + " ".join(f"{m}={r[m]:.4f}" for m in methods))


# ------------------------------------------------------------
# STEP 14 — Plot strong alphas
# ------------------------------------------------------------

def step_14_plot_strong_alphas(cfg: dict) -> None:
    print_step_header(14, "Plot strong alphas (box + sorted)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    box_png    = os.path.join(results_root, "strong_alphas_boxplot.png")
    sorted_png = os.path.join(results_root, "strong_alphas_sorted.png")
    if file_exists(box_png) and file_exists(sorted_png):
        print(f"  [SKIP] alpha plots already exist.")
        return

    strong_alphas = _predict_strong(cfg)
    oracle = _load_oracle_alphas(cfg)
    ds = _load_strong_dataset(cfg)

    by_ds: Dict[str, List[float]] = {d: [] for d in cfg["datasets"]}
    for d, q in zip(ds["ds_te"], ds["qids_te"]):
        by_ds[d].append(strong_alphas[(d, q)])

    alpha_box_plot(
        by_ds,
        title="Strong Router — Predicted α distribution per dataset (test set)",
        out_path=box_png,
    )

    oracle_list = [oracle[(d, q)] for d, q in zip(ds["ds_te"], ds["qids_te"])]
    pred_list   = [strong_alphas[(d, q)] for d, q in zip(ds["ds_te"], ds["qids_te"])]
    alpha_sorted_plot(
        oracle_list, pred_list,
        title="Strong Router — Oracle α (sorted) vs Predicted α (test set)",
        out_path=sorted_png,
    )


# ------------------------------------------------------------
# STEP 15 — MoE meta-learner dataset
# ------------------------------------------------------------

def _oof_predictions_for(model_bundle_kind: str, cfg: dict, device: torch.device,
                         n_folds: int, seed: int) -> np.ndarray:
    """
    Compute out-of-fold base-model predictions on the train+dev portion.

    `model_bundle_kind` ∈ {"weak", "strong"}; uses the corresponding
    best params from data/results/{kind}_best_params.json.  Each fold
    refits the base model on (n_folds - 1) folds and predicts on the
    held-out fold.  The set of (n_folds) folds is identical to that used
    by the grid searches (deterministic seed).
    """
    results_root = get_config_path(cfg, "results_folder", "data/results")
    if model_bundle_kind == "weak":
        ds      = _load_weak_dataset(cfg)
        best    = load_json(os.path.join(results_root, "weak_best_params.json"))
        bundle  = load_pickle(os.path.join(
            get_config_path(cfg, "models_folder", "data/models"), "weak_model.pkl"))
        cols    = bundle["feature_cols"]
        Xc      = ds["X_td"][:, cols]
    else:
        ds      = _load_strong_dataset(cfg)
        best    = load_json(os.path.join(results_root, "strong_best_params.json"))
        Xc      = ds["X_td"]

    model_name = best["model"]; params = best["params"]
    folds = kfold_indices(len(ds["y_td"]), n_folds, seed)

    aw_oof = np.full(len(ds["y_td"]), 0.5, dtype=np.float32)
    for fi, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = Xc[tr_idx], ds["y_td"][tr_idx]
        X_va       = Xc[va_idx]
        mu, sigma  = zscore_stats(X_tr)
        X_tr_z = (X_tr - mu) / sigma
        X_va_z = (X_va - mu) / sigma
        y_fit  = (y_tr >= 0.5).astype(int) if model_name in CLASSIFIER_MODELS else y_tr
        try:
            mdl = make_model(model_name, params, seed)
            mdl.fit(X_tr_z, y_fit)
            preds = predict_alpha_from_model(mdl, X_va_z, model_name)
        except Exception as exc:
            warnings.warn(f"OOF [{model_bundle_kind}] fold {fi}: {exc}")
            preds = np.full(len(va_idx), 0.5, dtype=np.float32)
        aw_oof[va_idx] = preds
        print(f"    {model_bundle_kind} OOF fold {fi+1}/{n_folds}  (val={len(va_idx)})")
    return aw_oof


def step_15_moe_dataset(cfg: dict, device: torch.device) -> None:
    print_step_header(15, "MoE meta-learner dataset (OOF base predictions)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "moe_dataset.csv")
    if file_exists(out_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    seed    = int(cfg["sampling"]["random_seed"])
    n_folds = int(cfg["sampling"]["cv_n_folds"])

    print("\n  Computing OOF predictions for the WEAK base model ...")
    aw_oof = _oof_predictions_for("weak", cfg, device, n_folds, seed)
    print("\n  Computing OOF predictions for the STRONG base model ...")
    as_oof = _oof_predictions_for("strong", cfg, device, n_folds, seed)

    weak_ds   = _load_weak_dataset(cfg)
    strong_ds = _load_strong_dataset(cfg)
    # Sanity: rows of weak/strong train+dev AND test must be aligned because
    # we index by position when packaging the MoE traindev rows below.
    assert weak_ds["qids_td"] == strong_ds["qids_td"], \
        "weak and strong datasets are not aligned (train+dev qids)"
    assert weak_ds["ds_td"]   == strong_ds["ds_td"], \
        "weak and strong datasets are not aligned (train+dev ds_names)"
    assert weak_ds["qids_te"] == strong_ds["qids_te"], \
        "weak and strong datasets are not aligned (test qids)"

    # Test predictions from the final base models.
    weak_alphas_te   = _predict_weak(cfg)
    strong_alphas_te = _predict_strong(cfg)

    rows: List[dict] = []
    # Train+dev rows (with OOF alphas)
    for i, (ds_name, qid, gt) in enumerate(zip(
        weak_ds["ds_td"], weak_ds["qids_td"], weak_ds["y_td"]
    )):
        rows.append({
            "ds_name":      ds_name,
            "qid":          qid,
            "split":        "traindev",
            "alpha_weak":   float(aw_oof[i]),
            "alpha_strong": float(as_oof[i]),
            "alpha_gt":     float(gt),
        })
    # Test rows (final-base-model alphas)
    for ds_name, qid, gt in zip(weak_ds["ds_te"], weak_ds["qids_te"], weak_ds["y_te"]):
        rows.append({
            "ds_name":      ds_name,
            "qid":          qid,
            "split":        "test",
            "alpha_weak":   float(weak_alphas_te[(ds_name, qid)]),
            "alpha_strong": float(strong_alphas_te[(ds_name, qid)]),
            "alpha_gt":     float(gt),
        })
    save_csv_dicts(
        rows,
        ["ds_name", "qid", "split", "alpha_weak", "alpha_strong", "alpha_gt"],
        out_csv,
    )
    print(f"  MoE dataset saved: {out_csv}  ({len(rows)} rows)")


def _load_moe_dataset(cfg: dict):
    rows = load_csv_dicts(os.path.join(
        get_config_path(cfg, "results_folder", "data/results"), "moe_dataset.csv"))
    td = [r for r in rows if r["split"] == "traindev"]
    te = [r for r in rows if r["split"] == "test"]
    def _arr(rows_):
        aw = np.array([float(r["alpha_weak"])   for r in rows_], dtype=np.float32)
        as_ = np.array([float(r["alpha_strong"]) for r in rows_], dtype=np.float32)
        gt = np.array([float(r["alpha_gt"])     for r in rows_], dtype=np.float32)
        ds = [r["ds_name"] for r in rows_]
        qid = [r["qid"]    for r in rows_]
        return aw, as_, gt, ds, qid
    aw_td, as_td, gt_td, ds_td, qid_td = _arr(td)
    aw_te, as_te, gt_te, ds_te, qid_te = _arr(te)
    return {
        "aw_td": aw_td, "as_td": as_td, "gt_td": gt_td,
        "ds_td": ds_td, "qid_td": qid_td,
        "aw_te": aw_te, "as_te": as_te, "gt_te": gt_te,
        "ds_te": ds_te, "qid_te": qid_te,
    }


def _moe_features(aw, as_, model_name):
    """3 features for non-tree models, 2 for trees (mirrors legacy logic)."""
    aw  = np.asarray(aw,  dtype=np.float32)
    as_ = np.asarray(as_, dtype=np.float32)
    if model_name in {"random_forest", "extra_trees", "xgboost", "lightgbm"}:
        return np.column_stack([aw, as_])
    return np.column_stack([aw, as_, np.abs(aw - as_)])


# ------------------------------------------------------------
# STEP 16 — MoE grid search
# ------------------------------------------------------------

def _cv_score_moe_combo(model_name: str, params: dict,
                        moe: dict, retrieval_data: Dict[str, dict],
                        folds: list, seed: int, rrf_k: int, ndcg_k: int) -> float:
    fold_means = []
    for tr_idx, va_idx in folds:
        X_tr = _moe_features(moe["aw_td"][tr_idx], moe["as_td"][tr_idx], model_name)
        X_va = _moe_features(moe["aw_td"][va_idx], moe["as_td"][va_idx], model_name)
        y_tr = moe["gt_td"][tr_idx]
        y_fit = (y_tr >= 0.5).astype(int) if model_name in CLASSIFIER_MODELS else y_tr
        try:
            mdl = make_model(model_name, params, seed)
            mdl.fit(X_tr, y_fit)
            preds = predict_alpha_from_model(mdl, X_va, model_name)
        except Exception as exc:
            warnings.warn(f"_cv_score_moe_combo {model_name} {params}: {exc}")
            preds = np.full(len(va_idx), 0.5, dtype=np.float32)
        ndcgs = []
        for i, alpha in zip(va_idx, preds):
            qid = moe["qid_td"][i]; ds = moe["ds_td"][i]
            rd  = retrieval_data[ds]
            ndcgs.append(_wrrf_query_ndcg(
                float(alpha), qid,
                rd["bm25_results"], rd["dense_results"], rd["qrels"],
                rrf_k, ndcg_k,
            ))
        fold_means.append(float(np.mean(ndcgs)) if ndcgs else 0.0)
    return float(np.mean(fold_means))


def step_16_moe_grid_search(cfg: dict, device: torch.device) -> None:
    print_step_header(16, "MoE meta-learner grid search")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    models_root  = get_config_path(cfg, "models_folder",  "data/models")
    out_csv  = os.path.join(results_root, "moe_grid_search_top.csv")
    out_best = os.path.join(results_root, "moe_best_params.json")
    out_pkl  = os.path.join(models_root,  "moe_model.pkl")
    if file_exists(out_csv) and file_exists(out_best) and file_exists(out_pkl):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    seed   = int(cfg["sampling"]["random_seed"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    n_folds = int(cfg["sampling"]["cv_n_folds"])
    cap    = int(cfg.get("max_models_per_grid", 1500))
    grid_cfg = cfg.get("moe_grid_search", {}).get("models", {}) or {}

    moe = _load_moe_dataset(cfg)
    retrieval = _retrieval_data_per_dataset(cfg, device, cfg["datasets"])
    folds = kfold_indices(len(moe["gt_td"]), n_folds, seed)

    combos = expand_grid(grid_cfg)
    combos = truncate_combos(combos, cap)
    print(f"  Total combos: {len(combos)}")

    n_jobs = 1 if torch.cuda.is_available() else -1

    t0 = time.time()
    scores = list(tqdm(
        Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_cv_score_moe_combo)(
                model_name, params, moe, retrieval, folds, seed, rrf_k, ndcg_k,
            )
            for model_name, params in combos
        ),
        total=len(combos), desc="  MoE grid", dynamic_ncols=True,
    ))
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    ranked = sorted(zip(combos, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:100]
    rows = [{"rank": i + 1,
             "model": mn, "params_json": json.dumps(p, sort_keys=True),
             f"cv_ndcg@{ndcg_k}": s}
            for i, ((mn, p), s) in enumerate(top)]
    save_csv_dicts(rows, ["rank", "model", "params_json", f"cv_ndcg@{ndcg_k}"], out_csv)

    (best_model, best_params), best_score = ranked[0]
    save_json({
        "model":  best_model,
        "params": best_params,
        f"cv_ndcg@{ndcg_k}": float(best_score),
    }, out_best)

    # Final MoE model on full train+dev meta features.
    X_td = _moe_features(moe["aw_td"], moe["as_td"], best_model)
    y_td = moe["gt_td"]
    y_fit = (y_td >= 0.5).astype(int) if best_model in CLASSIFIER_MODELS else y_td
    final_mdl = make_model(best_model, best_params, seed)
    final_mdl.fit(X_td, y_fit)
    save_pickle({
        "model_name": best_model,
        "params":     best_params,
        "model":      final_mdl,
    }, out_pkl)
    print(f"  Best: {best_model}  {best_params}  CV NDCG@{ndcg_k}={best_score:.4f}")
    print(f"  Final MoE model saved to {out_pkl}")


def _predict_moe(cfg: dict) -> Dict[Tuple[str, str], float]:
    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "models_folder", "data/models"), "moe_model.pkl"))
    mn  = bundle["model_name"]; mdl = bundle["model"]
    moe = _load_moe_dataset(cfg)
    X_te = _moe_features(moe["aw_te"], moe["as_te"], mn)
    preds = predict_alpha_from_model(mdl, X_te, mn)
    return {(d, q): float(p) for d, q, p in zip(moe["ds_te"], moe["qid_te"], preds)}


# ------------------------------------------------------------
# STEP 17 — MoE decision heatmap
# ------------------------------------------------------------

def step_17_moe_decision_heatmap(cfg: dict) -> None:
    print_step_header(17, "MoE decision heatmap")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_png = os.path.join(results_root, "moe_decision_heatmap.png")
    if file_exists(out_png):
        print(f"  [SKIP] {out_png} already exists.")
        return

    bundle = load_pickle(os.path.join(
        get_config_path(cfg, "models_folder", "data/models"), "moe_model.pkl"))
    mn = bundle["model_name"]; mdl = bundle["model"]
    moe = _load_moe_dataset(cfg)

    grid = 100
    aw_vals = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    as_vals = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    AW, AS  = np.meshgrid(aw_vals, as_vals)
    X_grid = _moe_features(AW.ravel(), AS.ravel(), mn)
    Z = predict_alpha_from_model(mdl, X_grid, mn).reshape(grid, grid)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    cf = ax.contourf(AS, AW, Z, levels=30, cmap="RdYlBu_r", alpha=0.85)
    plt.colorbar(cf, ax=ax,
                 label="Predicted α  (1 = prefer BM25,  0 = prefer Dense)")

    for ds_name, color in DS_PALETTE.items():
        mask_td = np.array([d == ds_name for d in moe["ds_td"]])
        mask_te = np.array([d == ds_name for d in moe["ds_te"]])
        if mask_td.any():
            ax.scatter(moe["as_td"][mask_td], moe["aw_td"][mask_td],
                       c=color, s=12, alpha=0.30, edgecolors="none", zorder=3)
        if mask_te.any():
            ax.scatter(moe["as_te"][mask_te], moe["aw_te"][mask_te],
                       c=color, s=40, alpha=0.85,
                       edgecolors="k", linewidths=0.4,
                       label=ds_name, zorder=4)

    ax.set_xlabel("α_strong", fontsize=10)
    ax.set_ylabel("α_weak",  fontsize=10)
    ax.set_title(
        f"MoE Meta-Learner Prediction Surface — {mn}\n"
        "(small = traindev, large = test)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right", title="dataset")
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {out_png}")


# ------------------------------------------------------------
# STEP 18 — MoE retrieval comparison (all 6 methods)
# ------------------------------------------------------------

def step_18_moe_retrieval_comparison(cfg: dict, device: torch.device) -> None:
    print_step_header(18, "MoE retrieval comparison (all 6 methods)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "moe_retrieval_comparison.csv")
    out_png = os.path.join(results_root, "moe_retrieval_comparison.png")
    if file_exists(out_csv) and file_exists(out_png):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    weak_alphas   = _predict_weak(cfg)
    strong_alphas = _predict_strong(cfg)
    moe_alphas    = _predict_moe(cfg)

    rows = _per_dataset_test_evaluate(
        cfg, device, moe_alphas, METHOD_KEYS_6,
        {"wrrf_weak": weak_alphas, "wrrf_strong": strong_alphas, "moe": moe_alphas},
    )

    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    save_csv_dicts(rows, ["group"] + METHOD_KEYS_6, out_csv)
    grouped_bar_chart(rows, METHOD_KEYS_6, METHOD_LABELS_6, METHOD_COLORS_6,
                      ylabel=f"NDCG@{ndcg_k}",
                      title=f"All 6 Methods Retrieval Comparison — NDCG@{ndcg_k}",
                      out_path=out_png)
    for r in rows:
        print(f"    {r['group']:<10} " + " ".join(f"{m}={r[m]:.4f}" for m in METHOD_KEYS_6))


# ------------------------------------------------------------
# STEP 19 — Plot MoE alphas
# ------------------------------------------------------------

def step_19_plot_moe_alphas(cfg: dict) -> None:
    print_step_header(19, "Plot MoE alphas (box + sorted)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    box_png    = os.path.join(results_root, "moe_alphas_boxplot.png")
    sorted_png = os.path.join(results_root, "moe_alphas_sorted.png")
    if file_exists(box_png) and file_exists(sorted_png):
        print(f"  [SKIP] alpha plots already exist.")
        return

    moe_alphas = _predict_moe(cfg)
    moe = _load_moe_dataset(cfg)
    oracle = _load_oracle_alphas(cfg)

    by_ds: Dict[str, List[float]] = {d: [] for d in cfg["datasets"]}
    for d, q in zip(moe["ds_te"], moe["qid_te"]):
        by_ds[d].append(moe_alphas[(d, q)])

    alpha_box_plot(
        by_ds,
        title="MoE — Predicted α distribution per dataset (test set)",
        out_path=box_png,
    )
    oracle_list = [oracle[(d, q)] for d, q in zip(moe["ds_te"], moe["qid_te"])]
    pred_list   = [moe_alphas[(d, q)] for d, q in zip(moe["ds_te"], moe["qid_te"])]
    alpha_sorted_plot(
        oracle_list, pred_list,
        title="MoE — Oracle α (sorted) vs Predicted α (test set)",
        out_path=sorted_png,
    )


# ------------------------------------------------------------
# STEP 20 — Recall@100 (all 6 methods)
# ------------------------------------------------------------

def step_20_recall_at_100(cfg: dict, device: torch.device) -> None:
    print_step_header(20, "Recall@100 (test set)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv   = os.path.join(results_root, "recall_at_100.csv")
    out_png   = os.path.join(results_root, "recall_at_100.png")
    ttest_csv = os.path.join(results_root, "recall_ttest.csv")
    if file_exists(out_csv) and file_exists(out_png) and file_exists(ttest_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    weak_alphas   = _predict_weak(cfg)
    strong_alphas = _predict_strong(cfg)
    moe_alphas    = _predict_moe(cfg)
    split     = _load_split(cfg)
    top_k     = int(cfg["benchmark"]["top_k"])
    rrf_k     = int(cfg["benchmark"]["rrf"]["k"])
    sig_alpha = float(cfg.get("significance_test", {}).get("alpha", 0.05))

    methods = METHOD_KEYS_6 + ["union"]
    labels  = METHOD_LABELS_6 + ["Union BM25 ∪ Dense (ceiling)"]
    colors  = METHOD_COLORS_6 + ["#BBBBBB"]

    rows: List[dict] = []
    # Accumulate per-query recall across all datasets for pairwise t-tests.
    # Only METHOD_KEYS_6 (not union) participate in t-tests.
    all_recalls: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}

    for ds_name in cfg["datasets"]:
        bm25_res, dense_res, qrels = _load_active_retrieval(cfg, ds_name, device)
        test_qids = split[ds_name]["test"]
        recalls = {m: [] for m in methods}
        for qid in test_qids:
            qrel = qrels.get(qid, {})
            if not qrel or all(g <= 0 for g in qrel.values()):
                continue
            bm = bm25_res.get(qid, [])
            de = dense_res.get(qid, [])
            cands = {
                "bm25":        [d for d, _ in bm[:top_k]],
                "dense":       [d for d, _ in de[:top_k]],
                "static_rrf":  [d for d, _ in wrrf_top_k(0.5, bm, de, rrf_k, top_k)],
                "wrrf_weak":   [d for d, _ in wrrf_top_k(weak_alphas[(ds_name, qid)],   bm, de, rrf_k, top_k)],
                "wrrf_strong": [d for d, _ in wrrf_top_k(strong_alphas[(ds_name, qid)], bm, de, rrf_k, top_k)],
                "moe":         [d for d, _ in wrrf_top_k(moe_alphas[(ds_name, qid)],    bm, de, rrf_k, top_k)],
            }
            cands["union"] = list({d for d, _ in bm} | {d for d, _ in de})

            # Compute all METHOD_KEYS_6 recalls together so alignment is guaranteed.
            pq_recalls = {}
            for m in METHOD_KEYS_6:
                r = query_recall_at_k(cands[m], qrel, len(cands[m]))
                pq_recalls[m] = r
            if all(v is not None for v in pq_recalls.values()):
                for m in METHOD_KEYS_6:
                    recalls[m].append(pq_recalls[m])
                    all_recalls[m].append(pq_recalls[m])
            # union recall (not used in t-tests)
            r_union = query_recall_at_k(cands["union"], qrel, len(cands["union"]))
            if r_union is not None:
                recalls["union"].append(r_union)

        row = {"group": ds_name}
        for m in methods:
            row[m] = float(np.mean(recalls[m])) if recalls[m] else 0.0
        rows.append(row)

    macro = {"group": "MACRO"}
    for m in methods:
        macro[m] = float(np.mean([r[m] for r in rows]))
    rows.append(macro)

    save_csv_dicts(rows, ["group"] + methods, out_csv)
    grouped_bar_chart(rows, methods, labels, colors,
                      ylabel=f"Recall@{top_k}",
                      title=f"Recall@{top_k} by Retrieval Method",
                      out_path=out_png)
    for r in rows:
        print(f"    {r['group']:<10} " + " ".join(f"{m}={r[m]:.3f}" for m in methods))

    # Pairwise paired t-tests on per-query Recall@top_k (all test queries merged).
    ttest_rows: List[dict] = []
    for i, m_a in enumerate(METHOD_KEYS_6):
        for m_b in METHOD_KEYS_6[i + 1:]:
            res = paired_t_test(all_recalls[m_a], all_recalls[m_b])
            sig = "yes" if res["p"] <= sig_alpha else "no"
            ttest_rows.append({
                "method_a":    m_a,
                "method_b":    m_b,
                "n":           res["n"],
                "mean_diff":   res["mean_diff"],
                "t":           res["t"],
                "p_value":     res["p"],
                "significant": sig,
            })
    save_csv_dicts(
        ttest_rows,
        ["method_a", "method_b", "n", "mean_diff", "t", "p_value", "significant"],
        ttest_csv,
    )
    print(f"  Recall@{top_k} t-tests saved to {ttest_csv}")
    for r in ttest_rows:
        print(f"    {r['method_a']:<13} vs {r['method_b']:<13}  "
              f"Δ={r['mean_diff']:+.4f}  p={r['p_value']:.4f}  significant={r['significant']}")


# ------------------------------------------------------------
# STEP 21 — Cross-encoder reranking
# ------------------------------------------------------------

def step_21_rerank(cfg: dict, device: torch.device) -> None:
    print_step_header(21, "Cross-encoder reranking (test set)")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv   = os.path.join(results_root, "rerank_ndcg.csv")
    out_png   = os.path.join(results_root, "rerank_ndcg.png")
    out_gain  = os.path.join(results_root, "rerank_gain.png")
    ttest_csv = os.path.join(results_root, "rerank_ttest.csv")
    if file_exists(out_csv) and file_exists(out_png) and file_exists(out_gain) and file_exists(ttest_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    rer_cfg   = cfg.get("reranker", {}) or {}
    ce_name   = str(rer_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    bs        = int(rer_cfg.get("batch_size_cuda", 128) if device.type == "cuda"
                    else rer_cfg.get("batch_size_cpu", 32))
    sig_alpha = float(cfg.get("significance_test", {}).get("alpha", 0.05))
    print(f"  Reranker: {ce_name}  batch_size={bs}")

    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder(ce_name, max_length=512, device=str(device))

    weak_alphas   = _predict_weak(cfg)
    strong_alphas = _predict_strong(cfg)
    moe_alphas    = _predict_moe(cfg)
    split  = _load_split(cfg)
    top_k  = int(cfg["benchmark"]["top_k"])
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])

    ce_short = ce_name.replace("/", "_").replace("-", "_")

    rows: List[dict] = []
    # Global per-query scores across all datasets for significance testing.
    all_orig: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    all_rer:  Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}

    for ds_name in cfg["datasets"]:
        print(f"\n  --- {ds_name} ---")
        bm25_res, dense_res, qrels = _load_active_retrieval(cfg, ds_name, device)
        ds_dir  = dataset_processed_dir(cfg, ds_name)
        corpus  = load_full_corpus(os.path.join(ds_dir, "corpus.jsonl"))
        queries = load_queries(os.path.join(ds_dir, "queries.jsonl"))
        test_qids = split[ds_name]["test"]

        method_cands_by_qid: Dict[str, Dict[str, List[str]]] = {qid: {} for qid in test_qids}
        required_pairs: set = set()
        for qid in test_qids:
            bm = bm25_res.get(qid, [])
            de = dense_res.get(qid, [])
            cands = {
                "bm25":        [d for d, _ in bm[:top_k]],
                "dense":       [d for d, _ in de[:top_k]],
                "static_rrf":  [d for d, _ in wrrf_top_k(0.5, bm, de, rrf_k, top_k)],
                "wrrf_weak":   [d for d, _ in wrrf_top_k(weak_alphas[(ds_name, qid)],   bm, de, rrf_k, top_k)],
                "wrrf_strong": [d for d, _ in wrrf_top_k(strong_alphas[(ds_name, qid)], bm, de, rrf_k, top_k)],
                "moe":         [d for d, _ in wrrf_top_k(moe_alphas[(ds_name, qid)],    bm, de, rrf_k, top_k)],
            }
            method_cands_by_qid[qid] = cands
            for m in METHOD_KEYS_6:
                for doc_id in cands[m]:
                    required_pairs.add((qid, doc_id))

        cache_path = os.path.join(results_root, f"rerank_scores_{ce_short}_{ds_name}.pkl")
        score_map = _ensure_ce_scores(
            cache_path, ce_model, bs, queries, corpus, required_pairs,
        )

        orig: Dict[str, list] = {m: [] for m in METHOD_KEYS_6}
        rer:  Dict[str, list] = {m: [] for m in METHOD_KEYS_6}
        for qid in test_qids:
            qrel = qrels.get(qid, {})
            if not qrel or all(g <= 0 for g in qrel.values()):
                continue
            for m in METHOD_KEYS_6:
                doc_ids    = method_cands_by_qid[qid][m]
                orig_pairs = [(d, 0.0) for d in doc_ids]   # order = wRRF rank
                o_score    = query_ndcg_at_k(orig_pairs, qrel, ndcg_k)
                reranked   = sorted(
                    [(d, score_map.get((qid, d), -1e9)) for d in doc_ids],
                    key=lambda t: t[1], reverse=True,
                )
                r_score = query_ndcg_at_k(reranked, qrel, ndcg_k)
                orig[m].append(o_score)
                rer[m].append(r_score)
                all_orig[m].append(o_score)
                all_rer[m].append(r_score)

        row = {"group": ds_name}
        for m in METHOD_KEYS_6:
            row[f"{m}_orig"]   = float(np.mean(orig[m])) if orig[m] else 0.0
            row[f"{m}_rerank"] = float(np.mean(rer[m]))  if rer[m]  else 0.0
        rows.append(row)
        print(f"    {ds_name}:")
        for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
            print(f"      {lab:<22} {row[f'{m}_orig']:.4f} -> {row[f'{m}_rerank']:.4f}  "
                  f"(Δ {row[f'{m}_rerank']-row[f'{m}_orig']:+.4f})")

    macro = {"group": "MACRO"}
    for m in METHOD_KEYS_6:
        macro[f"{m}_orig"]   = float(np.mean([r[f'{m}_orig']   for r in rows]))
        macro[f"{m}_rerank"] = float(np.mean([r[f'{m}_rerank'] for r in rows]))
    rows.append(macro)

    fieldnames = ["group"] + [f"{m}_{k}" for m in METHOD_KEYS_6 for k in ("orig", "rerank")]
    save_csv_dicts(rows, fieldnames, out_csv)

    plot_rows = [{"group": r["group"], **{m: r[f"{m}_rerank"] for m in METHOD_KEYS_6}} for r in rows]
    grouped_bar_chart(plot_rows, METHOD_KEYS_6, METHOD_LABELS_6, METHOD_COLORS_6,
                      ylabel=f"NDCG@{ndcg_k}",
                      title=f"NDCG@{ndcg_k} after Cross-Encoder Reranking",
                      out_path=out_png)

    gain_rows = [{"group": r["group"],
                  **{m: r[f"{m}_rerank"] - r[f"{m}_orig"] for m in METHOD_KEYS_6}} for r in rows]
    grouped_bar_chart(gain_rows, METHOD_KEYS_6, METHOD_LABELS_6, METHOD_COLORS_6,
                      ylabel=f"Δ NDCG@{ndcg_k}",
                      title=f"NDCG@{ndcg_k} Gain from Cross-Encoder Reranking",
                      out_path=out_gain, yzero=False)

    # Paired t-tests: for each method, does CE reranking significantly improve NDCG?
    ttest_rows: List[dict] = []
    print("\n  Rerank significance tests (reranked vs original per method):")
    for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
        res = paired_t_test(all_rer[m], all_orig[m])   # reranked - original
        sig = "yes" if res["p"] <= sig_alpha else "no"
        ttest_rows.append({
            "comparison":  "rerank_vs_orig",
            "method_a":    f"{m}_rerank",
            "method_b":    f"{m}_orig",
            "n":           res["n"],
            "mean_diff":   res["mean_diff"],
            "t":           res["t"],
            "p_value":     res["p"],
            "significant": sig,
        })
        print(f"    {lab:<22}  Δ={res['mean_diff']:+.4f}  p={res['p']:.4f}  significant={sig}")

    # Pairwise t-tests among reranked methods (15 pairs).
    print("\n  Reranked method pairwise t-tests:")
    for i, m_a in enumerate(METHOD_KEYS_6):
        for m_b in METHOD_KEYS_6[i + 1:]:
            res = paired_t_test(all_rer[m_a], all_rer[m_b])
            sig = "yes" if res["p"] <= sig_alpha else "no"
            ttest_rows.append({
                "comparison":  "reranked_pairwise",
                "method_a":    f"{m_a}_rerank",
                "method_b":    f"{m_b}_rerank",
                "n":           res["n"],
                "mean_diff":   res["mean_diff"],
                "t":           res["t"],
                "p_value":     res["p"],
                "significant": sig,
            })
            print(f"    {m_a}_rerank vs {m_b}_rerank  "
                  f"Δ={res['mean_diff']:+.4f}  p={res['p']:.4f}  significant={sig}")

    save_csv_dicts(
        ttest_rows,
        ["comparison", "method_a", "method_b", "n", "mean_diff", "t", "p_value", "significant"],
        ttest_csv,
    )
    print(f"  Rerank t-tests saved to {ttest_csv}")


def _ensure_ce_scores(cache_path: str, ce_model, batch_size: int,
                      queries: Dict[str, str], corpus: Dict[str, str],
                      required_pairs: set) -> Dict[Tuple[str, str], float]:
    """Cached cross-encoder scoring for a set of (qid, doc_id) pairs."""
    if file_exists(cache_path):
        try:
            cached = load_pickle(cache_path)
            if all(p in cached for p in required_pairs):
                print(f"  CE cache hit ({len(required_pairs):,} pairs).")
                return cached
            print(f"  CE cache partial — rebuilding.")
        except Exception as exc:
            print(f"  [WARN] CE cache corrupt: {exc}; rebuilding.")

    pair_list   = sorted(required_pairs)   # deterministic order
    pair_texts: List[Tuple[str, str]] = []
    for qid, doc_id in pair_list:
        q_text = queries.get(qid, "")
        d_text = corpus.get(doc_id, "")
        pair_texts.append((q_text, d_text))

    print(f"  Scoring {len(pair_texts):,} unique (q, d) pairs ...")
    scores = ce_model.predict(
        pair_texts, batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    )
    out = {pair: float(scores[i]) for i, pair in enumerate(pair_list)}
    save_pickle(out, cache_path)
    print(f"  CE scores cached: {cache_path}")
    return out


# ------------------------------------------------------------
# STEP 22 — Paired t-tests
# ------------------------------------------------------------

def step_22_significance(cfg: dict, device: torch.device) -> None:
    print_step_header(22, "Paired t-tests on per-query NDCG@10")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_csv = os.path.join(results_root, "significance_tests.csv")
    if file_exists(out_csv):
        print(f"  [SKIP] {out_csv} already exists.")
        return

    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])
    split  = _load_split(cfg)
    weak_alphas   = _predict_weak(cfg)
    strong_alphas = _predict_strong(cfg)
    moe_alphas    = _predict_moe(cfg)

    # Collect per-query NDCG@10 across all test queries (merged).
    per_method_scores: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    for ds_name in cfg["datasets"]:
        bm25_res, dense_res, qrels = _load_active_retrieval(cfg, ds_name, device)
        for qid in split[ds_name]["test"]:
            qrel = qrels.get(qid, {})
            bm = bm25_res.get(qid, [])
            de = dense_res.get(qid, [])
            method_ranked = {
                "bm25":        bm,
                "dense":       de,
                "static_rrf":  wrrf_fuse(0.5, bm, de, rrf_k),
                "wrrf_weak":   wrrf_fuse(weak_alphas[(ds_name, qid)],   bm, de, rrf_k),
                "wrrf_strong": wrrf_fuse(strong_alphas[(ds_name, qid)], bm, de, rrf_k),
                "moe":         wrrf_fuse(moe_alphas[(ds_name, qid)],    bm, de, rrf_k),
            }
            for m in METHOD_KEYS_6:
                per_method_scores[m].append(query_ndcg_at_k(method_ranked[m], qrel, ndcg_k))

    rows: List[dict] = []
    for i, m_a in enumerate(METHOD_KEYS_6):
        for m_b in METHOD_KEYS_6[i + 1:]:
            res = paired_t_test(per_method_scores[m_a], per_method_scores[m_b])
            sig = "yes" if res["p"] <= float(cfg.get("significance_test", {}).get("alpha", 0.05)) else "no"
            rows.append({
                "method_a": m_a, "method_b": m_b,
                "n":         res["n"],
                "mean_diff": res["mean_diff"],
                "t":         res["t"],
                "p_value":   res["p"],
                "significant": sig,
            })
    save_csv_dicts(
        rows, ["method_a", "method_b", "n", "mean_diff", "t", "p_value", "significant"],
        out_csv,
    )
    print(f"  Saved {len(rows)} paired tests to {out_csv}")
    for r in rows:
        print(f"    {r['method_a']:<13} vs {r['method_b']:<13}  "
              f"Δ={r['mean_diff']:+.4f}  p={r['p_value']:.4f}  significant={r['significant']}")


# ============================================================
# Section C — Pretty headers and main()
# ============================================================

def print_step_header(n: int, title: str) -> None:
    bar = "=" * 72
    print(f"\n\n{bar}\nSTEP {n:02d} — {title}\n{bar}")


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg = load_config()
    seed = int(cfg["sampling"]["random_seed"])
    set_global_seed(seed)
    device = get_device()
    print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")

    step_01_download(cfg)
    step_02_preprocess(cfg, device)
    step_03_optimize_bm25(cfg, device)
    step_04_oracle_alpha(cfg, device)
    step_05_weak_dataset(cfg, device)
    step_06_weak_grid_search(cfg, device)
    step_07_weak_ablation(cfg, device)
    step_08_weak_retrieval_comparison(cfg, device)
    step_09_plot_weak_alphas(cfg)
    step_10_weak_shap(cfg)
    step_11_strong_dataset(cfg, device)
    step_12_strong_grid_search(cfg, device)
    step_13_strong_retrieval_comparison(cfg, device)
    step_14_plot_strong_alphas(cfg)
    step_15_moe_dataset(cfg, device)
    step_16_moe_grid_search(cfg, device)
    step_17_moe_decision_heatmap(cfg)
    step_18_moe_retrieval_comparison(cfg, device)
    step_19_plot_moe_alphas(cfg)
    step_20_recall_at_100(cfg, device)
    step_21_rerank(cfg, device)
    step_22_significance(cfg, device)

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
