"""
eval_trec_covid.py
==================

Cross-domain evaluation on TREC-COVID using pre-trained models from the
Query-Adaptive Hybrid Retrieval pipeline.  TREC-COVID is a biomedical /
COVID-19 dataset completely outside the training distribution (scifact,
nfcorpus, arguana, fiqa, scidocs), making it a clean zero-shot probe.

All expensive outputs are cached so re-running is fast.

Output files  (data/results/trec_covid/):
  metrics.csv          - NDCG@10, Recall@100, Reranked-NDCG@10 per method
  ttest_ndcg.csv       - Pairwise paired t-tests on per-query NDCG@10
  ttest_recall.csv     - Pairwise paired t-tests on per-query Recall@100
  ttest_rerank.csv     - Rerank improvement per method + pairwise reranked tests

Run:
    python src/eval_trec_covid.py
    python src/eval_trec_covid.py --corpus-chunk 256 --rerank-batch 64
"""

import argparse
import itertools
import json
import math
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import (
    download_beir_dataset,
    ensure_dir,
    ensure_english_stopwords,
    file_exists,
    get_config_path,
    load_beir_dataset,
    load_config,
    load_json,
    load_pickle,
    model_short_name,
    paired_t_test,
    query_ndcg_at_k,
    query_recall_at_k,
    save_csv_dicts,
    save_pickle,
    stem_and_tokenize,
    wrrf_fuse,
    wrrf_top_k,
)
from src.pipeline import (
    CLASSIFIER_MODELS,
    FEATURE_NAMES,
    METHOD_COLORS_6,
    METHOD_KEYS_6,
    METHOD_LABELS_6,
    _compute_query_features,
    _moe_features,
    make_model,
    predict_alpha_from_model,
    zscore_stats,
)

DATASET_NAME = "trec-covid"


# ─────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────

def _ds_dir(cfg: dict) -> str:
    short = model_short_name(cfg["embeddings"]["model_name"])
    root  = get_config_path(cfg, "processed_folder", "data/processed")
    return os.path.join(root, short, DATASET_NAME)


def _results_dir(cfg: dict) -> str:
    root = get_config_path(cfg, "results_folder", "data/results")
    return os.path.join(root, "trec_covid")


def _active_bm25_params(cfg: dict) -> dict:
    results_root = get_config_path(cfg, "results_folder", "data/results")
    best_json    = os.path.join(results_root, "bm25_best_params.json")
    if file_exists(best_json):
        d = load_json(best_json)
        return {"k1": float(d["k1"]), "b": float(d["b"]),
                "use_stemming": bool(d["use_stemming"])}
    raw = cfg.get("bm25", {}) or {}
    return {"k1": float(raw.get("k1", 1.5)), "b": float(raw.get("b", 0.75)),
            "use_stemming": bool(raw.get("use_stemming", True))}


def _print_step(n: int, title: str) -> None:
    bar = "=" * 72
    print(f"\n\n{bar}\nSTEP {n:02d} — {title}\n{bar}")


# ─────────────────────────────────────────────────────────────
# Step 1 — Download TREC-COVID
# ─────────────────────────────────────────────────────────────

def step_01_download(cfg: dict) -> Tuple[dict, dict, dict]:
    _print_step(1, f"Download {DATASET_NAME}")
    datasets_root = get_config_path(cfg, "datasets_folder", "data/datasets")
    ds_path = os.path.join(datasets_root, DATASET_NAME)
    if not os.path.isdir(ds_path):
        print(f"  Downloading {DATASET_NAME} ...")
        download_beir_dataset(DATASET_NAME, datasets_root)
    else:
        print(f"  [SKIP] {ds_path} already exists.")
    corpus, queries, qrels, _ = load_beir_dataset(ds_path)
    n_judgments = sum(len(v) for v in qrels.values())
    print(f"  corpus={len(corpus):,}  queries={len(queries):,}  "
          f"judgments={n_judgments:,}")
    return corpus, queries, qrels


# ─────────────────────────────────────────────────────────────
# Step 2 — Tokenise corpus for BM25 + feature computation
# ─────────────────────────────────────────────────────────────

def step_02_tokenise(cfg: dict, corpus: dict, bm25_params: dict,
                     stemmer_lang: str
                     ) -> Tuple[list, dict, dict, dict, int, int]:
    """
    Returns (doc_ids, doc_id_to_tokens, word_freq, doc_freq,
             total_corpus_tokens, total_docs).
    """
    _print_step(2, "Tokenise corpus")
    ds_dir = _ds_dir(cfg)
    ensure_dir(ds_dir)
    suffix       = "stem" if bm25_params["use_stemming"] else "nostem"
    cache_tok    = os.path.join(ds_dir, f"doc_tokens_{suffix}.pkl")
    cache_wf     = os.path.join(ds_dir, f"word_freq_{suffix}.pkl")
    cache_df     = os.path.join(ds_dir, f"doc_freq_{suffix}.pkl")

    if file_exists(cache_tok) and file_exists(cache_wf) and file_exists(cache_df):
        print("  [SKIP] tokenisation cache hit.")
        doc_id_to_tokens = load_pickle(cache_tok)
        word_freq, total_corpus_tokens = load_pickle(cache_wf)
        doc_freq,  total_docs          = load_pickle(cache_df)
        return (list(doc_id_to_tokens.keys()), doc_id_to_tokens,
                word_freq, doc_freq, total_corpus_tokens, total_docs)

    stemmer = None
    if bm25_params["use_stemming"]:
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer(stemmer_lang)

    doc_id_to_tokens: dict  = {}
    word_freq:  Dict[str, int] = {}
    doc_freq:   Dict[str, int] = {}
    total_corpus_tokens = 0

    for doc_id, doc in tqdm(corpus.items(), desc="  Tokenising", dynamic_ncols=True):
        text   = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        tokens = stem_and_tokenize(text, stemmer)
        doc_id_to_tokens[doc_id] = tokens
        total_corpus_tokens += len(tokens)
        seen: set = set()
        for t in tokens:
            word_freq[t] = word_freq.get(t, 0) + 1
            if t not in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                seen.add(t)

    total_docs = len(corpus)
    save_pickle(doc_id_to_tokens, cache_tok)
    save_pickle((word_freq, total_corpus_tokens), cache_wf)
    save_pickle((doc_freq, total_docs), cache_df)
    print(f"  {total_docs:,} docs  |  {total_corpus_tokens:,} tokens  |  "
          f"vocab={len(word_freq):,}")
    return (list(doc_id_to_tokens.keys()), doc_id_to_tokens,
            word_freq, doc_freq, total_corpus_tokens, total_docs)


# ─────────────────────────────────────────────────────────────
# Step 3 — Build BM25 index
# ─────────────────────────────────────────────────────────────

def step_03_build_bm25(cfg: dict, doc_ids: list, doc_id_to_tokens: dict,
                       bm25_params: dict):
    _print_step(3, "Build BM25 index")
    ds_dir = _ds_dir(cfg)
    suffix = "stem" if bm25_params["use_stemming"] else "nostem"
    cache  = os.path.join(ds_dir,
                          f"bm25_index_{suffix}_{bm25_params['k1']}_{bm25_params['b']}.pkl")
    if file_exists(cache):
        print("  [SKIP] BM25 index cached.")
        return load_pickle(cache)

    from rank_bm25 import BM25Okapi
    token_lists = [doc_id_to_tokens[d] for d in doc_ids]
    print(f"  Building BM25Okapi for {len(doc_ids):,} docs "
          f"(k1={bm25_params['k1']}, b={bm25_params['b']}) ...")
    bm25 = BM25Okapi(token_lists, k1=bm25_params["k1"], b=bm25_params["b"])
    save_pickle(bm25, cache)
    print(f"  Index cached: {cache}")
    return bm25


# ─────────────────────────────────────────────────────────────
# Step 4 — Encode corpus (batched, CUDA-safe)
# ─────────────────────────────────────────────────────────────

def step_04_encode_corpus(cfg: dict, corpus: dict, doc_ids: list,
                          device: torch.device,
                          corpus_chunk: int) -> torch.Tensor:
    """
    Encodes the corpus in `corpus_chunk`-document batches to keep GPU memory
    under control.  Each batch is moved to CPU before the next batch begins.
    """
    _print_step(4, f"Encode corpus  (chunk={corpus_chunk}, device={device})")
    ds_dir    = _ds_dir(cfg)
    cache_emb = os.path.join(ds_dir, "corpus_embeddings.pt")
    cache_ids = os.path.join(ds_dir, "corpus_ids.pkl")

    if file_exists(cache_emb) and file_exists(cache_ids):
        if load_pickle(cache_ids) == doc_ids:
            print(f"  [SKIP] corpus embeddings cached ({len(doc_ids):,} docs).")
            return torch.load(cache_emb, weights_only=True)
        print("  [WARN] cached doc_ids mismatch — re-encoding.")

    model_name = cfg["embeddings"]["model_name"]
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name, device=str(device))

    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
             for d in doc_ids]

    all_embs: List[torch.Tensor] = []
    n_chunks = math.ceil(len(texts) / corpus_chunk)
    for start in tqdm(range(0, len(texts), corpus_chunk),
                      total=n_chunks, desc="  Corpus chunks", dynamic_ncols=True):
        batch = texts[start: start + corpus_chunk]
        with torch.no_grad():
            embs = st_model.encode(
                batch, convert_to_tensor=True,
                show_progress_bar=False, device=str(device),
            )
        all_embs.append(embs.cpu().float())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del st_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    corpus_embs = torch.cat(all_embs, dim=0)
    torch.save(corpus_embs, cache_emb)
    save_pickle(doc_ids, cache_ids)
    print(f"  Saved: {cache_emb}  shape={tuple(corpus_embs.shape)}")
    return corpus_embs


# ─────────────────────────────────────────────────────────────
# Step 5 — Encode queries
# ─────────────────────────────────────────────────────────────

def step_05_encode_queries(cfg: dict, queries: dict,
                           device: torch.device,
                           query_chunk: int) -> Tuple[torch.Tensor, list]:
    _print_step(5, f"Encode queries  (chunk={query_chunk})")
    ds_dir    = _ds_dir(cfg)
    cache_vec = os.path.join(ds_dir, "query_vectors.pt")
    cache_ids = os.path.join(ds_dir, "query_ids.pkl")
    qids      = list(queries.keys())

    if file_exists(cache_vec) and file_exists(cache_ids):
        if load_pickle(cache_ids) == qids:
            print(f"  [SKIP] query vectors cached ({len(qids)} queries).")
            return torch.load(cache_vec, weights_only=True), qids
        print("  [WARN] cached query ids mismatch — re-encoding.")

    model_name = cfg["embeddings"]["model_name"]
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name, device=str(device))

    texts    = [queries[q] for q in qids]
    all_embs: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), query_chunk),
                      desc="  Query chunks", dynamic_ncols=True):
        batch = texts[start: start + query_chunk]
        with torch.no_grad():
            embs = st_model.encode(
                batch, convert_to_tensor=True,
                show_progress_bar=False, device=str(device),
            )
        all_embs.append(embs.cpu().float())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del st_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    q_vecs = torch.cat(all_embs, dim=0)
    torch.save(q_vecs, cache_vec)
    save_pickle(qids, cache_ids)
    print(f"  Saved: {cache_vec}  shape={tuple(q_vecs.shape)}")
    return q_vecs, qids


# ─────────────────────────────────────────────────────────────
# Step 6 — BM25 retrieval
# ─────────────────────────────────────────────────────────────

def step_06_bm25_retrieval(cfg: dict, bm25, doc_ids: list,
                           query_tokens: Dict[str, list],
                           bm25_params: dict,
                           top_k: int) -> Dict[str, List[Tuple[str, float]]]:
    _print_step(6, f"BM25 retrieval  (top-{top_k})")
    ds_dir = _ds_dir(cfg)
    suffix = "stem" if bm25_params["use_stemming"] else "nostem"
    cache  = os.path.join(ds_dir,
                          f"bm25_results_{suffix}_{bm25_params['k1']}_"
                          f"{bm25_params['b']}_top{top_k}.pkl")
    if file_exists(cache):
        print("  [SKIP] BM25 results cached.")
        return load_pickle(cache)

    results: Dict[str, List[Tuple[str, float]]] = {}
    for qid, q_tokens in tqdm(query_tokens.items(),
                               desc="  BM25 retrieval", dynamic_ncols=True):
        if not q_tokens:
            results[qid] = []
            continue
        scores  = bm25.get_scores(q_tokens)
        k       = min(top_k, len(scores))
        if k == 0:
            results[qid] = []
            continue
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in top_idx]

    save_pickle(results, cache)
    print(f"  Cached BM25 results ({len(results)} queries).")
    return results


# ─────────────────────────────────────────────────────────────
# Step 7 — Dense retrieval
# ─────────────────────────────────────────────────────────────

def step_07_dense_retrieval(cfg: dict, q_vecs: torch.Tensor, qids: list,
                             corpus_embs: torch.Tensor, doc_ids: list,
                             device: torch.device,
                             top_k: int) -> Dict[str, List[Tuple[str, float]]]:
    _print_step(7, f"Dense retrieval  (top-{top_k})")
    ds_dir   = _ds_dir(cfg)
    short    = model_short_name(cfg["embeddings"]["model_name"])
    cache    = os.path.join(ds_dir, f"dense_results_{short}_top{top_k}.pkl")

    if file_exists(cache):
        print("  [SKIP] dense results cached.")
        return load_pickle(cache)

    from sentence_transformers import util as st_util
    dense_cfg = cfg.get("dense_search", {}) or {}
    q_chunk   = int(dense_cfg.get("query_chunk_size", 100))
    c_chunk   = int(dense_cfg.get("corpus_chunk_size", 50000))

    print(f"  Moving corpus embeddings to {device} ...")
    try:
        corpus_on_device = corpus_embs.to(device)
    except RuntimeError:
        print("  [WARN] OOM moving corpus to GPU; using CPU.")
        corpus_on_device = corpus_embs.cpu()
        device = torch.device("cpu")

    results: Dict[str, List[Tuple[str, float]]] = {}
    for start in tqdm(range(0, len(qids), q_chunk),
                      desc="  Dense retrieval", dynamic_ncols=True):
        end   = min(start + q_chunk, len(qids))
        batch = q_vecs[start:end].to(corpus_on_device.device)
        hits  = st_util.semantic_search(
            batch, corpus_on_device, top_k=top_k, corpus_chunk_size=c_chunk,
        )
        for i, hit_list in enumerate(hits):
            qid = qids[start + i]
            results[qid] = [(doc_ids[h["corpus_id"]], float(h["score"]))
                            for h in hit_list]

    del corpus_on_device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    save_pickle(results, cache)
    print(f"  Cached dense results ({len(results)} queries).")
    return results


# ─────────────────────────────────────────────────────────────
# Step 8 — Weak-model features
# ─────────────────────────────────────────────────────────────

def step_08_compute_features(cfg: dict, queries: dict,
                              query_tokens: Dict[str, list],
                              bm25_results: dict, dense_results: dict,
                              word_freq: dict, total_corpus_tokens: int,
                              doc_freq: dict, total_docs: int,
                              stopword_stems: frozenset,
                              ) -> Dict[str, np.ndarray]:
    """Compute the 16 hand-crafted features for every query."""
    _print_step(8, "Compute weak-model features")
    ds_dir = _ds_dir(cfg)
    cache  = os.path.join(ds_dir, "query_features.pkl")

    if file_exists(cache):
        print("  [SKIP] feature cache hit.")
        return load_pickle(cache)

    rcfg           = cfg.get("routing_features", {}) or {}
    overlap_k      = int(rcfg.get("overlap_k", 10))
    feature_stat_k = int(rcfg.get("feature_stat_k", 10))
    epsilon        = float(rcfg.get("epsilon", 1e-8))
    ce_alpha       = float(rcfg.get("ce_smoothing_alpha", 1.0))

    feat_dict: Dict[str, np.ndarray] = {}
    for qid, raw_text in tqdm(queries.items(), desc="  Features", dynamic_ncols=True):
        feats = _compute_query_features(
            raw_text,
            query_tokens.get(qid, []),
            bm25_results.get(qid, []),
            dense_results.get(qid, []),
            word_freq, total_corpus_tokens,
            doc_freq,  total_docs,
            stopword_stems, overlap_k, feature_stat_k, epsilon, ce_alpha,
        )
        feat_dict[qid] = np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)

    save_pickle(feat_dict, cache)
    print(f"  Computed features for {len(feat_dict)} queries.")
    return feat_dict


# ─────────────────────────────────────────────────────────────
# Step 9 — Predict alphas (weak / strong / MoE)
# ─────────────────────────────────────────────────────────────

def step_09_predict_alphas(cfg: dict, queries: dict,
                            feat_dict: Dict[str, np.ndarray],
                            q_vecs: torch.Tensor,
                            qids_ordered: list,
                            ) -> Tuple[dict, dict, dict]:
    """
    Apply pretrained weak, strong, and MoE models.
    Returns three {qid: float} dicts.
    """
    _print_step(9, "Predict alphas  (weak / strong / MoE)")
    models_root = get_config_path(cfg, "models_folder", "data/models")
    all_qids    = list(queries.keys())   # same order as feat_dict was built

    # ── Weak model ──────────────────────────────────────────────────────────
    w_b     = load_pickle(os.path.join(models_root, "weak_model.pkl"))
    w_cols  = w_b["feature_cols"]
    X_all   = np.stack([feat_dict[q] for q in all_qids], axis=0)
    Xc      = X_all[:, w_cols]
    Xz      = (Xc - w_b["scaler_mu"]) / w_b["scaler_sigma"]
    w_preds = predict_alpha_from_model(w_b["model"], Xz, w_b["model_name"])
    weak_alphas = {q: float(p) for q, p in zip(all_qids, w_preds)}
    print(f"  Weak  ({w_b['model_name']}):  "
          f"mean={np.mean(w_preds):.3f}  std={np.std(w_preds):.3f}  "
          f"[{np.min(w_preds):.3f}, {np.max(w_preds):.3f}]")

    # ── Strong model ─────────────────────────────────────────────────────────
    s_b       = load_pickle(os.path.join(models_root, "strong_model.pkl"))
    qid_to_vi = {q: i for i, q in enumerate(qids_ordered)}
    X_strong  = np.stack([q_vecs[qid_to_vi[q]].numpy() for q in all_qids], axis=0)
    Xz_s      = (X_strong - s_b["scaler_mu"]) / s_b["scaler_sigma"]
    s_preds   = predict_alpha_from_model(s_b["model"], Xz_s, s_b["model_name"])
    strong_alphas = {q: float(p) for q, p in zip(all_qids, s_preds)}
    print(f"  Strong ({s_b['model_name']}): "
          f"mean={np.mean(s_preds):.3f}  std={np.std(s_preds):.3f}  "
          f"[{np.min(s_preds):.3f}, {np.max(s_preds):.3f}]")

    # ── MoE model ────────────────────────────────────────────────────────────
    m_b      = load_pickle(os.path.join(models_root, "moe_model.pkl"))
    aw_arr   = np.array([weak_alphas[q]   for q in all_qids], dtype=np.float32)
    as_arr   = np.array([strong_alphas[q] for q in all_qids], dtype=np.float32)
    X_moe    = _moe_features(aw_arr, as_arr, m_b["model_name"])
    m_preds  = predict_alpha_from_model(m_b["model"], X_moe, m_b["model_name"])
    moe_alphas = {q: float(p) for q, p in zip(all_qids, m_preds)}
    print(f"  MoE   ({m_b['model_name']}):  "
          f"mean={np.mean(m_preds):.3f}  std={np.std(m_preds):.3f}  "
          f"[{np.min(m_preds):.3f}, {np.max(m_preds):.3f}]")

    return weak_alphas, strong_alphas, moe_alphas


# ─────────────────────────────────────────────────────────────
# Step 10 — Evaluate NDCG@k + Recall@k
# ─────────────────────────────────────────────────────────────

def step_10_evaluate(cfg: dict, queries: dict, qrels: dict,
                     bm25_results: dict, dense_results: dict,
                     weak_alphas: dict, strong_alphas: dict, moe_alphas: dict,
                     top_k: int,
                     ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], list]:
    """
    Returns (per_ndcg, per_recall, all_qids_ordered).
    per_ndcg[m]   — one NDCG@k per query (all queries, including no-qrel → 0.0)
    per_recall[m] — one Recall@k per query with relevant docs (aligned across methods)
    all_qids_ordered — iteration order of queries dict (needed for alignment later)
    """
    _print_step(10, "Evaluate NDCG@10 + Recall@100  (all 6 methods)")
    rrf_k  = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k = int(cfg["benchmark"]["ndcg_k"])

    per_ndcg:   Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    per_recall: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    all_qids_ordered: list = []

    for qid in tqdm(queries, desc="  Evaluating", dynamic_ncols=True):
        all_qids_ordered.append(qid)
        qrel    = qrels.get(qid, {})
        has_rel = bool(qrel) and any(g > 0 for g in qrel.values())
        bm      = bm25_results.get(qid, [])
        de      = dense_results.get(qid, [])

        ranked = {
            "bm25":        bm,
            "dense":       de,
            "static_rrf":  wrrf_fuse(0.5,                  bm, de, rrf_k),
            "wrrf_weak":   wrrf_fuse(weak_alphas[qid],     bm, de, rrf_k),
            "wrrf_strong": wrrf_fuse(strong_alphas[qid],   bm, de, rrf_k),
            "moe":         wrrf_fuse(moe_alphas[qid],      bm, de, rrf_k),
        }

        # Per-method recall: only appended if query has relevant docs (all methods together
        # so lists stay aligned).
        if has_rel:
            recall_ok = True
            pq_recalls: Dict[str, Optional[float]] = {}
            for m in METHOD_KEYS_6:
                cands = [d for d, _ in ranked[m][:top_k]]
                r     = query_recall_at_k(cands, qrel, top_k)
                pq_recalls[m] = r
                if r is None:
                    recall_ok = False
            if recall_ok:
                for m in METHOD_KEYS_6:
                    per_recall[m].append(pq_recalls[m])

        for m in METHOD_KEYS_6:
            per_ndcg[m].append(query_ndcg_at_k(ranked[m], qrel, ndcg_k))

    n_rel = len(per_recall[METHOD_KEYS_6[0]])
    print(f"\n  n_queries={len(queries)}  n_with_relevant_docs={n_rel}")
    print(f"\n  {'Method':<22}  NDCG@{ndcg_k}   Recall@{top_k}")
    print("  " + "-" * 46)
    for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
        nd = np.mean(per_ndcg[m])   if per_ndcg[m]   else 0.0
        rc = np.mean(per_recall[m]) if per_recall[m] else 0.0
        print(f"  {lab:<22}  {nd:.4f}     {rc:.4f}")

    return per_ndcg, per_recall, all_qids_ordered


# ─────────────────────────────────────────────────────────────
# Step 11 — Cross-encoder reranking
# ─────────────────────────────────────────────────────────────

def step_11_rerank(cfg: dict, queries: dict, corpus: dict, qrels: dict,
                   bm25_results: dict, dense_results: dict,
                   weak_alphas: dict, strong_alphas: dict, moe_alphas: dict,
                   device: torch.device, top_k: int,
                   rerank_batch: int,
                   ) -> Tuple[Dict[str, List[float]], list]:
    """
    CE reranking.
    Returns (per_rer, has_rel_qids_ordered).
    per_rer[m] — per-query reranked NDCG@k for queries with relevant docs.
    has_rel_qids_ordered — the qids in the same order as per_rer rows.
    """
    _print_step(11, f"Cross-encoder reranking  (top-{top_k})")
    results_dir = _results_dir(cfg)
    ensure_dir(results_dir)
    rrf_k   = int(cfg["benchmark"]["rrf"]["k"])
    ndcg_k  = int(cfg["benchmark"]["ndcg_k"])
    rer_cfg = cfg.get("reranker", {}) or {}
    ce_name = str(rer_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    ce_short = ce_name.replace("/", "_").replace("-", "_")
    cache    = os.path.join(results_dir, f"rerank_scores_{ce_short}.pkl")

    # Build candidate sets and required (qid, doc_id) pairs.
    method_cands: Dict[str, Dict[str, List[str]]] = {}
    required_pairs: set = set()
    for qid in queries:
        bm = bm25_results.get(qid, [])
        de = dense_results.get(qid, [])
        c = {
            "bm25":        [d for d, _ in bm[:top_k]],
            "dense":       [d for d, _ in de[:top_k]],
            "static_rrf":  [d for d, _ in wrrf_top_k(0.5,                bm, de, rrf_k, top_k)],
            "wrrf_weak":   [d for d, _ in wrrf_top_k(weak_alphas[qid],   bm, de, rrf_k, top_k)],
            "wrrf_strong": [d for d, _ in wrrf_top_k(strong_alphas[qid], bm, de, rrf_k, top_k)],
            "moe":         [d for d, _ in wrrf_top_k(moe_alphas[qid],    bm, de, rrf_k, top_k)],
        }
        method_cands[qid] = c
        for m in METHOD_KEYS_6:
            for doc_id in c[m]:
                required_pairs.add((qid, doc_id))

    # Load or compute CE scores.
    score_map: Optional[dict] = None
    if file_exists(cache):
        try:
            cached = load_pickle(cache)
            if all(p in cached for p in required_pairs):
                score_map = cached
                print(f"  [SKIP] CE cache hit ({len(required_pairs):,} pairs).")
        except Exception as exc:
            print(f"  [WARN] CE cache corrupt: {exc}")

    if score_map is None:
        from sentence_transformers import CrossEncoder
        print(f"  Loading CrossEncoder: {ce_name}")
        ce_model  = CrossEncoder(ce_name, max_length=512, device=str(device))
        pair_list = sorted(required_pairs)
        pair_texts = []
        for qid, doc_id in pair_list:
            q_text = queries[qid]
            d_text = (corpus[doc_id].get("title", "") + " " +
                      corpus[doc_id].get("text", "")).strip()
            pair_texts.append((q_text, d_text))

        print(f"  Scoring {len(pair_texts):,} (q, d) pairs  batch={rerank_batch} ...")
        scores = ce_model.predict(
            pair_texts, batch_size=rerank_batch,
            show_progress_bar=True, convert_to_numpy=True,
        )
        score_map = {pair: float(scores[i]) for i, pair in enumerate(pair_list)}
        save_pickle(score_map, cache)
        print(f"  Scores cached: {cache}")

    # Compute per-query reranked NDCG (only queries with relevant docs).
    per_rer: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    has_rel_qids_ordered: list = []

    for qid in queries:
        qrel = qrels.get(qid, {})
        if not qrel or all(g <= 0 for g in qrel.values()):
            continue
        has_rel_qids_ordered.append(qid)
        for m in METHOD_KEYS_6:
            doc_ids  = method_cands[qid][m]
            reranked = sorted(
                [(d, score_map.get((qid, d), -1e9)) for d in doc_ids],
                key=lambda t: t[1], reverse=True,
            )
            per_rer[m].append(query_ndcg_at_k(reranked, qrel, ndcg_k))

    print(f"\n  Reranked NDCG@{ndcg_k}  (n={len(has_rel_qids_ordered)} queries with qrels):")
    for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
        nd = np.mean(per_rer[m]) if per_rer[m] else 0.0
        print(f"  {lab:<22}  {nd:.4f}")

    return per_rer, has_rel_qids_ordered


# ─────────────────────────────────────────────────────────────
# Step 12 — Paired t-tests + save all results
# ─────────────────────────────────────────────────────────────

def step_12_ttest_and_save(cfg: dict,
                            per_ndcg: Dict[str, List[float]],
                            per_recall: Dict[str, List[float]],
                            per_rer: Dict[str, List[float]],
                            all_qids_ordered: list,
                            has_rel_qids_ordered: list,
                            ) -> None:
    _print_step(12, "Paired t-tests + save results")
    results_dir = _results_dir(cfg)
    ensure_dir(results_dir)
    sig_alpha = float(cfg.get("significance_test", {}).get("alpha", 0.05))
    ndcg_k    = int(cfg["benchmark"]["ndcg_k"])
    top_k     = int(cfg["benchmark"]["top_k"])

    def _ttest_rows(pairs_iter, scores_a_fn, scores_b_fn,
                    comparison_tag: str) -> List[dict]:
        rows = []
        for m_a, m_b in pairs_iter:
            a = scores_a_fn(m_a)
            b = scores_b_fn(m_b)
            n = min(len(a), len(b))
            if n == 0:
                continue
            res = paired_t_test(a[:n], b[:n])
            rows.append({
                "comparison":  comparison_tag,
                "method_a":    m_a,
                "method_b":    m_b,
                "n":           res["n"],
                "mean_diff":   res["mean_diff"],
                "t":           res["t"],
                "p_value":     res["p"],
                "significant": "yes" if res["p"] <= sig_alpha else "no",
            })
        return rows

    fields_base   = ["method_a", "method_b", "n", "mean_diff", "t", "p_value", "significant"]
    fields_tagged = ["comparison"] + fields_base

    # Pairwise NDCG@k t-tests (all queries).
    ndcg_pairs = [(m_a, m_b) for i, m_a in enumerate(METHOD_KEYS_6)
                  for m_b in METHOD_KEYS_6[i + 1:]]
    ttest_ndcg = _ttest_rows(ndcg_pairs,
                              lambda m: per_ndcg[m], lambda m: per_ndcg[m],
                              "ndcg_pairwise")
    save_csv_dicts(ttest_ndcg, fields_base,
                   os.path.join(results_dir, "ttest_ndcg.csv"))
    print(f"\n  NDCG@{ndcg_k} pairwise t-tests:")
    for r in ttest_ndcg:
        print(f"    {r['method_a']:<13} vs {r['method_b']:<13}  "
              f"Δ={r['mean_diff']:+.4f}  p={r['p_value']:.4f}  sig={r['significant']}")

    # Pairwise Recall@k t-tests (aligned per-recall lists).
    ttest_recall = _ttest_rows(ndcg_pairs,
                                lambda m: per_recall[m], lambda m: per_recall[m],
                                "recall_pairwise")
    save_csv_dicts(ttest_recall, fields_base,
                   os.path.join(results_dir, "ttest_recall.csv"))
    print(f"\n  Recall@{top_k} pairwise t-tests:")
    for r in ttest_recall:
        print(f"    {r['method_a']:<13} vs {r['method_b']:<13}  "
              f"Δ={r['mean_diff']:+.4f}  p={r['p_value']:.4f}  sig={r['significant']}")

    # Rerank t-tests.
    # Need aligned orig NDCG for queries with relevant docs (same order as per_rer).
    has_rel_set = set(has_rel_qids_ordered)
    aligned_orig: Dict[str, List[float]] = {m: [] for m in METHOD_KEYS_6}
    for i, qid in enumerate(all_qids_ordered):
        if qid in has_rel_set:
            for m in METHOD_KEYS_6:
                aligned_orig[m].append(per_ndcg[m][i])

    ttest_rerank: List[dict] = []
    # Per-method: does reranking help?
    for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
        n = min(len(per_rer[m]), len(aligned_orig[m]))
        if n == 0:
            continue
        res = paired_t_test(per_rer[m][:n], aligned_orig[m][:n])
        ttest_rerank.append({
            "comparison":  "rerank_vs_orig",
            "method_a":    f"{m}_rerank",
            "method_b":    f"{m}_orig",
            "n":           res["n"],
            "mean_diff":   res["mean_diff"],
            "t":           res["t"],
            "p_value":     res["p"],
            "significant": "yes" if res["p"] <= sig_alpha else "no",
        })
    # Pairwise between reranked methods.
    for i, m_a in enumerate(METHOD_KEYS_6):
        for m_b in METHOD_KEYS_6[i + 1:]:
            n = min(len(per_rer[m_a]), len(per_rer[m_b]))
            if n == 0:
                continue
            res = paired_t_test(per_rer[m_a][:n], per_rer[m_b][:n])
            ttest_rerank.append({
                "comparison":  "reranked_pairwise",
                "method_a":    f"{m_a}_rerank",
                "method_b":    f"{m_b}_rerank",
                "n":           res["n"],
                "mean_diff":   res["mean_diff"],
                "t":           res["t"],
                "p_value":     res["p"],
                "significant": "yes" if res["p"] <= sig_alpha else "no",
            })
    save_csv_dicts(ttest_rerank, fields_tagged,
                   os.path.join(results_dir, "ttest_rerank.csv"))

    print(f"\n  Reranking significance (reranked vs original):")
    for r in [x for x in ttest_rerank if x["comparison"] == "rerank_vs_orig"]:
        print(f"    {r['method_a']:<20}  Δ={r['mean_diff']:+.4f}  "
              f"p={r['p_value']:.4f}  sig={r['significant']}")

    # Summary CSV + console table.
    summary_rows = []
    for m, lab in zip(METHOD_KEYS_6, METHOD_LABELS_6):
        summary_rows.append({
            "method":                  m,
            "label":                   lab,
            f"ndcg@{ndcg_k}":         float(np.mean(per_ndcg[m]))   if per_ndcg[m]   else 0.0,
            f"recall@{top_k}":        float(np.mean(per_recall[m])) if per_recall[m] else 0.0,
            f"rerank_ndcg@{ndcg_k}":  float(np.mean(per_rer[m]))    if per_rer[m]    else 0.0,
        })
    save_csv_dicts(
        summary_rows,
        ["method", "label", f"ndcg@{ndcg_k}", f"recall@{top_k}",
         f"rerank_ndcg@{ndcg_k}"],
        os.path.join(results_dir, "metrics.csv"),
    )

    print(f"\n{'=' * 72}")
    print(f"TREC-COVID Cross-Domain Evaluation  —  Summary")
    print(f"{'=' * 72}")
    print(f"  {'Method':<22}  NDCG@{ndcg_k}   Recall@{top_k}   "
          f"Rerank-NDCG@{ndcg_k}")
    print("  " + "-" * 65)
    for r in summary_rows:
        print(f"  {r['label']:<22}  "
              f"{r[f'ndcg@{ndcg_k}']:.4f}    "
              f"{r[f'recall@{top_k}']:.4f}       "
              f"{r[f'rerank_ndcg@{ndcg_k}']:.4f}")
    print(f"\n  Results saved to: {results_dir}/")
    print(f"    metrics.csv       — summary table")
    print(f"    ttest_ndcg.csv    — pre-reranking NDCG@{ndcg_k} pairwise t-tests")
    print(f"    ttest_recall.csv  — Recall@{top_k} pairwise t-tests")
    print(f"    ttest_rerank.csv  — reranking improvement + reranked pairwise t-tests")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-domain evaluation on TREC-COVID using pre-trained router models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",        default="config.yaml",
                        help="Path to pipeline config YAML.")
    parser.add_argument("--corpus-chunk",  type=int, default=512,
                        help="Documents per batch during corpus encoding.")
    parser.add_argument("--query-chunk",   type=int, default=50,
                        help="Queries per batch during query encoding.")
    parser.add_argument("--top-k",         type=int, default=None,
                        help="Retrieval top-k (default: from config benchmark.top_k).")
    parser.add_argument("--rerank-batch",  type=int, default=None,
                        help="CE reranker batch size (default: from config reranker).")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}  |  CUDA: {torch.cuda.is_available()}")

    top_k = args.top_k or int(cfg["benchmark"]["top_k"])
    rer_cfg      = cfg.get("reranker", {}) or {}
    rerank_batch = args.rerank_batch or int(
        rer_cfg.get("batch_size_cuda", 128) if device.type == "cuda"
        else rer_cfg.get("batch_size_cpu", 32)
    )
    bm25_params  = _active_bm25_params(cfg)
    stemmer_lang = cfg.get("preprocessing", {}).get("stemmer_language", "english")

    print(f"BM25   : {bm25_params}")
    print(f"Top-K  : {top_k}  |  Rerank batch: {rerank_batch}")
    print(f"Chunks : corpus={args.corpus_chunk}  query={args.query_chunk}")

    # Stopwords / stemmer shared across steps 2 and 8.
    stopwords = ensure_english_stopwords()
    if bm25_params["use_stemming"]:
        from nltk.stem.snowball import SnowballStemmer
        stemmer        = SnowballStemmer(stemmer_lang)
        stopword_stems = frozenset(stemmer.stem(w) for w in stopwords)
    else:
        stemmer        = None
        stopword_stems = frozenset(w.lower() for w in stopwords)

    # ── Steps ──────────────────────────────────────────────────────────────

    corpus, queries, qrels = step_01_download(cfg)

    (doc_ids, doc_id_to_tokens,
     word_freq, doc_freq,
     total_corpus_tokens, total_docs) = step_02_tokenise(cfg, corpus, bm25_params,
                                                          stemmer_lang)

    query_tokens = {qid: stem_and_tokenize(text, stemmer)
                    for qid, text in queries.items()}

    bm25 = step_03_build_bm25(cfg, doc_ids, doc_id_to_tokens, bm25_params)

    corpus_embs = step_04_encode_corpus(cfg, corpus, doc_ids, device, args.corpus_chunk)

    q_vecs, qids_ordered = step_05_encode_queries(cfg, queries, device, args.query_chunk)

    bm25_results = step_06_bm25_retrieval(cfg, bm25, doc_ids, query_tokens,
                                          bm25_params, top_k)

    dense_results = step_07_dense_retrieval(cfg, q_vecs, qids_ordered,
                                            corpus_embs, doc_ids, device, top_k)

    feat_dict = step_08_compute_features(
        cfg, queries, query_tokens, bm25_results, dense_results,
        word_freq, total_corpus_tokens, doc_freq, total_docs, stopword_stems,
    )

    weak_alphas, strong_alphas, moe_alphas = step_09_predict_alphas(
        cfg, queries, feat_dict, q_vecs, qids_ordered,
    )

    per_ndcg, per_recall, all_qids_ordered = step_10_evaluate(
        cfg, queries, qrels, bm25_results, dense_results,
        weak_alphas, strong_alphas, moe_alphas, top_k,
    )

    per_rer, has_rel_qids_ordered = step_11_rerank(
        cfg, queries, corpus, qrels, bm25_results, dense_results,
        weak_alphas, strong_alphas, moe_alphas, device, top_k, rerank_batch,
    )

    step_12_ttest_and_save(
        cfg, per_ndcg, per_recall, per_rer,
        all_qids_ordered, has_rel_qids_ordered,
    )


if __name__ == "__main__":
    main()
