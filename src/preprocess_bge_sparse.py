"""
preprocess_bge_sparse.py

Builds BGE-M3 sparse retrieval artifacts for all configured datasets.
run preprocess.py first so that corpus.jsonl / queries.jsonl exist.

This script only adds sparse-head artifacts; existing dense files are
left completely untouched.

Artifacts written alongside the dense files in
  data/processed_data/bge-m3/<dataset>/  :

  bge_sparse_corpus_ids.pkl          [doc_id, ...]  (ordered)
  bge_sparse_inverted_index.pkl      {token_id: [(doc_idx, weight), ...]}
  bge_sparse_doc_freq.pkl            {token_id: int}  n docs with nonzero weight
  bge_sparse_query_vectors.pkl       {qid: {token_id: float}}
  bge_sparse_results_topk_<k>.pkl    {qid: [(doc_id, score), ...]}

Usage
-----
  python src/preprocess_bge_sparse.py
"""

import collections
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import (
    ensure_dir,
    file_exists,
    get_config_path,
    load_config,
    load_corpus_batch_generator,
    load_queries,
    load_pickle,
    model_short_name,
    save_pickle,
)

# Batch size for sparse encoding (smaller than dense — sparse head is heavier).
_ENCODE_BATCH = 32


def _is_nonempty(path):
    return file_exists(path) and os.path.getsize(path) > 0


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode_sparse(model, texts, max_length=8192):
    """
    Run the BGE-M3 sparse head on a list of texts.
    Returns a list of  {token_id (int): weight (float)}  dicts.
    """
    with torch.no_grad():
        out = model.encode(
            texts,
            batch_size=len(texts),
            max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    return [{int(k): float(v) for k, v in sv.items() if v > 0}
            for sv in out["lexical_weights"]]


# ── Corpus sparse index ───────────────────────────────────────────────────────

def build_sparse_corpus_index(corpus_jsonl, model, batch_size, max_length=8192):
    """
    Encode every corpus document with the BGE-M3 sparse head and build
    an inverted index plus a document-frequency table.

    Returns
    -------
    corpus_ids      : list[str]                    ordered doc IDs
    inverted_index  : dict[int, list[(int, float)]]  token_id -> [(doc_idx, w)]
    sparse_doc_freq : dict[int, int]               token_id -> n_docs_with_nonzero_w
    """
    # Load all texts into memory first (largest corpus ~57k docs ≈ few hundred MB).
    all_ids, all_texts = [], []
    for ids_batch, texts_batch in load_corpus_batch_generator(corpus_jsonl, 1024):
        all_ids.extend(ids_batch)
        all_texts.extend(texts_batch)

    n = len(all_ids)
    print(f"  Corpus: {n:,} documents")

    corpus_ids      = []
    inverted_index  = collections.defaultdict(list)
    sparse_doc_freq = collections.defaultdict(int)
    doc_idx         = 0

    n_batches = math.ceil(n / batch_size)
    for b in tqdm(range(n_batches), desc="  Encoding corpus (sparse)", dynamic_ncols=True):
        start     = b * batch_size
        end       = min(start + batch_size, n)
        b_ids     = all_ids[start:end]
        b_texts   = all_texts[start:end]
        sparse_vecs = _encode_sparse(model, b_texts, max_length)

        for did, sv in zip(b_ids, sparse_vecs):
            corpus_ids.append(did)
            for tok_id, w in sv.items():
                inverted_index[tok_id].append((doc_idx, w))
                sparse_doc_freq[tok_id] += 1
            doc_idx += 1

    return corpus_ids, dict(inverted_index), dict(sparse_doc_freq)


# ── Query sparse vectors ──────────────────────────────────────────────────────

def build_sparse_query_vectors(queries, model, batch_size, max_length=8192):
    """
    Encode all queries with the BGE-M3 sparse head.
    Returns  {qid: {token_id: float}}.
    """
    qids   = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    result = {}

    n_batches = math.ceil(len(qids) / batch_size)
    for b in tqdm(range(n_batches), desc="  Encoding queries (sparse)", dynamic_ncols=True):
        start = b * batch_size
        end   = min(start + batch_size, len(qids))
        svecs = _encode_sparse(model, qtexts[start:end], max_length)
        for qid, sv in zip(qids[start:end], svecs):
            result[qid] = sv

    return result


# ── Sparse retrieval ──────────────────────────────────────────────────────────

def run_sparse_retrieval(query_sparse_vecs, inverted_index, corpus_ids, top_k):
    """
    For every query, scan the inverted index and return top-k documents
    ranked by the dot product of query and document sparse weights.

    Returns  {qid: [(doc_id, score), ...]}.
    """
    n_docs  = len(corpus_ids)
    results = {}

    for qid, qsv in tqdm(query_sparse_vecs.items(),
                          desc="  Sparse retrieval", dynamic_ncols=True):
        scores = np.zeros(n_docs, dtype=np.float32)
        for tok_id, w_q in qsv.items():
            posting = inverted_index.get(tok_id)
            if posting is None:
                continue
            for d_idx, w_d in posting:
                scores[d_idx] += w_q * w_d

        k       = min(top_k, n_docs)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results[qid] = [(corpus_ids[i], float(scores[i])) for i in top_idx]

    return results


# ── Per-dataset driver ────────────────────────────────────────────────────────

def run_for_dataset(ds_name, cfg, model):
    short_model    = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir         = os.path.join(processed_root, short_model, ds_name)
    top_k          = int(cfg["benchmark"]["top_k"])

    ensure_dir(ds_dir)

    corpus_jsonl  = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")

    ids_pkl      = os.path.join(ds_dir, "bge_sparse_corpus_ids.pkl")
    index_pkl    = os.path.join(ds_dir, "bge_sparse_inverted_index.pkl")
    docfreq_pkl  = os.path.join(ds_dir, "bge_sparse_doc_freq.pkl")
    qvec_pkl     = os.path.join(ds_dir, "bge_sparse_query_vectors.pkl")
    results_pkl  = os.path.join(ds_dir, f"bge_sparse_results_topk_{top_k}.pkl")

    print(f"\n{'='*60}")
    print(f"Dataset : {ds_name}")
    print(f"Dir     : {ds_dir}")
    print(f"Top-k   : {top_k}")
    print(f"{'='*60}")

    # ── Step 1: Corpus index ──────────────────────────────────────────────────
    if _is_nonempty(ids_pkl) and _is_nonempty(index_pkl) and _is_nonempty(docfreq_pkl):
        print("[1/3] Sparse corpus index exists — skipping.")
        corpus_ids      = load_pickle(ids_pkl)
        inverted_index  = load_pickle(index_pkl)
        sparse_doc_freq = load_pickle(docfreq_pkl)
    else:
        print("[1/3] Building BGE-M3 sparse corpus index ...")
        corpus_ids, inverted_index, sparse_doc_freq = build_sparse_corpus_index(
            corpus_jsonl, model, _ENCODE_BATCH
        )
        save_pickle(corpus_ids,      ids_pkl)
        save_pickle(inverted_index,  index_pkl)
        save_pickle(sparse_doc_freq, docfreq_pkl)
        print(
            f"  Saved: {len(corpus_ids):,} docs | "
            f"{len(inverted_index):,} unique tokens in index"
        )

    total_docs = len(corpus_ids)

    # ── Step 2: Query sparse vectors ──────────────────────────────────────────
    if _is_nonempty(qvec_pkl):
        print("[2/3] Sparse query vectors exist — skipping.")
        query_sparse_vecs = load_pickle(qvec_pkl)
    else:
        print("[2/3] Encoding queries with BGE-M3 sparse head ...")
        queries = load_queries(queries_jsonl)
        query_sparse_vecs = build_sparse_query_vectors(queries, model, _ENCODE_BATCH)
        save_pickle(query_sparse_vecs, qvec_pkl)
        print(f"  Saved: {len(query_sparse_vecs):,} query sparse vectors")

    # ── Step 3: Retrieval ─────────────────────────────────────────────────────
    if _is_nonempty(results_pkl):
        print(f"[3/3] Sparse retrieval results (top-{top_k}) exist — skipping.")
    else:
        print(f"[3/3] Running BGE-M3 sparse retrieval (top-{top_k}) ...")
        sparse_results = run_sparse_retrieval(
            query_sparse_vecs, inverted_index, corpus_ids, top_k
        )
        save_pickle(sparse_results, results_pkl)
        print(f"  Saved: results for {len(sparse_results):,} queries")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg        = load_config()
    datasets   = cfg["datasets"]
    model_name = cfg["embeddings"]["model_name"]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device     : {device}")
    print(f"Model      : {model_name}")
    print(f"Datasets   : {', '.join(datasets)}")

    if "bge-m3" not in model_name.lower() and "bge_m3" not in model_name.lower():
        print(
            f"[WARN] config model_name is '{model_name}'. "
            "This script is designed for BAAI/bge-m3. "
            "The sparse head may not work correctly with other models."
        )

    print("\nLoading BGEM3FlagModel ...")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"))

    for ds_name in datasets:
        run_for_dataset(ds_name, cfg, model)

    print("\nBGE-M3 sparse preprocessing complete.")


if __name__ == "__main__":
    main()
