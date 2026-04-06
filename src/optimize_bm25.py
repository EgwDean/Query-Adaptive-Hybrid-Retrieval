"""optimize_bm25.py

Grid-search globally optimal BM25 parameters (k1, b, use_stemming) across all
configured datasets and write a macro-averaged summary plus the best config.

For each parameter combination:
  1. Build/reuse tokenized corpus, frequency indexes, and BM25 index (all cached).
  2. Run/reuse BM25 retrieval results (cached per combination).
  3. Evaluate NDCG@k per dataset; compute macro-average across datasets.

Outputs (written to data/results/):
  - bm25_optimization_macro.csv  — all combinations sorted by macro NDCG@k
  - bm25_optimization_best.json  — the single best configuration

Run:
    python src/optimize_bm25.py
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.preprocess import (
    build_bm25_and_word_freq_index,
    build_doc_freq_index,
    preprocess_corpus,
    preprocess_queries,
)
from src.utils import (
    download_beir_dataset,
    ensure_dir,
    file_exists,
    get_config_path,
    load_beir_dataset,
    load_config,
    load_pickle,
    load_qrels,
    load_queries,
    model_short_name,
    save_pickle,
    stem_and_tokenize,
    write_corpus_jsonl,
    write_qrels_tsv,
    write_queries_jsonl,
)


# ---------------------------------------------------------------------------
# NDCG helpers (self-contained so this script has no legacy dependencies)
# ---------------------------------------------------------------------------

def _query_ndcg_at_k(ranked_pairs, rels, k):
    if not rels:
        return 0.0
    dcg = sum(
        ((2.0 ** rels.get(doc_id, 0)) - 1.0) / math.log2(rank + 1)
        for rank, (doc_id, _) in enumerate(ranked_pairs[:k], start=1)
    )
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(((2.0 ** r) - 1.0) / math.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _dataset_ndcg_at_k(score_map, qrels, k):
    ndcgs = []
    for qid, rels in qrels.items():
        ranked = sorted(score_map.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        ndcgs.append(_query_ndcg_at_k(ranked, rels, k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ---------------------------------------------------------------------------
# BM25 retrieval (self-contained)
# ---------------------------------------------------------------------------

def _run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k, use_stemming):
    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
    results = {}
    for qid, qtext in queries.items():
        tokens = stem_and_tokenize(qtext, stemmer)
        scores = bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k <= 0:
            results[qid] = []
            continue
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in top_idx]
    return results


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _nonempty(path):
    return file_exists(path) and os.path.getsize(path) > 0


def _ensure_base_exports(dataset_name, cfg):
    """Write corpus/queries/qrels to processed_data if not already there."""
    datasets_folder = get_config_path(cfg, "datasets_folder", "data/datasets")
    processed_folder = get_config_path(cfg, "processed_folder", "data/processed_data")
    short_model = model_short_name(cfg["embeddings"]["model_name"])
    ds_dir = os.path.join(processed_folder, short_model, dataset_name)
    ensure_dir(ds_dir)

    corpus_jsonl  = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv     = os.path.join(ds_dir, "qrels.tsv")

    if _nonempty(corpus_jsonl) and _nonempty(queries_jsonl) and _nonempty(qrels_tsv):
        return ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv

    dataset_path = download_beir_dataset(dataset_name, datasets_folder)
    if dataset_path is None:
        raise RuntimeError(f"Dataset download/verification failed: {dataset_name}")
    corpus, queries, qrels, split = load_beir_dataset(dataset_path)
    if corpus is None:
        raise RuntimeError(f"Failed to load BEIR dataset: {dataset_name}")

    print(f"  Exporting {dataset_name} (split={split}, corpus={len(corpus):,}, queries={len(queries):,})")
    write_corpus_jsonl(corpus, corpus_jsonl)
    write_queries_jsonl(queries, queries_jsonl)
    write_qrels_tsv(qrels, qrels_tsv)
    return ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv


def _ensure_sparse_artifacts(dataset_name, cfg, k1, b, use_stemming):
    """Build/reuse tokenized corpus, frequency indexes, and BM25 index."""
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    top_k = int(cfg.get("benchmark", {}).get("top_k", 100))
    ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv = _ensure_base_exports(dataset_name, cfg)
    paths = u.bm25_artifact_paths(ds_dir, k1, b, use_stemming, top_k=top_k)

    if not _nonempty(paths["tokenized_corpus_jsonl"]):
        preprocess_corpus(corpus_jsonl, paths["tokenized_corpus_jsonl"], stemmer_lang, use_stemming)

    preprocess_queries(queries_jsonl, paths["tokenized_queries_jsonl"],
                       paths["query_tokens_pkl"], stemmer_lang, use_stemming)

    needs_freq = not _nonempty(paths["word_freq_pkl"]) or not _nonempty(paths["doc_freq_pkl"])
    needs_bm25 = not _nonempty(paths["bm25_pkl"]) or not _nonempty(paths["bm25_docids_pkl"])

    bm25 = bm25_ids = None
    if needs_freq or needs_bm25:
        bm25, bm25_ids, global_counts, total_tokens = build_bm25_and_word_freq_index(
            paths["tokenized_corpus_jsonl"], k1=k1, b=b
        )
    if needs_freq:
        doc_freq, total_docs = build_doc_freq_index(paths["tokenized_corpus_jsonl"])
        save_pickle((global_counts, total_tokens), paths["word_freq_pkl"])
        save_pickle((doc_freq, total_docs), paths["doc_freq_pkl"])
    if needs_bm25:
        save_pickle(bm25, paths["bm25_pkl"])
        save_pickle(bm25_ids, paths["bm25_docids_pkl"])

    return paths, queries_jsonl, qrels_tsv, stemmer_lang


def _run_or_load_results(dataset_name, cfg, k1, b, use_stemming):
    """Load cached BM25 results for one dataset/config, or run and cache them."""
    top_k = int(cfg.get("benchmark", {}).get("top_k", 100))
    paths, queries_jsonl, qrels_tsv, stemmer_lang = _ensure_sparse_artifacts(
        dataset_name, cfg, k1, b, use_stemming
    )

    bm25_results = None
    if _nonempty(paths["bm25_results_pkl"]):
        try:
            bm25_results = load_pickle(paths["bm25_results_pkl"])
        except Exception as exc:
            print(f"  [WARN] BM25 results cache corrupt for {dataset_name}; rebuilding. ({exc})")

    if bm25_results is None:
        queries = load_queries(queries_jsonl)
        try:
            bm25     = load_pickle(paths["bm25_pkl"])
            bm25_ids = load_pickle(paths["bm25_docids_pkl"])
        except Exception:
            bm25, bm25_ids, _, _ = build_bm25_and_word_freq_index(
                paths["tokenized_corpus_jsonl"], k1=k1, b=b
            )
            save_pickle(bm25,     paths["bm25_pkl"])
            save_pickle(bm25_ids, paths["bm25_docids_pkl"])

        bm25_results = _run_bm25_retrieval(bm25, bm25_ids, queries, stemmer_lang, top_k, use_stemming)
        save_pickle(bm25_results, paths["bm25_results_pkl"])

    qrels = load_qrels(qrels_tsv)
    return bm25_results, qrels


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def optimize_bm25(cfg):
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets in config.")

    ndcg_k  = int(cfg.get("benchmark", {}).get("ndcg_k", 10))
    grid    = cfg.get("bm25_optimization", {}) or {}
    k1_vals  = [float(v) for v in grid.get("k1_values",  [0.8, 1.2, 1.5, 1.6, 2.0])]
    b_vals   = [float(v) for v in grid.get("b_values",   [0.0, 0.25, 0.5, 0.75, 1.0])]
    stem_vals = [bool(v) for v in grid.get("use_stemming_values", [True, False])]

    combos = list(itertools.product(k1_vals, b_vals, stem_vals))
    macro_rows = []

    for k1, b, use_stemming in tqdm(combos, desc="BM25 grid search", dynamic_ncols=True):
        ds_ndcgs = []
        for ds in datasets:
            bm25_results, qrels = _run_or_load_results(ds, cfg, k1, b, use_stemming)
            score_map = {qid: dict(pairs) for qid, pairs in bm25_results.items()}
            ds_ndcgs.append(_dataset_ndcg_at_k(score_map, qrels, ndcg_k))

        macro_rows.append({
            "k1": k1, "b": b, "use_stemming": use_stemming,
            f"macro_ndcg@{ndcg_k}": float(np.mean(ds_ndcgs)) if ds_ndcgs else 0.0,
        })

    macro_rows.sort(key=lambda r: r[f"macro_ndcg@{ndcg_k}"], reverse=True)
    return macro_rows, ndcg_k


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(cfg, macro_rows, ndcg_k):
    results_root = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_root)

    macro_csv  = os.path.join(results_root, "bm25_optimization_macro.csv")
    best_json  = os.path.join(results_root, "bm25_optimization_best.json")
    best = macro_rows[0]

    with open(macro_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "b", "use_stemming", f"macro_ndcg@{ndcg_k}"])
        for row in macro_rows:
            writer.writerow([
                f"{row['k1']:.4f}", f"{row['b']:.4f}",
                int(row["use_stemming"]),
                f"{row[f'macro_ndcg@{ndcg_k}']:.6f}",
            ])

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump({
            "k1": float(best["k1"]),
            "b":  float(best["b"]),
            "use_stemming": bool(best["use_stemming"]),
            f"macro_ndcg@{ndcg_k}": float(best[f"macro_ndcg@{ndcg_k}"]),
        }, f, indent=2)

    return macro_csv, best_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find optimal BM25 parameters across configured datasets."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()
    random.seed(int(cfg.get("routing_features", {}).get("seed", 42)))

    datasets = cfg.get("datasets", [])
    grid     = cfg.get("bm25_optimization", {}) or {}
    k1_vals  = grid.get("k1_values",  [0.8, 1.2, 1.5, 1.6, 2.0])
    b_vals   = grid.get("b_values",   [0.0, 0.25, 0.5, 0.75, 1.0])
    stem_vals = grid.get("use_stemming_values", [True, False])
    n_combos = len(k1_vals) * len(b_vals) * len(stem_vals)

    print("=" * 60)
    print("BM25 parameter optimization")
    print(f"Datasets  : {', '.join(datasets)}")
    print(f"Grid size : {n_combos} combinations")
    print("=" * 60)

    macro_rows, ndcg_k = optimize_bm25(cfg)
    macro_csv, best_json = write_results(cfg, macro_rows, ndcg_k)

    best = macro_rows[0]
    print(f"\nBest: k1={best['k1']:.4f}  b={best['b']:.4f}  "
          f"stemming={bool(best['use_stemming'])}  "
          f"macro_ndcg@{ndcg_k}={best[f'macro_ndcg@{ndcg_k}']:.6f}")
    print(f"\nOutputs:\n  {macro_csv}\n  {best_json}")


if __name__ == "__main__":
    main()
