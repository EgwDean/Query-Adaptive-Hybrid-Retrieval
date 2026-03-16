"""
preproces.py

Builds preprocessing artifacts for configured datasets and caches them on disk.
The script always skips steps that are already available in processed_data.

Artifacts built here:
  - corpus.jsonl / queries.jsonl / qrels.tsv
  - tokenized_corpus.jsonl
  - tokenized_queries.jsonl
  - bm25_index.pkl + bm25_doc_ids.pkl
  - word_freq_index.pkl
  - corpus_embeddings.pt + corpus_ids.pkl
  - query_vectors.pt + query_ids.pkl
  - query_tokens.pkl
"""

import argparse
import json
import os
import sys
import torch
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer

# Make sure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    load_config,
    ensure_dir,
    file_exists,
    save_pickle,
    load_pickle,
    write_corpus_jsonl,
    write_queries_jsonl,
    write_qrels_tsv,
    load_queries,
    download_beir_dataset,
    load_beir_dataset,
    model_short_name,
    stem_and_tokenize,
)
from src.pipeline import (
    preprocess_corpus,
    build_bm25_and_word_freq_index,
    build_corpus_embeddings,
    build_dense_query_vectors,
)


def preprocess_queries(queries_jsonl, tokenized_queries_jsonl, query_tokens_pkl, stemmer_lang):
    """Tokenize/stem queries and cache both JSONL and dict-by-id forms."""
    if file_exists(tokenized_queries_jsonl) and file_exists(query_tokens_pkl):
        print("  Query preprocessing cache exists. Skipping.")
        return

    queries = load_queries(queries_jsonl)
    stemmer = SnowballStemmer(stemmer_lang)
    token_map = {}

    with open(tokenized_queries_jsonl, "w", encoding="utf-8") as out:
        for qid, qtext in queries.items():
            tokens = stem_and_tokenize(qtext, stemmer)
            token_map[qid] = tokens
            out.write(json.dumps({"_id": qid, "tokens": tokens}) + "\n")

    save_pickle(token_map, query_tokens_pkl)


def run_for_dataset(dataset_name, cfg, model, device):
    datasets_folder = u.get_config_path(cfg, "datasets_folder", "data/datasets")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    ds_dir = os.path.join(processed_folder, short_model, dataset_name)
    ensure_dir(ds_dir)

    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    emb_batch_size = cfg["embeddings"]["batch_size"]

    corpus_jsonl = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    tokenized_corpus_jsonl = os.path.join(ds_dir, "tokenized_corpus.jsonl")
    tokenized_queries_jsonl = os.path.join(ds_dir, "tokenized_queries.jsonl")
    query_tokens_pkl = os.path.join(ds_dir, "query_tokens.pkl")
    bm25_pkl = os.path.join(ds_dir, "bm25_index.pkl")
    bm25_docids_pkl = os.path.join(ds_dir, "bm25_doc_ids.pkl")
    word_freq_pkl = os.path.join(ds_dir, "word_freq_index.pkl")
    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Processed output: {ds_dir}")
    print(f"{'=' * 60}")

    # 1) Ensure dataset is available locally
    dataset_path = download_beir_dataset(dataset_name, datasets_folder)
    if dataset_path is None:
        print("  [ERROR] Dataset download/verification failed. Skipping dataset.")
        return

    # 2) Export corpus/queries/qrels once
    if file_exists(corpus_jsonl) and file_exists(queries_jsonl) and file_exists(qrels_tsv):
        print("[1/6] Corpus/queries/qrels already cached. Skipping.")
    else:
        print("[1/6] Loading BEIR data and writing corpus/queries/qrels ...")
        corpus, queries, qrels, split = load_beir_dataset(dataset_path)
        if corpus is None:
            print("  [ERROR] Failed to load BEIR dataset. Skipping dataset.")
            return
        print(f"  Split: {split} | Corpus: {len(corpus):,} | Queries: {len(queries):,}")
        write_corpus_jsonl(corpus, corpus_jsonl)
        write_queries_jsonl(queries, queries_jsonl)
        write_qrels_tsv(qrels, qrels_tsv)

    # 3) Preprocess corpus and queries
    if file_exists(tokenized_corpus_jsonl):
        print("[2/6] Tokenized corpus exists. Skipping.")
    else:
        print("[2/6] Tokenizing/stemming corpus ...")
        preprocess_corpus(corpus_jsonl, tokenized_corpus_jsonl, stemmer_lang)

    print("[3/6] Preprocessing queries ...")
    preprocess_queries(
        queries_jsonl,
        tokenized_queries_jsonl,
        query_tokens_pkl,
        stemmer_lang,
    )

    # 4) BM25 + frequency index
    has_index = file_exists(bm25_pkl) and file_exists(bm25_docids_pkl) and file_exists(word_freq_pkl)
    if has_index:
        print("[4/6] BM25 and frequency indexes exist. Skipping.")
    else:
        print("[4/6] Building BM25 + frequency indexes ...")
        bm25, bm25_doc_ids, global_counts, total_corpus_tokens = build_bm25_and_word_freq_index(
            tokenized_corpus_jsonl
        )
        save_pickle(bm25, bm25_pkl)
        save_pickle(bm25_doc_ids, bm25_docids_pkl)
        save_pickle((global_counts, total_corpus_tokens), word_freq_pkl)

    # 5) Corpus embeddings
    if file_exists(corpus_emb_pt) and file_exists(corpus_ids_pkl):
        print("[5/6] Corpus embeddings exist. Skipping.")
    else:
        print("[5/6] Encoding corpus ...")
        corpus_embeddings, corpus_ids = build_corpus_embeddings(
            corpus_jsonl,
            model,
            emb_batch_size,
            device,
        )
        torch.save(corpus_embeddings, corpus_emb_pt)
        save_pickle(corpus_ids, corpus_ids_pkl)

    # 6) Query embeddings
    if file_exists(query_vectors_pt) and file_exists(query_ids_pkl):
        print("[6/6] Query embeddings exist. Skipping.")
    else:
        print("[6/6] Encoding queries ...")
        queries = load_queries(queries_jsonl)
        query_vectors, query_ids = build_dense_query_vectors(
            queries,
            model,
            emb_batch_size,
            device,
        )
        torch.save(query_vectors, query_vectors_pt)
        save_pickle(query_ids, query_ids_pkl)


def main():
    parser = argparse.ArgumentParser(
        description="Build cached preprocessing artifacts for configured datasets."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets configured. Nothing to preprocess.")
        return

    # Always ensure the three top-level data folders exist.
    datasets_folder = u.get_config_path(cfg, "datasets_folder", "data/datasets")
    results_folder = u.get_config_path(cfg, "results_folder", "data/results")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")
    ensure_dir(datasets_folder)
    ensure_dir(results_folder)
    ensure_dir(processed_folder)

    model_name = cfg["embeddings"]["model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")

    print("\nLoading embedding model ...")
    model = SentenceTransformer(model_name, device=device)

    max_seq = cfg["embeddings"].get("max_seq_length")
    if max_seq is not None:
        model.max_seq_length = int(max_seq)

    for ds_name in datasets:
        run_for_dataset(ds_name, cfg, model, device)

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
