"""
Utility functions for the RAG-LLM retrieval pipeline.

Contains I/O helpers, data loaders, pipeline state management,
and corpus append/merge routines used across the notebook cells.
"""

import os
import csv
import json
import pickle
import hashlib
import yaml

CONFIG_PATH = "config.yaml"


# ═══════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════

def load_config():
    """Load configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration file not found at: {os.path.abspath(CONFIG_PATH)}"
        )
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════
# File / Directory helpers
# ═══════════════════════════════════════════════════

def ensure_dir(path):
    """Create directory (and parents) if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def count_lines(filepath):
    """Count lines in a file efficiently using binary mode."""
    count = 0
    with open(filepath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ═══════════════════════════════════════════════════
# Serialization helpers
# ═══════════════════════════════════════════════════

def save_pickle(data, filepath):
    """Serialize *data* to a pickle file, creating parent dirs as needed."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """Deserialize data from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════

def load_queries(filepath):
    """Load queries from a JSONL file → {query_id: text}."""
    queries = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    return queries


def load_qrels(filepath):
    """Load relevance judgments from a TSV file → nested dict {qid: {docid: score}}."""
    qrels = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][row["corpus-id"]] = int(row["score"])
    return qrels


# ═══════════════════════════════════════════════════
# Corpus / Queries / Qrels merge helpers
# ═══════════════════════════════════════════════════

def initialize_output_files(corpus_path, queries_path, qrels_path):
    """Initialize empty output files and write the qrels TSV header."""
    open(corpus_path, "w", encoding="utf-8").close()
    open(queries_path, "w", encoding="utf-8").close()
    ensure_dir(os.path.dirname(qrels_path))
    with open(qrels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])


def append_corpus_to_jsonl(corpus_dict, filepath, dataset_prefix):
    """Append corpus documents to a JSONL file, prefixing IDs."""
    if not corpus_dict:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        for doc_id, doc in corpus_dict.items():
            entry = {
                "_id": f"{dataset_prefix}_{doc_id}",
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }
            json.dump(entry, f)
            f.write("\n")


def append_queries_to_jsonl(queries_dict, filepath, dataset_prefix):
    """Append queries to a JSONL file, prefixing IDs."""
    if not queries_dict:
        return
    with open(filepath, "a", encoding="utf-8") as f:
        for q_id, q_text in queries_dict.items():
            entry = {"_id": f"{dataset_prefix}_{q_id}", "text": q_text, "metadata": {}}
            json.dump(entry, f)
            f.write("\n")


def append_qrels_to_tsv(qrels_dict, filepath, dataset_prefix):
    """Append relevance judgments to a TSV file, prefixing IDs."""
    if not qrels_dict:
        return
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for q_id, doc_map in qrels_dict.items():
            new_q_id = f"{dataset_prefix}_{q_id}"
            for doc_id, score in doc_map.items():
                writer.writerow([new_q_id, f"{dataset_prefix}_{doc_id}", int(score)])


# ═══════════════════════════════════════════════════
# Embedding batch generator
# ═══════════════════════════════════════════════════

def load_corpus_batch_generator(filepath, batch_size):
    """Yield batches of (doc_ids, texts) from a JSONL corpus file."""
    batch_ids, batch_texts = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
                full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                batch_ids.append(doc["_id"])
                batch_texts.append(full_text)
                if len(batch_texts) >= batch_size:
                    yield batch_ids, batch_texts
                    batch_ids, batch_texts = [], []
            except json.JSONDecodeError:
                continue
    if batch_texts:
        yield batch_ids, batch_texts


# ═══════════════════════════════════════════════════
# Pipeline State Management
# ═══════════════════════════════════════════════════

def get_datasets_hash(datasets):
    """Compute a deterministic hash of the sorted dataset list for change detection."""
    return hashlib.md5(json.dumps(sorted(datasets)).encode()).hexdigest()


def load_pipeline_state(state_path):
    """Load pipeline state from JSON, or return a default empty state."""
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"datasets_hash": None, "completed_steps": []}


def save_pipeline_state(state, state_path):
    """Persist pipeline state to a JSON file."""
    ensure_dir(os.path.dirname(state_path))
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def is_step_complete(state, step_name, required_files):
    """Return True if *step_name* was recorded as complete AND all output files exist."""
    if step_name not in state.get("completed_steps", []):
        return False
    return all(os.path.exists(f) for f in required_files)


def mark_step_complete(state, step_name, state_path):
    """Record *step_name* as completed and flush state to disk."""
    if step_name not in state.get("completed_steps", []):
        state.setdefault("completed_steps", []).append(step_name)
    save_pipeline_state(state, state_path)


def clear_pipeline_artifacts(paths, state_path):
    """Delete all generated artifacts and reset pipeline state for a full rebuild."""
    artifact_keys = [
        "tokenized_corpus", "freq_index", "bm25_index",
        "corpus_embeddings", "vector_doc_ids", "tokenized_queries",
        "bm25_results", "dense_query_vectors", "dense_query_ids",
        "dense_results", "results", "corpus", "queries", "qrels",
    ]
    removed = 0
    for key in artifact_keys:
        fpath = paths.get(key)
        if fpath and os.path.exists(fpath):
            os.remove(fpath)
            removed += 1
    save_pipeline_state({"datasets_hash": None, "completed_steps": []}, state_path)
    return removed
