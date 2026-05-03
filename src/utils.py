"""
utils.py
========

Shared, non-algorithmic helpers for the Query-Adaptive Hybrid Retrieval
pipeline.  All "plumbing" lives here — configuration, file I/O, BEIR
download/load, tokenisation, plotting, simple metrics — so that
`pipeline.py` can stay focused on the math/algorithmic flow.

Sections
--------
1.  Configuration
2.  File / directory helpers
3.  Serialisation (pickle, JSON, CSV)
4.  BEIR download / load
5.  Corpus / queries / qrels readers + writers
6.  Tokenisation
7.  Metrics  (NDCG@k, Recall@k)
8.  wRRF fusion
9.  BM25 / Dense retrieval
10. Stratified train / dev / test split
11. K-fold CV index helper
12. Plot helpers
13. Statistical helpers
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# ============================================================
# 1. Configuration
# ============================================================

CONFIG_PATH = "config.yaml"


def load_config(path: str = None) -> dict:
    """Read and return the YAML configuration dictionary."""
    cfg_path = path or CONFIG_PATH
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {os.path.abspath(cfg_path)}"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_path(cfg: dict, key: str, default_value: str) -> str:
    """Read a path key from cfg['paths'] with a fallback default."""
    paths_cfg = cfg.get("paths", {}) or {}
    return paths_cfg.get(key, default_value)


def model_short_name(full_name: str) -> str:
    """'BAAI/bge-m3' -> 'bge-m3'."""
    return full_name.split("/")[-1]


def bm25_signature(k1: float, b: float, use_stemming: bool) -> str:
    """Return a filesystem-safe signature for BM25 cache files."""
    def fmt(x: float) -> str:
        s = f"{float(x):.4f}".rstrip("0").rstrip(".")
        return s if "." in s else s + ".0"
    return f"bm25_k1_{fmt(k1)}_b_{fmt(b)}_stem_{1 if use_stemming else 0}"


def bm25_artifact_paths(
    ds_dir: str, k1: float, b: float, use_stemming: bool, top_k: int = None
) -> Dict[str, str]:
    """Return the standard artifact paths for one (BM25 config, dataset) pair."""
    sig = bm25_signature(k1, b, use_stemming)
    stem_flag = f"stem_{1 if use_stemming else 0}"
    if top_k is None:
        results_name = f"{sig}_results.pkl"
    else:
        results_name = f"{sig}_topk_{int(top_k)}_results.pkl"
    return {
        "tokenized_corpus_jsonl":  os.path.join(ds_dir, f"tokenized_corpus_{stem_flag}.jsonl"),
        "tokenized_queries_jsonl": os.path.join(ds_dir, f"tokenized_queries_{stem_flag}.jsonl"),
        "query_tokens_pkl":        os.path.join(ds_dir, f"query_tokens_{stem_flag}.pkl"),
        "word_freq_pkl":           os.path.join(ds_dir, f"word_freq_index_{stem_flag}.pkl"),
        "doc_freq_pkl":            os.path.join(ds_dir, f"doc_freq_index_{stem_flag}.pkl"),
        "bm25_pkl":                os.path.join(ds_dir, f"{sig}.pkl"),
        "bm25_docids_pkl":         os.path.join(ds_dir, f"{sig}_doc_ids.pkl"),
        "bm25_results_pkl":        os.path.join(ds_dir, results_name),
        "bm25_signature":          sig,
    }


# ============================================================
# 2. File / directory helpers
# ============================================================

def ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if missing."""
    if path:
        os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def is_nonempty_file(path: str) -> bool:
    return file_exists(path) and os.path.getsize(path) > 0


def count_lines(filepath: str) -> int:
    """Count the lines in *filepath* (binary mode for speed)."""
    n = 0
    with open(filepath, "rb") as f:
        for _ in f:
            n += 1
    return n


# ============================================================
# 3. Serialisation
# ============================================================

def save_pickle(data, filepath: str) -> None:
    """Pickle *data* to *filepath* (creates parent directories)."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(data, filepath: str) -> None:
    """Write *data* to *filepath* as JSON with parent-dir creation."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def save_csv_dicts(rows: Sequence[dict], fieldnames: Sequence[str], filepath: str) -> None:
    """Write a list of dicts as a CSV with explicit column order."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_csv_dicts(filepath: str) -> List[dict]:
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md5_of_obj(obj) -> str:
    """Stable MD5 of any JSON-serialisable object — used for cache keys."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


# ============================================================
# 4. BEIR download / load
# ============================================================

def download_beir_dataset(dataset_name: str, datasets_folder: str) -> Optional[str]:
    """Download and unzip a BEIR dataset if not already present."""
    from beir import util as beir_util  # local import for fast startup

    dataset_path = os.path.join(datasets_folder, dataset_name)
    if os.path.exists(dataset_path):
        return dataset_path

    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{dataset_name}.zip"
    )
    try:
        beir_util.download_and_unzip(url, datasets_folder)
        return dataset_path
    except Exception as exc:
        print(f"  [ERROR] Download failed for {dataset_name}: {exc}")
        return None


def load_beir_dataset(dataset_path: str):
    """Load a BEIR dataset, picking test -> dev -> train automatically."""
    from beir.datasets.data_loader import GenericDataLoader

    loader = GenericDataLoader(dataset_path)
    split = None
    for candidate in ["test", "dev", "train"]:
        if os.path.exists(os.path.join(dataset_path, "qrels", f"{candidate}.tsv")):
            split = candidate
            break
    if split is None:
        print(f"  [ERROR] No valid qrels split in {dataset_path}")
        return None, None, None, None

    try:
        corpus, queries, qrels = loader.load(split=split)
        return corpus, queries, qrels, split
    except Exception as exc:
        print(f"  [ERROR] Failed to load {dataset_path}: {exc}")
        return None, None, None, None


# ============================================================
# 5. Corpus / queries / qrels readers + writers
# ============================================================

def write_corpus_jsonl(corpus_dict: dict, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for doc_id, doc in corpus_dict.items():
            entry = {
                "_id": str(doc_id),
                "title": doc.get("title", ""),
                "text":  doc.get("text", ""),
            }
            json.dump(entry, f)
            f.write("\n")


def write_queries_jsonl(queries_dict: dict, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        for q_id, q_text in queries_dict.items():
            json.dump({"_id": str(q_id), "text": q_text}, f)
            f.write("\n")


def write_qrels_tsv(qrels_dict: dict, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for q_id, doc_map in qrels_dict.items():
            for doc_id, score in doc_map.items():
                writer.writerow([str(q_id), str(doc_id), int(score)])


def load_queries(filepath: str) -> Dict[str, str]:
    queries = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    return queries


def load_qrels(filepath: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            qrels.setdefault(qid, {})[row["corpus-id"]] = int(row["score"])
    return qrels


def load_corpus_batch_generator(filepath: str, batch_size: int):
    """Yield (doc_ids, texts) batches from a JSONL corpus file."""
    batch_ids: List[str] = []
    batch_texts: List[str] = []
    malformed = 0
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
                malformed += 1
                continue
    if batch_texts:
        yield batch_ids, batch_texts
    if malformed:
        print(f"  [WARN] Skipped {malformed} malformed JSON lines in {filepath}")


def load_full_corpus(corpus_jsonl: str, batch_size: int = 1024) -> Dict[str, str]:
    """Return {doc_id: 'title text'} for the whole corpus."""
    out: Dict[str, str] = {}
    for ids, texts in load_corpus_batch_generator(corpus_jsonl, batch_size):
        for d, t in zip(ids, texts):
            out[d] = t
    return out


# ============================================================
# 6. Tokenisation
# ============================================================

def stem_and_tokenize(text: str, stemmer=None) -> List[str]:
    """Lowercase + whitespace split, optionally stemmed."""
    tokens = text.lower().split()
    if stemmer is None:
        return tokens
    return [stemmer.stem(w) for w in tokens]


# Module-level stemmer for the multiprocessing worker.
_worker_stemmer = None


def init_stem_worker(stemmer_lang: str, use_stemming: bool = True) -> None:
    """ProcessPoolExecutor initialiser — one stemmer per worker."""
    from nltk.stem.snowball import SnowballStemmer

    global _worker_stemmer
    _worker_stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None


def stem_batch_worker(args) -> List[str]:
    """Multiprocessing worker: stem-tokenise a batch of documents."""
    batch_ids, batch_texts = args
    out = []
    for doc_id, text in zip(batch_ids, batch_texts):
        if _worker_stemmer is None:
            tokens = text.lower().split()
        else:
            tokens = [_worker_stemmer.stem(w) for w in text.lower().split()]
        out.append(json.dumps({"_id": doc_id, "tokens": tokens}))
    return out


def ensure_english_stopwords() -> set:
    """Return an English stopword set, downloading once on demand."""
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


# ============================================================
# 7. Metrics
# ============================================================

def query_ndcg_at_k(ranked_pairs: Sequence[Tuple[str, float]],
                    rels: Dict[str, int], k: int) -> float:
    """NDCG@k using exponential gain  (2^rel - 1) / log2(rank+1)."""
    if not rels:
        return 0.0
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(ranked_pairs[:k], start=1):
        gain = (2.0 ** rels.get(doc_id, 0)) - 1.0
        dcg += gain / math.log2(rank + 1)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(((2.0 ** r) - 1.0) / math.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def query_recall_at_k(retrieved_ids: Iterable[str],
                      rels: Dict[str, int], k: int) -> Optional[float]:
    """Recall@k.  Returns None when no relevant document exists for the query."""
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return None
    top = list(retrieved_ids)[:k]
    return len(relevant & set(top)) / len(relevant)


def query_mrr_at_k(ranked_pairs: Sequence[Tuple[str, float]],
                   rels: Dict[str, int], k: int) -> Optional[float]:
    """Reciprocal Rank at k for a single query.

    Returns 1 / rank_of_first_relevant_doc (within top-k), or 0.0 if no
    relevant doc appears in the top-k.  Returns None when no relevant
    document exists for the query (matches `query_recall_at_k`)."""
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return None
    for rank, (doc_id, _) in enumerate(ranked_pairs[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ============================================================
# 8. wRRF fusion
# ============================================================

def wrrf_fuse(alpha: float,
              bm25_pairs: Sequence[Tuple[str, float]],
              dense_pairs: Sequence[Tuple[str, float]],
              rrf_k: int) -> List[Tuple[str, float]]:
    """
    Weighted Reciprocal Rank Fusion of two ranked lists.

    score(d) = alpha * 1/(rrf_k + rank_bm25(d))
             + (1 - alpha) * 1/(rrf_k + rank_dense(d))

    Documents missing from one ranking get an "out-of-list" rank equal to
    (list_length + 1) so they contribute a small but non-zero score.
    Returns the merged list sorted by descending score.
    """
    alpha = float(alpha)
    bm_rank = {d: r for r, (d, _) in enumerate(bm25_pairs, 1)}
    de_rank = {d: r for r, (d, _) in enumerate(dense_pairs, 1)}
    bm_miss = len(bm25_pairs) + 1
    de_miss = len(dense_pairs) + 1
    fused = {
        d: alpha / (rrf_k + bm_rank.get(d, bm_miss))
           + (1.0 - alpha) / (rrf_k + de_rank.get(d, de_miss))
        for d in set(bm_rank) | set(de_rank)
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def wrrf_top_k(alpha: float,
               bm25_pairs: Sequence[Tuple[str, float]],
               dense_pairs: Sequence[Tuple[str, float]],
               rrf_k: int, k: int) -> List[Tuple[str, float]]:
    """Convenience wrapper around `wrrf_fuse` that truncates to k."""
    return wrrf_fuse(alpha, bm25_pairs, dense_pairs, rrf_k)[:k]


# ============================================================
# 9. BM25 / Dense retrieval
# ============================================================

def run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k, use_stemming,
                       desc: str = "BM25 retrieval"):
    """BM25 retrieval over all queries; returns {qid: [(doc_id, score), ...]}."""
    from nltk.stem.snowball import SnowballStemmer
    from tqdm import tqdm

    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
    out: Dict[str, List[Tuple[str, float]]] = {}
    for qid, text in tqdm(queries.items(), desc=desc, dynamic_ncols=True):
        tokens = stem_and_tokenize(text, stemmer)
        scores = bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k <= 0:
            out[qid] = []
            continue
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        out[qid] = [(doc_ids[i], float(scores[i])) for i in top_idx]
    return out


def run_dense_retrieval(q_vecs, q_ids, c_vecs, c_ids, top_k, cfg,
                        desc: str = "Dense retrieval"):
    """Dense retrieval via sentence-transformers' semantic_search."""
    from sentence_transformers import util as st_util
    from tqdm import tqdm

    dense_cfg = cfg.get("dense_search", {}) or {}
    q_chunk = int(dense_cfg.get("query_chunk_size", 100))
    c_chunk = int(dense_cfg.get("corpus_chunk_size", 50000))
    device = c_vecs.device
    out: Dict[str, List[Tuple[str, float]]] = {}
    for start in tqdm(range(0, len(q_ids), q_chunk), desc=desc, dynamic_ncols=True):
        end = min(start + q_chunk, len(q_ids))
        batch = q_vecs[start:end].to(device)
        hits = st_util.semantic_search(batch, c_vecs, top_k=top_k, corpus_chunk_size=c_chunk)
        for i, hit_list in enumerate(hits):
            qid = q_ids[start + i]
            out[qid] = [(c_ids[h["corpus_id"]], float(h["score"])) for h in hit_list]
    return out


# ============================================================
# 10. Stratified train / dev / test split
# ============================================================

def dataset_seed_offset(name: str) -> int:
    """Deterministic per-dataset seed offset (for reproducible splits)."""
    return int(hashlib.md5(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16) % (2 ** 31)


def stratified_split(qids_per_ds: Dict[str, Sequence[str]],
                     test_fraction: float, dev_fraction: float,
                     base_seed: int) -> Dict[str, dict]:
    """
    Return  {ds_name: {"train": [qid], "dev": [qid], "test": [qid]}}.

    A separate, deterministic permutation is drawn per dataset so the
    fractions test_fraction / dev_fraction are honoured *within* each
    dataset (uniform stratification).  Identical seeds across runs.
    """
    out = {}
    for ds_name, qids in qids_per_ds.items():
        rng  = np.random.RandomState(base_seed + dataset_seed_offset(ds_name))
        perm = rng.permutation(len(qids))
        n    = len(qids)
        n_test = max(1, int(round(test_fraction * n)))
        n_dev  = max(1, int(round(dev_fraction  * n)))
        n_dev  = min(n_dev, n - n_test - 1)  # ensure at least 1 train query
        train_idx = perm[: n - n_test - n_dev]
        dev_idx   = perm[n - n_test - n_dev: n - n_test]
        test_idx  = perm[n - n_test:]
        out[ds_name] = {
            "train": [qids[i] for i in train_idx],
            "dev":   [qids[i] for i in dev_idx],
            "test":  [qids[i] for i in test_idx],
        }
    return out


# ============================================================
# 11. K-fold CV index helper
# ============================================================

def kfold_indices(n: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Standard k-fold CV (shuffled) — returns list of (train_idx, val_idx).

    Uses np.array_split so the val sizes are as balanced as possible.
    """
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    out = []
    for fi, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        out.append((tr_idx, val_idx))
    return out


# ============================================================
# 12. Plot helpers
# ============================================================

def grouped_bar_chart(rows: Sequence[dict],
                      methods: Sequence[str],
                      labels: Sequence[str],
                      colors: Sequence[str],
                      ylabel: str,
                      title: str,
                      out_path: str,
                      group_key: str = "group",
                      figsize: Tuple[int, int] = (16, 6),
                      annotate: bool = True,
                      yzero: bool = True) -> None:
    """Generic grouped bar chart used by every comparison plot in pipeline.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    groups  = [r[group_key] for r in rows]
    x       = np.arange(len(groups))
    n_m     = len(methods)
    width   = 0.85 / max(n_m, 1)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    by_method = {m: [r[m] for r in rows] for m in methods}

    fig, ax = plt.subplots(figsize=figsize)
    for method, label, color, off in zip(methods, labels, colors, offsets):
        bars = ax.bar(x + off, by_method[method], width,
                      label=label, color=color, alpha=0.85, edgecolor="white")
        if annotate:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (0.003 if h >= 0 else -0.010),
                    f"{h:+.3f}" if not yzero else f"{h:.3f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    if yzero:
        all_vals = [v for vals in by_method.values() for v in vals]
        ax.set_ylim(0, min(1.0, max(all_vals) + 0.12) if all_vals else 1.0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    else:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alpha_box_plot(alphas_per_dataset: Dict[str, Sequence[float]],
                   title: str, out_path: str) -> None:
    """Box plot of predicted alpha values per dataset and overall macro."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys   = list(alphas_per_dataset.keys()) + ["MACRO"]
    macro  = np.concatenate([np.asarray(a, dtype=np.float32)
                             for a in alphas_per_dataset.values()]) \
             if alphas_per_dataset else np.array([])
    data   = [np.asarray(alphas_per_dataset[k], dtype=np.float32) for k in keys[:-1]] + [macro]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=keys, showmeans=True, meanline=True,
               patch_artist=True,
               boxprops=dict(facecolor="#82C6E2", alpha=0.7),
               medianprops=dict(color="darkblue"),
               meanprops=dict(color="red", linestyle="--"))
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(r"Predicted $\alpha$", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def alpha_sorted_plot(oracle_alphas: Sequence[float],
                      predicted_alphas: Sequence[float],
                      title: str, out_path: str) -> None:
    """
    Scatter plot of *oracle* alphas (sorted ascending) versus the
    *predicted* alphas for the same queries, drawn on top in another colour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oracle    = np.asarray(oracle_alphas, dtype=np.float32)
    predicted = np.asarray(predicted_alphas, dtype=np.float32)
    order     = np.argsort(oracle, kind="stable")
    ranks     = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(ranks, oracle[order],    s=8, alpha=0.85,
               color="#1f77b4", label="Oracle α  (sorted)")
    ax.scatter(ranks, predicted[order], s=8, alpha=0.55,
               color="#d62728", label="Predicted α")
    ax.set_xlabel("Query rank (after sorting by oracle α)", fontsize=10)
    ax.set_ylabel(r"$\alpha$", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 13. Statistical helpers
# ============================================================

def paired_t_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict:
    """
    Two-sided paired t-test on per-query scores  (a - b).

    Returns a dict with mean difference, t-statistic, p-value, the number of
    queries used, and Cohen's d for paired samples (mean_diff / sd_of_diff).
    An empty / all-equal sample yields p = 1.0 and d = 0.0.
    """
    from scipy.stats import ttest_rel

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) == 0 or len(a) != len(b):
        return {"n": int(min(len(a), len(b))), "mean_diff": 0.0,
                "t": 0.0, "p": 1.0, "d": 0.0}

    diff = a - b
    if np.allclose(diff, 0.0):
        return {"n": int(len(a)), "mean_diff": 0.0, "t": 0.0, "p": 1.0, "d": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ttest_rel(a, b)

    # Cohen's d for paired samples.  Guard against near-zero SD (would
    # otherwise produce a meaningless huge number for degenerate inputs).
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) >= 2 else 0.0
    cohens_d = float(np.mean(diff) / sd_diff) if sd_diff > 1e-12 else 0.0

    return {
        "n":         int(len(a)),
        "mean_diff": float(np.mean(diff)),
        "t":         float(result.statistic),
        "p":         float(result.pvalue),
        "d":         cohens_d,
    }


def bootstrap_ci_mean(scores: Sequence[float],
                      n_resamples: int = 1000,
                      ci: float = 0.95,
                      seed: int = 42) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of *scores*.

    Vectorised resampling for speed.  Returns (mean, lower, upper).  An
    empty array yields all zeros.  Identical seed → identical CI across
    runs (reproducibility).
    """
    a = np.asarray(scores, dtype=np.float64)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    if len(a) == 1:
        v = float(a[0])
        return v, v, v
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(int(n_resamples), len(a)))
    boots = a[idx].mean(axis=1)
    alpha = (1.0 - float(ci)) / 2.0
    return (float(a.mean()),
            float(np.percentile(boots, 100.0 * alpha)),
            float(np.percentile(boots, 100.0 * (1.0 - alpha))))


def holm_correction(p_values: Sequence[float],
                    alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down family-wise correction.

    Given a vector of raw p-values, returns:
      - rejected: boolean array — True for hypotheses rejected at level *alpha*.
      - p_adj:    Holm-adjusted p-values (monotone in sorted order, clipped to 1.0).

    Standard procedure: order p-values ascending, multiply rank-i p-value by
    (n - i + 1), then enforce monotonicity from smallest to largest.  A
    hypothesis is rejected iff its adjusted p-value is <= alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    multipliers = np.arange(n, 0, -1, dtype=np.float64)        # n, n-1, ..., 1
    p_adj_sorted = np.maximum.accumulate(p_sorted * multipliers)
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)

    p_adj = np.empty(n, dtype=np.float64)
    p_adj[order] = p_adj_sorted
    rejected = p_adj <= float(alpha)
    return rejected, p_adj
