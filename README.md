# Query-Adaptive Hybrid Retrieval

Diploma Thesis — Konstantinos Anastasopoulos, CEID

---

## Problem

Standard retrieval systems apply a fixed strategy to every query. For some queries
BM25 (sparse, lexical matching) is the better retriever; for others a dense
embedding model is. A query-adaptive system learns to weight the two methods
per query, improving retrieval quality across diverse query types.

---

## Algorithm

For each query $q$:

1. **Sparse retrieval** — BM25 produces a ranked list of documents.
2. **Dense retrieval** — a bi-encoder (BAAI/bge-m3) produces a ranked list via cosine similarity.
3. **Router** — a classifier predicts a weight $\hat{\alpha}(q) \in [0, 1]$ from
   query features. Values near 1 favour sparse; values near 0 favour dense.
4. **Weighted RRF fusion** — the two ranked lists are combined:

$$\text{score}(q, d) = \hat{\alpha}(q) \cdot \frac{1}{k + r_\text{sparse}(d)} + (1 - \hat{\alpha}(q)) \cdot \frac{1}{k + r_\text{dense}(d)}$$

with $k = 60$ (RRF damping constant). Setting $\hat{\alpha} = 0.5$ recovers
static RRF, which serves as a baseline.

The router is trained with **soft labels**: the target for each query is derived
from the relative NDCG@10 of BM25 vs dense on that query (see `docs/routing_features.md`).

---

## Baselines compared

For each dataset the benchmark reports NDCG@10 for:

- **Sparse only** — BM25
- **Dense only** — bi-encoder cosine similarity
- **Static RRF** — equal-weight reciprocal rank fusion ($\hat{\alpha} = 0.5$)
- **Weighted RRF** — per-query $\hat{\alpha}$ predicted by the trained router

---

## Datasets

Five BEIR datasets: `scifact`, `nfcorpus`, `arguana`, `fiqa`, `scidocs`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

---

## Run order

```bash
# 1. Download BEIR datasets
python src/download.py

# 2. Build preprocessing caches (BM25 index, embeddings, frequency indexes)
python src/preprocess.py
```

Further scripts are added incrementally as the pipeline is developed.

---

## Documentation

- `docs/routing_features.md` — full feature definitions with formulas
