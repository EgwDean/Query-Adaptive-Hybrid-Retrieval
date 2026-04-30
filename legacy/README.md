# Legacy Pipeline

This folder contains the original implementation of the Query-Adaptive Hybrid Retrieval pipeline.

## Key Difference: Alpha Label Formula

The central algorithmic difference between this legacy code and the current pipeline in `src/` is **how the fusion weight α is labelled for each training query**.

### Legacy approach — soft NDCG-difference label

The legacy pipeline computes α as a closed-form function of the per-query NDCG@10 scores of BM25 and Dense retrieval:

```
label = 0.5 * ((NDCG_sparse - NDCG_dense) / (max(NDCG_sparse, NDCG_dense) + ε) + 1)
```

This maps to the interval [0, 1]:
- **0.5** — both retrievers perform equally
- **→ 1.0** — BM25 is strictly better (prefer sparse retrieval)
- **→ 0.0** — Dense is strictly better (prefer dense retrieval)

It is fast and requires no search, but it is only a heuristic proxy. The label does not directly maximise retrieval quality under the chosen fusion formula; it merely reflects which retriever scored higher in isolation.

### New approach — oracle alpha label

The current pipeline (`src/pipeline.py`) replaces the formula above with **oracle alpha labels**. For each query individually, a brute-force grid search is run over 101 candidate values:

```
α ∈ {0.00, 0.01, 0.02, …, 1.00}
```

For each candidate α the weighted Reciprocal Rank Fusion (wRRF) ranking is computed:

```
score(d) = α / (60 + rank_bm25(d)) + (1 − α) / (60 + rank_dense(d))
```

and NDCG@10 is evaluated against the full qrels. The label assigned to the query is:

```
α* = argmax_α  NDCG@10( wRRF(α | q) )
```

These oracle labels represent the **theoretical upper bound** on what any router using the wRRF formula can achieve — if a model predicted α* perfectly for every query, the resulting retrieval would be the best possible under wRRF. This makes the router learning problem more precisely defined and the evaluation ceiling meaningful.
