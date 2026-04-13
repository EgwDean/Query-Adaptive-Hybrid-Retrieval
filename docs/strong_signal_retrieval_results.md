# Strong-Signal Retrieval Comparison Results

## What was done

For each of the five BEIR datasets, an XGBoost router was trained on the
85% train+dev split using the per-dataset best hyperparameters from the
strong-signal grid search. The 15% held-out test set was then evaluated
against four retrieval methods:

- **BM25-only** — sparse BM25 ranking, no fusion
- **Dense-only** — BAAI/bge-m3 cosine similarity, no fusion
- **Static RRF** — wRRF with α = 0.5 for every query (no routing)
- **wRRF (XGBoost+Emb)** — wRRF with per-query α predicted from the
  query embedding vector (~1024 dims, BAAI/bge-m3)

The 85/15 split uses the same seed formula as all previous scripts, so the
15% test queries are identical across all experiments — a prerequisite for
a fair comparison between the weak-signal and strong-signal routers.

Script: `src/xgboost_retrieval.py`
Config: `strong_signal_xgboost_per_dataset` in `config.yaml`
Outputs: `data/results/strong_signal_retrieval_comparison.csv`,
         `data/results/strong_signal_retrieval_comparison.png`

---

## Benchmark results

All NDCG@10 values on the 15% per-dataset held-out test set.

| Dataset  | BM25   | Dense  | Static RRF | wRRF (XGBoost+Emb) |
|----------|--------|--------|------------|---------------------|
| scifact  | 0.6008 | 0.6703 | 0.6488     | **0.6559**          |
| nfcorpus | 0.2912 | 0.3800 | **0.3452** | 0.3408              |
| arguana  | 0.2525 | 0.3804 | 0.3283     | **0.3665**          |
| fiqa     | 0.1602 | 0.4830 | 0.3859     | **0.4733**          |
| scidocs  | 0.1288 | 0.1540 | 0.1598     | **0.1622**          |
| **MACRO**| 0.2867 | **0.4135** | 0.3736 | **0.3997**          |

---

## Final assessment

### Summary

The embedding-based wRRF router achieves a macro NDCG@10 of **0.3997**:

- **+0.0261 over Static RRF** (0.3736) — consistent gain from adaptive routing
- **+0.1130 over BM25** (0.2867) — fusion with dense is always beneficial
- **−0.0138 below Dense-only** (0.4135) — the router does not surpass the
  best individual retriever

The router improves over static equal-weight fusion on 4 of 5 datasets. The
only exception is nfcorpus, where it falls 0.004 short of Static RRF.

---

### Per-dataset analysis

**scifact** — Dense is the best single retriever (0.6703). The embedding router
(0.6559) outperforms Static RRF (0.6488) by +0.0071. It correctly up-weights
dense for scientific claim verification, where semantic similarity to the corpus
of scientific statements is the primary matching signal. BM25 is competitive
here (0.6008) compared to other datasets because scientific terminology is
precise and unambiguous.

**nfcorpus** — Dense is again the best retriever (0.3800). The router (0.3408)
falls marginally short of Static RRF (0.3452) by −0.0044. nfcorpus is the
hardest dataset for the strong-signal router: with 38.2 relevant documents per
query on average, the optimal alpha is rarely near 0 or 1 — the routing
problem is not binary. The embedding router cannot reliably detect which
queries benefit from a slight BM25 contribution, so it occasionally
over-weights dense and loses the marginal gain from sparse signals. This is
consistent with the weak-signal router's behaviour on this dataset.

**arguana** — The router's strongest per-dataset result relative to static
fusion: wRRF (0.3665) beats Static RRF (0.3283) by **+0.0382**. Arguana is
a counter-argument retrieval task where queries are argumentative passages and
the target is the counter-argument. Dense retrieval (0.3804) handles the
semantic relationship well; BM25 (0.2525) is very weak because
counter-arguments deliberately avoid repeating their opponent's key terms.
The embedding router reliably recognises this query type and assigns near-zero
alpha (high dense weight) consistently across the test set. The embedding
vector directly encodes the argumentative style, which is a strong and reliable
routing signal.

**fiqa** — The largest absolute gain: wRRF (0.4733) vs Static RRF (0.3859),
**+0.0874**. fiqa is a financial QA dataset where BM25 fails catastrophically
(0.1602) due to the informal, conversational style of financial forum queries.
The router learns to almost entirely discard BM25 for this dataset, acting
effectively as an adaptive dense retriever — wRRF (0.4733) is within 0.010
of dense-only (0.4830). The full-dimensional embedding vector is an
especially reliable routing signal here: the writing style and domain of
financial queries are encoded directly in the embedding space.

**scidocs** — The only dataset where neither router nor dense-only beats
Static RRF cleanly. scidocs is a citation recommendation task with inherently
ambiguous relevance (NDCG@10 ≈ 0.16 across all methods). The embedding
router (0.1622) provides a marginal +0.0024 gain over Static RRF (0.1598),
and both exceed dense-only (0.1540). This means BM25 actually contributes
positively when fused with dense even at equal weight — the sparse citation
vocabulary partially captures document identity. The router successfully
recognises some of this and maintains a slight BM25 contribution, marginally
outperforming the static baseline, but the gains are within noise for this
test set size.

---

## Comparison with the weak-signal router

The weak-signal router (15 hand-crafted features) and the strong-signal router
(~1024-dim query embeddings) produce nearly identical macro performance on
the held-out test sets:

| Dataset  | wRRF (weak signal) | wRRF (XGBoost+Emb) | Delta (strong − weak) |
|----------|--------------------|--------------------|----------------------|
| scifact  | 0.6628             | 0.6559             | −0.0069              |
| nfcorpus | 0.3538             | 0.3408             | −0.0130              |
| arguana  | 0.3629             | 0.3665             | +0.0036              |
| fiqa     | 0.4679             | 0.4733             | +0.0054              |
| scidocs  | 0.1567             | 0.1622             | +0.0055              |
| **MACRO**| **0.4008**         | **0.3997**         | **−0.0011**          |

The macro gap is 0.001 — well within the variance of these small test sets.
The pattern mirrors what was observed at every earlier stage (model selection
and per-dataset parameter search): the two representations are **complementary**
rather than one being uniformly superior.

- **Weak signal outperforms on scifact and nfcorpus** — datasets where
  retriever-derived statistics (BM25 entropy, score confidence, rank
  correlation) are strong routing cues. The hand-crafted features directly
  measure the relationship between the query and both retrievers, which is
  exactly the information needed for the routing decision.

- **Strong signal outperforms on arguana, fiqa, and scidocs** — datasets
  where the routing decision correlates with query type or writing style
  rather than with per-query retriever behaviour. An embedding vector
  encodes the semantic and stylistic properties of the query directly, giving
  the router a head start on queries whose domain or register strongly
  predicts which retriever will win.

This complementarity suggests that a combined router — one that feeds both
representations as input — could outperform either individually, particularly
on the datasets where one representation is currently weakest.

---

## Limitations

- Test sets are small (48–648 queries per dataset); per-dataset scores have
  high variance and should be read as indicative, not definitive.
- The routing signal is derived from the query alone — the router has no
  visibility into the actual retrieved documents at inference time, which
  limits its ability to correct for retriever failures on individual queries.
- scidocs remains unsolved by routing; a more powerful representation or a
  corpus-aware router would be needed.
- Generalisation to datasets not seen during hyperparameter selection is not
  demonstrated.
