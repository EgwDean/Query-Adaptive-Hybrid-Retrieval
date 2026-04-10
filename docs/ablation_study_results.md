# Ablation Study Results

## What was done

Using the XGBoost configuration selected in the previous step, a feature ablation
study was run to measure the contribution of each individual feature and each
feature group to the wRRF router's performance.

Two ablation modes were tested:

1. **Leave-one-feature-out** — 16 configurations, each dropping exactly one of
   the 16 routing features.
2. **Leave-one-group-out** — 5 configurations, each dropping all features
   belonging to one of the five groups (A–E).

Each configuration used the same evaluation protocol as model selection:
10-fold Monte Carlo CV with 80/20 train/test splits across all 5 BEIR datasets,
with macro NDCG@10 as the metric. The XGBoost hyperparameters were fixed at the
rank-1 configuration from model selection throughout.

Script: `src/ablation_study.py`
Outputs: `data/results/ablation_study.csv`, `data/results/ablation_study.png`

---

## Result: two features are net noise

The full model (all 16 features) achieved macro NDCG@10 of **0.3858**.
Most ablation configurations score lower, but **two features improve performance
when removed**: `query_length` (+0.0003) and `average_idf` (+0.0003). All other
features hurt when dropped, confirming they carry genuine signal.

---

## Leave-one-feature-out

Sorted by delta vs full model (most negative = most important; positive = net noise).

| Feature | Macro NDCG@10 | Delta |
|---|---|---|
| `top_dense_score` | 0.3833 | −0.0025 |
| `sparse_entropy_topk` | 0.3837 | −0.0021 |
| `dense_entropy_topk` | 0.3841 | −0.0017 |
| `sparse_confidence` | 0.3842 | −0.0016 |
| `stopword_ratio` | 0.3843 | −0.0015 |
| `first_shared_doc_rank` | 0.3843 | −0.0015 |
| `rare_term_ratio` | 0.3847 | −0.0011 |
| `dense_confidence` | 0.3847 | −0.0011 |
| `spearman_topk` | 0.3847 | −0.0011 |
| `max_idf` | 0.3850 | −0.0008 |
| `has_question_word` | 0.3853 | −0.0005 |
| `overlap_at_k` | 0.3853 | −0.0005 |
| `cross_entropy` | 0.3854 | −0.0004 |
| `top_sparse_score` | 0.3855 | −0.0003 |
| `average_idf` | 0.3861 | **+0.0003** |
| `query_length` | 0.3861 | **+0.0003** |

The most impactful individual features are `top_dense_score` and
`sparse_entropy_topk`. The raw normalized top score from the dense retriever
and the entropy of the BM25 score distribution directly encode retriever
reliability per query. The least impactful useful feature is `top_sparse_score`.

`query_length` and `average_idf` are mildly harmful: including them adds noise
the model has to route around. This is consistent with other features already
capturing the relevant signal — `stopword_ratio`, `rare_term_ratio`, and `max_idf`
subsume what `query_length` and `average_idf` were trying to express.

---

## Leave-one-group-out

Sorted by delta vs full model (largest drop = most important group).

| Group | Macro NDCG@10 | Delta |
|---|---|---|
| E: Distribution Shape | 0.3823 | −0.0035 |
| C: Retriever Confidence | 0.3829 | −0.0029 |
| D: Retriever Agreement | 0.3829 | −0.0029 |
| A: Query Surface | 0.3834 | −0.0024 |
| B: Vocabulary Match | 0.3839 | −0.0019 |

Group E (Distribution Shape) is the single most important group. Removing both
entropy features causes the largest aggregate drop, confirming that the shape of
the retriever score distributions is the clearest signal for routing.

Groups C and D are nearly tied for second. Retriever Confidence (top scores,
score gaps) and Retriever Agreement (overlap, shared-rank correlation) together
directly encode *when one retriever is more reliable than the other* — which is
exactly what the router needs to predict.

Group B (Vocabulary Match) is the least important group. Two of its four
features (`average_idf`, `cross_entropy`) carry marginal or slightly harmful
signal, and the other two (`max_idf`, `rare_term_ratio`) are partially redundant
with Group E and D features.

---

## Full vs smaller model comparison

Following the leave-one-out results, a direct comparison was run between the
full 16-feature model and a 14-feature model with both `query_length` and
`average_idf` removed.

Script: `src/full_vs_smaller_model_ablation_study.py`
Outputs: `data/results/full_vs_smaller_ablation.csv`, `data/results/full_vs_smaller_ablation.png`

| Model | Features | Macro NDCG@10 | Delta |
|---|---|---|---|
| Full | 16 | 0.3858 | — |
| Reduced (no query_length, no average_idf) | 14 | 0.3844 | −0.0014 |

Removing both features together hurts more than removing either one alone. The
two features carry partially complementary signal: each individually is noise,
but together they provide a weak compound proxy that the model exploits.

**Decision**: remove only `query_length`, keeping `average_idf`. The
15-feature model (no `query_length`) scores **0.3861** — the highest of all
configurations tested — and retains the richer IDF-based vocabulary signal.

---

## Observations

- The full model is not the optimal feature set: `query_length` is net noise
  and its removal (+0.0003) represents the best single-feature configuration.
- The differences are small in absolute terms (max group drop 0.0035), which is
  expected: XGBoost is robust to mildly redundant features.
- Group-level drops are consistently larger than single-feature drops, confirming
  that within each group the features carry partially independent information.
- The 15-feature model (all features except `query_length`) is used for all
  subsequent steps (per-dataset hyperparameter search and final evaluation).
