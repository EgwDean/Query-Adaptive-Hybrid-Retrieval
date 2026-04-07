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

## Result: full model wins

The full model (all 16 features) achieved the highest macro NDCG@10 of **0.3851**.
Every ablation configuration scored lower, confirming that all features and all
groups carry at least some signal and that none are harmful noise.

---

## Leave-one-feature-out

Sorted by performance drop (largest drop = most important feature).

| Feature | Macro NDCG@10 | Drop |
|---|---|---|
| `sparse_entropy_topk` | 0.3825 | −0.0026 |
| `top_dense_score` | 0.3829 | −0.0022 |
| `first_shared_doc_rank` | 0.3835 | −0.0016 |
| `stopword_ratio` | 0.3836 | −0.0015 |
| `max_idf` | 0.3838 | −0.0013 |
| `dense_entropy_topk` | 0.3839 | −0.0012 |
| `spearman_topk` | 0.3839 | −0.0012 |
| `top_sparse_score` | 0.3842 | −0.0008 |
| `rare_term_ratio` | 0.3843 | −0.0008 |
| `dense_confidence` | 0.3844 | −0.0007 |
| `query_length` | 0.3846 | −0.0005 |
| `has_question_word` | 0.3846 | −0.0005 |
| `average_idf` | 0.3847 | −0.0004 |
| `sparse_confidence` | 0.3847 | −0.0004 |
| `overlap_at_k` | 0.3847 | −0.0004 |
| `cross_entropy` | 0.3849 | −0.0002 |

The most impactful individual features are `sparse_entropy_topk` and
`top_dense_score`. The distribution shape of the BM25 ranking (entropy) and the
raw normalized top score from the dense retriever both carry signal that the model
relies on. The least impactful is `cross_entropy`, which overlaps informationally
with IDF-based features.

---

## Leave-one-group-out

Sorted by performance drop (largest drop = most important group).

| Group | Macro NDCG@10 | Drop |
|---|---|---|
| E: Distribution Shape | 0.3814 | −0.0037 |
| D: Retriever Agreement | 0.3815 | −0.0036 |
| B: Vocabulary Match | 0.3826 | −0.0025 |
| C: Retriever Confidence | 0.3831 | −0.0020 |
| A: Query Surface | 0.3835 | −0.0016 |

Groups E and D are the most valuable, nearly tied. Removing either one causes the
largest drop. This makes sense: the distribution shape features (entropy of the
top-k score distributions) and the retriever agreement features (how much the two
retrievers agree on document ranking) directly encode *when* one retriever is more
reliable than the other, which is exactly the signal the router needs.

Group A (query surface features: length, stopword ratio, question word) causes the
smallest group-level drop, reflecting that surface-level query properties are
weaker proxies for retriever preference than the retrieval outputs themselves.

---

## Observations

- No feature hurts — every removal reduces performance, so the feature set is
  clean with no redundant or adversarial signals.
- The differences are small in absolute terms (max drop 0.0037), which is
  expected: XGBoost is robust to mildly redundant features, and the 16 features
  encode complementary but correlated signals.
- Group-level drops are consistently larger than single-feature drops, confirming
  that within each group the features carry partially independent information.
- The full model is therefore the right choice for the next step (hyperparameter
  fine-tuning).
