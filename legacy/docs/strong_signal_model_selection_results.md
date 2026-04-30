# Strong-Signal Model Selection Results

## What was done

A second model selection grid search was run using a fundamentally different
input representation: instead of the 15 hand-crafted weak-signal routing
features, the raw query embedding vectors from the dense encoder (BAAI/bge-m3,
~1024 dimensions per query) were used directly as input to the router.

The motivation is to ask whether a model that has access to the full semantic
content of a query — not just aggregate statistics derived from it — can learn
a better routing function.

Everything else was held constant compared to the weak-signal grid search:
- Same 5 BEIR datasets (scifact, nfcorpus, arguana, fiqa, scidocs)
- Same 10-fold Monte Carlo CV with 80/20 splits and identical fold seeds
- Same soft labels (relative NDCG@10 of BM25 vs dense per query)
- Same wRRF NDCG@10 evaluation metric
- Same top-100 output format

Model families tested: Ridge Regression, ElasticNet, SVR (RBF kernel), KNN,
MLP, XGBoost, Random Forest, Extra Trees (~470 total combinations).

Script: `src/strong_signal_model_grid_search.py`
Output: `data/results/strong_signal_grid_search_top100.csv`
Config: `strong_signal_model_grid_search` in `config.yaml`

---

## Result: XGBoost dominates again

**All 100 entries in the top-100 CSV are XGBoost configurations.**

The same model family that dominated the weak-signal grid search also dominates
when the input is a 1024-dimensional embedding vector. Linear models (Ridge,
ElasticNet), SVR, KNN, MLP, Random Forest, and Extra Trees all ranked below
the XGBoost threshold for inclusion in the top 100.

---

## Top 10 configurations

| Rank | colsample_bytree | learning_rate | max_depth | min_child_weight | n_estimators | subsample | Macro NDCG@10 |
|------|-----------------|---------------|-----------|-----------------|--------------|-----------|---------------|
| 1  | 0.8 | 0.3 | 8 | 1 | 300 | 0.8 | 0.3851 |
| 2  | 0.3 | 0.1 | 8 | 1 | 300 | 0.8 | 0.3849 |
| 3  | 0.3 | 0.3 | 8 | 1 | 300 | 0.8 | 0.3846 |
| 4  | 0.8 | 0.3 | 6 | 1 | 300 | 0.8 | 0.3843 |
| 5  | 0.3 | 0.1 | 6 | 1 | 300 | 0.8 | 0.3841 |
| 6  | 0.8 | 0.1 | 8 | 1 | 300 | 0.8 | 0.3840 |
| 7  | 0.8 | 0.3 | 4 | 1 | 300 | 0.8 | 0.3838 |
| 8  | 0.5 | 0.3 | 8 | 1 | 300 | 0.8 | 0.3838 |
| 9  | 0.1 | 0.3 | 6 | 1 | 300 | 0.8 | 0.3837 |
| 10 | 0.1 | 0.3 | 8 | 1 | 300 | 0.8 | 0.3837 |

Per-dataset NDCG@10 for rank 1:

| scifact | nfcorpus | arguana | fiqa   | scidocs |
|---------|----------|---------|--------|---------|
| 0.6312  | 0.3265   | 0.3791  | 0.4142 | 0.1745  |

---

## Comparison with weak-signal model selection

The key comparison is rank-1 strong signal vs rank-1 weak signal:

| Representation | Macro NDCG@10 | scifact | nfcorpus | arguana | fiqa   | scidocs |
|---------------|---------------|---------|----------|---------|--------|---------|
| Weak signal (15 features) | **0.3858** | **0.6392** | **0.3357** | 0.3746 | 0.4073 | 0.1722 |
| Strong signal (1024-dim embeddings) | 0.3851 | 0.6312 | 0.3265 | **0.3791** | **0.4142** | **0.1745** |
| Delta (strong − weak) | −0.0007 | −0.0080 | −0.0092 | +0.0045 | +0.0069 | +0.0023 |

The two representations produce **essentially identical macro performance**
(gap of 0.0007), but with complementary per-dataset strengths:

- Weak signal is better on **scifact** and **nfcorpus** — datasets where
  lexical overlap and term-frequency signals (IDF, entropy, stopword ratio)
  are strong routing cues.
- Strong signal is better on **arguana**, **fiqa**, and **scidocs** — datasets
  where the routing decision correlates more with query semantics (query topic
  or phrasing style) than with surface statistics.

---

## Hyperparameter patterns

**`n_estimators = 300` in every top-10 entry** — same as the weak-signal case.
Larger ensembles consistently help regardless of input representation.

**`subsample = 0.8` and `min_child_weight = 1` in every top-10 entry** — also
identical to the weak-signal case. These appear to be fundamental properties
of the routing task, not of the feature space.

**`colsample_bytree` varies widely** (0.1, 0.3, 0.5, and 0.8 all appear in
the top 10). In the weak-signal grid, `colsample_bytree = 0.8` dominated
9 of the top 10 entries. The greater spread here is consistent with the
hypothesis that with 1024 input dimensions, random feature subsampling at
different rates all produce useful diversity — there is no single "correct"
column fraction.

**`max_depth = 4` appears** (rank 7), unlike the weak-signal top 10 where
max_depth was always 6 or 8. Shallower trees can be sufficient when the
relevant routing signal is spread across many embedding dimensions and
individual deep splits are less meaningful.

---

## Observations

**XGBoost works on high-dimensional embeddings.** The expectation that linear
models would dominate on embedding input — based on the general principle that
embedding spaces are often linearly separable — did not hold here. XGBoost's
ability to learn non-linear interactions between embedding dimensions provides
a stronger routing signal than any linear projection.

**The 15 weak-signal features are information-dense.** The fact that 15
hand-crafted numbers encode the same macro routing quality as 1024 embedding
dimensions is the central finding. The features deliberately capture
retriever-specific signals (entropy, confidence, overlap, rank correlation)
that are not directly accessible from the query text alone. An embedding
vector describes the query; the weak-signal features describe the relationship
between the query and the two retrievers — which is exactly what the router
needs to predict.

**The representations are complementary, not redundant.** Each performs better
on a different subset of datasets. This opens the possibility that a combined
representation (concatenated features + embeddings, or a two-stage router)
could outperform either individually.

**Macro spread is very narrow.** The top 100 entries span only 0.3851 to
roughly 0.367 — the same pattern of hyperparameter insensitivity seen in the
weak-signal case. The routing task itself has a performance ceiling that neither
representation has broken through.
