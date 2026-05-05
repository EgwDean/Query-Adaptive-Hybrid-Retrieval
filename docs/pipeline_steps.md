# Pipeline — Complete Step-by-Step Reference

This document describes every step performed by `src/pipeline.py`.  The pipeline
is a single end-to-end script that produces every artefact (data, models,
metrics, plots) required by the thesis on top of **oracle** α labels found
by per-query brute-force search.

The pipeline is **idempotent**: every step caches its outputs to disk and
skips itself on a subsequent run if the outputs already exist, so the script
can be killed and restarted at any point without losing progress.  The skip
gate uses `is_nonempty_file()` (not just `os.path.exists()`) so zero-byte
crash files are never treated as valid cache.

---

## Dataset and split overview

The merged dataset used throughout the pipeline is the concatenation of the
six BEIR datasets **scifact**, **nfcorpus**, **arguana**, **fiqa**,
**scidocs**, and **trec-covid**, capped at **300 queries per dataset** where
available.  The 300 queries are drawn by stratified random sampling
(seed = `sampling.random_seed`, default 42) so that each unique relevance
label is proportionally represented.

> **Note on trec-covid:** The BEIR trec-covid split contains only ~50
> queries — fewer than the 300-query cap.  All available queries are used,
> so this dataset contributes ~50 queries rather than 300.  The total across
> all six datasets is therefore approximately **1 550** (5 × 300 + ~50).

Each dataset's queries are split **per-dataset** into:

| Split | Fraction | Queries/dataset (typical) | Approx. total |
|-------|----------|---------------------------|---------------|
| Train | 70 % | 210 (≈35 for trec-covid) | ~1 085 |
| Dev   | 15 % | 45  (≈7 for trec-covid)  | ~232  |
| Test  | 15 % | 45  (8 for trec-covid)   | **233** |

Train + dev (85 %, ~1 317 queries) is used by all grid searches via
10-fold cross-validation.  The test split (15 %, **233 queries**) is
**held out and never seen by any model** until the per-dataset evaluation
steps.  The split assignments are cached to `data/results/merged_split.json`
and are never recomputed once written, so every re-run of the pipeline
evaluates on the same test queries.

All grid searches honour the global cap `max_models_per_grid = 1500`;
combinations beyond the cap are deterministically truncated (deterministic
because Python dicts preserve insertion order and grid lists are defined in
the config).

---

## STEP 1 — Download datasets

Downloads each BEIR dataset listed under `datasets:` in `config.yaml` to
`data/datasets/<dataset>/`.  If a dataset is already present (checked by
directory existence), the download is skipped.  The six datasets together
total roughly 1.5 GB on disk after extraction.  The function uses BEIR's
official `util.download_and_unzip` so file integrity is guaranteed.  No
preprocessing happens at this stage.

**Output:** `data/datasets/<dataset>/` (one directory per dataset)

---

## STEP 2 — Preprocessing

For every dataset the script writes the canonical export files
(`corpus.jsonl`, `queries.jsonl`, `qrels.tsv`) and then builds all artefacts
that downstream stages need:

### 2a. Tokenised corpus and queries

Tokenisation is lowercase + whitespace split.  Optionally a
**Snowball English stemmer** is applied (`use_stemming` flag, optimised in
Step 3).  The pipeline builds *two* variants in parallel — stemmed
(`stem_1`) and unstemmed (`stem_0`) — so Step 3's grid search can switch
between them without re-tokenising.

Corpus tokenisation is parallelised with `ProcessPoolExecutor`.  Each
worker receives a batch of raw documents and a fresh stemmer instance (one
per process to avoid sharing state across threads).  The tokenised output is
appended in the original document order to
`tokenized_corpus_stem_<0|1>.jsonl`.

Query tokens are written to `tokenized_queries_stem_<0|1>.jsonl` and to a
`query_tokens_stem_<flag>.pkl` dictionary keyed by query ID.

### 2b. BM25 index

`BM25Okapi(tokenized_corpus, k1, b)` is constructed for every
`(k1, b, use_stemming)` triple encountered by the grid search.  The BM25
object and its corresponding ordered list of document IDs are pickled
separately so they can be loaded without re-building.

### 2c. Frequency indices (word freq and doc freq)

A single pass over the tokenised corpus produces both:

* `word_freq_index` — mapping `term → total count across corpus`
* `doc_freq_index` — mapping `term → number of documents containing term`

These are used by the cross-entropy and average/max-IDF features in Step 5.
Building them in a single pass (merged into `_build_bm25_and_freq_indices`)
avoids scanning the corpus twice.

After `BM25Okapi()` is constructed from `tokenized_docs`, the list is
deleted (`del tokenized_docs`) immediately to free memory before
BM25's internal structures have finished being built.

### 2d. Dense embeddings (BGE-M3)

Corpus and query embeddings are produced by `BAAI/bge-m3` (1024-dimensional
dense head) on the GPU when available, falling back to CPU otherwise.  An
OOM-resilient helper (`_embed_with_oom_retry`) automatically halves the
sub-batch size on `torch.cuda.OutOfMemoryError` and retries; if the GPU
raises an unrecoverable error the helper falls back to CPU for the remainder
of the batch.  After encoding is complete, `torch.cuda.empty_cache()` is
called to release unused VRAM.

The corpus embeddings tensor and its corresponding ordered document-ID list
are saved separately (`corpus_embeddings.pt` and `corpus_ids.pkl`) so they
can be loaded without a model dependency.  The same pattern is used for
query embeddings.

**Output:** under `data/processed/<embedding_model>/<dataset>/`:
```
corpus.jsonl, queries.jsonl, qrels.tsv
tokenized_corpus_stem_<0|1>.jsonl
tokenized_queries_stem_<0|1>.jsonl
query_tokens_stem_<flag>.pkl
word_freq_index_stem_<flag>.pkl
doc_freq_index_stem_<flag>.pkl
bm25_k1_<…>_b_<…>_stem_<…>.pkl
bm25_k1_<…>_b_<…>_stem_<…>_doc_ids.pkl
bm25_k1_<…>_b_<…>_stem_<…>_topk_<k>_results.pkl
dense_results_topk_<k>.pkl
corpus_embeddings.pt, corpus_ids.pkl
query_vectors.pt, query_ids.pkl
```

---

## STEP 3 — Optimise BM25 (k1, b, use_stemming)

Runs a grid search over the BM25 hyperparameter space defined under
`bm25_grid_search` in `config.yaml`:

```yaml
bm25_grid_search:
  k1:  [0.8, 1.2, 1.5, 1.6, 2.0]
  b:   [0.0, 0.25, 0.5, 0.75, 1.0]
  use_stemming: [true, false]
```

This yields 5 × 5 × 2 = **50 combinations**.

For every combination the pipeline ensures the corresponding tokenised
corpus, frequency indices, and BM25 index exist (building them on demand and
caching them keyed by the parameter triple so subsequent steps with the same
params are free), then performs BM25 retrieval for the 300 sampled queries
per dataset.  NDCG@100 is computed per query and averaged per dataset; the
score for a combination is the **macro-average across all six datasets**.

The full grid is written sorted descending by macro NDCG@100 to
`bm25_grid_search.csv`.  The single best configuration is persisted to
`bm25_best_params.json`.  From this point on every subsequent step calls
`get_active_bm25_params(cfg)` which reads this file and returns the winning
triple.

**Best parameters found:**
```json
{ "k1": 1.2, "b": 0.75, "use_stemming": true, "macro_ndcg@100": 0.3265 }
```

**Output:** `data/results/bm25_grid_search.csv`,
`data/results/bm25_best_params.json`

---

## STEP 4 — Oracle alpha grid search (per query)

For each of the ~1 550 selected queries the pipeline finds the optimal
fusion weight:

```
α* = argmax_{α ∈ {0.00, 0.01, …, 1.00}} NDCG@100( wRRF(α | q) )
```

where the weighted Reciprocal Rank Fusion score for a document d is:

```
score(d) = α · 1/(k + rank_bm25(d))  +  (1−α) · 1/(k + rank_dense(d))
           with k = rrf.k = 60
```

**Algorithm.**  The candidate pool is the union of BM25 top-100 and Dense
top-100 (using the best BM25 parameters from Step 3 and the BGE-M3 dense
retriever).  Within the loop the rank arrays

```
R_bm[d] = 1 / (60 + rank_bm25(d))
R_de[d] = 1 / (60 + rank_dense(d))
```

are precomputed once per query.  Each of the 101 alpha values then evaluates
a *vectorised* fused score `α·R_bm + (1−α)·R_de`, sorts it, takes the top
100, and computes NDCG@100 against the full qrels.

**Tie-breaking.**  When multiple alpha values achieve the same maximum NDCG,
the lowest alpha (first in the ascending iteration) is chosen.  This is
deterministic.

**No-relevance queries.**  Queries with no relevant document in the qrels
are emitted with α = 0.5 and `oracle_ndcg = 0`.  They remain in the dataset
and contribute uniform noise rather than disappearing.  They are excluded
from the per-dataset oracle NDCG average reported in
`oracle_ndcg_per_dataset.json` (i.e. the average is over queries with at
least one relevant document).

**Oracle NDCG@100 per dataset:**
| Dataset | Oracle NDCG@100 |
|---------|----------------|
| scifact | 0.7697 |
| nfcorpus | 0.3340 |
| arguana | 0.4782 |
| fiqa | 0.5486 |
| scidocs | 0.3045 |
| trec-covid | 0.4610 |
| **MACRO** | **0.4827** |

This is the theoretical ceiling for any alpha-fusion method on this dataset.

**Output:** `data/results/oracle_alphas.csv` (`ds_name, qid, oracle_alpha,
oracle_ndcg`), `data/results/oracle_ndcg_per_dataset.json`

---

## STEP 5 — Weak-model feature dataset

Builds the 16-dimensional hand-crafted feature matrix that the weak router
will be trained on.  Every row corresponds to one of the ~1 550 queries.
Features are organised in five groups:

### Group A — Query Surface (3 features)
| Feature | Description |
|---------|-------------|
| `query_length` | Number of tokens in the query |
| `stopword_ratio` | Fraction of query tokens that are NLTK English stop words |
| `has_question_word` | Binary: 1 if query starts with who/what/when/where/why/how |

### Group B — Vocabulary Match (4 features)
| Feature | Description |
|---------|-------------|
| `average_idf` | Mean IDF of query tokens against the corpus (`log(N/df + 1)`) |
| `max_idf` | Maximum IDF among all query tokens |
| `rare_term_ratio` | Fraction of query tokens with df < `rare_term_threshold` |
| `cross_entropy` | Cross-entropy of the query language model against the corpus unigram distribution; Laplace-smoothed with `ce_smoothing_alpha` |

### Group C — Retriever Confidence (4 features)
| Feature | Description |
|---------|-------------|
| `top_sparse_score` | Top-1 BM25 score after min-max normalisation of the ranked list |
| `sparse_confidence` | Top-1 minus top-2 BM25 score margin (normalised) |
| `top_dense_score` | Top-1 dense cosine similarity |
| `dense_confidence` | Top-1 minus top-2 cosine similarity margin |

### Group D — Retriever Agreement (3 features)
| Feature | Description |
|---------|-------------|
| `overlap_at_k` | Jaccard overlap of BM25 top-k and dense top-k (`feature_stat_k`) |
| `first_shared_doc_rank` | Harmonic mean rank of the first document in the intersection |
| `spearman_topk` | Spearman rank correlation of BM25 and dense scores over the top-k union |

### Group E — Distribution Shape (2 features)
| Feature | Description |
|---------|-------------|
| `sparse_entropy_topk` | Shannon entropy of normalised BM25 top-k score distribution |
| `dense_entropy_topk` | Shannon entropy of normalised dense top-k score distribution |

The label for every row is the **oracle alpha** read from
`oracle_alphas.csv`.  The split column (`train`/`dev`/`test`) is taken from
`merged_split.json` and is included in the CSV so any later step can filter
by split without re-reading the split file.

**Output:** `data/results/weak_dataset.csv` (~1 550 rows × 19 columns:
`ds_name, qid, split, oracle_alpha, oracle_ndcg` + 16 features)

---

## STEP 6 — Weak-model grid search

For every `(model_family, hyperparameter_combination)` in
`weak_model_grid_search.models` the pipeline runs a **10-fold stratified
cross-validation** on the train + dev rows (~1 317 queries across all six
datasets, before optional reduction due to missing qrels).

### Normalisation (critical for correctness)

Inside each fold:
1. The scaler is **fit on the training fold only** (~1 317 × 9/10 ≈ 1 185
   queries per training fold in the inner loop).
2. The same scaler is applied to transform the validation fold.
3. **No statistics from the validation fold ever influence the scaler.**

This ensures zero data leakage between splits.  The scaler used is
`sklearn.preprocessing.StandardScaler` (z-score: subtract mean, divide by
std).  When the final model is trained on the full 85 % after grid search,
a fresh `StandardScaler` is fit on all ~1 317 train+dev rows and saved
alongside the model so it can be applied at inference time.

### Prediction protocol

* **Regressors** (XGBoost, Ridge, SVR, …): predict a continuous alpha in
  `[0, 1]`, clipped to that range after prediction.
* **Classifiers** (LogisticRegression, GaussianNB, LDA): output
  `predict_proba[:, pos_class_idx]`, where `pos_class_idx` is looked up via
  `model.classes_` to handle the degenerate case where the positive class
  (α = 1) never appears in a fold.

### Scoring

Validation alpha predictions are fed into the wRRF formula against the
actual BM25 and dense retrieval results (not a proxy loss).  NDCG@100 is
computed per validation query and averaged per fold, then macro-averaged
across the 10 folds.  This means the grid optimises the exact retrieval
metric that matters.

Models that fail to fit (degenerate folds, single-class classifier inputs)
silently fall back to a uniform α = 0.5 prediction for that fold, so the
grid search continues without interruption.

**Best model:** XGBoost with `cv_ndcg@100 = 0.4362`
```json
{
  "model": "xgboost",
  "params": { "colsample_bytree": 0.8, "gamma": 0.0,
              "learning_rate": 0.05, "max_depth": 4,
              "min_child_weight": 1, "n_estimators": 300, "subsample": 0.8 }
}
```

The final weak model is retrained on all ~1 317 train+dev queries using the
best params and a fresh scaler, then saved to `data/models/weak_model.pkl`
(containing: `model`, `scaler`, `feature_cols`, `feature_names`).

**Output:** `data/results/weak_grid_search_top.csv` (top 100 combinations),
`data/results/weak_best_params.json`, `data/models/weak_model.pkl`

---

## STEP 7 — Weak-model feature ablation

Holds the best `(model, params)` from Step 6 fixed and re-evaluates the
same 10-fold CV protocol with three families of feature subsets:

* **Full** — all 16 features (baseline, cv_ndcg@100 = 0.4362).
* **Leave-one-feature-out** — 16 configurations, one per individual feature.
* **Leave-one-group-out** — 5 configurations, one per feature group A–E.

The normalisation protocol is unchanged: the scaler is fit on the training
fold only in every inner loop.

Results are sorted by `cv_ndcg@100` descending.  A two-panel horizontal-bar
plot is saved to `weak_ablation.png` (top panel: individual features; bottom
panel: groups), with each bar annotated by its delta against the full-model
score and a green dashed reference line at the full-model score.

The feature configuration with the **highest CV NDCG@100** — which may be a
subset rather than the full set — is selected as the final weak feature set.
The best model is retrained on the full 85 % train+dev portion using only
those columns and its scaler, and the bundle is saved to
`data/models/weak_model.pkl` (overwriting the Step 6 version with the
ablation-selected feature set).

**Key ablation finding:**  Removing `top_sparse_score` (Group C) slightly
*improved* CV NDCG (0.4367 > 0.4362), suggesting it introduces mild noise.
All group-level ablations decreased performance vs. the full model, with
Group A (Query Surface) being the most harmful to remove (−0.003).

**Output:** `data/results/weak_ablation.csv`,
`data/results/weak_ablation_combo.csv`,
`data/results/weak_ablation.png`,
`data/models/weak_model.pkl`

---

## STEP 8 — Weak retrieval comparison (test set)

Evaluates four retrieval methods on the **held-out 15 % test set** (never
seen during training or grid search):

| Method | Description |
|--------|-------------|
| BM25 | Pure sparse retrieval, top-100, best BM25 params |
| Dense | Pure dense retrieval (BGE-M3), top-100 |
| Static RRF (α = 0.5) | wRRF with fixed α = 0.5 (equal weight) |
| wRRF (weak) | α predicted per query by the saved weak model (with its scaler applied) |

NDCG@100 is computed per query, averaged per dataset, and macro-averaged
across the six datasets.  A grouped bar chart with one bar per method per
dataset (+ MACRO) is saved to `weak_retrieval_comparison.png`.

**Output:** `data/results/weak_retrieval_comparison.csv`,
`data/results/weak_retrieval_comparison.png`

---

## STEP 9 — Plot weak router alpha distributions

Two diagnostic plots over the **test set** queries:

* **Box plot** (`weak_alphas_boxplot.png`): distribution of weak-router
  predicted α per dataset plus a MACRO column.  Mean is a dashed red line,
  median is a dark-blue line.  Reveals per-dataset routing bias
  (e.g. arguana strongly prefers dense, trec-covid strongly prefers BM25).

* **Sorted overlay** (`weak_alphas_sorted.png`): oracle α values sorted
  ascending (blue dots) with the weak router's predicted α for the same
  queries overlaid (red dots, in the same sort order).  The closer the red
  trace approaches the blue diagonal, the better the routing calibration.

**Output:** `data/results/weak_alphas_boxplot.png`,
`data/results/weak_alphas_sorted.png`

---

## STEP 10 — SHAP explainability (weak model)

Computes a SHAP summary plot for the merged dataset (all ~1 550 queries).
For tree-based models (XGBoost, LightGBM, RandomForest, ExtraTrees) the
fast `shap.TreeExplainer` is used; for any other model family the script
falls back to `shap.KernelExplainer` with a 100-row background sample and
200 samples per explanation.

The plot shows all 16 features sorted by mean absolute SHAP value, with
individual points coloured by feature value (blue = low, red = high).
The model name is included in the figure title for provenance.

**Output:** `data/results/weak_shap.png`

---

## STEP 11 — Strong-model feature dataset

Constructs the strong-router dataset whose features are the **1 024-dimensional
BGE-M3 query embeddings** for the same ~1 550 queries, with the same split
assignments.  The embeddings are loaded from each dataset's cached
`query_vectors.pt`, re-aligned to the canonical qid order in
`merged_qids.json`, vertically stacked, and saved as a single pickle
containing `rows` (split metadata matching the weak dataset) and `X`
(a `(~1550, 1024)` float32 matrix).

Pickle is used rather than CSV because 1.5 M floats round-trip much faster
without text serialisation overhead and without precision loss.

**Output:** `data/results/strong_dataset.pkl`

---

## STEP 12 — Strong-model grid search

Identical protocol to Step 6 (10-fold CV, fit scaler on training fold only,
predict alpha, score via actual wRRF NDCG@100) but on the 1 024-dimensional
embedding features.  The model families are: Ridge, ElasticNet, KNN, SVR,
MLP, XGBoost, RandomForest, ExtraTrees.

**Normalisation** is identical: `StandardScaler` fit on training fold only,
applied to validation fold.  Because the embedding dimensions can have
very different variances, this normalisation step is critical for
distance-based (KNN) and kernel-based (SVR) models.

**Best model:** KNN with `cv_ndcg@100 = 0.4343`
```json
{ "model": "knn", "params": { "n_neighbors": 5, "weights": "distance" } }
```

The final strong model is retrained on all ~1 317 train+dev queries and saved
to `data/models/strong_model.pkl`.

**Output:** `data/results/strong_grid_search_top.csv`,
`data/results/strong_best_params.json`,
`data/models/strong_model.pkl`

---

## STEP 13 — Strong retrieval comparison (test set)

Same as Step 8 but evaluating **five methods**: BM25, Dense, Static RRF,
wRRF (weak), wRRF (strong).  Having both routers in the same chart enables
direct comparison on identical test queries.  The strong router's predicted
alphas are produced by loading the saved strong bundle and applying the saved
scaler to the test embeddings.

**Output:** `data/results/strong_retrieval_comparison.csv`,
`data/results/strong_retrieval_comparison.png`

---

## STEP 14 — Plot strong router alpha distributions

Same two plots as Step 9 but using the strong router's predicted alphas.

**Output:** `data/results/strong_alphas_boxplot.png`,
`data/results/strong_alphas_sorted.png`

---

## STEP 15 — MoE meta-learner dataset

Builds the two-dimensional meta-feature matrix on which the MoE router is
trained.  The critical design goal is **zero leakage** between the base
models (weak, strong) and the MoE on the train+dev portion.

### OOF (out-of-fold) predictions for train+dev

For every train+dev query a prediction is generated by a base model that
has **not** seen that query during its training:

1. The pipeline runs 10-fold CV on the ~1 317 train+dev rows using the
   **same fold definition** (same seed, same `StratifiedKFold` object) as
   Steps 6 and 12.
2. For each fold, the weak and strong models are re-trained from scratch on
   the training portion of that fold (with fresh scalers) using the best
   `(model, params, feature_cols)` bundles from Steps 7 and 12.
3. The held-out fold's alpha prediction comes from these freshly trained
   base models, so no train+dev query ever contaminates its own OOF
   prediction.

### Test predictions

For the 225–233 test queries the already-trained bundles from
`data/models/weak_model.pkl` and `data/models/strong_model.pkl` are used
directly (they were trained on all ~1 317 train+dev queries, so they have
never seen any test query).

### Alignment assertion

Before packaging the meta-rows, the pipeline asserts that the weak and
strong OOF qid lists are **identical in position order** for both train+dev
and test portions.  If they diverge (e.g. because one side was re-generated
with a different seed), the script aborts with a clear error message.

### Meta-dataset columns

`ds_name, qid, split, alpha_weak, alpha_strong, alpha_gt`

where `alpha_gt` is the oracle alpha from `oracle_alphas.csv`.

**Output:** `data/results/moe_dataset.csv`

---

## STEP 16 — MoE meta-learner grid search

The MoE grid (`moe_grid_search.models`) covers ten relatively shallow model
families — Ridge, Lasso, ElasticNet, SVR, KNN, RandomForest, ExtraTrees,
MLP, XGBoost, LightGBM — because the input is only 2-dimensional (or
3-dimensional for models that benefit from interaction terms).

### Feature construction

* For tree-based models: `[α_weak, α_strong]` (2 features).
* For linear, distance, kernel, and neural models: additionally include
  `|α_weak − α_strong|` (3 features), giving them the ability to express
  interaction terms without sacrificing identifiability.

### Normalisation

Same `StandardScaler` protocol: fit on train fold, apply to val fold.
Because the input range is bounded within `[0, 1]`, normalisation has less
impact here than in the strong model, but it remains correct.

### Scoring

NDCG@100 is evaluated on the validation queries using the actual BM25 +
dense retrieval results — the MoE's predicted alpha is plugged into the wRRF
formula against real candidate lists.

**Best model:** SVR with `cv_ndcg@100 = 0.4355`
```json
{ "model": "svr", "params": { "C": 1.0, "epsilon": 0.05 } }
```

The final MoE model is retrained on the full traindev meta-dataset and saved
to `data/models/moe_model.pkl`.

**Output:** `data/results/moe_grid_search_top.csv`,
`data/results/moe_best_params.json`,
`data/models/moe_model.pkl`

---

## STEP 17 — MoE decision heatmap

Renders the prediction surface of the saved MoE model over the unit square
`(α_strong, α_weak) ∈ [0, 1]²`.  The grid is sampled at 100 × 100 =
10 000 points; predictions are reshaped into a 2-D array and drawn as a
30-level filled contour plot with the `RdYlBu_r` palette (blue = MoE prefers
Dense (α→0), red = MoE prefers BM25 (α→1)).

Real queries are scatter-overlaid colour-coded by dataset.  Small
translucent markers are train+dev queries; large markers with black edges
are the held-out test queries.

**Output:** `data/results/moe_decision_heatmap.png`

---

## STEP 18 — Full retrieval comparison: six methods (test set)

The complete head-to-head evaluation of all six methods on the held-out test
set:

| Method | Description |
|--------|-------------|
| BM25 | Sparse-only, best k1/b/stemming |
| Dense | Dense-only (BGE-M3) |
| Static RRF | wRRF with α = 0.5 (no adaptation) |
| wRRF (weak) | 16-feature XGBoost router |
| wRRF (strong) | 1024-dim BGE-M3 embedding KNN router |
| wRRF (MoE) | SVR meta-learner combining weak + strong |

NDCG@100 per dataset and macro is written to CSV and plotted.

**Output:** `data/results/moe_retrieval_comparison.csv`,
`data/results/moe_retrieval_comparison.png`

---

## STEP 19 — Plot MoE router alpha distributions

Same two diagnostic plots as Steps 9 and 14, this time for the MoE router's
predicted alphas on the test set.

**Output:** `data/results/moe_alphas_boxplot.png`,
`data/results/moe_alphas_sorted.png`

---

## STEP 20 — Recall@100

Recall@100 is computed for **seven** sets of candidates on the test set:

| Candidate set | Description |
|--------------|-------------|
| BM25 top-100 | Sparse retrieval only |
| Dense top-100 | Dense retrieval only |
| Static RRF top-100 | wRRF with α = 0.5 |
| wRRF (weak) top-100 | Weak-router fusion |
| wRRF (strong) top-100 | Strong-router fusion |
| wRRF (MoE) top-100 | MoE-router fusion |
| BM25 ∪ Dense (union) | All 200 unique docs from both top-100 lists — **ceiling** |

The union set is the theoretical recall ceiling for any re-ranker drawing
candidates from both first-stage retrievers.  It is included as a reference
bar and is unranked (Recall@100 is rank-invariant).

Queries with no relevant document are excluded from the per-method averages
to avoid deflating recall with structurally zero-recall queries.

A grouped bar chart with 95 % bootstrap CI error bars is saved alongside
the CSV.  The CIs use `n_resamples = 1000` with `seed = 42`.  The plot is
regenerated from the cached CSV if the main outputs exist but the PNG is
missing (standalone recovery).

**Output:** `data/results/recall_at_100.csv`,
`data/results/recall_at_100.png`,
`data/results/recall_ci.csv`

---

## STEP 21 — Cross-encoder reranking

Loads the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` and reranks
the top-100 candidate set produced by each of the six retrieval methods.

### Efficiency: shared scoring

Across the six methods the candidate sets overlap heavily (all draw from
BM25 ∪ Dense).  To avoid scoring the same `(query, document)` pair twice:
1. The union of all required `(qid, doc_id)` pairs across all six methods
   for one dataset is enumerated.
2. The `needed_doc_ids` set is extracted and only those documents are loaded
   from disk via `load_corpus_subset()` (streaming JSONL scan, stops early
   when all target documents are found).
3. All pairs are scored in a single batched `predict` call (batch size 128
   on CUDA, 32 on CPU).
4. Scores are cached to `rerank_scores_<ce_short>_<dataset>.pkl`.

On subsequent runs the cache is loaded and checked for completeness.  If
any required pair is missing (e.g. because new queries were added), only the
missing pairs are scored and merged into the cache — previously scored pairs
are never discarded.

### Metrics

For each method, NDCG@100 is computed *before* (original retrieval ranking)
and *after* (re-sorted by cross-encoder score descending) reranking.  Two
charts are produced:

* `rerank_ndcg.png` — re-ranked NDCG@100 per method per dataset (six bars
  per group), with 95 % bootstrap CI error bars.
* `rerank_gain.png` — `ΔNDCG@100 = re-ranked − original` per method per
  dataset; a horizontal dashed zero line marks the break-even point.

**Output:** `data/results/rerank_ndcg.csv`,
`data/results/rerank_ndcg.png`,
`data/results/rerank_gain.png`,
`data/results/rerank_mrr.csv`,
`data/results/<rerank_cache>_<dataset>.pkl`

---

## STEP 22 — Paired significance tests (NDCG@100)

For every unordered pair of methods among (BM25, Dense, Static RRF,
wRRF weak, wRRF strong, wRRF MoE) — **15 pairs total** — the pipeline:

1. Collects per-query NDCG@100 scores for the two methods on the merged
   test set (n = 233 queries).
2. Runs `scipy.stats.ttest_rel` (paired two-sided t-test).
3. Reports: `n`, `mean_diff`, `t`, `p_value`, `cohens_d`, raw significance
   at α = 0.05, and Holm-Bonferroni corrected significance.

**Holm-Bonferroni correction** is applied across all 15 comparisons to
control the family-wise error rate.  Each row therefore has both a
`significant` column (raw α = 0.05) and a `significant_holm` column
(corrected).

Cohen's d effect size is computed as `mean_diff / pooled_std`.

A grouped bar chart with 95 % bootstrap CI error bars is also generated
from the NDCG scores.

**Output:** `data/results/significance_tests.csv`,
`data/results/ndcg_ci.csv`,
`data/results/ndcg_ci.png`

---

## STEP 23 — MRR@100 and Recall significance tests

Analogous to Step 22 but for MRR@100 and Recall@100.  The same 15 pairs are
tested for each metric.  Outputs are the per-query MRR and recall score
tables plus their t-test summaries.

**Output:** `data/results/mrr_at_100.csv`,
`data/results/mrr_ci.csv`,
`data/results/mrr_ttest.csv`,
`data/results/mrr_ci.png`,
`data/results/recall_ttest.csv`,
`data/results/recall_ci.csv`,
`data/results/recall_ci.png`

---

## STEP 24 — Oracle NDCG summary

Verifies and exports the oracle NDCG per dataset (already computed in
Step 4) to a readable JSON for reference.  The macro is computed only over
queries that have at least one relevant document (same filter as Step 4).

**Output:** `data/results/oracle_ndcg_per_dataset.json`

---

## STEP 25 — Latency benchmarking

Benchmarks end-to-end query latency for every method on every dataset.

### What is measured

For each `(method, dataset)` combination the pipeline times n repetitions
(default: equal to the test set size) of the full query path:

* **BM25:** tokenise query → BM25 score → argsort top-100.
* **Dense:** embed query (GPU inference) → cosine similarity → argsort
  top-100.  Query embedding time is included.
* **Static / wRRF / MoE:** dense time + BM25 time + wRRF merge step +
  (router inference for adaptive methods).
* **With reranking (+`_rer` columns):** the above plus cross-encoder
  `predict()` on the top-100 candidates.
* **CE-only (`_ce_only` columns):** cross-encoder scoring time in isolation,
  excluding first-stage retrieval.

All timings use `time.perf_counter()` (high-resolution wall-clock time).
The GPU is warmed up before measurement begins.

### Reported statistics

For each `(method, dataset)` cell the table records:
- `mean` latency per query (ms)
- `median` latency per query (ms)
- `p95` (95th percentile) latency per query (ms)

The macro row is the **dataset-weighted mean** of per-dataset summary
statistics (consistent with how NDCG/MRR/Recall macros are computed).

**Output:** `data/results/latency.csv`,
`data/results/latency.png`

---

## Data and model layout

```
data/
├── datasets/                      # BEIR raw datasets
│   ├── arguana/
│   ├── fiqa/
│   ├── nfcorpus/
│   ├── scidocs/
│   ├── scifact/
│   └── trec-covid/
├── processed/<embedding_model>/   # per-dataset preprocessed artefacts
│   └── <dataset>/
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       ├── qrels.tsv
│       ├── tokenized_corpus_stem_<0|1>.jsonl
│       ├── tokenized_queries_stem_<0|1>.jsonl
│       ├── query_tokens_stem_<flag>.pkl
│       ├── word_freq_index_stem_<flag>.pkl
│       ├── doc_freq_index_stem_<flag>.pkl
│       ├── bm25_k1_<k1>_b_<b>_stem_<flag>.pkl
│       ├── bm25_k1_<k1>_b_<b>_stem_<flag>_doc_ids.pkl
│       ├── bm25_k1_<k1>_b_<b>_stem_<flag>_topk_<k>_results.pkl
│       ├── dense_results_topk_<k>.pkl
│       ├── corpus_embeddings.pt
│       ├── corpus_ids.pkl
│       ├── query_vectors.pt
│       └── query_ids.pkl
├── results/                       # CSV / JSON / PNG outputs
│   ├── merged_qids.json
│   ├── merged_split.json
│   ├── bm25_grid_search.csv
│   ├── bm25_best_params.json
│   ├── oracle_alphas.csv
│   ├── oracle_ndcg_per_dataset.json
│   ├── weak_dataset.csv
│   ├── weak_grid_search_top.csv
│   ├── weak_best_params.json
│   ├── weak_ablation.csv
│   ├── weak_ablation_combo.csv
│   ├── weak_ablation.png
│   ├── weak_retrieval_comparison.csv / .png
│   ├── weak_alphas_boxplot.png
│   ├── weak_alphas_sorted.png
│   ├── weak_shap.png
│   ├── strong_dataset.pkl
│   ├── strong_grid_search_top.csv
│   ├── strong_best_params.json
│   ├── strong_retrieval_comparison.csv / .png
│   ├── strong_alphas_boxplot.png
│   ├── strong_alphas_sorted.png
│   ├── moe_dataset.csv
│   ├── moe_grid_search_top.csv
│   ├── moe_best_params.json
│   ├── moe_decision_heatmap.png
│   ├── moe_retrieval_comparison.csv / .png
│   ├── moe_alphas_boxplot.png
│   ├── moe_alphas_sorted.png
│   ├── recall_at_100.csv / .png
│   ├── recall_ci.csv / .png
│   ├── recall_ttest.csv
│   ├── rerank_ndcg.csv / .png
│   ├── rerank_mrr.csv
│   ├── rerank_gain.png
│   ├── rerank_scores_<ce>_<dataset>.pkl
│   ├── significance_tests.csv
│   ├── ndcg_ci.csv / .png
│   ├── mrr_at_100.csv
│   ├── mrr_ci.csv / .png
│   ├── mrr_ttest.csv
│   ├── latency.csv / .png
│   └── hardware.json
└── models/
    ├── weak_model.pkl
    ├── strong_model.pkl
    └── moe_model.pkl
```

---

## Reproducibility

* All random seeds derive from `sampling.random_seed` (default 42) plus a
  deterministic, dataset-specific MD5 offset, so per-dataset shuffles do not
  collide and the pipeline is fully reproducible across machines.
* The query selection is cached at `data/results/merged_qids.json` and the
  70/15/15 split at `data/results/merged_split.json`; both are recomputed
  only if missing.
* All grid searches use the **same 10 CV folds** (same seed + same
  `StratifiedKFold` object) so the OOF predictions in Step 15 are guaranteed
  to use queries that no base model has ever seen during training in the same
  fold.
* Every cache file is keyed by the parameters that affect its content
  (BM25 by `k1, b, use_stemming, top_k`; dense by the embedding model short
  name) so changing `config.yaml` automatically invalidates the relevant
  caches.
* Hardware used for the published results: AMD Ryzen 9 5950X, NVIDIA RTX 4090
  (24 GB VRAM), 62.7 GB RAM, Ubuntu 24.04, CUDA 13.0, PyTorch 2.11.
