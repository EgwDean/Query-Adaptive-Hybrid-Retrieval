# Pipeline — Step-by-Step Reference

This document describes every step performed by `src/pipeline.py`. The pipeline
is a single end-to-end script that produces every artefact (data, models,
metrics, plots) required by the thesis on top of *oracle* alpha labels found
by per-query grid search rather than the closed-form NDCG-difference soft
labels used in the legacy version.

The pipeline is **idempotent**: every step caches its outputs to disk and
skips itself on a subsequent run if they already exist, so the script can be
killed and restarted at any point.

The merged dataset used throughout the pipeline is the concatenation of the
five BEIR datasets `scifact`, `nfcorpus`, `arguana`, `fiqa`, `scidocs`,
truncated to **300 queries per dataset** (1500 total), then split per-dataset
into **train (70 %) / dev (15 %) / test (15 %)**. The 15 % test portion is
**held out and never seen by any model** until the per-dataset evaluations
(steps 8 / 13 / 18 / 20 / 21 / 22).  The remaining 85 % (train + dev) is
used by every grid search via 10-fold cross-validation.

All grids honour the global cap `max_models_per_grid = 1500`; combos beyond
the cap are deterministically truncated.

---

## STEP 1 — Download datasets

Downloads each BEIR dataset listed under `datasets:` in `config.yaml` to
`data/datasets/<dataset>/`. If a dataset is already present, it is skipped.
All five datasets together total ~1.5 GB on disk after extraction. The
function uses BEIR's official `util.download_and_unzip` so file integrity is
guaranteed. No further preprocessing happens at this stage.

**Output:** `data/datasets/<dataset>/`

## STEP 2 — Preprocessing

For every dataset, the script writes the canonical export files
(`corpus.jsonl`, `queries.jsonl`, `qrels.tsv`), then builds the artefacts
that downstream stages need:

* **Tokenised corpus** — lowercase + whitespace split + Snowball stem
  (`stemmer_language: english`). Built in parallel by a `ProcessPoolExecutor`
  whose workers each construct one stemmer instance and reuse it across all
  batches assigned to them. The output is appended in the original order to
  `tokenized_corpus_stem_<0|1>.jsonl`.
* **Tokenised queries** — same stemming pipeline, written to
  `tokenized_queries_stem_<0|1>.jsonl` and a `query_tokens_<flag>.pkl`
  dictionary.
* **BM25 index** — `BM25Okapi(tokens, k1, b)` plus its document-id list,
  pickled separately.
* **Word & document frequency indices** — used by the cross-entropy and IDF
  features in Step 5.
* **Corpus and query embeddings** — produced by `BAAI/bge-m3` on the GPU
  when available, with an automatic OOM-resilient batch-shrinking helper that
  halves the sub-batch size on `torch.cuda.OutOfMemoryError` and falls back
  to CPU if the GPU emits an unrecoverable error. The embeddings tensor and
  the corresponding ID lists are saved separately so they can be loaded
  without a model dependency.

**Output:** under `data/processed/<model>/<dataset>/` —
`corpus.jsonl`, `queries.jsonl`, `qrels.tsv`,
tokenized files, BM25 + frequency indices,
`corpus_embeddings.pt`, `corpus_ids.pkl`,
`query_vectors.pt`, `query_ids.pkl`.

## STEP 3 — Optimise BM25 (k1, b, use_stemming)

Runs a grid search over the BM25 hyperparameter space defined under
`bm25_grid_search` in the config. The grid for the new pipeline is
`5 × 5 × 2 = 50` combinations: `k1 ∈ {0.8, 1.2, 1.5, 1.6, 2.0}`,
`b ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`, `use_stemming ∈ {true, false}`.

For every combination the pipeline ensures the corresponding tokenised
corpus, frequency indices and BM25 index exist (building them on demand and
caching them keyed by the parameter signature so that subsequent runs of any
script that uses the same `(k1, b, stem)` triple are free), then performs
BM25 retrieval restricted to the **300 queries per dataset** that will form
the merged dataset later. NDCG@10 is computed per query and averaged per
dataset; the final score for a combination is the macro-average across the
five datasets.

The full grid is written sorted descending by macro NDCG@10 to
`bm25_grid_search.csv`, and the single best configuration is persisted as
`bm25_best_params.json`. From this point on the helper
`get_active_bm25_params(cfg)` returns the best parameters and **every
subsequent step uses them**.

**Output:** `data/results/bm25_grid_search.csv`,
`data/results/bm25_best_params.json`, plus per-config BM25 indices and
retrieval caches under each dataset's `data/processed/<model>/<dataset>/`.

## STEP 4 — Oracle alpha grid search (per query)

For each of the 1500 selected queries the pipeline finds the best
fusion-weight `α* = argmax_{α} NDCG@10(wRRF(α | q))` by brute force over the
101 values `α ∈ {0.00, 0.01, …, 1.00}`.

The candidate pool for a query is the union of BM25 top-100 and Dense
top-100 (using the active BM25 parameters from Step 3 and the BGE-M3 dense
retriever). Within the loop the rank arrays
`R_bm[d] = 1/(rrf_k + rank_bm25(d))` and `R_de[d] = 1/(rrf_k + rank_dense(d))`
are precomputed once per query. Each alpha then evaluates a *vectorised*
fused score `α · R_bm + (1−α) · R_de`, sorts the result, takes the top
`top_k = 100`, and computes NDCG@10. The IDCG used as the denominator is the
ideal DCG over the *full qrels* (matching `query_ndcg_at_k`), so the
reported `oracle_ndcg` is comparable across queries; the chosen oracle alpha
is mathematically invariant to this choice.

Ties at the maximum NDCG are broken in favour of the **lowest** alpha (the
first alpha to achieve the max in the ascending iteration), which is
deterministic. Queries with no relevant document are emitted with α = 0.5
and `oracle_ndcg = 0` so they remain in the dataset and contribute uniform
noise rather than disappearing.

**Output:** `data/results/oracle_alphas.csv` with columns
`(ds_name, qid, oracle_alpha, oracle_ndcg)`, plus
`data/results/oracle_ndcg_per_dataset.json` — the per-dataset and macro
oracle NDCG@10 for sanity checking.

## STEP 5 — Weak-model dataset

Builds the 16-feature hand-crafted dataset that the weak router will be
trained on. The features replicate exactly those of the legacy pipeline,
organised in five groups:

* **A. Query Surface** — `query_length`, `stopword_ratio`,
  `has_question_word`.
* **B. Vocabulary Match** — `average_idf`, `max_idf`, `rare_term_ratio`,
  `cross_entropy` (with Laplace smoothing, see
  `routing_features.ce_smoothing_alpha`).
* **C. Retriever Confidence** — top score and top-1 vs top-2 confidence
  margin for both retrievers, after min-max normalisation of each ranked
  list.
* **D. Retriever Agreement** — `overlap_at_k`, `first_shared_doc_rank`,
  `spearman_topk` over the top-`feature_stat_k` of each retriever.
* **E. Distribution Shape** — Shannon entropy of the normalised top-k score
  distributions for BM25 and Dense.

The label per query is the **oracle alpha** read from
`oracle_alphas.csv`. The split column reflects the cached
`merged_split.json` and is used by every later step that needs to filter
rows by split.

**Output:** `data/results/weak_dataset.csv` (1500 rows × 19 columns) and the
companion files `data/results/merged_qids.json` and
`data/results/merged_split.json` produced lazily on first use.

## STEP 6 — Weak-model grid search

For every combination of (model, hyperparameters) generated from
`weak_model_grid_search.models`, the pipeline:

1. z-scores the weak-feature matrix using statistics fitted **on the training
   fold only** (so the test fold remains unseen);
2. fits the model on the training fold (`y_oracle` directly for regressors;
   binarised at 0.5 for the classifier branch — `logistic_regression`,
   `gaussian_nb`, `lda`);
3. predicts a per-query alpha on the validation fold (clipped to `[0, 1]`
   for regressors, `predict_proba[:, 1]` for classifiers);
4. evaluates wRRF NDCG@10 on each validation query (using the actual BM25
   and Dense retrieval results, not a proxy loss), averages over the queries
   in the fold, and finally averages across the 10 folds.

Combinations that fail to fit (degenerate folds, single-class classifier
inputs, etc.) silently fall back to a uniform `α = 0.5` prediction so the
grid search continues. The top 100 combinations are written to
`weak_grid_search_top.csv` and the single best one to
`weak_best_params.json`. With CUDA available, `n_jobs = 1` is used for the
outer joblib loop because XGBoost on CUDA is not safe to share across
threads; otherwise the loop uses every core.

**Output:** `data/results/weak_grid_search_top.csv`,
`data/results/weak_best_params.json`.

## STEP 7 — Weak-model feature ablation

Holds the best `(model, params)` from Step 6 fixed and re-evaluates the
same 10-fold CV protocol with three families of feature subsets:

* **Full** — all 16 features (baseline).
* **Leave-one-feature-out** — 16 configurations, one per feature.
* **Leave-one-group-out** — 5 configurations, one per `FEATURE_GROUPS`
  bucket above.

Every configuration's `cv_ndcg@10` is written to `weak_ablation.csv` along
with the JSON-encoded list of feature column indices that produced it.
A two-panel horizontal-bar plot saves to `weak_ablation.png` (top panel:
leave-one-feature-out; bottom panel: leave-one-group-out), with each bar
annotated by its delta against the full-model score and a green dashed
reference line at the full-model NDCG@10.

The configuration with the highest CV NDCG@10 — which in some experiments is
the full set and in others a reduced one — is selected as the **final weak
feature set**. The best model is retrained on the full 85 % train + dev
portion of the merged dataset using only those columns, and persisted to
`data/models/weak_model.pkl` with its scaler statistics, model object, the
selected `feature_cols` and `feature_names`. Every later step that needs a
weak alpha applies this saved bundle directly.

**Output:** `data/results/weak_ablation.csv`,
`data/results/weak_ablation.png`,
`data/models/weak_model.pkl`.

## STEP 8 — Weak retrieval comparison (test set)

Evaluates four methods on the 15 % held-out test set:

* **BM25** — pure sparse retrieval (top-100).
* **Dense** — pure dense retrieval (top-100).
* **Static RRF (α = 0.5)** — Reciprocal Rank Fusion with constant α.
* **wRRF (weak)** — α predicted per query by the saved weak model.

NDCG@10 is computed per dataset and macro-averaged across the five
datasets. The 4-bar grouped chart is saved to
`weak_retrieval_comparison.png` and the underlying numbers to the matching
CSV.

**Output:** `data/results/weak_retrieval_comparison.csv`,
`data/results/weak_retrieval_comparison.png`.

## STEP 9 — Plot weak alphas

Two diagnostic plots over the test set:

* **Box plot** — distribution of the weak-router predicted α per dataset
  plus a **MACRO** column (concatenation of all datasets). Mean is drawn as
  a dashed red line, median as a dark-blue line.
* **Sorted overlay** — the oracle α values are sorted ascending and plotted
  as blue dots; the weak-router's predicted α for the same queries is drawn
  on top as red dots (in the same order). The closer the red trace
  approaches the blue diagonal, the better the routing.

**Output:** `data/results/weak_alphas_boxplot.png`,
`data/results/weak_alphas_sorted.png`.

## STEP 10 — Weak SHAP

Computes a single SHAP summary plot for the merged dataset.  For tree-based
models (`xgboost`, `lightgbm`, `random_forest`, `extra_trees`) the fast
`shap.TreeExplainer` is used; for any other model family the script falls
back to `shap.KernelExplainer` with a 100-row background sample and 200
samples per explanation. The plot displays all features sorted by mean
absolute SHAP value, dots coloured by feature value (blue = low,
red = high). The model name is included in the figure title for
provenance.

**Output:** `data/results/weak_shap.png`.

## STEP 11 — Strong-model dataset

Constructs the strong-router dataset whose features are the **1024-dim
BGE-M3 query embeddings** for the same 1500 queries used by the weak
dataset, with the same train / dev / test split assignments. The
embeddings are loaded from each dataset's cached `query_vectors.pt`,
re-aligned to the canonical qid order, vertically stacked, and saved as a
single pickle with `rows` (split metadata) and `X` (a `(1500, 1024)`
float32 matrix). Saved as a single binary file (rather than a CSV) because
1.5 M floats round-trip much faster through pickle.

**Output:** `data/results/strong_dataset.pkl`.

## STEP 12 — Strong-model grid search

Identical protocol to Step 6 but on the 1024-dim embedding features. The
grid (`strong_model_grid_search.models`) keeps the model families that
performed well in the legacy strong-signal selection — Ridge, ElasticNet,
KNN, SVR, MLP, XGBoost, RandomForest, ExtraTrees. With CUDA available the
outer loop is again sequential because XGBoost on CUDA cannot be shared
across joblib threads.

After the grid finishes the best combination is retrained on the full
85 % train + dev and saved together with its scaler statistics to
`data/models/strong_model.pkl`. The top-100 grid rows go to
`strong_grid_search_top.csv` and the winner to `strong_best_params.json`.

**Output:** `data/results/strong_grid_search_top.csv`,
`data/results/strong_best_params.json`,
`data/models/strong_model.pkl`.

## STEP 13 — Strong retrieval comparison (test set)

Same as Step 8 but with **five** methods: BM25, Dense, Static RRF (α = 0.5),
wRRF (weak) and wRRF (strong). The weak alphas are produced by the saved
weak bundle and the strong ones by the saved strong bundle. Reporting both
in the same chart makes the two routers directly comparable on identical
test queries.

**Output:** `data/results/strong_retrieval_comparison.csv`,
`data/results/strong_retrieval_comparison.png`.

## STEP 14 — Plot strong alphas

Same two plots as Step 9 but using the strong router's predicted alphas.

**Output:** `data/results/strong_alphas_boxplot.png`,
`data/results/strong_alphas_sorted.png`.

## STEP 15 — MoE meta-learner dataset

Builds the meta-dataset on which the MoE router is trained. The crucial
detail is **zero leakage** on the train + dev portion: every traindev row's
`alpha_weak` and `alpha_strong` is produced by a base model that has *not*
seen that query during training. Concretely:

* The pipeline runs **standard 10-fold CV** on the train + dev rows
  (matching the seed and fold definition used by the grid searches), using
  the best `(model, params)` and feature-subset bundle saved by the weak
  and strong stages. Each query's prediction comes from the fold in which
  it was the validation row, never the training row.
* For the **test** rows, the final base models trained on the full
  85 % train + dev (i.e. the bundles in `data/models/`) are queried
  directly, so no traindev query ever contaminates a test prediction.

The meta-dataset has the columns `ds_name, qid, split, alpha_weak,
alpha_strong, alpha_gt`, where `alpha_gt` is the oracle alpha. It is
written as a CSV (small enough not to need pickling) so it can be
inspected manually.

The pipeline asserts that the weak and strong datasets are **aligned
position-by-position** on both the train+dev qids and the test qids before
packaging the rows; if they ever diverge (e.g. because someone re-ran one
side with a different seed) the script aborts with a clear error.

**Output:** `data/results/moe_dataset.csv`.

## STEP 16 — MoE meta-learner grid search

The MoE grid (`moe_grid_search.models`) covers ten relatively shallow model
families — Ridge, Lasso, ElasticNet, SVR, KNN, RandomForest, ExtraTrees,
MLP, XGBoost, LightGBM — because the input is only 2-dimensional. For tree
models the meta-feature vector is `[α_weak, α_strong]`; for linear,
distance, kernel and neural models we additionally include
`|α_weak − α_strong|` so they can express interaction terms without
sacrificing identifiability.

The grid uses the same 10-fold protocol as Steps 6 / 12. NDCG@10 is
evaluated on the validation queries using their actual BM25 + Dense
retrieval results — not a proxy loss on the alphas — so the grid optimises
the metric we ultimately care about. The best combination is retrained on
the full traindev meta-dataset and saved to `data/models/moe_model.pkl`.

**Output:** `data/results/moe_grid_search_top.csv`,
`data/results/moe_best_params.json`,
`data/models/moe_model.pkl`.

## STEP 17 — MoE decision heatmap

Renders the prediction surface of the saved MoE model over the unit square
`(α_strong, α_weak) ∈ [0, 1]^2`. The grid is sampled at 100 × 100 = 10 000
points; predictions are reshaped into a 2-D grid and drawn as a 30-level
filled contour plot with the `RdYlBu_r` palette (blue = MoE prefers Dense,
red = MoE prefers BM25).

Real queries are scatter-overlaid in their dataset's palette colour; small
markers are train + dev queries (drawn translucently), large markers with
black edges are the held-out test queries. The plot reproduces the
"decision boundary" plot from the legacy MoE pipeline.

**Output:** `data/results/moe_decision_heatmap.png`.

## STEP 18 — MoE retrieval comparison (six methods)

The full comparison: BM25, Dense, Static RRF, wRRF (weak), wRRF (strong),
wRRF (MoE). The MoE alphas come from the saved MoE bundle applied to the
test rows of the meta-dataset.

**Output:** `data/results/moe_retrieval_comparison.csv`,
`data/results/moe_retrieval_comparison.png`.

## STEP 19 — Plot MoE alphas

Same two plots as Steps 9 and 14, this time for the MoE router.

**Output:** `data/results/moe_alphas_boxplot.png`,
`data/results/moe_alphas_sorted.png`.

## STEP 20 — Recall@100

Recall@100 is computed for **seven** sets of candidates on the test set:

* The six top-100 lists produced by BM25, Dense, Static RRF, wRRF (weak),
  wRRF (strong) and wRRF (MoE).
* The **union** of BM25's and Dense's full top-100 lists, *unranked* — this
  is the theoretical Recall@100 ceiling for any re-ranker that draws
  candidates from both first-stage retrievers, included as a reference
  bar.

Queries with no relevant document are excluded from the per-method
averages so that they are not double-counted as zero. The seven-bar
grouped chart and the matching CSV are saved.

**Output:** `data/results/recall_at_100.csv`,
`data/results/recall_at_100.png`.

## STEP 21 — Cross-encoder reranking

Loads the cross-encoder defined under `reranker.model_name`
(`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) and reranks the top-100
candidate set produced by each of the six retrieval methods.

**Efficiency.** Across the six methods the candidate sets overlap heavily
(they all draw from BM25 ∪ Dense). To avoid ever scoring the same
`(query, document)` pair twice, the pipeline enumerates the *union* of
required pairs across all six methods for one dataset, scores them in a
single batched `predict` call (batch size 128 on CUDA, 32 on CPU), and
caches the results to `rerank_scores_<ce_short>_<dataset>.pkl`. Subsequent
runs check that the cache contains every required pair before reusing it,
and rebuild it if any pair is missing.

For each method, NDCG@10 is computed *before* and *after* reranking. The
"before" ordering is the original retrieval order; the "after" ordering
sorts the same candidate set by the cross-encoder score (descending). Two
charts are produced:

* `rerank_ndcg.png` — re-ranked NDCG@10 per method per dataset (six bars
  per group).
* `rerank_gain.png` — `Δ NDCG@10 = re-ranked − original` per method per
  dataset (positive = gain, negative = loss; a horizontal dashed zero
  line is drawn for reference).

**Output:** `data/results/rerank_ndcg.csv`,
`data/results/rerank_ndcg.png`,
`data/results/rerank_gain.png`,
plus the per-dataset `rerank_scores_<…>_<dataset>.pkl` caches.

## STEP 22 — Paired t-tests

For every unordered pair of methods (15 pairs) the pipeline computes
`scipy.stats.ttest_rel` over the per-query NDCG@10 difference on the merged
test set (~225 queries). The output table records the sample size, mean
difference, t-statistic, two-sided p-value and a binary
`significant`/`not significant` flag at the threshold
`significance_test.alpha` (default 0.05).

This step turns "method A beats method B by Δ NDCG@10 = 0.003" into a
defensible "method A beats method B with mean Δ = 0.003, p = …" statement.

**Output:** `data/results/significance_tests.csv`.

---

## Data and model layout produced by the pipeline

```
data/
├── datasets/                      # BEIR raw datasets + zips
│   ├── arguana/
│   ├── fiqa/
│   ├── nfcorpus/
│   ├── scidocs/
│   └── scifact/
├── processed/<embedding model>/   # per-dataset preprocessed artefacts
│   └── <dataset>/
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       ├── qrels.tsv
│       ├── tokenized_corpus_stem_<flag>.jsonl
│       ├── tokenized_queries_stem_<flag>.jsonl
│       ├── query_tokens_stem_<flag>.pkl
│       ├── word_freq_index_stem_<flag>.pkl
│       ├── doc_freq_index_stem_<flag>.pkl
│       ├── bm25_k1_<…>_b_<…>_stem_<…>.pkl
│       ├── bm25_k1_<…>_b_<…>_stem_<…>_doc_ids.pkl
│       ├── bm25_k1_<…>_b_<…>_stem_<…>_topk_<k>_results.pkl
│       ├── dense_results_topk_<k>.pkl
│       ├── corpus_embeddings.pt
│       ├── corpus_ids.pkl
│       ├── query_vectors.pt
│       └── query_ids.pkl
├── results/                       # csv / json / png outputs
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
│   ├── rerank_ndcg.csv / .png
│   ├── rerank_gain.png
│   ├── rerank_scores_<…>_<dataset>.pkl
│   └── significance_tests.csv
└── models/
    ├── weak_model.pkl
    ├── strong_model.pkl
    └── moe_model.pkl
```

## Reproducibility checklist

* All random seeds derive from `sampling.random_seed` (default 42) plus a
  deterministic, dataset-specific MD5 offset, so per-dataset shuffles do not
  collide and the pipeline is reproducible across machines.
* The 1500-query selection is cached at
  `data/results/merged_qids.json` and the 70 / 15 / 15 split at
  `data/results/merged_split.json`; both are recomputed only if missing.
* All grid searches use the **same 10 CV folds** (same seed) so the OOF
  predictions in Step 15 are guaranteed to use queries that no base model
  ever saw during training in the same fold — this is the leakage barrier
  between Steps 6 / 12 and Steps 15 / 16.
* Every cache file is keyed by the parameters that affect its content
  (BM25 by `k1, b, use_stemming, top_k`; embeddings by the embedding-model
  short name) so changing `config.yaml` automatically invalidates the
  affected caches.

## Cleanup

`src/cleanup.py` exists solely to wipe stale outputs from `data/results/`
and `data/models/`, plus a curated set of stale caches under
`data/processed/`. It runs once on the HPC node before the first end-to-end
pipeline run and is then deleted (per the TODO instructions). Pass
`--dry-run` to print everything that would be removed without actually
deleting it.
