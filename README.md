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
3. **Router** — an XGBoost model predicts a weight $\hat{\alpha}(q) \in [0, 1]$ from
   15 query features. Values near 1 favour sparse; values near 0 favour dense.
4. **Weighted RRF fusion** — the two ranked lists are combined:

$$\text{score}(q, d) = \hat{\alpha}(q) \cdot \frac{1}{k + r_\text{sparse}(d)} + (1 - \hat{\alpha}(q)) \cdot \frac{1}{k + r_\text{dense}(d)}$$

with $k = 60$ (RRF damping constant). Setting $\hat{\alpha} = 0.5$ recovers
static RRF, which serves as the primary fusion baseline.

The router is trained with **soft labels** derived from the relative NDCG@10 of
BM25 vs dense on each query (see `docs/routing_features.md`).

---

## Key results

Macro NDCG@10 across 5 BEIR datasets on 15% held-out test sets:

| Method | Macro NDCG@10 |
|--------|---------------|
| BM25-only | 0.2867 |
| Dense-only | 0.4135 |
| Static RRF (α = 0.5) | 0.3768 |
| **wRRF — XGBoost router** | **0.4008** |

The router adds **+0.024** over static RRF and closes **80%** of the gap
between static RRF and dense-only. See `docs/retrieval_explainability_results.md`
for the full per-dataset breakdown and SHAP explainability analysis.

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

## Pipeline — run order

Each step produces cached artifacts consumed by the next.

```bash
# 1. Download BEIR datasets
python src/download.py

# 2. Build preprocessing caches (BM25 index, dense embeddings, frequency indexes)
python src/preprocess.py

# 3. Optimise BM25 parameters (k1, b, stemming) via grid search
python src/optimize_bm25.py

# 4. Model family and global hyperparameter selection (grid search across 9 models)
python src/weak_signal_model_grid_search.py

# 5. Feature ablation study (leave-one-out and leave-one-group-out)
python src/ablation_study.py

# 6. Full vs reduced model comparison (16 features vs 14 features)
python src/full_vs_smaller_model_ablation_study.py

# 7. Per-dataset XGBoost hyperparameter grid search (expanded grid, 85/15 split)
python src/weak_signal_params_grid_search.py

# 8. Final evaluation: compare all four retrieval methods + SHAP explainability
python src/retrieval_explainability.py
```

---

## Source files

### Pipeline scripts (`src/`)

| File | Purpose |
|------|---------|
| `src/download.py` | Downloads the five BEIR datasets from the Hugging Face hub into `data/datasets/`. |
| `src/preprocess.py` | Builds all preprocessing artifacts: BM25 tokenised corpus and query tokens, dense corpus embeddings and query vectors, word/doc frequency indexes. Skips steps whose outputs already exist and are non-empty. |
| `src/optimize_bm25.py` | Grid-searches BM25 parameters (k1, b, use_stemming) and writes the best combination to `data/results/bm25_optimization_macro.csv` and `bm25_optimization_best.json`. The best parameters should be copied to `config.yaml` before proceeding. |
| `src/weak_signal_model_grid_search.py` | Grid-searches 9 model families with ~2 600 total hyperparameter combinations under 10-fold Monte Carlo CV. Saves the top-100 results to `data/results/model_grid_search_top100.csv`. Also contains all shared utilities: feature computation, soft-label derivation, NDCG@k, dataset loading, caching. |
| `src/ablation_study.py` | Feature ablation with the global-best XGBoost configuration. Evaluates 22 configurations (full model + 16 leave-one-feature-out + 5 leave-one-group-out) and produces `data/results/ablation_study.csv` and `ablation_study.png`. |
| `src/full_vs_smaller_model_ablation_study.py` | Direct comparison between the full 16-feature model and the 14-feature model with `query_length` and `average_idf` removed. Produces `data/results/full_vs_smaller_ablation.csv` and `full_vs_smaller_ablation.png`. |
| `src/weak_signal_params_grid_search.py` | Per-dataset XGBoost hyperparameter grid search (~6 480 combinations, expanded grid including gamma). Uses a fixed 85/15 traindev/test split per dataset. Writes `data/results/per_dataset_best_params.csv`. |
| `src/retrieval_explainability.py` | Final evaluation on per-dataset held-out test sets. Compares BM25-only, Dense-only, Static RRF, and wRRF (XGBoost). Generates SHAP beeswarm plots per dataset. Writes `data/results/retrieval_comparison.csv`, `retrieval_comparison.png`, and `shap_<dataset>.png` for each dataset. |
| `src/utils.py` | Shared utilities: config loading, path resolution, BM25 signature, model short name, pickle helpers, directory helpers. |

### Maintenance

| File | Purpose |
|------|---------|
| `cleanup_hpc.py` | Removes stale data artifacts from previous pipeline runs that do not match the current configuration. Run `python cleanup_hpc.py` for a dry-run preview, or `python cleanup_hpc.py --confirm` to delete. Never touches raw BEIR downloads or dense embeddings. |

---

## Configuration (`config.yaml`)

All pipeline parameters live in a single YAML file. Key sections:

| Section | Controls |
|---------|----------|
| `datasets` | List of BEIR dataset names to process |
| `embeddings.model_name` | SentenceTransformer model for dense retrieval |
| `bm25` | k1, b, use_stemming, stemmer_lang |
| `benchmark` | top_k, ndcg_k, rrf.k |
| `routing_features` | Feature computation parameters (overlap_k, feature_stat_k, etc.) |
| `xgboost_best` | Global best XGBoost hyperparameters (from model selection) |
| `xgboost_params_grid` | Grid definition for per-dataset search |
| `xgboost_per_dataset` | Best hyperparameters per dataset (from per-dataset search) |
| `ablation_study` | n_folds, train_fraction, n_jobs for ablation |

---

## Documentation (`docs/`)

| File | Contents |
|------|---------|
| `docs/routing_features.md` | All 15 routing feature definitions with formulas and group assignments. Includes the soft-label derivation formula. |
| `docs/model_selection_results.md` | Model grid search results. Top-10 configurations, per-dataset scores for rank-1, and interpretation of which hyperparameters matter. |
| `docs/ablation_study_results.md` | Full leave-one-feature-out and leave-one-group-out tables. Rationale for removing `query_length`. Full vs smaller model comparison. |
| `docs/per_dataset_params_results.md` | Per-dataset grid search results: best hyperparameters, CV and test NDCG@10 per dataset, and interpretation of why parameters differ across datasets. |
| `docs/retrieval_explainability_results.md` | Final benchmark results across all four retrieval methods. Per-dataset analysis. SHAP plot interpretation per dataset and cross-dataset feature importance patterns. |

---

## Data layout

```
data/
  datasets/           # Raw BEIR downloads (never deleted by cleanup_hpc.py)
  processed_data/
    <model_short_name>/
      <dataset>/
        corpus.jsonl            # Exported corpus
        queries.jsonl           # Exported queries
        qrels.tsv               # Relevance judgements
        corpus_embeddings.pt    # Dense corpus embeddings (expensive)
        corpus_ids.pkl
        query_vectors.pt        # Dense query embeddings
        query_ids.pkl
        tokenized_corpus_stem_*.jsonl
        tokenized_queries_stem_*.jsonl
        bm25_k1_*_b_*_stem_*.pkl
        bm25_k1_*_b_*_stem_*_topk_*_results.pkl
        features_labels_<hash>.pkl
  results/
    bm25_optimization_macro.csv
    bm25_optimization_best.json
    model_grid_search_top100.csv
    ablation_study.csv / .png
    full_vs_smaller_ablation.csv / .png
    per_dataset_best_params.csv
    retrieval_comparison.csv / .png
    shap_<dataset>.png
```
