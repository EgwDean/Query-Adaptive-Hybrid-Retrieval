# Query-Adaptive Hybrid Retrieval

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20073287.svg)](https://doi.org/10.5281/zenodo.20073287)

> **Diploma Thesis — Computer Engineering and Informatics Department (CEID), University of Patras**
>
> **Author:** Konstantinos Anastasopoulos &lt;up1093320@ac.upatras.gr&gt;
> **Supervisor:** Prof. Spyridon Sioutas &lt;ssioutas@ceid.upatras.gr&gt;, Computer Engineering and Informatics Department (CEID), University of Patras
> **Year:** 2026

A query-adaptive hybrid retrieval system that learns, per query, the
optimal weight α between BM25 and BGE-M3 dense retrieval and merges the
two ranked lists via weighted Reciprocal Rank Fusion. Three router
variants — a 16-feature LightGBM, a 1024-dim BGE-M3 embedding XGBoost,
and an SVR-based Mixture-of-Experts meta-learner — are trained and
compared against BM25, dense, and a Static-RRF (α = 0.5) baseline on
five BEIR datasets. See [`docs/results.md`](docs/results.md) for the
full empirical analysis.

---

## Problem

Standard retrieval systems apply a fixed strategy to every query.  For some
queries BM25 (sparse, lexical matching) is the better retriever; for others
a dense embedding model is.  A query-adaptive system learns to predict, for
each individual query, how much to trust BM25 vs. dense retrieval.

Different queries have fundamentally different needs:
- A clinical keyword query like *"COVID-19 ACE2 receptor binding"* needs
  exact term matching — BM25.
- A paraphrase-heavy question like *"what makes chocolate pleasurable"*
  needs semantic understanding — dense retrieval.
- Most queries fall somewhere in between.

Fixed-weight fusion (Static RRF, α = 0.5) ignores this variability.  This
project learns α per query from observable query signals.

---

## Algorithm

For each query q the system:

1. **Sparse retrieval** — BM25 (Okapi BM25, tuned k1/b/stemming) produces a
   ranked list of up to 100 documents.
2. **Dense retrieval** — `BAAI/bge-m3` (1024-dim) encodes the query and
   retrieves the top-100 by cosine similarity.
3. **Router** — predicts α(q) ∈ [0, 1].  Three router variants are trained:
   - **Weak router:** 16 hand-crafted features (query length, IDF statistics,
     retriever confidence margins, retriever agreement, score entropy) →
     LightGBM regressor (best of a 66-combination grid).
   - **Strong router:** 1024-dimensional BGE-M3 query embedding → XGBoost
     regressor (best of a 108-combination grid).
   - **MoE meta-learner:** takes the weak and strong predictions as input →
     SVR regressor (best of a 77-combination grid).
4. **Weighted RRF fusion** — the two ranked lists are merged:

```
score(d) = α · 1/(60 + rank_bm25(d))  +  (1−α) · 1/(60 + rank_dense(d))
```

α = 1 → pure BM25; α = 0 → pure Dense; α = 0.5 → Static RRF (baseline).

The ground-truth α per query is found by brute-force search over
α ∈ {0.00, 0.01, …, 1.00} maximising NDCG@100 against gold relevance
judgements (oracle alpha).

---

## Key results

Macro NDCG@100, MRR@100, Recall@100 across **5 BEIR datasets**, **225
held-out test queries** (45 per dataset):

| Method | NDCG@100 | MRR@100 | Recall@100 |
|--------|----------|---------|------------|
| BM25 | 0.327 | 0.362 | 0.513 |
| Dense (BGE-M3) | 0.420 | 0.449 | 0.644 |
| Static RRF (α = 0.5) | 0.404 | 0.431 | 0.641 |
| wRRF Weak (LightGBM) | 0.422 | 0.457 | 0.648 |
| wRRF Strong (XGBoost) | 0.424 | 0.453 | **0.651** |
| **wRRF MoE (SVR)** | **0.426** | **0.462** | 0.647 |
| Oracle ceiling | 0.487 | — | — |

All three adaptive methods **significantly outperform BM25** (paired
t-test, p < 1e-7, Cohen's d ≈ 0.4) on NDCG, MRR, and Recall@100. They
also significantly beat Static RRF on NDCG (p ≤ 0.029, n = 225). Each
method pair is treated as an independent, pre-specified comparison;
no multiple-comparison correction is applied.

The cheap 16-feature weak router performs **statistically indistinguishably**
from the expensive 1024-dim strong router (p = 0.65, Δ = 0.002 NDCG) and
from the MoE ensemble (p = 0.52, Δ = 0.004) — at a fraction of the
inference cost (~1 ms router overhead vs. 13 ms dense + ~120 ms BM25).

A cross-encoder reranker only meaningfully helps **BM25-only** retrieval
(+0.037 NDCG, p = 0.002); for Dense, Static RRF, and all adaptive
methods the rerank delta is statistically null. Adaptive wRRF is therefore
a complete first-stage solution — no downstream reranker required.

See [`docs/results.md`](docs/results.md) for the full per-dataset breakdown,
significance tables, ablation analysis, reranking analysis, and latency
benchmarks.

---

## Datasets

Five BEIR datasets: `scifact`, `nfcorpus`, `arguana`, `fiqa`, `scidocs`.
300 queries per dataset are sampled (1 500 total), split
70 % train / 15 % dev / 15 % test per dataset (1 050 / 225 / 225).

---

## Setup

```bash
# Clone the repository
git clone https://github.com/EgwDean/Query-Adaptive-Hybrid-Retrieval.git
cd Query-Adaptive-Hybrid-Retrieval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Download NLTK stop words used by the tokeniser
python -c "import nltk; nltk.download('stopwords')"
```

A GPU with ≥ 8 GB VRAM is recommended.  The pipeline falls back to CPU if
no GPU is available (embedding and reranking steps will be significantly
slower).

---

## Running the pipeline

```bash
python -m src.pipeline
```

The pipeline runs all 25 steps in sequence, caching every intermediate result
to disk.  It prints a per-step summary to stdout.  All outputs are written to
`data/results/` and `data/models/`.

The three trained routers from the published run are saved under
[`data/models/`](data/models/) — `weak_model.pkl`, `strong_model.pkl`, and
`moe_model.pkl` — so you can load them directly without rerunning the
pipeline. Each pickle is a dict containing the trained estimator, the
fitted `StandardScaler`, and the feature column list used at training time.

**Resuming after interruption:** re-run the same command.  Every step checks
whether its outputs already exist before running and skips if so.

**Running a specific step range:**
```bash
python -m src.pipeline --start 6 --end 8
```

**Forcing recomputation:** delete the step's output files and re-run.

---

## Quickstart — load a saved router and predict α

The three trained routers in [`data/models/`](data/models/) are dictionaries
containing `model`, `scaler`, `feature_cols`, and `feature_names`. The
snippet below loads the **weak router** and predicts α for a single
feature vector. The 16 feature values must be supplied in the order given
by `feature_names` (and computed against the same BM25 + dense top-100
lists used at training time — see
[`docs/pipeline_steps.md`](docs/pipeline_steps.md) §STEP 5).

```python
import joblib, numpy as np

bundle = joblib.load("data/models/weak_model.pkl")
model, scaler, feat_cols = bundle["model"], bundle["scaler"], bundle["feature_cols"]
print("Feature order:", bundle["feature_names"])

# Replace the zeros below with the 16 features for your query, in the
# order printed above. See docs/pipeline_steps.md §STEP 5 for definitions
# (query_length, stopword_ratio, has_question_word, average_idf, ...).
features = np.zeros((1, 16), dtype=np.float32)

X_scaled = scaler.transform(features[:, feat_cols])
alpha = float(np.clip(model.predict(X_scaled)[0], 0.0, 1.0))
print(f"Predicted alpha = {alpha:.3f}")
```

For an end-to-end example that runs BM25, dense retrieval, computes the
16 features, predicts α, and produces the wRRF-fused ranked list, see
the test-set evaluation block in
[`src/pipeline.py`](src/pipeline.py) (Step 8 — `step_08_weak_retrieval_comparison`).
The strong and MoE routers load the same way; the strong bundle expects
a 1024-dim BGE-M3 query embedding as input, the MoE bundle expects
`[α_weak, α_strong, |α_weak − α_strong|]`.

---

## Source files

| File | Purpose |
|------|---------|
| `src/pipeline.py` | Main entry point — 25-step end-to-end pipeline |
| `src/utils.py` | Shared helpers: embedder, BM25 builder, frequency indices, plots, normalisation |
| `config.yaml` | All hyperparameters (edit here, not in code) |

---

## Configuration (`config.yaml`)

Key sections:

| Section | Controls |
|---------|----------|
| `datasets` | List of BEIR dataset names |
| `embeddings.model_name` | SentenceTransformer model (`BAAI/bge-m3`) |
| `bm25_grid_search` | k1, b, use_stemming search space |
| `benchmark` | top_k, ndcg_k, rrf.k, bootstrap settings |
| `sampling` | n_queries_per_dataset, test_fraction, dev_fraction, random_seed, cv_n_folds |
| `routing_features` | Feature computation parameters |
| `weak_model_grid_search` | Model families and hyperparameter grids for weak router |
| `strong_model_grid_search` | Model families and hyperparameter grids for strong router |
| `moe_grid_search` | Model families and hyperparameter grids for MoE |
| `reranker` | Cross-encoder model name |
| `significance_test` | Alpha threshold for t-tests |

---

## Documentation

| File | Contents |
|------|---------|
| [`docs/pipeline_steps.md`](docs/pipeline_steps.md) | Detailed description of all 25 pipeline steps: feature definitions, normalisation protocol, model training, split details, output file layout |
| [`docs/results.md`](docs/results.md) | Complete experimental analysis: NDCG/MRR/Recall tables, significance tests, feature ablation, reranking analysis, latency benchmarks |

---

## Data layout

```
data/
├── datasets/                      # Raw BEIR downloads (~1.5 GB)
├── processed/<embedding_model>/   # Per-dataset preprocessed artefacts
│   └── <dataset>/
│       ├── corpus_embeddings.pt   # Dense corpus embeddings (GPU-computed)
│       ├── query_vectors.pt
│       ├── bm25_k1_*_b_*_stem_*.pkl
│       └── ...
├── results/                       # All CSV / JSON / PNG outputs
│   ├── bm25_best_params.json
│   ├── oracle_ndcg_per_dataset.json
│   ├── moe_retrieval_comparison.csv
│   ├── significance_tests.csv
│   ├── latency.csv
│   └── ...
└── models/
    ├── weak_model.pkl
    ├── strong_model.pkl
    └── moe_model.pkl
```

---

## Reproducibility

All random seeds derive from `sampling.random_seed = 42`.  The 1 500-query
selection and 70/15/15 split are cached to `data/results/merged_qids.json`
and `data/results/merged_split.json` and never recomputed once written.

Hardware used for published results: AMD Ryzen 9 5950X, NVIDIA RTX 4090
(24 GB), 62.7 GB RAM, Ubuntu 24.04, CUDA 13.0, PyTorch 2.11.

---

## Citing this work

If you use this code or its results, please cite the repository via the
metadata in [`CITATION.cff`](CITATION.cff). GitHub renders a "Cite this
repository" button on the project page that returns BibTeX and APA forms
generated from that file.

The repository is archived on Zenodo. Please cite the **concept DOI**
[10.5281/zenodo.20073287](https://doi.org/10.5281/zenodo.20073287) — this
identifier always resolves to the latest archived version. Per-version
DOIs (e.g. `10.5281/zenodo.20073721` for v1.0.1) are listed on the
Zenodo record under "View all versions" if you need to pin to a specific
snapshot.

BibTeX:

```bibtex
@software{anastasopoulos_qahr_2026,
  author    = {Anastasopoulos, Konstantinos},
  title     = {Query-Adaptive Hybrid Retrieval},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20073287},
  url       = {https://github.com/EgwDean/Query-Adaptive-Hybrid-Retrieval}
}
```
