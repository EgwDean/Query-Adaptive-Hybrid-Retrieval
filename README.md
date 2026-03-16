# RAG-LLM Hybrid Retrieval Benchmark

This repository benchmarks sparse, dense, and hybrid retrieval on BEIR datasets with a two-phase workflow:

1. Preprocess and cache all heavy artifacts once.
2. Run retrieval and evaluation repeatedly on the cached data.

The current hybrid logic includes a dynamic, statistically normalized wRRF pipeline with grid search over query-cleaning and routing parameters.

## Methods evaluated

For each configured dataset, retrieval evaluates:

1. BM25 Only
2. Dense Only
3. RRF (static)
4. Dynamic JSD-wRRF (best over grid)
5. Dynamic KLD-wRRF (best over grid)
6. Dynamic CE-wRRF (best over grid)

All methods are scored with NDCG@k (default k=10).

## Repository layout

```text
config.yaml
requirements.txt
src/
  pipeline.py      # legacy reference pipeline (kept for comparison)
  utils.py
  donwload.py      # dataset downloader
  preproces.py     # preprocessing/cache builder
  retrieval.py     # retrieval + evaluation on cached artifacts
  correct.py       # migrate old cached files from results -> processed_data
  fix.py           # one-time stale-cache cleanup for new retrieval logic
data/
  datasets/        # raw BEIR datasets
  processed_data/  # reusable cache artifacts per model/dataset
  results/         # model-level summaries and charts
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Edit config.yaml to choose:

1. Active datasets (all uncommented entries are processed/evaluated).
2. Embedding model and batch size.
3. Retrieval/evaluation settings.
4. Dynamic wRRF grid values.

Important sections:

```yaml
datasets:
  - scifact
  - nfcorpus

paths:
  datasets_folder: "data/datasets"
  results_folder: "data/results"
  processed_folder: "data/processed_data"

embeddings:
  model_name: "BAAI/bge-m3"
  batch_size: 64

benchmark:
  top_k: 100
  ndcg_k: 10
  rrf:
    k: 60
  dynamic_wrrf:
    max_df_values: [0.5, 0.8]
    k_values: [1.0, 2.0, 3.0]
```

## Run order

1. Download datasets

```bash
python src/donwload.py
```

2. Build caches (corpus/query files, BM25 index, frequency index, embeddings)

```bash
python src/preproces.py
```

3. Optional migration for previously generated caches

```bash
python src/correct.py
```

4. Optional one-time stale cleanup before new retrieval logic

```bash
python src/fix.py
```

5. Run retrieval + evaluation

```bash
python src/retrieval.py
```

## Dynamic wRRF details

The retrieval stage applies the following per dataset:

1. Query cleaning:
   - Remove English stopwords.
   - Remove high document-frequency terms above max_df.
   - If query becomes empty, force alpha=0.0.
2. Metric computation on cleaned queries:
   - KLD, JSD, and length-normalized CE.
3. Standardization:
   - Convert metric values to z-scores per dataset.
4. Routing:
   - alpha = 1 / (1 + exp(-k * z)).
5. Fusion:
   - Min-max normalize BM25 and Dense scores first.
   - Score = alpha * BM25 + (1 - alpha) * Dense.
6. Lightweight grid search:
   - max_df in [0.5, 0.8]
   - k in [1.0, 2.0, 3.0]

## Outputs

Retrieval writes model-level summary files under data/results/<model>/:

1. summary_ndcg.csv
2. best_dynamic_params.csv
3. summary_ndcg.png
4. retrieval_timing.csv

Processed artifacts are cached under data/processed_data/<model>/<dataset>/ and reused across runs.

## Notes

1. CPU and CUDA are both supported.
2. Preprocessing uses multiprocessing for corpus tokenization.
3. Dense retrieval uses chunking controls from config for memory safety.
