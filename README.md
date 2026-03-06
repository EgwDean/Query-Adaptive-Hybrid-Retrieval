# RAG-LLM Hybrid Retrieval Benchmark

A benchmarking pipeline that compares **sparse (BM25)**, **dense (sentence-transformer)**, and several **hybrid fusion** strategies on [BEIR](https://github.com/beir-cellar/beir) datasets.  The fusion methods use information-theoretic divergences (JSD, KLD, cross-entropy) to dynamically weight BM25 and dense scores per query.

## Retrieval methods evaluated

| # | Method | Description |
|---|--------|-------------|
| 1 | BM25 Only | Classical bag-of-words retrieval |
| 2 | Dense Only | Cosine similarity over sentence embeddings |
| 3 | Naive RRF | Reciprocal Rank Fusion with equal weights |
| 4 | JSD (Linear) | Weighted RRF, alpha = Jensen-Shannon divergence |
| 5 | JSD (Sigmoid) | Same, but alpha passed through a sigmoid |
| 6 | KLD (0-1 Norm) | Weighted RRF, alpha = min-max normalised KL divergence |
| 7 | KLD (0-1 + Sigmoid) | Same, plus sigmoid |
| 8 | Cross-Entropy (0-1 Norm) | Weighted RRF, alpha = min-max normalised cross-entropy |
| 9 | Cross-Entropy (0-1 + Sigmoid) | Same, plus sigmoid |

All methods are evaluated with **NDCG@10**.

---

## Repository layout

```
config.yaml             # all tunable parameters
requirements.txt        # Python dependencies
src/
    pipeline.py         # single entry point -- run this
    utils.py            # shared I/O and data-loading helpers
data/
    datasets/           # BEIR datasets (auto-downloaded)
    results/            # generated per model + dataset
        <model>/
            <dataset>/
                results.csv           # NDCG@10 scores for all 9 methods
                results.png           # bar chart of the scores
                timing.csv            # per-step wall-clock times
                corpus.jsonl          # local copy of the corpus
                queries.jsonl         # local copy of queries
                qrels.tsv             # relevance judgments
                tokenized_corpus.jsonl
                bm25_index.pkl
                bm25_doc_ids.pkl
                word_freq_index.pkl
                corpus_embeddings.pt
                corpus_ids.pkl
                bm25_results.pkl
                query_vectors.pt
                query_ids.pkl
                dense_results.pkl
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pyyaml` | Read config.yaml |
| `tqdm` | Progress bars |
| `beir` | Download + load BEIR datasets |
| `nltk` | Snowball stemmer for BM25 tokenization |
| `rank_bm25` | BM25Okapi implementation |
| `sentence-transformers` | Dense embedding models |
| `torch` | Tensor operations, GPU acceleration |
| `numpy` | Numerical computations |
| `matplotlib` | Bar-chart output |

Python **3.10+** is required.  A CUDA-capable GPU is strongly recommended for the dense encoding steps.

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd RAG-LLM

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit **config.yaml** to choose datasets, the embedding model, and other parameters.  The important sections:

```yaml
# Process each dataset independently (recommended for benchmarking)
merge: false

# Uncomment the datasets you want to run
datasets:
  - scifact
  #- nfcorpus
  #- arguana

# The embedding model to use
embeddings:
  model_name: "BAAI/bge-m3"
  batch_size: 32

# Dense search chunking
dense_search:
  query_chunk_size: 100
  corpus_chunk_size: 50000

# Benchmark parameters
benchmark:
  top_k: 100
  ndcg_k: 10
  rrf:
    k: 60                # RRF damping constant
  sigmoid:
    center: 0.5          # sigmoid center for WRRF alpha
    slope: 10            # sigmoid slope for WRRF alpha
  smoothing:
    laplace_alpha: 1     # Laplace smoothing for corpus distribution
```

See the comments inside config.yaml for the full list of BEIR datasets and their sizes.

---

## Running the pipeline

```bash
python src/pipeline.py
```

Or with a custom config:

```bash
python src/pipeline.py --config my_config.yaml
```

### What happens

1. Each dataset listed in config.yaml is downloaded (if not already present).
2. The corpus is preprocessed (stemmed and tokenized for BM25).
3. A BM25 index and a corpus-wide word-frequency index are built in a single pass.
4. The corpus is encoded with the sentence-transformer model.
5. BM25 and dense retrievals are performed.
6. All 9 fusion methods are evaluated and scored with NDCG@10.
7. A CSV table, a bar chart, and a timing CSV are saved to `data/results/<model>/<dataset>/`.

Every intermediate artifact (index, embeddings, retrieval results) is cached on disk.  If you re-run the pipeline, completed steps are detected and skipped automatically.

### Example output

```
Device : cuda
Model  : BAAI/bge-m3
Datasets (1): scifact
Mode   : PER-DATASET

Loading embedding model ...
  Model loaded in 4.2s
  CPU threads (torch)     : 12
  CPU inter-op threads    : 1
  Workers (preprocessing) : 11

============================================================
  Dataset: scifact
  Output : data/results/bge-m3/scifact
============================================================

[Step 1/10] Downloading / verifying dataset ...
[Step 2/10] Loading BEIR data and writing local copies ...
  Split: test | Corpus: 5,084 | Queries: 300
[Step 3/10] Preprocessing corpus (stemming + tokenization) ...
  Preprocessing corpus: 100%|████████████| 10/10 [00:00<00:00]
[Step 4/10] Building BM25 index & word frequency index ...
[Step 5/10] Encoding corpus with embedding model ...
  Encoding corpus: 100%|████████████| 80/80 [00:12<00:00]
[Step 6/10] Running BM25 retrieval ...
[Step 7/10] Encoding queries ...
[Step 8/10] Running dense retrieval ...
[Step 9/10] Evaluating fusion methods ...

  Method                           NDCG@10
  -------------------------------- --------
  BM25 Only                          0.6653
  Dense Only                         0.7102
  Naive RRF                          0.7214
  JSD (Linear)                       0.7189
  JSD (Sigmoid)                      0.7201
  KLD (0-1 Norm)                     0.7156
  KLD (0-1 + Sigmoid)                0.7178
  Cross-Entropy (0-1 Norm)           0.7134
  Cross-Entropy (0-1 + Sigmoid)      0.7167

[Step 10/10] Saving results ...
  Results saved to data/results/bge-m3/scifact/results.csv
  Chart saved to data/results/bge-m3/scifact/results.png
  Timing saved to data/results/bge-m3/scifact/timing.csv

  Step                                          Time (s)
  --------------------------------------------- ----------
  Step 1: Download / verify dataset                  0.01
  Step 2: Load / write corpus, queries, qrels        1.23
  Step 3: Preprocessing (stem + tokenize)            0.45
  Step 4: BM25 + word-freq index                     0.82
  Step 5: Corpus embeddings                         12.34
  Step 6: BM25 retrieval                             0.67
  Step 7: Query embeddings                           0.31
  Step 8: Dense retrieval                            0.18
  Step 9: Evaluation (all fusion methods)            0.05
  Step 10: Save results                              0.15
  --------------------------------------------- ----------
  Total                                             16.21

  Finished scifact.

============================================================
  All done.
============================================================
```

*(The NDCG values and timings above are illustrative.)*

---

## Output files

Each run produces three result files in `data/results/<model>/<dataset>/`:

| File | Contents |
|------|----------|
| `results.csv` | NDCG@10 score for each of the 9 retrieval methods |
| `results.png` | Horizontal bar chart of the scores |
| `timing.csv` | Wall-clock time (seconds) for each pipeline step, plus total |

---

## Resource requirements

| Dataset | Corpus size | RAM (approx.) | GPU VRAM |
|---------|-------------|---------------|----------|
| scifact | 5K docs | ~1 GB | ~2 GB |
| nfcorpus | 3.6K docs | ~1 GB | ~2 GB |
| arguana | 8.7K docs | ~1 GB | ~2 GB |
| fiqa | 57K docs | ~2 GB | ~3 GB |
| scidocs | 25K docs | ~2 GB | ~3 GB |
| nq | 2.7M docs | ~20 GB | ~6 GB |
| msmarco | 8.8M docs | ~50 GB | ~8 GB |

Estimates assume BAAI/bge-m3.  Using `all-MiniLM-L6-v2` uses roughly 3× less VRAM and RAM for embeddings.

---
