# MLP Per-Dataset Hyperparameter Search Results

## What was done

A per-dataset hyperparameter grid search was run for a PyTorch MLP router using
query embedding vectors (~1024 dims, BAAI/bge-m3) as input — the same
representation used in the strong-signal XGBoost search.

**Motivation**: The sklearn MLP in the strong-signal model selection grid search
was missing the three ingredients that make MLPs competitive on high-dimensional
input: Dropout, BatchNorm, and proper learning rate scheduling. This experiment
tests whether a fully-specified PyTorch MLP can match or surpass XGBoost when
those ingredients are present.

**Model architecture** (`AlphaMLP`):

```
Linear(1024 → h₁)  →  [BatchNorm1d]  →  GELU  →  Dropout
  ...
Linear(hₙ₋₁ → hₙ)  →  [BatchNorm1d]  →  GELU  →  Dropout
Linear(hₙ → 1)  →  Sigmoid
```

Training: Adam optimizer, cosine LR annealing over 200 epochs, MSE loss,
batch size 64.

**Split strategy**: identical to all other scripts — 85% train+dev (10-fold
80/20 CV for hyperparameter selection), 15% test (evaluated once with best params,
same held-out queries as XGBoost experiments).

**Grid**: 96 combinations per dataset (4 architectures × 3 dropouts × 2
BatchNorm × 2 LRs × 2 weight decays).

Script: `src/mlp_params_grid_search.py`
Output: `data/results/mlp_per_dataset_best_params.csv`
Config: `mlp_params_grid` in `config.yaml`

---

## Results

| Dataset  | hidden_sizes | dropout | batchnorm | learning_rate | weight_decay | CV NDCG@10 | Test NDCG@10 |
|----------|-------------|---------|-----------|---------------|-------------|------------|--------------|
| scifact  | 256-64 | 0.1 | True  | 0.001  | 0.0001 | 0.6726 | 0.6396 |
| nfcorpus | 128    | 0.3 | True  | 0.0001 | 0.001  | 0.2928 | 0.3419 |
| arguana  | 256    | 0.1 | True  | 0.001  | 0.0001 | 0.3747 | 0.3608 |
| fiqa     | 128    | 0.5 | False | 0.001  | 0.0001 | 0.3771 | 0.4392 |
| scidocs  | 512-256| 0.1 | True  | 0.0001 | 0.0001 | 0.1752 | 0.1587 |
| **MACRO**|        |     |       |        |        | **0.3786** | **0.3880** |

---

## Comparison with XGBoost routers

| Dataset  | MLP test | XGB strong-signal | XGB weak-signal | MLP vs XGB (strong) |
|----------|----------|-------------------|-----------------|---------------------|
| scifact  | 0.6396   | 0.6559            | 0.6628          | −0.0163             |
| nfcorpus | 0.3419   | 0.3408            | 0.3538          | +0.0011             |
| arguana  | 0.3608   | 0.3665            | 0.3629          | −0.0057             |
| fiqa     | 0.4392   | 0.4733            | 0.4679          | −0.0341             |
| scidocs  | 0.1587   | 0.1622            | 0.1567          | −0.0035             |
| **MACRO**| **0.3880**| **0.3997**       | **0.4008**      | **−0.0117**         |

XGBoost outperforms the MLP on 4 of 5 datasets. The MLP is approximately tied on
nfcorpus (+0.001). The largest gap is fiqa (−0.034), where XGBoost's routing
is particularly precise. Despite fixing the sklearn MLP's missing ingredients,
XGBoost remains the stronger router on this task and data regime.

---

## Observations

### Hyperparameter patterns

**BatchNorm=True in 4 of 5 datasets.** The single exception is fiqa. This
confirms that normalising across the 1024 embedding dimensions improves gradient
flow and acts as an effective regularizer — the same reasoning that motivated
including it in the grid. The fiqa exception is discussed below.

**All selected architectures are small.** The largest selected architecture is
[512, 256] (scidocs); the others are [256, 64], [128], and [256]. No dataset
benefited from the widest architectures available. With 200–1000 training
queries, smaller models generalise better. This directly validates the concern
raised before designing the grid: the small-data regime limits neural model
capacity.

**Dropout varies by routing difficulty.** A clear pattern emerges:
- `dropout=0.1` — scifact, arguana, scidocs: datasets where routing provides a
  moderate signal; light regularisation is sufficient.
- `dropout=0.3` — nfcorpus: the hardest routing case (38 relevant docs per query,
  weak routing signal); moderate regularisation needed.
- `dropout=0.5` — fiqa: the easiest routing case (BM25 is nearly useless), but
  the model needs heavy dropout to prevent memorising specific training queries
  and generalise the "always prefer dense" rule uniformly.

**Learning rate splits cleanly.** lr=0.001 for scifact, arguana, and fiqa —
datasets where the routing signal is relatively consistent. lr=0.0001 for nfcorpus
and scidocs — the two datasets with the weakest and most ambiguous routing
signals, where smaller gradient updates prevent the model from chasing noise.

**nfcorpus gets maximum regularisation.** The winning config for nfcorpus uses
every available regularisation mechanism simultaneously: dropout=0.3, wd=0.001
(10× higher than all others), lr=0.0001, BatchNorm=True. This is consistent
with nfcorpus being the dataset where the optimal alpha is least extreme — the
routing problem has a low signal-to-noise ratio and the model needs to be strongly
constrained to avoid overfit.

### Why fiqa rejects BatchNorm

fiqa selects BatchNorm=False, dropout=0.5, hidden=[128]. This is the most
extreme departure from the other datasets. fiqa's routing task is nearly
constant: BM25 is deeply ineffective (NDCG@10 = 0.16) so the correct α is
near zero for virtually every query. The model only needs to learn a very
simple function, and a large hidden dimension with BatchNorm may over-parameterise
it. A tiny network (128 hidden units) with heavy dropout provides the right
inductive bias: "output a small constant for all inputs."

### CV vs test discrepancy

**scifact** shows the largest negative gap (CV=0.673 → test=0.640, −0.033).
This is the biggest CV-to-test drop across all experiments, suggesting the best
config selected by CV generalises poorly to the 15% hold-out. With only ~45 test
queries (15% of 300), high variance is expected, but this gap is large enough
to suggest mild overfitting to the training distribution.

**nfcorpus and fiqa** again show test > CV (the small-sample variance pattern
seen in every other script in this project). With 48 and 97 test queries
respectively, these fluctuations are expected and the CV score is the more
reliable generalisation estimate.

---

## Interpretation: why XGBoost outperforms

The result contradicts the "neural model for neural data" intuition. Four
possible explanations:

**1. Small-data regime.** With 200–1000 training queries per dataset, XGBoost's
tree-based inductive biases (axis-aligned decision boundaries, built-in subsampling,
leaf regularisation) fit the data more efficiently than gradient descent on a
neural network. Neural networks typically need more data to offset their lack of
these structural priors.

**2. Embedding subsampling.** XGBoost's `colsample_bytree` randomly selects a
subset of the 1024 embedding dimensions at each split, providing implicit
feature-level regularisation that is well-matched to the high-dimensional,
redundant structure of bge-m3 embeddings. Dropout operates on neuron activations,
not input features, and does not provide the same dimension-level diversity.

**3. Routing function structure.** The routing function — "for this query, which
retriever wins?" — may be better approximated by a small number of axis-aligned
conditions on specific embedding dimensions than by the smooth, dense projections
that MLPs learn. Tree stumps can say "if embedding dim 347 > 0.8, use dense" in
a single split; an MLP must compose many smooth projections to achieve the same.

**4. Training instability.** CV workers use unseed model initialisation to avoid
thread-race conditions on the global PyTorch RNG. This introduces fold-level
variance that is absent in the XGBoost CV (which is fully deterministic). Some
"winning" configurations may have been selected partly due to favourable random
initialisation rather than genuine superiority.

---

## Summary

The properly-specified PyTorch MLP (with Dropout, BatchNorm, Adam, cosine LR)
achieves a macro test NDCG@10 of **0.3880**, compared to **0.3997** for XGBoost
on the same embedding input — a gap of −0.012. This is a meaningful but not
large difference. The result does not entirely close the door on neural routers:
the gap is concentrated on fiqa (−0.034), where the XGBoost router is near-optimal.
On nfcorpus the MLP is essentially tied. A larger training set (cross-dataset
joint training) or a more powerful architecture may close the remaining gap.
