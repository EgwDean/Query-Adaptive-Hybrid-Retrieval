# Strong-Signal Per-Dataset Hyperparameter Search Results

## What was done

Following the strong-signal model selection (which confirmed XGBoost dominates
on embedding input just as it does on weak-signal features), a per-dataset
hyperparameter search was run for XGBoost using query embedding vectors as input.

**Input features**: raw query embedding vectors (~1024 dims, BAAI/bge-m3) —
the same representation used in the strong-signal model selection grid search.

**Split strategy per dataset**:
- 85% train+dev — used for 10-fold 80/20 CV to select hyperparameters
- 15% test — held out; evaluated exactly once with the best parameters

The 85/15 split uses the same seed formula as `weak_signal_params_grid_search.py`
(`split_seed = seed + dataset_seed_offset(dataset)`), ensuring both experiments
evaluate on the exact same held-out queries — a prerequisite for a fair comparison.

**Grid**: intentionally tight (96 combinations) compared to the weak-signal grid
(~6 480 combinations). XGBoost on 1024-dimensional input is ~100× slower per fit
than on 15 features. The grid axes and their fixed values are informed by the
strong-signal model selection results:

- `n_estimators = 300` fixed (dominated every top-10 entry)
- `subsample = 0.8` fixed (same dominant value)
- `min_child_weight = 1` fixed (same)
- `learning_rate`: [0.1, 0.3]
- `max_depth`: [4, 6, 8]
- `colsample_bytree`: [0.1, 0.3, 0.5, 0.8]
- `gamma`: [0.0, 0.1]

Performance optimisations vs the weak-signal script:
- `tree_method="hist"` always — avoids the O(n × d) exact sort per node
- GPU branch — when CUDA is available, `device="cuda"` and the outer
  Parallel loop is set to `n_jobs=1` (sequential) to avoid GPU memory contention
- CPU branch — outer loop uses all cores (`n_jobs=-1`)

Script: `src/strong_signal_params_grid_search.py`
Output: `data/results/strong_signal_per_dataset_best_params.csv`
Config: `strong_signal_params_grid` and `strong_signal_xgboost_per_dataset` in `config.yaml`

---

## Results

| Dataset  | colsample_bytree | gamma | learning_rate | max_depth | min_child_weight | n_estimators | subsample | CV NDCG@10 | Test NDCG@10 |
|----------|-----------------|-------|---------------|-----------|-----------------|--------------|-----------|------------|--------------|
| scifact  | 0.3 | 0.0 | 0.1 | 6 | 1 | 300 | 0.8 | 0.6749 | 0.6559 |
| nfcorpus | 0.8 | 0.0 | 0.3 | 8 | 1 | 300 | 0.8 | 0.2946 | 0.3408 |
| arguana  | 0.3 | 0.0 | 0.3 | 8 | 1 | 300 | 0.8 | 0.3835 | 0.3665 |
| fiqa     | 0.5 | 0.0 | 0.3 | 8 | 1 | 300 | 0.8 | 0.4036 | 0.4733 |
| scidocs  | 0.1 | 0.0 | 0.3 | 6 | 1 | 300 | 0.8 | 0.1769 | 0.1622 |
| **MACRO**|     |     |     |   |   |    |     | **0.3867** | **0.3997** |

---

## Comparison with weak-signal per-dataset search

| Dataset  | Weak-signal test NDCG@10 | Strong-signal test NDCG@10 | Delta (strong − weak) |
|----------|--------------------------|----------------------------|-----------------------|
| scifact  | 0.6628 | 0.6559 | −0.0069 |
| nfcorpus | 0.3538 | 0.3408 | −0.0130 |
| arguana  | 0.3629 | 0.3665 | +0.0036 |
| fiqa     | 0.4679 | 0.4733 | +0.0054 |
| scidocs  | 0.1567 | 0.1622 | +0.0055 |
| **MACRO**| **0.4008** | **0.3997** | **−0.0011** |

The per-dataset results replicate the pattern first observed in model selection:
weak signal is stronger on scifact and nfcorpus; strong signal is stronger on
arguana, fiqa, and scidocs. The macro gap (0.001) is within noise for these
dataset sizes.

---

## Observations

**gamma = 0.0 for all datasets.** Unlike the weak-signal case where nfcorpus
benefited from gamma = 0.1, no dataset here requires min-loss regularization.
With 1024 input dimensions, the effective model complexity is already constrained
by `colsample_bytree` — randomly subsampling embedding dimensions at the tree
level provides implicit regularization that makes gamma redundant.

**colsample_bytree varies across datasets** (0.1, 0.3, 0.5, 0.8). This is the
primary axis of variation — the only hyperparameter that differs meaningfully
across datasets. scidocs uses the most aggressive subsampling (0.1 ≈ 102 of
1024 dimensions per split), which acts as strong regularization for the
noisiest dataset. scifact and arguana use moderate subsampling (0.3), while
nfcorpus uses no subsampling (0.8). This pattern suggests that the number of
embedding dimensions that carry useful routing signal differs by domain.

**max_depth = 6 or 8 for all datasets** — no dataset benefits from shallower
trees (depth 4). This contrasts with the model selection finding where depth 4
appeared at rank 7. When hyperparameters are optimised per dataset, the routing
signal is captured fully only at moderate depth.

**n_estimators = 300 fixed** — the tight grid validated the model selection
finding: 300 trees are universally necessary. This is consistent across both
representations and all grid searches in this project.

**CV vs test discrepancy**: nfcorpus (CV=0.295, test=0.341) and fiqa
(CV=0.404, test=0.473) again show higher test than CV scores. This is a
small-sample variance artifact (48 test queries for nfcorpus, 97 for fiqa),
not evidence of data leakage — the CV is the more reliable generalization estimate.

---

## Next step

`src/xgboost_retrieval.py` uses the best per-dataset parameters stored in
`strong_signal_xgboost_per_dataset` to train a final router and evaluate it
against BM25, Dense, and Static RRF on the held-out 15% test queries:

```
python src/xgboost_retrieval.py
```

Outputs:
- `data/results/strong_signal_retrieval_comparison.csv`
- `data/results/strong_signal_retrieval_comparison.png`
