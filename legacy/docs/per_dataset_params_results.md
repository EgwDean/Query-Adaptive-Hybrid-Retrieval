# Per-Dataset XGBoost Hyperparameter Search Results

## What was done

Following the ablation study, a full hyperparameter grid search was run
independently for each dataset to find the best XGBoost configuration per
dataset. This is an expanded grid (~6 480 combinations, 8× the model selection
grid) that additionally tunes `gamma`.

**Feature set**: 15 features — all features except `query_length`, which was
found to be net noise in the ablation study.

**Split strategy per dataset**:
- 85% train+dev — used for 10-fold Monte Carlo CV (80/20 splits) to select
  hyperparameters
- 15% test — held out; evaluated exactly once with the best parameters

The CV score is the mean wRRF NDCG@10 across 10 folds of the train+dev split.
The test score is the final single-evaluation result on the held-out 15%.

Script: `src/weak_signal_params_grid_search.py`
Output: `data/results/per_dataset_best_params.csv`
Config: `xgboost_per_dataset` in `config.yaml`

---

## Results

| Dataset | colsample_bytree | gamma | learning_rate | max_depth | min_child_weight | n_estimators | subsample | CV NDCG@10 | Test NDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| scifact | 0.8 | 0.0 | 0.3 | 8 | 1 | 100 | 0.8 | 0.6908 | 0.6628 |
| nfcorpus | 0.8 | 0.1 | 0.2 | 4 | 1 | 100 | 0.9 | 0.3018 | 0.3538 |
| arguana | 0.9 | 0.0 | 0.3 | 10 | 1 | 500 | 0.7 | 0.3813 | 0.3629 |
| fiqa | 0.8 | 0.0 | 0.2 | 10 | 1 | 500 | 0.9 | 0.3980 | 0.4679 |
| scidocs | 0.7 | 0.0 | 0.2 | 6 | 2 | 100 | 0.8 | 0.1763 | 0.1567 |

---

## Observations

**Per-dataset optimal depth varies widely.** `nfcorpus` (max_depth=4) uses a much
shallower tree than `arguana` and `fiqa` (max_depth=10). nfcorpus has the highest
average relevant documents per query (38.2), so the routing decision is less
binary — the optimal alpha lies in a narrower range and a shallow tree is
sufficient. Arguana and fiqa have sparser relevance (1.0 and 2.6 avg), so the
model needs more capacity to learn which extreme of alpha to predict.

**nfcorpus is the only dataset that benefits from gamma regularization** (gamma=0.1
vs 0.0 for all others). The small, high-recall corpus (3 633 docs) means the
feature distributions are more uniform, and gamma prevents the model from fitting
noise.

**scifact uses the same depth and rate as the global best** (max_depth=8, lr=0.3),
but with fewer estimators (100 vs 300). The per-dataset search found that earlier
stopping is sufficient for this dataset.

**min_child_weight=2 for scidocs** (vs 1 everywhere else). scidocs is the hardest
dataset (NDCG@10 ≈ 0.16); extra regularization on the leaf size prevents
overfit to a noisy training signal.

**CV vs test discrepancy**: nfcorpus (CV=0.302, test=0.354) and fiqa
(CV=0.398, test=0.468) both show higher test than CV scores. This is a small-
sample artifact — the 15% test split is 48 queries for nfcorpus and 97 for fiqa,
so single-run test scores have high variance. The CV score is the more reliable
estimate of generalization.

---