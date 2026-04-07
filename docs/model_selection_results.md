# Model Selection Results

## What was done

A grid search was run across 9 model families to find the best weak-signal router
for query-adaptive hybrid retrieval (wRRF). Each model was evaluated under
all combinations of its hyperparameters as defined in `config.yaml` under
`model_grid_search.models`, yielding roughly 2 600 total combinations.

For each combination, 10 repeated random 80/20 train/test splits (Monte Carlo CV)
were applied per dataset. The evaluation metric is macro-averaged NDCG@10 across
all 5 BEIR datasets (scifact, nfcorpus, arguana, fiqa, scidocs). The top 100
results were saved to `data/results/model_grid_search_top100.csv`.

Models tested: Logistic Regression, Random Forest, Extra Trees, XGBoost,
LightGBM, Gaussian NB, AdaBoost, LDA, MLP.

## Result

**All 100 entries in the top-100 CSV are XGBoost configurations.** No other model
family appeared anywhere in the top 100. This is a strong, unambiguous signal that
XGBoost is the right model family for this routing task.

XGBoost is trained with `objective="binary:logistic"` on the soft labels directly
(no binarization), which means it fits a gradient-boosted sigmoid model with
binary cross-entropy loss. The predicted probability is used as alpha for wRRF.

## Top 10 configurations

| Rank | colsample_bytree | learning_rate | max_depth | min_child_weight | n_estimators | subsample | Macro NDCG@10 |
|------|-----------------|---------------|-----------|-----------------|--------------|-----------|---------------|
| 1  | 0.8 | 0.1  | 8 | 1 | 300 | 0.8 | 0.3851 |
| 2  | 0.8 | 0.1  | 8 | 1 | 200 | 0.8 | 0.3845 |
| 3  | 1.0 | 0.3  | 8 | 1 | 200 | 0.8 | 0.3844 |
| 4  | 1.0 | 0.3  | 6 | 1 | 200 | 0.8 | 0.3844 |
| 5  | 1.0 | 0.3  | 8 | 1 | 300 | 0.8 | 0.3843 |
| 6  | 0.8 | 0.3  | 8 | 1 | 100 | 0.8 | 0.3843 |
| 7  | 1.0 | 0.3  | 6 | 1 | 300 | 0.8 | 0.3842 |
| 8  | 0.8 | 0.3  | 8 | 1 | 300 | 0.8 | 0.3842 |
| 9  | 0.8 | 0.1  | 6 | 1 | 300 | 0.8 | 0.3840 |
| 10 | 1.0 | 0.1  | 8 | 1 | 300 | 0.8 | 0.3840 |

Per-dataset NDCG@10 for rank 1:

| scifact | nfcorpus | arguana | fiqa   | scidocs |
|---------|----------|---------|--------|---------|
| 0.6392  | 0.3336   | 0.3742  | 0.4054 | 0.1730  |

## Observations

- `subsample=0.8` is present in every top-10 entry — full-data training
  (`subsample=1.0`) consistently underperforms with subsampling off.
- `min_child_weight=1` dominates the top 10, suggesting the router benefits
  from fitting fine-grained leaf nodes on this relatively small feature space.
- `max_depth` of 6 or 8 works best; shallower trees (3–4) do not appear.
- The spread across the top 10 is very narrow (0.3851 → 0.3840), meaning the
  exact hyperparameter choice within this region matters little.
- The macro gap between BM25-only (0.2927) and wRRF with the best XGBoost
  router (0.3851) is substantial, confirming that query-adaptive fusion
  adds meaningful value over static retrieval.
