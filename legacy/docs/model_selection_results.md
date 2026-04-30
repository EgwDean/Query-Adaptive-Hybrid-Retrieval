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

**~90 of the top-100 entries are XGBoost configurations.** The remaining entries
include Extra Trees and MLP, but XGBoost dominates throughout. All top-10 entries
are XGBoost. This is a strong, unambiguous signal that XGBoost is the right model
family for this routing task.

XGBoost is trained with `objective="binary:logistic"` on the soft labels directly
(no binarization), which means it fits a gradient-boosted sigmoid model with
binary cross-entropy loss. The predicted probability is used as alpha for wRRF.

## Top 10 configurations

| Rank | colsample_bytree | learning_rate | max_depth | min_child_weight | n_estimators | subsample | Macro NDCG@10 |
|------|-----------------|---------------|-----------|-----------------|--------------|-----------|---------------|
| 1  | 0.8 | 0.1  | 8 | 1 | 300 | 0.8 | 0.3858 |
| 2  | 1.0 | 0.1  | 8 | 1 | 300 | 0.8 | 0.3857 |
| 3  | 0.8 | 0.1  | 8 | 1 | 200 | 0.8 | 0.3857 |
| 4  | 1.0 | 0.1  | 8 | 1 | 200 | 0.8 | 0.3854 |
| 5  | 0.8 | 0.3  | 6 | 1 | 200 | 0.8 | 0.3851 |
| 6  | 0.8 | 0.1  | 6 | 1 | 300 | 0.8 | 0.3850 |
| 7  | 1.0 | 0.05 | 8 | 1 | 300 | 0.8 | 0.3850 |
| 8  | 1.0 | 0.3  | 8 | 1 | 300 | 0.8 | 0.3848 |
| 9  | 0.8 | 0.3  | 8 | 1 | 200 | 0.8 | 0.3848 |
| 10 | 0.8 | 0.3  | 8 | 1 | 100 | 0.8 | 0.3847 |

Per-dataset NDCG@10 for rank 1:

| scifact | nfcorpus | arguana | fiqa   | scidocs |
|---------|----------|---------|--------|---------|
| 0.6392  | 0.3357   | 0.3746  | 0.4073 | 0.1722  |

## Observations

- `subsample=0.8` is present in every top-10 entry — full-data training
  (`subsample=1.0`) consistently underperforms.
- `min_child_weight=1` dominates the top 10, suggesting the router benefits
  from fitting fine-grained leaf nodes on this relatively small feature space.
- `max_depth` of 6 or 8 works best; shallower trees (3–4) do not appear.
- The spread across the top 10 is very narrow (0.3858 → 0.3847), meaning the
  exact hyperparameter choice within this region matters little.
- The macro gap between BM25-only (0.2927) and wRRF with the best XGBoost
  router (0.3858) is substantial, confirming that query-adaptive fusion
  adds meaningful value over static retrieval.
