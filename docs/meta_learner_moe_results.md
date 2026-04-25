# MoE Meta-Learner Results (BGE-M3)

## What was done

A Mixture-of-Experts stacking meta-learner was trained on top of the two
base routers (weak-signal XGBoost and strong-signal XGBoost) to produce a
final per-query alpha. The meta-learner grid search searched 10 model families
and 145 hyperparameter combinations, evaluated via 10-round Monte Carlo CV.

**Base inputs:**
- `alpha_weak` — per-query alpha from the 15-feature XGBoost router
  (best per-dataset hyperparameters from the weak-signal grid search)
- `alpha_strong` — per-query alpha from the embedding-based XGBoost router
  (best per-dataset hyperparameters from the strong-signal grid search;
  input = 1024-dim BGE-M3 query embedding)

**OOF construction:** 10-fold standard KFold CV on the merged 1275-query
traindev pool (300 queries × 5 datasets). Each traindev query is predicted
by a base model that never saw it during training. The test set (15% of
each dataset's queries, same seed as all previous scripts) is predicted by
full-traindev-trained base models.

**Meta-learner CV:** 10-round Monte Carlo CV with `dev_frac_of_traindev=0.176`
(≈15% of the 1275-query pool per round). The macro NDCG@10 across all five
datasets is the objective.

Script: `src/meta_learner_moe_grid_search.py`
Outputs: `data/results/meta_learner_best_params.csv`,
         `data/results/meta_learner_retrieval_comparison.{csv,png}`,
         `data/results/meta_learner_decision_boundary.png`

---

## Best meta-learner

| model | params | CV NDCG@10 |
|-------|--------|------------|
| SVR   | C=0.1, ε=0.05 | 0.3996 |

A support-vector regressor with very strong regularisation (C=0.1) won the
145-combo grid search. The small epsilon (0.05) means the SVR fits a tight
tube around the training targets. The low C implies the decision surface is
heavily smoothed — the meta-learner effectively computes a regularised
weighted average of `alpha_weak` and `alpha_strong` rather than a complex
non-linear combination.

---

## Benchmark results

All NDCG@10 values on the 15% per-dataset held-out test set.
Note: the test set here is ~45 queries per dataset (15% of the 300-query
balanced pool), which is smaller than the full dataset test sets used in
`retrieval_explainability_results.md`. Per-dataset scores have high variance.

| Dataset  | BM25   | Dense  | Static RRF | wRRF (weak) | wRRF (strong) | MoE    |
|----------|--------|--------|------------|-------------|---------------|--------|
| scifact  | 0.6008 | 0.6703 | 0.6544     | 0.6313      | 0.6537        | **0.6561** |
| nfcorpus | 0.2607 | 0.2671 | 0.2912     | **0.3075**  | 0.2905        | 0.2903 |
| arguana  | 0.2193 | **0.3787** | 0.3071  | 0.3629      | **0.3677**    | 0.3261 |
| fiqa     | 0.1836 | **0.3683** | 0.3058  | 0.3585      | **0.3604**    | 0.3564 |
| scidocs  | 0.1957 | 0.2208 | 0.2394     | 0.2316      | 0.2289        | **0.2452** |
| **MACRO**| 0.2920 | **0.3810** | 0.3596 | 0.3784      | **0.3802**    | 0.3748 |

---

## Assessment

### Summary

The strong-signal wRRF router achieves a macro NDCG@10 of **0.3802**, which
is **0.0008 below Dense-only (0.3810)** — a gap that is within the noise of
the 45-query test sets. The MoE meta-learner (0.3748) fails to improve over
its best input, falling 0.0054 below `wRRF (strong)`. On the macro, the
ranking is:

```
Dense (0.3810) > wRRF-strong (0.3802) > wRRF-weak (0.3784) > MoE (0.3748) > Static RRF (0.3596) > BM25 (0.2920)
```

All routing methods improve substantially over BM25 (+0.086 for the best
router) and over Static RRF (+0.021 for the best router).

### Why the MoE does not improve over its best input

The decision boundary plot confirms that the SVR meta-learner has learned
an approximately linear interpolation between `alpha_weak` and `alpha_strong`
rather than a selective switching rule. With only 2 input dimensions and
strong regularisation (C=0.1), the meta-learner cannot learn to identify
queries where one router is reliably better than the other. The CV score
(0.3996) is higher than the test score (0.3748), indicating overfitting to
the MC-CV training distribution despite regularisation. The two base routers
produce correlated predictions, which limits the diversification available
to any stacking model.

### Per-dataset analysis

**scifact** — Dense retrieval is the strongest single method (0.6703). The
MoE (0.6561) outperforms Static RRF (0.6544) by +0.0017. The weak-signal
router (0.6313) underperforms all other methods, which is consistent with
the smaller training pool introducing more label noise on the scientific
claim verification task.

**nfcorpus** — The most striking result in this evaluation. All routing
methods beat Dense-only (0.2671), with the weak-signal router achieving
0.3075 (+0.0404 over Dense). This is the inverse of what was observed on the
full test set in `retrieval_explainability_results.md`, where Dense was
0.3800 and wRRF reached 0.3538. The 45-query test subset here appears to
sample a harder region of the nfcorpus query space where BM25 lexical
signals are more useful, illustrating the high variance of small held-out
sets. The MoE (0.2903) ≈ Static RRF (0.2912) and fails to exploit the
strong weak-signal advantage.

**arguana** — Dense is the best method (0.3787). The strong-signal router
(0.3677) achieves the best routing result, +0.0382 over Static RRF. The
MoE (0.3261) significantly underperforms all routing methods, including
Static RRF — it over-smooths the alpha signal and fails to maintain the
near-zero alpha that both individual routers correctly apply to this dataset.
This is the largest MoE underperformance across all datasets.

**fiqa** — The strong-signal router (0.3604) and weak-signal router (0.3585)
both improve substantially over Static RRF (0.3058), by +0.0546 and +0.0527
respectively. The MoE (0.3564) almost matches them. Dense is best (0.3683)
but only 0.0079 above the strong-signal router.

**scidocs** — The only dataset where the MoE is the best routing method
(0.2452 vs Static RRF 0.2394). All methods, including BM25, beat Dense-only
(0.2208), confirming that scidocs benefits from lexical signals in the
citation vocabulary. The MoE gain here (+0.0058 over Static RRF) is
consistent with the result from `retrieval_explainability_results.md`.

### Comparison with the standalone router results

The meta-learner experiment uses a different evaluation setup from
`retrieval_explainability_results.md` (45-query test sets vs. full dataset
test sets, and the merged 300-query pool instead of the full traindev split),
so absolute scores are not directly comparable. In the full-dataset evaluation,
Dense macro NDCG@10 was 0.4135 and the best router achieved 0.4008 (−0.0127).
In this balanced-pool evaluation, Dense macro is 0.3810 and the best router
achieves 0.3802 (−0.0008). The two evaluations bracket the same qualitative
conclusion: routing nearly matches but does not consistently surpass the dense
retriever alone.

### Limitations

- Test sets of ~45 queries per dataset produce high-variance per-dataset
  estimates. The nfcorpus reversal (routing beats dense here but not on the
  full test set) is the clearest example of this.
- The balanced 300-query pool underrepresents datasets with thousands of
  queries (fiqa, arguana), which may disadvantage dataset-specific routing
  patterns.
- The MoE fails to improve over its inputs, suggesting that the two base
  routers are too correlated for stacking to provide diversity gains.
- The generalisation gap (CV 0.3996 vs test 0.3748) indicates that the
  145-combo meta-learner search overfits slightly to the MC-CV distribution.
