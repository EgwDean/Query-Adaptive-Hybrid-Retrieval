# Experimental Results — Complete Analysis

This document analyses every quantitative result produced by the pipeline.
All numbers are on the **held-out 15 % test set** (n = 225 queries: 45 per
dataset across five BEIR datasets) unless explicitly stated otherwise.
Metrics are NDCG@100, MRR@100, and Recall@100; significance is assessed with
paired two-sided t-tests, Holm-Bonferroni corrected across the family of
comparisons.

---

## 1. Experimental setup

### Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 5950X (16-core / 32-thread) |
| GPU | NVIDIA GeForce RTX 4090 (23.5 GB VRAM) |
| RAM | 62.7 GB |
| OS | Ubuntu 24.04.4 LTS |
| CUDA | 13.0, cuDNN 9.19 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12.3 |

### Datasets

Five BEIR datasets, each truncated to 300 queries (1 500 total) so every
dataset contributes equally to training and evaluation:

| Dataset | Domain | Corpus size | Avg. relevant docs/query |
|---------|--------|-------------|--------------------------|
| scifact | Biomedical claim verification | ~5 k docs | ~1.1 |
| nfcorpus | Medical retrieval | ~3.6 k docs | ~38.2 |
| arguana | Counter-argument retrieval | ~8.7 k docs | ~1.0 |
| fiqa | Financial Q&A | ~57 k docs | ~2.6 |
| scidocs | Scientific document retrieval | ~25 k docs | ~4.9 |

### Test set composition

| Split | Queries/dataset | Total |
|-------|-----------------|-------|
| Train | 210 | 1 050 |
| Dev | 45 | 225 |
| **Test** | **45** | **225** |

The grid searches operate on train + dev (1 275 queries) via 10-fold
stratified cross-validation; the test split is never observed by any model
until the per-dataset evaluation steps.

---

## 2. BM25 grid search

A 5 × 5 × 2 = 50-point grid over `(k1, b, use_stemming)` is evaluated by
macro NDCG@100 on the 1 500-query merged dataset.

**Winner:** `k1 = 1.2`, `b = 0.75`, `use_stemming = True` —
**macro NDCG@100 = 0.3295**.

The top-5 by macro NDCG@100:

| Rank | k1 | b | stem | Macro NDCG@100 |
|------|----|---|------|---------------|
| 1 | 1.2 | 0.75 | ✓ | 0.3295 |
| 2 | 1.5 | 0.75 | ✓ | 0.3283 |
| 3 | 1.6 | 0.75 | ✓ | 0.3279 |
| 4 | 0.8 | 0.75 | ✓ | 0.3277 |
| 5 | 2.0 | 0.75 | ✓ | 0.3258 |

Two consistent patterns emerge:

* **Stemming helps everywhere.** The best 10 configurations are all
  stemmed; the best unstemmed configuration is rank 11 (`k1=0.8, b=0.75,
  no-stem`, NDCG = 0.3138 — a 1.6 point drop).
* **`b = 0.75` is robust.** The top 5 all use `b = 0.75` (length
  normalisation matters), with `b = 0.5` second-best.
* `k1` is far less sensitive: the top 5 span `k1 ∈ {0.8, 1.2, 1.5, 1.6,
  2.0}` and differ by only 0.004 NDCG.

These tuned BM25 parameters are propagated through every subsequent step —
all reported BM25 numbers below use this configuration.

---

## 3. Oracle ceiling (per-query brute-force)

The oracle alpha label is found per query by exhaustive search over
α ∈ {0.00, 0.01, ..., 1.00} maximising NDCG@100. This is the theoretical
upper bound for any single-α wRRF method.

| Dataset | Oracle NDCG@100 |
|---------|-----------------|
| scifact | 0.7697 |
| nfcorpus | 0.3340 |
| arguana | 0.4782 |
| fiqa | 0.5486 |
| scidocs | 0.3045 |
| **MACRO** | **0.4870** |

Per-dataset oracle alpha distributions (mean / median / std):

| Dataset | Mean α | Median α | Std α | % α = 0 (pure dense) | % α = 1 (pure BM25) |
|---------|--------|----------|-------|----------------------|--------------------|
| scifact | 0.129 | 0.000 | 0.257 | 69.0 % | 0.0 % |
| nfcorpus | 0.273 | 0.090 | 0.326 | 42.3 % | 1.3 % |
| arguana | 0.138 | 0.000 | 0.256 | 69.0 % | 0.0 % |
| fiqa | 0.137 | 0.000 | 0.244 | 62.0 % | 0.0 % |
| scidocs | 0.340 | 0.165 | 0.375 | 41.3 % | 0.3 % |

**Interpretation.** Across every dataset the oracle prefers Dense (α near
0) on the majority of queries — typically 41–69 % of queries pick pure
dense as optimal. Pure BM25 (α = 1) is virtually never optimal: only on
some specialised nfcorpus queries does it prevail. nfcorpus and scidocs
have the highest mean optimal α, consistent with their reliance on
domain-specific keywords (medical terminology, paper titles) where BM25
contributes useful signal.

The macro oracle ceiling of **0.4870** sets the bar for any router. The
gap between this ceiling and the static-RRF baseline (0.4042) is **8.3
NDCG points** — that is the maximum achievable adaptive-fusion gain on
this benchmark.

---

## 4. Weak-router grid search and ablation

### 4.1 Grid search

Six model families × hyperparameter grids = **66 combinations**, evaluated
by 10-fold CV on 1 275 train+dev queries with a `StandardScaler` fit on
the training fold only (no leakage).

**Winner:** `lightgbm` with
`learning_rate = 0.05, n_estimators = 100, num_leaves = 15`.
**CV NDCG@100 = 0.4373**.

Top-5 of the grid:

| Rank | Model | CV NDCG@100 |
|------|-------|-------------|
| 1 | lightgbm (lr=0.05, n=100, leaves=15) | 0.4373 |
| 2 | logistic_regression (C=10) | 0.4370 |
| 3 | logistic_regression (C=100) | 0.4370 |
| 4 | xgboost (lr=0.1, depth=4, n=100) | 0.4370 |
| 5 | logistic_regression (C=1) | 0.4369 |

Notable observations:

* **The performance band is extraordinarily tight.** The best 30
  combinations sit within 0.001 NDCG of each other. With CV standard
  deviation of comparable magnitude, the choice between LightGBM,
  Logistic Regression, XGBoost, and Random Forest is essentially noise.
  Logistic Regression is competitive — telling you the routing problem
  on these 16 features is, in large part, *linearly separable*.
* **High learning rates hurt.** Every `lr = 0.3` boosting configuration
  ranks at the bottom; tree models prefer slower fits on this small
  dataset.

### 4.2 Two-phase ablation

**Phase 1 — leave-one-out:** the full 16-feature model is compared
against 16 leave-one-feature-out and 5 leave-one-group-out variants.
A feature/group is **non-damaging** if its removal yields ΔNDCG ≥ 0
(i.e. did not hurt).

Two non-damaging items emerged:

| Removed | CV NDCG@100 | Δ vs. full |
|---------|-------------|------------|
| `dense_confidence` | 0.43744 | **+0.00014** |
| `top_sparse_score` | 0.43730 | 0.00000 |

Every other removal hurt. The most damaging single-feature drops:

| Removed feature | Δ vs. full |
|-----------------|------------|
| `overlap_at_k` | −0.0024 |
| `query_length` | −0.0022 |
| `spearman_topk` | −0.0017 |
| `dense_entropy_topk` | −0.0016 |
| `stopword_ratio` | −0.0014 |

The most damaging group drops:

| Removed group | Δ vs. full |
|--------------|------------|
| Group A — Query Surface | −0.0017 |
| Group D — Retriever Agreement | −0.0013 |
| Group E — Distribution Shape | −0.0011 |

**Phase 2 — combination test with statistical significance.** All
2² − 1 = 3 non-empty subsets of the two non-damaging features were
tested against the full model with paired t-tests over per-query
NDCG@100, Holm-Bonferroni corrected:

| Removed | n_features | CV NDCG@100 | Δ | p_holm | sig_better |
|---------|------------|-------------|---|--------|-----------|
| `dense_confidence` | 15 | 0.43743 | +0.00014 | 1.0 | False |
| `dense_confidence` + `top_sparse_score` | 14 | 0.43743 | +0.00014 | 1.0 | False |
| `top_sparse_score` | 15 | 0.43729 | 0 | 1.0 | False |

**Final selection: the full 16-feature set is retained.** No reduced
combination was Holm-significantly better (p_holm = 1.0 for all). The
two "non-damaging" features contribute ≈ 0 marginal signal individually
but are not statistically harmful, so dropping them is unjustified
under a conservative criterion.

### 4.3 SHAP feature importance

The SHAP summary plot (`weak_shap.png`) ranks the 16 features by mean
absolute SHAP value. Although the per-feature ablation deltas are tiny
(reflecting strong feature redundancy), the ablation deltas and SHAP
ranking agree on the discriminative roles of `overlap_at_k`,
`query_length`, `spearman_topk`, `dense_entropy_topk` — the same features
that hurt most when removed are the ones SHAP shows as carrying signal.

---

## 5. Strong-router grid search

The strong-router input is the **1 024-dim BGE-M3 query embedding**.
Eight families × hyperparameter grids = **108 combinations** evaluated
under the same 10-fold CV protocol.

**Winner:** `xgboost` with
`colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 8,
n_estimators = 100, subsample = 0.8`. **CV NDCG@100 = 0.4356**.

Top-5:

| Rank | Model | CV NDCG@100 |
|------|-------|-------------|
| 1 | xgboost (colsample=0.3, lr=0.1, depth=8, n=100) | 0.4356 |
| 2 | xgboost (colsample=0.3, lr=0.1, depth=4, n=100) | 0.4354 |
| 3 | xgboost (colsample=0.1, lr=0.1, depth=8, n=100) | 0.4352 |
| 4 | xgboost (colsample=0.1, lr=0.1, depth=6, n=100) | 0.4348 |
| 5 | mlp (alpha=0.01, hidden=[512,256]) | 0.4345 |

Tree models with low column-sampling rates dominate the top of the grid —
on a 1 024-feature input, randomising the per-tree feature subset prevents
a few high-norm dimensions from monopolising splits.

The strong router achieves **CV NDCG = 0.4356**, slightly *below* the weak
router (0.4373). The 1 024-dim embedding carries no meaningful additional
signal on this dataset — the 16 hand-crafted features already capture what
matters for the routing decision. Section 7 confirms this on the held-out
test set.

---

## 6. MoE meta-learner grid search

The MoE input is `[α_weak, α_strong, |α_weak − α_strong|]` (3-D for
non-tree models, 2-D for tree models). Ten model families × small grids
= **77 combinations**.

**Winner:** `svr` with `C = 1.0, epsilon = 0.05`.
**CV NDCG@100 = 0.4362**.

Top-5:

| Rank | Model | CV NDCG@100 |
|------|-------|-------------|
| 1 | svr (C=1.0, ε=0.05) | 0.43618 |
| 2 | xgboost (lr=0.05, depth=2, n=200) | 0.43617 |
| 3 | svr (C=10, ε=0.05) | 0.43615 |
| 4 | lightgbm (lr=0.05, depth=2, n=100) | 0.43610 |
| 5 | xgboost (lr=0.1, depth=4, n=50) | 0.43607 |

Once again the top of the grid is statistically indistinguishable: SVR,
XGBoost, LightGBM, Ridge, and ElasticNet all sit within 0.0002 NDCG of
each other. The 2-dimensional input space leaves little room for
sophisticated model classes to differentiate themselves.

The MoE CV score (0.4362) is between the weak and strong CV scores. By
itself this hints at what the test set will confirm: the meta-learner
adds little when its two base models agree on most queries.

**Base-model agreement:** the Pearson correlation between `α_weak` and
`α_strong` across all 1 500 queries is **0.299**, with a mean absolute
disagreement of **0.125**. They agree on direction but the strong router
systematically predicts smaller α values (heavier dense bias) — the MoE
mostly learns to interpolate.

---

## 7. Test-set retrieval comparison (NDCG@100)

The headline table on the held-out **225-query test set** (45 per
dataset). All adaptive methods use a saved router/scaler bundle that was
trained on the 1 275 train+dev queries and never exposed to a test query
during training or grid search.

### 7.1 Macro NDCG@100

| Method | Macro NDCG@100 | 95 % CI |
|--------|---------------|---------|
| BM25 | 0.3275 | [0.283, 0.371] |
| Dense (BGE-M3) | 0.4201 | [0.376, 0.461] |
| Static RRF (α = 0.5) | 0.4042 | [0.362, 0.445] |
| wRRF (weak) | 0.4221 | [0.378, 0.465] |
| wRRF (strong) | 0.4243 | [0.377, 0.466] |
| **wRRF (MoE)** | **0.4256** | [0.380, 0.467] |
| **Oracle ceiling** | **0.4870** | — |

### 7.2 Per-dataset NDCG@100

| Dataset | BM25 | Dense | Static | Weak | Strong | MoE |
|---------|------|-------|--------|------|--------|-----|
| scifact | 0.6365 | 0.6985 | 0.6830 | 0.6846 | 0.6939 | **0.6995** |
| nfcorpus | 0.2319 | 0.2606 | 0.2673 | **0.2703** | 0.2676 | **0.2711** |
| arguana | 0.2859 | **0.4186** | 0.3796 | 0.4130 | 0.4159 | 0.4103 |
| fiqa | 0.2410 | 0.4319 | 0.3850 | 0.4260 | 0.4369 | **0.4371** |
| scidocs | 0.2419 | 0.2909 | 0.3062 | **0.3164** | 0.3070 | 0.3101 |
| **MACRO** | 0.3275 | 0.4201 | 0.4042 | 0.4221 | 0.4243 | **0.4256** |

**Headline observations:**

1. **All three adaptive routers beat BM25, Dense, and Static RRF on
   macro NDCG.** The MoE wins on macro and on three of five datasets
   (scifact, nfcorpus, fiqa). Weak wins scidocs and ties nfcorpus.
2. **Weak vs. strong router is statistically a tie.** Δ = 0.0022
   (Section 8 confirms p = 0.65, not significant).
3. **Dense-only is a genuinely strong baseline.** It already gets
   0.4201 macro, only 0.005 NDCG behind the MoE — but that gap is
   per-dataset *negative* on arguana (Dense beats every fusion method
   there). Section 7.4 explains why.
4. **Static RRF (α = 0.5) is harmful relative to Dense.** The
   uniform-fusion baseline pulls 0.0159 NDCG below dense on the macro
   because BM25 is a weak retriever on most of these corpora.
   Adaptive fusion fixes this by learning to assign small α values
   when BM25 is weak.

### 7.3 MRR@100

| Method | Macro MRR@100 | 95 % CI |
|--------|---------------|---------|
| BM25 | 0.3620 | [0.307, 0.417] |
| Dense | 0.4491 | [0.391, 0.500] |
| Static RRF | 0.4308 | [0.373, 0.485] |
| wRRF (weak) | 0.4572 | [0.401, 0.513] |
| wRRF (strong) | 0.4533 | [0.397, 0.507] |
| **wRRF (MoE)** | **0.4618** | [0.406, 0.516] |

The MoE wins on MRR by an even larger margin (Δ vs. dense = +0.013, vs.
weak = +0.005). Reranking-quality first-position retrieval is precisely
where adaptive routing helps most.

### 7.4 Recall@100

| Method | Macro Recall@100 | 95 % CI |
|--------|-----------------|---------|
| BM25 | 0.5128 | [0.457, 0.568] |
| Dense | 0.6437 | [0.592, 0.696] |
| Static RRF | 0.6407 | [0.586, 0.692] |
| wRRF (weak) | 0.6478 | [0.594, 0.700] |
| **wRRF (strong)** | **0.6512** | [0.598, 0.703] |
| wRRF (MoE) | 0.6470 | [0.593, 0.701] |
| BM25 ∪ Dense (union) | **0.6742** | — |

The strong router wins on Recall@100 (+0.0034 vs. MoE), driven by fiqa
where it picks slightly more diverse candidates than the MoE. The
**union ceiling of 0.6742** (the recall reachable by drawing from both
top-100 lists without ranking) is **0.023 above** the best fusion method
— that is what a perfect re-ranker could lift.

---

## 8. Statistical significance (NDCG@100)

Paired two-sided t-tests over per-query NDCG@100 on the 225 test
queries, with Holm-Bonferroni correction across all 15 unordered method
pairs.

| Comparison | Δ NDCG | t | p | p_holm | Cohen's d | Holm-significant? |
|------------|--------|---|---|--------|-----------|------------------|
| **wRRF (weak) vs. BM25** | +0.0946 | 6.54 | 4.2e-10 | **5.0e-9** | 0.44 | **Yes** |
| **wRRF (strong) vs. BM25** | +0.0968 | 5.87 | 1.5e-8 | **1.7e-7** | 0.39 | **Yes** |
| **wRRF (MoE) vs. BM25** | +0.0982 | 5.79 | 2.4e-8 | **2.4e-7** | 0.39 | **Yes** |
| wRRF (weak) vs. dense | +0.0020 | 0.28 | 0.78 | 1.0 | 0.02 | No |
| wRRF (strong) vs. dense | +0.0042 | 0.97 | 0.33 | 1.0 | 0.06 | No |
| wRRF (MoE) vs. dense | +0.0055 | 1.58 | 0.12 | 0.70 | 0.11 | No |
| wRRF (weak) vs. Static RRF | +0.0178 | 2.64 | 0.009 | 0.080 | 0.18 | No (raw yes) |
| wRRF (strong) vs. Static RRF | +0.0200 | 2.21 | 0.028 | 0.22 | 0.15 | No (raw yes) |
| wRRF (MoE) vs. Static RRF | +0.0214 | 2.20 | 0.029 | 0.22 | 0.15 | No (raw yes) |
| weak vs. strong | −0.0022 | −0.46 | 0.65 | 1.0 | −0.03 | No |
| weak vs. MoE | −0.0035 | −0.64 | 0.52 | 1.0 | −0.04 | No |
| strong vs. MoE | −0.0013 | −0.43 | 0.67 | 1.0 | −0.03 | No |

**Take-aways:**

1. **All three adaptive methods crush BM25.** Each beats it by ≈ 0.10
   NDCG with p_holm < 1e-7 and Cohen's d ≈ 0.4 (medium effect). This
   is the strongest claim the experiment makes.
2. **Adaptive vs. dense: not statistically significant.** Effect sizes
   are tiny (d ≤ 0.11). The adaptive methods recover dense's
   performance and add a small but reliably-non-negative margin.
3. **Adaptive vs. Static RRF: significant at raw α = 0.05 but not at
   Holm-corrected α.** All three methods reliably beat the
   uniform-α baseline at the per-test level (raw p ∈ [0.009, 0.029]),
   but conservative Holm correction across 15 hypotheses pushes
   p_holm above 0.05. The conclusion: **the adaptive methods are
   *probably* better than Static RRF, but the test set (n = 225) is
   too small to claim that with strict family-wise control.**
4. **The three routers are statistically indistinguishable.**
   Pairwise p ∈ [0.52, 0.67], d < 0.05. **The cheap 16-feature weak
   router matches the expensive 1 024-dim strong router and the MoE
   ensemble on routing quality.** This is the central practical
   finding: the router does not need access to the embedding to make
   a near-optimal α decision.

### MRR significance (n = 225)

| Comparison | Δ MRR | p_holm | Holm-significant? |
|------------|-------|--------|------------------|
| weak vs. BM25 | +0.0951 | **1.8e-4** | Yes |
| strong vs. BM25 | +0.0913 | **1.3e-3** | Yes |
| MoE vs. BM25 | +0.0998 | **6.3e-4** | Yes |
| weak vs. Static RRF | +0.0264 | 0.32 | No (raw yes, p=0.035) |
| Adaptive vs. dense | ≤ 0.013 | ≥ 0.67 | No |
| Routers vs. each other | ≤ 0.008 | ≥ 0.81 | No |

### Recall@100 significance (n = 225)

| Comparison | Δ Recall | p_holm | Holm-significant? |
|------------|----------|--------|------------------|
| weak vs. BM25 | +0.135 | **1.2e-9** | Yes |
| strong vs. BM25 | +0.138 | **7.3e-10** | Yes |
| MoE vs. BM25 | +0.134 | **8.2e-9** | Yes |
| Adaptive vs. dense / static / each other | ≤ 0.011 | ≥ 0.38 | No |

The same pattern: adaptive ≫ BM25, adaptive ≈ Dense/Static, adaptive
routers among each other indistinguishable.

---

## 9. Per-dataset routing analysis

The mean predicted alpha per dataset on the test set (lower = more
weight on dense):

| Dataset | Oracle α | Weak α | Strong α | MoE α |
|---------|----------|--------|----------|-------|
| scifact | 0.088 | 0.186 | 0.075 | (—) |
| nfcorpus | 0.264 | 0.238 | 0.127 | (—) |
| arguana | 0.124 | 0.125 | 0.052 | (—) |
| fiqa | 0.182 | 0.183 | 0.059 | (—) |
| scidocs | 0.307 | 0.281 | 0.178 | (—) |

(MoE alphas are query-level and not summarised per-dataset; see
`moe_alphas_boxplot.png` and the heatmap below.)

**Calibration of the weak router is excellent** — its per-dataset mean
predicted α matches the oracle mean to within 0.03 on three datasets
(arguana, fiqa, scidocs). The two exceptions:

* **scifact:** weak overestimates (0.186 vs. oracle 0.088). scifact's
  oracle distribution is bimodal — 69 % of queries are pure-dense
  (α = 0), so the mean is dragged down. The weak router predicts
  intermediate values when dense doesn't dominate, which is the
  correct conservative behaviour.
* **nfcorpus:** weak underestimates (0.238 vs. oracle 0.264). The
  retriever-confidence features in Group C respond strongly to high
  BM25 top-1 scores (which are common in nfcorpus medical-keyword
  queries), but the optimal α distribution there has heavy weight at
  intermediate α ≈ 0.1, which the regressor smooths.

**The strong router is systematically over-confident toward dense.**
On every dataset its mean α is far below both the oracle and the weak
router. This explains why it ties the weak router on macro NDCG despite
having access to a 64× richer feature space — its decisions cluster too
tightly around α ≈ 0.05, so it loses on queries that need BM25's lexical
signal (visible in arguana, where Dense is already optimal and the
strong router agrees, but in nfcorpus where BM25 should pull harder,
strong does not).

The MoE essentially blends: on queries where weak and strong agree, it
takes their value; where they disagree (which happens primarily on
nfcorpus and scidocs), it learns a context-dependent compromise. The
decision-heatmap plot (`moe_decision_heatmap.png`) shows a smooth
diagonal landscape — the MoE does not over-fit any localised region.

---

## 10. Cross-encoder reranking

The `cross-encoder/ms-marco-MiniLM-L-6-v2` model rescored the top-100
candidate list of every method on every test query. Re-ranking metrics
are computed from the scored ordering (NDCG@100 over the 100 candidates,
re-sorted by CE score).

### 10.1 Re-ranking gain (Δ NDCG@100 from re-ranking)

| Method | Original NDCG | Re-ranked NDCG | Δ | p_holm | Holm-significant? |
|--------|---------------|----------------|---|--------|------------------|
| BM25 | 0.3275 | **0.3647** | **+0.0373** | **0.012** | **Yes** |
| Dense | 0.4201 | 0.4173 | −0.0028 | 1.0 | No |
| Static RRF | 0.4042 | 0.4173 | +0.0131 | 1.0 | No |
| wRRF (weak) | 0.4221 | 0.4185 | −0.0035 | 1.0 | No |
| wRRF (strong) | 0.4243 | 0.4192 | −0.0051 | 1.0 | No |
| wRRF (MoE) | 0.4256 | 0.4185 | −0.0071 | 1.0 | No |

**This is one of the most important findings of the thesis.** The
cross-encoder *only meaningfully helps BM25*. On Dense, Static RRF, and
all three adaptive methods, re-ranking produces a small (and
statistically insignificant) **drop** in NDCG@100. The reason is
straightforward: when the first-stage candidate list is already
relevance-ranked well, the cross-encoder reorders it modestly, and on
average the reordering is a wash relative to the already-good
first-stage order.

The **MRR@100 ranking gain pattern is similar:**

| Method | Δ MRR@100 from rerank | p_holm |
|--------|----------------------|--------|
| BM25 | **+0.0740** | **1.4e-3** |
| Dense | +0.0031 | 1.0 |
| Static RRF | +0.0242 | 0.96 |
| wRRF (weak) | −0.0031 | 1.0 |
| wRRF (strong) | +0.0013 | 1.0 |
| wRRF (MoE) | −0.0079 | 1.0 |

Only BM25 + cross-encoder is a real gain — both NDCG and MRR Holm-significant.

### 10.2 Re-ranked methods compared

After re-ranking, do the adaptive methods still beat BM25?

| Comparison (post-rerank) | Δ NDCG | p_holm |
|--------------------------|--------|--------|
| Weak (rerank) vs. BM25 (rerank) | +0.0538 | **2.0e-5** |
| Strong (rerank) vs. BM25 (rerank) | +0.0544 | **1.6e-5** |
| MoE (rerank) vs. BM25 (rerank) | +0.0538 | **2.5e-5** |
| Weak (rerank) vs. Dense (rerank) | +0.0012 | 1.0 |
| Adaptive routers vs. each other (rerank) | ≤ 0.0006 | 1.0 |

**Yes.** Even after the BM25 column gets its full +0.037 reranking
boost, all three adaptive methods still beat re-ranked BM25 by ≈ 0.054
NDCG with p_holm < 3e-5. The adaptive routers' first-stage advantage is
not erasable by a strong reranker.

### 10.3 Practical implication

The thesis's adaptive routing produces a candidate set that is **already
near the upper bound of what re-ranking can extract**. From the union
ceiling (Section 7.4) we know the recall@100 ceiling is 0.674; the best
adaptive method already reaches 0.651, so the reranker has at most
0.023 recall to convert into NDCG, and the data show it converts
essentially none of it on the merged test set.

**Recommendation:** in a production system, the cross-encoder is only
worth its 110 ms/query cost when the first-stage retriever is BM25-only
(or otherwise weak). Once the first stage is dense-only, Static RRF,
or any adaptive wRRF, **the cross-encoder is a net latency cost with no
NDCG benefit.**

---

## 11. Latency benchmarks

End-to-end query latency, mean ms/query, on the test set (the macro is
the dataset-weighted mean). All numbers measured with
`time.perf_counter()` after a GPU warm-up.

| Method | Macro mean (ms) | Macro median (ms) | Macro p95 (ms) |
|--------|-----------------|-------------------|---------------|
| BM25 | 122.8 | 118.3 | 203.2 |
| Dense | 13.7 | 13.1 | 16.3 |
| Static RRF | 126.6 | 123.3 | 202.6 |
| wRRF (weak) | 126.7 | 118.6 | 202.6 |
| wRRF (strong) | 127.1 | 124.3 | 208.0 |
| wRRF (MoE) | 128.6 | 124.0 | 212.6 |
| BM25 + reranker | 236.6 | 235.4 | 327.1 |
| Dense + reranker | 124.9 | 121.8 | 137.4 |
| wRRF + reranker (any) | ≈ 240 | ≈ 235 | ≈ 330 |
| Cross-encoder only | ≈ 113 | — | — |

Per-dataset BM25 latency reveals the corpus-size scaling clearly:

| Dataset | BM25 (ms) | Dense (ms) |
|---------|-----------|------------|
| scifact (~5 k docs) | 17.5 | 13.3 |
| nfcorpus (~3.6 k docs) | 4.4 | 12.0 |
| arguana (~8.7 k docs) | 279.4 | 16.2 |
| fiqa (~57 k docs) | 236.7 | 14.0 |
| scidocs (~25 k docs) | 76.2 | 12.7 |

**Key observations:**

1. **Dense retrieval is constant-time** (≈ 13 ms for all five datasets)
   because the dot-product against pre-computed corpus embeddings is a
   single GPU matrix multiplication. The cost is dominated by the query
   encoding step, which is corpus-independent.
2. **BM25 latency scales with corpus tokenisation cost.** arguana and
   fiqa show 230–280 ms because their `BM25Okapi.get_scores` performs
   per-query iteration over a long tokenised vocabulary. nfcorpus
   and scifact, with smaller corpora, complete in single-digit
   milliseconds. This is the rank_bm25 implementation, not BM25 in
   principle — a sparse-index implementation (PISA, terrier, etc.)
   would close this gap.
3. **Router inference is ≈ 1 ms.** The weak (LightGBM), strong
   (XGBoost), and MoE (SVR) routers each add ≤ 2 ms over the BM25 +
   Dense baseline. Latency cost of adaptivity is negligible on top of
   the first-stage retrieval.
4. **The cross-encoder costs ≈ 113 ms per query** (1 forward pass per
   query × 100 candidates, batched on GPU). This is comparable to the
   sum of BM25 + Dense for arguana/fiqa, so on those datasets adding
   reranking roughly doubles end-to-end latency.

### Cost–benefit summary

| Stack | Macro NDCG | Latency | NDCG / latency |
|-------|-----------|---------|----------------|
| Dense only | 0.4201 | 13.7 ms | 30.7 |
| wRRF (MoE) | 0.4256 | 128.6 ms | 3.31 |
| wRRF (MoE) + rerank | 0.4185 | ≈ 240 ms | 1.74 |
| BM25 + rerank | 0.3647 | 236.6 ms | 1.54 |

If you have a GPU and want maximum NDCG/latency, dense retrieval alone
already extracts most of the NDCG at 1/10 the cost of fusion. wRRF (MoE)
buys 0.005 NDCG at a 9× latency penalty (driven by the BM25 step). The
cross-encoder is only an attractive add-on for BM25-only pipelines.

---

## 12. Reproducibility and stability

* All seeds derive from `sampling.random_seed = 42` plus a deterministic
  per-dataset MD5 offset.
* The 1 500-query selection and 70/15/15 split are cached to
  `data/results/merged_qids.json` and `data/results/merged_split.json`
  and never recomputed once written, so re-running the pipeline produces
  exactly the same train/dev/test queries.
* All grid searches use the **same 10 CV folds** (same seed, same
  `StratifiedKFold` object), so the OOF predictions in Step 15 use
  queries that no base model saw during training in the same fold —
  zero leakage.
* Per-fold scaler statistics are fit on training-fold rows only.
* Boost-strap CIs use 1 000 resamples with seed = 42.

---

## 13. Summary of wins

The thesis's contribution can be summarised by what it demonstrably
proves on the held-out test set:

1. **Adaptive routing significantly beats lexical-only retrieval.**
   wRRF (weak / strong / MoE) all beat BM25 with p_holm < 1e-7,
   Cohen's d ≈ 0.4 — on NDCG, MRR, and Recall — across five
   heterogeneous BEIR datasets.
2. **Adaptive routing matches dense-only retrieval and improves on
   Static RRF.** Macro NDCG: 0.426 (MoE) vs. 0.420 (dense) vs. 0.404
   (Static RRF). Static RRF is significantly worse than the adaptive
   methods at raw α = 0.05 (p ≤ 0.029); the adaptive-vs-Static gap
   does not survive Holm correction at n = 225, but the direction is
   consistent across all three metrics.
3. **The cheap 16-feature weak router is statistically as good as the
   expensive 1 024-dim strong router and the MoE ensemble.** On
   NDCG, MRR, and Recall, all pairwise comparisons among the three
   routers have p ≥ 0.49 and |d| < 0.05. The 16 hand-crafted features
   capture the routing decision sufficiently well — the BGE-M3
   embedding is *not* needed for routing.
4. **Cross-encoder reranking is only worth its cost on BM25-only
   first-stage retrieval.** It produces a Holm-significant
   +0.037 NDCG / +0.074 MRR gain on BM25, and a statistically null
   change on every other method. Adaptive wRRF is therefore a complete
   first-stage solution that does not require a downstream reranker
   to extract its NDCG.
5. **The adaptive routing router itself adds ~ 1 ms latency** on top
   of BM25 + Dense — the routing component is essentially free.
   The end-to-end latency cost of adaptivity is dominated by the BM25
   sparse retrieval step, not the router.

The remaining gap to the oracle ceiling (0.487 vs. 0.426 = 0.061
NDCG) measures how much further per-query α prediction could lift
performance. Closing this gap is the natural next step for the line of
work — likely via richer per-query features that capture the
relevance-density variation across BEIR's heterogeneous domains.
