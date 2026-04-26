# BGE-M3 Dual-Mode Pipeline Results

## What was done

A full hybrid retrieval pipeline was built using **both output heads of the same BGE-M3 model**:

- **Lexical leg**: BGE-M3 sparse head — term-weighting over the vocabulary, retrieved via an inverted index with dot-product scoring.
- **Semantic leg**: BGE-M3 dense head — CLS-pooled 1024-dim embedding, retrieved via cosine similarity.

The motivation: using a single encoder isolates the **retrieval paradigm** (sparse vs. dense) as the only experimental variable. BM25 vs. dense confounds the lexical retrieval model with the vocabulary model; BGE-M3 sparse vs. dense does not.

The wRRF fusion formula is identical to the BM25+Dense pipeline:
`α/(60+rank_sparse) + (1-α)/(60+rank_dense)`, α ∈ [0,1].

**Routing features**: same 15 features as the BM25+Dense pipeline, with two Group B adaptations:
- Average/max IDF computed from the sparse doc-frequency index (pseudo-IDF) instead of BM25 IDF.
- Cross-entropy replaced by Shannon entropy of normalized query sparse weights.

**Evaluation split**: balanced 5×300 query pool (300 queries per dataset, same truncation seed 31415), 70/15/15 train/dev/test split. Test set ≈ 45 queries per dataset.

**Grid searches**:
- Weak-signal XGBoost: 10-fold KFold CV, same grid as `weak_signal_params_grid_search.py`.
- Strong-signal XGBoost (1024-dim BGE-M3 query embeddings): same grid as `strong_signal_params_grid_search.py`.
- MoE meta-learner: 10-round Monte Carlo CV, 10 model families, 145 combos. Best model: **MLP**.

Scripts: `src/preprocess_bge_sparse.py`, `src/weighted_dual_bge_pipeline.py`
Outputs: `data/results/bge_dual_retrieval_comparison.{csv,png}`, `data/results/bge_dual_decision_boundary.png`

---

## Best MoE meta-learner

| model | CV NDCG@10 |
|-------|------------|
| MLP   | (best of 145 combos) |

The decision boundary plot shows a non-linear surface: when `alpha_strong` is high the MoE correctly drives α toward dense; when both inputs are low the MoE exploits the marginal sparse advantage on nfcorpus-type queries.

---

## Benchmark results

All NDCG@10 values on the ≈45-query per-dataset held-out test set (balanced 300-query pool).

| Dataset  | Sparse | Dense  | Static wRRF (α=0.5) | wRRF (weak) | wRRF (strong) | MoE    |
|----------|--------|--------|---------------------|-------------|---------------|--------|
| scifact  | 0.5871 | **0.6703** | 0.6246          | 0.6381      | **0.6435**    | 0.6353 |
| nfcorpus | 0.2524 | 0.2671 | 0.2817              | **0.2863**  | 0.2784        | 0.2777 |
| arguana  | 0.2542 | **0.3787** | 0.3302          | 0.3521      | **0.3642**    | 0.3416 |
| fiqa     | 0.2656 | **0.3683** | 0.3327          | **0.3554**  | 0.3404        | 0.3404 |
| scidocs  | 0.1512 | **0.2208** | 0.2037          | **0.2095**  | 0.2011        | 0.2000 |
| **MACRO**| 0.3021 | **0.3810** | 0.3546          | **0.3683**  | 0.3655        | 0.3590 |

Macro ranking:
```
Dense (0.3810) > wRRF-weak (0.3683) > wRRF-strong (0.3655) > MoE (0.3590) > Static wRRF (0.3546) > Sparse (0.3021)
```

---

## Assessment

### Summary

Dense retrieval (0.3810) is the strongest single method. The best routing method, wRRF-weak (0.3683), closes **−0.0127** below Dense. All routing methods improve substantially over Static wRRF (+0.014 for the best router) and over BGE-M3 Sparse-only (+0.066). The MoE (0.3590) does not surpass its best input.

### BGE-M3 Sparse as lexical leg vs. BM25

Comparing with the BM25+Dense pipeline (evaluated on the same 45-query balanced pool test sets via `meta_learner_moe_grid_search.py`):

| Method        | BM25+Dense | BGE-M3 Dual | Δ       |
|---------------|------------|-------------|---------|
| Dense         | 0.3810     | 0.3810      | 0.0000  |
| Static wRRF   | 0.3596     | 0.3546      | −0.0050 |
| wRRF (weak)   | 0.3784     | 0.3683      | −0.0101 |
| wRRF (strong) | 0.3802     | 0.3655      | −0.0147 |
| MoE           | 0.3748     | 0.3590      | −0.0158 |

**BGE-M3 Sparse is a weaker fusion partner than BM25 at every routing level.** The gap widens as the routing signal becomes stronger: the MoE loses 0.0158 vs. BM25+Dense, the strong-signal router loses 0.0147. This is the key finding of this experiment.

### Why BGE-M3 Sparse underperforms BM25 as a fusion partner

BGE-M3 Sparse and BGE-M3 Dense share the same encoder weights. Their retrieval signals are produced by the same representation of the text — the sparse head performs learned term-weighting on top of the same contextual token representations that the dense head pools. As a result the two signals are **more correlated** than BM25 and Dense: BM25 uses exact term frequency over the raw vocabulary, a signal that is structurally independent of the neural encoder.

Lower signal diversity means lower fusion ceiling. The wRRF alpha that maximises NDCG on a given query needs to trade off two meaningfully different ranked lists; when the lists are similar, the optimal alpha is near 0.5 for most queries, and routing provides less incremental gain.

This is confirmed by the Static wRRF result: BGE-M3 Dual Static wRRF (0.3546) < BM25+Dense Static wRRF (0.3596). Even the unrouted fusion is weaker when the two legs are more correlated.

### Per-dataset analysis

**scifact** — Dense is dominant (0.6703). All routing methods improve over Sparse alone and over Static wRRF. wRRF-strong (0.6435) is the best router, confirming that query embedding similarity captures the scifact relevance signal well.

**nfcorpus** — The only dataset where all routing methods beat Dense (0.2671). wRRF-weak (0.2863, +0.0192 over Dense) is the best method overall. The sparse leg provides genuine lexical complementarity here: nfcorpus queries are short biomedical keyword lookups where vocabulary matching is informative. This is the same qualitative pattern seen in the BM25+Dense pipeline, though the magnitude is smaller (+0.0192 BGE-M3 Dual vs. +0.0404 BM25+Dense — BM25 is more complementary to Dense on this dataset).

**arguana** — Dense is clearly best (0.3787). Arguana queries are full argument paragraphs; arguana retrieval is essentially passage-to-argument matching, where dense semantic similarity dominates. wRRF-strong (0.3642) is the best router, correctly identifying these queries as dense-preferred. MoE (0.3416) is weaker than wRRF-strong by −0.0226, a larger gap than expected.

**fiqa** — Dense best (0.3683). wRRF-weak (0.3554) is the best router, +0.0227 over Static wRRF. wRRF-strong (0.3404) ≈ MoE (0.3404). The weak-signal (feature-based) router outperforms the strong-signal (embedding-based) router on fiqa, the reverse of most other datasets — suggesting fiqa's query characteristics (financial questions) are captured better by surface features than by the query embedding alone.

**scidocs** — Dense is best (0.2208). All routing methods fall below Dense. This is the inverse of the BM25+Dense result, where routing beat dense on scidocs. The explanation: BM25 excels on scidocs because the retrieval task is citation-based (matching paper titles and abstracts by technical vocabulary), and BM25's exact term frequency is very informative there. BGE-M3 Sparse, while lexical, does not provide the same strong signal — its learned term weights are smoother and less discriminative than raw TF-IDF on technical citation vocabulary.

### MoE failure pattern

The MoE (MLP) does not improve over the best routing input in macro average. This mirrors the SVR finding in the BM25+Dense pipeline. The two base inputs (`alpha_weak`, `alpha_strong`) are correlated — when both agree, the MoE has no room to exploit disagreement; when they disagree, the MoE must generalise from too few training examples (45 queries per test dataset). The generalisation gap between CV and test is the primary cause of underperformance.

### Limitations

- Test sets of ≈45 queries per dataset produce high-variance estimates; per-dataset conclusions should be treated qualitatively.
- The balanced 300-query pool underrepresents large-query-set datasets (fiqa, arguana).
- The BGE-M3 Sparse index uses top-100 retrieval; BM25 also uses top-100. The recall@100 of BGE-M3 Sparse may be lower than BM25's for certain lexical query types (exact-match lookups), which would cap the fusion ceiling independently of routing quality.
