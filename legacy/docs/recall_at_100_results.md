# Recall@100 Analysis

## What was done

Recall@100 was computed for all 6 retrieval methods on the same test split as the
NDCG@10 evaluation (balanced 5×300 query pool, ≈45 test queries per dataset).
Alpha values (weak, strong) were loaded from the cached `meta_learner_dataset.csv`
and the best MoE meta-learner was retrained on the traindev portion of that CSV.

An additional "Union BM25∪Dense" column gives the theoretical Recall@100 ceiling —
the fraction of relevant documents present anywhere in the combined BM25 top-100
and Dense top-100 candidate pools, regardless of how they are ranked.

Script: `src/recall_at_100.py`
Outputs: `data/results/recall_at_100.{csv,png}`

---

## Results

All values are mean Recall@100 over ≈45 test queries per dataset.

| Dataset  | BM25   | Dense  | Static wRRF | wRRF (weak) | wRRF (strong) | MoE    | Union (ceiling) |
|----------|--------|--------|-------------|-------------|---------------|--------|-----------------|
| scifact  | 0.7956 | 0.8844 | 0.9067      | 0.9067      | 0.9067        | 0.9067 | **0.9067**      |
| nfcorpus | 0.2202 | 0.2636 | 0.2522      | 0.2576      | 0.2601        | 0.2529 | **0.3019**      |
| arguana  | 0.7778 | 0.9556 | 0.9556      | **0.9778**  | 0.9556        | 0.9556 | **0.9778**      |
| fiqa     | 0.3738 | 0.6848 | 0.6504      | 0.6689      | **0.6867**    | 0.6689 | **0.7078**      |
| scidocs  | 0.3267 | 0.4300 | 0.4300      | 0.4389      | **0.4433**    | 0.4300 | **0.4700**      |
| **MACRO**| 0.4988 | 0.6437 | 0.6390      | 0.6500      | **0.6505**    | 0.6428 | **0.6728**      |

---

## Assessment

### The two bottleneck types

Every dataset falls into one of two categories, and identifying which drives the
correct choice of improvement strategy.

**Ranking-bottlenecked**: relevant documents are already in the top-100 candidate pool
(high recall ceiling), but they are not ranked high enough to appear in the top-10
evaluated by NDCG@10. The fix is better re-ranking, not better retrieval.

**Retrieval-bottlenecked**: many relevant documents are absent from the top-100 pool
entirely. No re-ranker can surface them. The fix is larger retrieval pools or better
first-stage retrievers.

### Per-dataset diagnosis

**scifact — purely ranking-bottlenecked**
- Dense Recall@100 = 0.884; Union ceiling = 0.907.
- Every fusion method (including Static wRRF α=0.5) reaches the union ceiling of 0.907.
- Adding BM25 candidates adds no new relevant documents — the union ceiling equals the
  fused recall, meaning all relevant docs in the union are already in Dense's top-100.
- Dense NDCG@10 = 0.670, so the system retrieves 90.7% of relevant docs but only ranks
  a fraction in the top-10. A cross-encoder re-ranker on the Dense top-100 alone would
  directly close this gap.

**nfcorpus — severely retrieval-bottlenecked**
- Dense Recall@100 = 0.264; Union ceiling = 0.302 — the lowest recall of any dataset
  by a large margin.
- Even the combined BM25+Dense top-100 contains only 30% of relevant documents.
  nfcorpus queries have many relevant documents per query (broad medical topics), so
  any fixed top-100 pool captures a small fraction.
- All routing methods lie between BM25 and the union; the routing signal has almost
  no effect because the binding constraint is the retrieval stage, not the ranking stage.
- Expanding the retrieval pool (top-500 or top-1000) or using a domain-specific
  retriever would be the correct first lever here. A re-ranker cannot help with
  documents that were never retrieved.

**arguana — ranking-bottlenecked, Dense dominant**
- Dense Recall@100 = 0.956; wRRF (weak) and Union ceiling = 0.978.
- Dense alone almost saturates the recall ceiling. BM25 contributes a marginal 2.2
  percentage-point gain via the union.
- BM25 Recall@100 = 0.778 — much lower, confirming that argument-to-argument
  matching is purely a semantic task where exact-term overlap is misleading.
- Dense NDCG@10 = 0.379 against a recall@100 of 0.956: the relevant document is
  almost always retrieved but consistently ranked below the top-10 by the dense
  retriever alone. A re-ranker would have high-quality input.

**fiqa — mixed, re-ranking dominant**
- Dense Recall@100 = 0.685; Union ceiling = 0.708 (+0.023 over Dense).
- BM25 brings in 2.3 pp of additional relevant documents that Dense misses, making
  fusion at the candidate level modestly worthwhile.
- **Static wRRF recall (0.650) is lower than Dense recall (0.685)**: mixing BM25
  candidates at α=0.5 pushes some Dense-retrieved relevant documents below rank 100.
  The router correctly compensates — wRRF (strong) recovers to 0.687 ≈ Dense.
  This confirms that on fiqa, the router's primary job is preventing BM25 from
  diluting Dense's candidate pool, not boosting recall.

**scidocs — mixed bottleneck**
- Dense Recall@100 = 0.430; Union ceiling = 0.470 (+0.040 over Dense).
- BM25 contributes meaningful additional relevant candidates (+4 pp union gain),
  which is consistent with scidocs being a citation-retrieval task where exact term
  overlap on paper titles matters.
- Routing modestly improves on Dense (wRRF strong = 0.443 vs Dense = 0.430), but
  the ceiling of 0.470 limits overall potential. Both a stronger first-stage retriever
  and better re-ranking would help here.

### Macro summary

| Method       | Recall@100 | vs Dense  |
|--------------|------------|-----------|
| Union (ceil) | 0.6728     | +0.0291   |
| wRRF (strong)| 0.6505     | +0.0068   |
| wRRF (weak)  | 0.6500     | +0.0063   |
| Dense        | 0.6437     | —         |
| MoE          | 0.6428     | −0.0009   |
| Static wRRF  | 0.6390     | −0.0047   |
| BM25         | 0.4988     | −0.1449   |

The routing methods (wRRF weak/strong) improve macro Recall@100 by ~0.006-0.007
over Dense alone. The union ceiling is 0.029 above Dense — meaning BM25 brings
additional relevant candidates that current fusion does not fully surface into
the top-100. A re-ranker on the full union (up to 200 docs per query) rather than
on a fixed top-100 would capture this residual gain.

**Static wRRF reduces recall below Dense** (0.639 vs 0.644): the fixed α=0.5 lets
BM25 displace Dense-retrieved relevant documents from the fused top-100. Routing
(which drives α toward 0 on dense-preferred datasets) is necessary just to preserve
Dense's recall, not only to improve it.

**MoE recall (0.643) is essentially equal to Dense (0.644)**: the MoE meta-learner
is not making the candidate pool worse but is also not meaningfully expanding it.
Its benefit, if any, comes from ranking quality within the candidates (NDCG@10),
not from recall.

### Implications for improvement

1. **Cross-encoder re-ranker on the union candidate pool** — highest expected gain.
   On scifact (recall ceiling 0.907) and arguana (recall ceiling 0.978), the
   relevant documents are almost certainly in the pool; a cross-encoder (e.g.
   cross-encoder/ms-marco-MiniLM-L-12-v2) on the BM25∪Dense top-100 would re-rank
   them to the top-10 far more accurately than wRRF. The routing wRRF step becomes
   the candidate generator for the re-ranker.

2. **Larger retrieval pools for nfcorpus** — routing and re-ranking cannot help when
   recall@100 is 0.30. The retrieval depth (top-k) must be increased for this dataset,
   or a domain-adapted retriever must replace the general-purpose dense model.

3. **Routing is already necessary to protect Dense recall on fiqa** — Static wRRF
   hurts recall; routing corrects this. This is a concrete, measurable benefit of
   the router that is separate from NDCG@10 gains.
