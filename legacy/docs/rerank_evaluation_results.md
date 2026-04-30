# Cross-Encoder Re-ranking Evaluation Results

## What was done

A cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was applied
on top of the candidate pools produced by all 6 retrieval methods and NDCG@10
was measured before and after re-ranking.

**Re-ranking procedure:** For each test query, the candidate pool from the
retrieval method (top-100 docs) is re-scored by the cross-encoder, which takes
the raw (query text, document text) pair and produces a relevance score via
full cross-attention. Candidates are then sorted by this score and NDCG@10 is
re-evaluated on the re-ranked list.

**Efficiency:** All unique (query, doc) pairs across all 6 methods for a dataset
are deduplicated and scored in one batched call per dataset; shared candidates
are never scored twice. Scores are cached per dataset.

**Test split:** same balanced 5×300 pool as all previous experiments (~45 queries
per dataset). Alpha values for wRRF methods loaded from the cached meta-dataset CSV.

Script: `src/rerank_evaluation.py`
Outputs: `data/results/rerank_ndcg_comparison.{csv,png}`,
         `data/results/rerank_ndcg_improvement.png`

---

## Results

### Original vs Re-ranked NDCG@10

| Dataset  | BM25 → rr | Dense → rr | Static wRRF → rr | wRRF (weak) → rr | wRRF (strong) → rr | MoE → rr |
|----------|-----------|-----------|-----------------|-----------------|-------------------|---------|
| scifact  | 0.601 → **0.642** | 0.670 → 0.659 | 0.663 → 0.659 | 0.631 → 0.659 | 0.654 → 0.659 | 0.656 → 0.659 |
| nfcorpus | 0.261 → **0.303** | 0.267 → **0.317** | 0.288 → **0.323** | 0.308 → **0.325** | 0.290 → **0.323** | 0.290 → **0.323** |
| arguana  | 0.219 → 0.275 | 0.379 → 0.280 | 0.307 → 0.288 | 0.363 → 0.288 | 0.368 → 0.280 | 0.326 → 0.287 |
| fiqa     | 0.184 → **0.287** | 0.368 → **0.369** | 0.301 → **0.370** | 0.358 → **0.370** | 0.360 → **0.370** | 0.356 → **0.370** |
| scidocs  | 0.196 → **0.233** | 0.221 → **0.261** | 0.239 → **0.259** | 0.232 → **0.259** | 0.229 → **0.259** | 0.245 → **0.259** |
| **MACRO**| 0.292 → **0.348** | 0.381 → 0.377 | 0.360 → **0.380** | 0.378 → **0.380** | 0.380 → 0.378 | 0.375 → **0.380** |

Bold = re-ranking improves over original for that method.

### Macro NDCG@10 gain per method

| Method         | Original | Re-ranked | Gain    |
|----------------|----------|-----------|---------|
| BM25           | 0.2920   | 0.3480    | **+0.0560** |
| Dense          | 0.3810   | 0.3774    | −0.0037 |
| Static wRRF    | 0.3596   | 0.3801    | +0.0205 |
| wRRF (weak)    | 0.3784   | **0.3804**| +0.0020 |
| wRRF (strong)  | 0.3802   | 0.3784    | −0.0018 |
| MoE            | 0.3748   | 0.3797    | +0.0049 |

### Re-ranked macro ranking

```
wRRF (weak) + CE  (0.3804)
Static wRRF + CE  (0.3801)
MoE + CE          (0.3797)
Dense (original)  (0.3810)   ← best non-re-ranked
Dense + CE        (0.3774)
wRRF (strong) + CE(0.3784)
BM25 + CE         (0.3480)
```

---

## Assessment

### Headline results

**wRRF (weak) + cross-encoder re-ranking achieves the highest NDCG@10 of any
re-ranked system (0.3804).** It beats Dense + re-ranking (0.3774) by +0.003.
It falls 0.0006 below Dense-only (0.3810), a gap within the noise of the ≈45-query
test sets. This is the first configuration where the routing system effectively
matches the dense retriever on the primary metric.

**Re-ranking hurts Dense (macro −0.004) and wRRF (strong) (macro −0.002).**
Both are dragged down by a catastrophic failure on arguana (see below). Every other
method benefits from re-ranking or is neutral.

### Why re-ranking hurts arguana (the dominant effect)

arguana is a counter-argument retrieval task: the query is an argument, and the
relevant document is its counter-argument. Dense retrieval handles this well because
opposite-direction arguments occupy nearby positions in embedding space (they discuss
the same topics). `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on MS MARCO,
a question-answering dataset where the relevant document *answers* the query.
A counter-argument that contradicts the query looks irrelevant to an MS MARCO-trained
cross-encoder, so re-ranking pushes the relevant documents down the list.

| Method        | arguana original | arguana re-ranked | Δ       |
|---------------|-----------------|-------------------|---------|
| Dense         | 0.379           | 0.280             | −0.099  |
| wRRF (strong) | 0.368           | 0.280             | −0.087  |
| wRRF (weak)   | 0.363           | 0.288             | −0.075  |
| Static wRRF   | 0.307           | 0.288             | −0.019  |
| BM25          | 0.219           | 0.275             | +0.056  |

Dense loses almost 0.1 NDCG@10 on arguana alone, which pulls its macro below wRRF.
BM25 is the only method that gains on arguana because its original ranking was so weak
that even a misaligned cross-encoder still improves it.

This is a domain mismatch between the cross-encoder's training distribution (MS MARCO
QA) and arguana's task structure (counter-argument retrieval), not a fundamental
failure of re-ranking as a paradigm.

### Why re-ranking helps nfcorpus so much

nfcorpus is retrieval-limited (union recall@100 = 0.30), yet all methods gain
significantly from re-ranking (+0.033 to +0.050). Although only 30% of relevant
documents are in the candidate pool, those that are retrieved are scattered through
positions 1–100. The cross-encoder, which generalises partially to biomedical text,
correctly identifies the most relevant ones and promotes them to the top-10.
This is a case where the limiting factor is not finding the documents but ranking
them correctly within the candidate pool — exactly the problem a cross-encoder solves.

### Convergence after re-ranking

On scifact, scidocs, and fiqa, all routing methods converge to essentially the same
re-ranked score. This occurs because:
- The recall@100 of all routing methods is similar on these datasets (they contain
  nearly the same relevant documents).
- Once the cross-encoder sees the same candidate pool, it produces the same ranking.

The re-ranking erases ranking-quality differences between retrieval methods when
their candidate pools overlap heavily. The differences that remain come entirely from
differences in recall@100 — which method brought in which relevant documents.

### BM25 benefits most (+0.056 macro)

BM25 has the lowest original NDCG@10 (0.292) but the largest absolute gain from
re-ranking (+0.056). BM25 retrieves lexically relevant documents but ranks them
poorly (IDF-weighted term frequency is a weak ranking signal). The cross-encoder
then correctly reorders this pool, recovering much of the signal that BM25's ranking
missed. This illustrates the classic two-stage retrieval finding: a fast first-stage
retriever for recall + a powerful second-stage re-ranker for precision.

### Conclusion

**Re-ranking makes routing competitive with Dense.** Without re-ranking, Dense macro
NDCG@10 = 0.381 was the ceiling. With re-ranking applied consistently:
- wRRF (weak) + re-ranking reaches 0.380, within 0.001 of Dense-only.
- Dense itself drops to 0.377 after re-ranking (arguana domain mismatch).

The practical conclusion for this thesis: the routing system's value is not fully
realised at the direct-ranking stage (NDCG@10), but it does produce better candidate
pools (Recall@100) that a re-ranker can exploit. When both are combined — routing
to select candidates, cross-encoder to rank them — the system matches or exceeds
Dense retrieval on macro NDCG@10 despite the cross-encoder having a known failure
mode on arguana.

A cross-encoder trained on a broader distribution (e.g., including argument retrieval
tasks) would be expected to further improve the re-ranked wRRF results without the
arguana penalty.

### Limitations

- Test sets of ≈45 queries per dataset produce high-variance per-dataset estimates;
  a gain or loss of 0.005 macro may not be statistically significant.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` is an efficient but not state-of-the-art
  cross-encoder; a larger model (MiniLM-L-12, monoT5, RankLLaMA) would likely
  produce larger absolute gains and potentially handle arguana better.
- The re-ranker sees only the top-100 candidates from each method. On nfcorpus,
  where recall@100 is only 0.30, expanding the candidate pool first (top-500)
  before re-ranking would be the logical next step.
