# Routing Feature Notes

## 1) Active routing features

Current active features used by logistic routing:

- `cross_entropy`
- `agreement`
- `overlap_at_3`
- `query_length`
- `dense_confidence`
- `sparse_confidence`
- `confidence_gap`
- `top_dense_score`
- `top_sparse_score`
- `top_score_gap`
- `average_idf`
- `max_idf`
- `idf_std`
- `rare_term_ratio`
- `stopword_ratio`

Key formulas:

- Per-query score normalization:
  - $\hat{s}_i = (s_i - \min_j s_j)/(\max_j s_j - \min_j s_j + \epsilon)$
- Agreement:
  - $\mathrm{agreement} = |\mathrm{TopK}_{\text{sparse}} \cap \mathrm{TopK}_{\text{dense}}| / K$
- Top-3 overlap:
  - $\mathrm{overlap\_at\_3} = |\mathrm{Top3}_{\text{sparse}} \cap \mathrm{Top3}_{\text{dense}}| / 3$
- Confidence margins:
  - $\mathrm{dense\_confidence} = \hat{s}^{(d)}_1 - \hat{s}^{(d)}_2$
  - $\mathrm{sparse\_confidence} = \hat{s}^{(s)}_1 - \hat{s}^{(s)}_2$
- Gap features:
  - $\mathrm{confidence\_gap} = \mathrm{dense\_confidence} - \mathrm{sparse\_confidence}$
  - $\mathrm{top\_score\_gap} = \mathrm{top\_dense\_score} - \mathrm{top\_sparse\_score}$
- Smoothed IDF:
  - $\mathrm{idf}(t)=\log((N+1)/(\mathrm{df}(t)+1))+1$
- Rare-term ratio threshold:
  - $\tau=\mathrm{average\_idf}+\mathrm{idf\_std}$
  - $\mathrm{rare\_term\_ratio}=|\{t:\mathrm{idf}(t)\ge\tau\}|/|Q|$

## 2) Previously considered / omitted / replaced features

Documented for thesis discussion (not active as standalone raw features):

- Raw `top_dense_score` (no normalization)
  - Formula: top dense score from original dense retrieval scale.
  - Rationale: incomparable with BM25 score scale across retrievers.

- Raw `top_sparse_score` (no normalization)
  - Formula: top BM25 score from original sparse retrieval scale.
  - Rationale: incomparable with dense score scale.

- Raw `dense_confidence` (no normalization)
  - Formula: $s^{(d)}_1 - s^{(d)}_2$ on raw dense scores.
  - Rationale: sensitive to retriever-specific scale and score spread.

- Raw `sparse_confidence` (no normalization)
  - Formula: $s^{(s)}_1 - s^{(s)}_2$ on raw BM25 scores.
  - Rationale: not directly comparable to dense confidence magnitudes.

- Cross-retriever absolute score comparisons on raw scales
  - Example: direct subtraction between raw dense and raw BM25 scores.
  - Rationale: violates scale comparability assumptions.

- `average_idf` only as sole rarity signal
  - Rationale: misses tail behavior and rarity concentration.

- `max_idf` only as sole rarity signal
  - Rationale: too sensitive to one token and unstable alone.

- Overlap@100 as primary agreement signal
  - Formula: $|\mathrm{Top100}_s \cap \mathrm{Top100}_d|/100$.
  - Rationale: weaker top-rank discrimination than overlap@10/overlap@3.

- Redundant absolute-score features superseded by gap features
  - Rationale: normalized gap features better encode relative retriever strength.

## 3) Notes on score normalization

- Normalization is applied per query and per retriever list independently.
- If a result list is empty, normalization returns an empty list.
- If score range is near zero, normalized scores are set to zero to avoid unstable divisions.
- Normalized scores are used only for routing feature computation.
- Raw rankings/scores remain unchanged for:
  - NDCG label construction
  - static RRF
  - dynamic wRRF fusion
