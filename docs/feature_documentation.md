# Feature Documentation

## 1) Current Features (Active)

The supervised routing model uses the following active query-level features.

### cross_entropy
- Formula:
  - $\mathrm{cross\_entropy} = -\frac{1}{|Q|}\sum_{t \in Q} \log_2\left(\frac{\mathrm{tf}(t)+\alpha}{T+\alpha |V|}\right)$
- Explanation:
  - Measures lexical surprisal of query tokens under corpus unigram statistics.

### agreement
- Formula:
  - $\mathrm{agreement} = \frac{|\mathrm{TopK}_{\text{sparse}} \cap \mathrm{TopK}_{\text{dense}}|}{K}$
- Explanation:
  - Captures overlap between sparse and dense retrieved sets at cutoff $K$.

### overlap_at_3
- Formula:
  - $\mathrm{overlap\_at\_3} = \frac{|\mathrm{Top3}_{\text{sparse}} \cap \mathrm{Top3}_{\text{dense}}|}{3}$
- Explanation:
  - Strong top-rank agreement signal focused on the first three documents.

### query_length
- Formula:
  - $\mathrm{query\_length} = |Q_{\text{raw tokens}}|$
- Explanation:
  - Number of tokens in the tokenized query before stopword removal.

### dense_confidence
- Formula:
  - $\mathrm{dense\_confidence} = \hat{s}^{(d)}_1 - \hat{s}^{(d)}_2$
- Explanation:
  - Margin between top-1 and top-2 dense scores after per-query min-max normalization.

### sparse_confidence
- Formula:
  - $\mathrm{sparse\_confidence} = \hat{s}^{(s)}_1 - \hat{s}^{(s)}_2$
- Explanation:
  - Margin between top-1 and top-2 sparse scores after per-query min-max normalization.

### confidence_gap
- Formula:
  - $\mathrm{confidence\_gap} = \mathrm{dense\_confidence} - \mathrm{sparse\_confidence}$
- Explanation:
  - Relative certainty difference between dense and sparse retrievers.

### average_idf
- Formula:
  - $\mathrm{average\_idf} = \frac{1}{|Q|}\sum_{t \in Q}\left(\log\frac{N+1}{\mathrm{df}(t)+1}+1\right)$
- Explanation:
  - Mean smoothed IDF across stopword-filtered query tokens.

### max_idf
- Formula:
  - $\mathrm{max\_idf} = \max_{t \in Q}\left(\log\frac{N+1}{\mathrm{df}(t)+1}+1\right)$
- Explanation:
  - Highest smoothed IDF token in the query.

### idf_std
- Formula:
  - $\mathrm{idf\_std} = \mathrm{std}\left(\{\log\frac{N+1}{\mathrm{df}(t)+1}+1 : t \in Q\}\right)$
- Explanation:
  - Dispersion of token specificity within the query.

### rare_term_ratio
- Formula:
  - $\tau = \mathrm{average\_idf} + \mathrm{idf\_std}$
  - $\mathrm{rare\_term\_ratio} = \frac{|\{t \in Q : \mathrm{idf}(t) \ge \tau\}|}{|Q|}$
- Explanation:
  - Fraction of tokens considered unusually specific for that query.

### stopword_ratio
- Formula:
  - $\mathrm{stopword\_ratio} = \frac{|Q_{\text{stopwords}}|}{|Q_{\text{raw tokens}}|}$
- Explanation:
  - Measures stopword density in the original tokenized query.

## 2) Removed Features

The following features were removed:
- `top_dense_score`
- `top_sparse_score`
- `top_score_gap`

Reason for removal:
- After per-query min-max normalization, top-1 normalized scores are near 1 by construction.
- This makes `top_dense_score` and `top_sparse_score` near-constant.
- Their difference (`top_score_gap`) becomes near 0 and contributes little information.
- Keeping them can add noise/redundancy without improving routing decisions.
