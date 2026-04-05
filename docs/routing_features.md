# Routing Features — Weak Signal Classifier

This document defines the 16 features used to train the weak-signal query router.
All features are computed at query time from cached preprocessing artifacts and
retrieval results, with **no access to relevance judgments**.

---

## Notation

| Symbol | Meaning |
|---|---|
| $q$ | A single query |
| $\text{tokens}(q)$ | Stemmed, lowercased token list of $q$ |
| $\text{tokens}^*(q)$ | Same list with stopwords removed |
| $n_q = \lvert\text{tokens}(q)\rvert$ | Total token count |
| $n_q^* = \lvert\text{tokens}^*(q)\rvert$ | Non-stopword token count |
| $N$ | Total number of documents in the corpus |
| $\text{df}(t)$ | Number of corpus documents containing token $t$ |
| $\text{cf}(t)$ | Total occurrences of token $t$ across all corpus documents |
| $T$ | Total token occurrences in the entire corpus |
| $V$ | Vocabulary size (number of unique tokens in the corpus) |
| $\mathcal{B}(q)$ | BM25 ranked list: $[(d_1, s_1^B), (d_2, s_2^B), \ldots]$, score-descending |
| $\mathcal{D}(q)$ | Dense ranked list: $[(d_1, s_1^D), (d_2, s_2^D), \ldots]$, score-descending |
| $\tilde{s}_i^B,\ \tilde{s}_i^D$ | Min-max normalized scores within the query's result list |
| $k_\text{agree}$ | Overlap window for agreement features (config: `overlap_k`) |
| $k_f$ | Window for distribution/rank features (config: `feature_stat_k`) |
| $\varepsilon$ | Small numerical stabilizer (config: `epsilon`) |
| $\alpha_\text{sm}$ | Laplace smoothing constant for cross-entropy (config: `ce_smoothing_alpha`) |

### Min-max normalization

All score-derived features use per-query min-max normalization to make
them comparable across queries and corpora:

$$\tilde{s}_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min} + \varepsilon}$$

If $s_{\max} - s_{\min} < \varepsilon$, all normalized scores are set to 0.

---

## Group A — Query Surface

Features derived solely from the raw query text, independent of any index.

### `query_length`

Number of stemmed tokens in the query before stopword removal.

$$\text{query\_length} = n_q$$

### `stopword_ratio`

Fraction of query tokens that are English stopwords (after stemming, so the
stopword list is itself stemmed for a fair comparison).

$$\text{stopword\_ratio} = \frac{\lvert\{t \in \text{tokens}(q) : t \in \mathcal{S}\}\rvert}{n_q}$$

where $\mathcal{S}$ is the set of stemmed English stopwords. Returns 0 for an empty query.

BM25 effectively ignores stopwords through low IDF; dense embeddings treat them as
part of the overall query meaning. A high stopword ratio may indicate that dense
better captures the query's intent.

### `has_question_word`

Binary indicator: does the query start with a natural-language question word?

$$\text{has\_question\_word} = \mathbf{1}\!\left[\text{lower}(w_1) \in \{\textit{who, what, when, where, why, how, which, whose, whom}\}\right]$$

where $w_1$ is the first whitespace-delimited token of the **raw** (unstemmed) query text.
"Why" and "whose" can change under stemming, so this check is on the raw text.

Question-form queries have a well-formed semantic intent that dense embeddings
capture well, while BM25 reduces them to a bag of keywords.

---

## Group B — Query–Corpus Vocabulary Match

Features that measure how well the query's vocabulary aligns with what the corpus contains.
All use $\text{tokens}^*(q)$ (stopwords removed) unless noted.

### `average_idf`

Mean smoothed IDF of the non-stopword query tokens.

$$\text{idf}(t) = \ln\!\frac{N + 1}{\text{df}(t) + 1} + 1$$

$$\text{average\_idf} = \frac{1}{n_q^*} \sum_{t \in \text{tokens}^*(q)} \text{idf}(t)$$

The $+1$ smoothing keeps the formula finite for unseen tokens and avoids log(0).
Higher values indicate that the query terms are rare in the corpus, which typically
means BM25 can discriminate well on those terms.

### `max_idf`

The single highest IDF value among the non-stopword query tokens.

$$\text{max\_idf} = \max_{t \in \text{tokens}^*(q)}\ \text{idf}(t)$$

Captures "does the query contain at least one highly specific term?" even when
the average is pulled down by more common terms.

### `rare_term_ratio`

Fraction of non-stopword query tokens whose IDF exceeds the within-query mean plus
one within-query standard deviation. Measures whether the query is dominated by
rare, specific terminology.

$$\sigma_\text{idf} = \operatorname{std}\!\left(\{\text{idf}(t) : t \in \text{tokens}^*(q)\}\right)$$

$$\text{rare\_term\_ratio} = \frac{\lvert\{t \in \text{tokens}^*(q) : \text{idf}(t) \geq \overline{\text{idf}} + \sigma_\text{idf}\}\rvert}{n_q^*}$$

where $\overline{\text{idf}}$ is `average_idf`. Note: the within-query IDF standard
deviation is used as a threshold parameter here but is not itself a feature.

### `cross_entropy`

Cross-entropy between the empirical query unigram distribution and the
Laplace-smoothed corpus unigram distribution, computed over the non-stopword
query tokens.

Let $p_q(t) = \text{count}(t\ \text{in}\ \text{tokens}^*(q))\ /\ n_q^*$ be the
empirical query distribution. The smoothed corpus probability is:

$$\hat{p}_c(t) = \frac{\text{cf}(t) + \alpha_\text{sm}}{T + \alpha_\text{sm} \cdot V}$$

Then:

$$\text{cross\_entropy} = H(p_q \| \hat{p}_c) = -\sum_{t \in \text{tokens}^*(q)} p_q(t)\, \log_2 \hat{p}_c(t)$$

In practice this is computed by iterating over the token list (with repeats), so a
token appearing twice contributes twice — which is exactly the $p_q(t)$ weighting.
Laplace smoothing assigns a small but non-zero probability to tokens absent from
the corpus, so no tokens are excluded.

Distinct from `average_idf`: IDF is based on *document* frequency (how many
documents contain a term), while cross-entropy is based on *corpus frequency*
(how often a term occurs across all documents). A technical term appearing
intensively in a few documents can have high IDF but low surprisal, or vice versa.

---

## Group C — Retriever Confidence

Features that measure how decisively each retriever ranks its top result above the rest.
All use min-max normalized scores $\tilde{s}$.

### `top_dense_score`

The normalized score of the top-ranked document in the dense result list.

$$\text{top\_dense\_score} = \tilde{s}_1^D$$

Because scores are min-max normalized within each query's result list, this value
is always 1.0 unless all scores are identical (in which case it is 0.0). Its signal
is therefore mainly whether the dense retriever produces a degenerate ranking.

### `top_sparse_score`

Analogous for BM25.

$$\text{top\_sparse\_score} = \tilde{s}_1^B$$

### `dense_confidence`

Score margin between the top-1 and top-2 dense results. Measures how much the
top dense document stands out.

$$\text{dense\_confidence} = \tilde{s}_1^D - \tilde{s}_2^D$$

Set to $\tilde{s}_1^D$ if the result list contains only one document.

### `sparse_confidence`

Analogous for BM25.

$$\text{sparse\_confidence} = \tilde{s}_1^B - \tilde{s}_2^B$$

---

## Group D — Retriever Agreement

Features that measure how much BM25 and dense agree on which documents are relevant.

Let $\mathcal{T}^B_{k}$ and $\mathcal{T}^D_{k}$ denote the top-$k$ document sets
from BM25 and dense respectively.

### `overlap_at_k`

Fraction of the top-$k_\text{agree}$ window that both retrievers agree on.

$$\text{overlap\_at\_k} = \frac{\lvert \mathcal{T}^B_{k_a} \cap \mathcal{T}^D_{k_a} \rvert}{k_\text{agree}}$$

Since both lists have exactly $k_\text{agree}$ documents this equals the overlap
coefficient, but is **not** Jaccard (which would divide by the union).
High overlap suggests both retrievers surface the same evidence, which usually
makes fusion less critical.

### `first_shared_doc_rank`

The average rank position at which the first shared document appears in both lists.

Let $\mathcal{C} = \mathcal{T}^B_{k_f} \cap \mathcal{T}^D_{k_f}$ be the shared
documents within the top-$k_f$ window:

$$\text{first\_shared\_doc\_rank} = \min_{d \in \mathcal{C}}\ \frac{r^B(d) + r^D(d)}{2}$$

where $r^B(d)$ and $r^D(d)$ are the 1-indexed ranks of $d$ in their respective
top-$k_f$ lists. If $\mathcal{C} = \emptyset$, the value is set to $k_f + 1$.

A low value means agreement starts near the top of both rankings; a high value
means the retrievers agree only on lower-ranked documents (or not at all).

### `spearman_topk`

Spearman rank correlation computed over the shared documents within the top-$k_f$
window, using the ranks each document has in its respective retriever's list.

For $\lvert\mathcal{C}\rvert \geq 2$:

$$\text{spearman\_topk} = 1 - \frac{6 \displaystyle\sum_{d \in \mathcal{C}} \bigl(r^B(d) - r^D(d)\bigr)^2}{\lvert\mathcal{C}\rvert \bigl(\lvert\mathcal{C}\rvert^2 - 1\bigr)}$$

For $\lvert\mathcal{C}\rvert < 2$: value is 0.

Captures whether the two retrievers not only surface the same documents but also
rank them in the same order. Complements `agreement` (set overlap) and
`first_shared_doc_rank` (where overlap begins).

---

## Group E — Ranking Distribution Shape

Features that measure how concentrated or spread the retrieval scores are across
the top-$k_f$ results. A highly concentrated distribution suggests the retriever
is confident; a flat distribution suggests uncertainty.

Let $\tilde{s}^R_1 \geq \cdots \geq \tilde{s}^R_{k_f}$ be the normalized top-$k_f$
scores for retriever $R \in \{B, D\}$. Define the score-mass probability:

$$p_i^R = \frac{\max(\tilde{s}^R_i,\ 0)}{\displaystyle\sum_{j=1}^{k_f} \max(\tilde{s}^R_j,\ 0)}$$

If the total mass is $\leq \varepsilon$, entropy is set to 0.

### `dense_entropy_topk`

Shannon entropy (bits) of the normalized dense score distribution over the top-$k_f$ results.

$$\text{dense\_entropy\_topk} = -\sum_{i=1}^{k_f} p_i^D \log_2 \max\!\left(p_i^D,\ \varepsilon\right)$$

### `sparse_entropy_topk`

Same formula applied to BM25 scores.

$$\text{sparse\_entropy\_topk} = -\sum_{i=1}^{k_f} p_i^B \log_2 \max\!\left(p_i^B,\ \varepsilon\right)$$

---

## Soft Label

The target value for each query. Derived from query-level NDCG@k of each retriever,
requiring relevance judgments — **used only at training time, not at inference**.

$$\text{label}(q) = \frac{1}{2}\left(\frac{\text{NDCG@}k^B(q) - \text{NDCG@}k^D(q)}{\max\!\left(\text{NDCG@}k^B(q),\;\text{NDCG@}k^D(q)\right) + \varepsilon} + 1\right) \in [0, 1]$$

- $\text{label} = 1$ → sparse (BM25) is strictly better
- $\text{label} = 0$ → dense is strictly better
- $\text{label} = 0.5$ → tie (both return zero NDCG, or equal NDCG)

The label is clipped to $[0, 1]$ to guard against floating-point edge cases.

---

## Weighted RRF Fusion

At inference the router predicts $\hat{\alpha}(q) \in [0, 1]$ from the 17 features.
The final document score is:

$$\text{score}(q, d) = \hat{\alpha}(q) \cdot \frac{1}{k_\text{rrf} + r^B(d)} + \bigl(1 - \hat{\alpha}(q)\bigr) \cdot \frac{1}{k_\text{rrf} + r^D(d)}$$

where $k_\text{rrf} = 60$ is the RRF damping constant and $r^B(d)$, $r^D(d)$ are
the 1-indexed BM25 and dense ranks. Documents absent from one list are assigned
rank = list length + 1 (just-below-tail imputation).

When $\hat{\alpha} = 0.5$ this reduces to static RRF.
