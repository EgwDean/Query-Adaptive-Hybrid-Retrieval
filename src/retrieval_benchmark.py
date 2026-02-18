import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from utils import load_config, load_qrels, load_pickle, ensure_dir
from fusion_methods import (
    apply_rrf_fusion,
    apply_wrrf_fusion,
    compute_divergence_alpha,
    get_sorted_docs
)

def calculate_ndcg_at_k(result_ids, qrels, k=10):
    """NDCG@k for a single query given ranked doc IDs and ground-truth relevance."""
    top_hits = result_ids[:k]
    if not top_hits:
        return 0.0

    predicted = [qrels.get(doc, 0) for doc in top_hits]
    ideal = sorted(qrels.values(), reverse=True)[:k]

    if len(ideal) < k:
        ideal += [0] * (k - len(ideal))

    return ndcg_score([ideal], [predicted], k=k)

def main():
    config = load_config()
    paths = config['paths']
    bench = config['benchmark']

    print("Running retrieval benchmark...")

    qrels = load_qrels(paths['qrels'])
    ndcg_k = bench['ndcg_k']

    print("Loading BM25 data...")
    bm25_data = load_pickle(paths['bm25_results'])
    sparse_rankings = bm25_data['rankings']

    tokenized_queries = load_pickle(paths['tokenized_queries'])

    print("Loading dense data...")
    dense_data = load_pickle(paths['dense_results'])
    dense_rankings = dense_data['rankings']

    query_ids = [qid for qid in sparse_rankings if qid in qrels]
    print(f"Evaluating on {len(query_ids)} queries.")

    freq_data = load_pickle(paths['freq_index'])

    methods = [
        ("BM25 Only", "bm25_only"),
        ("Dense Only", "dense_only"),
        ("Naive RRF", "rrf"),
        ("JSD (Linear)", "jsd_linear"),
        ("JSD (Sigmoid)", "jsd_sigmoid"),
        ("KLD (Sigmoid)", "kld_sigmoid"),
    ]

    final_metrics = []

    for name, mode in methods:
        ndcg_scores = []
        alpha_values = []

        for qid in tqdm(query_ids, desc=f"Eval {name}"):
            s_rank = sparse_rankings[qid]
            d_rank = dense_rankings[qid]

            if mode == "bm25_only":
                sorted_docs = s_rank
                alpha_values.append(0.0)

            elif mode == "dense_only":
                sorted_docs = d_rank
                alpha_values.append(1.0)

            elif mode == "rrf":
                k_rrf = bench['rrf']['k']
                fused = apply_rrf_fusion(s_rank, d_rank, k_rrf)
                sorted_docs = get_sorted_docs(fused)
                alpha_values.append(0.5)

            else:
                use_sig = "sigmoid" in mode
                metric = "kld" if "kld" in mode else "jsd"

                alpha = compute_divergence_alpha(
                    tokenized_queries[qid], freq_data, config,
                    method=metric, use_sigmoid=use_sig
                )
                alpha_values.append(alpha)

                k_rrf = bench['rrf']['k']
                fused = apply_wrrf_fusion(s_rank, d_rank, alpha, k_rrf)
                sorted_docs = get_sorted_docs(fused)

            score = calculate_ndcg_at_k(sorted_docs, qrels[qid], k=ndcg_k)
            ndcg_scores.append(score)

        avg_ndcg = np.mean(ndcg_scores)
        avg_alpha = np.mean(alpha_values)
        min_alpha = np.min(alpha_values)
        max_alpha = np.max(alpha_values)

        print(f"   > {name}: NDCG@{ndcg_k} = {avg_ndcg:.4f} (Alpha Avg: {avg_alpha:.2f})")

        final_metrics.append({
            "Method": name,
            f"NDCG@{ndcg_k}": avg_ndcg,
            "Min_Alpha": min_alpha,
            "Max_Alpha": max_alpha,
            "Avg_Alpha": avg_alpha,
        })

    results_path = paths['results']
    ensure_dir(os.path.dirname(results_path))
    file_exists = os.path.isfile(results_path)

    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Method", f"NDCG@{ndcg_k}", "Min_Alpha", "Max_Alpha", "Avg_Alpha"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(final_metrics)

    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
