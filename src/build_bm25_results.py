import numpy as np
from tqdm import tqdm
from utils import load_config, load_pickle, save_pickle

def main():
    config = load_config()
    paths = config['paths']
    top_k = config['benchmark']['top_k']

    print("Building BM25 results...")

    print("Loading BM25 index...")
    bm25_data = load_pickle(paths['bm25_index'])
    bm25_model = bm25_data['model']
    bm25_doc_map = bm25_data['doc_ids']

    print("Loading tokenized queries...")
    tokenized_queries = load_pickle(paths['tokenized_queries'])

    print(f"Scoring {len(tokenized_queries)} queries against the index...")
    bm25_results = {}
    bm25_rankings = {}

    for qid, tokens in tqdm(tokenized_queries.items(), desc="BM25 Search"):
        scores = bm25_model.get_scores(tokens)
        top_n_indices = np.argsort(scores)[::-1][:top_k]

        bm25_results[qid] = {}
        ranking = []
        for idx in top_n_indices:
            doc_id = bm25_doc_map[idx]
            bm25_results[qid][doc_id] = float(scores[idx])
            ranking.append(doc_id)

        bm25_rankings[qid] = ranking

    output_file = paths['bm25_results']
    save_pickle({'results': bm25_results, 'rankings': bm25_rankings}, output_file)

    print(f"BM25 results saved to {output_file}")

if __name__ == "__main__":
    main()
