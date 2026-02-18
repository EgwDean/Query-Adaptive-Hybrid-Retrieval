import torch
from tqdm import tqdm
from sentence_transformers import util
from utils import load_config, load_pickle, save_pickle

def main():
    config = load_config()
    paths = config['paths']
    top_k = config['benchmark']['top_k']
    search_cfg = config['dense_search']

    print("Building dense results...")

    print("Loading corpus embeddings...")
    corpus_emb = torch.load(paths['corpus_embeddings'], map_location='cpu')

    print("Loading document ID mapping...")
    vector_doc_map = load_pickle(paths['vector_doc_ids'])

    print("Loading query embeddings...")
    query_emb = torch.load(paths['dense_query_vectors'], map_location='cpu')
    query_ids = load_pickle(paths['dense_query_ids'])

    print(f"Running semantic search ({len(query_ids)} queries, top_k={top_k})...")
    hits_list = util.semantic_search(
        query_emb,
        corpus_emb,
        top_k=top_k,
        query_chunk_size=search_cfg['query_chunk_size'],
        corpus_chunk_size=search_cfg['corpus_chunk_size']
    )

    dense_results = {}
    dense_rankings = {}

    for i, hits in enumerate(tqdm(hits_list, desc="Collecting results")):
        qid = query_ids[i]
        dense_results[qid] = {}
        dense_rankings[qid] = []

        for hit in hits:
            doc_id = vector_doc_map[hit['corpus_id']]
            dense_results[qid][doc_id] = float(hit['score'])
            dense_rankings[qid].append(doc_id)

    output_file = paths['dense_results']
    save_pickle({'results': dense_results, 'rankings': dense_rankings}, output_file)

    print(f"Dense results saved to {output_file}")

if __name__ == "__main__":
    main()
