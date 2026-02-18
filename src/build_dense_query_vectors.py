import os
import torch
from sentence_transformers import SentenceTransformer
from utils import load_config, load_queries, save_pickle, ensure_dir

def main():
    config = load_config()
    paths = config['paths']
    model_name = config['embeddings']['model_name']

    print("Building dense query vectors...")

    queries = load_queries(paths['queries'])
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Encoding {len(query_ids)} queries...")
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

    embeddings_file = paths['dense_query_vectors']
    ids_file = paths['dense_query_ids']
    ensure_dir(os.path.dirname(embeddings_file))

    print(f"Saving query embeddings to {embeddings_file}...")
    torch.save(query_embeddings.cpu(), embeddings_file)

    print(f"Saving query IDs to {ids_file}...")
    save_pickle(query_ids, ids_file)

    print(f"Dense query vectors saved ({query_embeddings.shape})")

if __name__ == "__main__":
    main()
