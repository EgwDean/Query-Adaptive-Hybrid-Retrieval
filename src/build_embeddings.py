import os
import json
import torch
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import load_config, ensure_dir

def load_corpus_batch_generator(filepath, batch_size):
    """Yields batches of (doc_id, text) for efficient processing."""
    batch_ids = []
    batch_texts = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                full_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                
                batch_ids.append(doc["_id"])
                batch_texts.append(full_text)
                
                if len(batch_texts) >= batch_size:
                    yield batch_ids, batch_texts
                    batch_ids = []
                    batch_texts = []
            except json.JSONDecodeError:
                continue
    
    if batch_texts:
        yield batch_ids, batch_texts

def main():
    config = load_config()
    paths = config['paths']
    embeddings_config = config['embeddings']
    
    print("Building vector index...")
    
    corpus_path = paths['corpus']
    output_file = paths['corpus_embeddings']
    doc_ids_file = paths['vector_doc_ids']
    batch_size = embeddings_config['batch_size']
    model_name = embeddings_config['model_name']
    
    ensure_dir(os.path.dirname(output_file))
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus not found at {corpus_path}")
        return

    print(f"Loading model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name, device=device)

    all_embeddings = []
    all_doc_ids = []
    
    print("Estimating corpus size...")
    total_docs = sum(1 for _ in open(corpus_path, 'r', encoding='utf-8'))
    
    print(f"Encoding {total_docs} documents...")
    
    loader = load_corpus_batch_generator(corpus_path, batch_size)
    
    for batch_ids, batch_texts in tqdm(loader, total=total_docs//batch_size, desc="Encoding batches"):
        embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu())
        all_doc_ids.extend(batch_ids)

    print("Concatenating embeddings...")
    final_tensor = torch.cat(all_embeddings, dim=0)
    
    print(f"Final embedding shape: {final_tensor.shape}")
    print(f"Saving embeddings to {output_file}...")
    torch.save(final_tensor, output_file)
    
    print(f"Saving document IDs to {doc_ids_file}...")
    with open(doc_ids_file, "wb") as f:
        pickle.dump(all_doc_ids, f)

    print(f"Vector index saved to {output_file}")

if __name__ == "__main__":
    main()