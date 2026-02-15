import os
import json
import pickle
import yaml
from collections import Counter
from tqdm import tqdm

CONFIG_PATH = "config.yaml"

def load_config():
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    config = load_config()
    paths = config['paths']
    
    print("Building frequency index...")
    
    tokenized_file = paths['tokenized_corpus']
    output_file = paths['freq_index']
    ensure_dir(os.path.dirname(output_file))
    
    global_counter = Counter()
    total_tokens = 0
    
    with open(tokenized_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting tokens"):
            entry = json.loads(line)
            tokens = entry['tokens']
            
            global_counter.update(tokens)
            total_tokens += len(tokens)

    with open(output_file, "wb") as f:
        pickle.dump({"counts": dict(global_counter), "total_tokens": total_tokens}, f)
    
    print(f"Frequency index saved to {output_file}")

if __name__ == "__main__":
    main()