import os
import csv
import json
import pickle
import yaml

CONFIG_PATH = "config.yaml"

def load_config():
    """Load configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(CONFIG_PATH)}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_queries(filepath):
    """Load queries from a JSONL file into a {query_id: text} dictionary."""
    queries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            queries[d['_id']] = d['text']
    return queries

def load_qrels(filepath):
    """Load relevance judgments from a TSV file into a nested dictionary."""
    qrels = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            qid = row['query-id']
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][row['corpus-id']] = int(row['score'])
    return qrels

def save_pickle(data, filepath):
    """Serialize data to a pickle file, creating parent directories as needed."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    """Deserialize data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
