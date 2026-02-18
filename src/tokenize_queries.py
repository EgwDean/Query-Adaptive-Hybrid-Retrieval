from nltk.stem import SnowballStemmer
from utils import load_config, load_queries, save_pickle

def main():
    config = load_config()
    paths = config['paths']

    print("Tokenizing queries...")

    queries = load_queries(paths['queries'])
    stemmer = SnowballStemmer(config['preprocessing']['stemmer_language'])

    print(f"Stemming {len(queries)} queries...")
    tokenized_queries = {}
    for qid, text in queries.items():
        tokenized_queries[qid] = [stemmer.stem(t) for t in text.lower().split()]

    output_file = paths['tokenized_queries']
    save_pickle(tokenized_queries, output_file)

    print(f"Tokenized queries saved to {output_file}")

if __name__ == "__main__":
    main()
