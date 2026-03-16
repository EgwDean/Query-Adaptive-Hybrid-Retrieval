"""fix.py -- One-time cleanup of stale cached artifacts.

Removes files produced by older retrieval logic that can conflict with the
new dynamic routing pipeline while keeping embeddings and reusable indexes.
"""

import argparse
import os
import sys

# Ensure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import load_config, get_config_path


# These files are safe to regenerate and may encode stale retrieval behavior.
FILES_TO_REMOVE = {
    "tokenized_queries.jsonl",
    "query_tokens.pkl",
    "bm25_results.pkl",
    "dense_results.pkl",
    "results.csv",
    "results.png",
    "timing.csv",
    "timing_retrieval.csv",
    "summary_ndcg.csv",
    "summary_ndcg.png",
    "best_dynamic_params.csv",
    "retrieval_timing.csv",
}


def remove_stale_files(root_dir, dry_run=False):
    """Delete stale files under root_dir recursively."""
    removed = 0
    scanned = 0

    if not os.path.isdir(root_dir):
        return scanned, removed

    for walk_root, _, files in os.walk(root_dir):
        for filename in files:
            scanned += 1
            if filename not in FILES_TO_REMOVE:
                continue
            abs_path = os.path.join(walk_root, filename)
            rel_path = os.path.relpath(abs_path, root_dir)
            if dry_run:
                print(f"[REMOVE] {os.path.join(root_dir, rel_path)}")
                removed += 1
                continue
            os.remove(abs_path)
            print(f"[REMOVE] {os.path.join(root_dir, rel_path)}")
            removed += 1

    return scanned, removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove stale cached files from previous retrieval runs."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be removed without deleting them.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    results_root = get_config_path(cfg, "results_folder", "data/results")
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")

    print(f"Mode          : {'DRY RUN' if args.dry_run else 'DELETE'}")
    print(f"Results root  : {results_root}")
    print(f"Processed root: {processed_root}")

    scanned_a, removed_a = remove_stale_files(results_root, dry_run=args.dry_run)
    scanned_b, removed_b = remove_stale_files(processed_root, dry_run=args.dry_run)

    print("\nSummary")
    print(f"  Files scanned: {scanned_a + scanned_b}")
    print(f"  Files removed: {removed_a + removed_b}")
    print("  Embedding artifacts were preserved.")


if __name__ == "__main__":
    main()
