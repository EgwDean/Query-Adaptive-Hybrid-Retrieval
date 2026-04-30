"""
cleanup.py
==========

One-shot disk cleanup script.  It is meant to be executed *once* on the HPC
node before running `pipeline.py` for the first time, and then deleted.

What it deletes
---------------
1. Everything under  data/results/  — model parameters, retrieval scores,
   plots, etc.  All of these are recomputed by the new pipeline and the
   legacy file naming differs from the new one.
2. Everything under  data/models/   — final pickled models from older runs.
3. Stale per-dataset cache files inside  data/processed/<model>/<ds>/  that
   the new pipeline would either rebuild differently or no longer needs:
       *  features_labels_*.pkl              (legacy weak-feature cache)
       *  query_tokens_stem_*.pkl            (rebuilt deterministically)
       *  bm25_results_*.pkl                 (cheap, dependent on best params)
       *  dense_results_topk_*.pkl           (cheap, parameter-dependent)
       *  *_grid_search*.csv                 (any leftover CSV from legacy)

What it KEEPS
-------------
*  data/datasets/                            (BEIR raw archives + folders)
*  data/processed/<model>/<ds>/corpus.jsonl
*  data/processed/<model>/<ds>/queries.jsonl
*  data/processed/<model>/<ds>/qrels.tsv
*  data/processed/<model>/<ds>/tokenized_corpus_stem_*.jsonl
*  data/processed/<model>/<ds>/tokenized_queries_stem_*.jsonl
*  data/processed/<model>/<ds>/word_freq_index_stem_*.pkl
*  data/processed/<model>/<ds>/doc_freq_index_stem_*.pkl
*  data/processed/<model>/<ds>/bm25_*.pkl + *_doc_ids.pkl
*  data/processed/<model>/<ds>/corpus_embeddings.pt + corpus_ids.pkl
*  data/processed/<model>/<ds>/query_vectors.pt + query_ids.pkl

These are expensive to recompute and the new pipeline would produce
identical results, so we skip them on purpose (per the TODO requirements).

Run:
    python src/cleanup.py [--dry-run]
"""

import argparse
import os
import shutil
import sys
import fnmatch

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils import get_config_path, load_config, model_short_name


# ── Patterns of files to delete inside data/processed/<model>/<ds>/ ──────────

STALE_FILE_GLOBS = [
    "features_labels_*.pkl",            # legacy weak-feature cache
    "query_tokens_stem_*.pkl",          # rebuilt by Step 2
    "bm25_results_*.pkl",               # depends on chosen BM25 + top_k
    "*_results.pkl",                    # generic name used in legacy too
    "dense_results_topk_*.pkl",
    "rerank_scores_*.pkl",              # legacy CE cache
    "rerank_*.csv",
    "*.params_hash",
    "*_grid_search*.csv",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:7.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:7.2f} PB"


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _wipe_directory_contents(dir_path: str, dry_run: bool) -> int:
    """Delete every file and sub-folder inside `dir_path` (keep the folder)."""
    bytes_freed = 0
    if not os.path.isdir(dir_path):
        return 0
    for entry in sorted(os.listdir(dir_path)):
        full = os.path.join(dir_path, entry)
        try:
            if os.path.isdir(full):
                size = _dir_size(full)
                bytes_freed += size
                action = "[DRY-RUN]" if dry_run else "[REMOVE]"
                print(f"  {action} {full}/  ({_human_size(size)})")
                if not dry_run:
                    shutil.rmtree(full, ignore_errors=False)
            else:
                size = _file_size(full)
                bytes_freed += size
                action = "[DRY-RUN]" if dry_run else "[REMOVE]"
                print(f"  {action} {full}     ({_human_size(size)})")
                if not dry_run:
                    os.remove(full)
        except OSError as exc:
            print(f"  [WARN] could not remove {full}: {exc}")
    return bytes_freed


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            total += _file_size(os.path.join(dirpath, fn))
    return total


def _matches_any(name: str, globs: list) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _clean_processed(processed_root: str, model_short: str,
                     datasets: list, dry_run: bool) -> int:
    """Remove only stale per-dataset artifacts, keeping the heavy ones."""
    bytes_freed = 0
    base = os.path.join(processed_root, model_short)
    if not os.path.isdir(base):
        print(f"  (no processed folder at {base})")
        return 0

    for ds_name in datasets:
        ds_dir = os.path.join(base, ds_name)
        if not os.path.isdir(ds_dir):
            continue
        for entry in sorted(os.listdir(ds_dir)):
            full = os.path.join(ds_dir, entry)
            if not os.path.isfile(full):
                continue
            if _matches_any(entry, STALE_FILE_GLOBS):
                size = _file_size(full)
                bytes_freed += size
                action = "[DRY-RUN]" if dry_run else "[REMOVE]"
                print(f"  {action} {full}  ({_human_size(size)})")
                if not dry_run:
                    try:
                        os.remove(full)
                    except OSError as exc:
                        print(f"  [WARN] could not remove {full}: {exc}")
    return bytes_freed


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean stale pipeline artefacts from data/."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be deleted without removing anything.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config (default: config.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    datasets         = cfg.get("datasets", []) or []
    results_folder   = get_config_path(cfg, "results_folder", "data/results")
    models_folder    = get_config_path(cfg, "models_folder",  "data/models")
    processed_folder = get_config_path(cfg, "processed_folder", "data/processed")
    short_model      = model_short_name(cfg["embeddings"]["model_name"])

    print("=" * 64)
    print("Pipeline data cleanup")
    print("=" * 64)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Datasets          : {', '.join(datasets) or '(none)'}")
    print(f"Embedding model   : {short_model}")
    print(f"Results folder    : {results_folder}")
    print(f"Models folder     : {models_folder}")
    print(f"Processed folder  : {processed_folder}")
    print(f"Mode              : {'DRY-RUN' if args.dry_run else 'DELETE'}")
    print("-" * 64)

    total = 0

    print("\n[1/3] Wiping  data/results/  (all old metrics, params, plots) ...")
    total += _wipe_directory_contents(results_folder, args.dry_run)

    print("\n[2/3] Wiping  data/models/   (all old pickled models) ...")
    total += _wipe_directory_contents(models_folder,  args.dry_run)

    print("\n[3/3] Cleaning stale per-dataset artefacts under  data/processed/  ...")
    total += _clean_processed(processed_folder, short_model, datasets, args.dry_run)

    print("\n" + "=" * 64)
    verb = "Would free" if args.dry_run else "Freed"
    print(f"{verb} {_human_size(total)} of disk space.")
    print("=" * 64)
    if not args.dry_run:
        print("\nYou can now delete this script (per TODO instructions) and run:")
        print("    python src/pipeline.py")


if __name__ == "__main__":
    main()
