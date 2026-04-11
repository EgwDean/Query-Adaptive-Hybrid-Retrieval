"""
cleanup_hpc.py

Removes stale data artifacts from previous pipeline runs that do not match
the current configuration in config.yaml.  Targets three kinds of staleness:

  1. Processed-data directories built with a different embedding model
     (e.g., data/processed_data/all-MiniLM-L6-v2/ when the current model is
     bge-m3).

  2. BM25 artifact files whose k1 / b / stemming signature differs from the
     current config (e.g., leftover bm25_k1_1.2_b_0.5_stem_1.pkl when config
     now uses k1=1.5, b=0.75).

  3. Result files in data/results/ whose names are not produced by any current
     script (legacy CSVs, PNGs from old experiments, etc.).

The script never touches:
  - data/datasets/         (raw BEIR downloads -- expensive to re-download)
  - corpus.jsonl / queries.jsonl / qrels.tsv
  - corpus_embeddings.pt / corpus_ids.pkl / query_vectors.pt / query_ids.pkl
    (dense embeddings -- expensive to recompute on HPC)
  - Any file whose name matches the current pipeline's expected outputs

Usage
-----
  # Preview what would be deleted (dry-run, default):
  python cleanup_hpc.py

  # Actually delete:
  python cleanup_hpc.py --confirm
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR          # cleanup_hpc.py lives at project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    bm25_signature,
    get_bm25_params,
    get_config_path,
    load_config,
    model_short_name,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ── Stale-file collection ─────────────────────────────────────────────────────

def collect_stale(cfg):
    """Return a sorted list of (abs_path, size_bytes) for stale files."""
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    results_root   = get_config_path(cfg, "results_folder",   "data/results")
    datasets       = cfg["datasets"]
    top_k          = int(cfg["benchmark"]["top_k"])

    bm25_params  = get_bm25_params(cfg)
    current_sig  = bm25_signature(
        bm25_params["k1"], bm25_params["b"], bm25_params["use_stemming"]
    )
    stem_flag    = f"stem_{1 if bm25_params['use_stemming'] else 0}"
    short_model  = model_short_name(cfg["embeddings"]["model_name"])

    stale = []

    # ── processed_data ────────────────────────────────────────────────────────
    if os.path.isdir(processed_root):
        for model_dir in os.listdir(processed_root):
            model_path = os.path.join(processed_root, model_dir)
            if not os.path.isdir(model_path):
                continue

            # Entire model directory belongs to an old embedding model → stale.
            if model_dir != short_model:
                for root, _dirs, files in os.walk(model_path):
                    for fname in files:
                        fp = os.path.join(root, fname)
                        stale.append((fp, _file_size(fp)))
                continue

            # Within the current model directory, check each dataset folder.
            for ds_dir_name in os.listdir(model_path):
                ds_path = os.path.join(model_path, ds_dir_name)
                if not os.path.isdir(ds_path):
                    continue

                for fname in os.listdir(ds_path):
                    fp = os.path.join(ds_path, fname)
                    if not os.path.isfile(fp):
                        continue

                    # ── Whitelist ─────────────────────────────────────────────
                    keep = False

                    # Base exports (cheap to regenerate but always small)
                    if fname in ("corpus.jsonl", "queries.jsonl", "qrels.tsv"):
                        keep = True

                    # Dense embeddings and retrieval results (expensive on GPU)
                    elif fname in (
                        "corpus_embeddings.pt",
                        "corpus_ids.pkl",
                        "query_vectors.pt",
                        "query_ids.pkl",
                        f"dense_results_topk_{top_k}.pkl",
                    ):
                        keep = True

                    # Current BM25 artifacts (keyed by signature + stem flag)
                    elif fname in (
                        f"tokenized_corpus_{stem_flag}.jsonl",
                        f"tokenized_queries_{stem_flag}.jsonl",
                        f"query_tokens_{stem_flag}.pkl",
                        f"word_freq_index_{stem_flag}.pkl",
                        f"doc_freq_index_{stem_flag}.pkl",
                        f"{current_sig}.pkl",
                        f"{current_sig}_doc_ids.pkl",
                        f"{current_sig}_topk_{top_k}_results.pkl",
                    ):
                        keep = True

                    # Feature / label cache — any hash suffix is valid; the hash
                    # encodes all config params so a stale hash simply goes unused.
                    elif fname.startswith("features_labels_") and fname.endswith(".pkl"):
                        keep = True

                    if not keep:
                        stale.append((fp, _file_size(fp)))

    # ── results ───────────────────────────────────────────────────────────────
    shap_files = {f"shap_{ds}.png" for ds in datasets}
    expected_results = {
        "bm25_optimization_macro.csv",
        "bm25_optimization_best.json",
        "model_grid_search_top100.csv",
        "ablation_study.csv",
        "ablation_study.png",
        "full_vs_smaller_ablation.csv",
        "full_vs_smaller_ablation.png",
        "per_dataset_best_params.csv",
        "retrieval_comparison.csv",
        "retrieval_comparison.png",
    } | shap_files

    if os.path.isdir(results_root):
        for fname in os.listdir(results_root):
            fp = os.path.join(results_root, fname)
            if os.path.isfile(fp) and fname not in expected_results:
                stale.append((fp, _file_size(fp)))

    return sorted(stale, key=lambda x: x[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Remove stale data artifacts from previous pipeline runs.\n"
            "Run without --confirm to preview (dry-run mode)."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files. Default is dry-run (print only).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml).",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    print("=" * 60)
    print("Pipeline cleanup utility")
    print(f"Config        : {args.config}")
    print(f"Embedding model : {cfg['embeddings']['model_name']}")
    bm25_p = get_bm25_params(cfg)
    print(f"BM25 signature  : {bm25_signature(bm25_p['k1'], bm25_p['b'], bm25_p['use_stemming'])}")
    print(f"Mode          : {'CONFIRM — files will be DELETED' if args.confirm else 'DRY-RUN — no files deleted'}")
    print("=" * 60)

    stale = collect_stale(cfg)

    if not stale:
        print("\nNothing to clean up — all artifacts match the current configuration.")
        return

    total_bytes = sum(sz for _, sz in stale)
    rel = lambda p: os.path.relpath(p, PROJECT_ROOT)  # noqa: E731

    print(f"\nFound {len(stale)} stale file(s) totalling {_fmt_bytes(total_bytes)}:\n")
    for fp, sz in stale:
        action = "DELETE" if args.confirm else "WOULD DELETE"
        print(f"  [{action}]  {rel(fp)}  ({_fmt_bytes(sz)})")

    if not args.confirm:
        print(
            f"\nDry-run complete.  Re-run with --confirm to free "
            f"{_fmt_bytes(total_bytes)} across {len(stale)} file(s)."
        )
        return

    deleted = freed = failed = 0
    for fp, sz in stale:
        try:
            os.remove(fp)
            deleted += 1
            freed   += sz
        except OSError as exc:
            print(f"  [ERROR] Could not delete {rel(fp)}: {exc}")
            failed += 1

    # Remove any now-empty subdirectories (non-recursive, best-effort)
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    if os.path.isdir(processed_root):
        for root, dirs, files in os.walk(processed_root, topdown=False):
            if root == processed_root:
                continue
            try:
                os.rmdir(root)  # only succeeds if directory is empty
            except OSError:
                pass

    print(
        f"\nDeleted {deleted} file(s), freed {_fmt_bytes(freed)}."
        + (f"  Errors: {failed}." if failed else "")
    )


if __name__ == "__main__":
    main()
