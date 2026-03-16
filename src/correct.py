"""
correct.py

Moves preprocessing/cache artifacts from data/results/... to data/processed_data/...
for already-run experiments.

This is a migration helper for the new split:
  - data/results -> final benchmark outputs
  - data/processed_data -> cached intermediate artifacts
"""

import argparse
import os
import shutil
import sys

# Make sure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import load_config, ensure_dir, get_processing_artifact_filenames


def move_if_needed(src_path, dst_path, dry_run=False):
    """Move src_path -> dst_path if destination is absent.

    Returns one of: moved, skipped_exists, skipped_missing.
    """
    if not os.path.isfile(src_path):
        return "skipped_missing"

    ensure_dir(os.path.dirname(dst_path))

    if os.path.isfile(dst_path):
        return "skipped_exists"

    if dry_run:
        return "moved"

    shutil.move(src_path, dst_path)
    return "moved"


def main():
    parser = argparse.ArgumentParser(
        description="Move cached preprocessing artifacts from results to processed_data."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    results_folder = u.get_config_path(cfg, "results_folder", "data/results")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")
    ensure_dir(results_folder)
    ensure_dir(processed_folder)

    artifact_names = set(get_processing_artifact_filenames())

    moved = 0
    skipped_exists = 0
    scanned_files = 0

    print(f"Results folder  : {results_folder}")
    print(f"Processed folder: {processed_folder}")
    print(f"Mode            : {'DRY RUN' if args.dry_run else 'MOVE'}")

    for root, _, files in os.walk(results_folder):
        for filename in files:
            scanned_files += 1
            if filename not in artifact_names:
                continue

            src_path = os.path.join(root, filename)
            rel_path = os.path.relpath(src_path, results_folder)
            dst_path = os.path.join(processed_folder, rel_path)

            status = move_if_needed(src_path, dst_path, dry_run=args.dry_run)
            if status == "moved":
                moved += 1
                print(f"[MOVE] {rel_path}")
            elif status == "skipped_exists":
                skipped_exists += 1
                print(f"[SKIP] Exists in processed_data: {rel_path}")

    print("\nSummary")
    print(f"  Scanned files         : {scanned_files}")
    print(f"  Moved artifacts       : {moved}")
    print(f"  Skipped (already there): {skipped_exists}")


if __name__ == "__main__":
    main()
