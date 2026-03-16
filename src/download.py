"""
donwload.py

Downloads the datasets listed in config.yaml (uncommented entries only).
If a dataset already exists locally, it is skipped.
"""

import argparse
import os
import sys

# Make sure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import load_config, ensure_dir, download_beir_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download configured BEIR datasets if not present."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets configured. Nothing to download.")
        return

    datasets_folder = u.get_config_path(cfg, "datasets_folder", "data/datasets")
    results_folder = u.get_config_path(cfg, "results_folder", "data/results")
    processed_folder = u.get_config_path(cfg, "processed_folder", "data/processed_data")

    ensure_dir(datasets_folder)
    ensure_dir(results_folder)
    ensure_dir(processed_folder)

    print(f"Datasets folder : {datasets_folder}")
    print(f"Results folder  : {results_folder}")
    print(f"Processed folder: {processed_folder}")
    print(f"Configured datasets ({len(datasets)}): {', '.join(datasets)}")

    downloaded = 0
    skipped = 0
    failed = 0

    for ds_name in datasets:
        ds_path = os.path.join(datasets_folder, ds_name)
        if os.path.isdir(ds_path):
            print(f"[SKIP] {ds_name} already exists at {ds_path}")
            skipped += 1
            continue

        print(f"[DOWNLOAD] {ds_name}")
        out = download_beir_dataset(ds_name, datasets_folder)
        if out is None or not os.path.isdir(out):
            print(f"[FAIL] {ds_name}")
            failed += 1
        else:
            print(f"[OK] {ds_name} -> {out}")
            downloaded += 1

    print("\nSummary")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")


if __name__ == "__main__":
    main()
