"""ablation_study.py

Single-feature leave-one-feature-out (LOFO) ablation for XGBoost query routing.

Design:
- Within-dataset only
- Paired train/test splits across baseline and each ablation run
- Delta definition: NDCG_without_feature - NDCG_full_features
- Reports per-dataset and macro-average deltas
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.retrieve_and_evaluate import (
    apply_zscore,
    build_or_load_query_feature_cache,
    compute_zscore_stats,
    dataset_seed_offset,
    ensure_retrieval_results_cached,
    evaluate_benchmark_methods_for_qids,
    get_selected_feature_names,
    predict_router_alpha,
    set_global_seed,
    split_rows_train_test,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def rows_to_matrix(rows, feature_names):
    """Convert cached feature rows to matrix format for selected feature order."""
    qids = [r["query_id"] for r in rows]
    X = np.asarray(
        [[float(r["features"][name]) for name in feature_names] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(r["soft_label"]) for r in rows], dtype=np.float32)
    return X, y, qids


def run_model_once(train_rows, test_rows, feature_names, cfg, dataset_name, ds_cache, device, ndcg_k, rrf_k):
    """Train/evaluate one model on a fixed paired split and return dynamic NDCG."""
    X_train_raw, y_train, _ = rows_to_matrix(train_rows, feature_names)
    X_test_raw, _, test_qids = rows_to_matrix(test_rows, feature_names)

    train_mean, train_std = compute_zscore_stats(X_train_raw)
    X_train = apply_zscore(X_train_raw, train_mean, train_std)
    X_test = apply_zscore(X_test_raw, train_mean, train_std)

    model_bundle = train_router_model(
        X_train,
        y_train,
        cfg,
        device,
        dataset_name=dataset_name,
        optimization_mode="within_dataset",
    )
    alphas = predict_router_alpha(model_bundle, X_test, cfg, device)
    alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}

    metrics = evaluate_benchmark_methods_for_qids(
        bm25_results=ds_cache["bm25_results"],
        dense_results=ds_cache["dense_results"],
        qrels=ds_cache["qrels"],
        ndcg_k=ndcg_k,
        rrf_k=rrf_k,
        query_ids=test_qids,
        alpha_map=alpha_map,
    )
    return float(metrics["dynamic_wrrf_ndcg"])


def save_per_dataset_delta_csv(rows, output_csv):
    """Save per-dataset per-feature LOFO delta results."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "feature_removed",
                "baseline_dynamic_ndcg",
                "ablated_dynamic_ndcg",
                "delta_ndcg",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["feature_removed"],
                    f"{row['baseline_dynamic_ndcg']:.6f}",
                    f"{row['ablated_dynamic_ndcg']:.6f}",
                    f"{row['delta_ndcg']:.6f}",
                ]
            )


def save_macro_delta_csv(rows, output_csv):
    """Save macro-average LOFO delta across datasets."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature_removed",
                "macro_baseline_dynamic_ndcg",
                "macro_ablated_dynamic_ndcg",
                "macro_delta_ndcg",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["feature_removed"],
                    f"{row['macro_baseline_dynamic_ndcg']:.6f}",
                    f"{row['macro_ablated_dynamic_ndcg']:.6f}",
                    f"{row['macro_delta_ndcg']:.6f}",
                ]
            )


def save_delta_plot(macro_rows, output_png):
    """Save sorted macro delta bar plot."""
    sorted_rows = sorted(macro_rows, key=lambda r: r["macro_delta_ndcg"])
    labels = [r["feature_removed"] for r in sorted_rows]
    values = [r["macro_delta_ndcg"] for r in sorted_rows]

    colors = ["#D1495B" if v < 0 else "#2A9D8F" for v in values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("LOFO Macro Delta NDCG@10 (without feature - full)")
    ax.set_ylabel("Delta NDCG@10")
    ax.set_xlabel("Removed feature")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run within-dataset paired LOFO ablation for XGBoost routing."
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
        raise ValueError("No datasets configured.")

    routing_cfg = cfg.get("supervised_routing", {})
    cfg.setdefault("supervised_routing", {})
    cfg["supervised_routing"]["model_type"] = "xgboost"

    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    within_cfg = cfg.get("within_dataset_evaluation", {})
    train_fraction = float(within_cfg.get("train_fraction", 0.8))
    n_repeats = int(within_cfg.get("n_repeats", 5))
    shuffle = bool(within_cfg.get("shuffle", True))

    if n_repeats <= 0:
        raise ValueError("within_dataset_evaluation.n_repeats must be > 0.")

    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))
    rrf_k = int(cfg["benchmark"].get("rrf", {}).get("k", 60))
    short_model = model_short_name(cfg["embeddings"]["model_name"])

    final_features = get_selected_feature_names()

    print("=" * 72)
    print("XGBoost LOFO ablation (within-dataset, paired splits)")
    print(f"Device          : {device}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Features ({len(final_features)}): {final_features}")
    print(f"Delta definition: ndcg_without_feature - ndcg_full")
    print("=" * 72)

    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Building or loading per-query feature cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)
    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/4] Running paired LOFO ablation ...")
    per_dataset_feature_rows = []

    for dataset_name in datasets:
        ds_rows = list(rows_by_dataset[dataset_name])
        if len(ds_rows) < 2:
            raise ValueError(f"Dataset {dataset_name} has only {len(ds_rows)} rows; need at least 2.")

        ds_offset = dataset_seed_offset(dataset_name)
        print(f"\nDataset: {dataset_name} | rows={len(ds_rows)}")

        paired_splits = []
        baseline_scores = []
        for repeat_idx in range(n_repeats):
            repeat_seed = seed + repeat_idx + ds_offset
            train_rows, test_rows = split_rows_train_test(
                ds_rows,
                train_fraction=train_fraction,
                repeat_seed=repeat_seed,
                shuffle=shuffle,
            )
            paired_splits.append((train_rows, test_rows))
            baseline_ndcg = run_model_once(
                train_rows=train_rows,
                test_rows=test_rows,
                    feature_names=final_features,
                cfg=cfg,
                dataset_name=dataset_name,
                ds_cache=dataset_cache_map[dataset_name],
                device=device,
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )
            baseline_scores.append(baseline_ndcg)

        for feature_removed in final_features:
            ablated_scores = []

            ablated_features = [f for f in final_features if f != feature_removed]
            if not ablated_features:
                raise ValueError("Ablated feature set is empty; cannot train model.")

            for train_rows, test_rows in paired_splits:
                ablated_ndcg = run_model_once(
                    train_rows=train_rows,
                    test_rows=test_rows,
                    feature_names=ablated_features,
                    cfg=cfg,
                    dataset_name=dataset_name,
                    ds_cache=dataset_cache_map[dataset_name],
                    device=device,
                    ndcg_k=ndcg_k,
                    rrf_k=rrf_k,
                )

                ablated_scores.append(ablated_ndcg)

            baseline_mean = float(np.mean(baseline_scores))
            ablated_mean = float(np.mean(ablated_scores))
            delta = float(ablated_mean - baseline_mean)

            per_dataset_feature_rows.append(
                {
                    "dataset": dataset_name,
                    "feature_removed": feature_removed,
                    "baseline_dynamic_ndcg": baseline_mean,
                    "ablated_dynamic_ndcg": ablated_mean,
                    "delta_ndcg": delta,
                }
            )
            print(
                f"  remove={feature_removed:>28} | "
                f"baseline={baseline_mean:.4f} | ablated={ablated_mean:.4f} | delta={delta:.4f}"
            )

    print("\n[4/4] Writing ablation outputs ...")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_dir = os.path.join(results_root, short_model, "ablation")
    ensure_dir(out_dir)

    per_dataset_csv = os.path.join(out_dir, "lofo_per_dataset_delta.csv")
    macro_csv = os.path.join(out_dir, "lofo_macro_delta.csv")
    delta_plot = os.path.join(out_dir, "lofo_macro_delta_plot.png")

    save_per_dataset_delta_csv(per_dataset_feature_rows, per_dataset_csv)

    macro_rows = []
    for feat in final_features:
        rows = [r for r in per_dataset_feature_rows if r["feature_removed"] == feat]
        macro_rows.append(
            {
                "feature_removed": feat,
                "macro_baseline_dynamic_ndcg": float(np.mean([r["baseline_dynamic_ndcg"] for r in rows])),
                "macro_ablated_dynamic_ndcg": float(np.mean([r["ablated_dynamic_ndcg"] for r in rows])),
                "macro_delta_ndcg": float(np.mean([r["delta_ndcg"] for r in rows])),
            }
        )

    save_macro_delta_csv(macro_rows, macro_csv)
    save_delta_plot(macro_rows, delta_plot)

    print("\n" + "=" * 72)
    print("LOFO ablation completed.")
    print(f"Per-dataset delta CSV : {per_dataset_csv}")
    print(f"Macro delta CSV       : {macro_csv}")
    print(f"Delta plot PNG        : {delta_plot}")
    print("=" * 72)


if __name__ == "__main__":
    main()
