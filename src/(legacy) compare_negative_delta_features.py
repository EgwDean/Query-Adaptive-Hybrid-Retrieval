"""compare_negative_delta_features.py

Run a model ladder between compact and full routing features.

Default setup targets 14 total models:
- 1 dense-only retrieval baseline (no router training)
- 1 full-feature router
- 12 compact-based ladder models, where each next model adds one omitted
    feature (ordered by least harmful macro LOFO delta first)

Method:
- Within-dataset only
- Paired train/test splits
- Delta definitions:
    - delta_vs_full = model_ndcg - full_ndcg
    - delta_vs_prev = model_ndcg - previous_model_ndcg (ladder order)
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
    rows_to_matrix_with_features,
    set_global_seed,
    split_rows_train_test,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def load_negative_feature_rows(macro_delta_csv, threshold):
    """Load LOFO macro delta rows and keep features with delta < threshold."""
    rows = []
    with open(macro_delta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = str(row["feature_removed"]).strip()
            delta = float(row["macro_delta_ndcg"])
            if delta < threshold:
                rows.append({"feature_removed": feature, "macro_delta_ndcg": delta})
    return rows


def load_macro_delta_map(macro_delta_csv):
    """Load full feature->macro_delta mapping from LOFO macro CSV."""
    delta_map = {}
    with open(macro_delta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = str(row["feature_removed"]).strip()
            delta_map[feature] = float(row["macro_delta_ndcg"])
    return delta_map


def run_model_once(train_rows, test_rows, feature_names, cfg, dataset_name, ds_cache, device, ndcg_k, rrf_k):
    """Train and evaluate one model on one paired split."""
    X_train_raw, y_train, _ = rows_to_matrix_with_features(train_rows, feature_names)
    X_test_raw, _, test_qids = rows_to_matrix_with_features(test_rows, feature_names)

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


def evaluate_dense_baseline_for_qids(test_qids, ds_cache, ndcg_k, rrf_k):
    """Evaluate dense-only retrieval on a fixed query subset."""
    metrics = evaluate_benchmark_methods_for_qids(
        bm25_results=ds_cache["bm25_results"],
        dense_results=ds_cache["dense_results"],
        qrels=ds_cache["qrels"],
        ndcg_k=ndcg_k,
        rrf_k=rrf_k,
        query_ids=test_qids,
        alpha_map=None,
    )
    return float(metrics["dense_only_ndcg"])


def save_negative_feature_list(rows, output_csv):
    """Persist the chosen compact feature set with macro deltas."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_kept", "macro_delta_ndcg"])
        for row in sorted(rows, key=lambda r: r["macro_delta_ndcg"]):
            writer.writerow([row["feature_removed"], f"{row['macro_delta_ndcg']:.6f}"])


def save_ladder_plan_csv(rows, output_csv):
    """Persist model ladder plan and feature composition."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_index",
                "model_name",
                "model_type",
                "feature_count",
                "added_feature",
                "added_feature_macro_delta",
                "feature_list",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model_index"],
                    row["model_name"],
                    row["model_type"],
                    row["feature_count"],
                    row.get("added_feature", ""),
                    "" if row.get("added_feature_macro_delta") is None else f"{row['added_feature_macro_delta']:.6f}",
                    "|".join(row.get("feature_names", [])),
                ]
            )


def save_per_dataset_comparison(rows, output_csv):
    """Save per-dataset model comparison against full and previous model."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "model_index",
                "model_name",
                "model_type",
                "feature_count",
                "dynamic_ndcg",
                "delta_vs_full",
                "delta_vs_previous",
                "added_feature",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["model_index"],
                    row["model_name"],
                    row["model_type"],
                    row["feature_count"],
                    f"{row['dynamic_ndcg']:.6f}",
                    f"{row['delta_vs_full']:.6f}",
                    "" if row["delta_vs_previous"] is None else f"{row['delta_vs_previous']:.6f}",
                    row.get("added_feature", ""),
                ]
            )


def save_macro_comparison(rows, output_csv):
    """Save macro-average comparison across all evaluated models."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_index",
                "model_name",
                "model_type",
                "feature_count",
                "macro_dynamic_ndcg",
                "macro_delta_vs_full",
                "macro_delta_vs_previous",
                "added_feature",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model_index"],
                    row["model_name"],
                    row["model_type"],
                    row["feature_count"],
                    f"{row['macro_dynamic_ndcg']:.6f}",
                    f"{row['macro_delta_vs_full']:.6f}",
                    "" if row["macro_delta_vs_previous"] is None else f"{row['macro_delta_vs_previous']:.6f}",
                    row.get("added_feature", ""),
                ]
            )


def save_macro_delta_plot(macro_rows, output_png):
    """Save macro delta-vs-full bar plot for all models."""
    labels = [f"{r['model_index']}: {r['model_name']}" for r in macro_rows]
    values = [r["macro_delta_vs_full"] for r in macro_rows]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Macro Delta vs Full (NDCG@10)")
    ax.set_ylabel("model_ndcg - full_ndcg")
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def save_dataset_delta_plot(per_dataset_rows, output_png):
    """Save per-dataset delta-vs-full line plot across model index."""
    datasets = sorted({r["dataset"] for r in per_dataset_rows})
    fig, ax = plt.subplots(figsize=(12, 6))

    for ds in datasets:
        ds_rows = sorted(
            [r for r in per_dataset_rows if r["dataset"] == ds],
            key=lambda r: r["model_index"],
        )
        x = [r["model_index"] for r in ds_rows]
        y = [r["delta_vs_full"] for r in ds_rows]
        ax.plot(x, y, marker="o", linewidth=1.5, label=ds)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Per-Dataset Delta vs Full Across Ladder")
    ax.set_xlabel("Model index")
    ax.set_ylabel("model_ndcg - full_ndcg")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare dense baseline, full router, and compact-based feature-addition ladder models."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--macro-delta-csv",
        default=None,
        help="Optional path to lofo_macro_delta.csv. Defaults to model results folder.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.0,
        help="Keep features with macro_delta_ndcg < threshold (default: 0.0).",
    )
    parser.add_argument(
        "--n-ladder-models",
        type=int,
        default=12,
        help="Number of compact-based ladder models (default: 12).",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    cfg.setdefault("supervised_routing", {})
    cfg["supervised_routing"]["model_type"] = "xgboost"
    routing_cfg = cfg.get("supervised_routing", {})

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

    results_root = get_config_path(cfg, "results_folder", "data/results")
    ablation_dir = os.path.join(results_root, short_model, "ablation")
    macro_delta_csv = args.macro_delta_csv or os.path.join(ablation_dir, "lofo_macro_delta.csv")
    if not os.path.exists(macro_delta_csv):
        raise FileNotFoundError(
            f"Macro delta CSV not found: {macro_delta_csv}. Run LOFO ablation first."
        )

    full_features = get_selected_feature_names()
    macro_delta_map = load_macro_delta_map(macro_delta_csv)
    negative_rows = load_negative_feature_rows(macro_delta_csv, args.delta_threshold)
    negative_feature_set = {r["feature_removed"] for r in negative_rows}
    compact_features = [f for f in full_features if f in negative_feature_set]

    if not compact_features:
        raise ValueError(
            "No compact features selected. Try increasing --delta-threshold or verify LOFO CSV values."
        )

    omitted_features = [f for f in full_features if f not in negative_feature_set]
    if not omitted_features:
        raise ValueError("No omitted features available to build a ladder.")

    omitted_sorted = sorted(
        omitted_features,
        key=lambda f: (macro_delta_map.get(f, float("inf")), f),
    )

    n_ladder = int(args.n_ladder_models)
    if n_ladder <= 0:
        raise ValueError("--n-ladder-models must be > 0.")

    max_additions = min(len(omitted_sorted), max(0, n_ladder - 1))
    if max_additions < (n_ladder - 1):
        print(
            f"[WARN] Requested {n_ladder} ladder models but only {len(omitted_sorted)} omitted features are "
            f"available. Using {max_additions + 1} ladder models instead."
        )
    n_ladder_effective = max_additions + 1

    ladder_models = []
    for idx in range(n_ladder_effective):
        added = omitted_sorted[:idx]
        added_feature = omitted_sorted[idx - 1] if idx > 0 else ""
        added_delta = macro_delta_map.get(added_feature) if added_feature else None
        feature_names = [f for f in full_features if (f in compact_features or f in added)]
        ladder_models.append(
            {
                "model_name": f"compact_plus_{idx}",
                "model_type": "router",
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "added_feature": added_feature,
                "added_feature_macro_delta": added_delta,
            }
        )

    print("=" * 72)
    print("Dense + full + compact ladder comparison")
    print(f"Device                 : {device}")
    print(f"Datasets ({len(datasets)})         : {', '.join(datasets)}")
    print(f"Full feature count     : {len(full_features)}")
    print(f"Compact feature count  : {len(compact_features)}")
    print(f"Omitted feature count  : {len(omitted_features)}")
    print(f"Ladder models          : {n_ladder_effective}")
    print(f"Total models (target)  : dense + full + ladder = {2 + n_ladder_effective}")
    print(f"Delta threshold        : {args.delta_threshold} (keep delta < threshold)")
    print(f"Macro delta CSV        : {macro_delta_csv}")
    print("Delta definition       : model_ndcg - full_ndcg")
    print("=" * 72)

    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Loading or building query feature cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)
    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/4] Running paired within-dataset comparison across all models ...")

    model_specs = []
    model_specs.append(
        {
            "model_name": "dense_only",
            "model_type": "dense_baseline",
            "feature_names": [],
            "feature_count": 0,
            "added_feature": "",
            "added_feature_macro_delta": None,
        }
    )
    model_specs.append(
        {
            "model_name": "full_router",
            "model_type": "router",
            "feature_names": list(full_features),
            "feature_count": len(full_features),
            "added_feature": "",
            "added_feature_macro_delta": None,
        }
    )
    model_specs.extend(ladder_models)

    for idx, spec in enumerate(model_specs):
        spec["model_index"] = idx

    per_dataset_rows = []
    per_dataset_model_scores = {}

    for dataset_name in datasets:
        ds_rows = list(rows_by_dataset[dataset_name])
        if len(ds_rows) < 2:
            raise ValueError(f"Dataset {dataset_name} has only {len(ds_rows)} rows; need at least 2.")

        ds_offset = dataset_seed_offset(dataset_name)
        per_model_scores = {spec["model_name"]: [] for spec in model_specs}

        for repeat_idx in range(n_repeats):
            repeat_seed = seed + repeat_idx + ds_offset
            train_rows, test_rows = split_rows_train_test(
                ds_rows,
                train_fraction=train_fraction,
                repeat_seed=repeat_seed,
                shuffle=shuffle,
            )

            dense_ndcg = evaluate_dense_baseline_for_qids(
                test_qids=[r["query_id"] for r in test_rows],
                ds_cache=dataset_cache_map[dataset_name],
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )
            per_model_scores["dense_only"].append(dense_ndcg)

            for spec in model_specs:
                if spec["model_type"] != "router":
                    continue
                ndcg = run_model_once(
                    train_rows=train_rows,
                    test_rows=test_rows,
                    feature_names=spec["feature_names"],
                    cfg=cfg,
                    dataset_name=dataset_name,
                    ds_cache=dataset_cache_map[dataset_name],
                    device=device,
                    ndcg_k=ndcg_k,
                    rrf_k=rrf_k,
                )
                per_model_scores[spec["model_name"]].append(ndcg)

        per_dataset_model_scores[dataset_name] = {
            k: float(np.mean(v)) for k, v in per_model_scores.items()
        }

        full_mean = per_dataset_model_scores[dataset_name]["full_router"]
        print(f"\n  Dataset={dataset_name} | full={full_mean:.4f}")
        previous_mean = None
        for spec in model_specs:
            model_mean = per_dataset_model_scores[dataset_name][spec["model_name"]]
            delta_vs_full = float(model_mean - full_mean)
            delta_vs_prev = None if previous_mean is None else float(model_mean - previous_mean)
            previous_mean = model_mean

            per_dataset_rows.append(
                {
                    "dataset": dataset_name,
                    "model_index": spec["model_index"],
                    "model_name": spec["model_name"],
                    "model_type": spec["model_type"],
                    "feature_count": spec["feature_count"],
                    "dynamic_ndcg": model_mean,
                    "delta_vs_full": delta_vs_full,
                    "delta_vs_previous": delta_vs_prev,
                    "added_feature": spec.get("added_feature", ""),
                }
            )
            print(
                f"    idx={spec['model_index']:>2} | {spec['model_name']:<16} | "
                f"ndcg={model_mean:.4f} | d_full={delta_vs_full:+.4f}"
            )

    macro_rows = []
    previous_macro = None
    macro_full = float(np.mean([per_dataset_model_scores[ds]["full_router"] for ds in datasets]))
    for spec in model_specs:
        model_macro = float(np.mean([per_dataset_model_scores[ds][spec["model_name"]] for ds in datasets]))
        macro_delta_vs_full = float(model_macro - macro_full)
        macro_delta_vs_previous = None if previous_macro is None else float(model_macro - previous_macro)
        previous_macro = model_macro
        macro_rows.append(
            {
                "model_index": spec["model_index"],
                "model_name": spec["model_name"],
                "model_type": spec["model_type"],
                "feature_count": spec["feature_count"],
                "macro_dynamic_ndcg": model_macro,
                "macro_delta_vs_full": macro_delta_vs_full,
                "macro_delta_vs_previous": macro_delta_vs_previous,
                "added_feature": spec.get("added_feature", ""),
            }
        )

    print("\n[4/4] Writing comparison outputs ...")
    out_dir = os.path.join(ablation_dir, "negative_feature_subset_comparison")
    ensure_dir(out_dir)

    kept_features_csv = os.path.join(out_dir, "kept_negative_delta_features.csv")
    ladder_plan_csv = os.path.join(out_dir, "model_ladder_plan.csv")
    per_dataset_csv = os.path.join(out_dir, "model_ladder_per_dataset.csv")
    macro_csv = os.path.join(out_dir, "model_ladder_macro.csv")
    macro_plot_png = os.path.join(out_dir, "model_ladder_macro_delta_vs_full.png")
    dataset_plot_png = os.path.join(out_dir, "model_ladder_dataset_delta_vs_full.png")

    save_negative_feature_list(negative_rows, kept_features_csv)
    save_ladder_plan_csv(model_specs, ladder_plan_csv)
    save_per_dataset_comparison(per_dataset_rows, per_dataset_csv)
    save_macro_comparison(macro_rows, macro_csv)
    save_macro_delta_plot(macro_rows, macro_plot_png)
    save_dataset_delta_plot(per_dataset_rows, dataset_plot_png)

    print("\nComparison completed.")
    print(f"Kept features CSV      : {kept_features_csv}")
    print(f"Ladder plan CSV        : {ladder_plan_csv}")
    print(f"Per-dataset CSV        : {per_dataset_csv}")
    print(f"Macro summary CSV      : {macro_csv}")
    print(f"Macro delta plot       : {macro_plot_png}")
    print(f"Per-dataset delta plot : {dataset_plot_png}")
    print(f"Macro full_router      : {macro_full:.6f}")
    for row in macro_rows:
        print(
            f"  idx={row['model_index']:>2} | {row['model_name']:<16} | "
            f"macro_ndcg={row['macro_dynamic_ndcg']:.6f} | "
            f"d_full={row['macro_delta_vs_full']:+.6f}"
        )


if __name__ == "__main__":
    main()
