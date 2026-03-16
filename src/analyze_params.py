"""Analyze dynamic wRRF parameter winners and derive zero-shot consensus.

Reads:
    data/results/<model_name>/best_dynamic_params.csv

Writes:
    data/results/<model_name>/parameter_distributions.png

Prints:
    YAML-formatted zero-shot parameter summary for JSD/KLD/CE.
"""

import argparse
import csv
import os
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "Dataset",
    "Metric",
    "Best max_df",
    "Best k",
    "Best center",
    "Best rrf_k",
    "Best NDCG@10",
}

METRICS = ("JSD", "KLD", "CE")
PARAMS = (
    ("Best max_df", "max_df"),
    ("Best rrf_k", "rrf_k"),
    ("Best k", "k"),
    ("Best center", "center"),
)


def to_float(value, col_name):
    """Convert a CSV field to float with context-rich errors."""
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Could not parse '{value}' as float in column '{col_name}'.") from exc


def to_int(value, col_name):
    """Convert a CSV field to int with context-rich errors."""
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Could not parse '{value}' as int in column '{col_name}'.") from exc


def load_rows(csv_path):
    """Load and validate rows from best_dynamic_params.csv."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - header
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in CSV: {missing_str}")

        for raw in reader:
            metric = (raw["Metric"] or "").strip().upper()
            if metric not in METRICS:
                continue

            rows.append(
                {
                    "Dataset": (raw["Dataset"] or "").strip(),
                    "Metric": metric,
                    "Best max_df": to_float(raw["Best max_df"], "Best max_df"),
                    "Best rrf_k": to_int(raw["Best rrf_k"], "Best rrf_k"),
                    "Best k": to_float(raw["Best k"], "Best k"),
                    "Best center": to_float(raw["Best center"], "Best center"),
                    "Best NDCG@10": to_float(raw["Best NDCG@10"], "Best NDCG@10"),
                }
            )

    if not rows:
        raise ValueError("No valid rows found for metrics JSD/KLD/CE.")

    return rows


def mode_with_tiebreak(values, prefer):
    """Return mode with explicit tie-break behavior.

    prefer='higher': select max among tied values
    prefer='lower' : select min among tied values
    """
    counts = Counter(values)
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    if prefer == "higher":
        return max(tied)
    if prefer == "lower":
        return min(tied)
    raise ValueError(f"Unknown tie-break preference: {prefer}")


def compute_consensus(rows):
    """Compute per-metric zero-shot consensus parameters."""
    by_metric = defaultdict(list)
    for row in rows:
        by_metric[row["Metric"]].append(row)

    consensus = {}
    for metric in METRICS:
        metric_rows = by_metric.get(metric, [])
        if not metric_rows:
            continue

        max_df_vals = [r["Best max_df"] for r in metric_rows]
        rrf_k_vals = [r["Best rrf_k"] for r in metric_rows]
        k_vals = [r["Best k"] for r in metric_rows]
        center_vals = [r["Best center"] for r in metric_rows]

        consensus[metric] = {
            "max_df": mode_with_tiebreak(max_df_vals, prefer="higher"),
            "rrf_k": mode_with_tiebreak(rrf_k_vals, prefer="lower"),
            "k": float(statistics.median(k_vals)),
            "center": float(statistics.median(center_vals)),
        }

    return consensus


def save_distribution_plot(rows, output_path):
    """Create a 2x2 grouped-frequency chart for winning parameters."""
    colors = {"JSD": "#1f77b4", "KLD": "#ff7f0e", "CE": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for ax, (col_name, label) in zip(axes, PARAMS):
        values_by_metric = {
            metric: [r[col_name] for r in rows if r["Metric"] == metric]
            for metric in METRICS
        }

        unique_vals = sorted(set(v for vals in values_by_metric.values() for v in vals))
        x_positions = list(range(len(unique_vals)))
        width = 0.22

        for idx, metric in enumerate(METRICS):
            counts = Counter(values_by_metric[metric])
            heights = [counts.get(v, 0) for v in unique_vals]
            offset = (idx - 1) * width
            ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=width,
                label=metric,
                color=colors[metric],
            )

        ax.set_title(f"{label} winners")
        ax.set_xticks(x_positions)

        # Keep integer-style tick labels for rrf_k and compact float labels otherwise.
        if label == "rrf_k":
            tick_labels = [str(int(v)) for v in unique_vals]
        else:
            tick_labels = [f"{v:g}" for v in unique_vals]

        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Dynamic wRRF Winning Parameter Distributions", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_yaml_summary(model_name, consensus):
    """Print zero-shot consensus as copy-paste friendly YAML."""
    print("zero_shot_dynamic_wrrf:")
    for metric in METRICS:
        if metric not in consensus:
            continue
        params = consensus[metric]
        print(f"  {metric}:")
        print(f"    max_df: {params['max_df']:g}")
        print(f"    rrf_k: {params['rrf_k']}")
        print(f"    k: {params['k']:g}")
        print(f"    center: {params['center']:g}")
    print(f"  model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best dynamic params and produce zero-shot consensus + plots."
    )
    parser.add_argument(
        "--model-name",
        default="bge-m3",
        help="Model folder name under data/results (default: bge-m3)",
    )
    args = parser.parse_args()

    base_dir = os.path.join("data", "results", args.model_name)
    csv_path = os.path.join(base_dir, "best_dynamic_params.csv")
    plot_path = os.path.join(base_dir, "parameter_distributions.png")

    rows = load_rows(csv_path)
    consensus = compute_consensus(rows)
    save_distribution_plot(rows, plot_path)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved plot : {plot_path}")
    print_yaml_summary(args.model_name, consensus)


if __name__ == "__main__":
    main()
