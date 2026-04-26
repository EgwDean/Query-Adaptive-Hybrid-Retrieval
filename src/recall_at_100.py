"""
recall_at_100.py

Computes Recall@100 for 6 retrieval methods on the same test split used in
the NDCG evaluation (balanced 5×300 query pool from meta_learner_moe_grid_search.py,
≈45 test queries per dataset).

Alpha values (weak, strong) are loaded directly from the cached meta-dataset
CSV so no feature recomputation is needed.  The MoE alpha is obtained by
retraining the best saved meta-learner on the traindev portion of the CSV.

An additional "Union BM25∪Dense" column / bar shows the theoretical Recall
ceiling for any re-ranker that draws candidates from both lists.

Methods:
  BM25                 : pure BM25 top-100
  Dense                : pure dense top-100
  Static wRRF (α=0.5)  : fused top-100 with fixed α
  wRRF (weak)          : fused top-100 with per-query α from weak-signal XGBoost
  wRRF (strong)        : fused top-100 with per-query α from strong-signal XGBoost
  wRRF (MoE)           : fused top-100 with per-query α from MoE meta-learner
  Union BM25∪Dense     : all unique docs in both candidate sets (theoretical ceiling)

Outputs:
  data/results/recall_at_100.csv
  data/results/recall_at_100.png

Usage:
  python src/recall_at_100.py
"""

import csv
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from src.utils import ensure_dir, get_config_path, load_config
from src.weak_signal_model_grid_search import (
    dataset_seed_offset,   # noqa: F401 – kept for parity; split lives in the CSV
    load_dataset_for_grid_search,
    set_global_seed,
)
from src.meta_learner_moe_grid_search import (
    _fit_meta,
    _load_meta_dataset,
    _meta_features,
    _pred_meta,
)


# ── Recall / retrieval helpers ────────────────────────────────────────────────

def _query_recall(retrieved_ids, qrel):
    """
    Recall for one query.
    retrieved_ids : iterable of doc IDs (order does not matter — already sliced)
    qrel          : {doc_id: relevance_grade}
    Returns None when the query has no relevant documents (excluded from mean).
    """
    relevant = {d for d, g in qrel.items() if g > 0}
    if not relevant:
        return None
    return len(relevant & set(retrieved_ids)) / len(relevant)


def _rrf_top_k(alpha, qid, bm25_res, dense_res, rrf_k, k):
    """
    Fuse BM25 and Dense via wRRF for one query and return the top-k doc IDs.
    Documents missing from one list are penalised by using (list_len + 1) as rank.
    """
    bm_pairs = bm25_res.get(qid, [])
    de_pairs = dense_res.get(qid, [])
    bm_rank  = {d: r for r, (d, _) in enumerate(bm_pairs, 1)}
    de_rank  = {d: r for r, (d, _) in enumerate(de_pairs, 1)}
    bm_miss  = len(bm_pairs) + 1
    de_miss  = len(de_pairs) + 1
    alpha    = float(alpha)
    fused    = {
        d: alpha / (rrf_k + bm_rank.get(d, bm_miss))
           + (1.0 - alpha) / (rrf_k + de_rank.get(d, de_miss))
        for d in set(bm_rank) | set(de_rank)
    }
    ranked = sorted(fused, key=fused.__getitem__, reverse=True)
    return ranked[:k]


def _mean_recall_single(qids, qrel_map, get_ids_fn):
    """
    Average Recall over qids.
    get_ids_fn(qid) -> iterable of retrieved doc IDs
    Queries with no relevant documents are excluded from the average.
    """
    scores = []
    for qid in qids:
        r = _query_recall(get_ids_fn(qid), qrel_map.get(qid, {}))
        if r is not None:
            scores.append(r)
    return float(np.mean(scores)) if scores else 0.0


# ── Bar chart ─────────────────────────────────────────────────────────────────

def _save_bar_chart(rows, methods, labels, colors, top_k, out_path):
    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    n_m     = len(methods)
    width   = 0.11
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(18, 6))
    for method, label, color, off in zip(methods, labels, colors, offsets):
        vals = [r[method] for r in rows]
        bars = ax.bar(x + off, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"Recall@{top_k}", fontsize=10)
    ax.set_title(f"Recall@{top_k} by Retrieval Method", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    all_vals = [r[m] for r in rows for m in methods]
    ax.set_ylim(0, min(1.02, max(all_vals) + 0.14))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg  = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    cuda_available = torch.cuda.is_available()
    device         = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device: {device}")

    dataset_names = cfg["datasets"]
    top_k         = int(cfg["benchmark"]["top_k"])   # 100
    rrf_k         = int(cfg["benchmark"]["rrf"]["k"])  # 60

    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)

    # ── Load meta-dataset CSV (has alpha_weak / alpha_strong per query) ────────
    meta_csv = os.path.join(results_folder, "meta_learner_dataset.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"Meta-dataset CSV not found: {meta_csv}\n"
            "Run  python src/meta_learner_moe_grid_search.py  first."
        )
    all_rows = _load_meta_dataset(meta_csv)
    td_rows  = [r for r in all_rows if r["split"] == "traindev"]
    te_rows  = [r for r in all_rows if r["split"] == "test"]
    print(f"Meta-dataset: {len(td_rows)} traindev | {len(te_rows)} test queries")

    td_aw = np.array([r["alpha_weak"]   for r in td_rows], dtype=np.float32)
    td_as = np.array([r["alpha_strong"] for r in td_rows], dtype=np.float32)
    td_gt = np.array([r["alpha_gt"]     for r in td_rows], dtype=np.float32)
    te_aw = np.array([r["alpha_weak"]   for r in te_rows], dtype=np.float32)
    te_as = np.array([r["alpha_strong"] for r in te_rows], dtype=np.float32)

    # ── Load best MoE params and retrain on full traindev ─────────────────────
    moe_params_csv = os.path.join(results_folder, "meta_learner_best_params.csv")
    if not os.path.exists(moe_params_csv):
        raise FileNotFoundError(
            f"Best MoE params not found: {moe_params_csv}\n"
            "Run  python src/meta_learner_moe_grid_search.py  first."
        )
    with open(moe_params_csv, newline="", encoding="utf-8") as f:
        moe_row        = next(iter(csv.DictReader(f)))
        moe_model_name = moe_row["model"]
        moe_params     = json.loads(moe_row["params_json"])

    print(f"MoE model : {moe_model_name}  |  params: {moe_params}")
    X_td_meta = _meta_features(td_aw, td_as, moe_model_name)
    X_te_meta = _meta_features(te_aw, te_as, moe_model_name)
    print("Retraining MoE on full traindev ...")
    final_moe = _fit_meta(moe_model_name, moe_params, X_td_meta, td_gt)
    te_moe    = _pred_meta(final_moe, X_te_meta)

    # ── Load BM25 + Dense retrieval results ───────────────────────────────────
    print("\n=== Loading retrieval results ===")
    retrieval = {}
    for ds_name in dataset_names:
        print(f"  {ds_name} ...")
        wd = load_dataset_for_grid_search(ds_name, cfg, device)
        retrieval[ds_name] = {
            "bm25":  wd["bm25_results"],
            "dense": wd["dense_results"],
            "qrels": wd["qrels"],
        }
    if cuda_available:
        torch.cuda.empty_cache()

    # ── Per-dataset Recall@100 ────────────────────────────────────────────────
    print(f"\n=== Recall@{top_k} (test set) ===")
    comparison_rows = []

    for ds_name in dataset_names:
        ds_te = [r for r in te_rows if r["ds_name"] == ds_name]
        if not ds_te:
            print(f"  {ds_name}: no test queries — skipping")
            continue

        qids_ds  = [r["qid"] for r in ds_te]
        te_idx   = [i for i, r in enumerate(te_rows) if r["ds_name"] == ds_name]
        aw_ds    = te_aw[te_idx]
        as_ds    = te_as[te_idx]
        moe_ds   = te_moe[te_idx]

        bm25_r = retrieval[ds_name]["bm25"]
        den_r  = retrieval[ds_name]["dense"]
        qrels  = retrieval[ds_name]["qrels"]

        # Single-ranker recalls — captured vars are safe inside list comprehension
        bm25_recall  = _mean_recall_single(
            qids_ds, qrels,
            lambda q, b=bm25_r: [d for d, _ in b.get(q, [])[:top_k]],
        )
        dense_recall = _mean_recall_single(
            qids_ds, qrels,
            lambda q, d=den_r: [doc for doc, _ in d.get(q, [])[:top_k]],
        )

        # Static wRRF
        srrf_recall = _mean_recall_single(
            qids_ds, qrels,
            lambda q, b=bm25_r, d=den_r: _rrf_top_k(0.5, q, b, d, rrf_k, top_k),
        )

        # Per-query alpha methods
        wrrf_w_scores, wrrf_s_scores, moe_scores = [], [], []
        for qid, aw, as_, ma in zip(qids_ds, aw_ds, as_ds, moe_ds):
            qrel = qrels.get(qid, {})
            for score_list, alpha in [
                (wrrf_w_scores, aw),
                (wrrf_s_scores, as_),
                (moe_scores,    ma),
            ]:
                r = _query_recall(
                    _rrf_top_k(alpha, qid, bm25_r, den_r, rrf_k, top_k), qrel
                )
                if r is not None:
                    score_list.append(r)

        wrrf_w_recall  = float(np.mean(wrrf_w_scores))  if wrrf_w_scores  else 0.0
        wrrf_s_recall  = float(np.mean(wrrf_s_scores))  if wrrf_s_scores  else 0.0
        moe_recall     = float(np.mean(moe_scores))      if moe_scores     else 0.0

        # Theoretical ceiling: union of both candidate sets (unranked)
        union_scores = []
        for qid in qids_ds:
            qrel     = qrels.get(qid, {})
            relevant = {d for d, g in qrel.items() if g > 0}
            if not relevant:
                continue
            union = (
                {d for d, _ in bm25_r.get(qid, [])}
                | {d for d, _ in den_r.get(qid, [])}
            )
            union_scores.append(len(relevant & union) / len(relevant))
        union_recall = float(np.mean(union_scores)) if union_scores else 0.0

        print(f"\n  {ds_name}  ({len(qids_ds)} test queries):")
        for lbl, val in [
            ("BM25",              bm25_recall),
            ("Dense",             dense_recall),
            ("Static wRRF",       srrf_recall),
            ("wRRF (weak)",       wrrf_w_recall),
            ("wRRF (strong)",     wrrf_s_recall),
            ("wRRF (MoE)",        moe_recall),
            ("Union BM25∪Dense", union_recall),
        ]:
            print(f"    {lbl:<22} Recall@{top_k} = {val:.4f}")

        comparison_rows.append({
            "group":       ds_name,
            "bm25":        bm25_recall,
            "dense":       dense_recall,
            "static_rrf":  srrf_recall,
            "wrrf_weak":   wrrf_w_recall,
            "wrrf_strong": wrrf_s_recall,
            "moe":         moe_recall,
            "union":       union_recall,
        })

    # Macro average
    method_keys = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe", "union"]
    macro = {m: float(np.mean([r[m] for r in comparison_rows])) for m in method_keys}
    comparison_rows.append({"group": "MACRO", **macro})

    print(f"\n{'=' * 60}")
    print("Macro averages:")
    for lbl, key in [
        ("BM25",              "bm25"),
        ("Dense",             "dense"),
        ("Static wRRF",       "static_rrf"),
        ("wRRF (weak)",       "wrrf_weak"),
        ("wRRF (strong)",     "wrrf_strong"),
        ("wRRF (MoE)",        "moe"),
        ("Union BM25∪Dense", "union"),
    ]:
        print(f"  {lbl:<22} Recall@{top_k} = {macro[key]:.4f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_out = os.path.join(results_folder, "recall_at_100.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "bm25_recall@100", "dense_recall@100",
            "static_rrf_recall@100",
            "wrrf_weak_recall@100", "wrrf_strong_recall@100",
            "moe_recall@100", "union_recall@100",
        ])
        for r in comparison_rows:
            w.writerow([
                r["group"],
                f"{r['bm25']:.6f}",      f"{r['dense']:.6f}",
                f"{r['static_rrf']:.6f}",
                f"{r['wrrf_weak']:.6f}",  f"{r['wrrf_strong']:.6f}",
                f"{r['moe']:.6f}",        f"{r['union']:.6f}",
            ])
    print(f"\nCSV saved: {csv_out}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    methods = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe", "union"]
    labels  = [
        "BM25", "Dense", "Static wRRF (α=0.5)",
        "wRRF (weak)", "wRRF (strong)", "MoE Meta-Learner",
        "Union BM25∪Dense (ceiling)",
    ]
    colors  = [
        "#4878D0", "#EE854A", "#6ACC65",
        "#D65F5F", "#B47CC7", "#82C6E2",
        "#BBBBBB",
    ]
    png_out = os.path.join(results_folder, "recall_at_100.png")
    _save_bar_chart(comparison_rows, methods, labels, colors, top_k, png_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
