"""
rerank_evaluation.py

Applies a cross-encoder re-ranker to candidate pools from all 6 retrieval
methods and measures NDCG@10 before and after re-ranking.

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 / BAAI/bge-reranker-base
  (download once via sentence-transformers on first run)

Efficiency:
  All unique (query, doc) pairs across every method for a dataset are
  collected and scored in a single batched cross-encoder call, so shared
  candidates are never scored twice.  CUDA is used when available.
  CE scores are cached per dataset; subsequent runs skip re-encoding.

Test split: same balanced 5x300 pool used in meta_learner_moe_grid_search.py
  (alpha_weak / alpha_strong loaded from the cached meta-dataset CSV;
   MoE alpha reproduced by retraining the saved best meta-learner).

Outputs:
  data/results/rerank_ndcg_comparison.csv   -- original + re-ranked NDCG@10
  data/results/rerank_ndcg_comparison.png   -- re-ranked NDCG@10 bar chart
  data/results/rerank_ndcg_improvement.png  -- NDCG@10 gain bar chart

Usage:
  pip install sentence-transformers   # if not already installed
  python src/rerank_evaluation.py
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
from src.utils import (
    ensure_dir,
    get_config_path,
    load_config,
    load_corpus_batch_generator,
    load_pickle,
    load_queries,
    model_short_name,
    save_pickle,
)
from src.weak_signal_model_grid_search import (
    load_dataset_for_grid_search,
    query_ndcg_at_k,
    set_global_seed,
)
from src.meta_learner_moe_grid_search import (
    _fit_meta,
    _load_meta_dataset,
    _meta_features,
    _pred_meta,
)

CE_MODEL           = "BAAI/bge-reranker-base"
CE_BATCH_CUDA      = 128
CE_BATCH_CPU       = 32
METHODS            = ["bm25", "dense", "static_rrf", "wrrf_weak", "wrrf_strong", "moe"]
METHOD_LABELS      = [
    "BM25", "Dense", "Static wRRF (α=0.5)",
    "wRRF (weak)", "wRRF (strong)", "MoE Meta-Learner",
]
METHOD_COLORS      = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#B47CC7", "#82C6E2"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_corpus(corpus_jsonl):
    """Return {doc_id: text} for the whole corpus."""
    corpus = {}
    for ids_batch, texts_batch in load_corpus_batch_generator(corpus_jsonl, 1024):
        for did, txt in zip(ids_batch, texts_batch):
            corpus[did] = txt
    return corpus


def _rrf_ranked(alpha, qid, bm25_res, dense_res, rrf_k, top_k):
    """
    wRRF fusion for one query.
    Returns [(doc_id, rrf_score)] sorted descending, length <= top_k.
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
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _candidates(alpha, qid, bm25_res, dense_res, rrf_k, top_k, method):
    """Return list of (doc_id, original_score) for a method's candidate pool."""
    if method == "bm25":
        return bm25_res.get(qid, [])[:top_k]
    if method == "dense":
        return dense_res.get(qid, [])[:top_k]
    return _rrf_ranked(alpha, qid, bm25_res, dense_res, rrf_k, top_k)


# ── Cross-encoder scoring (per dataset) ──────────────────────────────────────

def _score_dataset(
    ce_model, qids, query_texts, corpus,
    bm25_res, dense_res, rrf_k, top_k,
    aw_arr, as_arr, moe_arr,
    cache_path,
):
    """
    Score all unique (query, doc) pairs for a dataset with the cross-encoder.
    Returns score_map: {(qid, doc_id): float}.
    Scores are loaded from cache_path if it exists and covers all pairs.
    """
    # Build the full set of required pairs
    required = {}   # (qid, doc_id) -> True
    alphas_by_qid = {
        qid: {"aw": aw, "as_": as_, "moe": moe}
        for qid, aw, as_, moe in zip(qids, aw_arr, as_arr, moe_arr)
    }
    for qid in qids:
        ab    = alphas_by_qid[qid]
        bm_ids = [d for d, _ in bm25_res.get(qid, [])[:top_k]]
        de_ids = [d for d, _ in dense_res.get(qid, [])[:top_k]]
        sr_ids = [d for d, _ in _rrf_ranked(0.5,        qid, bm25_res, dense_res, rrf_k, top_k)]
        ww_ids = [d for d, _ in _rrf_ranked(ab["aw"],   qid, bm25_res, dense_res, rrf_k, top_k)]
        ws_ids = [d for d, _ in _rrf_ranked(ab["as_"],  qid, bm25_res, dense_res, rrf_k, top_k)]
        mo_ids = [d for d, _ in _rrf_ranked(ab["moe"],  qid, bm25_res, dense_res, rrf_k, top_k)]
        for doc_id in set(bm_ids + de_ids + sr_ids + ww_ids + ws_ids + mo_ids):
            required[(qid, doc_id)] = True

    # Try cache
    if os.path.exists(cache_path):
        try:
            cached = load_pickle(cache_path)
            if all(k in cached for k in required):
                print(f"  CE score cache hit ({len(required):,} pairs).")
                return cached
            print(f"  CE score cache partial — rebuilding ({len(required):,} pairs).")
        except Exception as exc:
            print(f"  [WARN] CE score cache corrupt; rebuilding. ({exc})")

    # Build pair list (preserve insertion order for index mapping)
    pair_index = {}
    pair_texts = []
    for qid in qids:
        q_text = query_texts[qid]
        ab     = alphas_by_qid[qid]
        bm_ids = [d for d, _ in bm25_res.get(qid, [])[:top_k]]
        de_ids = [d for d, _ in dense_res.get(qid, [])[:top_k]]
        sr_ids = [d for d, _ in _rrf_ranked(0.5,        qid, bm25_res, dense_res, rrf_k, top_k)]
        ww_ids = [d for d, _ in _rrf_ranked(ab["aw"],   qid, bm25_res, dense_res, rrf_k, top_k)]
        ws_ids = [d for d, _ in _rrf_ranked(ab["as_"],  qid, bm25_res, dense_res, rrf_k, top_k)]
        mo_ids = [d for d, _ in _rrf_ranked(ab["moe"],  qid, bm25_res, dense_res, rrf_k, top_k)]
        unique_docs = dict.fromkeys(bm_ids + de_ids + sr_ids + ww_ids + ws_ids + mo_ids)
        for doc_id in unique_docs:
            if (qid, doc_id) not in pair_index:
                doc_text = corpus.get(doc_id, "")
                if not doc_text:
                    print(f"  [WARN] doc_id '{doc_id}' not found in corpus.")
                pair_index[(qid, doc_id)] = len(pair_texts)
                pair_texts.append((q_text, doc_text))

    print(f"  Scoring {len(pair_texts):,} unique (query, doc) pairs with cross-encoder ...")
    raw_scores = ce_model.predict(
        pair_texts,
        batch_size=CE_BATCH_CUDA if torch.cuda.is_available() else CE_BATCH_CPU,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    score_map = {pair: float(raw_scores[idx]) for pair, idx in pair_index.items()}
    save_pickle(score_map, cache_path)
    print(f"  CE scores cached: {cache_path}")
    return score_map


# ── Bar charts ────────────────────────────────────────────────────────────────

def _save_comparison_chart(rows, ndcg_k, out_path):
    """Re-ranked NDCG@k bar chart, one bar per method per dataset group."""
    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    n_m     = len(METHODS)
    width   = 0.12
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(16, 6))
    for method, label, color, off in zip(METHODS, METHOD_LABELS, METHOD_COLORS, offsets):
        vals = [r[f"{method}_reranked"] for r in rows]
        bars = ax.bar(x + off, vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"NDCG@{ndcg_k}", fontsize=10)
    ax.set_title(f"NDCG@{ndcg_k} after Cross-Encoder Re-ranking", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    all_vals = [r[f"{m}_reranked"] for r in rows for m in METHODS]
    ax.set_ylim(0, min(1.0, max(all_vals) + 0.12))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


def _save_improvement_chart(rows, ndcg_k, out_path):
    """NDCG@k gain (re-ranked minus original) bar chart."""
    groups  = [r["group"] for r in rows]
    x       = np.arange(len(groups))
    n_m     = len(METHODS)
    width   = 0.12
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(16, 6))
    for method, label, color, off in zip(METHODS, METHOD_LABELS, METHOD_COLORS, offsets):
        deltas = [r[f"{method}_reranked"] - r[f"{method}_original"] for r in rows]
        bars   = ax.bar(x + off, deltas, width, label=label,
                        color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.002 if h >= 0 else -0.010),
                f"{h:+.3f}",
                ha="center", va="bottom", fontsize=5.5, rotation=90,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(f"ΔNDCG@{ndcg_k}  (re-ranked − original)", fontsize=10)
    ax.set_title(f"NDCG@{ndcg_k} Gain from Cross-Encoder Re-ranking", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg  = load_config()
    seed = int(cfg.get("routing_features", {}).get("seed", 42))
    set_global_seed(seed)

    cuda_available = torch.cuda.is_available()
    device_str     = "cuda" if cuda_available else "cpu"
    device         = torch.device(device_str)
    print(f"Device      : {device}")

    dataset_names  = cfg["datasets"]
    top_k          = int(cfg["benchmark"]["top_k"])
    ndcg_k         = int(cfg["benchmark"]["ndcg_k"])
    rrf_k          = int(cfg["benchmark"]["rrf"]["k"])
    short_model    = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    results_folder = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_folder)

    # ── Load cross-encoder ────────────────────────────────────────────────────
    print(f"\nLoading cross-encoder: {CE_MODEL}")
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Run:  pip install sentence-transformers"
        )
    ce_model = CrossEncoder(CE_MODEL, max_length=512, device=device_str)
    print(f"Cross-encoder loaded on {device_str}.")

    # ── Load meta-dataset CSV (alpha_weak / alpha_strong per query) ───────────
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

    # ── Retrain MoE on traindev → test predictions ────────────────────────────
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

    print(f"MoE model  : {moe_model_name}  |  params: {moe_params}")
    X_td_meta = _meta_features(td_aw, td_as, moe_model_name)
    X_te_meta = _meta_features(te_aw, te_as, moe_model_name)
    print("Retraining MoE on full traindev ...")
    final_moe = _fit_meta(moe_model_name, moe_params, X_td_meta, td_gt)
    te_moe    = _pred_meta(final_moe, X_te_meta)

    # ── Per-dataset evaluation ────────────────────────────────────────────────
    print(f"\n=== Cross-encoder re-ranking evaluation (NDCG@{ndcg_k}) ===")
    comparison_rows = []

    ce_short = CE_MODEL.replace("/", "_").replace("-", "_")

    for ds_name in dataset_names:
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_name}")

        # Identify test queries for this dataset
        ds_te = [r for r in te_rows if r["ds_name"] == ds_name]
        if not ds_te:
            print(f"  No test queries — skipping.")
            continue

        qids_ds = [r["qid"] for r in ds_te]
        te_idx  = [i for i, r in enumerate(te_rows) if r["ds_name"] == ds_name]
        aw_ds   = te_aw[te_idx]
        as_ds   = te_as[te_idx]
        moe_ds  = te_moe[te_idx]

        # Load retrieval results and qrels
        print(f"  Loading retrieval results ...")
        wd     = load_dataset_for_grid_search(ds_name, cfg, device)
        bm25_r = wd["bm25_results"]
        den_r  = wd["dense_results"]
        qrels  = wd["qrels"]

        if cuda_available:
            torch.cuda.empty_cache()

        # Load corpus texts
        corpus_jsonl = os.path.join(processed_root, short_model, ds_name, "corpus.jsonl")
        if not os.path.exists(corpus_jsonl):
            raise FileNotFoundError(
                f"Corpus JSONL not found: {corpus_jsonl}\n"
                "Run  python src/preprocess.py  first."
            )
        print(f"  Loading corpus texts ...")
        corpus = _load_corpus(corpus_jsonl)

        # Load query texts
        queries_jsonl = os.path.join(processed_root, short_model, ds_name, "queries.jsonl")
        if not os.path.exists(queries_jsonl):
            raise FileNotFoundError(
                f"Queries JSONL not found: {queries_jsonl}\n"
                "Run  python src/preprocess.py  first."
            )
        query_texts = load_queries(queries_jsonl)

        # Cross-encoder scoring (batched, cached)
        cache_path = os.path.join(
            results_folder, f"rerank_scores_{ce_short}_{ds_name}.pkl"
        )
        score_map = _score_dataset(
            ce_model, qids_ds, query_texts, corpus,
            bm25_r, den_r, rrf_k, top_k,
            aw_ds, as_ds, moe_ds,
            cache_path,
        )

        # ── Per-query NDCG@10 before and after re-ranking ─────────────────────
        orig_scores    = {m: [] for m in METHODS}
        reranked_scores = {m: [] for m in METHODS}

        alpha_map = {
            "bm25":        None,
            "dense":       None,
            "static_rrf":  0.5,
        }

        for qid, aw, as_, ma in zip(qids_ds, aw_ds, as_ds, moe_ds):
            qrel = qrels.get(qid, {})
            if not any(g > 0 for g in qrel.values()):
                continue   # skip queries with no relevant docs

            per_query_alpha = {
                "bm25":        None,
                "dense":       None,
                "static_rrf":  0.5,
                "wrrf_weak":   float(aw),
                "wrrf_strong": float(as_),
                "moe":         float(ma),
            }

            for method in METHODS:
                alpha = per_query_alpha[method]
                cands = _candidates(alpha, qid, bm25_r, den_r, rrf_k, top_k, method)

                # Original NDCG@10
                orig_ndcg = query_ndcg_at_k(cands, qrel, ndcg_k)
                orig_scores[method].append(orig_ndcg)

                # Re-ranked NDCG@10
                doc_ids   = [d for d, _ in cands]
                reranked  = sorted(
                    doc_ids,
                    key=lambda d: score_map.get((qid, d), -1e9),
                    reverse=True,
                )
                reranked_with_scores = [
                    (d, score_map.get((qid, d), -1e9)) for d in reranked
                ]
                reranked_ndcg = query_ndcg_at_k(reranked_with_scores, qrel, ndcg_k)
                reranked_scores[method].append(reranked_ndcg)

        row = {"group": ds_name}
        print(f"\n  {ds_name}  ({len(qids_ds)} test queries):")
        print(f"  {'Method':<20} {'Original':>10} {'Re-ranked':>10} {'Gain':>8}")
        print(f"  {'─'*50}")
        for method, label in zip(METHODS, METHOD_LABELS):
            orig_mean     = float(np.mean(orig_scores[method]))    if orig_scores[method]     else 0.0
            reranked_mean = float(np.mean(reranked_scores[method])) if reranked_scores[method] else 0.0
            delta         = reranked_mean - orig_mean
            row[f"{method}_original"] = orig_mean
            row[f"{method}_reranked"] = reranked_mean
            print(f"  {label:<20} {orig_mean:>10.4f} {reranked_mean:>10.4f} {delta:>+8.4f}")
        comparison_rows.append(row)

    # Macro averages
    macro = {"group": "MACRO"}
    for method in METHODS:
        macro[f"{method}_original"] = float(np.mean(
            [r[f"{method}_original"] for r in comparison_rows]
        ))
        macro[f"{method}_reranked"] = float(np.mean(
            [r[f"{method}_reranked"] for r in comparison_rows]
        ))
    comparison_rows.append(macro)

    print(f"\n{'='*60}")
    print("Macro averages:")
    print(f"  {'Method':<20} {'Original':>10} {'Re-ranked':>10} {'Gain':>8}")
    print(f"  {'─'*50}")
    for method, label in zip(METHODS, METHOD_LABELS):
        o = macro[f"{method}_original"]
        r = macro[f"{method}_reranked"]
        print(f"  {label:<20} {o:>10.4f} {r:>10.4f} {r-o:>+8.4f}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_out = os.path.join(results_folder, "rerank_ndcg_comparison.csv")
    header  = ["dataset"]
    for m in METHODS:
        header += [f"{m}_original_ndcg@{ndcg_k}", f"{m}_reranked_ndcg@{ndcg_k}"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in comparison_rows:
            row_vals = [r["group"]]
            for m in METHODS:
                row_vals += [
                    f"{r[f'{m}_original']:.6f}",
                    f"{r[f'{m}_reranked']:.6f}",
                ]
            w.writerow(row_vals)
    print(f"\nCSV saved: {csv_out}")

    # ── Charts ────────────────────────────────────────────────────────────────
    _save_comparison_chart(
        comparison_rows, ndcg_k,
        os.path.join(results_folder, "rerank_ndcg_comparison.png"),
    )
    _save_improvement_chart(
        comparison_rows, ndcg_k,
        os.path.join(results_folder, "rerank_ndcg_improvement.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
