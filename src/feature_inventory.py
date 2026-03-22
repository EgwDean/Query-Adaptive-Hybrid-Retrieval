"""feature_inventory.py

Central feature inventory definitions for supervised routing and ablation.

This module keeps deterministic ordering, group definitions, and a compact
feature catalog used for audit outputs.
"""

CURRENT_FEATURES = [
    "cross_entropy",
    "agreement",
    "overlap_at_3",
    "query_length",
    "dense_confidence",
    "sparse_confidence",
    "confidence_gap",
    "average_idf",
    "max_idf",
    "idf_std",
    "rare_term_ratio",
    "stopword_ratio",
]

RESTORED_LEGACY_FEATURES = [
    "top_dense_score",
    "top_sparse_score",
    "top_score_gap",
]

EXPANDED_FEATURES = CURRENT_FEATURES + RESTORED_LEGACY_FEATURES

FEATURE_GROUPS_CURRENT = {
    "overlap": ["agreement", "overlap_at_3"],
    "confidence": ["dense_confidence", "sparse_confidence", "confidence_gap"],
    "idf": ["average_idf", "max_idf", "idf_std", "rare_term_ratio"],
    "query": ["query_length", "stopword_ratio"],
    "entropy": ["cross_entropy"],
}

FEATURE_GROUPS_EXPANDED = {
    **FEATURE_GROUPS_CURRENT,
    "legacy_topscore": ["top_dense_score", "top_sparse_score", "top_score_gap"],
}

# Keep this aligned with feature definition changes to force cache refresh when needed.
FEATURE_SCHEMA_VERSION = "routing_features_v2_expanded_legacy_topscore"


def get_feature_inventory(inventory_name):
    """Return feature list and grouped definitions for the requested inventory."""
    key = str(inventory_name).strip().lower()
    if key == "current":
        return list(CURRENT_FEATURES), dict(FEATURE_GROUPS_CURRENT)
    if key == "expanded":
        return list(EXPANDED_FEATURES), dict(FEATURE_GROUPS_EXPANDED)
    raise ValueError(f"Unsupported feature inventory: {inventory_name!r}.")


def inventory_membership(feature_name):
    """Return inventory membership labels for one feature."""
    labels = []
    if feature_name in CURRENT_FEATURES:
        labels.append("current")
    if feature_name in EXPANDED_FEATURES:
        labels.append("expanded")
    return labels


def build_feature_catalog():
    """Return audit catalog with implemented and skipped doc-grounded features."""
    rows = []

    for name in CURRENT_FEATURES:
        rows.append(
            {
                "feature": name,
                "source": "current",
                "inventories": inventory_membership(name),
                "implemented": True,
                "reason": "Active feature in current routing pipeline.",
            }
        )

    for name in RESTORED_LEGACY_FEATURES:
        rows.append(
            {
                "feature": name,
                "source": "restored_from_docs",
                "inventories": inventory_membership(name),
                "implemented": True,
                "reason": (
                    "Restored from docs as normalized score-shape features; "
                    "raw-score variants remain excluded."
                ),
            }
        )

    rows.extend(
        [
            {
                "feature": "raw_top_dense_score",
                "source": "restored_from_docs",
                "inventories": [],
                "implemented": False,
                "reason": "Skipped: raw dense score scale is not comparable/reliable across settings.",
            },
            {
                "feature": "raw_top_sparse_score",
                "source": "restored_from_docs",
                "inventories": [],
                "implemented": False,
                "reason": "Skipped: raw BM25 score scale is not comparable to dense scale.",
            },
            {
                "feature": "raw_dense_confidence",
                "source": "restored_from_docs",
                "inventories": [],
                "implemented": False,
                "reason": "Skipped: raw margin is retriever-scale dependent and unstable.",
            },
            {
                "feature": "raw_sparse_confidence",
                "source": "restored_from_docs",
                "inventories": [],
                "implemented": False,
                "reason": "Skipped: raw margin is retriever-scale dependent and unstable.",
            },
            {
                "feature": "raw_cross_retriever_score_differences",
                "source": "restored_from_docs",
                "inventories": [],
                "implemented": False,
                "reason": "Skipped: invalid cross-retriever raw-scale comparison.",
            },
        ]
    )

    return rows
