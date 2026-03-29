"""analyze_feature_signal.py

Comprehensive feature-signal diagnostics for router label prediction.

This script is intended as a thesis-safe analysis utility to answer:
- Do features contain predictive signal for the label?
- Is the signal mostly linear or nonlinear?
- How far above dummy / shuffled-label controls are we?

Outputs are written under:
  data/results/<model_short_name>/analysis/feature_signal_diagnostics/
"""

import argparse
import csv
import os
import sys
import threading
from collections import defaultdict
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise RuntimeError(
        "xgboost is required for analyze_feature_signal.py. Install dependencies from requirements.txt."
    ) from exc

# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def load_table(csv_path):
    """Load CSV table and return (rows, fieldnames)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"Input CSV has no rows: {csv_path}")
    if not fieldnames:
        raise ValueError(f"Input CSV has no header: {csv_path}")

    return rows, fieldnames


def format_seconds(seconds):
    """Format elapsed seconds to a compact human-readable string."""
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - (60 * minutes)
    if minutes < 60:
        return f"{minutes}m {rem:.1f}s"
    hours = int(minutes // 60)
    minutes = minutes % 60
    return f"{hours}h {minutes}m {rem:.1f}s"


def log_line(message):
    """Write logs without breaking tqdm progress bars."""
    try:
        tqdm.write(str(message))
    except Exception:
        print(str(message), flush=True)


def run_with_heartbeat(fn, label, heartbeat_seconds=20.0, enabled=False, tracker=None):
    """Run a callable and print periodic heartbeat logs while it executes."""
    start = perf_counter()
    if enabled:
        log_line(f"  [progress] starting {label}")
    if (not enabled) or heartbeat_seconds <= 0:
        return fn(), perf_counter() - start

    stop_event = threading.Event()

    def _heartbeat_loop():
        while not stop_event.wait(heartbeat_seconds):
            now = perf_counter()
            msg = f"  [progress] still running {label} ({format_seconds(now - start)} elapsed)"
            if tracker is not None:
                done = int(tracker.get("completed", 0))
                total = int(tracker.get("total", 0))
                stage_start = float(tracker.get("stage_start", now))
                stage_elapsed = max(0.0, now - stage_start)
                if done > 0 and total > done:
                    avg_step = stage_elapsed / float(done)
                    eta = max(0.0, (total - done) * avg_step)
                    msg += f" | stage {done}/{total}, ETA ~ {format_seconds(eta)}"
                elif total > 0:
                    msg += f" | stage {done}/{total}"
            log_line(msg)

    t = threading.Thread(target=_heartbeat_loop, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop_event.set()
        t.join(timeout=0.1)

    elapsed = perf_counter() - start
    log_line(f"  [progress] completed {label} in {format_seconds(elapsed)}")
    return result, elapsed


def build_speed_profile(speed_profile_name):
    """Return runtime controls for full/fast/seconds diagnostic execution."""
    if speed_profile_name == "full":
        return {
            "name": "full",
            "max_rows": None,
            "max_cv_splits": None,
            "models_to_run": ["dummy_mean", "ridge", "random_forest", "xgboost"],
            "shuffled_models": ["ridge", "random_forest", "xgboost"],
            "xgb_n_estimators_cap": None,
            "xgb_max_depth_cap": None,
            "rf_n_estimators_cap": None,
            "perm_max_folds": None,
            "perm_repeats_cap": None,
            "shap_max_samples_cap": None,
            "heartbeat_default_seconds": 5.0,
        }
    if speed_profile_name == "fast":
        return {
            "name": "fast",
            "max_rows": 1800,
            "max_cv_splits": 3,
            "models_to_run": ["dummy_mean", "ridge", "xgboost"],
            "shuffled_models": ["ridge", "xgboost"],
            "xgb_n_estimators_cap": 120,
            "xgb_max_depth_cap": 4,
            "rf_n_estimators_cap": 120,
            "perm_max_folds": 2,
            "perm_repeats_cap": 3,
            "shap_max_samples_cap": 80,
            "heartbeat_default_seconds": 3.0,
        }
    if speed_profile_name == "seconds":
        return {
            "name": "seconds",
            "max_rows": 500,
            "max_cv_splits": 2,
            "models_to_run": ["dummy_mean", "xgboost"],
            "shuffled_models": ["xgboost"],
            "xgb_n_estimators_cap": 20,
            "xgb_max_depth_cap": 3,
            "rf_n_estimators_cap": 80,
            "perm_max_folds": 1,
            "perm_repeats_cap": 1,
            "shap_max_samples_cap": 20,
            "heartbeat_default_seconds": 2.0,
        }
    raise ValueError(f"Unknown speed profile: {speed_profile_name!r}")


def subsample_rows_by_group(X, y, groups, max_rows, seed):
    """Subsample rows with per-group quotas to preserve grouped-CV viability."""
    if max_rows is None or X.shape[0] <= int(max_rows):
        return X, y, groups, np.arange(X.shape[0], dtype=np.int64)

    max_rows = int(max_rows)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    unique_groups = np.unique(groups)

    group_indices = {}
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        group_indices[g] = idx

    quotas = {}
    for g in unique_groups:
        size_g = group_indices[g].shape[0]
        q = int(round((size_g / float(n)) * max_rows))
        quotas[g] = max(1, min(size_g, q))

    selected = []
    for g in unique_groups:
        idx = group_indices[g]
        q = quotas[g]
        if q >= idx.shape[0]:
            chosen = idx
        else:
            chosen = rng.choice(idx, size=q, replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))

    selected = np.concatenate(selected)

    if selected.shape[0] > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)
    elif selected.shape[0] < max_rows:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), selected, assume_unique=False)
        add = min(max_rows - selected.shape[0], remaining.shape[0])
        if add > 0:
            extra = rng.choice(remaining, size=add, replace=False)
            selected = np.concatenate([selected, extra])

    selected = np.sort(selected)
    return X[selected], y[selected], groups[selected], selected


def infer_feature_columns(fieldnames, label_column):
    """Infer feature columns by excluding known metadata columns."""
    excluded = {"query_id", "dataset", "alpha", "soft_label", label_column}
    feat_cols = [c for c in fieldnames if c not in excluded]
    if not feat_cols:
        raise ValueError("No feature columns inferred from input CSV.")
    return feat_cols


def rows_to_arrays(rows, feature_cols, label_column):
    """Convert analysis rows to arrays."""
    X = np.zeros((len(rows), len(feature_cols)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    groups = []

    for i, row in enumerate(rows):
        groups.append(str(row.get("dataset", "")))
        try:
            y[i] = float(row[label_column])
        except KeyError as exc:
            raise KeyError(
                f"Label column {label_column!r} not found. Use --label-column with an existing field."
            ) from exc

        for j, col in enumerate(feature_cols):
            try:
                X[i, j] = float(row[col])
            except Exception as exc:
                raise ValueError(f"Non-numeric value in feature column {col!r}, row {i}.") from exc

    return X, y, np.asarray(groups)


def pearson_corr(x, y):
    """Pearson correlation with zero-variance guard."""
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def distance_correlation_1d(x, y):
    """Distance correlation for 1D variables using the standard centered-distance form."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.shape[0]

    if n < 2:
        return float("nan")

    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])

    Ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
    Ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()

    dcov2 = float(np.mean(Ax * Ay))
    dvarx2 = float(np.mean(Ax * Ax))
    dvary2 = float(np.mean(Ay * Ay))

    denom = np.sqrt(max(dvarx2, 0.0) * max(dvary2, 0.0))
    if denom <= 1.0e-18:
        return 0.0

    dcor2 = max(dcov2, 0.0) / denom
    return float(np.sqrt(max(dcor2, 0.0)))


def make_models(cfg, seed, speed_profile):
    """Build linear and nonlinear regressors for comparison."""
    rf_cfg = cfg.get("random_forest", {}) or {}
    xgb_cfg = cfg.get("xgboost", {}) or {}

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    rf_n_estimators = int(rf_cfg.get("n_estimators", 300))
    if speed_profile.get("rf_n_estimators_cap") is not None:
        rf_n_estimators = min(rf_n_estimators, int(speed_profile["rf_n_estimators_cap"]))

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=None if rf_cfg.get("max_depth", 8) is None else int(rf_cfg.get("max_depth", 8)),
        min_samples_split=int(rf_cfg.get("min_samples_split", 8)),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 4)),
        max_features=rf_cfg.get("max_features", "sqrt"),
        bootstrap=bool(rf_cfg.get("bootstrap", True)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=seed,
    )

    xgb_n_estimators = int(xgb_cfg.get("n_estimators", 300))
    xgb_max_depth = int(xgb_cfg.get("max_depth", 6))
    if speed_profile.get("xgb_n_estimators_cap") is not None:
        xgb_n_estimators = min(xgb_n_estimators, int(speed_profile["xgb_n_estimators_cap"]))
    if speed_profile.get("xgb_max_depth_cap") is not None:
        xgb_max_depth = min(xgb_max_depth, int(speed_profile["xgb_max_depth_cap"]))

    xgb_params = {
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample": float(xgb_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
        "reg_lambda": float(xgb_cfg.get("reg_lambda", 1.0)),
        "reg_alpha": float(xgb_cfg.get("reg_alpha", 0.0)),
        "min_child_weight": float(xgb_cfg.get("min_child_weight", 1.0)),
        "objective": str(xgb_cfg.get("objective", "reg:squarederror")),
        "n_jobs": int(xgb_cfg.get("n_jobs", -1)),
        "random_state": seed,
    }

    # For diagnostic speed, prefer histogram tree method when not explicitly configured.
    if "tree_method" not in xgb_cfg and speed_profile["name"] != "full":
        xgb_params["tree_method"] = "hist"

    for opt_key in ["tree_method", "device", "predictor", "max_bin", "grow_policy", "sampling_method"]:
        if opt_key in xgb_cfg and xgb_cfg[opt_key] is not None:
            xgb_params[opt_key] = xgb_cfg[opt_key]

    xgb = XGBRegressor(**xgb_params)

    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": ridge,
        "random_forest": rf,
        "xgboost": xgb,
    }


def generate_splits(cv_mode, X, y, groups, n_splits, n_repeats, seed):
    """Generate CV splits with either grouped or random strategy."""
    n_samples = X.shape[0]
    if n_samples < 4:
        raise ValueError("Need at least 4 samples for stable cross-validation diagnostics.")

    if cv_mode == "grouped":
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Grouped CV requires at least 2 distinct dataset groups.")

        n_splits_eff = min(int(n_splits), len(unique_groups))
        if n_splits_eff < 2:
            raise ValueError("Grouped CV effective n_splits is < 2.")

        splitter = GroupKFold(n_splits=n_splits_eff)
        return list(splitter.split(X, y, groups=groups))

    n_splits_eff = min(int(n_splits), n_samples)
    if n_splits_eff < 2:
        raise ValueError("Random CV effective n_splits is < 2.")

    n_repeats_eff = max(1, int(n_repeats))
    splitter = RepeatedKFold(n_splits=n_splits_eff, n_repeats=n_repeats_eff, random_state=seed)
    return list(splitter.split(X, y))


def evaluate_models_with_controls(
    models,
    X,
    y,
    splits,
    seed,
    show_progress=True,
    heartbeat_seconds=20.0,
    shuffled_models=None,
):
    """Evaluate each model and shuffled-label control over CV splits."""
    rng = np.random.default_rng(seed)
    rows = []

    model_items = list(models.items())
    n_folds = len(splits)
    shuffled_models = set(shuffled_models or [])

    total_steps = 0
    for model_name, _ in model_items:
        total_steps += n_folds
        if model_name in shuffled_models:
            total_steps += n_folds

    pbar = None
    if show_progress:
        pbar = tqdm(total=total_steps, desc="CV model fits", dynamic_ncols=True)

    tracker = {
        "stage_start": perf_counter(),
        "completed": 0,
        "total": total_steps,
    }

    for model_name, model in model_items:
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            fitted = clone(model)
            fit_label = f"{model_name} fold {fold_idx}/{n_folds} (original fit)"
            _, _ = run_with_heartbeat(
                fn=lambda: fitted.fit(X_train, y_train),
                label=fit_label,
                heartbeat_seconds=heartbeat_seconds,
                enabled=show_progress,
                tracker=tracker,
            )
            pred = fitted.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "control": "original",
                    "fold": fold_idx,
                    "r2": float(r2_score(y_test, pred)),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "pred_label_pearson": pearson_corr(pred, y_test),
                }
            )

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(model=model_name, fold=f"{fold_idx}/{n_folds}")
            tracker["completed"] += 1

            if model_name not in shuffled_models:
                continue

            y_train_shuf = rng.permutation(y_train)
            fitted_shuf = clone(model)
            fit_label = f"{model_name} fold {fold_idx}/{n_folds} (shuffled fit)"
            _, _ = run_with_heartbeat(
                fn=lambda: fitted_shuf.fit(X_train, y_train_shuf),
                label=fit_label,
                heartbeat_seconds=heartbeat_seconds,
                enabled=show_progress,
                tracker=tracker,
            )
            pred_shuf = fitted_shuf.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "control": "shuffled_label",
                    "fold": fold_idx,
                    "r2": float(r2_score(y_test, pred_shuf)),
                    "mae": float(mean_absolute_error(y_test, pred_shuf)),
                    "pred_label_pearson": pearson_corr(pred_shuf, y_test),
                }
            )

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(model=f"{model_name}(shuf)", fold=f"{fold_idx}/{n_folds}")
            tracker["completed"] += 1

    if pbar is not None:
        pbar.close()

    return rows


def summarize_cv_rows(cv_rows):
    """Aggregate CV rows into summary statistics."""
    grouped = defaultdict(list)
    for row in cv_rows:
        grouped[(row["model"], row["control"])].append(row)

    summary = []
    for (model, control), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        r2_vals = np.asarray([r["r2"] for r in rows], dtype=np.float64)
        mae_vals = np.asarray([r["mae"] for r in rows], dtype=np.float64)
        corr_vals = np.asarray([r["pred_label_pearson"] for r in rows], dtype=np.float64)

        if np.all(np.isnan(corr_vals)):
            corr_mean = float("nan")
            corr_std = float("nan")
        else:
            corr_mean = float(np.nanmean(corr_vals))
            corr_std = float(np.nanstd(corr_vals))

        summary.append(
            {
                "model": model,
                "control": control,
                "n_folds": len(rows),
                "r2_mean": float(np.nanmean(r2_vals)),
                "r2_std": float(np.nanstd(r2_vals)),
                "mae_mean": float(np.nanmean(mae_vals)),
                "mae_std": float(np.nanstd(mae_vals)),
                "pred_label_pearson_mean": corr_mean,
                "pred_label_pearson_std": corr_std,
            }
        )

    return summary


def write_cv_outputs(cv_rows, cv_summary, out_dir):
    """Write detailed and summary CV outputs to CSV."""
    detail_csv = os.path.join(out_dir, "cv_detailed_rows.csv")
    summary_csv = os.path.join(out_dir, "cv_summary.csv")

    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "control", "fold", "r2", "mae", "pred_label_pearson"])
        for row in cv_rows:
            writer.writerow(
                [
                    row["model"],
                    row["control"],
                    row["fold"],
                    f"{row['r2']:.12f}",
                    f"{row['mae']:.12f}",
                    f"{row['pred_label_pearson']:.12f}",
                ]
            )

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "control",
                "n_folds",
                "r2_mean",
                "r2_std",
                "mae_mean",
                "mae_std",
                "pred_label_pearson_mean",
                "pred_label_pearson_std",
            ]
        )
        for row in cv_summary:
            writer.writerow(
                [
                    row["model"],
                    row["control"],
                    row["n_folds"],
                    f"{row['r2_mean']:.12f}",
                    f"{row['r2_std']:.12f}",
                    f"{row['mae_mean']:.12f}",
                    f"{row['mae_std']:.12f}",
                    f"{row['pred_label_pearson_mean']:.12f}",
                    f"{row['pred_label_pearson_std']:.12f}",
                ]
            )

    return detail_csv, summary_csv


def compute_feature_dependence_scores(X, y, feature_cols, seed, show_progress=True):
    """Compute per-feature dependence scores (linear + nonlinear)."""
    pearsons = []
    dcor_vals = []

    feature_iter = range(X.shape[1])
    if show_progress:
        feature_iter = tqdm(feature_iter, desc="Feature dependence", dynamic_ncols=True)

    for j in feature_iter:
        xj = X[:, j]
        pearsons.append(pearson_corr(xj, y))
        dcor_vals.append(distance_correlation_1d(xj, y))

    try:
        mi = mutual_info_regression(X, y, random_state=seed)
        mi = np.asarray(mi, dtype=np.float64)
    except Exception:
        mi = np.full(X.shape[1], np.nan, dtype=np.float64)

    rows = []
    for j, feat in enumerate(feature_cols):
        rows.append(
            {
                "feature_name": feat,
                "pearson_corr": float(pearsons[j]),
                "distance_corr": float(dcor_vals[j]),
                "mutual_info": float(mi[j]),
            }
        )

    rows.sort(
        key=lambda r: (
            -np.nan_to_num(r["distance_corr"], nan=-1.0),
            -np.nan_to_num(r["mutual_info"], nan=-1.0),
        )
    )
    return rows


def write_feature_dependence_csv(rows, out_csv):
    """Write per-feature dependence metrics CSV."""
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "pearson_corr", "distance_corr", "mutual_info"])
        for r in rows:
            writer.writerow(
                [
                    r["feature_name"],
                    f"{r['pearson_corr']:.12f}",
                    f"{r['distance_corr']:.12f}",
                    f"{r['mutual_info']:.12f}",
                ]
            )


def aggregate_permutation_importance(
    models,
    model_name,
    X,
    y,
    splits,
    feature_cols,
    seed,
    n_repeats,
    show_progress=True,
    heartbeat_seconds=20.0,
):
    """Estimate permutation importance on test folds and aggregate mean/std."""
    if model_name not in models:
        raise ValueError(f"Model {model_name!r} not found for permutation importance.")

    rng = np.random.default_rng(seed)
    fold_importances = []

    split_iter = splits
    if show_progress:
        split_iter = tqdm(splits, desc="Permutation importance folds", dynamic_ncols=True)

    total_folds = len(splits)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model = clone(models[model_name])
        fold_tracker = {
            "stage_start": perf_counter(),
            "completed": fold_idx - 1,
            "total": total_folds,
        }
        _, _ = run_with_heartbeat(
            fn=lambda: model.fit(X_train, y_train),
            label=f"permutation fold {fold_idx}/{total_folds} model fit",
            heartbeat_seconds=heartbeat_seconds,
            enabled=show_progress,
            tracker=fold_tracker,
        )

        perm_seed = int(rng.integers(0, 2**31 - 1))
        perm, _ = run_with_heartbeat(
            fn=lambda: permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=max(1, int(n_repeats)),
                random_state=perm_seed,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
            ),
            label=f"permutation fold {fold_idx}/{total_folds} scoring",
            heartbeat_seconds=heartbeat_seconds,
            enabled=show_progress,
            tracker=fold_tracker,
        )
        fold_importances.append(np.asarray(perm.importances_mean, dtype=np.float64))

    imp = np.vstack(fold_importances)
    rows = []
    for j, feat in enumerate(feature_cols):
        rows.append(
            {
                "feature_name": feat,
                "importance_mean": float(np.mean(imp[:, j])),
                "importance_std": float(np.std(imp[:, j])),
            }
        )

    rows.sort(key=lambda r: -r["importance_mean"])
    return rows


def write_permutation_csv(rows, out_csv):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "importance_mean", "importance_std"])
        for r in rows:
            writer.writerow(
                [
                    r["feature_name"],
                    f"{r['importance_mean']:.12f}",
                    f"{r['importance_std']:.12f}",
                ]
            )


def plot_cv_summary(cv_summary, out_png):
    """Plot CV r2 and MAE means with control bars."""
    labels = sorted({r["model"] for r in cv_summary})
    original = {r["model"]: r for r in cv_summary if r["control"] == "original"}
    shuffled = {r["model"]: r for r in cv_summary if r["control"] == "shuffled_label"}

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # R2 panel.
    r2_orig = [original.get(m, {}).get("r2_mean", np.nan) for m in labels]
    r2_shuf = [shuffled.get(m, {}).get("r2_mean", np.nan) for m in labels]
    axes[0].bar(x - width / 2, r2_orig, width=width, label="original")
    axes[0].bar(x + width / 2, r2_shuf, width=width, label="shuffled_label")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Cross-validated R2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("R2")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    # MAE panel (lower is better).
    mae_orig = [original.get(m, {}).get("mae_mean", np.nan) for m in labels]
    mae_shuf = [shuffled.get(m, {}).get("mae_mean", np.nan) for m in labels]
    axes[1].bar(x - width / 2, mae_orig, width=width, label="original")
    axes[1].bar(x + width / 2, mae_shuf, width=width, label="shuffled_label")
    axes[1].set_title("Cross-validated MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_feature_dependence(rows, out_png):
    """Plot top feature dependence metrics (distance corr + MI)."""
    labels = [r["feature_name"] for r in rows]
    dcor = [r["distance_corr"] for r in rows]
    mi = [r["mutual_info"] for r in rows]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.42

    plt.figure(figsize=(max(10, 0.55 * len(labels)), 5.5))
    plt.bar(x - width / 2, dcor, width=width, label="distance_corr")
    plt.bar(x + width / 2, mi, width=width, label="mutual_info")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Per-feature nonlinear dependence with label")
    plt.grid(axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_permutation_importance(rows, out_png):
    """Plot permutation importances with error bars."""
    labels = [r["feature_name"] for r in rows]
    means = [r["importance_mean"] for r in rows]
    stds = [r["importance_std"] for r in rows]

    x = np.arange(len(labels), dtype=np.float64)
    plt.figure(figsize=(max(10, 0.55 * len(labels)), 5.5))
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Importance (increase in MAE when permuted)")
    plt.title("Permutation feature importance")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def maybe_compute_shap_interactions(
    models,
    model_name,
    X,
    feature_cols,
    out_dir,
    max_samples,
    verbose=True,
    heartbeat_seconds=20.0,
):
    """Optionally compute SHAP interaction strength matrix for tree models."""
    if model_name != "xgboost":
        return None, None

    try:
        import shap
    except Exception:
        return None, None

    n = min(int(max_samples), X.shape[0])
    if n < 2:
        return None, None

    X_sample = X[:n]
    model = clone(models[model_name])
    # Fit on full data for interaction diagnostics.
    if verbose:
        print(f"[SHAP] Fitting xgboost for interaction analysis on {X.shape[0]} rows ...")
    _, _ = run_with_heartbeat(
        fn=lambda: model.fit(X, y_global_ref["y"]),
        label="SHAP prep model fit",
        heartbeat_seconds=heartbeat_seconds,
        enabled=verbose,
        tracker=None,
    )

    if verbose:
        print(f"[SHAP] Computing interaction values for {n} samples (this can take time) ...")
    explainer = shap.TreeExplainer(model)
    inter, _ = run_with_heartbeat(
        fn=lambda: explainer.shap_interaction_values(X_sample),
        label=f"SHAP interaction values ({n} samples)",
        heartbeat_seconds=heartbeat_seconds,
        enabled=verbose,
        tracker=None,
    )
    inter = np.asarray(inter, dtype=np.float64)

    # Mean absolute interaction matrix.
    M = np.mean(np.abs(inter), axis=0)

    csv_path = os.path.join(out_dir, "shap_interaction_strength.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name"] + list(feature_cols))
        for i, fi in enumerate(feature_cols):
            writer.writerow([fi] + [f"{float(M[i, j]):.12f}" for j in range(len(feature_cols))])

    png_path = os.path.join(out_dir, "shap_interaction_heatmap.png")
    plt.figure(figsize=(8, 7))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="mean |SHAP interaction|")
    plt.xticks(np.arange(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(np.arange(len(feature_cols)), feature_cols)
    plt.title("SHAP interaction strength (sample)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close()

    return csv_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run linear/nonlinear feature-signal diagnostics versus label: "
            "CV predictive test, dummy/shuffled controls, dependence metrics, and permutation importance."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional path to alpha_analysis.csv. Defaults to current analysis output folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <analysis>/feature_signal_diagnostics.",
    )
    parser.add_argument(
        "--label-column",
        default="soft_label",
        help="Label column in input CSV (default: soft_label).",
    )
    parser.add_argument(
        "--cv-mode",
        choices=["grouped", "random"],
        default="grouped",
        help="Cross-validation mode: grouped by dataset (safer) or random.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repeats for random CV mode (ignored in grouped mode).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--perm-repeats",
        type=int,
        default=10,
        help="Permutation repeats per fold for importance estimation.",
    )
    parser.add_argument(
        "--compute-shap-interactions",
        action="store_true",
        help="Compute optional SHAP interaction matrix for xgboost model.",
    )
    parser.add_argument(
        "--max-shap-samples",
        type=int,
        default=1000,
        help="Max samples used for SHAP interaction computation.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=["seconds", "fast", "full"],
        default="seconds",
        help=(
            "Execution profile. 'seconds' is optimized for very fast turnaround; "
            "'full' keeps the original exhaustive diagnostics."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce progress output (final summary is still printed).",
    )
    parser.add_argument(
        "--fit-heartbeat-seconds",
        type=float,
        default=None,
        help="Seconds between heartbeat logs while one long fit/scoring step is running (<=0 disables).",
    )
    args = parser.parse_args()

    verbose = not bool(args.quiet)
    global_start = perf_counter()
    speed_profile = build_speed_profile(args.speed_profile)

    if args.fit_heartbeat_seconds is None:
        heartbeat_seconds = float(speed_profile["heartbeat_default_seconds"])
    else:
        heartbeat_seconds = float(args.fit_heartbeat_seconds)

    u.CONFIG_PATH = args.config
    cfg = load_config()

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    results_root = get_config_path(cfg, "results_folder", "data/results")
    analysis_dir = os.path.join(results_root, short_model, "analysis")

    input_csv = args.input_csv or os.path.join(analysis_dir, "alpha_analysis.csv")
    out_dir = args.output_dir or os.path.join(analysis_dir, "feature_signal_diagnostics")
    ensure_dir(out_dir)

    if verbose:
        print("[1/7] Loading input table ...")

    rows, fieldnames = load_table(input_csv)
    feature_cols = infer_feature_columns(fieldnames, args.label_column)
    X, y, groups = rows_to_arrays(rows, feature_cols, args.label_column)
    X, y, groups, selected_idx = subsample_rows_by_group(
        X,
        y,
        groups,
        max_rows=speed_profile["max_rows"],
        seed=args.seed,
    )

    if verbose:
        print(f"  rows={X.shape[0]}, features={X.shape[1]}, label={args.label_column}")
        if speed_profile["max_rows"] is not None:
            print(
                f"  speed_profile={speed_profile['name']} (sampled {X.shape[0]} rows from {len(rows)} total)"
            )
        else:
            print(f"  speed_profile={speed_profile['name']} (full dataset)")
        print("[2/7] Building CV splits ...")

    n_splits_effective = int(args.n_splits)
    if speed_profile.get("max_cv_splits") is not None:
        n_splits_effective = min(n_splits_effective, int(speed_profile["max_cv_splits"]))

    splits = generate_splits(
        cv_mode=args.cv_mode,
        X=X,
        y=y,
        groups=groups,
        n_splits=n_splits_effective,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )

    if verbose:
        print(f"  splits={len(splits)} ({args.cv_mode} mode)")
        print("[3/7] Running CV predictive tests (models + controls) ...")
        print("  Note: xgboost fits are typically the slowest steps in this stage.")

    models = make_models(cfg, args.seed, speed_profile)
    models = {k: v for k, v in models.items() if k in set(speed_profile["models_to_run"]) }
    shuffled_models = [m for m in speed_profile["shuffled_models"] if m in models]
    cv_t0 = perf_counter()

    cv_rows = evaluate_models_with_controls(
        models=models,
        X=X,
        y=y,
        splits=splits,
        seed=args.seed,
        show_progress=verbose,
        heartbeat_seconds=heartbeat_seconds,
        shuffled_models=shuffled_models,
    )
    if verbose:
        print(f"  CV stage finished in {format_seconds(perf_counter() - cv_t0)}")

    if verbose:
        print("[4/7] Summarizing and saving CV results ...")
    cv_summary = summarize_cv_rows(cv_rows)

    cv_detail_csv, cv_summary_csv = write_cv_outputs(cv_rows, cv_summary, out_dir)

    if verbose:
        print("[5/7] Computing feature dependence metrics (Pearson, dCor, MI) ...")
    dep_t0 = perf_counter()
    dep_rows = compute_feature_dependence_scores(X, y, feature_cols, args.seed, show_progress=verbose)
    if verbose:
        print(f"  Dependence stage finished in {format_seconds(perf_counter() - dep_t0)}")

    dep_csv = os.path.join(out_dir, "feature_dependence_scores.csv")
    write_feature_dependence_csv(dep_rows, dep_csv)

    if verbose:
        print("[6/7] Computing permutation importance (xgboost) ...")
    perm_t0 = perf_counter()

    perm_repeats_effective = int(args.perm_repeats)
    if speed_profile.get("perm_repeats_cap") is not None:
        perm_repeats_effective = min(perm_repeats_effective, int(speed_profile["perm_repeats_cap"]))

    perm_splits = splits
    if speed_profile.get("perm_max_folds") is not None:
        perm_splits = splits[: int(speed_profile["perm_max_folds"])]

    perm_rows = aggregate_permutation_importance(
        models=models,
        model_name="xgboost",
        X=X,
        y=y,
        splits=perm_splits,
        feature_cols=feature_cols,
        seed=args.seed,
        n_repeats=perm_repeats_effective,
        show_progress=verbose,
        heartbeat_seconds=heartbeat_seconds,
    )
    if verbose:
        print(f"  Permutation stage finished in {format_seconds(perf_counter() - perm_t0)}")

    perm_csv = os.path.join(out_dir, "permutation_importance_xgboost.csv")
    write_permutation_csv(perm_rows, perm_csv)

    cv_plot = os.path.join(out_dir, "cv_model_vs_controls.png")
    dep_plot = os.path.join(out_dir, "feature_dependence_scores.png")
    perm_plot = os.path.join(out_dir, "permutation_importance_xgboost.png")

    plot_cv_summary(cv_summary, cv_plot)
    plot_feature_dependence(dep_rows, dep_plot)
    plot_permutation_importance(perm_rows, perm_plot)

    shap_csv = None
    shap_png = None
    if args.compute_shap_interactions:
        if verbose:
            print("[7/7] Computing optional SHAP interactions ...")
        shap_t0 = perf_counter()
        y_global_ref["y"] = y
        shap_samples_effective = int(args.max_shap_samples)
        if speed_profile.get("shap_max_samples_cap") is not None:
            shap_samples_effective = min(
                shap_samples_effective,
                int(speed_profile["shap_max_samples_cap"]),
            )
        shap_csv, shap_png = maybe_compute_shap_interactions(
            models=models,
            model_name="xgboost",
            X=X,
            feature_cols=feature_cols,
            out_dir=out_dir,
            max_samples=shap_samples_effective,
            verbose=verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if verbose:
            print(f"  SHAP interaction stage finished in {format_seconds(perf_counter() - shap_t0)}")

    if verbose:
        print(f"Total runtime: {format_seconds(perf_counter() - global_start)}")

    print("Feature-signal diagnostics completed.")
    print(f"Input CSV             : {input_csv}")
    print(f"Rows                  : {X.shape[0]}")
    print(f"Features              : {X.shape[1]}")
    print(f"Label column          : {args.label_column}")
    print(f"CV mode               : {args.cv_mode}")
    print(f"Folds evaluated       : {len(splits)}")
    print(f"Output directory      : {out_dir}")
    print(f"- {cv_detail_csv}")
    print(f"- {cv_summary_csv}")
    print(f"- {dep_csv}")
    print(f"- {perm_csv}")
    print(f"- {cv_plot}")
    print(f"- {dep_plot}")
    print(f"- {perm_plot}")
    if args.compute_shap_interactions:
        if shap_csv and shap_png:
            print(f"- {shap_csv}")
            print(f"- {shap_png}")
        else:
            print("- SHAP interaction outputs skipped (shap unavailable or insufficient data).")


# Mutable reference used only for optional SHAP helper to avoid passing y repeatedly.
y_global_ref = {}


if __name__ == "__main__":
    main()
