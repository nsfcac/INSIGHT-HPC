from __future__ import annotations

from src.phase1.isolation_forest_module.constants import *
import gc, io, json, os, pickle, time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

from src.utils.io_utils import (
    load_config,
    load_parquet,
    save_parquet,
    apply_node_limit,
    phase1_workers,
)
from src.utils.maintenance import load_maintenance_windows, apply_maintenance_mask


# Read only the support and feature columns needed for IF scoring.
def read_feature_subset(path: Path, feature_cols: list) -> Optional[pd.DataFrame]:
    try:
        schema_cols = set(pq.read_schema(path).names)
        support_cols = [
            "timestamp",
            "hostname",
            "split",
            "audit_any",
            "maintenance_flag",
            "is_running_job",
            "active_job_count",
        ]
        read_cols = [c for c in support_cols if c in schema_cols]
        read_cols += [
            c for c in feature_cols if c in schema_cols and c not in read_cols
        ]
        if "timestamp" not in read_cols:
            return None
        return pd.read_parquet(path, engine="pyarrow", columns=read_cols)
    except Exception:
        return None


# Pick feature columns by suffix, dropping excluded keywords and adding explicit ones.
def select_features(columns: list) -> list:
    feats = []
    for c in columns:
        if not any(c.endswith(sfx) for sfx in FEATURE_SUFFIXES):
            continue
        if any(kw in c.lower() for kw in EXCLUDE_KEYWORDS):
            continue
        feats.append(c)
    for c in EXPLICIT_FEATURES:
        if c in columns and c not in feats:
            feats.append(c)
    return feats


# Stack clean train rows across a component's nodes into one feature matrix.
def load_train_matrix(
    feat_dir: Path,
    component: str,
    feature_cols: list,
    windows: list,
    only_under_load: bool,
) -> Optional[np.ndarray]:
    comp_dir = feat_dir / component
    if not comp_dir.exists():
        return None

    matrices = []
    for p in apply_node_limit(sorted(comp_dir.glob("*.parquet"))):
        df = read_feature_subset(p, feature_cols)
        if df is None or df.empty:
            continue

        df = apply_maintenance_mask(df, windows, "timestamp", "hostname")

        mask = pd.Series(True, index=df.index)
        if "split" in df.columns:
            mask &= df["split"] == "train"
        if "audit_any" in df.columns:
            mask &= ~df["audit_any"].fillna(False)
        if "maintenance_flag" in df.columns:
            mask &= ~df["maintenance_flag"].fillna(False)
        try:
            from src.utils.gt_mask import in_gt_window

            gt = in_gt_window(df)
            if gt.any():
                mask &= ~gt
        except Exception:
            pass

        if only_under_load:
            load_col = "is_running_job"
            if load_col not in df.columns and "active_job_count" in df.columns:
                df["is_running_job"] = (
                    df["active_job_count"].fillna(0).astype(float) > 0
                ).astype("float32")
                load_col = "is_running_job"
            if load_col in df.columns:
                mask &= df[load_col].fillna(0).astype(float) > 0

        train_df = df[mask]
        if len(train_df) < 10:
            del df
            gc.collect()
            continue

        existing = [c for c in feature_cols if c in train_df.columns]
        if not existing:
            del df
            gc.collect()
            continue

        mat = np.zeros((len(train_df), len(feature_cols)), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            if col in train_df.columns:
                v = pd.to_numeric(train_df[col], errors="coerce").values
                mat[:, i] = np.nan_to_num(v.astype(np.float32), nan=0.0)

        matrices.append(mat)
        del df, train_df, mat
        gc.collect()

    if not matrices:
        return None
    return np.vstack(matrices)


# Score one node with the IF model and attach its top contributing features.
def score_node(
    df: pd.DataFrame,
    hostname: str,
    component: str,
    feature_cols: list,
    model: IsolationForest,
    windows: list,
    norm_stats: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    ts_col = "timestamp"
    if ts_col not in df.columns:
        return None

    df = df.sort_values(ts_col).copy()
    df = apply_maintenance_mask(df, windows, ts_col, "hostname")

    mat = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").values
            mat[:, i] = np.nan_to_num(v.astype(np.float32), nan=0.0)

    raw_scores = model.decision_function(mat)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm_scores = (
        1.0 - (raw_scores - s_min) / (s_max - s_min)
        if s_max > s_min
        else np.full_like(raw_scores, 0.5)
    )

    is_anomaly = raw_scores < 0

    is_running_job = np.nan
    if "is_running_job" in df.columns:
        is_running_job = df["is_running_job"].values
    elif "active_job_count" in df.columns:
        is_running_job = (
            (df["active_job_count"].fillna(0).astype(float) > 0)
            .astype("float32")
            .values
        )

    tf1, tf2, tf3, tz1, tz2, tz3 = top_features(
        mat, is_anomaly, feature_cols, norm_stats, hostname, component
    )

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], utc=True),
            "hostname": hostname,
            "component": component,
            "split": df["split"].values if "split" in df.columns else "unknown",
            "is_running_job": is_running_job,
            "maintenance_flag": df["maintenance_flag"].values,
            "if_anomaly_score": norm_scores.astype("float32"),
            "if_is_anomaly": is_anomaly,
            "top_feature_1": tf1,
            "top_feature_2": tf2,
            "top_feature_3": tf3,
            "top_feature_z_1": tz1.astype("float32"),
            "top_feature_z_2": tz2.astype("float32"),
            "top_feature_z_3": tz3.astype("float32"),
        }
    )


# Collect the union of usable feature columns across a component's nodes.
def discover_features(feat_dir: Path, component: str) -> list:
    comp_dir = feat_dir / component
    if not comp_dir.exists():
        return []

    all_cols: set = set()
    for p in apply_node_limit(sorted(comp_dir.glob("*.parquet"))):
        try:
            all_cols.update(select_features(list(pq.read_schema(p).names)))
        except Exception:
            pass
    return sorted(all_cols)


# Load the per-feature normalization stats parquet, searching the fallback path.
def load_norm_stats(feat_dir: Path) -> Optional[pd.DataFrame]:
    p = feat_dir / "norm_stats.parquet"
    if not p.exists():
        p = feat_dir.parent / "features" / "norm_stats.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p, engine="pyarrow")


# Find the top-k most-deviating features per anomalous row, as z-scores.
def top_features(
    mat: np.ndarray,
    is_anomaly: np.ndarray,
    feature_cols: list,
    norm_stats: Optional[pd.DataFrame],
    hostname: str,
    component: str,
    top_k: int = 3,
) -> tuple:
    T, F = mat.shape
    top_names = [np.full(T, "", dtype=object) for _ in range(top_k)]
    top_zs = [np.zeros(T, dtype=np.float32) for _ in range(top_k)]

    anom_idx = np.where(is_anomaly)[0]
    if len(anom_idx) == 0 or F == 0:
        return tuple(top_names + top_zs)

    STD_FLOOR = 1e-3
    Z_CLIP = 50.0
    if norm_stats is not None:
        ns = norm_stats[
            (norm_stats["hostname"] == hostname)
            & (norm_stats["component"] == component)
        ].set_index("feature")
        means = np.array(
            [float(ns.loc[c, "mean"]) if c in ns.index else 0.0 for c in feature_cols],
            dtype=np.float32,
        )
        stds = np.array(
            [float(ns.loc[c, "std"]) if c in ns.index else 1.0 for c in feature_cols],
            dtype=np.float32,
        )
        stds = np.where(stds < STD_FLOOR, STD_FLOOR, stds)
    else:
        means = mat[anom_idx].mean(axis=0)
        stds = mat[anom_idx].std(axis=0)
        stds = np.where(stds < STD_FLOOR, STD_FLOOR, stds)

    feat_arr = np.array(feature_cols)

    for i in anom_idx:
        devs = np.abs(mat[i] - means) / stds
        np.clip(devs, 0.0, Z_CLIP, out=devs)
        order = np.argsort(devs)[::-1]
        for k in range(top_k):
            if k < len(order):
                top_names[k][i] = feat_arr[order[k]]
                top_zs[k][i] = float(devs[order[k]])

    return tuple(top_names + top_zs)


# Compute each node's anomaly score relative to the rolling cluster median.
def add_cluster_relative_scores(
    scores_dir: Path, rolling_window_min: int = 60, total_rows: int = 1
) -> None:
    score_paths = sorted(scores_dir.glob("*.parquet"))
    if not score_paths:
        return

    STRIDE = 10
    frames = []
    for p in score_paths:
        try:
            df = pd.read_parquet(
                p, engine="pyarrow", columns=["timestamp", "if_anomaly_score"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").iloc[::STRIDE].reset_index(drop=True)
            df["_node"] = p.stem
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return

    all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp")

    cluster_med = (
        all_df.groupby("timestamp")["if_anomaly_score"].median().rename("cluster_med")
    )
    cluster_med_roll = (
        cluster_med.rolling(f"{rolling_window_min}min", min_periods=5)
        .median()
        .fillna(cluster_med.expanding().median())
    )
    del all_df, frames
    gc.collect()

    IF_WORKER_CONTEXT.clear()
    IF_WORKER_CONTEXT["cluster_med_roll"] = cluster_med_roll
    workers = phase1_workers()

    if workers == 1 or len(score_paths) <= 1:
        results = [cluster_rel_one(str(p)) for p in score_paths]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(score_paths))) as ex:
            results = list(ex.map(cluster_rel_one, [str(p) for p in score_paths]))
    IF_WORKER_CONTEXT.clear()

    updated = rel_anom = 0
    for stem, n_rel, warn in sorted(results, key=lambda r: r[0]):
        if warn:
            print(warn, flush=True)
            continue
        rel_anom += n_rel
        updated += 1

    if updated:
        print(
            f"  Cluster-relative scores added to {updated} node files  "
            f"relative_anomaly_total={rel_anom:,}  "
            f"({100*rel_anom/max(total_rows,1):.2f}%)"
        )


# Read-only context shared with scoring workers via fork inheritance.
IF_WORKER_CONTEXT: dict = {}


# Score one node in a worker process and write its scores parquet.
def score_one_node(
    comp: str, hostname: str, parquet_str: str, score_str: str, force: bool
):
    ctx = IF_WORKER_CONTEXT
    score_path = Path(score_str)
    if score_path.exists() and not force:
        return hostname, f"    skip  {hostname}", 0, 0

    clf = ctx["models"][comp]["clf"]
    feature_cols = ctx["models"][comp]["feature_cols"]
    try:
        clf.n_jobs = 1
    except Exception:
        pass

    buf = io.StringIO()
    with redirect_stdout(buf):
        df = read_feature_subset(Path(parquet_str), feature_cols)
        if df is None or df.empty:
            return hostname, None, 0, 0
        scored = score_node(
            df,
            hostname,
            comp,
            feature_cols,
            clf,
            ctx["windows"],
            norm_stats=ctx["norm_stats"],
        )
        if scored is None:
            return hostname, None, 0, 0
        n_rows = len(scored)
        n_anomaly = int(scored["if_is_anomaly"].sum())
        save_parquet(scored, score_path)
        p95 = scored["if_anomaly_score"].quantile(0.95)
        del df, scored
        gc.collect()
    summary = (
        f"    {hostname:20s}: {n_rows:>8,} rows  "
        f"anomaly={n_anomaly:>6,} ({100*n_anomaly/max(n_rows,1):5.2f}%)  "
        f"score_p95={p95:.3f}"
    )
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    return hostname, log, n_rows, n_anomaly


# Add cluster-relative anomaly scores to one node's scores file.
def cluster_rel_one(score_str: str):
    cluster_med_roll = IF_WORKER_CONTEXT["cluster_med_roll"]
    p = Path(score_str)
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        ts_idx = df["timestamp"]
        cmed = cluster_med_roll.reindex(
            ts_idx, method="nearest", tolerance=pd.Timedelta("2min")
        )
        cmed = cmed.fillna(cluster_med_roll.median()).values.astype(np.float32)

        df["if_anomaly_score_rel"] = (
            df["if_anomaly_score"].values.astype(np.float32) - cmed
        )
        df["if_is_anomaly_rel"] = df["if_anomaly_score_rel"] > 0.20

        save_parquet(df, p)
        n_rel = int(df["if_is_anomaly_rel"].sum())
        del df
        return p.stem, n_rel, None
    except Exception as exc:
        return p.stem, 0, f"  [WARN] cluster-rel score failed for {p.stem}: {exc}"
