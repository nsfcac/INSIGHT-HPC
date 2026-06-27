from __future__ import annotations

from offline.phase2.profiling_and_clustering_module.constants import *
import gc, os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

from shared.utils.io_utils import apply_node_limit, load_config, save_parquet
from shared.utils.maintenance import load_maintenance_windows
from shared.utils.parsers import series_to_ns, timestamp_to_ns
from shared.utils.rack_topology import build_rack_map_from_paths

from offline.phase2.profiling_and_clustering_module.profiles import *

#   Per-cluster IF training + segment scoring + reproducibility

IF_FEATURE_SUFFIXES = ("_avg", "_rmean15", "_rstd15")
IF_EXCLUDE_KEYWORDS = [
    "audit_",
    "split",
    "hostname",
    "component",
    "timestamp",
    "primary_job",
    "jobs_json",
    "cpu_shares",
    "active_job",
    "node_busy",
    "node_idle",
    "primary_job_run",
    "primary_job_changed",
    "job_set_changed",
    "is_running",
    "is_multi",
    "pdu__rack_audit",
]

FEATURE_FRAME_CACHE: dict = {}
FEATURE_MATRIX_CACHE: dict = {}


# Return whether feature cache is enabled.
def feature_cache_enabled() -> bool:
    return os.environ.get("INSIGHT_HPC_PHASE2_FEATURE_CACHE", "1") != "0"


# Load and cache a node's feature columns, sorted by timestamp.
def load_feature_frame(
    feat_dir: Path, component: str, hostname: str, feature_cols: list
) -> Optional[pd.DataFrame]:
    p = feat_dir / component / f"{hostname}.parquet"
    if not p.exists():
        return None

    key = (str(p), tuple(feature_cols))
    if feature_cache_enabled() and key in FEATURE_FRAME_CACHE:
        return FEATURE_FRAME_CACHE[key]

    try:
        schema_cols = set(pq.read_schema(p).names)
        read_cols = ["timestamp"] + [c for c in feature_cols if c in schema_cols]
        read_cols = [c for c in read_cols if c in schema_cols]
        if "timestamp" not in read_cols:
            return None
        df = pd.read_parquet(p, engine="pyarrow", columns=read_cols)
    except Exception:
        return None

    if df.empty or "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if feature_cache_enabled():
        FEATURE_FRAME_CACHE[key] = df
    return df


# Load and cache a node's feature matrix with timestamps as int64 nanoseconds.
def load_feature_matrix(
    feat_dir: Path, component: str, hostname: str, feature_cols: list
):
    p = feat_dir / component / f"{hostname}.parquet"
    if not p.exists():
        return None

    key = (str(p), tuple(feature_cols))
    if feature_cache_enabled() and key in FEATURE_MATRIX_CACHE:
        return FEATURE_MATRIX_CACHE[key]

    df = load_feature_frame(feat_dir, component, hostname, feature_cols)
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    ts_ns = series_to_ns(df["timestamp"])
    mat = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            mat[:, i] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    payload = (ts_ns, mat)
    if feature_cache_enabled():
        FEATURE_MATRIX_CACHE[key] = payload
        FEATURE_FRAME_CACHE.pop(key, None)
    return payload


# Drop cached frames and matrices for a component to free memory.
def clear_feature_cache(feat_dir: Path, component: str) -> None:
    prefix = str(feat_dir / component)
    for store in (FEATURE_FRAME_CACHE, FEATURE_MATRIX_CACHE):
        for key in list(store):
            if key[0].startswith(prefix):
                del store[key]
    gc.collect()


# Pick IF feature columns by suffix, excluding context keywords.
def select_if_features(columns: list) -> list:
    feats = []
    for c in columns:
        if not any(c.endswith(sfx) for sfx in IF_FEATURE_SUFFIXES):
            continue
        if any(kw in c.lower() for kw in IF_EXCLUDE_KEYWORDS):
            continue
        feats.append(c)
    return feats


# Collect the union of usable IF feature columns across a component's nodes.
def discover_features(feat_dir: Path, component: str) -> list:
    comp_dir = feat_dir / component
    if not comp_dir.exists():
        return []
    all_cols: set = set()
    for p in apply_node_limit(sorted(comp_dir.glob("*.parquet"))):
        try:
            all_cols.update(select_if_features(list(pq.read_schema(p).names)))
        except Exception:
            pass
    return sorted(all_cols)


# Slice a node's cached feature matrix to one segment's time window.
def build_feature_matrix(
    feat_dir: Path,
    component: str,
    feature_cols: list,
    hostname: str,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
) -> Optional[np.ndarray]:
    payload = load_feature_matrix(feat_dir, component, hostname, feature_cols)
    if payload is None:
        return None
    ts_ns, mat_all = payload
    left = int(np.searchsorted(ts_ns, timestamp_to_ns(seg_start), side="left"))
    right = int(np.searchsorted(ts_ns, timestamp_to_ns(seg_end), side="right"))
    if right - left < 2:
        return None
    return mat_all[left:right]


# Stack feature rows across all of a cluster's training segments.
def load_cluster_train_matrix(
    feat_dir: Path,
    component: str,
    feature_cols: list,
    clustered_segs: pd.DataFrame,
    windows: list,
) -> Optional[np.ndarray]:
    matrices = []
    for _, seg in clustered_segs.iterrows():
        mat = build_feature_matrix(
            feat_dir,
            component,
            feature_cols,
            seg["hostname"],
            pd.Timestamp(seg["seg_start"]),
            pd.Timestamp(seg["seg_end"]),
        )
        if mat is not None:
            matrices.append(mat)
    return np.vstack(matrices) if matrices else None


# Return a segment's mean IF anomaly score and scored-row count.
def score_segment(
    feat_dir: Path,
    component: str,
    feature_cols: list,
    model: IsolationForest,
    hostname: str,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
) -> tuple:
    mat = build_feature_matrix(
        feat_dir, component, feature_cols, hostname, seg_start, seg_end
    )
    if mat is None or len(mat) == 0:
        return np.nan, 0

    raw = model.decision_function(mat)
    s_min, s_max = raw.min(), raw.max()
    if s_max > s_min:
        norm = 1.0 - (raw - s_min) / (s_max - s_min)
    else:
        norm = np.full_like(raw, 0.5)
    return float(norm.mean()), len(mat)
