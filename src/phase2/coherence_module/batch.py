from __future__ import annotations

import json, os, pickle, time
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io_utils import load_config, save_parquet, apply_node_limit
from src.utils.parsers import series_to_ns, timestamp_to_ns
from src.phase2.profiling_and_clustering import (
    BASE_FEATURES,
    H100_FEATURES,
    COMPONENT_FEATURES,
)


# Load a component's saved StandardScaler.
def load_scaler(cluster_dir: Path, component: str):
    path = cluster_dir / f"{component}_scaler.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# Load a component's saved cluster feature list.
def load_feature_list(cluster_dir: Path, component: str) -> list:
    path = cluster_dir / f"{component}_features.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# Convert a profile row to a feature vector, filling gaps with medians.
def profile_to_vector(
    row: pd.Series, feature_cols: list, medians: dict
) -> Optional[np.ndarray]:
    vec = []
    all_nan = True
    for col in feature_cols:
        if col in row.index:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                val = medians.get(col, 0.0)
            else:
                all_nan = False
            vec.append(float(val))
        else:
            vec.append(float(medians.get(col, 0.0)))
    if all_nan:
        return None
    return np.array(vec, dtype=np.float32)


# Return the top-k features deviating most from the peer median, as JSON.
def top_deviant_features(
    node_vec: np.ndarray, median_vec: np.ndarray, feature_cols: list, top_k: int = 5
) -> str:
    deltas = np.abs(node_vec - median_vec)
    ranked = sorted(
        zip(feature_cols, node_vec.tolist(), median_vec.tolist(), deltas.tolist()),
        key=lambda x: x[3],
        reverse=True,
    )[:top_k]
    result = [
        {
            "feature": feat,
            "node_val": round(float(nv), 4),
            "peer_median": round(float(mv), 4),
            "abs_delta": round(float(d), 4),
        }
        for feat, nv, mv, d in ranked
        if d > 1e-6
    ]
    return json.dumps(result)


# Score each node in a multi-node job by its distance from the peer-median profile.
def coherence_for_job(
    job_profiles: pd.DataFrame,
    feature_cols: list,
    scaler,
    medians: dict,
    z_thresh: float,
) -> list:
    hostnames = job_profiles["hostname"].tolist()
    vectors = []
    valid_idx = []

    for i, (_, row) in enumerate(job_profiles.iterrows()):
        vec = profile_to_vector(row, feature_cols, medians)
        if vec is not None:
            vectors.append(vec)
            valid_idx.append(i)

    n_valid = len(vectors)
    n_peers = len(hostnames)

    records = []

    if n_valid < 2:
        for i, (_, row) in enumerate(job_profiles.iterrows()):
            records.append(
                {
                    "hostname": row["hostname"],
                    "cluster_id": int(row.get("cluster_id", -1)),
                    "n_peer_nodes": n_peers,
                    "n_valid_profiles": n_valid,
                    "coherence_score": np.nan,
                    "peer_median_dist": np.nan,
                    "peer_mad_dist": np.nan,
                    "coherence_z": np.nan,
                    "is_coherence_anomaly": False,
                    "top_deviant_features_json": "[]",
                    "split": str(row.get("split", "unknown")),
                }
            )
        return records

    X = np.vstack(vectors)
    if scaler is not None:
        X = scaler.transform(X)

    median_vec = np.median(X, axis=0)
    dists = np.linalg.norm(X - median_vec, axis=1)
    med_dist = float(np.median(dists))
    mad_dist = float(np.median(np.abs(dists - med_dist)))
    scale = 1.4826 * mad_dist

    valid_set = set(valid_idx)
    vi = 0
    for i, (_, row) in enumerate(job_profiles.iterrows()):
        if i in valid_set:
            dist = float(dists[vi])
            z = (dist - med_dist) / scale if scale > 1e-6 else 0.0
            max_dist = float(dists.max())
            cs = dist / max_dist if max_dist > 1e-6 else 0.0
            top_feats_json = top_deviant_features(X[vi], median_vec, feature_cols)
            records.append(
                {
                    "hostname": row["hostname"],
                    "cluster_id": int(row.get("cluster_id", -1)),
                    "n_peer_nodes": n_peers,
                    "n_valid_profiles": n_valid,
                    "coherence_score": float(cs),
                    "peer_median_dist": med_dist,
                    "peer_mad_dist": float(mad_dist),
                    "coherence_z": float(z),
                    "is_coherence_anomaly": bool(z > z_thresh),
                    "top_deviant_features_json": top_feats_json,
                    "split": str(row.get("split", "unknown")),
                }
            )
            vi += 1
        else:
            records.append(
                {
                    "hostname": row["hostname"],
                    "cluster_id": int(row.get("cluster_id", -1)),
                    "n_peer_nodes": n_peers,
                    "n_valid_profiles": n_valid,
                    "coherence_score": np.nan,
                    "peer_median_dist": med_dist,
                    "peer_mad_dist": float(mad_dist),
                    "coherence_z": np.nan,
                    "is_coherence_anomaly": False,
                    "top_deviant_features_json": "[]",
                    "split": str(row.get("split", "unknown")),
                }
            )
    return records


PM_MIN_Z = 3.5
PM_MIN_RUN = 5
PM_MIN_PEERS = 2

GLOBAL_MASTER_CACHE: dict = {}
GLOBAL_ART_CACHE: dict = {}
GLOBAL_PARAMS: dict = {}


# Load per-minute max system power for every node across components.
def load_per_minute_power(master_dir: Path) -> pd.DataFrame:
    frames = []
    for comp in ("zen4", "h100"):
        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue
        for p in apply_node_limit(sorted(comp_dir.glob("*.parquet"))):
            hostname = p.stem
            try:
                df = pd.read_parquet(p, engine="pyarrow")
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue
            pcols = [
                c
                for c in df.columns
                if c.endswith("_avg")
                and (
                    "systeminputpower" in c.lower()
                    or "systempowerconsumption" in c.lower()
                )
            ]
            if not pcols:
                continue
            pwr = df[pcols].apply(pd.to_numeric, errors="coerce").max(axis=1)
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(df["timestamp"], utc=True).dt.floor(
                            "1min"
                        ),
                        "hostname": hostname,
                        "power_w": pwr.astype("float32"),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=["timestamp", "hostname", "power_w"])
    return pd.concat(frames, ignore_index=True)
