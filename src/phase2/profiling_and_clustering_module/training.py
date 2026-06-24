from __future__ import annotations

from src.phase2.profiling_and_clustering_module.constants import *
import gc, json, os, pickle, time, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit
from src.utils.maintenance import load_maintenance_windows
from src.utils.rack_topology import build_rack_map_from_paths

from src.phase2.profiling_and_clustering_module.profiles import *
from src.phase2.profiling_and_clustering_module.clustering_features import *


# Compute per-job coefficient-of-variation reproducibility across repeat runs.
def compute_reproducibility(profiles: pd.DataFrame) -> pd.DataFrame:
    cv_features = [
        "pwr_mean",
        "cpu_load_mean",
        "mem_usage_mean",
        "joules_per_job",
        "watts_per_cpu_pct",
        "gpu_util_mean",  # NaN for zen4
        "pdu_idrac_efficiency",  # NaN if PDU disabled
    ]

    records = []
    for (jid, comp), grp in profiles.groupby(
        ["job_id", "component"], dropna=True, sort=False
    ):
        if len(grp) < 2:
            continue
        cvs = []
        row = {"job_id": jid, "component": comp, "n_runs": len(grp)}
        for feat in cv_features:
            if feat not in grp.columns:
                row[f"{feat}_cv"] = np.nan
                continue
            s = pd.to_numeric(grp[feat], errors="coerce").dropna()
            if len(s) < 2 or s.mean() == 0 or np.isnan(s.mean()):
                row[f"{feat}_cv"] = np.nan
                continue
            cv = float(s.std() / abs(s.mean()))
            row[f"{feat}_cv"] = cv
            cvs.append(cv)
        if cvs:
            mean_cv = float(np.mean(cvs))
            row["profile_consistency_score"] = float(np.clip(1.0 - mean_cv, 0.0, 1.0))
        else:
            row["profile_consistency_score"] = np.nan
        records.append(row)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")
    result["is_reproducible"] = result["profile_consistency_score"].ge(0.5)

    float_cols = [c for c in result.columns if result[c].dtype in (float, np.float64)]
    for c in float_cols:
        result[c] = result[c].astype("float32")
    return result
