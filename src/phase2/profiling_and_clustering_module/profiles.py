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


# JSON encoder that handles numpy scalars (float32, int16, etc.)
class NpEncoder(json.JSONEncoder):
    # Encode numpy scalars and arrays as JSON-native types.
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Return "_avg" columns matching all of the given keywords.
def avg(cols: list, *kws: str) -> list:
    kws_l = [k.lower() for k in kws]
    return [
        c for c in cols if c.endswith("_avg") and all(k in c.lower() for k in kws_l)
    ]


# Return "_avg" columns matching any of the given keywords.
def any_col(cols: list, *kws: str) -> list:
    kws_l = [k.lower() for k in kws]
    return [
        c for c in cols if c.endswith("_avg") and any(k in c.lower() for k in kws_l)
    ]


# Return the first "_avg" column matching all (then any) of the keywords.
def first(cols: list, *kws: str) -> Optional[str]:
    found = avg(cols, *kws) or any_col(cols, *kws)
    return found[0] if found else None


# Return a single aggregate (mean/max/min/std) of a numeric series.
def stat(series: pd.Series, agg: str) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    fn = {"mean": s.mean, "max": s.max, "min": s.min, "std": s.std}
    return float(fn[agg]())


class PduCache:
    def __init__(self, master_dir: Path, rack_map):
        self.pdu_dir = master_dir / "infra" / "pdu"
        self.rack_map = rack_map
        self.cache: dict = {}

    # Return the cached per-minute summed PDU power for a rack.
    def get_rack_power(self, rack_id: int) -> Optional[pd.Series]:
        if rack_id in self.cache:
            return self.cache[rack_id]

        pdu_units = self.rack_map.rack_to_pdus.get(rack_id, [])
        if not pdu_units or not self.pdu_dir.exists():
            self.cache[rack_id] = None
            return None

        frames = []
        for uid in pdu_units:
            p = self.pdu_dir / f"{uid}.parquet"
            if not p.exists():
                continue
            df = load_parquet(p)
            if df is None or df.empty or TS not in df.columns:
                continue
            pdu_cols = [
                c for c in df.columns if c.endswith("_avg") and "pdu" in c.lower()
            ]
            if not pdu_cols:
                continue
            tmp = pd.DataFrame(
                {
                    "ts": pd.to_datetime(df[TS], utc=True),
                    "w": df[pdu_cols].apply(pd.to_numeric, errors="coerce").max(axis=1),
                }
            ).dropna()
            frames.append(tmp)

        if not frames:
            self.cache[rack_id] = None
            return None

        combined = pd.concat(frames, ignore_index=True)
        series = combined.set_index("ts")["w"].resample("1min").sum(min_count=1)
        self.cache[rack_id] = series
        return series


# Compute power/thermal/GPU/PDU profile features for one job segment.
def profile_segment(
    seg: pd.Series,
    master_df: pd.DataFrame,
    component: str,
    pdu_cache: PduCache,
    rack_id: Optional[int],
) -> dict:
    start, end = seg["seg_start"], seg["seg_end"]
    mask = (master_df[TS] >= start) & (master_df[TS] <= end)
    grp = master_df[mask]

    if len(grp) < 2:
        return {"profile_valid": False}

    cols = list(grp.columns)
    rec: dict = {"profile_valid": True}

    # Power
    pwr_col = first(cols, "systeminputpower") or first(cols, "systempowerconsumption")
    if pwr_col:
        s = pd.to_numeric(grp[pwr_col], errors="coerce").dropna()
        rec["pwr_mean"] = float(s.mean())
        rec["pwr_max"] = float(s.max())
        rec["pwr_std"] = float(s.std()) if len(s) > 1 else np.nan
        diffs = s.diff().dropna()
        rec["power_ramp_rate"] = float(diffs.std()) if len(diffs) > 1 else np.nan
        duration_s = float((end - start).total_seconds()) + 60.0
        rec["joules_per_job"] = float(s.mean()) * duration_s
    else:
        rec.update(
            pwr_mean=np.nan,
            pwr_max=np.nan,
            pwr_std=np.nan,
            power_ramp_rate=np.nan,
            joules_per_job=np.nan,
        )

    # CPU
    cpu_pwr_col = first(cols, "totalcpupower")
    cpu_load_col = "slurm_cpu_load" if "slurm_cpu_load" in cols else None

    rec["cpu_pwr_mean"] = stat(grp[cpu_pwr_col], "mean") if cpu_pwr_col else np.nan
    rec["cpu_load_mean"] = stat(grp[cpu_load_col], "mean") if cpu_load_col else np.nan

    # Watts per CPU% — efficiency proxy
    if not np.isnan(rec.get("pwr_mean", np.nan)) and not np.isnan(
        rec.get("cpu_load_mean", np.nan)
    ):
        cpu_pct = rec["cpu_load_mean"] / 100.0
        rec["watts_per_cpu_pct"] = (
            (rec["pwr_mean"] / cpu_pct) if cpu_pct > 0.01 else np.nan
        )
    else:
        rec["watts_per_cpu_pct"] = np.nan

    # Memory
    mem_pwr_col = first(cols, "totalmemorypower")
    mem_usage_col = "slurm_memory_usage" if "slurm_memory_usage" in cols else None

    rec["mem_pwr_mean"] = stat(grp[mem_pwr_col], "mean") if mem_pwr_col else np.nan
    rec["mem_usage_mean"] = (
        stat(grp[mem_usage_col], "mean") if mem_usage_col else np.nan
    )

    # Thermal — multi-channel families averaged across channels first, then time.
    inlet_cols = avg(cols, "temperaturereading", "inlet") or [
        c for c in any_col(cols, "temperaturereading") if "inlet" in c.lower()
    ]
    exhaust_cols = [
        c for c in any_col(cols, "temperaturereading") if "exhaust" in c.lower()
    ]
    fan_cols = any_col(cols, "rpmreading", "fanspeed")

    inlet_s = (
        grp[inlet_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        if inlet_cols
        else pd.Series(dtype=float)
    )
    exhaust_s = (
        grp[exhaust_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        if exhaust_cols
        else pd.Series(dtype=float)
    )
    fan_s = (
        grp[fan_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        if fan_cols
        else pd.Series(dtype=float)
    )

    rec["inlet_temp_mean"] = float(inlet_s.mean()) if len(inlet_s) > 0 else np.nan
    rec["inlet_temp_max"] = float(inlet_s.max()) if len(inlet_s) > 0 else np.nan
    rec["exhaust_temp_mean"] = float(exhaust_s.mean()) if len(exhaust_s) > 0 else np.nan
    rec["fan_rpm_mean"] = float(fan_s.mean()) if len(fan_s) > 0 else np.nan

    if (
        not np.isnan(rec.get("pwr_mean", np.nan))
        and rec["pwr_mean"] > 0
        and not np.isnan(rec.get("inlet_temp_max", np.nan))
    ):
        rec["max_temp_per_watt"] = rec["inlet_temp_max"] / rec["pwr_mean"]
    else:
        rec["max_temp_per_watt"] = np.nan

    # GPU (h100 only)
    if component == "h100":
        util_cols = any_col(cols, "gpuusage")
        gmem_cols = any_col(cols, "gpumemoryusage")
        gpwr_cols = [
            c
            for c in cols
            if c.endswith("_avg")
            and "powerconsumption" in c.lower()
            and "systempowerconsumption" not in c.lower()
        ]
        # iDRAC reports GPU temps via temperaturereading + gputempNN sensors.
        gtemp_cols = [
            c
            for c in cols
            if c.endswith("_avg")
            and "temperaturereading" in c.lower()
            and "gputemp" in c.lower()
        ]

        gpu_util = (
            grp[util_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if util_cols
            else pd.Series(dtype=float)
        )
        gpu_mem = (
            grp[gmem_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if gmem_cols
            else pd.Series(dtype=float)
        )
        gpu_pwr = (
            grp[gpwr_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if gpwr_cols
            else pd.Series(dtype=float)
        )
        gpu_temp = (
            grp[gtemp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if gtemp_cols
            else pd.Series(dtype=float)
        )

        rec["gpu_util_mean"] = float(gpu_util.mean()) if len(gpu_util) > 0 else np.nan
        rec["gpu_util_max"] = float(gpu_util.max()) if len(gpu_util) > 0 else np.nan
        rec["gpu_mem_mean"] = float(gpu_mem.mean()) if len(gpu_mem) > 0 else np.nan
        rec["gpu_pwr_mean"] = float(gpu_pwr.mean()) if len(gpu_pwr) > 0 else np.nan
        rec["gpu_pwr_max"] = float(gpu_pwr.max()) if len(gpu_pwr) > 0 else np.nan
        rec["gpu_temp_mean"] = float(gpu_temp.mean()) if len(gpu_temp) > 0 else np.nan
        if len(gpu_pwr) > 0 and not np.isnan(rec["joules_per_job"]):
            duration_s = float((end - start).total_seconds()) + 60.0
            rec["gpu_joules"] = float(gpu_pwr.mean()) * duration_s
        else:
            rec["gpu_joules"] = np.nan
    else:
        for k in [
            "gpu_util_mean",
            "gpu_util_max",
            "gpu_mem_mean",
            "gpu_pwr_mean",
            "gpu_pwr_max",
            "gpu_temp_mean",
            "gpu_joules",
        ]:
            rec[k] = np.nan

    # PDU (rack-level)
    if rack_id is not None:
        rack_pwr = pdu_cache.get_rack_power(rack_id)
        if rack_pwr is not None:
            window = rack_pwr[
                (rack_pwr.index >= start) & (rack_pwr.index <= end)
            ].dropna()
            rec["pdu_mean_w"] = float(window.mean()) if len(window) > 0 else np.nan
            rec["pdu_max_w"] = float(window.max()) if len(window) > 0 else np.nan
        else:
            rec["pdu_mean_w"] = rec["pdu_max_w"] = np.nan

        if (
            not np.isnan(rec.get("pwr_mean", np.nan))
            and rec["pwr_mean"] > 0
            and not np.isnan(rec.get("pdu_mean_w", np.nan))
        ):
            rec["pdu_idrac_efficiency"] = rec["pdu_mean_w"] / rec["pwr_mean"]
        else:
            rec["pdu_idrac_efficiency"] = np.nan
    else:
        rec["pdu_mean_w"] = rec["pdu_max_w"] = rec["pdu_idrac_efficiency"] = np.nan

    # Power presence is required for a valid profile.
    if np.isnan(rec.get("pwr_mean", np.nan)):
        rec["profile_valid"] = False

    return rec


# Pick component feature columns with enough non-constant data to cluster on.
def select_features(component: str, df: pd.DataFrame) -> list:
    candidates = COMPONENT_FEATURES.get(component, BASE_FEATURES)
    result = []
    for c in candidates:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) < 5 or vals.std() < 1e-6:
            continue
        result.append(c)
    return result


# Search candidate k values and return the best-silhouette K-means model.
def best_kmeans(
    X: np.ndarray, k_values: list, min_cluster_size: int, random_state: int = 42
) -> tuple:
    best_model = None
    best_k = k_values[0]
    best_sil = -1.0
    best_db = np.inf

    for k in k_values:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        counts = np.bincount(labels)
        if counts.min() < min_cluster_size:
            continue
        if len(np.unique(labels)) < 2:
            continue

        n_sample = min(5000, len(X))
        sil = silhouette_score(
            X, labels, sample_size=n_sample, random_state=random_state
        )
        db = davies_bouldin_score(X, labels)
        print(
            f"      k={k:2d}  sil={sil:.4f}  db={db:.4f}  "
            f"min_cluster={counts.min():3d}  max_cluster={counts.max():5d}"
        )

        if sil > best_sil:
            best_sil = sil
            best_db = db
            best_k = k
            best_model = km

    if best_model is None:
        k = k_values[0]
        best_model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        best_model.fit(X)
        best_k = k
        best_sil = 0.0
        best_db = np.inf
        print(f"      fallback k={k}")

    return best_model, best_k, best_sil, best_db


# Fit HDBSCAN and return labels with silhouette/DB/noise metrics.
def run_hdbscan(X: np.ndarray, min_cluster_size: int, random_state: int = 42) -> tuple:
    hdb = HDBSCAN(min_cluster_size=max(min_cluster_size, 3), min_samples=2, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        labels = hdb.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = float((labels == -1).mean())

    if n_clusters < 2:
        return labels, -1.0, np.inf, n_clusters, noise_frac

    mask = labels != -1
    if mask.sum() < 10:
        return labels, -1.0, np.inf, n_clusters, noise_frac

    n_sample = min(5000, mask.sum())
    idx = np.where(mask)[0]
    if len(idx) > n_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, n_sample, replace=False)

    sil = silhouette_score(X[idx], labels[idx], random_state=random_state)
    db = davies_bouldin_score(X[mask], labels[mask])
    return labels, sil, db, n_clusters, noise_frac


# Merge undersized clusters into their nearest centroid.
def merge_small(labels: np.ndarray, centroids: np.ndarray, min_size: int) -> np.ndarray:
    unique = np.unique(labels[labels >= 0])
    counts = {int(c): int((labels == c).sum()) for c in unique}
    small = [c for c, n in counts.items() if n < min_size]

    new_labels = labels.copy()
    for sid in small:
        dists = np.linalg.norm(centroids - centroids[sid], axis=1)
        dists[sid] = np.inf
        for alt in small:
            dists[alt] = np.inf
        nearest = int(np.argmin(dists))
        new_labels[new_labels == sid] = nearest
        print(f"      merged cluster {sid} (n={counts[sid]}) -> {nearest}")

    return new_labels


# Cluster one component's job profiles, persist the model artifacts, and return assignments.
def cluster_component(
    comp: str,
    comp_df: pd.DataFrame,
    comp_job_ids: pd.Series,
    failed_job_ids: set,
    cluster_dir: Path,
    k_values: list,
    min_cluster: int,
    random_state: int,
    use_hdbscan: bool,
) -> Optional[tuple]:
    feature_cols = select_features(comp, comp_df)
    if not feature_cols:
        print(f"  [WARN] No valid features for {comp}, skipping")
        return None
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    train_mask = (
        (comp_df["split"] == "train")
        & (~comp_df["is_multi_job"].fillna(False))
        & (~comp_job_ids.isin(failed_job_ids))
    )
    n_excluded = int(
        (
            (comp_df["split"] == "train")
            & (~comp_df["is_multi_job"].fillna(False))
            & comp_job_ids.isin(failed_job_ids)
        ).sum()
    )
    if n_excluded:
        print(f"  Excluded {n_excluded} failed/timeout jobs from train set")

    X_all_raw = comp_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = X_all_raw.median()
    X_all_raw = X_all_raw.fillna(medians).values.astype(np.float32)

    X_train_raw = (
        comp_df.loc[train_mask, feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .values.astype(np.float32)
    )

    if len(X_train_raw) < max(min_cluster * 2, 4):
        print(
            f"  [WARN] Only {len(X_train_raw)} training segments "
            f"(need >= {max(min_cluster*2, 4)}), skipping {comp}"
        )
        return None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_scaled = scaler.transform(X_all_raw)

    t0 = time.perf_counter()
    print(f"  K-means: searching k in {k_values} (n_train={len(X_train):,}) ...")
    km, best_k, km_sil, km_db = best_kmeans(
        X_train, k_values, min_cluster, random_state
    )
    print(
        f"  K-means best: k={best_k}  sil={km_sil:.4f}  db={km_db:.4f}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    chosen_algo = "kmeans"
    chosen_labels = km.labels_.copy()
    hdb_sil = hdb_db = None

    if use_hdbscan:
        print(f"  HDBSCAN: min_cluster_size={min_cluster} ...")
        hdb_labels, hdb_sil, hdb_db, hdb_k, noise_frac = run_hdbscan(
            X_train, min_cluster, random_state
        )
        print(
            f"  HDBSCAN: k={hdb_k}  sil={hdb_sil:.4f}  db={hdb_db:.4f}  "
            f"noise={noise_frac:.1%}"
        )

        if hdb_sil > km_sil and hdb_k >= 2 and noise_frac < 0.5:
            chosen_algo = "hdbscan"
            chosen_labels = hdb_labels
            print(f"  -> Choosing HDBSCAN (sil {hdb_sil:.4f} > kmeans {km_sil:.4f})")
        else:
            print(f"  -> Choosing K-means (sil {km_sil:.4f} >= hdbscan {hdb_sil:.4f})")

        with open(cluster_dir / f"{comp}_hdbscan_results.json", "w") as f:
            json.dump(
                {
                    "n_clusters": int(hdb_k),
                    "silhouette": float(hdb_sil),
                    "davies_bouldin": float(hdb_db) if np.isfinite(hdb_db) else None,
                    "noise_fraction": float(noise_frac),
                    "chosen": chosen_algo == "hdbscan",
                },
                f,
                indent=2,
            )

    valid_labels = chosen_labels[chosen_labels >= 0]
    unique_cls = np.unique(valid_labels)

    if len(unique_cls) == 0:
        print(f"  [WARN] No valid clusters for {comp}, skipping")
        return None

    centroids = np.vstack(
        [X_train[chosen_labels == c].mean(axis=0) for c in unique_cls]
    )

    merged_train = merge_small(chosen_labels, centroids, min_cluster)
    unique_merged = np.unique(merged_train[merged_train >= 0])
    if len(unique_merged) == 0:
        unique_merged = np.array([0])
    new_centroids = np.vstack(
        [X_train[merged_train == c].mean(axis=0) for c in unique_merged]
    )

    # Assign every profile to its nearest centroid.
    diffs = X_scaled[:, None, :] - new_centroids[None, :, :]
    dists_all = np.linalg.norm(diffs, axis=2)
    assigned = dists_all.argmin(axis=1)
    min_dists = dists_all.min(axis=1)
    assigned_ids = unique_merged[assigned]

    summary: dict = {}
    for cid in unique_merged:
        cmask = assigned_ids == cid
        cdf = comp_df.iloc[np.where(cmask)[0]]
        entry = {
            "n_all": int(cmask.sum()),
            "n_train": int((merged_train == cid).sum()),
            "algorithm": chosen_algo,
            "pwr_mean": float(cdf["pwr_mean"].mean()) if "pwr_mean" in cdf else None,
            "joules_per_job": (
                float(cdf["joules_per_job"].mean())
                if "joules_per_job" in cdf.columns
                and cdf["joules_per_job"].notna().any()
                else None
            ),
            "pdu_efficiency": (
                float(cdf["pdu_idrac_efficiency"].mean())
                if "pdu_idrac_efficiency" in cdf.columns
                and cdf["pdu_idrac_efficiency"].notna().any()
                else None
            ),
        }
        if "gpu_util_mean" in cdf.columns and cdf["gpu_util_mean"].notna().any():
            entry["gpu_util_mean"] = float(cdf["gpu_util_mean"].mean())
        summary[int(cid)] = entry

    quality = {
        "kmeans_silhouette": km_sil,
        "kmeans_db": km_db,
        "chosen_algorithm": chosen_algo,
        "n_clusters": len(unique_merged),
        "n_train_segments": len(X_train),
    }
    if use_hdbscan:
        quality["hdbscan_silhouette"] = hdb_sil
        quality["hdbscan_db"] = hdb_db

    full_summary = {"clusters": summary, "quality": quality}
    (cluster_dir / f"{comp}_summary.json").write_text(
        json.dumps(full_summary, indent=2, cls=NpEncoder)
    )
    (cluster_dir / f"{comp}_algorithm.txt").write_text(chosen_algo)

    with open(cluster_dir / f"{comp}_kmeans.pkl", "wb") as f:
        pickle.dump((km, new_centroids, unique_merged), f)
    with open(cluster_dir / f"{comp}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(cluster_dir / f"{comp}_features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(cluster_dir / f"{comp}_medians.json", "w") as f:
        json.dump(medians.to_dict(), f, indent=2, cls=NpEncoder)

    print(
        f"  Cluster sizes: "
        + "  ".join(f"C{c}={summary[int(c)]['n_all']}" for c in unique_merged)
    )
    print(f"  Quality: chosen={chosen_algo}  n_clusters={len(unique_merged)}")

    del X_all_raw, X_train, X_scaled, X_train_raw
    gc.collect()
    return assigned_ids, min_dists, chosen_algo
