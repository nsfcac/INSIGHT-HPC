from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


from offline.phase2.coherence_module.batch import *


# Worker entry: score one job segment's streaming coherence from the cached global state.
def process_segment_worker(work_unit):
    jid, comp, seg_start, seg_end, peer_hosts = work_unit
    art = GLOBAL_ART_CACHE.get(comp)
    if art is None:
        return []
    scaler, feature_cols, medians = art
    if scaler is None:
        return []
    peer_master = {}
    for h in peer_hosts:
        df = GLOBAL_MASTER_CACHE.get((h, comp))
        if df is not None and len(df) > 0:
            peer_master[h] = df
    if len(peer_master) < GLOBAL_PARAMS["min_peers"]:
        return []
    return streaming_coherence_for_segment(
        int(jid),
        comp,
        seg_start,
        seg_end,
        peer_master,
        feature_cols,
        scaler,
        medians,
        GLOBAL_PARAMS["fingerprint_min"],
        GLOBAL_PARAMS["update_every_min"],
        GLOBAL_PARAMS["z_thresh"],
        GLOBAL_PARAMS["min_peers"],
    )


# Return the first column whose name contains the keyword.
def first_col(cols, kw: str) -> Optional[str]:
    for c in cols:
        if kw.lower() in c.lower():
            return c
    return None


# Build profile features from a job's accumulated rows up to the current minute.
def build_running_profile(
    accum: pd.DataFrame, component: str, duration_s: float
) -> dict:
    if len(accum) < 2:
        return {"profile_valid": False}
    cols = list(accum.columns)
    rec: dict = {"profile_valid": True}

    pwr_col = first_col(cols, "systeminputpower") or first_col(
        cols, "systempowerconsumption"
    )
    if pwr_col:
        s = pd.to_numeric(accum[pwr_col], errors="coerce").dropna()
        if len(s) > 0:
            rec["pwr_mean"] = float(s.mean())
            rec["pwr_max"] = float(s.max())
            rec["pwr_std"] = float(s.std()) if len(s) > 1 else np.nan
            diffs = s.diff().dropna()
            rec["power_ramp_rate"] = float(diffs.std()) if len(diffs) > 1 else np.nan
            rec["joules_per_job"] = float(s.mean()) * duration_s
        else:
            rec.update(
                pwr_mean=np.nan,
                pwr_max=np.nan,
                pwr_std=np.nan,
                power_ramp_rate=np.nan,
                joules_per_job=np.nan,
            )
    else:
        rec.update(
            pwr_mean=np.nan,
            pwr_max=np.nan,
            pwr_std=np.nan,
            power_ramp_rate=np.nan,
            joules_per_job=np.nan,
        )

    cpu_pwr_col = first_col(cols, "totalcpupower")
    cpu_load_col = "slurm_cpu_load" if "slurm_cpu_load" in cols else None
    rec["cpu_pwr_mean"] = (
        float(pd.to_numeric(accum[cpu_pwr_col], errors="coerce").dropna().mean())
        if cpu_pwr_col
        and pd.to_numeric(accum[cpu_pwr_col], errors="coerce").notna().any()
        else np.nan
    )
    rec["cpu_load_mean"] = (
        float(pd.to_numeric(accum[cpu_load_col], errors="coerce").dropna().mean())
        if cpu_load_col
        and pd.to_numeric(accum[cpu_load_col], errors="coerce").notna().any()
        else np.nan
    )
    if not np.isnan(rec.get("pwr_mean", np.nan)) and not np.isnan(
        rec.get("cpu_load_mean", np.nan)
    ):
        cpu_pct = rec["cpu_load_mean"] / 100.0
        rec["watts_per_cpu_pct"] = (
            (rec["pwr_mean"] / cpu_pct) if cpu_pct > 0.01 else np.nan
        )
    else:
        rec["watts_per_cpu_pct"] = np.nan

    mem_pwr_col = first_col(cols, "totalmemorypower")
    mem_usage_col = "slurm_memory_usage" if "slurm_memory_usage" in cols else None
    rec["mem_pwr_mean"] = (
        float(pd.to_numeric(accum[mem_pwr_col], errors="coerce").dropna().mean())
        if mem_pwr_col
        and pd.to_numeric(accum[mem_pwr_col], errors="coerce").notna().any()
        else np.nan
    )
    rec["mem_usage_mean"] = (
        float(pd.to_numeric(accum[mem_usage_col], errors="coerce").dropna().mean())
        if mem_usage_col
        and pd.to_numeric(accum[mem_usage_col], errors="coerce").notna().any()
        else np.nan
    )

    inlet_cols = [
        c
        for c in cols
        if "temperaturereading" in c.lower()
        and "inlet" in c.lower()
        and c.endswith("_avg")
    ]
    exhaust_cols = [
        c
        for c in cols
        if "temperaturereading" in c.lower()
        and "exhaust" in c.lower()
        and c.endswith("_avg")
    ]
    fan_cols = [
        c
        for c in cols
        if ("rpmreading" in c.lower() or "fanspeed" in c.lower()) and c.endswith("_avg")
    ]
    inlet_s = (
        accum[inlet_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        if inlet_cols
        else pd.Series(dtype=float)
    )
    exhaust_s = (
        accum[exhaust_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
        if exhaust_cols
        else pd.Series(dtype=float)
    )
    fan_s = (
        accum[fan_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
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

    if component == "h100":
        util_cols = [c for c in cols if "gpuusage" in c.lower() and c.endswith("_avg")]
        gpwr_cols = [
            c
            for c in cols
            if c.endswith("_avg")
            and "powerconsumption" in c.lower()
            and "systempowerconsumption" not in c.lower()
        ]
        gtemp_cols = [
            c
            for c in cols
            if "temperaturereading" in c.lower()
            and "gputemp" in c.lower()
            and c.endswith("_avg")
        ]
        gmem_cols = [
            c for c in cols if "gpumemoryusage" in c.lower() and c.endswith("_avg")
        ]
        gpu_util = (
            accum[util_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if util_cols
            else pd.Series(dtype=float)
        )
        gpu_pwr = (
            accum[gpwr_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if gpwr_cols
            else pd.Series(dtype=float)
        )
        gpu_temp = (
            accum[gtemp_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .dropna()
            if gtemp_cols
            else pd.Series(dtype=float)
        )
        gpu_mem = (
            accum[gmem_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).dropna()
            if gmem_cols
            else pd.Series(dtype=float)
        )
        rec["gpu_util_mean"] = float(gpu_util.mean()) if len(gpu_util) > 0 else np.nan
        rec["gpu_util_max"] = float(gpu_util.max()) if len(gpu_util) > 0 else np.nan
        rec["gpu_pwr_mean"] = float(gpu_pwr.mean()) if len(gpu_pwr) > 0 else np.nan
        rec["gpu_pwr_max"] = float(gpu_pwr.max()) if len(gpu_pwr) > 0 else np.nan
        rec["gpu_temp_mean"] = float(gpu_temp.mean()) if len(gpu_temp) > 0 else np.nan
        rec["gpu_mem_mean"] = float(gpu_mem.mean()) if len(gpu_mem) > 0 else np.nan
        if not np.isnan(rec.get("gpu_pwr_mean", np.nan)):
            rec["gpu_joules"] = rec["gpu_pwr_mean"] * duration_s
        else:
            rec["gpu_joules"] = np.nan

    return rec


# Replay a segment minute-by-minute, scoring each node against its peers' running profiles.
def streaming_coherence_for_segment(
    job_id: int,
    component: str,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    peer_master: dict,
    feature_cols: list,
    scaler,
    medians: dict,
    fingerprint_min: int,
    update_every_min: int,
    z_thresh: float,
    min_peers: int,
) -> list:
    if len(peer_master) < min_peers:
        return []

    sliced: dict = {}
    start_ns = timestamp_to_ns(seg_start)
    end_ns = timestamp_to_ns(seg_end)
    for host, df in peer_master.items():
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        ts_ns_raw = series_to_ns(ts)
        order = np.argsort(ts_ns_raw)
        ts_ns_all = ts_ns_raw[order]
        left = int(np.searchsorted(ts_ns_all, start_ns, side="left"))
        right = int(np.searchsorted(ts_ns_all, end_ns, side="right"))
        if right - left < 2:
            continue
        sub = df.iloc[order[left:right]].copy()
        sub["__ts"] = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce")
        sliced[host] = (sub, series_to_ns(sub["__ts"]))

    grid = pd.date_range(
        seg_start + pd.Timedelta(minutes=fingerprint_min),
        seg_end,
        freq=f"{update_every_min}min",
        tz="UTC",
    )
    if len(grid) == 0:
        return []

    out: list = []
    for T in grid:
        profiles: list = []
        t_ns = timestamp_to_ns(T)
        for host, (sub, ts_ns) in sliced.items():
            pos = int(np.searchsorted(ts_ns, t_ns, side="right"))
            if pos < 2:
                continue
            accum = sub.iloc[:pos]
            duration_s = float((T - seg_start).total_seconds()) + 60.0
            prof = build_running_profile(accum, component, duration_s)
            if not prof.get("profile_valid", False):
                continue
            profiles.append((host, prof))

        if len(profiles) < min_peers:
            continue

        vectors: list = []
        for host, prof in profiles:
            vec = profile_to_vector(pd.Series(prof), feature_cols, medians)
            if vec is None:
                continue
            vectors.append((host, vec))
        if len(vectors) < min_peers:
            continue

        X = np.vstack([v for _, v in vectors])
        try:
            X = scaler.transform(X)
        except Exception:
            continue
        median_vec = np.median(X, axis=0)
        dists = np.linalg.norm(X - median_vec, axis=1)
        med_dist = float(np.median(dists))
        mad_dist = float(np.median(np.abs(dists - med_dist)))
        scale = 1.4826 * mad_dist
        max_dist = float(dists.max())

        for i, (host, _) in enumerate(vectors):
            d = float(dists[i])
            z = (d - med_dist) / scale if scale > 1e-6 else 0.0
            cs = d / max_dist if max_dist > 1e-6 else 0.0
            top_feats = top_deviant_features(X[i], median_vec, feature_cols)
            out.append(
                {
                    "hostname": host,
                    "job_id": int(job_id),
                    "component": component,
                    "timestamp": T,
                    "streaming_coherence_z": float(z),
                    "is_streaming_anomaly": bool(z > z_thresh),
                    "streaming_coherence_score": float(cs),
                    "n_peer_nodes": len(vectors),
                    "top_deviant_features_json": top_feats,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                }
            )
    return out
