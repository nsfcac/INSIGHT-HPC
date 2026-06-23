from __future__ import annotations

import gc
import io
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.utils.parsers import first_existing, is_missing, parse_int_list, to_int
from src.utils.rack_topology import rack_id
from src.utils.regime_utils import REGIME_INSIDE_MAINTENANCE, attach_regime_id, regime_safe_slope, rolling_by_regime

TS = "timestamp"

# Defaults — overridden by config.yaml stage3.* when feature_engineering() runs.
ROLL_WINDOWS  = [5, 15, 60]
QUANT_WINDOWS = [15, 60]
QUANT_LEVELS  = [0.05, 0.25, 0.75, 0.95]
PHYSICS_LAGS  = [1, 5, 15]
# 1-day window catches fast ramps; 7-day catches slow thermal/power degradation.
DRIFT_WINDOWS = [1440, 10080]
TWO_PI        = 2 * np.pi

# Load per-job slurm metadata maps for a component.
def load_slurm_job_meta(cfg: dict, comp: str) -> tuple[dict, dict]:
    raw_base   = Path(cfg["paths"].get("raw_parquet", "data/raw_parquet"))
    slurm_path = raw_base / comp / "slurm" / "jobs.parquet"
    if not slurm_path.exists():
        return {}, {}

    need = ["job_id", "exit_code", "memory_per_node",
            "memory_per_cpu", "cpus", "start_time", "end_time"]
    try:
        df = pd.read_parquet(slurm_path, engine="pyarrow", columns=need)
    except Exception:
        try:
            df = pd.read_parquet(slurm_path, engine="pyarrow")
            for c in need:
                if c not in df.columns:
                    df[c] = None
        except Exception:
            return {}, {}

    df["job_id"] = pd.to_numeric(df["job_id"], errors="coerce")
    df = df.dropna(subset=["job_id"])
    df["job_id"] = df["job_id"].astype(int)

    mpn = pd.to_numeric(df["memory_per_node"], errors="coerce").fillna(0)
    mpc = pd.to_numeric(df["memory_per_cpu"],  errors="coerce").fillna(0)
    ncp = pd.to_numeric(df["cpus"],            errors="coerce").fillna(1)
    df["effective_memory_mb"] = np.where(mpn > 0, mpn, mpc * ncp)

    mem_lookup = dict(zip(df["job_id"].astype(int),
                          df["effective_memory_mb"].astype(float)))

    # OOM kills: exit_code=137 with runtime > 60 min.
    df["start_ts"] = pd.to_datetime(pd.to_numeric(df["start_time"], errors="coerce"),
                                    unit="s", utc=True)
    df["end_ts"]   = pd.to_datetime(pd.to_numeric(df["end_time"],   errors="coerce"),
                                    unit="s", utc=True)
    df["runtime_min"] = (df["end_ts"] - df["start_ts"]).dt.total_seconds() / 60.0

    ec = df["exit_code"].astype(str).str.strip()
    oom_mask = ec.str.startswith("137") & (df["runtime_min"] > 60)
    oom_df   = df[oom_mask & df["end_ts"].notna()]
    oom_jobs = dict(zip(oom_df["job_id"].astype(int), oom_df["end_ts"]))

    return mem_lookup, oom_jobs

# Safety-critical sensors that should not be NaN/zero while a node is running.
SILENCE_KEYWORDS = ("rpmreading", "fan", "rpm",
                    "temperaturereading", "temp", "inlet", "exhaust",
                    "systeminputpower", "systempowerconsumption",
                    "totalcpupower", "cpupower")

# Add NaN-streak / coverage silence features per value column.
def silence_features(df: pd.DataFrame, val_cols: list) -> pd.DataFrame:
    new_cols = {}
    for col in val_cols:
        col_lower = col.lower()
        if not any(kw in col_lower for kw in SILENCE_KEYWORDS):
            continue

        vals = df[col].to_numpy(dtype=np.float64)
        is_missing = np.isnan(vals)

        # Power/fan columns: also treat sustained-zero-while-reporting as silence
        is_power_or_fan = any(kw in col_lower for kw in
                              ("rpmreading", "fan", "systeminputpower",
                               "systempowerconsumption", "totalcpupower", "cpupower"))
        if is_power_or_fan:
            is_missing = is_missing | (vals == 0)

        streak = np.zeros(len(vals), dtype=np.float32)
        count = 0
        for i, miss in enumerate(is_missing):
            if miss:
                count += 1
            else:
                count = 0
            streak[i] = count

        new_cols[f"{col}_nan_streak"] = streak

    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Add rolling mean/std/quantile features (regime-aware).
def rolling_features(df: pd.DataFrame, val_cols: list, regime_id: np.ndarray | None = None) -> pd.DataFrame:
    new_cols = {}
    for col in val_cols:
        v = df[col].to_numpy(dtype=np.float64)
        roc = np.empty_like(v); roc[0] = np.nan
        roc[1:] = v[1:] - v[:-1]
        new_cols[f"{col}_roc1"] = roc.astype("float32")
        for w in ROLL_WINDOWS:
            if regime_id is not None:
                rmean = rolling_by_regime(df[col], regime_id, window=w, op="mean")
                rstd  = rolling_by_regime(df[col], regime_id, window=w, op="std")
                new_cols[f"{col}_rmean{w}"] = rmean.to_numpy(dtype=np.float32)
                new_cols[f"{col}_rstd{w}"]  = rstd.to_numpy(dtype=np.float32)
            else:
                roll = df[col].rolling(window=w, min_periods=max(1, w // 2))
                new_cols[f"{col}_rmean{w}"] = roll.mean().astype("float32")
                new_cols[f"{col}_rstd{w}"]  = roll.std().astype("float32")

        for w in QUANT_WINDOWS:
            if regime_id is not None:
                for q in QUANT_LEVELS:
                    qlabel = f"rp{int(q * 100):02d}"
                    res = rolling_by_regime(
                        df[col], regime_id, window=w, op="quantile", quantile=q
                    )
                    new_cols[f"{col}_{qlabel}_{w}"] = res.to_numpy(dtype=np.float32)
            else:
                roll_q = df[col].rolling(window=w, min_periods=max(1, w // 2))
                for q in QUANT_LEVELS:
                    qlabel = f"rp{int(q * 100):02d}"
                    new_cols[f"{col}_{qlabel}_{w}"] = roll_q.quantile(q).astype("float32")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Select the value columns used by physics residuals.
def physics_cols(val_cols: list) -> list:
    sel = []
    for kws in [
        ["power", "consumption", "watt", "dram"],
        ["temp", "temperature"],
        ["fan", "rpm"],
        ["gpuusage", "gpumemoryusage"],
    ]:
        found = [c for c in val_cols
                 if any(k in c.lower() for k in kws)
                 and c not in sel]
        sel.extend(found)
    return sel

# Add lag / rate-of-change features for the given columns.
def lag_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    new_cols = {}
    for col in cols:
        for lag in PHYSICS_LAGS:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Return the metric keywords that get slope features.
def slope_keywords() -> list:
    return ["temp", "temperature", "inlet", "exhaust", "power", "watt", "consumption"]

# Fast rolling least-squares slope over a window.
def rolling_slope_fast(vals: np.ndarray, window: int) -> np.ndarray:
    n = len(vals)
    if n < window:
        return np.full(n, np.nan, dtype=np.float32)

    x_mean    = (window - 1) / 2.0
    x_sq_sum  = window * (window ** 2 - 1) / 12.0   # Σ(x−x̄)²
    if x_sq_sum < 1e-10:
        return np.full(n, np.nan, dtype=np.float32)

    nan_mask   = np.isnan(vals)
    vals_clean = np.where(nan_mask, 0.0, vals.astype(np.float64))
    idx        = np.arange(n, dtype=np.float64)

    S_pad             = np.empty(n + 1, dtype=np.float64); S_pad[0] = 0.0
    np.cumsum(vals_clean, out=S_pad[1:])

    T_pad             = np.empty(n + 1, dtype=np.float64); T_pad[0] = 0.0
    np.cumsum(idx * vals_clean, out=T_pad[1:])

    nan_pad           = np.empty(n + 1, dtype=np.float64); nan_pad[0] = 0.0
    np.cumsum(nan_mask.astype(np.float64), out=nan_pad[1:])

    i_arr     = np.arange(window - 1, n)
    start_arr = i_arr - window + 1

    sum_y   = S_pad[i_arr + 1] - S_pad[start_arr]
    sum_jy  = T_pad[i_arr + 1] - T_pad[start_arr]
    n_nan   = nan_pad[i_arr + 1] - nan_pad[start_arr]

    y_mean  = sum_y / window
    sum_xy  = sum_jy - start_arr * sum_y
    slope   = (sum_xy / window - x_mean * y_mean) / x_sq_sum

    # Suppress windows with too many missing values.
    slope   = np.where(n_nan > window * 0.20, np.nan, slope)

    result  = np.full(n, np.nan, dtype=np.float32)
    result[window - 1:] = slope.astype(np.float32)
    return result

# Add long-window drift/slope features (regime-aware).
def drift_features(df: pd.DataFrame, val_cols: list, regime_id: np.ndarray | None = None) -> pd.DataFrame:
    kws        = slope_keywords()
    drift_cols = [c for c in val_cols if any(k in c.lower() for k in kws)]
    if not drift_cols or not DRIFT_WINDOWS:
        return df

    new_cols: dict = {}
    window_labels = {1440: "1d", 10080: "7d"}
    EPS = 1e-6

    for col in drift_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        for w in DRIFT_WINDOWS:
            label = window_labels.get(w, f"{w}m")
            if regime_id is not None:
                # Boundary-respecting: slope is fit per contiguous regime block, so a 7-day window on 2026-02-20 never fits OLS across a maintenance gap.
                slope_raw = regime_safe_slope(
                    vals, np.asarray(regime_id, dtype=np.int64),
                    w, rolling_slope_fast,
                )
                level = rolling_by_regime(
                    pd.Series(vals), regime_id, window=w,
                    min_periods=max(1, w // 4), op="mean",
                ).to_numpy(dtype=np.float64)
            else:
                slope_raw = rolling_slope_fast(vals, w)
                level = (pd.Series(vals)
                         .rolling(w, min_periods=max(1, w // 4))
                         .mean()
                         .to_numpy(dtype=np.float64))
            denom = np.where(np.abs(level) < EPS, EPS, np.abs(level))
            frac  = (slope_raw.astype(np.float64) * w) / denom
            new_cols[f"{col}_slope{label}"] = frac.astype(np.float32)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Add cyclical time-of-day / day-of-week features.
def time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts   = pd.to_datetime(df[ts_col], utc=True)
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(dtype=np.float64)
    dow  = ts.dt.dayofweek.to_numpy(dtype=np.float64)
    new_cols = {
        "hour_sin_enc": np.sin(TWO_PI * hour / 24).astype("float32"),
        "hour_cos_enc": np.cos(TWO_PI * hour / 24).astype("float32"),
        "dow_sin_enc":  np.sin(TWO_PI * dow  /  7).astype("float32"),
        "dow_cos_enc":  np.cos(TWO_PI * dow  /  7).astype("float32"),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

# Compute consecutive-True run lengths of a boolean mask.
def run_length(mask: pd.Series) -> np.ndarray:
    b = np.asarray(mask, dtype=bool)
    if len(b) == 0:
        return np.empty(0, dtype=np.float32)
    # Group id increments at each False; within each group, run length is cum - cum_at_group_start.
    group_id = np.cumsum(~b)
    group_id -= group_id[0]
    cum = np.cumsum(b)
    _, idx = np.unique(group_id, return_index=True)
    starts = np.zeros(len(idx), dtype=cum.dtype)
    starts[1:] = cum[idx[1:] - 1]
    first_cum = starts[group_id]
    return np.where(b, cum - first_cum, 0).astype(np.float32)

# Mark runs where the primary job id stays constant while active.
def same_job_run(primary: pd.Series, running: pd.Series) -> np.ndarray:
    n = len(primary)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    on  = np.asarray(running, dtype=bool)
    job = np.asarray(primary, dtype=object)

    # Replace NA with a sentinel so element-wise != never evaluates bool(pd.NA).
    is_na = pd.isna(job)
    sentinel = object()
    job_safe = job.copy()
    job_safe[is_na] = sentinel
    boundary = np.ones(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = (~on[1:]) | (job_safe[1:] != job_safe[:-1])
    boundary |= ~on | is_na

    group_id = np.cumsum(boundary)
    group_id -= group_id[0]
    cum_ones = np.cumsum(on & ~is_na)
    _, idx = np.unique(group_id, return_index=True)
    starts = np.zeros(len(idx), dtype=cum_ones.dtype)
    starts[1:] = cum_ones[idx[1:] - 1]
    first_cum = starts[group_id]
    return np.where(on & ~is_na, cum_ones - first_cum, 0).astype(np.float32)

# Add job-context derived features.
def job_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    jobs_col    = first_existing(list(out.columns), ["jobs_json"])
    fallback    = first_existing(list(out.columns), ["jobs", "job_ids", "active_jobs"])
    primary_col = first_existing(list(out.columns), ["primary_job_id", "job_id"])

    if jobs_col:
        job_lists = out[jobs_col].apply(parse_int_list)
    elif fallback:
        job_lists = out[fallback].apply(parse_int_list)
    elif primary_col:
        job_lists = out[primary_col].apply(
            lambda x: [v] if (v := to_int(x)) is not None else [])
    else:
        job_lists = pd.Series([[] for _ in range(len(out))], index=out.index)

    active_count = pd.array([len(v) for v in job_lists], dtype="int16")

    # Use pre-computed columns from Stage 2 when available; otherwise derive.
    if "active_job_count" in out.columns:
        existing = pd.to_numeric(out["active_job_count"],
                                 errors="coerce").fillna(0).astype("int16")
        active_count = np.maximum(active_count, existing.to_numpy(dtype="int16"))

    if "primary_job_id" not in out.columns:
        out["primary_job_id"] = pd.array(
            [v[0] if v else None for v in job_lists.tolist()], dtype="Int64")

    primary = pd.to_numeric(out["primary_job_id"], errors="coerce").astype("Int64")

    if "primary_job_cpu_share" not in out.columns:
        out["primary_job_cpu_share"] = np.nan
    if "total_job_cpu_share" not in out.columns:
        out["total_job_cpu_share"] = np.nan

    pcs = pd.to_numeric(out["primary_job_cpu_share"], errors="coerce").astype("float32")
    tcs = pd.to_numeric(out["total_job_cpu_share"],   errors="coerce").astype("float32")

    busy       = active_count > 0
    active_set = pd.Series(
        [json.dumps(v) if v else None for v in job_lists], index=out.index)

    new_cols = {
        "active_job_count":       active_count.astype("int16"),
        "is_running_job":         busy.astype("float32"),
        "is_multi_job":           (active_count > 1).astype("float32"),
        "primary_job_cpu_share":  pcs.values,
        "total_job_cpu_share":    tcs.values,
        "primary_job_cpu_frac":   (pcs / tcs.replace(0, np.nan)).astype("float32").values,
        "node_busy_run_min":      run_length(pd.Series(busy)),
        "node_idle_run_min":      run_length(pd.Series(~busy)),
        "primary_job_run_min":    same_job_run(primary, pd.Series(busy)),
        "primary_job_changed":    (
            pd.Series(busy) & primary.notna() & (primary != primary.shift(1))
        ).astype("float32").values,
        "active_job_set_changed": (
            active_set.notna() & (active_set != active_set.shift(1))
        ).astype("float32").values,
    }

    req_cpu_col = "job_req_cpus_per_task"  if "job_req_cpus_per_task"  in out.columns else None
    req_tpn_col = "job_req_tasks_per_node" if "job_req_tasks_per_node" in out.columns else None
    act_cpu_col = "total_job_cpu_share"    if "total_job_cpu_share"    in out.columns else None

    n_req_resource = 0

    if req_cpu_col and req_tpn_col and act_cpu_col:
        alloc_cpus = (
            pd.to_numeric(out[req_cpu_col], errors="coerce").fillna(1.0) *
            pd.to_numeric(out[req_tpn_col], errors="coerce").fillna(1.0)
        )
        actual_cpu = pd.to_numeric(out[act_cpu_col], errors="coerce")

        cpu_eff = (actual_cpu / alloc_cpus.replace(0, np.nan)).clip(0, 4).astype("float32")
        new_cols["job_req_cpu_efficiency_avg"] = cpu_eff.values
        n_req_resource += 1

    update_cols = {k: v for k, v in new_cols.items() if k in out.columns}
    add_cols    = {k: v for k, v in new_cols.items() if k not in out.columns}
    for k, v in update_cols.items():
        out[k] = v
    if add_cols:
        out = pd.concat([out, pd.DataFrame(add_cols, index=out.index)], axis=1)

    is_active = pd.Series(busy, index=out.index)

    req_tpn = pd.to_numeric(out.get("job_req_tasks_per_node", pd.Series(np.nan, index=out.index)), errors="coerce").fillna(np.nan)
    req_cpt = pd.to_numeric(out.get("job_req_cpus_per_task",  pd.Series(np.nan, index=out.index)), errors="coerce").fillna(np.nan)
    req_cpus_col = (req_tpn * req_cpt).where(
        req_tpn.notna() & req_cpt.notna(), other=np.nan
    )

    power_class = pd.Series(np.int8(0), index=out.index)  # default: idle

    is_h100 = any("gpuusage" in c.lower() for c in out.columns)

    if is_h100:
        power_class[is_active] = np.int8(4)
    else:
        # ZEN4: bin by req_cpus when active.
        small  = is_active & (req_cpus_col.fillna(0) > 0)  & (req_cpus_col <= 16)
        med    = is_active & (req_cpus_col > 16) & (req_cpus_col <= 64)
        large  = is_active & (req_cpus_col > 64)
        no_req = is_active & req_cpus_col.isna()  # active but no req info → mid-tier
        power_class[small]  = np.int8(1)
        power_class[med]    = np.int8(2)
        power_class[large]  = np.int8(3)
        power_class[no_req] = np.int8(2)

    if "req_power_class" not in out.columns:
        out = pd.concat([out,
            pd.DataFrame({"req_power_class": power_class.astype("int8")},
                         index=out.index)
        ], axis=1)
    else:
        out["req_power_class"] = power_class.astype("int8")

    out._n_req_resource = n_req_resource
    return out

# Compute per-feature train-split normalization stats.
def compute_norm_stats(df: pd.DataFrame, feat_cols: list, hostname: str, comp: str, quiet_only: bool = False) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    if "audit_any" in train.columns:
        train = train[~train["audit_any"].fillna(False)]

    if quiet_only:
        if "is_running_job" in train.columns:
            train = train[train["is_running_job"].fillna(1.0).eq(0.0)]
        elif "active_job_count" in train.columns:
            train = train[train["active_job_count"].fillna(1).eq(0)]
        if len(train) < 60:
            print(f"    [WARN] {hostname}: only {len(train)} idle train rows "
                  f"for norm stats — z-scores may be unreliable")

    # Streak values above this are treated as anomalous silence and excluded from normalization stats; routine collection gaps stay in.
    STREAK_NORMAL_MAX = 30

    records = []
    for col in feat_cols:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if col.endswith("_nan_streak"):
            s = s[s <= STREAK_NORMAL_MAX]
        records.append({
            "component": comp,
            "hostname":  hostname,
            "feature":   col,
            "mean":      float(s.mean()) if len(s) > 0 else 0.0,
            "std":       float(s.std())  if len(s) > 1 else 1.0,
            "count":     int(len(s)),
        })
    return pd.DataFrame(records)

# Apply z-score normalization using precomputed stats.
def normalize(df: pd.DataFrame, feat_cols: list, norm_stats: pd.DataFrame, hostname: str) -> pd.DataFrame:
    STD_FLOOR = 1e-3
    out = df.copy()
    idx = norm_stats[norm_stats["hostname"] == hostname].set_index("feature")
    for col in feat_cols:
        if col not in idx.index:
            continue
        mean = float(idx.loc[col, "mean"])
        std  = float(idx.loc[col, "std"])
        if std < STD_FLOOR:
            std = 1.0
        out[col] = (
            (pd.to_numeric(out[col], errors="coerce") - mean) / std
        ).clip(-10, 10).astype("float32")
    return out

# Build the set of feature-column suffixes.
def build_feat_suffixes() -> tuple:
    suffixes = ["_avg", "_roc1"]
    for w in ROLL_WINDOWS:
        suffixes += [f"_rmean{w}", f"_rstd{w}"]
    for w in QUANT_WINDOWS:
        for q in QUANT_LEVELS:
            suffixes.append(f"_rp{int(q * 100):02d}_{w}")
    for lag in PHYSICS_LAGS:
        suffixes.append(f"_lag{lag}")
    suffixes.append("_enc")
    window_labels = {1440: "1d", 10080: "7d"}
    for w in DRIFT_WINDOWS:
        label = window_labels.get(w, f"{w}m")
        suffixes.append(f"_slope{label}")
    suffixes.append("_nan_streak")
    return tuple(suffixes)

FEAT_SUFFIXES = build_feat_suffixes()

# Aggregate PDU parquets into per-rack power features.
def load_pdu_rack_features(pdu_paths: List[Path]) -> Optional[pd.DataFrame]:
    rows: List[pd.DataFrame] = []

    for p in pdu_paths:
        df = load_parquet(p)
        if df is None or df.empty or TS not in df.columns:
            continue

        df[TS] = pd.to_datetime(df[TS], utc=True).dt.floor("60s")
        df = df.drop_duplicates(TS, keep="last").sort_values(TS).reset_index(drop=True)

        avg_c     = [c for c in df.columns if c.endswith("_avg")
                     and not c.startswith("audit")]
        rmean_c   = [c for c in df.columns if c.endswith("_avg_rmean15")]
        rstd_c    = [c for c in df.columns if c.endswith("_avg_rstd15")]
        audit_col = "audit_any" if "audit_any" in df.columns else None

        outlet = pd.DataFrame({TS: df[TS]})
        outlet["_avg"] = (
            df[avg_c].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            if avg_c else np.nan
        )
        outlet["_rmean15"] = (
            df[rmean_c].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            if rmean_c else np.nan
        )
        outlet["_rstd15"] = (
            np.sqrt((df[rstd_c].apply(pd.to_numeric, errors="coerce") ** 2).mean(axis=1))
            if rstd_c else np.nan
        )
        outlet["_audit"] = (
            df[audit_col].fillna(False).astype(bool) if audit_col else False
        )
        rows.append(outlet)

    if not rows:
        return None

    long = pd.concat(rows, ignore_index=True)
    result = (
        long.groupby(TS, sort=True)
        .agg(
            pdu__rack_total_avg=("_avg",     "sum"),
            pdu__rack_rmean15  =("_rmean15", "mean"),
            pdu__rack_rstd15   =("_rstd15",  "mean"),
            pdu__rack_audit_any=("_audit",   "any"),
        )
        .reset_index()
    )
    result["pdu__rack_total_avg"] = result["pdu__rack_total_avg"].astype("float32")
    result["pdu__rack_rmean15"]   = result["pdu__rack_rmean15"].astype("float32")
    result["pdu__rack_rstd15"]    = result["pdu__rack_rstd15"].astype("float32")
    result["pdu__rack_audit_any"] = result["pdu__rack_audit_any"].astype("int8")
    return result

# Attach rack PDU features to a node frame.
def attach_pdu(node_df: pd.DataFrame, pdu_paths: List[Path]) -> pd.DataFrame:
    if not pdu_paths:
        return node_df
    pdu_df = load_pdu_rack_features(pdu_paths)
    if pdu_df is None or pdu_df.empty:
        return node_df

    n_before = len(node_df)
    node_df[TS] = pd.to_datetime(node_df[TS], utc=True).dt.floor("60s")
    pdu_df[TS]  = pd.to_datetime(pdu_df[TS],  utc=True).dt.floor("60s")
    merged = node_df.merge(pdu_df, on=TS, how="left")

    assert len(merged) == n_before, \
        f"PDU join changed row count {n_before} -> {len(merged)}"
    return merged

# Add node-vs-rack power delta features.
def rack_power_delta(df: pd.DataFrame) -> pd.DataFrame:
    sys_power_col = next(
        (c for c in df.columns if "systeminputpower" in c and c.endswith("_avg")), None
    )
    if not sys_power_col or "pdu__rack_rmean15" not in df.columns:
        return df
    return pd.concat([
        df,
        pd.DataFrame(
            {"node_vs_rack_power_z_avg":
                df[sys_power_col].sub(df["pdu__rack_rmean15"]).astype("float32")},
            index=df.index,
        ),
    ], axis=1)

# Second pass: attach PDU/rack features to all feature files.
def attach_pdu_pass(cfg: dict, force: bool) -> None:
    feat_dir = Path(cfg["paths"]["features"])
    out_base = Path(cfg["paths"].get("features_aligned", "data/features_aligned"))

    pdu_feat_dir = feat_dir / "infra" / "pdu"
    all_pdu_ids: List[str] = (
        [p.stem for p in sorted(pdu_feat_dir.glob("*.parquet"))]
        if pdu_feat_dir.exists() else []
    )
    n_racks = len({rack_id(u) for u in all_pdu_ids if rack_id(u)})
    print(f"\n[features] Pass 3: attach PDU rack features  "
          f"({len(all_pdu_ids)} outlets across {n_racks} racks)")

    pdu_attached = pdu_missing = 0

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue

        comp_feat = feat_dir / comp
        if not comp_feat.exists():
            continue

        (out_base / comp).mkdir(parents=True, exist_ok=True)
        # Resolve PDU paths + counters in the parent (deterministic); the I/O + compute per node runs serially or in a pool.
        tasks = []
        for feat_path in sorted(comp_feat.glob("*.parquet")):
            hostname = feat_path.stem
            out_path = out_base / comp / feat_path.name
            node_rack = rack_id(hostname)
            pdu_paths_node: List[Path] = []
            if node_rack is not None:
                units = [u for u in all_pdu_ids if rack_id(u) == node_rack]
                pdu_paths_node = [pdu_feat_dir / f"{u}.parquet" for u in units]
                if pdu_paths_node:
                    pdu_attached += 1
                else:
                    pdu_missing += 1
                    print(f"    [WARN] {hostname}: no PDU for rack {node_rack}")
            tasks.append((feat_path, out_path, pdu_paths_node, force))

        workers = fe_workers()
        if workers == 1 or not tasks:
            for t in tasks:
                log = fe_pass3_node(*t)[1]
                if log:
                    print(log, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass3_node, *t) for t in tasks]
                results = [fut.result() for fut in as_completed(futs)]
                for _, log, _ in sorted(results, key=lambda r: r[0]):
                    if log:
                        print(log, flush=True)

    # Copy infra/pdu tables across unchanged so downstream sees a complete tree.
    src_pdu = feat_dir / "infra" / "pdu"
    if src_pdu.exists():
        dst_pdu = out_base / "infra" / "pdu"
        dst_pdu.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_pdu.glob("*.parquet")):
            d = dst_pdu / p.name
            if not d.exists() or force:
                shutil.copy2(p, d)

    print(f"  [pdu] attached={pdu_attached}  missing={pdu_missing}  → {out_base}")

# Resolve FE worker count from env or SLURM CPUs (cap 12).
def fe_workers() -> int:
    env = os.environ.get("INSIGHT_HPC_FE_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(12, int(slurm_cpus)))
        except ValueError:
            return 1
    return 1


# Pass 1: compute features and collect norm stats for one node.
def fe_pass1_node(p: Path, comp: str, comp_in: Path, out_dir: Path, force: bool, cfg: dict, ts_col: str, oom_jobs: dict):
    hostname = p.stem
    rel      = p.relative_to(comp_in)
    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return hostname, None, None, f"    skip {hostname}", False

    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = pd.read_parquet(p, engine="pyarrow")
        if ts_col not in df.columns or "split" not in df.columns:
            detail = buf.getvalue().rstrip()
            msg = f"    [WARN] {hostname}: missing {ts_col} or split"
            return hostname, None, None, f"{detail}\n{msg}" if detail else msg, False

        df       = df.sort_values(ts_col).reset_index(drop=True)
        val_cols = [c for c in df.columns if c.endswith("_avg")]
        if not val_cols:
            detail = buf.getvalue().rstrip()
            msg = f"    [WARN] {hostname}: no _avg cols"
            return hostname, None, None, f"{detail}\n{msg}" if detail else msg, False

        df = attach_regime_id(df, cfg=cfg, ts_col=ts_col, exclude_split=True)
        regime_arr = df["regime_id"].to_numpy(dtype=np.int64)

        df = rolling_features(df, val_cols, regime_id=regime_arr)
        df = lag_features(df, physics_cols(val_cols))
        df = drift_features(df, val_cols, regime_id=regime_arr)
        df = silence_features(df, val_cols)
        df = job_features(df)

        fex = pd.Series(False, index=df.index)
        if oom_jobs and "primary_job_id" in df.columns:
            pjid   = pd.to_numeric(df["primary_job_id"], errors="coerce")
            ts_utc = pd.to_datetime(df[ts_col], utc=True)
            window = pd.Timedelta("30min")
            for jid, end_ts in oom_jobs.items():
                mask = (pjid == jid) & (ts_utc >= (end_ts - window)) & (ts_utc <= end_ts)
                fex |= mask
        df["failed_job_exclusion"] = fex.fillna(False).astype("bool")

        n_req_resource = getattr(df, "_n_req_resource", 0)
        df = time_features(df, ts_col)

        feat_cols = [c for c in df.columns if c.endswith(FEAT_SUFFIXES)]
        stats = compute_norm_stats(df, feat_cols, hostname, comp)
        idle  = compute_norm_stats(df, feat_cols, hostname, comp, quiet_only=True)
        save_parquet(df, out_path)
    summary = (f"    {hostname:20s}: {len(df):>8,} rows  {len(feat_cols)} features  "
               f"({n_req_resource} req-resource)  {time.perf_counter()-t0:.1f}s")
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    del df; gc.collect()
    return hostname, stats, idle, log, True


# Pass 2: normalize one node's features with component stats.
def fe_pass2_node(p: Path, comp_stats: pd.DataFrame):
    hostname = p.stem
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = pd.read_parquet(p, engine="pyarrow")
        feat_cols = [c for c in df.columns if c.endswith(FEAT_SUFFIXES)]
        df = normalize(df, feat_cols, comp_stats, hostname)
        save_parquet(df, p)
    summary = f"    normalized {hostname:20s}  {time.perf_counter()-t0:.1f}s"
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    del df; gc.collect()
    return hostname, log


# Pass 3: attach PDU/rack features for one node.
def fe_pass3_node(feat_path: Path, out_path: Path, pdu_paths_node: List[Path], force: bool):
    hostname = feat_path.stem
    if out_path.exists() and not force:
        return hostname, f"    {hostname}: skip (exists)", 0
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = load_parquet(feat_path)
        if df is None or df.empty:
            return hostname, buf.getvalue().rstrip(), 0
        df = attach_pdu(df, pdu_paths_node)
        df = rack_power_delta(df)
        pdu_n = len([c for c in df.columns if c.startswith("pdu__")])
        save_parquet(df, out_path)
    summary = (f"    {hostname:20s}: {len(df):>8,} rows  +pdu={pdu_n}  "
               f"{time.perf_counter() - t0:.1f}s")
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    return hostname, log, pdu_n


# Run the 3-pass feature-engineering pipeline for all nodes.
def feature_engineering(force: bool = False) -> None:
    global ROLL_WINDOWS, QUANT_WINDOWS, QUANT_LEVELS, DRIFT_WINDOWS, FEAT_SUFFIXES

    cfg        = load_config()
    master_dir = Path(cfg["paths"]["master"])
    out_base   = Path(cfg["paths"]["features"])
    ts_col     = "timestamp"

    s3 = cfg.get("stage3", {})
    ROLL_WINDOWS  = s3.get("rolling_windows_min",         ROLL_WINDOWS)
    QUANT_WINDOWS = s3.get("rolling_quantile_windows_min", QUANT_WINDOWS)
    QUANT_LEVELS  = s3.get("rolling_quantiles",            QUANT_LEVELS)
    DRIFT_WINDOWS = s3.get("drift_slope_windows_min",      DRIFT_WINDOWS)
    FEAT_SUFFIXES = build_feat_suffixes()

    stage_start    = time.perf_counter()
    all_stats      = []
    all_idle_stats = []
    total_nodes    = 0

    print("\n[features] Pass 1: compute features + collect norm stats")
    print(f"  rolling_windows={ROLL_WINDOWS}  quantile_windows={QUANT_WINDOWS}  "
          f"quantiles={QUANT_LEVELS}  drift_windows={DRIFT_WINDOWS}")
    for comp_cfg in cfg["components"]:
        comp    = comp_cfg["name"]
        comp_in = master_dir / comp
        if not comp_in.exists():
            continue

        out_dir = out_base / comp
        out_dir.mkdir(parents=True, exist_ok=True)

        if comp == "infra":
            parquets = [p for p in sorted(comp_in.rglob("*.parquet"))
                        if "irc" not in p.parts]
        else:
            parquets = sorted(comp_in.glob("*.parquet"))
        print(f"\n  {comp.upper()}  {len(parquets)} nodes")

        mem_lookup, oom_jobs = load_slurm_job_meta(cfg, comp)
        print(f"    SLURM meta: {len(mem_lookup):,} jobs, {len(oom_jobs):,} OOM kills")

        workers = fe_workers()
        args = [(p, comp, comp_in, out_dir, force, cfg, ts_col, oom_jobs)
                for p in parquets]
        results = []
        if workers == 1 or not args:
            for a in args:
                r = fe_pass1_node(*a)
                print(r[3], flush=True)
                results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass1_node, *a) for a in args]
                results = [fut.result() for fut in as_completed(futs)]
                for r in sorted(results, key=lambda r: r[0]):
                    print(r[3], flush=True)
        # deterministic reduce (independent of completion order)
        for _, stats, idle, _, processed in sorted(results, key=lambda r: r[0]):
            if processed:
                all_stats.append(stats)
                all_idle_stats.append(idle)
                total_nodes += 1
        gc.collect()

    if not all_stats:
        print("[features] No data processed")
        return

    norm_stats = pd.concat(all_stats, ignore_index=True)
    norm_path  = out_base / "norm_stats.parquet"
    save_parquet(norm_stats, norm_path)
    print(f"\n  norm_stats saved: {len(norm_stats)} entries → {norm_path}")

    if all_idle_stats:
        idle_stats = pd.concat(all_idle_stats, ignore_index=True)
        idle_path  = out_base / "norm_stats_idle.parquet"
        save_parquet(idle_stats, idle_path)
        n_with_idle = int((idle_stats["count"] > 0).sum())
        print(f"  norm_stats_idle saved: {len(idle_stats)} entries "
              f"({n_with_idle} with idle rows) → {idle_path}")

    print("\n[features] Pass 2: apply z-score normalization (clean train stats)")
    for comp_cfg in cfg["components"]:
        comp    = comp_cfg["name"]
        out_dir = out_base / comp
        if not out_dir.exists():
            continue
        comp_stats = norm_stats[norm_stats["component"] == comp]
        parquets   = sorted(out_dir.rglob("*.parquet")) if comp == "infra" \
                     else sorted(out_dir.glob("*.parquet"))
        parquets   = [p for p in parquets if "irc" not in p.parts]  # no-IRC
        workers = fe_workers()
        if workers == 1 or not parquets:
            for p in parquets:
                print(fe_pass2_node(p, comp_stats)[1], flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass2_node, p, comp_stats) for p in parquets]
                results = [fut.result() for fut in as_completed(futs)]
                for _, log in sorted(results, key=lambda r: r[0]):
                    print(log, flush=True)
        gc.collect()

    attach_pdu_pass(cfg, force=force)

    elapsed = time.perf_counter() - stage_start
    print(f"\nFeature Engineering completed. nodes={total_nodes}  time={elapsed:.1f}s")
    print(f"Norm stats: {norm_path}")

if __name__ == "__main__":
    feature_engineering(force=True)
