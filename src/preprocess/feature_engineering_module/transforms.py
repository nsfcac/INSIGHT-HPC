from __future__ import annotations

from src.preprocess.feature_engineering_module.constants import *
import gc, io, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.utils.parsers import first_existing, is_missing, parse_int_list, to_int
from src.utils.rack_topology import rack_id
from src.utils.regime_utils import (
    REGIME_INSIDE_MAINTENANCE,
    attach_regime_id,
    regime_safe_slope,
    rolling_by_regime,
)


# Load per-job memory allocations and OOM-killed job end-times from Slurm.
def load_slurm_job_meta(cfg: dict, comp: str) -> tuple[dict, dict]:
    raw_base = Path(cfg["paths"].get("raw_parquet", "data/raw_parquet"))
    slurm_path = raw_base / comp / "slurm" / "jobs.parquet"
    if not slurm_path.exists():
        return {}, {}

    need = [
        "job_id",
        "exit_code",
        "memory_per_node",
        "memory_per_cpu",
        "cpus",
        "start_time",
        "end_time",
    ]
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
    mpc = pd.to_numeric(df["memory_per_cpu"], errors="coerce").fillna(0)
    ncp = pd.to_numeric(df["cpus"], errors="coerce").fillna(1)
    df["effective_memory_mb"] = np.where(mpn > 0, mpn, mpc * ncp)

    mem_lookup = dict(
        zip(df["job_id"].astype(int), df["effective_memory_mb"].astype(float))
    )

    # OOM kills: exit_code=137 with runtime > 60 min.
    df["start_ts"] = pd.to_datetime(
        pd.to_numeric(df["start_time"], errors="coerce"), unit="s", utc=True
    )
    df["end_ts"] = pd.to_datetime(
        pd.to_numeric(df["end_time"], errors="coerce"), unit="s", utc=True
    )
    df["runtime_min"] = (df["end_ts"] - df["start_ts"]).dt.total_seconds() / 60.0

    ec = df["exit_code"].astype(str).str.strip()
    oom_mask = ec.str.startswith("137") & (df["runtime_min"] > 60)
    oom_df = df[oom_mask & df["end_ts"].notna()]
    oom_jobs = dict(zip(oom_df["job_id"].astype(int), oom_df["end_ts"]))

    return mem_lookup, oom_jobs


# Safety-critical sensors that should not be NaN/zero while a node is running.
SILENCE_KEYWORDS = (
    "rpmreading",
    "fan",
    "rpm",
    "temperaturereading",
    "temp",
    "inlet",
    "exhaust",
    "systeminputpower",
    "systempowerconsumption",
    "totalcpupower",
    "cpupower",
)


# Add per-sensor NaN/zero streak lengths for safety-critical sensors.
def silence_features(df: pd.DataFrame, val_cols: list) -> pd.DataFrame:
    new_cols = {}
    for col in val_cols:
        col_lower = col.lower()
        if not any(kw in col_lower for kw in SILENCE_KEYWORDS):
            continue

        vals = df[col].to_numpy(dtype=np.float64)
        is_missing = np.isnan(vals)

        # Power/fan columns: also treat sustained-zero-while-reporting as silence
        is_power_or_fan = any(
            kw in col_lower
            for kw in (
                "rpmreading",
                "fan",
                "systeminputpower",
                "systempowerconsumption",
                "totalcpupower",
                "cpupower",
            )
        )
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


# Add rate-of-change, rolling mean/std, and rolling quantile features per column.
def rolling_features(
    df: pd.DataFrame, val_cols: list, regime_id: np.ndarray | None = None
) -> pd.DataFrame:
    new_cols = {}
    for col in val_cols:
        v = df[col].to_numpy(dtype=np.float64)
        roc = np.empty_like(v)
        roc[0] = np.nan
        roc[1:] = v[1:] - v[:-1]
        new_cols[f"{col}_roc1"] = roc.astype("float32")
        for w in ROLL_WINDOWS:
            if regime_id is not None:
                rmean = rolling_by_regime(df[col], regime_id, window=w, op="mean")
                rstd = rolling_by_regime(df[col], regime_id, window=w, op="std")
                new_cols[f"{col}_rmean{w}"] = rmean.to_numpy(dtype=np.float32)
                new_cols[f"{col}_rstd{w}"] = rstd.to_numpy(dtype=np.float32)
            else:
                roll = df[col].rolling(window=w, min_periods=max(1, w // 2))
                new_cols[f"{col}_rmean{w}"] = roll.mean().astype("float32")
                new_cols[f"{col}_rstd{w}"] = roll.std().astype("float32")

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
                    new_cols[f"{col}_{qlabel}_{w}"] = roll_q.quantile(q).astype(
                        "float32"
                    )
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# Select power/temperature/fan/GPU columns eligible for physics lag features.
def physics_cols(val_cols: list) -> list:
    sel = []
    for kws in [
        ["power", "consumption", "watt", "dram"],
        ["temp", "temperature"],
        ["fan", "rpm"],
        ["gpuusage", "gpumemoryusage"],
    ]:
        found = [
            c for c in val_cols if any(k in c.lower() for k in kws) and c not in sel
        ]
        sel.extend(found)
    return sel


# Add lagged copies of the given columns at the configured lags.
def lag_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    new_cols = {}
    for col in cols:
        for lag in PHYSICS_LAGS:
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag).astype("float32")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def slope_keywords() -> list:
    return ["temp", "temperature", "inlet", "exhaust", "power", "watt", "consumption"]


# Compute a fast rolling OLS slope via cumulative sums.
def rolling_slope_fast(vals: np.ndarray, window: int) -> np.ndarray:
    n = len(vals)
    if n < window:
        return np.full(n, np.nan, dtype=np.float32)

    x_mean = (window - 1) / 2.0
    x_sq_sum = window * (window**2 - 1) / 12.0  # Σ(x−x̄)²
    if x_sq_sum < 1e-10:
        return np.full(n, np.nan, dtype=np.float32)

    nan_mask = np.isnan(vals)
    vals_clean = np.where(nan_mask, 0.0, vals.astype(np.float64))
    idx = np.arange(n, dtype=np.float64)

    S_pad = np.empty(n + 1, dtype=np.float64)
    S_pad[0] = 0.0
    np.cumsum(vals_clean, out=S_pad[1:])

    T_pad = np.empty(n + 1, dtype=np.float64)
    T_pad[0] = 0.0
    np.cumsum(idx * vals_clean, out=T_pad[1:])

    nan_pad = np.empty(n + 1, dtype=np.float64)
    nan_pad[0] = 0.0
    np.cumsum(nan_mask.astype(np.float64), out=nan_pad[1:])

    i_arr = np.arange(window - 1, n)
    start_arr = i_arr - window + 1

    sum_y = S_pad[i_arr + 1] - S_pad[start_arr]
    sum_jy = T_pad[i_arr + 1] - T_pad[start_arr]
    n_nan = nan_pad[i_arr + 1] - nan_pad[start_arr]

    y_mean = sum_y / window
    sum_xy = sum_jy - start_arr * sum_y
    slope = (sum_xy / window - x_mean * y_mean) / x_sq_sum

    # Suppress windows with too many missing values.
    slope = np.where(n_nan > window * 0.20, np.nan, slope)

    result = np.full(n, np.nan, dtype=np.float32)
    result[window - 1 :] = slope.astype(np.float32)
    return result


# Add normalised long-window drift slopes for thermal and power columns.
def drift_features(
    df: pd.DataFrame, val_cols: list, regime_id: np.ndarray | None = None
) -> pd.DataFrame:
    kws = slope_keywords()
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
                    vals,
                    np.asarray(regime_id, dtype=np.int64),
                    w,
                    rolling_slope_fast,
                )
                level = rolling_by_regime(
                    pd.Series(vals),
                    regime_id,
                    window=w,
                    min_periods=max(1, w // 4),
                    op="mean",
                ).to_numpy(dtype=np.float64)
            else:
                slope_raw = rolling_slope_fast(vals, w)
                level = (
                    pd.Series(vals)
                    .rolling(w, min_periods=max(1, w // 4))
                    .mean()
                    .to_numpy(dtype=np.float64)
                )
            denom = np.where(np.abs(level) < EPS, EPS, np.abs(level))
            frac = (slope_raw.astype(np.float64) * w) / denom
            new_cols[f"{col}_slope{label}"] = frac.astype(np.float32)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# Add cyclic sine/cosine encodings of hour-of-day and day-of-week.
def time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col], utc=True)
    hour = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(dtype=np.float64)
    dow = ts.dt.dayofweek.to_numpy(dtype=np.float64)
    new_cols = {
        "hour_sin_enc": np.sin(TWO_PI * hour / 24).astype("float32"),
        "hour_cos_enc": np.cos(TWO_PI * hour / 24).astype("float32"),
        "dow_sin_enc": np.sin(TWO_PI * dow / 7).astype("float32"),
        "dow_cos_enc": np.cos(TWO_PI * dow / 7).astype("float32"),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# Compute the running length of consecutive True values in a mask.
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


# Compute how long the same primary job has been running continuously.
def same_job_run(primary: pd.Series, running: pd.Series) -> np.ndarray:
    n = len(primary)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    on = np.asarray(running, dtype=bool)
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


# Derive job-occupancy, cpu-efficiency, and power-class features.
def job_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    jobs_col = first_existing(list(out.columns), ["jobs_json"])
    fallback = first_existing(list(out.columns), ["jobs", "job_ids", "active_jobs"])
    primary_col = first_existing(list(out.columns), ["primary_job_id", "job_id"])

    if jobs_col:
        job_lists = out[jobs_col].apply(parse_int_list)
    elif fallback:
        job_lists = out[fallback].apply(parse_int_list)
    elif primary_col:
        job_lists = out[primary_col].apply(
            lambda x: [v] if (v := to_int(x)) is not None else []
        )
    else:
        job_lists = pd.Series([[] for _ in range(len(out))], index=out.index)

    active_count = pd.array([len(v) for v in job_lists], dtype="int16")

    # Use pre-computed columns from Stage 2 when available; otherwise derive.
    if "active_job_count" in out.columns:
        existing = (
            pd.to_numeric(out["active_job_count"], errors="coerce")
            .fillna(0)
            .astype("int16")
        )
        active_count = np.maximum(active_count, existing.to_numpy(dtype="int16"))

    if "primary_job_id" not in out.columns:
        out["primary_job_id"] = pd.array(
            [v[0] if v else None for v in job_lists.tolist()], dtype="Int64"
        )

    primary = pd.to_numeric(out["primary_job_id"], errors="coerce").astype("Int64")

    if "primary_job_cpu_share" not in out.columns:
        out["primary_job_cpu_share"] = np.nan
    if "total_job_cpu_share" not in out.columns:
        out["total_job_cpu_share"] = np.nan

    pcs = pd.to_numeric(out["primary_job_cpu_share"], errors="coerce").astype("float32")
    tcs = pd.to_numeric(out["total_job_cpu_share"], errors="coerce").astype("float32")

    busy = active_count > 0
    active_set = pd.Series(
        [json.dumps(v) if v else None for v in job_lists], index=out.index
    )

    new_cols = {
        "active_job_count": active_count.astype("int16"),
        "is_running_job": busy.astype("float32"),
        "is_multi_job": (active_count > 1).astype("float32"),
        "primary_job_cpu_share": pcs.values,
        "total_job_cpu_share": tcs.values,
        "primary_job_cpu_frac": (pcs / tcs.replace(0, np.nan)).astype("float32").values,
        "node_busy_run_min": run_length(pd.Series(busy)),
        "node_idle_run_min": run_length(pd.Series(~busy)),
        "primary_job_run_min": same_job_run(primary, pd.Series(busy)),
        "primary_job_changed": (
            pd.Series(busy) & primary.notna() & (primary != primary.shift(1))
        )
        .astype("float32")
        .values,
        "active_job_set_changed": (
            active_set.notna() & (active_set != active_set.shift(1))
        )
        .astype("float32")
        .values,
    }

    req_cpu_col = (
        "job_req_cpus_per_task" if "job_req_cpus_per_task" in out.columns else None
    )
    req_tpn_col = (
        "job_req_tasks_per_node" if "job_req_tasks_per_node" in out.columns else None
    )
    act_cpu_col = (
        "total_job_cpu_share" if "total_job_cpu_share" in out.columns else None
    )

    n_req_resource = 0

    if req_cpu_col and req_tpn_col and act_cpu_col:
        alloc_cpus = pd.to_numeric(out[req_cpu_col], errors="coerce").fillna(
            1.0
        ) * pd.to_numeric(out[req_tpn_col], errors="coerce").fillna(1.0)
        actual_cpu = pd.to_numeric(out[act_cpu_col], errors="coerce")

        cpu_eff = (
            (actual_cpu / alloc_cpus.replace(0, np.nan)).clip(0, 4).astype("float32")
        )
        new_cols["job_req_cpu_efficiency_avg"] = cpu_eff.values
        n_req_resource += 1

    update_cols = {k: v for k, v in new_cols.items() if k in out.columns}
    add_cols = {k: v for k, v in new_cols.items() if k not in out.columns}
    for k, v in update_cols.items():
        out[k] = v
    if add_cols:
        out = pd.concat([out, pd.DataFrame(add_cols, index=out.index)], axis=1)

    is_active = pd.Series(busy, index=out.index)

    req_tpn = pd.to_numeric(
        out.get("job_req_tasks_per_node", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).fillna(np.nan)
    req_cpt = pd.to_numeric(
        out.get("job_req_cpus_per_task", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).fillna(np.nan)
    req_cpus_col = (req_tpn * req_cpt).where(
        req_tpn.notna() & req_cpt.notna(), other=np.nan
    )

    power_class = pd.Series(np.int8(0), index=out.index)  # default: idle

    is_h100 = any("gpuusage" in c.lower() for c in out.columns)

    if is_h100:
        power_class[is_active] = np.int8(4)
    else:
        # ZEN4: bin by req_cpus when active.
        small = is_active & (req_cpus_col.fillna(0) > 0) & (req_cpus_col <= 16)
        med = is_active & (req_cpus_col > 16) & (req_cpus_col <= 64)
        large = is_active & (req_cpus_col > 64)
        no_req = is_active & req_cpus_col.isna()  # active but no req info → mid-tier
        power_class[small] = np.int8(1)
        power_class[med] = np.int8(2)
        power_class[large] = np.int8(3)
        power_class[no_req] = np.int8(2)

    if "req_power_class" not in out.columns:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {"req_power_class": power_class.astype("int8")}, index=out.index
                ),
            ],
            axis=1,
        )
    else:
        out["req_power_class"] = power_class.astype("int8")

    out.n_req_resource_count = n_req_resource
    return out
