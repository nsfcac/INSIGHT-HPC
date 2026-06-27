from __future__ import annotations

from offline.phase3.physics_constraints_module.constants import *
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.parsers import find_avg, findany


# Return the sorted union of several datetime indexes, preserving tz.
def union_ts_index(indexes) -> pd.DatetimeIndex:
    indexes = list(indexes)
    if not indexes:
        return pd.DatetimeIndex([])
    arrs = [getattr(i, "values", i) for i in indexes]
    result = pd.DatetimeIndex(np.unique(np.concatenate(arrs)))
    tz = getattr(indexes[0], "tz", None)
    if tz is not None and result.tz is None:
        result = result.tz_localize(tz)
    return result


TS = "timestamp"

# Per-worker constraint params, populated by rack_jobs.worker_init in each
# worker process. Defaults keep check_node safe if invoked without init.
WORKER_C1_PARAMS: dict = {}
WORKER_C3_PARAMS: dict = {}


# Convert a timestamp series/index to nanosecond-resolution UTC.
def to_ns(s):
    dti = pd.to_datetime(s, utc=True)
    if hasattr(dti, "as_unit"):  # DatetimeIndex path
        return dti.as_unit("ns")
    if hasattr(dti, "dt"):  # Series path
        return dti.dt.as_unit("ns")
    return dti


# Return the inlet-temperature "_avg" column name, if any.
def inlet_col(cols: list):
    return next(
        (
            c
            for c in cols
            if "temperaturereading" in c.lower()
            and "inlet" in c.lower()
            and c.endswith("_avg")
        ),
        None,
    ) or next((c for c in cols if "inlet" in c.lower() and c.endswith("_avg")), None)


# Return the system-power "_avg" column name, if any.
def node_power_col(cols: list):
    return find_avg(cols, "systeminputpower") or findany(cols, "systempowerconsumption")


# C1: flag minutes where inlet temperature rises while fan speed stays low.
def check_temp_fan_decoupling(
    df: pd.DataFrame,
    temp_rise_rate: float = 0.15,
    fan_slack: float = 0.90,
    window_min: int = 5,
) -> pd.Series:
    cols = list(df.columns)
    temp_col = inlet_col(cols)
    fan_cols = [
        c
        for c in cols
        if ("rpmreading" in c.lower() or "fanspeed" in c.lower()) and c.endswith("_avg")
    ]

    if not temp_col or not fan_cols:
        return pd.Series(False, index=df.index)

    temp = pd.to_numeric(df[temp_col], errors="coerce")
    fan = df[fan_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Rolling rate of temperature change
    temp_rate = temp.diff(window_min)  # °C change over window_min minutes
    fan_series = fan.copy()

    # Baseline: training median fan speed
    train_fan_baseline = fan_series.quantile(0.50)
    if pd.isna(train_fan_baseline) or train_fan_baseline == 0:
        return pd.Series(False, index=df.index)

    temp_rising = temp_rate > temp_rise_rate * window_min
    fan_low = fan_series < train_fan_baseline * fan_slack

    fired = (temp_rising & fan_low).fillna(False)

    # Idle gate: only count minutes with no active job.
    if "active_job_count" in cols:
        idle = df["active_job_count"].fillna(0).astype(int).eq(0)
        fired = fired & idle

    return fired


# C3: flag minutes where power and temperature dynamics decouple.
def check_physics_dynamics_violation(
    df: pd.DataFrame,
    power_ramp_pct: float = 0.20,
    lag_window_min: int = 10,
    min_temp_rise: float = 1.5,
    temp_rise_per_min: float = 0.25,
    power_stable_min: int = 5,
    power_stable_w: float = 20.0,
    min_power_rise_w: float = 30.0,
    min_temp_rise_abs: float = 1.5,
) -> pd.Series:
    cols = list(df.columns)
    power_col = node_power_col(cols)
    temp_col = inlet_col(cols)

    if not power_col or not temp_col:
        return pd.Series(False, index=df.index)

    power = pd.to_numeric(df[power_col], errors="coerce")
    temp = pd.to_numeric(df[temp_col], errors="coerce")

    # --- Check A: did power rise lag ago, and has temp responded since? ---
    p_lag = power.shift(lag_window_min)  # power at T-lag
    p_2lag = power.shift(2 * lag_window_min)  # power at T-2*lag
    power_ramp_past = (p_lag - p_2lag) / p_2lag.replace(0, np.nan)
    power_abs_rise = p_lag - p_2lag
    power_rose_past = (power_ramp_past > power_ramp_pct) & (
        power_abs_rise > min_power_rise_w
    )

    t_lag = temp.shift(lag_window_min)  # inlet temp at T-lag (start of lag window)
    temp_responded = (temp - t_lag) >= min_temp_rise

    check_a = power_rose_past & ~temp_responded

    temp_rise_now = temp - temp.shift(power_stable_min)
    temp_is_rising = (temp_rise_now > (temp_rise_per_min * power_stable_min)) & (
        temp_rise_now > min_temp_rise_abs
    )

    p_prev = power.shift(power_stable_min)
    power_is_stable = (power - p_prev).abs() < power_stable_w

    check_b = temp_is_rising & power_is_stable
    if "active_job_count" in cols:
        idle = df["active_job_count"].fillna(0).astype(int).eq(0)
        check_b = check_b & idle

    return (check_a | check_b).fillna(False)


# C4: flag minutes where PDU power rises but the rack's node power stays flat.
def check_rack_cross_plane_disagreement(
    rack_node_dfs: dict,
    pdu_series: Optional[pd.Series],
    rack_id: int,
    pdu_rise_pct: float = 0.20,
    node_stable: float = 0.05,
    window_min: int = 10,
    min_node_frac: float = 0.5,
    min_pdu_rise_w: float = 200.0,
    min_mismatch_w: float = 100.0,
) -> pd.Series:
    if not rack_node_dfs or pdu_series is None or pdu_series.empty:
        return pd.Series(dtype=bool)

    node_power = {}
    for hostname, df in rack_node_dfs.items():
        cols = list(df.columns)
        power_col = node_power_col(cols)
        if not power_col or TS not in df.columns:
            continue
        power = pd.to_numeric(df[power_col], errors="coerce")
        power.index = pd.to_datetime(df[TS], utc=True)
        node_power[hostname] = power

    if not node_power:
        return pd.Series(dtype=bool)

    ts_index = union_ts_index([s.index for s in node_power.values()])
    node_df = pd.DataFrame({h: s.reindex(ts_index) for h, s in node_power.items()})
    reporting_frac = node_df.notna().mean(axis=1)
    node_total = node_df.sum(axis=1, min_count=1)

    target_frame = pd.DataFrame({TS: to_ns(ts_index)})
    source_frame = pdu_series.rename("_pdu_val").reset_index()
    source_frame.columns = [TS, "_pdu_val"]
    source_frame[TS] = to_ns(source_frame[TS])
    pdu_aligned = pd.merge_asof(
        target_frame.sort_values(TS),
        source_frame.sort_values(TS),
        on=TS,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=5),
    ).set_index(TS)["_pdu_val"]

    pdu_prev = pdu_aligned.shift(window_min)
    node_prev = node_total.shift(window_min)

    pdu_change = (pdu_aligned - pdu_prev) / pdu_prev.replace(0, np.nan)
    pdu_change_abs = pdu_aligned - pdu_prev  # absolute W
    node_change = (node_total - node_prev) / node_prev.replace(0, np.nan)
    node_change_abs = node_total - node_prev  # absolute W

    mismatch_abs = (pdu_change_abs - node_change_abs).abs()

    valid = (reporting_frac >= min_node_frac) & pdu_prev.notna() & node_prev.notna()
    violation = (
        valid
        & (pdu_change > pdu_rise_pct)
        & (pdu_change_abs > min_pdu_rise_w)  # 2026-04-21: absolute PDU rise floor
        & (node_change.abs() < node_stable)
        & (mismatch_abs > min_mismatch_w)  # 2026-04-21: absolute mismatch floor
    )
    return violation.fillna(False)


# Keep only constraint flags that persist for the minimum consecutive minutes.
def apply_persistence_filter(
    df: pd.DataFrame,
    min_consecutive: int = 3,
    per_constraint_min: Optional[dict] = None,
) -> pd.DataFrame:
    df = df.copy()
    if per_constraint_min is None:
        per_constraint_min = {"const3_dynamics": 4, "const4_crossplane": 4}
    flag_cols = [
        "const1_temp_fan",
        "const2_rack_therm",
        "const3_dynamics",
        "const4_crossplane",
    ]
    for col in flag_cols:
        if col not in df.columns:
            continue
        flags = df[col].astype(bool)
        if not flags.any():
            continue
        run_ids = (flags != flags.shift(1)).cumsum()
        run_lengths = flags.groupby(run_ids).transform("sum")
        col_min = int(per_constraint_min.get(col, min_consecutive))
        df[col] = flags & (run_lengths >= col_min)
    c_present = [c for c in flag_cols if c in df.columns]
    if "const5_alloc_idle" in df.columns:
        c_present = c_present + ["const5_alloc_idle"]
    df["n_constraints_violated"] = sum(df[c].astype(int) for c in c_present)

    # Bit positions: TEMP_FAN=0, RACK_THERM=1, DYNAMICS=2, CROSSPLANE=3, ALLOC_IDLE=4.
    bit_for_col = {
        "const1_temp_fan": 1,
        "const2_rack_therm": 2,
        "const3_dynamics": 4,
        "const4_crossplane": 8,
        "const5_alloc_idle": 16,
    }
    bit = np.zeros(len(df), dtype=np.uint8)
    for col, b in bit_for_col.items():
        if col in df.columns:
            bit |= df[col].astype(bool).values.astype(np.uint8) * b
    df["constraint_flags"] = FLAG_LOOKUP[bit]
    return df


# C5: flag sustained periods where a node is allocated but hardware is idle.
def check_allocation_vs_hardware_idle(
    df: pd.DataFrame,
    p_ratio: float = 0.85,
    cpu_ratio: float = 0.15,
    gpu_ratio: float = 0.15,
    roll_window: int = 60,
    sustain_min: int = 30,
    grace_min: int = 20,
    end_grace_min: int = 5,
) -> pd.Series:
    cols = list(df.columns)
    power_col = node_power_col(cols)
    cpu_col = next(
        (c for c in cols if c.startswith("cpuusage__") and c.endswith("_avg")), None
    )
    gpu_cols = [c for c in cols if c.startswith("gpuusage__") and c.endswith("_avg")]

    if not power_col or not cpu_col or "active_job_count" not in cols:
        return pd.Series(False, index=df.index)

    power = pd.to_numeric(df[power_col], errors="coerce")
    cpu = pd.to_numeric(df[cpu_col], errors="coerce")
    if gpu_cols:
        gpu = df[gpu_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        gpu = None

    roll_p = power.rolling(roll_window, min_periods=max(10, roll_window // 3)).max()
    roll_c = cpu.rolling(roll_window, min_periods=max(10, roll_window // 3)).max()

    power_low = (power < roll_p * p_ratio).fillna(False)
    cpu_low = (cpu < roll_c * cpu_ratio).fillna(False)
    if gpu is not None:
        roll_g = gpu.rolling(roll_window, min_periods=max(10, roll_window // 3)).max()
        gpu_low = (gpu < roll_g * gpu_ratio).fillna(False)
        collapse = power_low & cpu_low & gpu_low
    else:
        collapse = power_low & cpu_low

    active = df["active_job_count"].fillna(0).astype(int).ge(1)
    if "maintenance_flag" in cols:
        active &= ~df["maintenance_flag"].fillna(False).astype(bool)
    collapse = collapse & active

    # Warmup grace: skip the first grace_min rows after each primary_job_id change.
    pj = (
        df.get("primary_job_id", pd.Series(-1, index=df.index))
        .fillna(-1)
        .astype("int64")
    )
    changed = (pj != pj.shift(1)).fillna(True).to_numpy()
    n = len(df)
    idx = np.arange(n, dtype=np.int32)
    last_change = np.maximum.accumulate(np.where(changed, idx, 0).astype(np.int32))
    in_warmup = (idx - last_change) < grace_min

    from numpy.lib.stride_tricks import sliding_window_view

    fut = active.astype(int).to_numpy()
    pad = np.concatenate([fut, np.zeros(end_grace_min, dtype=int)])
    win = sliding_window_view(pad, end_grace_min)
    ending_soon = (win.sum(axis=1) < end_grace_min)[:n]

    flagged = collapse.to_numpy() & (~in_warmup) & (~ending_soon)

    sustained = np.zeros_like(flagged)
    if flagged.any():
        # Run id increments on every transition; gate to True rows via flagged mask.
        boundary = np.concatenate([[True], flagged[:-1] != flagged[1:]])
        run_id = np.cumsum(boundary)
        # Length of each run (across all rows), then look up per-row length.
        run_len = np.bincount(run_id)[run_id]
        sustained = flagged & (run_len >= sustain_min)

    return pd.Series(sustained, index=df.index)


# Run all node-level constraint checks and return per-minute violation flags.
def check_node(
    df: pd.DataFrame, hostname: str, component: str, seg_idx: Optional[dict] = None
) -> pd.DataFrame:
    if TS not in df.columns:
        return pd.DataFrame()

    df = df.sort_values(TS).copy()

    c1_enabled = (
        bool(WORKER_C1_PARAMS.get("enabled", False)) if WORKER_C1_PARAMS else False
    )
    if c1_enabled:
        c1_runtime_params = {
            k: v for k, v in (WORKER_C1_PARAMS or {}).items() if k != "enabled"
        }
        c1 = check_temp_fan_decoupling(df, **c1_runtime_params)
    else:
        c1 = pd.Series(False, index=df.index)
    c3 = check_physics_dynamics_violation(df, **(WORKER_C3_PARAMS or {}))
    # CROSSPLANE is attached later at rack scope using summed node power.
    c4 = pd.Series(False, index=df.index)

    # c2 (rack thermal/power mismatch) requires cross-node data — handled separately
    c2 = pd.Series(False, index=df.index)

    # ALLOC_IDLE: allocation-vs-hardware mismatch.
    c5 = check_allocation_vs_hardware_idle(df)

    violated = (
        c1.astype(int)
        + c2.astype(int)
        + c3.astype(int)
        + c4.astype(int)
        + c5.astype(int)
    )

    c1_i = c1.values.astype(np.uint8)
    c3_i = c3.values.astype(np.uint8)
    c4_i = c4.values.astype(np.uint8)
    c5_i = c5.values.astype(np.uint8)
    bit = c1_i | (c3_i << 2) | (c4_i << 3) | (c5_i << 4)
    flags = FLAG_LOOKUP[bit]

    return pd.DataFrame(
        {
            TS: pd.to_datetime(df[TS], utc=True),
            "hostname": hostname,
            "component": component,
            "const1_temp_fan": c1.values,
            "const2_rack_therm": c2.values,
            "const3_dynamics": c3.values,
            "const4_crossplane": c4.values,
            "const5_alloc_idle": c5.values,
            "n_constraints_violated": violated.astype("int8").values,
            "constraint_flags": flags,
        }
    )
