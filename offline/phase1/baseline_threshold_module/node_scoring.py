from __future__ import annotations

import gc, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import apply_node_limit, load_config, save_parquet
from shared.utils.maintenance import apply_maintenance_mask, load_maintenance_windows
from shared.utils.rack_topology import rack_id


from offline.phase1.baseline_threshold_module.threshold_loading import *


# Apply every threshold rule to one node and return per-rule flag columns.
def score_node(
    df: pd.DataFrame, hostname: str, component: str, cfg_thr: dict, windows: list
) -> Optional[pd.DataFrame]:
    ts_col = "timestamp"
    if ts_col not in df.columns:
        return None

    df = df.sort_values(ts_col).copy()
    df = apply_maintenance_mask(df, windows, ts_col, "hostname")

    cols = list(df.columns)

    if "is_running_job" in df.columns:
        is_running = pd.to_numeric(df["is_running_job"], errors="coerce").fillna(0) > 0
    elif "active_job_count" in df.columns:
        is_running = df["active_job_count"].fillna(0).astype(float) > 0
    else:
        is_running = pd.Series(False, index=df.index)

    rule_flags: dict[str, pd.Series] = {
        r: pd.Series(False, index=df.index) for r in RULES
    }

    train_rows = (
        df[df.get("split", "").astype(str) == "train"]
        if "split" in df.columns
        else df.iloc[0:0]
    )
    if not train_rows.empty:
        try:
            from offline.utils.gt_mask import in_gt_window

            tr_for_gt = train_rows.copy()
            tr_for_gt["hostname"] = hostname
            gt_mask = in_gt_window(tr_for_gt)
            if gt_mask.any():
                train_rows = train_rows[~gt_mask.values]
        except Exception:
            pass

    # Return the max of per-metric 99.5th percentiles over clean train rows.
    def host_p99_5(metric_cols: list) -> float:
        if train_rows.empty or not metric_cols:
            return 0.0
        vals = []
        for mc in metric_cols:
            if mc in train_rows.columns:
                s = pd.to_numeric(train_rows[mc], errors="coerce").dropna()
                if not s.empty:
                    vals.append(float(s.quantile(0.995)))
        return float(max(vals)) if vals else 0.0

    # Raise a fleet threshold to at least 1.1x the host's own p99.5 baseline.
    def thr_with_host_floor(fleet_thr: float, metric_cols: list) -> float:
        return max(float(fleet_thr), host_p99_5(metric_cols) * 1.10)

    if component == "h100":
        thr_cfg = cfg_thr.get(
            "h100_system_power_w", cfg_thr.get("system_power_w", 4500)
        )
    else:
        thr_cfg = cfg_thr.get("system_power_w", 1800)

    power_cols = any_avg(cols, "systeminputpower", "systempowerconsumption")
    thr = thr_with_host_floor(thr_cfg, power_cols)
    for pc in power_cols:
        rule_flags["HIGH_SYSTEM_POWER"] |= (
            pd.to_numeric(df[pc], errors="coerce").fillna(0) > thr
        )

    cpu_temp_cols = pick_avg(cols, "temperaturereading", "cpu")
    if not cpu_temp_cols:
        cpu_temp_cols = [
            c
            for c in any_avg(cols, "temperaturereading")
            if "inlet" not in c.lower() and "exhaust" not in c.lower()
        ]
    thr = thr_with_host_floor(cfg_thr.get("cpu_temp_c", 85), cpu_temp_cols)
    for tc in cpu_temp_cols:
        rule_flags["HIGH_CPU_TEMP"] |= (
            pd.to_numeric(df[tc], errors="coerce").fillna(0) > thr
        )

    inlet_cols = pick_avg(cols, "temperaturereading", "inlet")
    if not inlet_cols:
        inlet_cols = pick_avg(cols, "inlet", "temp")
    thr = thr_with_host_floor(cfg_thr.get("inlet_temp_c", 35), inlet_cols)
    for tc in inlet_cols:
        rule_flags["HIGH_INLET_TEMP"] |= (
            pd.to_numeric(df[tc], errors="coerce").fillna(0) > thr
        )

    fan_low_thr = cfg_thr.get("fan_rpm_low", 1500)
    fan_temp_thr = cfg_thr.get("fan_temp_crosscheck", 70)

    rpm_cols = any_avg(cols, "rpmreading", "fanspeed", "fanrpm")
    if rpm_cols:
        rpm_values = (
            df[rpm_cols]
            .apply(pd.to_numeric, errors="coerce")
            .min(axis=1)
            .fillna(np.nan)
        )
        fan_is_low = rpm_values < fan_low_thr
    else:
        fan_is_low = pd.Series(False, index=df.index)

    temp_high_for_fan = pd.Series(False, index=df.index)
    for tc in any_avg(cols, "temperaturereading"):
        temp_high_for_fan |= (
            pd.to_numeric(df[tc], errors="coerce").fillna(0) > fan_temp_thr
        )

    rule_flags["FAN_FAIL"] = fan_is_low & temp_high_for_fan

    cpu_comp_cols = any_avg(cols, "totalcpupower", "cpupower")
    thr = thr_with_host_floor(cfg_thr.get("cpu_component_power_w", 800), cpu_comp_cols)
    for pc in cpu_comp_cols:
        rule_flags["HIGH_CPU_COMPONENT_POWER"] |= (
            pd.to_numeric(df[pc], errors="coerce").fillna(0) > thr
        )

    fan_pwr_cols = any_avg(cols, "totalfanpower")
    thr = thr_with_host_floor(cfg_thr.get("fan_power_w_high", 230), fan_pwr_cols)
    for fc in fan_pwr_cols:
        rule_flags["HIGH_FAN_POWER"] |= (
            pd.to_numeric(df[fc], errors="coerce").fillna(0) > thr
        )

    if component == "h100":
        gpu_pwr_thr = cfg_thr.get("gpu_power_w", 380)
        gpu_util_thr = cfg_thr.get("gpu_util_low_pct", 10)

        gpu_pwr_cols = any_avg(cols, "powerconsumption")
        gpu_util_cols = any_avg(cols, "gpuusage")

        if gpu_pwr_cols and gpu_util_cols:
            gpu_pwr = (
                df[gpu_pwr_cols]
                .apply(pd.to_numeric, errors="coerce")
                .max(axis=1)
                .fillna(0)
            )
            gpu_util = (
                df[gpu_util_cols]
                .apply(pd.to_numeric, errors="coerce")
                .max(axis=1)
                .fillna(0)
            )
            rule_flags["GPU_POWER_IDLE"] = (
                is_running & (gpu_pwr > gpu_pwr_thr) & (gpu_util < gpu_util_thr)
            )

    rpm_pct_cap = float(cfg_thr.get("fan_rpm_pct_absolute_cap", 4000))
    rpm_pct_frac = float(cfg_thr.get("fan_rpm_pct_of_baseline", 0.50))
    if rpm_cols:
        rpm_min_per_min = df[rpm_cols].apply(pd.to_numeric, errors="coerce").min(axis=1)
        rpm_baseline = rpm_min_per_min.rolling(window=10080, min_periods=60).median()
        baseline_ok = rpm_baseline.notna() & (rpm_baseline > 100)
        current = rpm_min_per_min.fillna(np.inf)
        rule_flags["FAN_RPM_PCT_DROP"] = (
            baseline_ok
            & (current < rpm_pct_frac * rpm_baseline)
            & (current < rpm_pct_cap)
        )

    idle_delta_w = float(cfg_thr.get("idle_power_delta_w", 80))
    if power_cols:
        p_current = df[power_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        cpu_use_cols = any_avg(cols, "cpuusage")
        if cpu_use_cols:
            cpu_use = (
                df[cpu_use_cols]
                .apply(pd.to_numeric, errors="coerce")
                .max(axis=1)
                .fillna(0)
            )
            idle_mask = (~is_running) & (cpu_use < 5.0)
        else:
            idle_mask = ~is_running
        idle_power = p_current.where(idle_mask)
        idle_baseline = idle_power.rolling(window=10080, min_periods=120).median()
        baseline_ok = idle_baseline.notna()
        idle_surge_raw = (
            idle_mask & baseline_ok & (p_current > idle_baseline + idle_delta_w)
        )
        if idle_surge_raw.any():
            run_id = (idle_surge_raw != idle_surge_raw.shift()).cumsum()
            run_len = idle_surge_raw.groupby(run_id).transform("sum")
            rule_flags["IDLE_POWER_SURGE"] = (idle_surge_raw & (run_len >= 30)).astype(
                bool
            )
        else:
            rule_flags["IDLE_POWER_SURGE"] = idle_surge_raw

    FLATLINE_METRICS = {
        "rackinlettemperature1": 0.25,
        "maximumrackinlettemperature": 0.25,
        "returntemperature": 0.4,
        "supplytemperature": 0.4,
        "systeminputpower": 15.0,
        "systempowerconsumption": 15.0,
        "totalcpupower": 10.0,
        "cpupower": 10.0,
        "cpuusage": 5.0,
    }
    FLATLINE_MIN_VALUE = {
        "cpuusage": 95.0,
    }
    FLATLINE_STD_RATIO = 0.03
    FLATLINE_MIN_RUN = 10
    FLATLINE_REF_WIN = 360
    FLATLINE_REF_LAG = 60
    flatline_mask = pd.Series(False, index=df.index)
    for kw, std_floor in FLATLINE_METRICS.items():
        metric_cols = [c for c in cols if c.endswith("_avg") and kw in c.lower()]
        for mc in metric_cols:
            vals = pd.to_numeric(df[mc], errors="coerce")
            if vals.dropna().empty:
                continue
            std15 = vals.rolling(window=15, min_periods=10).std()
            std_ref = (
                vals.rolling(window=FLATLINE_REF_WIN, min_periods=60)
                .std()
                .shift(FLATLINE_REF_LAG)
            )
            normally_varies = std_ref.fillna(0) > std_floor
            zero_run = (
                (std15 < FLATLINE_STD_RATIO * std_ref) & vals.notna() & normally_varies
            )
            min_val = FLATLINE_MIN_VALUE.get(kw)
            if min_val is not None:
                zero_run = zero_run & (vals >= min_val)
            if zero_run.any():
                run_id = (zero_run != zero_run.shift()).cumsum()
                run_len = zero_run.groupby(run_id).transform("sum")
                flatline_mask |= zero_run & (run_len >= FLATLINE_MIN_RUN)
    rule_flags["FLATLINE"] = flatline_mask

    DROPOUT_FAMILIES = [
        ["systeminputpower", "systempowerconsumption"],
        ["rackinlettemperature1", "maximumrackinlettemperature"],
        ["rpmreading"],
        ["cpupower", "totalcpupower"],
        ["powerconsumption"],
        ["compositetemperature"],
    ]
    DROPOUT_MIN_RUN = 5
    DROPOUT_MIN_FAMILIES = 2
    family_nan = pd.DataFrame(index=df.index)
    for i, fam_kws in enumerate(DROPOUT_FAMILIES):
        fam_cols = []
        for kw in fam_kws:
            fam_cols.extend([c for c in cols if c.endswith("_avg") and kw in c.lower()])
        if not fam_cols:
            continue
        vals = df[fam_cols].apply(pd.to_numeric, errors="coerce")
        family_nan[f"fam_{i}"] = vals.isna().all(axis=1)
    if not family_nan.empty:
        n_fam_nan = family_nan.sum(axis=1)
        dropout_row = n_fam_nan >= DROPOUT_MIN_FAMILIES
        if dropout_row.any():
            run_id = (dropout_row != dropout_row.shift()).cumsum()
            run_len = dropout_row.groupby(run_id).transform("sum")
            rule_flags["DROPOUT"] = dropout_row & (run_len >= DROPOUT_MIN_RUN)

    if component == "h100":
        gpu_temp_rate_thr = float(cfg_thr.get("gpu_temp_rate_c_per_min", 1.0))
        gpu_temp_rate_min_run = int(cfg_thr.get("gpu_temp_rate_min_run_min", 10))

        gpu_temp_cols = [
            c
            for c in cols
            if c.endswith("_avg")
            and "temperaturereading" in c.lower()
            and "gpu" in c.lower()
        ]
        if not gpu_temp_cols:
            gpu_temp_cols = [
                c
                for c in any_avg(cols, "temperaturereading")
                if not any(
                    t in c.lower()
                    for t in ("inlet", "exhaust", "cpu", "dimm", "storage")
                )
            ]
        gpu_temp_thr = thr_with_host_floor(cfg_thr.get("gpu_temp_c", 85), gpu_temp_cols)
        for tc in gpu_temp_cols:
            v = pd.to_numeric(df[tc], errors="coerce")
            rule_flags["HIGH_GPU_TEMP"] |= v.fillna(0) > gpu_temp_thr
            diff_10 = v - v.shift(gpu_temp_rate_min_run)
            rate_hit = diff_10 >= (gpu_temp_rate_thr * gpu_temp_rate_min_run)
            rule_flags["HIGH_GPU_TEMP"] |= rate_hit.fillna(False)

    cold_inlet_thr = float(cfg_thr.get("cold_inlet_threshold_c", 15.0))
    cold_inlet_run = int(cfg_thr.get("cold_inlet_min_run_min", 10))
    cold_inlet_cols = pick_avg(cols, "temperaturereading", "inlet")
    if not cold_inlet_cols:
        cold_inlet_cols = pick_avg(cols, "inlet", "temp")
    cold_inlet_mask = pd.Series(False, index=df.index)
    for tc in cold_inlet_cols:
        vals = pd.to_numeric(df[tc], errors="coerce")
        is_cold = vals.notna() & (vals > 0) & (vals < cold_inlet_thr)
        cold_with_job = is_cold & is_running
        if cold_with_job.any():
            run_id = (cold_with_job != cold_with_job.shift()).cumsum()
            run_len = cold_with_job.groupby(run_id).transform("sum")
            cold_inlet_mask |= cold_with_job & (run_len >= cold_inlet_run)
    rule_flags["COLD_INLET_DROP"] = cold_inlet_mask

    for rname, min_run in PERSIST_RULES.items():
        m = rule_flags.get(rname)
        if m is None or not m.any():
            continue
        run_id = (m != m.shift()).cumsum()
        run_len = m.groupby(run_id).transform("sum")
        rule_flags[rname] = (m & (run_len >= min_run)).astype(bool)

    flag_df = pd.DataFrame(
        {f"rule_{r}": rule_flags[r].astype(bool) for r in RULES},
        index=df.index,
    )
    flag_df["flag_count"] = (
        flag_df[[f"rule_{r}" for r in RULES]].sum(axis=1).astype("int8")
    )
    flag_df["is_flagged"] = flag_df["flag_count"] > 0
    flag_df["triggered_rules"] = flag_df[[f"rule_{r}" for r in RULES]].apply(
        lambda row: "|".join(r for r in RULES if row[f"rule_{r}"]), axis=1
    )

    out = pd.DataFrame(
        {
            "timestamp": df[ts_col].values,
            "hostname": hostname,
            "component": component,
            "split": df["split"].values if "split" in df.columns else "unknown",
            "is_running_job": is_running.astype("float32").values,
            "maintenance_flag": df["maintenance_flag"].values,
        }
    )
    for col in flag_df.columns:
        out[col] = flag_df[col].values

    return out


# Compute per-PDU-outlet power thresholds from clean train data.
def compute_pdu_thresholds(
    master_dir: Path, pdu_percentile: float, windows: list
) -> Dict[str, float]:
    pdu_dir = master_dir / "infra" / "pdu"
    if not pdu_dir.exists():
        return {}

    thresholds: Dict[str, float] = {}
    for p in sorted(pdu_dir.glob("*.parquet")):
        df = read_threshold_frame(p)
        if df is None or df.empty:
            continue

        unit_id = p.stem
        df = apply_maintenance_mask(df, windows, "timestamp", None)

        train = df[df["split"] == "train"] if "split" in df.columns else df
        train = train[~train["maintenance_flag"].fillna(False)]
        try:
            from offline.utils.gt_mask import in_gt_window

            gt = in_gt_window(train)
            if gt.any():
                train = train[~gt]
        except Exception:
            pass

        pdu_cols = [
            c for c in train.columns if c.endswith("_avg") and "pdu" in c.lower()
        ]
        if not pdu_cols:
            continue

        vals = train[pdu_cols].apply(pd.to_numeric, errors="coerce").values.ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) < 10:
            continue

        thresholds[unit_id] = float(np.percentile(vals, pdu_percentile))

    return thresholds


# Sum per-node iDRAC system power into per-rack, per-minute totals.
def build_rack_idrac_power(master_dir: Path, cfg: dict) -> pd.DataFrame:
    frames = []
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue

        for p in apply_node_limit(sorted(comp_dir.glob("*.parquet"))):
            hostname = p.stem
            rid = rack_id(hostname)
            if rid is None:
                continue

            df = read_threshold_frame(p)
            if df is None or df.empty:
                continue

            ts_col = "timestamp"
            if ts_col not in df.columns:
                continue

            power_col = None
            for kw in ["systeminputpower", "systempowerconsumption"]:
                found = [
                    c for c in df.columns if c.endswith("_avg") and kw in c.lower()
                ]
                if found:
                    power_col = found[0]
                    break

            if power_col is None:
                continue

            tmp = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df[ts_col], utc=True),
                    "rack_id": rid,
                    "power_w": pd.to_numeric(df[power_col], errors="coerce"),
                }
            ).dropna()

            frames.append(tmp)
            del df, tmp
            gc.collect()

    if not frames:
        return pd.DataFrame(columns=["timestamp", "rack_id", "idrac_power_sum_w"])

    combined = pd.concat(frames, ignore_index=True)
    rack_power = (
        combined.set_index("timestamp")
        .groupby(["timestamp", "rack_id"])["power_w"]
        .sum()
        .rename("idrac_power_sum_w")
        .reset_index()
    )
    del combined
    gc.collect()
    return rack_power
