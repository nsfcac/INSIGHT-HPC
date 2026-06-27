from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from shared.utils.io_utils import load_parquet


# Return "_avg" columns matching all of the given keywords.
def pick_avg(columns: list, *keywords: str) -> list:
    kws = [k.lower() for k in keywords]
    return [
        c for c in columns if c.endswith("_avg") and all(k in c.lower() for k in kws)
    ]


# Return "_avg" columns matching any of the given keywords.
def any_avg(columns: list, *keywords: str) -> list:
    kws = [k.lower() for k in keywords]
    return [
        c for c in columns if c.endswith("_avg") and any(k in c.lower() for k in kws)
    ]


THRESHOLD_KEYWORDS = (
    "systeminputpower",
    "systempowerconsumption",
    "temperaturereading",
    "inlet",
    "temp",
    "rpmreading",
    "fanspeed",
    "fanrpm",
    "totalcpupower",
    "cpupower",
    "totalfanpower",
    "powerconsumption",
    "gpuusage",
    "cpuusage",
    "rackinlettemperature",
    "maximumrackinlettemperature",
    "returntemperature",
    "supplytemperature",
    "compositetemperature",
    "pdu",
)
SUPPORT_COLUMNS = (
    "timestamp",
    "hostname",
    "split",
    "is_running_job",
    "active_job_count",
    "maintenance_flag",
)


# Read the support columns plus threshold-relevant "_avg" sensors from a node.
def read_threshold_frame(path: Path) -> Optional[pd.DataFrame]:
    try:
        schema_cols = pq.read_schema(path).names
        keep = [c for c in SUPPORT_COLUMNS if c in schema_cols]
        for c in schema_cols:
            cl = c.lower()
            if c.endswith("_avg") and any(k in cl for k in THRESHOLD_KEYWORDS):
                keep.append(c)
        keep = [c for i, c in enumerate(keep) if c not in keep[:i]]
        if "timestamp" not in keep:
            return None
        return pd.read_parquet(path, engine="pyarrow", columns=keep)
    except Exception:
        return load_parquet(path)


RULES = [
    "HIGH_SYSTEM_POWER",
    "HIGH_CPU_TEMP",
    "HIGH_INLET_TEMP",
    "FAN_FAIL",
    "GPU_POWER_IDLE",
    "HIGH_CPU_COMPONENT_POWER",
    "HIGH_FAN_POWER",
    "FAN_RPM_PCT_DROP",
    "IDLE_POWER_SURGE",
    "HIGH_GPU_TEMP",
    "FLATLINE",
    "DROPOUT",
    "COLD_INLET_DROP",
]

PERSIST_RULES = {
    "HIGH_SYSTEM_POWER": 5,
    "HIGH_CPU_TEMP": 5,
    "HIGH_INLET_TEMP": 5,
    "HIGH_GPU_TEMP": 5,
    "HIGH_CPU_COMPONENT_POWER": 5,
    "HIGH_FAN_POWER": 5,
}


# Derive per-component threshold values from the metric-thresholds CSV percentiles.
def load_dynamic_thresholds(cfg: dict, comp: str) -> dict:
    thr_path = (
        Path(cfg["paths"]["visualization"]) / "thresholds" / "metric_thresholds.csv"
    )
    if not thr_path.exists():
        return {}

    try:
        df = pd.read_csv(thr_path)
    except Exception as e:
        print(f"  [dyn_thr] Cannot read {thr_path}: {e}")
        return {}

    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    platform = comp.lower()

    # Return a percentile-column value for a metric, or None if absent.
    def pct(
        metric: str, col: str, agg: str = "max", source_type: str = "idrac"
    ) -> Optional[float]:
        mask = (
            (df["platform"].str.lower() == platform)
            & (df["source_type"].str.lower() == source_type)
            & (df["metric"].str.lower() == metric.lower())
        )
        rows = df[mask]
        if rows.empty or col not in rows.columns:
            return None
        vals = pd.to_numeric(rows[col], errors="coerce").dropna()
        if vals.empty:
            return None
        return float(vals.max() if agg == "max" else vals.min())

    result: dict = {}
    logged: list = []

    v = pct("systeminputpower", "p99") or pct("systempowerconsumption", "p99")
    if v is not None:
        result["system_power_w"] = round(v * 1.05, 1)
        logged.append(f"sys_pwr>{result['system_power_w']:.0f}W")

    v = pct("compositetemperature", "p99") or pct("temperaturereading", "p99")
    if v is not None:
        result["cpu_temp_c"] = round(v * 1.05, 1)
        logged.append(f"cpu_temp>{result['cpu_temp_c']:.1f}°C")

    v = pct("rackinlettemperature1", "p99") or pct("maximumrackinlettemperature", "p99")
    if v is not None:
        result["inlet_temp_c"] = round(v * 1.05, 1)
        logged.append(f"inlet>{result['inlet_temp_c']:.1f}°C")

    v = pct("totalcpupower", "p99") or pct("cpupower", "p99")
    if v is not None:
        result["cpu_component_power_w"] = round(v * 1.05, 1)
        logged.append(f"cpu_comp_pwr>{result['cpu_component_power_w']:.0f}W")

    v = pct("totalfanpower", "p99")
    if v is not None:
        result["fan_power_w_high"] = round(v * 1.05, 1)
        logged.append(f"fan_pwr>{result['fan_power_w_high']:.0f}W")

    v = pct("rpmreading", "p001", agg="min")
    if v is not None and v > 100:
        result["fan_rpm_low"] = round(max(v * 0.90, 500), 0)
        logged.append(f"fan_rpm<{result['fan_rpm_low']:.0f}")

    v_cpu_p99 = pct("compositetemperature", "p99") or pct("temperaturereading", "p99")
    if v_cpu_p99 is not None:
        result["fan_temp_crosscheck"] = round(v_cpu_p99 * 0.80, 1)
        logged.append(f"fan_temp_xchk>{result['fan_temp_crosscheck']:.1f}°C")

    if platform == "h100":
        v = pct("powerconsumption", "p99")
        if v is not None:
            result["gpu_power_w"] = round(v * 0.70, 1)
            logged.append(f"gpu_pwr>{result['gpu_power_w']:.0f}mW@low_util")

    if logged:
        print(f"  [dyn_thr] {comp}: {', '.join(logged)}")
    else:
        print(f"  [dyn_thr] {comp}: no CSV matches — using config defaults")

    return result
