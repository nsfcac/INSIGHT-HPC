from __future__ import annotations

from typing import Optional

import pandas as pd


# Format a sensor peak/baseline line with its source channel.
def format_sensor_line(
    label: str,
    peak: float,
    baseline: Optional[float],
    unit: str,
    decimals: int,
    sources: list[str],
) -> str:
    unit_sp = f" {unit}" if unit else ""
    val_str = f"{peak:.{decimals}f}{unit_sp}"
    parts = [f"**{val_str}**"]
    if baseline is not None and abs(baseline) > 1e-9:
        diff = peak - baseline
        pct = 100.0 * diff / abs(baseline)
        parts.append(
            f"(pre-episode baseline {baseline:.{decimals}f}{unit_sp}, "
            f"Δ {diff:+.{decimals}f}{unit_sp} / {pct:+.0f}%)"
        )
    elif baseline is not None:
        parts.append(f"(pre-episode baseline {baseline:.{decimals}f}{unit_sp})")
    src_str = (
        f"`{sources[0]}`"
        if len(sources) == 1
        else f"`{sources[0]}` (+{len(sources) - 1} more)"
    )
    return f"- {label}: " + " ".join(parts) + f"  _(source: {src_str})_"


# Return the peak (or min) value, its pre-episode baseline, and source column.
def peak_with_baseline(
    df: pd.DataFrame, pre: pd.DataFrame, epi: pd.DataFrame, cols: list[str], agg: str
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    cols = [c for c in cols if c in df.columns]
    if not cols or len(epi) == 0:
        return None, None, None
    epi_vals = epi[cols].astype(float)
    per_col = epi_vals.max() if agg == "max" else epi_vals.min()
    if not per_col.notna().any():
        return None, None, None
    best_col = per_col.idxmax() if agg == "max" else per_col.idxmin()
    peak = float(per_col[best_col])
    baseline: Optional[float] = None
    if len(pre) > 0:
        pre_series = pre[best_col].dropna().astype(float)
        if len(pre_series) >= 3:
            baseline = float(pre_series.median())
    return peak, baseline, best_col


# Build the master-table context: per-sensor peaks, baselines, and dark channels.
def build_master_ctx(ep: pd.Series, df: Optional[pd.DataFrame]) -> dict:
    if df is None or len(df) == 0:
        return {}
    from offline.phase4.alert_plots import matching_columns  # local to avoid heavy import

    ep_start = pd.to_datetime(ep.get("episode_start"), utc=True, errors="coerce")
    ep_end = pd.to_datetime(ep.get("episode_end"), utc=True, errors="coerce")
    if "timestamp" in df.columns and pd.notna(ep_start) and pd.notna(ep_end):
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        pre = df[ts < ep_start]
        epi = df[(ts >= ep_start) & (ts <= ep_end)]
    else:
        pre = df.iloc[0:0]
        epi = df

    cpu_cols = [
        c for c in matching_columns(df, "cpu.socket") if "temperaturereading" in c
    ]
    inlet_cols = [c for c in matching_columns(df, "inlet") if "temperaturereading" in c]
    fan_cols = [
        c
        for c in df.columns
        if c.startswith("rpmreading__fan") and c.endswith("_fan_avg")
    ]
    power_cols = [
        "systeminputpower__powermetrics_avg",
        "systempowerconsumption__powermetrics_avg",
    ]

    candidates: list[dict] = []
    peak_power = peak_cpu_temp = peak_inlet_temp = min_fan_rpm = None

    for c in power_cols:
        peak, baseline, src = peak_with_baseline(df, pre, epi, [c], "max")
        if peak is not None:
            row = {
                "label": "System power peak",
                "peak": peak,
                "baseline": baseline,
                "unit": "W",
                "decimals": 0,
                "source": src,
            }
            candidates.append(row)
            if peak_power is None:
                peak_power = row

    cpu_peak, cpu_base, cpu_src = peak_with_baseline(df, pre, epi, cpu_cols, "max")
    if cpu_peak is not None:
        peak_cpu_temp = {
            "label": "CPU temp peak",
            "peak": cpu_peak,
            "baseline": cpu_base,
            "unit": "°C",
            "decimals": 1,
            "source": cpu_src,
        }
        candidates.append(peak_cpu_temp)

    in_peak, in_base, in_src = peak_with_baseline(df, pre, epi, inlet_cols, "max")
    if in_peak is not None:
        peak_inlet_temp = {
            "label": "Inlet temp peak",
            "peak": in_peak,
            "baseline": in_base,
            "unit": "°C",
            "decimals": 1,
            "source": in_src,
        }
        candidates.append(peak_inlet_temp)

    fan_peak, fan_base, fan_src = peak_with_baseline(df, pre, epi, fan_cols, "min")
    if fan_peak is not None:
        min_fan_rpm = {
            "label": "Min fan RPM (any channel)",
            "peak": fan_peak,
            "baseline": fan_base,
            "unit": "",
            "decimals": 0,
            "source": fan_src,
        }
        candidates.append(min_fan_rpm)

    sensor_cols = (
        cpu_cols
        + inlet_cols
        + fan_cols
        + power_cols
        + [
            c
            for c in df.columns
            if c.startswith("temperaturereading")
            and c.endswith("_avg")
            and c not in cpu_cols
            and c not in inlet_cols
        ]
    )
    sensor_cols = list(dict.fromkeys(sensor_cols))[:80]

    flatline_channels: list[str] = []
    if len(epi) >= 5 and len(pre) >= 5:
        for c in sensor_cols:
            if c not in df.columns:
                continue
            try:
                epi_std = float(epi[c].astype(float).std())
                pre_std = float(pre[c].astype(float).std())
            except (ValueError, TypeError):
                continue
            if (
                pd.notna(epi_std)
                and pd.notna(pre_std)
                and epi_std < 1e-6
                and pre_std > 1e-3
            ):
                flatline_channels.append(c)

    dropout_channels: list[str] = []
    if len(epi) > 0:
        for c in sensor_cols:
            if c not in df.columns:
                continue
            try:
                nan_frac = float(epi[c].isna().mean())
            except (ValueError, TypeError):
                continue
            if pd.notna(nan_frac) and nan_frac > 0.5:
                dropout_channels.append(c)

    return {
        "candidates": candidates,
        "peak_power": peak_power,
        "peak_cpu_temp": peak_cpu_temp,
        "peak_inlet_temp": peak_inlet_temp,
        "min_fan_rpm": min_fan_rpm,
        "flatline_channels": flatline_channels[:5],
        "dropout_channels": dropout_channels[:5],
    }
