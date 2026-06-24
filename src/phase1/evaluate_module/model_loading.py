from __future__ import annotations

import json, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.visualization.plot_utils import ensure_dir, save_fig, write_json


# Load the baseline threshold alerts parquet, parsing timestamps.
def load_threshold_alerts(model_dir: Path) -> Optional[pd.DataFrame]:
    p = model_dir / "baseline_threshold" / "alerts.parquet"
    if not p.exists():
        print(f"  [eval] Missing threshold alerts: {p}")
        return None
    df = load_parquet(p)
    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# Concatenate all per-node Isolation Forest score parquets.
def load_if_scores(model_dir: Path) -> Optional[pd.DataFrame]:
    scores_dir = model_dir / "isolation_forest" / "scores"
    if not scores_dir.exists():
        print(f"  [eval] Missing IF scores dir: {scores_dir}")
        return None
    frames = []
    for p in sorted(scores_dir.glob("*.parquet")):
        df = load_parquet(p)
        if df is not None:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# Load the PDU alerts parquet, if present.
def load_pdu_alerts(model_dir: Path) -> Optional[pd.DataFrame]:
    p = model_dir / "baseline_threshold" / "pdu_alerts.parquet"
    if not p.exists():
        return None
    df = load_parquet(p)
    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# Outer-join threshold and IF alerts and derive the both/either flags.
def merge_models(thr_df: pd.DataFrame, if_df: pd.DataFrame) -> pd.DataFrame:
    thr_keep = [
        "timestamp",
        "hostname",
        "component",
        "split",
        "is_running_job",
        "maintenance_flag",
        "is_flagged",
        "flag_count",
        "triggered_rules",
    ] + [c for c in thr_df.columns if c.startswith("rule_")]

    if_keep = ["timestamp", "hostname", "if_anomaly_score", "if_is_anomaly"]

    thr_small = thr_df[[c for c in thr_keep if c in thr_df.columns]].rename(
        columns={"is_flagged": "thr_is_flagged"}
    )
    if_small = if_df[[c for c in if_keep if c in if_df.columns]]

    merged = pd.merge(thr_small, if_small, on=["timestamp", "hostname"], how="outer")
    merged["thr_is_flagged"] = (
        merged.get("thr_is_flagged", pd.Series(False)).fillna(False).astype(bool)
    )
    merged["if_is_anomaly"] = (
        merged.get("if_is_anomaly", pd.Series(False)).fillna(False).astype(bool)
    )
    merged["both_flag"] = merged["thr_is_flagged"] & merged["if_is_anomaly"]
    merged["either_flag"] = merged["thr_is_flagged"] | merged["if_is_anomaly"]
    return merged


# Compute per-subset and per-component alert and agreement rates.
def compute_alert_rates(merged: pd.DataFrame) -> dict:
    rates: dict = {}
    for subset_name, subset in [
        ("all", merged),
        (
            "loaded",
            merged[merged.get("is_running_job", pd.Series(np.nan)).fillna(0) > 0],
        ),
        (
            "idle",
            merged[merged.get("is_running_job", pd.Series(np.nan)).fillna(1) == 0],
        ),
        ("train", merged[merged.get("split", pd.Series("val")) == "train"]),
        (
            "val_test",
            merged[merged.get("split", pd.Series("train")).isin(["val", "test"])],
        ),
    ]:
        n = max(len(subset), 1)
        rates[subset_name] = {
            "threshold_alert_rate": float(subset["thr_is_flagged"].sum() / n),
            "if_alert_rate": float(subset["if_is_anomaly"].sum() / n),
            "agreement_rate": float(subset["both_flag"].sum() / n),
            "either_rate": float(subset["either_flag"].sum() / n),
            "n_rows": int(n),
        }

    if "component" in merged.columns:
        for comp in merged["component"].dropna().unique():
            sub = merged[merged["component"] == comp]
            n = max(len(sub), 1)
            rates[f"component_{comp}"] = {
                "threshold_alert_rate": float(sub["thr_is_flagged"].sum() / n),
                "if_alert_rate": float(sub["if_is_anomaly"].sum() / n),
                "n_rows": int(n),
            }
    return rates


# Plot a grouped bar chart of threshold vs IF alert rates by subset.
def plot_alert_rates(rates: dict, out: Path, dpi: int = 140) -> None:
    subsets = ["all", "loaded", "idle", "train", "val_test"]
    thr_vals = [rates[s]["threshold_alert_rate"] * 100 for s in subsets if s in rates]
    if_vals = [rates[s]["if_alert_rate"] * 100 for s in subsets if s in rates]
    labels = [s for s in subsets if s in rates]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - w / 2,
        thr_vals,
        w,
        label="Threshold baseline",
        color="steelblue",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2, if_vals, w, label="Isolation Forest", color="darkorange", alpha=0.85
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Alert rate (%)")
    ax.set_title(
        "Phase I — Alert rate comparison by subset", fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.yaxis.grid(True, alpha=0.4)
    save_fig(fig, out, dpi=dpi)


# Plot IF score histograms for loaded/idle and train/val+test.
def plot_score_distributions(merged: pd.DataFrame, out: Path, dpi: int = 140) -> None:
    if "if_anomaly_score" not in merged.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    loaded = merged[merged.get("is_running_job", pd.Series(1.0)).fillna(0) > 0][
        "if_anomaly_score"
    ].dropna()
    idle = merged[merged.get("is_running_job", pd.Series(1.0)).fillna(1) == 0][
        "if_anomaly_score"
    ].dropna()
    bins = np.linspace(0, 1, 51)
    if len(loaded):
        ax.hist(
            loaded,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            label=f"Loaded (n={len(loaded):,})",
        )
    if len(idle):
        ax.hist(
            idle,
            bins=bins,
            density=True,
            alpha=0.7,
            color="slategray",
            label=f"Idle   (n={len(idle):,})",
        )
    ax.axvline(
        loaded.quantile(0.99) if len(loaded) else 0.9,
        color="steelblue",
        lw=1.5,
        ls="--",
        label="Loaded p99",
    )
    ax.set_xlabel("IF anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: loaded vs idle")
    ax.legend(fontsize=9)

    ax = axes[1]
    if "split" in merged.columns:
        train = merged[merged["split"] == "train"]["if_anomaly_score"].dropna()
        valtest = merged[merged["split"].isin(["val", "test"])][
            "if_anomaly_score"
        ].dropna()
        if len(train):
            ax.hist(
                train,
                bins=bins,
                density=True,
                alpha=0.7,
                color="mediumseagreen",
                label=f"Train   (n={len(train):,})",
            )
        if len(valtest):
            ax.hist(
                valtest,
                bins=bins,
                density=True,
                alpha=0.7,
                color="tomato",
                label=f"Val+Test (n={len(valtest):,})",
            )
    ax.set_xlabel("IF anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: train vs val+test")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Isolation Forest — Anomaly Score Distributions", fontsize=12, fontweight="bold"
    )
    save_fig(fig, out, dpi=dpi)


# Compute the both / threshold-only / IF-only / neither agreement breakdown.
def compute_agreement(merged: pd.DataFrame) -> dict:
    thr = merged["thr_is_flagged"].fillna(False).astype(bool)
    iff = merged["if_is_anomaly"].fillna(False).astype(bool)

    both = int((thr & iff).sum())
    thr_only = int((thr & ~iff).sum())
    if_only = int((~thr & iff).sum())
    neither = int((~thr & ~iff).sum())
    total = max(len(merged), 1)

    return {
        "both_flagged": both,
        "both_pct": 100 * both / total,
        "threshold_only": thr_only,
        "threshold_pct": 100 * thr_only / total,
        "iforest_only": if_only,
        "iforest_pct": 100 * if_only / total,
        "neither": neither,
        "neither_pct": 100 * neither / total,
        "total_rows": int(total),
        "agreement_of_flagged": 100 * both / max(both + thr_only + if_only, 1),
    }


# Plot the model-agreement pie and absolute-count bar charts.
def plot_agreement(agreement: dict, out: Path, dpi: int = 140) -> None:
    labels = ["Both flagged", "Threshold only", "IF only", "Neither"]
    values = [
        agreement["both_pct"],
        agreement["threshold_pct"],
        agreement["iforest_pct"],
        agreement["neither_pct"],
    ]
    colors = ["crimson", "steelblue", "darkorange", "lightgray"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    display_vals = list(values[:3])
    display_labels = labels[:3]
    display_colors = colors[:3]
    ax.pie(
        display_vals,
        labels=display_labels,
        colors=display_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.7,
    )
    ax.set_title(
        f"Alert attribution\n(excl. neither={agreement['neither_pct']:.1f}%)",
        fontsize=11,
    )

    ax = axes[1]
    x = np.arange(len(labels))
    counts = [
        agreement["both_flagged"],
        agreement["threshold_only"],
        agreement["iforest_only"],
        agreement["neither"],
    ]
    bars = ax.bar(x, counts, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Row count")
    ax.set_title("Agreement matrix (absolute counts)", fontsize=11)
    ax.yaxis.grid(True, alpha=0.4)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{cnt:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"Model Agreement  |  Agree-on-flagged: "
        f"{agreement['agreement_of_flagged']:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, out, dpi=dpi)


# Plot rolling alert rates over time for each detector.
def plot_temporal_alert_rate(
    merged: pd.DataFrame, window_min: int, out: Path, dpi: int = 140
) -> None:
    if "timestamp" not in merged.columns:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    ts_col = "timestamp"
    for ax, (flag_col, label, color) in zip(
        axes,
        [
            ("thr_is_flagged", "Threshold baseline", "steelblue"),
            ("if_is_anomaly", "Isolation Forest", "darkorange"),
            ("both_flag", "Both (high confidence)", "crimson"),
        ],
    ):
        if flag_col not in merged.columns:
            continue
        tmp = merged[[ts_col, flag_col]].copy()
        tmp[flag_col] = tmp[flag_col].fillna(False).astype(float)
        tmp = tmp.set_index(ts_col).resample("1min")[flag_col].mean()
        rolled = tmp.rolling(
            window=window_min, min_periods=max(1, window_min // 4)
        ).mean()

        ax.fill_between(rolled.index, 0, rolled.values * 100, color=color, alpha=0.4)
        ax.plot(rolled.index, rolled.values * 100, color=color, lw=0.8)
        ax.set_ylabel(f"Alert rate (%)\n{window_min}-min rolling")
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.yaxis.grid(True, alpha=0.35)

    fig.suptitle(
        "Temporal Alert Rate — Rolling window analysis", fontsize=12, fontweight="bold"
    )
    save_fig(fig, out, dpi=dpi)


# Rank time windows where many nodes are simultaneously flagged.
def find_suspicious_periods(
    merged: pd.DataFrame,
    window_min: int = 60,
    multi_node_thr: float = 0.25,
    top_n: int = 20,
) -> pd.DataFrame:
    if "timestamp" not in merged.columns or "hostname" not in merged.columns:
        return pd.DataFrame()

    ts_col = "timestamp"
    total_nodes = merged["hostname"].nunique()
    if total_nodes == 0:
        return pd.DataFrame()

    per_min = (
        merged[[ts_col, "hostname", "either_flag"]]
        .assign(either_flag=lambda d: d["either_flag"].fillna(False).astype(float))
        .groupby(ts_col)
        .agg(
            nodes_flagged=("either_flag", "sum"), nodes_present=("hostname", "nunique")
        )
        .reset_index()
    )
    per_min["frac_flagged"] = per_min["nodes_flagged"] / per_min["nodes_present"].clip(
        lower=1
    )
    per_min = per_min.sort_values(ts_col).set_index(ts_col)

    rolled = per_min["frac_flagged"].rolling(f"{window_min}min", min_periods=1).mean()

    above = rolled[rolled >= multi_node_thr]
    if above.empty:
        above = rolled.nlargest(top_n)

    events = []
    prev_t = None
    cur_peak = 0.0
    cur_peak_t = None

    for t, val in rolled.items():
        if prev_t is None or (t - prev_t).total_seconds() > window_min * 60:
            if prev_t is not None and cur_peak >= multi_node_thr:
                events.append({"window_end": cur_peak_t, "peak_frac": cur_peak})
            cur_peak = 0.0
            cur_peak_t = t
        if val > cur_peak:
            cur_peak = val
            cur_peak_t = t
        prev_t = t

    if cur_peak >= multi_node_thr:
        events.append({"window_end": cur_peak_t, "peak_frac": cur_peak})

    if not events:
        top_rows = rolled.nlargest(top_n).reset_index()
        top_rows.columns = ["window_end", "peak_frac"]
        return top_rows.head(top_n)

    result = (
        pd.DataFrame(events)
        .sort_values("peak_frac", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "rank", result.index + 1)
    result["window_start"] = result["window_end"] - pd.Timedelta(minutes=window_min)
    result["peak_frac_pct"] = (result["peak_frac"] * 100).round(1)
    return result[["rank", "window_start", "window_end", "peak_frac_pct"]]


# Build per-node flag-rate and peak-anomaly-day summary rows.
def node_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if "hostname" not in merged.columns:
        return pd.DataFrame()

    agg = []
    for hostname, grp in merged.groupby("hostname"):
        n = len(grp)
        rec = {
            "hostname": hostname,
            "component": grp["component"].iloc[0] if "component" in grp.columns else "",
            "total_rows": n,
            "thr_flag_rate": float(grp["thr_is_flagged"].fillna(False).mean()),
            "if_flag_rate": float(grp["if_is_anomaly"].fillna(False).mean()),
            "both_flag_rate": float(grp["both_flag"].fillna(False).mean()),
        }
        if "if_anomaly_score" in grp.columns:
            rec["if_score_p95"] = (
                float(grp["if_anomaly_score"].dropna().quantile(0.95))
                if len(grp["if_anomaly_score"].dropna()) > 0
                else 0.0
            )
            rec["if_score_max"] = float(grp["if_anomaly_score"].max())

        if "timestamp" in grp.columns and grp["both_flag"].any():
            anomaly_days = (
                grp[grp["both_flag"].fillna(False)]
                .groupby(grp["timestamp"].dt.date)["hostname"]
                .count()
            )
            if not anomaly_days.empty:
                rec["busiest_anomaly_day"] = str(anomaly_days.idxmax())
                rec["peak_day_count"] = int(anomaly_days.max())

        agg.append(rec)

    df = pd.DataFrame(agg)
    if not df.empty:
        df = df.sort_values("both_flag_rate", ascending=False).reset_index(drop=True)
    return df


# Compute the lag-k autocorrelation of a series.
def autocorrelation(series: pd.Series, lag: int = 1) -> float:
    n = len(series)
    if n <= lag:
        return 0.0
    x = series.iloc[:-lag].values.astype(float)
    y = series.iloc[lag:].values.astype(float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# Measure temporal clustering of alerts via lag-1 autocorrelation.
def compute_clustering(merged: pd.DataFrame) -> dict:
    result = {}
    ts_col = "timestamp"
    if ts_col not in merged.columns:
        return result

    for flag_col, label in [
        ("thr_is_flagged", "threshold"),
        ("if_is_anomaly", "iforest"),
        ("both_flag", "both"),
    ]:
        if flag_col not in merged.columns:
            continue
        per_min = (
            merged[[ts_col, flag_col]]
            .assign(**{flag_col: merged[flag_col].fillna(False).astype(float)})
            .set_index(ts_col)
            .resample("1min")[flag_col]
            .mean()
            .fillna(0)
        )
        ac = autocorrelation(per_min, lag=1)
        result[label] = {
            "autocorr_lag1": round(ac, 4),
            "clustered": ac > 0.4,
        }
    return result


# Write the human-readable evaluation README.
def write_readme(results: dict, out_dir: Path) -> None:
    rates = results.get("alert_rates", {}).get("all", {})
    agr = results.get("agreement", {})
    clust = results.get("clustering", {})

    lines = [
        "# Phase I Evaluation Report",
        "",
        "Unsupervised comparison of the threshold-only baseline and the",
        "Isolation Forest telemetry anomaly model.",
        "",
        "## Alert Rates (all rows)",
        f"- Threshold baseline: **{rates.get('threshold_alert_rate', 0)*100:.2f}%**",
        f"- Isolation Forest: **{rates.get('if_alert_rate', 0)*100:.2f}%**",
        f"- Both models agree: **{rates.get('agreement_rate', 0)*100:.2f}%**",
        "",
        "## Model Agreement",
        f"- Both flagged: {agr.get('both_pct', 0):.2f}%",
        f"- Threshold only: {agr.get('threshold_pct', 0):.2f}%",
        f"- IF only: {agr.get('iforest_pct', 0):.2f}%",
        f"- Agreement rate (of all flagged): {agr.get('agreement_of_flagged', 0):.1f}%",
        "",
        "## Temporal Clustering",
    ]
    for model, info in clust.items():
        icon = "Clustered" if info.get("clustered") else "Scattered"
        lines.append(f"- {model}: AC={info['autocorr_lag1']:.3f}  {icon}")

    lines += [
        "",
        "## Interpretation",
        "- Agreement rate > 50% of flagged rows → models capture overlapping signals.",
        "- Temporal clustering (AC > 0.4) → alerts correspond to real sustained events.",
        "- Loaded alert rate >> idle alert rate → anomalies are load-correlated (expected).",
        "- High val/test rate vs train → possible distribution shift in later data.",
        "",
        "## Files",
        "- `alert_rates.png`            — grouped bar chart by subset",
        "- `score_distributions.png`    — IF score histograms",
        "- `temporal_alert_rate.png`    — rolling alert rate over time",
        "- `model_agreement.png`        — agreement matrix",
        "- `node_summary.parquet`       — per-node metrics",
        "- `top_suspicious_periods.csv` — ranked candidate incident windows",
        "- `summary.json`               — full machine-readable results",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))
