from __future__ import annotations

import gc, json, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.utils.maintenance import load_maintenance_windows, apply_maintenance_mask

EXCLUDE = [
    "audit_",
    "split",
    "hostname",
    "component",
    "timestamp",
    "primary_job",
    "jobs_json",
    "cpu_shares",
    "active_job",
    "node_busy",
    "node_idle",
    "is_running",
    "is_multi",
    "primary_job_changed",
    "active_job_set",
    "pdu__rack_audit",
]


# Pick numeric "_avg" feature columns, excluding context and metadata columns.
def select_features(columns: list) -> list:
    return [
        c
        for c in columns
        if c.endswith("_avg") and not any(ex in c.lower() for ex in EXCLUDE)
    ]


# Flag a node where several features sustain a large absolute z-score deviation.
def zscore_node(
    df: pd.DataFrame, hostname: str, threshold: float
) -> Optional[pd.DataFrame]:
    ts_col = "timestamp"
    feat_cols = select_features(list(df.columns))
    if not feat_cols or ts_col not in df.columns or "split" not in df.columns:
        return None

    train = df[df["split"].isin(["train", "val"])].copy()
    if "audit_any" in train.columns:
        train = train[~train["audit_any"].eq(True)]
    if "maintenance_flag" in train.columns:
        train = train[~train["maintenance_flag"].fillna(False)]

    stats = {}
    for col in feat_cols:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if len(s) < 30:
            continue
        mu, sigma = float(s.mean()), float(s.std())
        if sigma < 1e-6:
            continue
        stats[col] = (mu, sigma)

    if not stats:
        return None

    n_exceeded = pd.Series(0, index=df.index, dtype="int32")
    max_z = pd.Series(0.0, index=df.index, dtype="float32")
    for col, (mu, sigma) in stats.items():
        z = (pd.to_numeric(df[col], errors="coerce").fillna(mu) - mu).abs() / sigma
        n_exceeded = n_exceeded + (z > threshold).astype("int32")
        max_z = np.maximum(max_z, z.fillna(0).astype("float32"))

    strong_thresh = max(threshold + 1.5, 4.5)
    is_flagged = (n_exceeded >= 2) & (max_z > strong_thresh)

    if is_flagged.any():
        run_id = (is_flagged != is_flagged.shift()).cumsum()
        run_len = is_flagged.groupby(run_id).transform("sum")
        is_flagged = (is_flagged & (run_len >= 5)).astype(bool)

    return pd.DataFrame(
        {
            "timestamp": df[ts_col].values,
            "hostname": hostname,
            "split": df["split"].values if "split" in df.columns else "unknown",
            "is_flagged": is_flagged.values,
            "max_z": max_z.values,
            "detector": "zscore",
        }
    )


# Flag a node where several features sustain a large deviation from their EWMA.
def ewma_node(
    df: pd.DataFrame, hostname: str, span: int = 30, k: float = 3.0
) -> Optional[pd.DataFrame]:
    ts_col = "timestamp"
    feat_cols = select_features(list(df.columns))
    if not feat_cols or ts_col not in df.columns or "split" not in df.columns:
        return None

    train = df[df["split"].isin(["train", "val"])].copy()
    if "audit_any" in train.columns:
        train = train[~train["audit_any"].eq(True)]
    if "maintenance_flag" in train.columns:
        train = train[~train["maintenance_flag"].fillna(False)]

    train_sigma = {}
    for col in feat_cols:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if len(s) < 30:
            continue
        sigma = float(s.std())
        if sigma < 1e-6:
            continue
        train_sigma[col] = sigma

    if not train_sigma:
        return None

    n_exceeded = pd.Series(0, index=df.index, dtype="int32")
    max_dev = pd.Series(0.0, index=df.index, dtype="float32")
    for col, sigma in train_sigma.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        ewma = vals.ewm(span=span, min_periods=1).mean()
        dev = (vals - ewma).abs() / sigma
        n_exceeded = n_exceeded + (dev > k).astype("int32")
        max_dev = np.maximum(max_dev, dev.fillna(0).astype("float32"))

    strong_k = max(k + 1.5, 4.5)
    is_flagged = (n_exceeded >= 2) & (max_dev > strong_k)

    if is_flagged.any():
        run_id = (is_flagged != is_flagged.shift()).cumsum()
        run_len = is_flagged.groupby(run_id).transform("sum")
        is_flagged = (is_flagged & (run_len >= 5)).astype(bool)

    return pd.DataFrame(
        {
            "timestamp": df[ts_col].values,
            "hostname": hostname,
            "split": df["split"].values if "split" in df.columns else "unknown",
            "is_flagged": is_flagged.values,
            "detector": "ewma",
        }
    )


# Run the z-score and EWMA baseline detectors over all nodes and summarise flag rates.
def run_baseline_comparison(
    zscore_threshold: float = 4.0,
    ewma_span: int = 30,
    ewma_k: float = 3.0,
    force: bool = False,
) -> dict:
    cfg = load_config()
    master_dir = Path(cfg["paths"]["master"])

    phase1_dir = Path(cfg["phase1"]["output_dir"])
    rep_dir = Path(cfg["phase1"]["evaluation"]["output_dir"])
    out_dir = phase1_dir / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    out_json = rep_dir / "baseline_comparison.json"
    if (
        (out_dir / "zscore_alerts.parquet").exists()
        and (out_dir / "ewma_alerts.parquet").exists()
        and not force
    ):
        print("[baseline] Outputs exist — loading")
        return json.loads(out_json.read_text()) if out_json.exists() else {}

    maint_windows = load_maintenance_windows(cfg)
    t0 = time.perf_counter()
    print(
        f"\n[baseline] z-score (thr={zscore_threshold})  "
        f"EWMA (span={ewma_span}, k={ewma_k})"
    )

    frames = {"zscore": [], "ewma": []}
    summary_rows = []

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue

        parquets = sorted(comp_dir.glob("*.parquet"))
        print(f"\n  [{comp.upper()}]  {len(parquets)} nodes")

        for p in parquets:
            hostname = p.stem
            df = load_parquet(p)
            if df is None or df.empty:
                continue
            apply_maintenance_mask(df, maint_windows, "timestamp", "hostname")

            for det_name, fn, kwargs in [
                ("zscore", zscore_node, {"threshold": zscore_threshold}),
                ("ewma", ewma_node, {"span": ewma_span, "k": ewma_k}),
            ]:
                result = fn(df, hostname, **kwargs)
                if result is not None:
                    n_flag = int(result["is_flagged"].sum())
                    frames[det_name].append(result)
                    summary_rows.append(
                        {
                            "hostname": hostname,
                            "component": comp,
                            "detector": det_name,
                            "n_rows": len(result),
                            "n_flagged": n_flag,
                            "flag_rate_pct": round(
                                100 * n_flag / max(len(result), 1), 3
                            ),
                        }
                    )

            del df
            gc.collect()

    agg = {}
    for det_name, frame_list in frames.items():
        if not frame_list:
            continue
        combined = pd.concat(frame_list, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        path = out_dir / f"{det_name}_alerts.parquet"
        save_parquet(combined, path)
        rate = 100 * combined["is_flagged"].mean()
        print(
            f"\n  {det_name:8s}: {len(combined):>10,} rows  "
            f"flagged={combined['is_flagged'].sum():>8,}  "
            f"rate={rate:.2f}%"
        )
        agg[det_name] = {
            "total_rows": int(len(combined)),
            "total_flagged": int(combined["is_flagged"].sum()),
            "flag_rate_pct": round(float(rate), 3),
        }

    summary_df = pd.DataFrame(summary_rows)
    for det in agg:
        sub = summary_df[summary_df["detector"] == det]
        agg[det]["per_component"] = (
            sub.groupby("component")["flag_rate_pct"].mean().round(3).to_dict()
        )

    out_json.write_text(json.dumps(agg, indent=2, default=str))
    print(f"\n[baseline] Done in {time.perf_counter()-t0:.1f}s  → {out_json}")
    return agg


if __name__ == "__main__":
    run_baseline_comparison(force=True)
