from __future__ import annotations

from offline.phase4.score_fusion_module.constants import *
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from offline.phase4.fusion_gbdt import build_fused_index, window_slice
from shared.utils.io_utils import load_parquet


# Resolve and create the phase-4 output, fused-input, and reports paths.
def resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    p4_out_dir = Path(cfg["phase4"]["output_dir"])
    p4_out_dir.mkdir(parents=True, exist_ok=True)
    fused_in = p4_out_dir / "fused_alerts.parquet"

    reports_dir = Path(cfg["phase4"].get("reports_dir", "offline/reports/phase4_eval"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    return p4_out_dir, fused_in, reports_dir


# Load ground-truth events, deriving the train/val/test split if absent.
def load_gt_events(cfg: dict) -> pd.DataFrame:
    gt_path = (
        Path(cfg["paths"].get("ground_truth", "offline/data/ground_truth")) / "events.parquet"
    )
    if not gt_path.exists():
        print(
            f"[fusion_v2] GT events not found at {gt_path} — episodes will be unlabeled"
        )
        return pd.DataFrame(
            columns=[
                "hostname",
                "event_start",
                "event_end",
                "category",
                "severity",
                "split",
            ]
        )
    gt = load_parquet(gt_path)
    if "split" not in gt.columns:
        train_end_raw = cfg.get("window", {}).get("train_end") or cfg.get(
            "data", {}
        ).get("train_end", "2026-02-28 23:59:59")
        val_end_raw = cfg.get("window", {}).get("val_end") or cfg.get("data", {}).get(
            "val_end", "2026-03-08 23:59:59"
        )
        train_end = pd.to_datetime(train_end_raw, utc=True, errors="coerce")
        val_end = pd.to_datetime(val_end_raw, utc=True, errors="coerce")
        ts = pd.to_datetime(gt["event_start"], utc=True, errors="coerce")
        gt = gt.copy()
        gt["split"] = np.where(
            ts <= train_end, "train", np.where(ts <= val_end, "val", "test")
        )
        print(
            f"[fusion_v2] Derived GT split from event_start "
            f"(train_end={train_end.date()}, val_end={val_end.date()}): "
            f"{dict(gt['split'].value_counts())}"
        )
    return gt


# Determine the dominant split (train/val/test) covering each episode.
def split_of_episode(
    fused: pd.DataFrame, episodes: pd.DataFrame, fused_index: Optional[dict] = None
) -> pd.Series:
    if "split" not in fused.columns or len(episodes) == 0:
        return pd.Series(["test"] * len(episodes), index=episodes.index)

    if fused_index is None:
        fused_index = build_fused_index(fused)
    sorted_fused = fused_index["sorted"]
    split_arr = sorted_fused["split"].to_numpy()

    out = []
    for _, ep in episodes.iterrows():
        lo, hi = window_slice(
            fused_index,
            ep["hostname"],
            ep["component"],
            ep["episode_start"],
            ep["episode_end"],
        )
        if hi <= lo:
            out.append("test")
            continue
        window = split_arr[lo:hi]
        # Mode via unique+argmax — faster than pd.Series.mode() on small arrays.
        vals, counts = np.unique(window, return_counts=True)
        out.append(str(vals[int(np.argmax(counts))]))
    return pd.Series(out, index=episodes.index)


# Label fused rows positive when inside a GT event window (with lead/lag padding).
def row_level_labels(
    fused: pd.DataFrame, gt_events: pd.DataFrame, lead_min: int = 60, lag_min: int = 30
) -> pd.Series:
    if len(fused) == 0 or gt_events is None or len(gt_events) == 0:
        return pd.Series([0] * len(fused), index=fused.index, dtype="int8")

    f = fused[["hostname", "component", "timestamp"]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)

    gt = gt_events.copy()
    gt["event_start"] = pd.to_datetime(gt["event_start"], utc=True) - pd.Timedelta(
        minutes=lead_min
    )
    gt["event_end"] = pd.to_datetime(gt["event_end"], utc=True) + pd.Timedelta(
        minutes=lag_min
    )

    labels = pd.Series(0, index=f.index, dtype="int8")
    wildcard = gt[gt["hostname"].fillna("*") == "*"]
    for _, g in gt.iterrows():
        host = g.get("hostname") or "*"
        if host == "*":
            continue
        m = (
            (f["hostname"] == host)
            & (f["timestamp"] >= g["event_start"])
            & (f["timestamp"] <= g["event_end"])
        )
        labels.loc[m] = 1
    for _, g in wildcard.iterrows():
        m = (f["timestamp"] >= g["event_start"]) & (f["timestamp"] <= g["event_end"])
        labels.loc[m] = 1
    return labels
