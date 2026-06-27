from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_GT_PATH = Path("offline/data/ground_truth/events.parquet")


# Load and normalise ground-truth events from the parquet at the given path.
@lru_cache(maxsize=1)
def load_events(gt_path_str: str) -> Optional[pd.DataFrame]:
    p = Path(gt_path_str)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return None
    if df.empty or "event_start" not in df.columns:
        return None
    df = df.copy()
    df["event_start"] = pd.to_datetime(df["event_start"], utc=True, errors="coerce")
    df["event_end"] = pd.to_datetime(df["event_end"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_start", "event_end"])
    return df[["hostname", "event_start", "event_end"]].reset_index(drop=True)


# Mark rows whose timestamp falls inside any ground-truth event window for their host.
def in_gt_window(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    hostname_col: str = "hostname",
    gt_path: Path = DEFAULT_GT_PATH,
) -> pd.Series:
    out = pd.Series(False, index=df.index)
    events = load_events(str(gt_path))
    if (
        events is None
        or events.empty
        or timestamp_col not in df.columns
        or hostname_col not in df.columns
    ):
        return out

    ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    hosts = df[hostname_col].astype(str)

    cluster = events[events["hostname"] == "*"]
    per_host = events[events["hostname"] != "*"]

    for _, ev in cluster.iterrows():
        out |= (ts >= ev["event_start"]) & (ts <= ev["event_end"])

    for host, grp in per_host.groupby("hostname"):
        host_mask = hosts.eq(host)
        if not host_mask.any():
            continue
        for _, ev in grp.iterrows():
            out |= host_mask & (ts >= ev["event_start"]) & (ts <= ev["event_end"])

    return out
