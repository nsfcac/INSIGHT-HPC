from __future__ import annotations

from typing import List, Optional

import pandas as pd


# Parse configured maintenance windows into normalised start/end/nodes dicts.
def load_maintenance_windows(cfg: dict) -> List[dict]:
    raw = cfg.get("maintenance_windows") or []
    windows = []
    for entry in raw:
        try:
            start = pd.Timestamp(entry["start"], tz="UTC")
            end = pd.Timestamp(entry["end"], tz="UTC")
        except Exception as e:
            print(f"  [WARN] maintenance_windows: bad entry {entry}: {e}")
            continue
        if end <= start:
            print(f"  [WARN] maintenance window end <= start, skipping: {entry}")
            continue
        nodes = set(entry.get("nodes") or [])
        windows.append(
            dict(start=start, end=end, nodes=nodes, reason=str(entry.get("reason", "")))
        )
    return windows


# Flag rows that fall within any maintenance window for their node.
def apply_maintenance_mask(
    df: pd.DataFrame,
    windows: List[dict],
    ts_col: str = "timestamp",
    hostname_col: Optional[str] = "hostname",
) -> pd.DataFrame:
    if "maintenance_flag" not in df.columns:
        df["maintenance_flag"] = False

    if not windows or ts_col not in df.columns:
        return df

    ts = df[ts_col]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is None:
        ts = ts.dt.tz_localize("UTC")

    for w in windows:
        in_window = (ts >= w["start"]) & (ts <= w["end"])
        if not in_window.any():
            continue

        if not w["nodes"] or hostname_col not in df.columns:
            df.loc[in_window, "maintenance_flag"] = True
        else:
            node_match = (
                df[hostname_col].isin(w["nodes"])
                if hostname_col in df.columns
                else pd.Series(True, index=df.index)
            )
            df.loc[in_window & node_match, "maintenance_flag"] = True

    return df
