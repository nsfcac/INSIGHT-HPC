from __future__ import annotations

from typing import List

import pandas as pd

# Parse maintenance windows from config.
def load_maintenance_windows(cfg: dict) -> List[dict]:
    raw     = cfg.get("maintenance_windows") or []
    windows = []
    for entry in raw:
        try:
            start = pd.Timestamp(entry["start"], tz="UTC")
            end   = pd.Timestamp(entry["end"],   tz="UTC")
        except Exception as e:
            print(f"  [WARN] maintenance_windows: bad entry {entry}: {e}")
            continue
        if end <= start:
            print(f"  [WARN] maintenance window end <= start, skipping: {entry}")
            continue
        nodes = set(entry.get("nodes") or [])
        windows.append(dict(start=start, end=end, nodes=nodes,
                            reason=str(entry.get("reason", ""))))
    return windows
