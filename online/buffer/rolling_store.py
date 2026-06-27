from __future__ import annotations


# Per-node 7-day warm buffer for incremental feature computation.
class RollingStore:
    def __init__(self, lookback_days: int = 7):
        self.lookback_days = lookback_days

    def append(self, window_df):
        raise NotImplementedError("append new minutes + drop tail")
