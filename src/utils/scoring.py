from __future__ import annotations

import numpy as np
import pandas as pd


# Robust z-score of a series using the train-set median and MAD.
def robust_z(s: pd.Series, train_mask: pd.Series) -> pd.Series:
    if train_mask.sum() < 30:
        return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
    vals = pd.to_numeric(s, errors="coerce")
    train_vals = vals[train_mask].dropna()
    if len(train_vals) < 30:
        return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
    med = float(train_vals.median())
    mad = float((train_vals - med).abs().median())
    scale = 1.4826 * mad if mad > 0 else train_vals.std() or 1.0
    if scale <= 0:
        return pd.Series(np.zeros(len(s), dtype=np.float32), index=s.index)
    return ((vals.fillna(med) - med) / scale).astype(np.float32)


# Count consecutive True values ending at each position.
def consecutive_true_run(s: pd.Series) -> pd.Series:
    arr = s.astype(bool).to_numpy()
    out = np.zeros_like(arr, dtype=np.int32)
    run = 0
    for i in range(len(arr)):
        run = run + 1 if arr[i] else 0
        out[i] = run
    return pd.Series(out, index=s.index, dtype=np.int32)


# Return a column as float32, filling missing or absent values with a default.
def coerce_numeric(fused: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in fused.columns:
        return pd.Series(
            np.full(len(fused), default, dtype=np.float32), index=fused.index
        )
    return pd.to_numeric(fused[col], errors="coerce").fillna(default).astype(np.float32)
