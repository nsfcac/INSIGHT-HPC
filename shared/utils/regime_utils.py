from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from shared.utils.maintenance import load_maintenance_windows

REGIME_INSIDE_MAINTENANCE = -1


# Assign an integer regime id to each timestamp based on maintenance windows.
def assign_regime_id(
    timestamps: pd.Series | np.ndarray,
    cfg: Optional[dict] = None,
    windows: Optional[List[dict]] = None,
) -> np.ndarray:
    if windows is None:
        if cfg is None:
            raise ValueError("assign_regime_id needs either cfg or windows")
        windows = load_maintenance_windows(cfg)

    ts = pd.to_datetime(timestamps, utc=True)
    # Strip tz to allow vectorised comparison with tz-naive np.datetime64
    if isinstance(ts, pd.Series):
        ts_arr = ts.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
    else:
        # DatetimeIndex / array-like
        if hasattr(ts, "tz_convert"):
            ts_arr = ts.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        else:
            ts_arr = np.asarray(ts, dtype="datetime64[ns]")

    n = len(ts_arr)
    regime = np.zeros(n, dtype=np.int8)

    if not windows:
        return regime

    # Sort windows by start; the regime index after the i-th window is i+1.
    sorted_win = sorted(windows, key=lambda w: w["start"])

    for i, w in enumerate(sorted_win):
        w_start = np.datetime64(w["start"].tz_convert(None), "ns")
        w_end = np.datetime64(w["end"].tz_convert(None), "ns")
        # After this window's end → bump regime to i+1
        regime = np.where(ts_arr >= w_end, np.int8(i + 1), regime)
        # Inside this window → mark -1 (overrides previous assignment)
        inside = (ts_arr >= w_start) & (ts_arr < w_end)
        regime = np.where(inside, np.int8(REGIME_INSIDE_MAINTENANCE), regime)

    return regime


# Attach regime_id to a DataFrame and exclude maintenance rows from the split.
def attach_regime_id(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
    windows: Optional[List[dict]] = None,
    ts_col: str = "timestamp",
    exclude_split: bool = True,
) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise KeyError(f"DataFrame has no {ts_col} column")

    regime = assign_regime_id(df[ts_col], cfg=cfg, windows=windows)
    df["regime_id"] = regime

    if exclude_split and "split" in df.columns:
        maint_mask = regime == REGIME_INSIDE_MAINTENANCE
        if maint_mask.any():

            split_col = df["split"]
            if (
                isinstance(split_col.dtype, pd.CategoricalDtype)
                and "exclude" not in split_col.cat.categories
            ):
                df["split"] = split_col.cat.add_categories(["exclude"])
            df.loc[maint_mask, "split"] = "exclude"
    return df


# Rolling aggregation computed independently within each contiguous regime block.
def rolling_by_regime(
    series: pd.Series,
    regime_id: pd.Series | np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    op: str = "mean",
    quantile: Optional[float] = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)

    r = np.asarray(regime_id, dtype=np.int64)
    s = pd.Series(series.values, index=series.index)
    out = pd.Series(np.full(len(s), np.nan, dtype=np.float64), index=s.index)

    # Process each contiguous regime block separately
    if len(r) == 0:
        return out
    change = np.concatenate(([True], r[1:] != r[:-1]))
    block_starts = np.where(change)[0]
    block_ends = np.concatenate((block_starts[1:], [len(r)]))

    for b0, b1 in zip(block_starts, block_ends):
        if r[b0] == REGIME_INSIDE_MAINTENANCE:
            continue
        block = s.iloc[b0:b1]
        rolled = block.rolling(window=window, min_periods=min_periods)
        if op == "mean":
            res = rolled.mean()
        elif op == "std":
            res = rolled.std()
        elif op == "quantile":
            if quantile is None:
                raise ValueError(
                    "rolling_by_regime op='quantile' requires `quantile=...`"
                )
            res = rolled.quantile(quantile)
        else:
            raise ValueError(f"Unsupported op: {op}")
        out.iloc[b0:b1] = res.values
    return out


# Apply a slope function within each regime block, skipping maintenance spans.
def regime_safe_slope(
    vals: np.ndarray, regime_id: np.ndarray, window: int, slope_fn
) -> np.ndarray:
    n = len(vals)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return out

    change = np.concatenate(([True], regime_id[1:] != regime_id[:-1]))
    block_starts = np.where(change)[0]
    block_ends = np.concatenate((block_starts[1:], [n]))

    for b0, b1 in zip(block_starts, block_ends):
        if regime_id[b0] == REGIME_INSIDE_MAINTENANCE:
            continue
        block = vals[b0:b1]
        if len(block) < 2:
            continue
        res = slope_fn(block.astype(np.float64), window)
        out[b0:b1] = res
    return out
