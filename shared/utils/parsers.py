from __future__ import annotations

import json, warnings
from typing import Any, Optional

import numpy as np
import pandas as pd


def first_existing(columns: list, candidates: list) -> Optional[str]:
    return next((c for c in candidates if c in columns), None)


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def to_int(x: Any) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x)) if not (isinstance(x, str) and not x.strip()) else None
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(x) if not (isinstance(x, str) and not x.strip()) else None
    except Exception:
        return None


# Parse a value (list, JSON, or comma-separated string) into a list via item_cast.
def parse_list(x: Any, item_cast) -> list:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [v for v in (item_cast(i) for i in x) if v is not None]
    if is_missing(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            return parse_list(json.loads(s), item_cast)
        except Exception:
            pass
        if "," in s:
            return [v for v in (item_cast(p) for p in s.split(",")) if v is not None]
    v = item_cast(x)
    return [v] if v is not None else []


def parse_int_list(x: Any) -> list:
    return parse_list(x, to_int)


def parse_float_list(x: Any) -> list:
    return parse_list(x, to_float)


# Parse a value into a list of non-empty strings (handles JSON and Postgres arrays).
def parse_str_list(x: Any) -> list:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    if is_missing(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # Postgres array format: {a,b} or {"a","b"}.
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1]
            return [
                v.strip().strip('"').strip("'") for v in inner.split(",") if v.strip()
            ]
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    str(v).strip() for v in parsed if v is not None and str(v).strip()
                ]
        except Exception:
            pass
        if "," in s:
            return [v.strip() for v in s.split(",") if v.strip()]
        return [s]
    return [str(x).strip()] if x is not None else []


# Coerce a series to UTC microsecond timestamps floored to the minute.
def as_utc_min(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        parsed = pd.to_datetime(series, unit="ns", utc=True)
    elif pd.api.types.is_datetime64_any_dtype(series):
        tz = getattr(series.dtype, "tz", None)
        parsed = (
            series.dt.tz_localize("UTC") if tz is None else series.dt.tz_convert("UTC")
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.astype("datetime64[us, UTC]").dt.floor("60s")


# Parse Slurm timestamps, detecting second vs nanosecond integer units.
def parse_slurm_ts(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        sample = series.dropna().iloc[0] if not series.dropna().empty else 0
        unit = "ns" if int(sample) >= int(1e12) else "s"
        return (
            pd.to_datetime(series, unit=unit, utc=True)
            .astype("datetime64[us, UTC]")
            .dt.floor("60s")
        )
    return as_utc_min(series)


# Ensure a DataFrame's timestamp column is UTC microsecond dtype.
def ensure_utc_us(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        return df
    col = df[ts_col]
    if not (
        pd.api.types.is_datetime64_any_dtype(col)
        and getattr(col.dtype, "tz", None) is not None
    ):
        df = df.copy()
        df[ts_col] = as_utc_min(col)
    elif str(col.dtype) != "datetime64[us, UTC]":
        df = df.copy()
        df[ts_col] = col.astype("datetime64[us, UTC]")
    return df


# Pad or truncate per-job CPU values to match the job list length.
def align_cpu(jobs: list, cpu: list) -> list:
    if not jobs:
        return []
    if not cpu:
        return [np.nan] * len(jobs)
    padded = [float(v) for v in cpu[: len(jobs)]]
    padded += [np.nan] * (len(jobs) - len(padded))
    return padded


# Find the "_avg" column matching all keywords.
def find_avg(cols, *kws):
    kws_l = [k.lower() for k in kws]
    return next(
        (c for c in cols if c.endswith("_avg") and all(k in c.lower() for k in kws_l)),
        None,
    )


# Find the "_avg" column matching any keyword.
def findany(cols, *kws):
    kws_l = [k.lower() for k in kws]
    return next(
        (c for c in cols if c.endswith("_avg") and any(k in c.lower() for k in kws_l)),
        None,
    )


# Convert a timestamp series to int64 nanoseconds (UTC, tz-naive).
def series_to_ns(series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (
        ts.dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
        .astype("int64")
        .to_numpy(copy=False)
    )


# Convert a single timestamp to int64 nanoseconds (UTC).
def timestamp_to_ns(ts) -> int:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return int(out.value)
