from __future__ import annotations

from src.phase3.physics_constraints_module.constants import *
import os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit
from src.utils.rack_topology import rack_id as get_rack_id

from src.phase3.physics_constraints_module.node_checks import *


# C2: flag minutes where idle nodes warm up while rack PDU power stays stable.
def check_rack_thermal_mismatch(
    rack_node_dfs: dict,
    pdu_series: Optional[pd.Series],
    rack_id: int,
    min_node_frac: float = 0.6,
    pdu_stable_pct: float = 0.10,
    temp_rise_min: float = 1.5,
    window_min: int = 15,
    seg_idx: Optional[dict] = None,
) -> pd.Series:
    if not rack_node_dfs:
        return pd.Series(dtype=bool)

    # Mark timestamps where a host is inside an active job segment.
    def vec_active_mask(hostname: str, ts_idx) -> np.ndarray:
        n = len(ts_idx)
        if seg_idx is None:
            return np.zeros(n, dtype=bool)
        segs = seg_idx.get(hostname, [])
        if not segs:
            return np.zeros(n, dtype=bool)
        starts = (
            pd.to_datetime([s["seg_start"] for s in segs], utc=True)
            .tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        ends = (
            pd.to_datetime([s["seg_end"] for s in segs], utc=True)
            .tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        order = np.argsort(starts)
        starts_s = starts[order]
        ends_s = ends[order]
        max_ends = np.maximum.accumulate(ends_s)
        ts_dt = pd.to_datetime(ts_idx, utc=True)
        if isinstance(ts_dt, pd.Series):
            ts_ns = ts_dt.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        else:
            ts_ns = ts_dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        k = np.searchsorted(starts_s, ts_ns, side="right") - 1
        active = np.zeros(n, dtype=bool)
        valid = k >= 0
        active[valid] = max_ends[k[valid]] >= ts_ns[valid]
        return active

    # Collect per-node temp rising signal — only for minutes when the node is idle
    rising_flags = {}
    for hostname, df in rack_node_dfs.items():
        cols = list(df.columns)
        temp_col = inlet_col(cols)
        if not temp_col or TS not in df.columns:
            continue
        temp = pd.to_numeric(df[temp_col], errors="coerce")
        ts_idx = pd.DatetimeIndex(pd.to_datetime(df[TS], utc=True))
        temp.index = ts_idx
        rising = (temp.diff(window_min) > temp_rise_min).fillna(False)
        # Mask out minutes when this node is running a job
        idle = ~vec_active_mask(hostname, ts_idx)
        rising = rising & pd.Series(idle, index=ts_idx)
        rising_flags[hostname] = rising

    if not rising_flags:
        return pd.Series(dtype=bool)

    # Align on common timestamps and compute fraction of rising nodes
    ts_index = union_ts_index([s.index for s in rising_flags.values()])
    frac_df = pd.DataFrame(
        {h: s.reindex(ts_index, fill_value=False) for h, s in rising_flags.items()}
    )
    node_frac = frac_df.mean(axis=1)

    # PDU stability check
    pdu_stable = pd.Series(True, index=ts_index)
    if pdu_series is not None:
        # merge_asof replaces deprecated reindex(method="nearest", tolerance=)
        target_frame = pd.DataFrame({TS: to_ns(ts_index)})
        source_frame = pdu_series.rename("_pdu_val").reset_index()
        source_frame.columns = [TS, "_pdu_val"]
        source_frame[TS] = to_ns(source_frame[TS])
        pdu_aligned = pd.merge_asof(
            target_frame.sort_values(TS),
            source_frame.sort_values(TS),
            on=TS,
            direction="nearest",
            tolerance=pd.Timedelta(minutes=5),
        ).set_index(TS)["_pdu_val"]
        pdu_prev = pdu_aligned.shift(window_min)
        pdu_change = (pdu_aligned - pdu_prev).abs() / pdu_prev.replace(0, np.nan)
        pdu_stable = pdu_change < pdu_stable_pct

    return ((node_frac >= min_node_frac) & pdu_stable).fillna(False)


# Index non-idle job segments by hostname for fast active-job lookups.
def load_segments_index(seg_path: Path) -> dict:
    from collections import defaultdict

    if not seg_path.exists():
        return {}
    segs = pd.read_parquet(seg_path, engine="pyarrow")
    idx: dict = defaultdict(list)
    for _, row in segs.iterrows():
        if row.get("is_idle", False):
            continue
        idx[str(row["hostname"])].append(
            {
                "job_id": int(row["job_id"]) if pd.notna(row.get("job_id")) else -1,
                "seg_start": pd.Timestamp(row["seg_start"]).tz_convert("UTC"),
                "seg_end": pd.Timestamp(row["seg_end"]).tz_convert("UTC"),
            }
        )
    return idx


# Return the pipe-joined active job ids for a host at one timestamp.
def active_job_ids_at(hostname: str, ts: pd.Timestamp, seg_idx: dict) -> str:
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    ids = sorted(
        {
            str(s["job_id"])
            for s in seg_idx.get(hostname, [])
            if s["seg_start"] <= ts_utc <= s["seg_end"]
        }
    )
    return "|".join(ids) if ids else ""


# Return active job ids for a host across a timestamp series.
def active_job_ids_vectorized(
    hostname: str, ts_series: pd.Series, seg_idx: dict
) -> pd.Series:
    segs = seg_idx.get(hostname, [])
    n = len(ts_series)
    if not segs or n == 0:
        return pd.Series([""] * n, index=ts_series.index, dtype=object)

    ts_ns = (
        pd.to_datetime(ts_series, utc=True)
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    starts = (
        pd.to_datetime([s["seg_start"] for s in segs], utc=True)
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    ends = (
        pd.to_datetime([s["seg_end"] for s in segs], utc=True)
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    job_ids = np.array([str(s["job_id"]) for s in segs], dtype=object)

    # Sort segments by start time for searchsorted.
    order = np.argsort(starts)
    starts_s = starts[order]
    ends_s = ends[order]
    jobs_s = job_ids[order]

    cand_counts = np.searchsorted(starts_s, ts_ns, side="right")

    out = [""] * n
    for i in range(n):
        k = cand_counts[i]
        if k == 0:
            continue
        mask = ends_s[:k] >= ts_ns[i]
        if not mask.any():
            continue
        ids = sorted(set(jobs_s[:k][mask].tolist()))
        if ids:
            out[i] = "|".join(ids)
    return pd.Series(out, index=ts_series.index, dtype=object)


# Worker entry: apply the persistence filter to one node's results.
def persist_worker(pickled: bytes):
    import pickle

    hostname, df = pickle.loads(pickled)
    return hostname, apply_persistence_filter(df, min_consecutive=PERSIST_MIN_GLOBAL)


PERSIST_MIN_GLOBAL: int = 3


# Worker initializer: set the global persistence-minimum.
def persist_init(min_consecutive: int) -> None:
    global PERSIST_MIN_GLOBAL
    PERSIST_MIN_GLOBAL = min_consecutive


# Apply the persistence filter to all nodes, in parallel when worthwhile.
def apply_persistence_parallel(
    node_results: dict, persist_min: int, n_workers: int
) -> None:
    if n_workers <= 1 or len(node_results) <= 1:
        for hostname in list(node_results.keys()):
            node_results[hostname] = apply_persistence_filter(
                node_results[hostname], min_consecutive=persist_min
            )
        return

    import pickle

    payloads = [pickle.dumps((h, df)) for h, df in node_results.items()]
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=persist_init,
        initargs=(persist_min,),
    ) as ex:
        for fut in as_completed([ex.submit(persist_worker, p) for p in payloads]):
            hostname, filtered = fut.result()
            node_results[hostname] = filtered


WORKER_SEG_IDX: Optional[dict] = None
WORKER_COMP: Optional[str] = None


# Worker initializer: store the segment index, component, and C1/C3 params.
def worker_init(
    seg_idx: dict,
    comp: str,
    c1_params: Optional[dict] = None,
    c3_params: Optional[dict] = None,
) -> None:
    global WORKER_SEG_IDX, WORKER_COMP
    WORKER_SEG_IDX = seg_idx
    WORKER_COMP = comp
    # C1/C3 params are read by check_node in the node_checks module namespace,
    # so set them there rather than rebinding this module's copies.
    from src.phase3.physics_constraints_module import node_checks

    node_checks.WORKER_C1_PARAMS = dict(c1_params or {})
    node_checks.WORKER_C3_PARAMS = dict(c3_params or {})


# Select the minimal column set a node needs for constraint checks.
def constraint_columns(path: Path) -> Optional[list]:
    try:
        import pyarrow.parquet as pq

        names = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None
    if TS not in names:
        return None
    keep = {TS}
    for c in ("active_job_count", "primary_job_id", "maintenance_flag"):
        if c in names:
            keep.add(c)
    for c in names:
        if not c.endswith("_avg"):
            continue
        cl = c.lower()
        if "inlet" in cl:
            keep.add(c)
        if "rpmreading" in cl or "fanspeed" in cl:
            keep.add(c)
        if "systeminputpower" in cl or "systempowerconsumption" in cl:
            keep.add(c)
        if c.startswith("cpuusage__") or c.startswith("gpuusage__"):
            keep.add(c)
    return [c for c in names if c in keep]


# Worker entry: run node-level checks and return results plus slim rack data.
def worker_process_node(parquet_path_str: str):
    p = Path(parquet_path_str)
    hostname = p.stem
    cols = constraint_columns(p)
    df = load_parquet_cols(p, cols) if cols else load_parquet(p)
    if df is None or df.empty:
        return hostname, None, None, None

    seg_idx = WORKER_SEG_IDX
    comp = WORKER_COMP

    result = check_node(df, hostname, comp, seg_idx=seg_idx)
    if not result.empty:
        result["active_job_ids"] = active_job_ids_vectorized(
            hostname, result[TS], seg_idx
        ).values
    else:
        result = None

    rid = get_rack_id(hostname)
    slim_df = None
    if rid is not None:
        cols = list(df.columns)
        temp_col = inlet_col(cols)
        power_col = node_power_col(cols)
        keep = [c for c in [TS, temp_col, power_col] if c is not None]
        if keep:
            slim_df = df[keep].copy()

    return hostname, result, slim_df, rid


# Build a merge-asof source frame from a flagged-timestamp series.
def build_merge_src(series: pd.Series, col_name: str) -> pd.DataFrame:
    src = series.rename(col_name).reset_index()
    src.columns = [TS, col_name]
    src[TS] = to_ns(src[TS])
    return src.sort_values(TS)


# Reindex a rack-level signal onto a node's timestamps.
def align_rack_to_node(rack_signal: pd.Series, node_ts) -> np.ndarray:
    if isinstance(node_ts, pd.Series):
        node_dti = pd.DatetimeIndex(pd.to_datetime(node_ts, utc=True))
    else:
        node_dti = pd.DatetimeIndex(node_ts)
        if node_dti.tz is None:
            node_dti = node_dti.tz_localize("UTC")
    src_idx = rack_signal.index
    if getattr(src_idx, "tz", None) is None:
        src_idx = pd.DatetimeIndex(src_idx).tz_localize("UTC")
    src = pd.Series(rack_signal.values, index=src_idx)
    aligned = src.reindex(node_dti, fill_value=False)
    return aligned.to_numpy(dtype=bool)


# Select the timestamp and PDU power columns from a PDU parquet.
def pdu_columns(path: Path) -> Optional[list]:
    try:
        import pyarrow.parquet as pq

        names = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return None
    if TS not in names:
        return None
    for c in names:
        if c.endswith("_avg") and "pdu" in c.lower():
            return [TS, c]
    return None


# Read selected columns from a parquet, returning None on failure.
def load_parquet_cols(path: Path, columns: list) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path, engine="pyarrow", columns=columns)
    except Exception as e:
        print(f"  [WARN] load_parquet_cols {path}: {e}")
        return None


# Preload per-rack PDU feeder power series and their timestamps.
def preload_pdu_by_rack(pdu_master_dir: Path) -> dict:
    out: dict = {}
    if not pdu_master_dir.exists():
        return out
    # feeder_series[rid][pdu_hostname] = pd.Series(values indexed by ts)
    feeder_series: dict = defaultdict(dict)
    hosts: dict = defaultdict(dict)
    for pf in sorted(pdu_master_dir.glob("*.parquet")):
        rid_pdu = get_rack_id(pf.stem)
        if rid_pdu is None:
            continue
        cols = pdu_columns(pf)
        if not cols:
            continue
        pf_df = load_parquet_cols(pf, cols)
        if pf_df is None or pf_df.empty:
            continue
        ts = pd.to_datetime(pf_df[TS], utc=True)
        val = pd.to_numeric(pf_df[cols[1]], errors="coerce")
        s = pd.Series(val.values, index=ts).sort_index()
        if s.index.has_duplicates:
            s = s.groupby(level=0).sum()
        feeder_series[rid_pdu][pf.stem] = s
        hosts[rid_pdu][pf.stem] = ts
    for rid_pdu, fdict in feeder_series.items():
        out[rid_pdu] = (fdict, hosts[rid_pdu])
    return out
