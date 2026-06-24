from __future__ import annotations

from src.preprocess.audit_raw_files_module.constants import *
import gc, io, json, os, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io_utils import (
    load_config,
    load_public_tables,
    load_metric_parquet,
    save_parquet,
    build_lookup_dicts,
)


# Flag points that deviate sharply from a per-host rolling mean (z-score spike).
def flag_spikes(
    df: pd.DataFrame,
    col: str,
    z_thresh: float,
    rolling_pts: int,
    min_abs_delta: float = 0.0,
) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int8)
    vals = df[col].to_numpy(dtype=np.float64)
    w = int(rolling_pts)

    for _, idx in df.groupby("hostname", sort=False).indices.items():
        v = vals[idx]
        n = len(v)
        pad = np.pad(v, (w, 0), mode="edge")

        cs = np.empty(n + w + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(pad, out=cs[1:])

        cs2 = np.empty(n + w + 1, dtype=np.float64)
        cs2[0] = 0.0
        np.cumsum(pad**2, out=cs2[1:])

        with np.errstate(invalid="ignore"):
            rm = (cs[w:-1] - cs[: -w - 1]) / w
            rv = np.maximum((cs2[w:-1] - cs2[: -w - 1]) / w - rm**2, 0.0)
            rs = np.sqrt(rv)
            rs[rs < 1e-6] = np.nan
            z = np.abs((v - rm) / rs)
            abs_dev = np.abs(v - rm)

        z_flag = np.where(np.isnan(z), False, z > z_thresh)
        abs_flag = (
            (abs_dev > min_abs_delta) if min_abs_delta > 0.0 else np.ones(n, dtype=bool)
        )
        out[idx] = (z_flag & abs_flag).astype(np.int8)

    return out


# Flag runs of identical values at least min_run long per host.
def flag_flatlines(df: pd.DataFrame, col: str, min_run: int) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int8)
    vals = df[col].to_numpy(dtype=np.float32)

    for _, idx in df.groupby("hostname", sort=False).indices.items():
        v = vals[idx]
        nan_mask = np.isnan(v)
        if nan_mask.all():
            continue

        chg = np.empty(len(v), dtype=bool)
        chg[0] = True
        chg[1:] = (v[1:] != v[:-1]) | nan_mask[1:] | nan_mask[:-1]

        starts = np.where(chg)[0]
        lens = np.diff(np.append(starts, len(v)))
        flags = (np.repeat(lens, lens) >= min_run).astype(np.int8)
        flags[nan_mask] = 0
        out[idx] = flags

    return out


# Look up a per-metric override, falling back to the short name then a default.
def resolve_metric_override(metric: str, overrides: dict, default):
    val = overrides.get(metric)
    if val is not None:
        return val
    if "_" in metric:
        short = metric.split("_", 1)[1]
        val = overrides.get(short)
        if val is not None:
            return val
    return default


# Compute the flatline run length in points from the configured window minutes.
def flatline_window(metric: str, audit_cfg: dict, interval: int) -> int:
    overrides = audit_cfg.get("flatline_overrides", {})
    minutes = resolve_metric_override(
        metric, overrides, audit_cfg["flatline_window_min"]
    )
    if minutes == -1:
        return -1
    return max(3, int(minutes * 60 / interval))


# Compute dup/spike/flatline/bad-range flags per sensor for one metric DataFrame.
def audit_df(
    df: pd.DataFrame,
    metric: str,
    source_name: str,
    cfg: dict,
    thresholds: pd.DataFrame = None,
    comp: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    t0 = time.perf_counter()

    source_cfg = cfg["sources"].get(source_name, {})
    audit_cfg = cfg.get("audit", {})
    interval = source_cfg.get("interval_seconds", 60)
    ts_col = source_cfg.get("schema", {}).get("timestamp_col", "timestamp")

    if source_cfg.get("skip_audit", False):
        df = df.sort_values(["hostname", ts_col]).reset_index(drop=True)
        df = df.copy()
        df["audit_flags"] = np.int8(0)
        df.drop(columns=["_dup_count"], errors="ignore", inplace=True)
        if verbose:
            print(
                f"      audit: skipped (skip_audit=true)  "
                f"{len(df):,} rows -> audit_flags=0 throughout"
            )
        return df

    enable_dup = audit_cfg.get("enable_dup", True)
    enable_spike = audit_cfg.get("enable_spike", True)
    enable_flatline = audit_cfg.get("enable_flatline", True)
    enable_bad_range = audit_cfg.get("enable_bad_range", True)

    default_ranges = source_cfg.get("expected_ranges", {})
    comp_ranges = (
        source_cfg.get("per_component_ranges", {}).get(comp, {}) if comp else {}
    )
    ranges = comp_ranges.get(metric) or default_ranges.get(metric)

    spike_overrides = audit_cfg.get("spike_zscore_overrides", {})
    effective_z = resolve_metric_override(
        metric, spike_overrides, audit_cfg["spike_zscore"]
    )
    abs_delta_overrides = audit_cfg.get("spike_abs_delta_overrides", {})
    effective_abs_delta = resolve_metric_override(
        metric, abs_delta_overrides, audit_cfg.get("spike_abs_delta", 0.0)
    )

    rolling_pts = max(10, int(120 * 60 / interval))
    min_run_pts = flatline_window(metric, audit_cfg, interval)

    df = df.sort_values(["hostname", ts_col]).reset_index(drop=True)

    avg_cols = [c for c in df.columns if c.endswith("_avg")]

    # One flag array per sensor so a bad sensor can be masked independently.
    col_flags: dict = {col: np.zeros(len(df), dtype=np.int8) for col in avg_cols}

    t_dup = time.perf_counter()
    dup_row = np.zeros(len(df), dtype=np.int8)
    if enable_dup and "_dup_count" in df.columns:
        multiplier = audit_cfg.get("dup_threshold_multiplier", 4)
        median_count = df["_dup_count"].median()
        dup_row = (
            df["_dup_count"].to_numpy() > max(multiplier * median_count, 2)
        ).astype(np.int8) * FLAG_DUP
        for col in avg_cols:
            col_flags[col] |= dup_row
    n_dup = int((dup_row & FLAG_DUP).astype(bool).sum())
    t_dup = time.perf_counter() - t_dup
    t_spike = t_flat = 0.0

    for col in avg_cols:
        if df[col].isna().all():
            if verbose:
                print(f"      {col}: all NaN, skipping")
            continue

        if enable_bad_range and thresholds is not None and comp:
            hi_buffer = audit_cfg.get("buffer_hi", 1.05)
            lo_buffer = audit_cfg.get("buffer_lo", 0.95)
            sensor_name = col[:-4] if col.endswith("_avg") else col
            key = f"{comp}_{source_name}_{metric}_{sensor_name}"
            if key in thresholds.index:
                limit_hi = float(thresholds.loc[key, "p999"]) * hi_buffer
                limit_lo = float(thresholds.loc[key, "p001"]) * lo_buffer
                vals = df[col].to_numpy()
                is_impossible = vals < 0
                oob = (vals > limit_hi) | (vals < limit_lo)
                col_flags[col] |= (oob | is_impossible).astype(np.int8) * FLAG_BAD_RANGE

        if enable_spike and effective_z != -1:
            t1 = time.perf_counter()
            col_flags[col] |= (
                flag_spikes(
                    df,
                    col,
                    float(effective_z),
                    rolling_pts,
                    min_abs_delta=float(effective_abs_delta),
                )
                * FLAG_SPIKE
            )
            t_spike += time.perf_counter() - t1

        if enable_flatline and min_run_pts != -1:
            t1 = time.perf_counter()
            col_flags[col] |= flag_flatlines(df, col, min_run_pts) * FLAG_FLATLINE
            t_flat += time.perf_counter() - t1

    agg_flags = np.zeros(len(df), dtype=np.int8)
    for f in col_flags.values():
        agg_flags |= f

    n_spike = int(((agg_flags & FLAG_SPIKE) > 0).sum())
    n_flat = int(((agg_flags & FLAG_FLATLINE) > 0).sum())
    n_range = int(((agg_flags & FLAG_BAD_RANGE) > 0).sum())
    n_any = int((agg_flags > 0).sum())
    t_elapsed = time.perf_counter() - t0

    df = df.copy()
    for col, f in col_flags.items():
        sensor_name = col[:-4] if col.endswith("_avg") else col
        df[f"audit_flags__{sensor_name}"] = f
    df.drop(columns=["_dup_count"], errors="ignore", inplace=True)

    return df


# Summarise per-sensor audit flags into row counts and percentages.
def extract_metric_stats(df: pd.DataFrame) -> dict:
    # Support both the new per-sensor schema (audit_flags__{sensor}) and the legacy single audit_flags column for parquets not yet re-audited.
    per_sensor_cols = [c for c in df.columns if c.startswith("audit_flags__")]
    if per_sensor_cols:
        agg = np.zeros(len(df), dtype=np.int64)
        for col in per_sensor_cols:
            agg |= df[col].fillna(0).astype("int64").to_numpy()
        flags = pd.Series(agg, index=df.index)
    elif "audit_flags" in df.columns:
        flags = df["audit_flags"].fillna(0).astype("int64")
    else:
        return {}

    n_rows = len(df)
    n_dup = int(((flags & int(FLAG_DUP)) > 0).sum())
    n_spk = int(((flags & int(FLAG_SPIKE)) > 0).sum())
    n_flt = int(((flags & int(FLAG_FLATLINE)) > 0).sum())
    n_rng = int(((flags & int(FLAG_BAD_RANGE)) > 0).sum())
    n_any = int((flags > 0).sum())

    return {
        "rows": n_rows,
        "clean": n_rows - n_any,
        "dup": n_dup,
        "spike": n_spk,
        "flatline": n_flt,
        "bad_range": n_rng,
        "any_flag": n_any,
        "flag_pct": round(100 * n_any / max(n_rows, 1), 3),
    }


# Write the aggregated audit statistics to a JSON report.
def write_audit_summary(summary: dict, cfg: dict) -> None:
    out_dir = Path(cfg.get("paths", {}).get("reports", "reports")) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_summary.json"

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "window": cfg.get("window", {}),
    }
    payload.update(summary)

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  [audit_summary] -> {out_path}")


# Load and index the metric threshold table from the visualization reports.
def load_thresholds(cfg: dict):
    path = Path(cfg["paths"]["visualization"]) / "thresholds" / "metric_thresholds.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    # Strip whitespace from column names and string values
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    if "key" in df.columns:
        return df.set_index("key")

    df["lookup_key"] = (
        df["platform"]
        + "_"
        + df["source_type"]
        + "_"
        + df["metric"]
        + "_"
        + df["fqdd"].fillna("none")
        + "_"
        + df["source"].fillna("none")
    )
    return df.set_index("lookup_key")


# Decode threshold rows' fqdd/source ids into sensor-name-keyed thresholds.
def make_sensor_thresholds(
    thresholds: pd.DataFrame, fqdd_map: dict, source_map: dict
) -> pd.DataFrame:
    if thresholds is None:
        return None

    records = []
    for _, row in thresholds.iterrows():
        try:
            fqdd_id = int(float(str(row["fqdd"])))
            fname = str(fqdd_map.get(fqdd_id, str(fqdd_id))).lower()
        except (ValueError, TypeError):
            fname = str(row["fqdd"]).lower()
        try:
            src_id = int(float(str(row["source"])))
            sname = str(source_map.get(src_id, str(src_id))).lower()
        except (ValueError, TypeError):
            sname = str(row["source"]).lower()

        # PDU use sentinel strings instead of numeric IDs.
        SENTINELS = {"global", "none", "metric", "nan", ""}
        if fname in SENTINELS and sname in SENTINELS:
            sensor_name = str(row.get("metric", "")).lower()
        else:
            sensor_name = fname if fname == sname else f"{fname}_{sname}"
        decoded_key = (
            f"{str(row.get('platform',''))}_"
            f"{str(row.get('source_type',''))}_"
            f"{str(row.get('metric',''))}_"
            f"{sensor_name}"
        )
        rec = row.to_dict()
        rec["decoded_key"] = decoded_key
        records.append(rec)

    if not records:
        return None

    out = pd.DataFrame(records).set_index("decoded_key")
    return out[~out.index.duplicated(keep="first")]


# Resolve the audit worker count from the environment or SLURM allocation.
def audit_workers() -> int:
    env = os.environ.get("INSIGHT_HPC_AUDIT_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(6, int(slurm_cpus)))
        except ValueError:
            return 1
    return 1


# Resolve how many node-group chunks to split giant files into.
def audit_split() -> int:
    try:
        return max(1, int(os.environ.get("INSIGHT_HPC_AUDIT_SPLIT", "1")))
    except ValueError:
        return 1


# Resolve the progress-heartbeat interval in seconds from the environment.
def log_heartbeat_seconds() -> int:
    try:
        return max(0, int(os.environ.get("INSIGHT_HPC_LOG_HEARTBEAT_SEC", "60")))
    except ValueError:
        return 60


# Return a parquet file's row count, or 0 if it cannot be read.
def parquet_num_rows(p: Path) -> int:
    try:
        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        return 0


# Audit one node-group chunk of a metric file and write a partial parquet.
def audit_chunk(
    parquet_path: Path,
    comp: str,
    source: str,
    source_dir: Path,
    partial_out: Path,
    cfg: dict,
    pub_tables: dict,
    comp_thresholds,
    node_ids,
):
    metric = parquet_path.stem.lower()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = load_metric_parquet(
            comp,
            source_dir,
            parquet_path,
            pub_tables,
            cfg,
            verbose=False,
            node_ids=node_ids,
        )
        if df is None or df.empty:
            return partial_out, 0, buf.getvalue()
        df = audit_df(
            df,
            metric,
            source,
            cfg,
            thresholds=comp_thresholds,
            comp=comp,
            verbose=False,
        )
        save_parquet(df, partial_out)
        n = len(df)
        del df
        gc.collect()
    return partial_out, n, buf.getvalue()


# Audit a whole metric file and write the flagged parquet, returning stats and a log line.
def audit_one_file(
    parquet_path: Path,
    comp: str,
    source: str,
    source_dir: Path,
    out_path: Path,
    force: bool,
    cfg: dict,
    pub_tables: dict,
    comp_thresholds,
    verbose: bool,
):
    metric = parquet_path.stem.lower()

    if out_path.exists() and not force:
        stats = None
        try:
            existing = pd.read_parquet(out_path, engine="pyarrow")
            stats = extract_metric_stats(existing)
        except Exception:
            pass
        return metric, stats, False, f"    skip  {metric}"

    t_file = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = load_metric_parquet(
            comp, source_dir, parquet_path, pub_tables, cfg, verbose=verbose
        )
        if df is None or df.empty:
            detail = buf.getvalue().rstrip()
            warn = f"    [WARN] {metric}: no data returned"
            return metric, None, False, f"{detail}\n{warn}" if detail else warn

        df = audit_df(
            df,
            metric,
            source,
            cfg,
            thresholds=comp_thresholds,
            comp=comp,
            verbose=verbose,
        )
    stats = extract_metric_stats(df)

    t_save = time.perf_counter()
    save_parquet(df, out_path)
    t_save = time.perf_counter() - t_save

    n_rows = len(df)
    n_cols = len(df.columns)
    per_sensor_flag_cols = [c for c in df.columns if c.startswith("audit_flags__")]
    if per_sensor_flag_cols:
        agg = np.zeros(n_rows, dtype=np.int64)
        for fc in per_sensor_flag_cols:
            agg |= df[fc].fillna(0).astype("int64").to_numpy()
        n_flagged = int((agg > 0).sum())
    elif "audit_flags" in df.columns:
        n_flagged = int((df["audit_flags"] > 0).sum())
    else:
        n_flagged = 0
    mb_out = out_path.stat().st_size / 1e6
    elapsed = time.perf_counter() - t_file
    summary = (
        f"    DONE {metric:30s}  {n_rows:>12,} rows × {n_cols} cols  "
        f"flagged={n_flagged:>8,} ({100*n_flagged/max(n_rows,1):5.1f}%)  "
        f"saved={mb_out:.1f}MB  save={t_save:.1f}s  total={elapsed:.1f}s  "
        f"({n_rows/max(elapsed,0.001)/1e3:.0f}K rows/s)"
    )
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    del df
    gc.collect()
    return metric, stats, True, log
