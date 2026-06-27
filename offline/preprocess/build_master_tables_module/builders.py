from __future__ import annotations

from offline.preprocess.build_master_tables_module.constants import *
import gc, io, os, time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import load_config, load_public_tables, save_parquet
from shared.utils.parsers import ensure_utc_us

from offline.preprocess.build_master_tables_module.metric_loading import *
from offline.preprocess.build_master_tables_module.job_context import *


# Build one node's master table by merging iDRAC metrics, Slurm, and job context.
def build_node(
    hostname: str,
    node_id: int,
    node_meta: dict,
    component: str,
    audited_dir: Path,
    raw_base: Path,
    out_path: Path,
    cfg: dict,
    force: bool,
    apply_mask: bool,
) -> Optional[int]:
    if out_path.exists() and not force:
        return None

    t0 = time.perf_counter()
    ts_col = "timestamp"
    w = cfg["window"]
    t_start = pd.Timestamp(w["start"], tz="UTC")
    t_end = pd.Timestamp(w["end"], tz="UTC")

    idrac_dir = audited_dir / component / "idrac"
    frames = []
    for p in sorted(idrac_dir.glob("*.parquet")):
        # Always load raw values — masking is deferred until after job join.
        mf = load_audited_metric(p, hostname, ts_col)
        if mf is not None and not mf.empty:
            frames.append(mf)

    if not frames:
        if component != "infra":
            print(f"      {hostname}: no iDRAC data, skipping")
        return None

    wide = merge_metric_frames(frames, ts_col)
    wide = wide[(wide[ts_col] >= t_start) & (wide[ts_col] <= t_end)].copy()
    if wide.empty:
        return None

    slurm_audited = audited_dir / component / "slurm"
    if slurm_audited.exists():
        sl = load_slurm_node(slurm_audited, hostname, ts_col, node_id, apply_mask)
        if sl is not None:
            wide = ensure_utc_us(wide, ts_col).merge(
                ensure_utc_us(sl, ts_col), on=ts_col, how="left"
            )

    # Expand jobs.parquet into a minute-resolution timeline; supplement with cpu_share polling data from node_jobs.parquet.
    slurm_raw = raw_base / component / "slurm"
    if slurm_raw.exists():
        expanded = load_jobs_for_node(slurm_raw, hostname, node_id, ts_col, cfg)
        snap = load_node_jobs(slurm_raw, hostname, ts_col, node_id)

        jdf = build_job_context(expanded, snap, ts_col)

        if jdf is not None:
            wide = ensure_utc_us(wide, ts_col).merge(
                ensure_utc_us(jdf, ts_col), on=ts_col, how="left"
            )

            # Timestamps not matched by jdf are idle minutes.
            idle = wide["active_job_count"].isna()
            if idle.any():
                wide.loc[idle, "active_job_count"] = 0
                wide.loc[idle, "is_multi_job"] = 0.0
                wide.loc[idle, "primary_job_id"] = pd.NA
                wide.loc[idle, "primary_job_cpu_share"] = np.nan
                wide.loc[idle, "total_job_cpu_share"] = np.nan
                wide.loc[idle, "jobs_json"] = None
                wide.loc[idle, "cpu_shares_json"] = None
                for rc in [
                    "job_req_mem_mb",
                    "job_req_tasks_per_node",
                    "job_req_cpus_per_task",
                ]:
                    if rc in wide.columns:
                        wide.loc[idle, rc] = np.nan

            # Diagnostic: warn if any master timestamp falls in a known job window but was not matched (Slurm data inconsistency).
            if expanded is not None:
                job_ts = set(pd.to_datetime(expanded[ts_col], utc=True).dt.floor("60s"))
                master_ts = set(pd.to_datetime(wide[ts_col], utc=True))
                unmatched_job_ts = job_ts & master_ts
                unmatched_idle = wide[
                    wide[ts_col].isin(unmatched_job_ts)
                    & (wide["active_job_count"] == 0)
                ]
                if len(unmatched_idle) > 0:
                    print(
                        f"      [WARN] {hostname}: {len(unmatched_idle)} master timestamps "
                        f"fall in a job window but show active_job_count=0 — "
                        f"check Slurm node assignment consistency"
                    )

    # Assign split BEFORE masking so apply_selective_mask can restrict NaN operations to train rows only (val/test anomalies must be preserved).
    wide = assign_split(wide, ts_col, cfg)

    master_cfg = cfg.get("master", {})
    if master_cfg.get("save_both", False):
        raw_master = Path(
            cfg["paths"].get(
                "master_raw",
                str(Path(cfg["paths"]["master"]).parent / "master_raw"),
            )
        )
        raw_out = raw_master / component / out_path.name
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        if not raw_out.exists() or force:
            save_parquet(wide, raw_out)

    if apply_mask:
        wide = apply_selective_mask(wide)
        wide = flag_sensor_silence(wide)

    wide = attach_metadata(wide, hostname, component, node_meta)

    val_cols = [c for c in wide.columns if c.endswith("_avg")]
    if val_cols:
        wide[val_cols] = (
            wide[val_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        )
    for col in ["primary_job_cpu_share", "total_job_cpu_share", "is_multi_job"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").astype("float32")
    if "active_job_count" in wide.columns:
        wide["active_job_count"] = (
            pd.to_numeric(wide["active_job_count"], errors="coerce")
            .fillna(0)
            .astype("int16")
        )

    save_parquet(wide, out_path)

    elapsed = time.perf_counter() - t0
    n_rows = len(wide)
    n_cols = len(wide.columns)
    mb = out_path.stat().st_size / 1e6
    val_cols = [c for c in wide.columns if c.endswith("_avg")]
    pct_nan = 100 * wide[val_cols].isna().values.mean() if val_cols else 0.0

    pct_any = 100 * wide["audit_any"].mean() if "audit_any" in wide.columns else 0.0

    if "active_job_count" in wide.columns:
        n_job_rows = int((wide["active_job_count"] > 0).sum())
        n_multi_rows = int((wide["active_job_count"] > 1).sum())
        job_note = f"  job_rows={n_job_rows:,} multi={n_multi_rows:,}"
    else:
        job_note = ""

    print(
        f"      {hostname:20s}: {n_rows:>8,} rows × {n_cols} cols  "
        f"NaN={pct_nan:.1f}%  audit_any={pct_any:.1f}%  "
        f"{mb:.1f}MB  {elapsed:.1f}s{job_note}"
    )
    return n_rows


# Build one infra unit's (PDU) master table from its audited metrics.
def build_infra_unit(
    unit_id: str,
    source: str,
    audited_dir: Path,
    out_path: Path,
    cfg: dict,
    force: bool,
    apply_mask: bool,
) -> Optional[int]:
    if out_path.exists() and not force:
        return None

    t0 = time.perf_counter()
    ts_col = "timestamp"
    w = cfg["window"]
    t_start = pd.Timestamp(w["start"], tz="UTC")
    t_end = pd.Timestamp(w["end"], tz="UTC")

    infra_dir = audited_dir / "infra" / source
    frames = []

    for p in sorted(infra_dir.glob("*.parquet")):
        try:
            mf = load_audited_metric(p, unit_id, ts_col, by_unit=True)
        except Exception as e:
            print(f"      [WARN] {unit_id} {p.name}: {e}")
            continue
        if mf is not None and not mf.empty:
            frames.append(mf)

    if not frames:
        return None

    wide = merge_metric_frames(frames, ts_col)
    wide = wide[(wide[ts_col] >= t_start) & (wide[ts_col] <= t_end)].copy()
    if wide.empty:
        return None

    wide.insert(0, "source", source)
    wide.insert(0, "unit_id", unit_id)
    # Assign split first so masking can restrict NaN-ing to train rows.
    wide = assign_split(wide, ts_col, cfg)

    if apply_mask:
        is_train = (
            (wide["split"] == "train")
            if "split" in wide.columns
            else pd.Series(True, index=wide.index)
        )
        flag_cols = [c for c in wide.columns if c.startswith("audit_flags__")]
        for fc in flag_cols:
            body = fc[len("audit_flags__") :]
            sensor_col = f"{body}_avg"
            if sensor_col in wide.columns:
                target_cols = [sensor_col]
            else:
                stem = body
                target_cols = [
                    c
                    for c in wide.columns
                    if c.startswith(f"{stem}__") and c.endswith("_avg")
                ]
            bad_train = (wide[fc].fillna(0).astype("int64") != 0) & is_train
            if target_cols and bad_train.any():
                wide.loc[bad_train, target_cols] = np.nan

    val_cols = [c for c in wide.columns if c.endswith("_avg")]
    if val_cols:
        wide[val_cols] = (
            wide[val_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        )

    save_parquet(wide, out_path)

    elapsed = time.perf_counter() - t0
    n_rows = len(wide)
    mb = out_path.stat().st_size / 1e6
    pct_nan = 100 * wide[val_cols].isna().values.mean() if val_cols else 0.0
    pct_any = 100 * wide["audit_any"].mean() if "audit_any" in wide.columns else 0.0
    print(
        f"      {unit_id:20s}: {n_rows:>8,} rows × {len(wide.columns)} cols  "
        f"NaN={pct_nan:.1f}%  audit_any={pct_any:.1f}%  {mb:.1f}MB  {elapsed:.1f}s"
    )
    return n_rows


# Resolve the master-build worker count from the environment or SLURM allocation.
def master_workers() -> int:
    env = os.environ.get("INSIGHT_HPC_MASTER_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(8, int(slurm_cpus)))
        except ValueError:
            return 1
    return 1


# Build one node in a worker process, capturing its stdout into a log string.
def master_build_one(kwargs: dict):
    buf = io.StringIO()
    with redirect_stdout(buf):
        n = build_node(**kwargs)
    return kwargs["hostname"], n, str(kwargs["out_path"]), buf.getvalue()
