from __future__ import annotations

import json, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit


# Parse a jobs_json value into a sorted tuple of positive job ids.
def active_job_signature(val) -> tuple[int, ...]:
    try:
        jobs = json.loads(val) if isinstance(val, str) else val
    except Exception:
        return tuple()
    if not isinstance(jobs, list):
        return tuple()
    cleaned = set()
    for job in jobs:
        try:
            jid = int(job)
        except Exception:
            continue
        if jid > 0:
            cleaned.add(jid)
    return tuple(sorted(cleaned))


# Split a node's timeline into contiguous job and idle segments.
def segment_one_node(
    df: pd.DataFrame, hostname: str, component: str
) -> Optional[pd.DataFrame]:
    ts_col = "timestamp"
    job_col = "primary_job_id"

    if ts_col not in df.columns:
        return None

    df = df.sort_values(ts_col).copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    if job_col not in df.columns:
        df[job_col] = pd.NA

    job_vals = pd.to_numeric(df[job_col], errors="coerce")
    # Treat NaN / <= 0 as idle (-1 sentinel)
    idle_mask = job_vals.isna() | (job_vals <= 0)
    job_vals = job_vals.where(~idle_mask, other=-1)

    if "jobs_json" in df.columns:
        job_sigs = df["jobs_json"].apply(active_job_signature)
    else:
        job_sigs = job_vals.apply(lambda v: tuple() if v == -1 else (int(v),))

    job_changed = job_vals != job_vals.shift(1)
    jobset_changed = job_sigs != job_sigs.shift(1)
    ts_gap = df[ts_col].diff().dt.total_seconds() > 120
    boundary = job_changed | jobset_changed | ts_gap
    seg_id = boundary.cumsum()

    records = []
    for _, grp in df.groupby(seg_id, sort=False):
        jid = job_vals.loc[grp.index].iloc[0]
        sig = job_sigs.loc[grp.index].iloc[0]
        is_idle = bool(jid == -1 or pd.isna(jid))
        actual_jid = None if is_idle else int(jid)

        is_multi = False
        if "is_multi_job" in grp.columns:
            is_multi = bool(grp["is_multi_job"].fillna(0).astype(float).mean() > 0.5)

        split_val = "unknown"
        if "split" in grp.columns and len(grp) > 0:
            mode_result = grp["split"].mode()
            split_val = mode_result.iloc[0] if len(mode_result) > 0 else "unknown"

        records.append(
            {
                "hostname": hostname,
                "component": component,
                "job_id": actual_jid,
                "seg_start": grp[ts_col].iloc[0],
                "seg_end": grp[ts_col].iloc[-1],
                "duration_min": max(
                    (grp[ts_col].iloc[-1] - grp[ts_col].iloc[0]).total_seconds() / 60.0,
                    1.0,  # minimum 1 minute for single-row segments
                ),
                "is_idle": is_idle,
                "is_multi_job": is_multi,
                "active_job_ids": json.dumps(list(sig)) if sig else "",
                "n_active_jobs": len(sig),
                "split": split_val,
                "n_rows": int(len(grp)),
            }
        )

    if not records:
        return None

    out = pd.DataFrame(records)
    out["job_id"] = pd.array(out["job_id"], dtype="Int64")
    out["n_rows"] = out["n_rows"].astype("int32")
    out["n_active_jobs"] = out["n_active_jobs"].astype("int16")
    out["duration_min"] = out["duration_min"].astype("float32")
    return out


# Extract per-node job segments from master tables and attach req_cpus from Slurm.
def run_segment_jobs(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    master_dir = Path(cfg["paths"]["master"])
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "offline/data/phase2"))
    out_path = phase2_dir / "job_segments.parquet"
    max_job_seg_min = (
        float(cfg.get("sources", {}).get("slurm", {}).get("max_job_duration_hours", 72))
        * 60.0
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[segment] job_segments.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    t0 = time.perf_counter()
    print("\n[segment] Extracting job segments from master tables ...")

    all_frames = []
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue

        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))
        print(f"  [{comp.upper()}]  {len(parquets)} nodes")

        for p in parquets:
            hostname = p.stem
            df = load_parquet(p)
            if df is None or df.empty:
                continue
            segs = segment_one_node(df, hostname, comp)
            if segs is not None:
                all_frames.append(segs)

    if not all_frames:
        print("[segment] No segments found.")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)

    # Propagate req_cpus from raw SLURM jobs.parquet
    raw_base = Path(cfg["paths"]["raw_parquet"])
    slurm_lookup: dict = {}
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        jobs_path = raw_base / comp / "slurm" / "jobs.parquet"
        if not jobs_path.exists():
            continue
        try:
            jdf = pd.read_parquet(jobs_path, engine="pyarrow")
            jdf.columns = [c.strip().lower() for c in jdf.columns]
            jid_col = next(
                (c for c in ["job_id", "jobid", "id"] if c in jdf.columns), None
            )
            cpu_col = next(
                (c for c in ["cpus", "num_cpus", "ncpus"] if c in jdf.columns), None
            )
            if jid_col and cpu_col:
                for _, jrow in jdf[[jid_col, cpu_col]].iterrows():
                    if pd.notna(jrow[jid_col]):
                        jid = int(jrow[jid_col])
                        slurm_lookup[jid] = (
                            float(jrow[cpu_col]) if pd.notna(jrow[cpu_col]) else np.nan
                        )
        except Exception as e:
            print(f"  [WARN] Could not read SLURM jobs for {comp}: {e}")

    result["req_cpus"] = (
        result["job_id"]
        .apply(lambda j: slurm_lookup.get(int(j), np.nan) if pd.notna(j) else np.nan)
        .astype("float32")
    )
    if slurm_lookup:
        n_mapped = result["req_cpus"].notna().sum()
        print(
            f"  req_cpus: mapped {n_mapped:,}/{len(result):,} segments from SLURM "
            f"({len(slurm_lookup):,} unique job_ids found)"
        )

    save_parquet(result, out_path)

    elapsed = time.perf_counter() - t0
    job_segs = result[~result["is_idle"]]
    idle_segs = result[result["is_idle"]]
    n_job = len(job_segs)
    n_idle = len(idle_segs)
    n_multi = int(job_segs["is_multi_job"].fillna(False).sum())

    print(f"\n[segment] Done in {elapsed:.1f}s")
    print(f"  Total segments  : {len(result):,}  job={n_job:,}  idle={n_idle:,}")

    if n_job > 0:
        jd = job_segs["duration_min"]
        print(
            f"  Job  duration   : p50={jd.median():.0f}  p95={jd.quantile(0.95):.0f}  "
            f"max={jd.max():.0f} min  "
            f"(p50={jd.median()/60:.1f}h  p95={jd.quantile(0.95)/60:.1f}h  "
            f"max={jd.max()/60:.1f}h)  multi_job_segs={n_multi}"
        )
        over = int((jd > max_job_seg_min).sum())
        if over:
            print(
                f"  [WARN] {over} job segments exceed cap {max_job_seg_min} min — check split logic"
            )

    if n_idle > 0:
        id_ = idle_segs["duration_min"]
        print(
            f"  Idle duration   : p50={id_.median():.0f}  p95={id_.quantile(0.95):.0f}  "
            f"max={id_.max():.0f} min  (max={id_.max()/60:.0f}h = expected for idle nodes)"
        )

    print(f"  Saved: {out_path}")
    return result


if __name__ == "__main__":
    run_segment_jobs(force=True)
