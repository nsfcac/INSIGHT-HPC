from __future__ import annotations

from offline.preprocess.build_master_tables_module.constants import *
import json, os
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.parsers import (
    align_cpu,
    as_utc_min,
    ensure_utc_us,
    first_existing,
    is_missing,
    parse_float_list,
    parse_int_list,
    to_int,
)


# Refuse to run heavy preprocessing outside a Slurm allocation.
def require_slurm_for_heavy(stage: str, slurm_hint: str) -> None:
    if (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("INSIGHT_HPC_ALLOW_LOGIN_HEAVY") == "1"
    ):
        return
    raise SystemExit(
        f"[{stage}] Refusing to run heavy preprocessing on a login node. "
        f"Submit through Slurm instead, e.g. `sbatch {slurm_hint}`. "
        "For tiny smoke tests only, set INSIGHT_HPC_ALLOW_LOGIN_HEAVY=1."
    )


# Tag each row train/val/test by timestamp using the configured window.
def assign_split(df: pd.DataFrame, ts_col: str, cfg: dict) -> pd.DataFrame:
    w = cfg["window"]
    train_end = pd.Timestamp(w["train_end"], tz="UTC")
    val_end = pd.Timestamp(w["val_end"], tz="UTC")
    ts = df[ts_col]
    df = df.copy()
    df["split"] = pd.Categorical(
        np.where(ts <= train_end, "train", np.where(ts <= val_end, "val", "test")),
        categories=["train", "val", "test"],
    )
    return df


# Load one audited metric for a node/unit and prefix its sensor and flag columns.
def load_audited_metric(
    path: Path, key: str, ts_col: str, by_unit: bool = False
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        print(f"- [WARN] {path.name}: {e}")
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    if by_unit:
        if "hostname" in df.columns:
            df = df[df["hostname"] == key].copy()
        else:
            nc = first_existing(list(df.columns), ["nodeid", "node_id"])
            df = df[df[nc].astype(str) == str(key)].copy() if nc else df.copy()
    else:
        if "hostname" not in df.columns or ts_col not in df.columns:
            return None
        df = df[df["hostname"] == key].copy()
    if df.empty:
        return None

    df[ts_col] = as_utc_min(df[ts_col])
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    stem = path.stem.lower()
    val_cols = [c for c in df.columns if c.endswith("_avg")]

    per_sensor_flag_cols = [c for c in df.columns if c.startswith("audit_flags__")]
    if not per_sensor_flag_cols and "audit_flags" in df.columns:
        old_flags = df["audit_flags"].fillna(0).astype("int64")
        for col in val_cols:
            sensor_name = col[:-4]
            df[f"audit_flags__{sensor_name}"] = old_flags.values
        per_sensor_flag_cols = [f"audit_flags__{col[:-4]}" for col in val_cols]

    rename_val = {c: f"{stem}__{c}" for c in val_cols}
    df.rename(columns=rename_val, inplace=True)
    prefixed_val_cols = list(rename_val.values())

    rename_flags = {
        c: f"audit_flags__{stem}__{c[len('audit_flags__'):]}"
        for c in per_sensor_flag_cols
    }
    df.rename(columns=rename_flags, inplace=True)
    prefixed_flag_cols = list(rename_flags.values())

    tag = f"_audit_any__{stem}"
    if prefixed_flag_cols:
        any_flagged = np.zeros(len(df), dtype=bool)
        for fc in prefixed_flag_cols:
            any_flagged |= (df[fc].fillna(0).astype("int64") != 0).to_numpy()
        df[tag] = any_flagged
    else:
        df[tag] = False

    keep = [ts_col] + prefixed_val_cols + prefixed_flag_cols + [tag]
    return df[keep].drop_duplicates(ts_col, keep="last").reset_index(drop=True)


# Outer-join per-metric frames on timestamp and combine their audit-any flags.
def merge_metric_frames(frames: list, ts_col: str) -> pd.DataFrame:
    wide = reduce(
        lambda a, b: pd.merge(
            ensure_utc_us(a, ts_col), ensure_utc_us(b, ts_col), on=ts_col, how="outer"
        ),
        frames,
    )

    audit_cols = [c for c in wide.columns if c.startswith("_audit_any__")]
    if audit_cols:
        wide["audit_any"] = wide[audit_cols].eq(True).any(axis=1)
        wide.drop(columns=audit_cols, inplace=True)
    else:
        wide["audit_any"] = False

    return wide.sort_values(ts_col).reset_index(drop=True)


# Flag sustained gaps in critical sensors while the node is still drawing power.
def flag_sensor_silence(
    wide: pd.DataFrame, silence_thresh_min: int = 60
) -> pd.DataFrame:
    CRITICAL_KEYWORDS = (
        "rpmreading__",
        "temperaturereading__idrac",
        "systeminputpower__",
        "totalcpupower__",
        "cpupower__",
    )
    critical_cols = [
        c
        for c in wide.columns
        if c.endswith("_avg") and any(kw in c for kw in CRITICAL_KEYWORDS)
    ]
    if not critical_cols:
        return wide

    # Power acts as the "node is alive" indicator: when a node consumes power, iDRAC should be reporting.
    power_cols = [
        c for c in wide.columns if "systeminputpower" in c and c.endswith("_avg")
    ]

    wide = wide.copy()
    silence_mask = pd.Series(False, index=wide.index)

    for col in critical_cols:
        vals = wide[col].to_numpy(dtype=np.float64)
        is_missing = np.isnan(vals) | (vals == 0)

        streak = np.zeros(len(vals), dtype=np.int32)
        count = 0
        for i, miss in enumerate(is_missing):
            count = count + 1 if miss else 0
            streak[i] = count

        sustained = streak > silence_thresh_min

        if power_cols:
            power_ok = np.zeros(len(wide), dtype=bool)
            for pc in power_cols:
                pvals = wide[pc].to_numpy(dtype=np.float64)
                power_ok |= ~np.isnan(pvals) & (pvals > 0)
        else:
            # No power sensor available — fall back to any other critical sensor.
            power_ok = np.zeros(len(wide), dtype=bool)
            for oc in critical_cols:
                if oc == col:
                    continue
                ovals = wide[oc].to_numpy(dtype=np.float64)
                power_ok |= ~np.isnan(ovals) & (ovals > 0)

        col_silence = pd.Series(sustained & power_ok, index=wide.index)
        silence_mask |= col_silence

    if silence_mask.any():
        if "audit_any" in wide.columns:
            wide["audit_any"] = wide["audit_any"] | silence_mask
        else:
            wide["audit_any"] = silence_mask

    return wide


# Clear legitimate idle/active flags and NaN bad-range training values per sensor.
def apply_selective_mask(wide: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [c for c in wide.columns if c.startswith("audit_flags__")]
    if not flag_cols:
        return wide

    if "active_job_count" in wide.columns:
        is_idle = wide["active_job_count"].fillna(0).astype("int64") == 0
        is_active = ~is_idle
    else:
        is_idle = pd.Series(False, index=wide.index)
        is_active = pd.Series(False, index=wide.index)

    wide = wide.copy()
    any_flags = pd.Series(False, index=wide.index)

    is_train = (
        (wide["split"] == "train")
        if "split" in wide.columns
        else pd.Series(True, index=wide.index)
    )

    for fc in flag_cols:
        body = fc[len("audit_flags__") :]
        sensor_col = f"{body}_avg"

        if sensor_col in wide.columns:
            sensor_cols = [sensor_col]
        else:
            stem = body
            sensor_cols = [
                c
                for c in wide.columns
                if c.startswith(f"{stem}__") and c.endswith("_avg")
            ]

        flags = wide[fc].fillna(0).astype("int64")

        # Idle flatlines are legitimate (sensor holds steady when no job runs).
        idle_flatline = is_idle & ((flags & FLAG_FLATLINE) > 0)
        if idle_flatline.any():
            flags = flags.copy()
            flags[idle_flatline] = flags[idle_flatline] & ~FLAG_FLATLINE
            wide[fc] = flags

        # Active-period spikes are legitimate job-induced transients for power/CPU/GPU/memory metrics. Temperature spikes during jobs are not cleared.
        TEMP_KEYWORDS = ("temp", "thermal", "inlet", "return", "supply")
        is_temp_sensor = any(kw in body.lower() for kw in TEMP_KEYWORDS)
        if not is_temp_sensor:
            active_spike = is_active & ((flags & FLAG_SPIKE) > 0)
            if active_spike.any():
                flags = flags.copy()
                flags[active_spike] = flags[active_spike] & ~FLAG_SPIKE
                wide[fc] = flags

        # Bad-range values are stripped only from training rows.
        bad_range_train = is_train & ((flags & FLAG_BAD_RANGE) > 0)
        if bad_range_train.any() and sensor_cols:
            wide.loc[bad_range_train, sensor_cols] = np.nan

        any_flags |= flags > 0

    if "audit_any" in wide.columns:
        wide["audit_any"] = any_flags

    return wide


# Load and merge a node's Slurm cpu_load/memory metrics into per-minute columns.
def load_slurm_node(
    slurm_dir: Path, hostname: str, ts_col: str, node_id: int, apply_mask: bool
) -> Optional[pd.DataFrame]:
    parts = []
    for metric in ["cpu_load", "memory_used", "memoryusage"]:
        p = slurm_dir / f"{metric}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, engine="pyarrow")
            df.columns = [c.strip().lower() for c in df.columns]

            node_col = first_existing(list(df.columns), ["nodeid", "node_id"])
            if node_col:
                df = df[df[node_col] == node_id]
            elif "hostname" in df.columns:
                df = df[df["hostname"] == hostname]
            if df.empty:
                continue

            val_col = first_existing(
                list(df.columns), ["value", f"{metric}_avg", metric]
            )
            if not val_col or ts_col not in df.columns:
                continue

            flags = (
                df["audit_flags"].fillna(0).astype("int64")
                if "audit_flags" in df.columns
                else pd.Series(0, index=df.index)
            )

            tmp = df[[ts_col, val_col]].copy()
            tmp[ts_col] = as_utc_min(tmp[ts_col])
            tmp = tmp.dropna(subset=[ts_col]).sort_values(ts_col)
            tmp = tmp.drop_duplicates(ts_col, keep="last")
            tmp = tmp.rename(columns={val_col: f"slurm_{metric}"})
            tmp[f"slurm_{metric}"] = pd.to_numeric(
                tmp[f"slurm_{metric}"], errors="coerce"
            ).astype("float32")

            if apply_mask:
                bad = flags.reindex(tmp.index).fillna(0) != 0
                tmp.loc[bad, f"slurm_{metric}"] = np.nan

            parts.append(tmp.reset_index(drop=True))
        except Exception as e:
            print(f"      [WARN] slurm {metric}: {e}")

    if not parts:
        return None
    return (
        reduce(
            lambda a, b: pd.merge(
                ensure_utc_us(a, ts_col),
                ensure_utc_us(b, ts_col),
                on=ts_col,
                how="outer",
            ),
            parts,
        )
        .sort_values(ts_col)
        .reset_index(drop=True)
    )


# Load node_jobs polling data into per-minute job-context columns.
def load_node_jobs(
    slurm_dir: Path, hostname: str, ts_col: str, node_id: int
) -> Optional[pd.DataFrame]:
    p = slurm_dir / "node_jobs.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        df.columns = [c.strip().lower() for c in df.columns]

        if "hostname" in df.columns:
            df = df[df["hostname"] == hostname]
        else:
            nc = first_existing(list(df.columns), ["nodeid", "node_id"])
            if nc:
                df = df[df[nc] == node_id]

        if df.empty or ts_col not in df.columns:
            return None

        df = df.copy()
        df[ts_col] = as_utc_min(df[ts_col])
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)

        jobs_col = first_existing(list(df.columns), ["jobs", "job_ids", "active_jobs"])
        cpu_col = first_existing(
            list(df.columns), ["cpu", "cpus", "cpu_share", "cpu_shares"]
        )
        pri_col = first_existing(list(df.columns), ["job_id", "primary_job_id"])

        if jobs_col is not None:
            job_lists = df[jobs_col].apply(parse_int_list)
        elif pri_col is not None:
            job_lists = df[pri_col].apply(
                lambda x: [v] if (v := to_int(x)) is not None else []
            )
        else:
            return None

        cpu_lists = (
            df[cpu_col].apply(parse_float_list)
            if cpu_col
            else pd.Series([[] for _ in range(len(df))], index=df.index)
        )
        aligned_cpu = [
            align_cpu(j, c) for j, c in zip(job_lists.tolist(), cpu_lists.tolist())
        ]

        out = pd.DataFrame(
            {
                ts_col: df[ts_col].values,
                "jobs_json": [json.dumps(v) if v else None for v in job_lists],
                "cpu_shares_json": [json.dumps(v) if v else None for v in aligned_cpu],
                "active_job_count": pd.array(
                    [len(v) for v in job_lists], dtype="int16"
                ),
                "primary_job_id": pd.array(
                    [v[0] if v else None for v in job_lists.tolist()], dtype="Int64"
                ),
                "primary_job_cpu_share": pd.Series(
                    [c[0] if c else np.nan for c in aligned_cpu], dtype="float32"
                ),
                "total_job_cpu_share": pd.Series(
                    [float(np.nansum(c)) if c else np.nan for c in aligned_cpu],
                    dtype="float32",
                ),
                "is_multi_job": pd.array(
                    [len(v) > 1 for v in job_lists], dtype="float32"
                ),
            }
        )
        return (
            out.drop_duplicates(ts_col, keep="last")
            .sort_values(ts_col)
            .reset_index(drop=True)
        )
    except Exception as e:
        print(f"      [WARN] node_jobs: {e}")
        return None
