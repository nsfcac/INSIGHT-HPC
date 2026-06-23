from __future__ import annotations

import gc, io, json, os, time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_public_tables, save_parquet
from src.utils.parsers import align_cpu, as_utc_min, ensure_utc_us, first_existing, is_missing, parse_float_list, parse_int_list, parse_slurm_ts, parse_str_list, to_float, to_int

# Bitmask constants matching audit_raw_files.py
FLAG_SPIKE     = np.int64(2)
FLAG_FLATLINE  = np.int64(4)
FLAG_BAD_RANGE = np.int64(8)


# Refuse to run heavy stages on a login node unless under SLURM.
def require_slurm_for_heavy(stage: str, slurm_hint: str) -> None:
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("INSIGHT_HPC_ALLOW_LOGIN_HEAVY") == "1":
        return
    raise SystemExit(
        f"[{stage}] Refusing to run heavy preprocessing on a login node. "
        f"Submit through Slurm instead, e.g. `sbatch {slurm_hint}`. "
        "For tiny smoke tests only, set INSIGHT_HPC_ALLOW_LOGIN_HEAVY=1."
    )


# Tag rows train/val/test by timestamp against the configured window.
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


# by_unit=True matches infra units by hostname or nodeid; else matches a node hostname.
def load_audited_metric(path: Path, key: str, ts_col: str, by_unit: bool = False) -> Optional[pd.DataFrame]:
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


# Outer-merge per-metric frames on timestamp and OR their audit flags.
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


# Flag sustained critical-sensor silence while the node still draws power.
def flag_sensor_silence(wide: pd.DataFrame, silence_thresh_min: int = 60) -> pd.DataFrame:
    CRITICAL_KEYWORDS = (
        "rpmreading__", "temperaturereading__idrac", "systeminputpower__",
        "totalcpupower__", "cpupower__",
    )
    critical_cols = [
        c for c in wide.columns
        if c.endswith("_avg") and any(kw in c for kw in CRITICAL_KEYWORDS)
    ]
    if not critical_cols:
        return wide

    # Power acts as the "node is alive" indicator: when a node consumes power, iDRAC should be reporting.
    power_cols = [c for c in wide.columns
                  if "systeminputpower" in c and c.endswith("_avg")]

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


# Clear idle flatlines / active spikes; NaN bad-range train rows.
def apply_selective_mask(wide: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [c for c in wide.columns if c.startswith("audit_flags__")]
    if not flag_cols:
        return wide

    if "active_job_count" in wide.columns:
        is_idle   = wide["active_job_count"].fillna(0).astype("int64") == 0
        is_active = ~is_idle
    else:
        is_idle   = pd.Series(False, index=wide.index)
        is_active = pd.Series(False, index=wide.index)

    wide      = wide.copy()
    any_flags = pd.Series(False, index=wide.index)

    is_train = (
        (wide["split"] == "train")
        if "split" in wide.columns
        else pd.Series(True, index=wide.index)
    )

    for fc in flag_cols:
        body = fc[len("audit_flags__"):]
        sensor_col = f"{body}_avg"

        if sensor_col in wide.columns:
            sensor_cols = [sensor_col]
        else:
            stem = body
            sensor_cols = [
                c for c in wide.columns
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

        any_flags |= (flags > 0)

    if "audit_any" in wide.columns:
        wide["audit_any"] = any_flags

    return wide


# Load and merge a node's slurm metric timeseries.
def load_slurm_node(slurm_dir: Path, hostname: str, ts_col: str, node_id: int, apply_mask: bool) -> Optional[pd.DataFrame]:
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

            val_col = first_existing(list(df.columns),
                                      ["value", f"{metric}_avg", metric])
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


# Load per-minute node_jobs snapshots (job lists, cpu shares).
def load_node_jobs(slurm_dir: Path, hostname: str, ts_col: str, node_id: int) -> Optional[pd.DataFrame]:
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


# Expand jobs.parquet into a per-minute job-requirement timeline.
def load_jobs_for_node(slurm_raw_dir: Path, hostname: str, node_id: int, ts_col: str, cfg: dict) -> Optional[pd.DataFrame]:
    jobs_path = slurm_raw_dir / "jobs.parquet"
    if not jobs_path.exists():
        return None

    slurm_cfg = cfg.get("sources", {}).get("slurm", {})

    try:
        df = pd.read_parquet(jobs_path, engine="pyarrow")
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        print(f"      [WARN] jobs.parquet: {e}")
        return None

    if df.empty:
        return None

    jid_col       = first_existing(list(df.columns), ["jobid", "job_id", "id"])
    start_col     = first_existing(list(df.columns), ["start_time", "starttime", "start"])
    end_col       = first_existing(list(df.columns), ["end_time",   "endtime",   "end"])
    nodes_col     = first_existing(list(df.columns), ["nodes", "nodelist", "node_list",
                                                        "alloc_nodes", "nodenames"])
    state_col     = first_existing(list(df.columns), ["job_state", "state", "jobstate"])
    cpus_col      = first_existing(list(df.columns), ["cpus", "num_cpus", "ncpus",
                                                        "num_nodes_cpus", "cpus_per_node"])
    # num_nodes converts total CPUs -> per-node CPU allocation for multi-node jobs.
    num_nodes_col = first_existing(list(df.columns), ["num_nodes", "nnodes", "node_count",
                                                        "alloc_nodes_count"])
    # 0 in Slurm = "not set" — treat as NaN, not 0.
    mem_per_node_col   = first_existing(list(df.columns), ["min_mem_per_node", "mem_per_node",
                                                             "mem", "memory_per_node"])
    mem_per_cpu_col    = first_existing(list(df.columns), ["mem_per_cpu", "min_mem_per_cpu",
                                                             "memory_per_cpu"])
    cpus_per_task_col  = first_existing(list(df.columns), ["cpus_per_task", "cpus_per_tre",
                                                             "num_cpus_per_task"])
    tasks_col          = first_existing(list(df.columns), ["tasks", "num_tasks", "ntasks"])
    tasks_per_node_col = first_existing(list(df.columns), ["tasks_per_node", "ntasks_per_node",
                                                             "num_tasks_per_node"])

    if any(c is None for c in [jid_col, start_col, end_col, nodes_col]):
        print(f"      [WARN] jobs.parquet missing required columns "
              f"(need job_id, start_time, end_time, nodes).  "
              f"Found: {list(df.columns[:10])}")
        return None

    df["_start"] = parse_slurm_ts(df[start_col])
    df["_end"]   = parse_slurm_ts(df[end_col])

    # Drop jobs with invalid or zero times (never actually started).
    epoch_floor = pd.Timestamp("1970-01-01 00:00:02", tz="UTC")
    df = df.dropna(subset=["_start", "_end"])
    df = df[(df["_start"] > epoch_floor) & (df["_end"] > df["_start"])]

    if df.empty:
        return None

    node_lists = df[nodes_col].apply(parse_str_list)
    df = df[node_lists.apply(len) > 0].copy()
    node_lists = node_lists[df.index]

    if df.empty:
        return None

    NEVER_RAN = {"pending", "cancelled", "revoked", "suspended", "requeued"}
    if state_col is not None:
        def had_execution(states) -> bool:
            sl = parse_str_list(states)
            if not sl:
                return True
            return any(s.lower() not in NEVER_RAN for s in sl)
        ran_mask = df[state_col].apply(had_execution)
        df = df[ran_mask].copy()
        node_lists = node_lists[df.index]

    if df.empty:
        return None

    hostname_str = str(hostname)
    node_id_str  = str(node_id)
    on_node = node_lists.apply(
        lambda nl: hostname_str in nl or node_id_str in nl
    )
    df = df[on_node].copy()

    if df.empty:
        return None

    w            = cfg["window"]
    window_start = pd.Timestamp(w["start"], tz="UTC")
    window_end   = pd.Timestamp(w["end"],   tz="UTC")

    records = []
    for _, row in df.iterrows():
        jid = to_int(row[jid_col])
        if jid is None:
            continue

        job_start = max(row["_start"], window_start)
        job_end   = min(row["_end"],   window_end)
        if job_end <= job_start:
            continue
        clipped_min = (job_end - job_start).total_seconds() / 60.0
        max_job_dur = float(slurm_cfg.get("max_job_duration_hours", 100)) * 60.0
        if clipped_min > max_job_dur:
            continue

        t_first = job_start.floor("60s")
        t_last  = job_end.floor("60s")

        total_cpus = to_float(row[cpus_col]) if cpus_col else None
        if total_cpus is not None and num_nodes_col is not None:
            nn = to_float(row[num_nodes_col])
            if nn and nn > 1:
                total_cpus = total_cpus / nn
        n_cpus = total_cpus

        # 0 in Slurm = "not explicitly set" → NaN, not 0.
        raw_mem_node = to_float(row[mem_per_node_col])   if mem_per_node_col   else None
        raw_mem_cpu  = to_float(row[mem_per_cpu_col])    if mem_per_cpu_col    else None
        raw_cpt      = to_float(row[cpus_per_task_col])  if cpus_per_task_col  else None
        raw_tasks    = to_float(row[tasks_col])           if tasks_col          else None
        raw_tpn      = to_float(row[tasks_per_node_col]) if tasks_per_node_col else None
        nn_val       = to_float(row[num_nodes_col])       if num_nodes_col      else None

        # Memory per node (MB): prefer explicit, else infer from mem_per_cpu × n_cpus.
        if raw_mem_node and raw_mem_node > 0:
            job_req_mem_mb = float(raw_mem_node)
        elif raw_mem_cpu and raw_mem_cpu > 0 and n_cpus and n_cpus > 0:
            job_req_mem_mb = float(raw_mem_cpu * n_cpus)
        else:
            job_req_mem_mb = np.nan

        # Tasks per node: prefer explicit, else tasks/num_nodes.
        if raw_tpn and raw_tpn > 0:
            job_req_tasks_per_node = float(raw_tpn)
        elif raw_tasks and raw_tasks > 0 and nn_val and nn_val > 0:
            job_req_tasks_per_node = float(raw_tasks / nn_val)
        else:
            job_req_tasks_per_node = np.nan

        job_req_cpus_per_task = float(raw_cpt) if raw_cpt and raw_cpt > 0 else 1.0

        ts = t_first
        while ts <= t_last:
            records.append({
                ts_col:                    ts,
                "job_id":                  jid,
                "job_num_cpus":            n_cpus,
                "job_req_mem_mb":          job_req_mem_mb,
                "job_req_tasks_per_node":  job_req_tasks_per_node,
                "job_req_cpus_per_task":   job_req_cpus_per_task,
            })
            ts += pd.Timedelta(minutes=1)

    if not records:
        return None

    result = pd.DataFrame(records)
    result[ts_col]         = pd.to_datetime(result[ts_col], utc=True)
    result["job_id"]       = pd.array(result["job_id"], dtype="Int64")
    result["job_num_cpus"]           = result["job_num_cpus"].astype("float32")
    result["job_req_mem_mb"]          = result["job_req_mem_mb"].astype("float32")
    result["job_req_tasks_per_node"]  = result["job_req_tasks_per_node"].astype("float32")
    result["job_req_cpus_per_task"]   = result["job_req_cpus_per_task"].astype("float32")
    return result.sort_values([ts_col, "job_id"]).reset_index(drop=True)


# Merge expanded jobs with snapshot data into per-minute job context.
def build_job_context(expanded: Optional[pd.DataFrame], snap: Optional[pd.DataFrame], ts_col: str) -> Optional[pd.DataFrame]:
    if expanded is None and snap is None:
        return None
    if expanded is None:
        return snap

    expanded = expanded.copy()

    expanded[ts_col] = pd.to_datetime(expanded[ts_col], utc=True).astype("datetime64[us, UTC]")

    ts_jobs:      dict = {}
    ts_ncpus:     dict = {}
    ts_req_mem:   dict = {}
    ts_req_tasks: dict = {}
    ts_req_cpt:   dict = {}
    for _, row in expanded.iterrows():
        ts  = row[ts_col]
        jid = row["job_id"]
        nc  = row["job_num_cpus"]
        if ts not in ts_jobs:
            ts_jobs[ts]      = []
            ts_ncpus[ts]     = []
            ts_req_mem[ts]   = float(row["job_req_mem_mb"])         if "job_req_mem_mb"         in row.index and not is_missing(row.get("job_req_mem_mb"))         else np.nan
            ts_req_tasks[ts] = float(row["job_req_tasks_per_node"]) if "job_req_tasks_per_node" in row.index and not is_missing(row.get("job_req_tasks_per_node")) else np.nan
            ts_req_cpt[ts]   = float(row["job_req_cpus_per_task"])  if "job_req_cpus_per_task"  in row.index and not is_missing(row.get("job_req_cpus_per_task"))  else 1.0
        if jid not in ts_jobs[ts]:
            ts_jobs[ts].append(jid)
            ts_ncpus[ts].append(nc)

    snap_lookup: dict = {}
    if snap is not None and not snap.empty:
        snap_s = snap.copy()
        snap_s[ts_col] = pd.to_datetime(snap_s[ts_col], utc=True).astype("datetime64[us, UTC]")
        snap_s = snap_s.sort_values(ts_col)
        exp_ts_list = [pd.Timestamp(t).tz_convert("UTC").as_unit("us") for t in sorted(ts_jobs.keys())]
        exp_df = pd.DataFrame({ts_col: pd.DatetimeIndex(exp_ts_list).astype("datetime64[us, UTC]")})
        exp_df = exp_df.sort_values(ts_col)
        matched = pd.merge_asof(
            exp_df,
            snap_s[[ts_col, "jobs_json", "cpu_shares_json"]].sort_values(ts_col),
            on=ts_col,
            direction="nearest",
            tolerance=pd.Timedelta(minutes=5),
        )
        for _, mrow in matched.iterrows():
            snap_lookup[mrow[ts_col]] = {
                "jobs_json":      mrow.get("jobs_json"),
                "cpu_shares_json": mrow.get("cpu_shares_json"),
            }

    records = []
    for ts in sorted(ts_jobs.keys()):
        job_ids = ts_jobs[ts]
        num_cpus = ts_ncpus[ts]

        # Resolve cpu_shares: prefer snap data for matching job_ids.
        cpu_shares: list = []
        snap_data = snap_lookup.get(ts, {})
        snap_jobs  = parse_int_list(snap_data.get("jobs_json"))
        snap_cpus  = parse_float_list(snap_data.get("cpu_shares_json"))
        snap_map   = dict(zip(snap_jobs, snap_cpus))

        for jid, nc in zip(job_ids, num_cpus):
            if jid in snap_map and not is_missing(snap_map[jid]):
                cpu_shares.append(float(snap_map[jid]))
            elif not is_missing(nc):
                cpu_shares.append(float(nc))
            else:
                cpu_shares.append(np.nan)

        primary_jid   = job_ids[0]
        primary_share = cpu_shares[0] if cpu_shares else np.nan
        valid_shares  = [c for c in cpu_shares if not is_missing(c)]
        total_share   = float(np.nansum(cpu_shares)) if valid_shares else np.nan

        records.append({
            ts_col:                    ts,
            "jobs_json":               json.dumps(job_ids),
            "cpu_shares_json":         json.dumps(
                [c if not is_missing(c) else None for c in cpu_shares]
            ),
            "active_job_count":        len(job_ids),
            "primary_job_id":          primary_jid,
            "primary_job_cpu_share":   float(primary_share) if not is_missing(primary_share) else np.nan,
            "total_job_cpu_share":     float(total_share)   if not is_missing(total_share)   else np.nan,
            "is_multi_job":            float(len(job_ids) > 1),
            "job_req_mem_mb":          ts_req_mem.get(ts, np.nan),
            "job_req_tasks_per_node":  ts_req_tasks.get(ts, np.nan),
            "job_req_cpus_per_task":   ts_req_cpt.get(ts, 1.0),
        })

    if not records:
        return snap

    result = pd.DataFrame(records)
    result[ts_col]                  = pd.to_datetime(result[ts_col], utc=True)
    result["primary_job_id"]        = pd.array(result["primary_job_id"], dtype="Int64")
    result["active_job_count"]      = result["active_job_count"].astype("int16")
    if "job_req_mem_mb" in result.columns:
        result["job_req_mem_mb"]         = result["job_req_mem_mb"].astype("float32")
        result["job_req_tasks_per_node"] = result["job_req_tasks_per_node"].astype("float32")
        result["job_req_cpus_per_task"]  = result["job_req_cpus_per_task"].astype("float32")
    result["primary_job_cpu_share"] = result["primary_job_cpu_share"].astype("float32")
    result["total_job_cpu_share"]   = result["total_job_cpu_share"].astype("float32")
    result["is_multi_job"]          = result["is_multi_job"].astype("float32")
    return (
        result.drop_duplicates(ts_col, keep="last")
        .sort_values(ts_col)
        .reset_index(drop=True)
    )


TOPO_COLS = [
    "rack",
    "rack_id",
    "rackname",
    "cabinet",
    "cabinet_id",
    "row",
    "row_id",
    "pdu",
    "pdu_id",
    "cooling_zone",
    "cooling_zone_id",
    "zone",
    "chassis",
    "cluster",
]


# Prepend hostname/component and attach topology columns.
def attach_metadata(wide: pd.DataFrame, hostname: str, component: str, node_meta: dict) -> pd.DataFrame:
    out = wide.copy()
    out.insert(0, "component", component)
    out.insert(0, "hostname", hostname)
    for col in TOPO_COLS:
        if col in node_meta and not is_missing(node_meta[col]):
            out[col] = node_meta[col]
    return out


# Build one node's wide master table (idrac + slurm + jobs).
def build_node(hostname: str, node_id: int, node_meta: dict, component: str, audited_dir: Path, raw_base: Path, out_path: Path, cfg: dict, force: bool, apply_mask: bool) -> Optional[int]:
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
                wide.loc[idle, "active_job_count"]      = 0
                wide.loc[idle, "is_multi_job"]          = 0.0
                wide.loc[idle, "primary_job_id"]        = pd.NA
                wide.loc[idle, "primary_job_cpu_share"] = np.nan
                wide.loc[idle, "total_job_cpu_share"]   = np.nan
                wide.loc[idle, "jobs_json"]             = None
                wide.loc[idle, "cpu_shares_json"]       = None
                for rc in ["job_req_mem_mb", "job_req_tasks_per_node", "job_req_cpus_per_task"]:
                    if rc in wide.columns:
                        wide.loc[idle, rc] = np.nan

            # Diagnostic: warn if any master timestamp falls in a known job window but was not matched (Slurm data inconsistency).
            if expanded is not None:
                job_ts = set(pd.to_datetime(expanded[ts_col], utc=True).dt.floor("60s"))
                master_ts = set(pd.to_datetime(wide[ts_col], utc=True))
                unmatched_job_ts = job_ts & master_ts
                unmatched_idle = wide[
                    wide[ts_col].isin(unmatched_job_ts) &
                    (wide["active_job_count"] == 0)
                ]
                if len(unmatched_idle) > 0:
                    print(f"      [WARN] {hostname}: {len(unmatched_idle)} master timestamps "
                          f"fall in a job window but show active_job_count=0 — "
                          f"check Slurm node assignment consistency")

    # Assign split BEFORE masking so apply_selective_mask can restrict NaN operations to train rows only (val/test anomalies must be preserved).
    wide = assign_split(wide, ts_col, cfg)

    master_cfg = cfg.get("master", {})
    if master_cfg.get("save_both", False):
        raw_master = Path(cfg["paths"].get(
            "master_raw",
            str(Path(cfg["paths"]["master"]).parent / "master_raw"),
        ))
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
        n_job_rows  = int((wide["active_job_count"] > 0).sum())
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


# Build one infra/PDU unit's wide master table.
def build_infra_unit(unit_id: str, source: str, audited_dir: Path, out_path: Path, cfg: dict, force: bool, apply_mask: bool) -> Optional[int]:
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
        is_train = (wide["split"] == "train") if "split" in wide.columns else pd.Series(True, index=wide.index)
        flag_cols = [c for c in wide.columns if c.startswith("audit_flags__")]
        for fc in flag_cols:
            body = fc[len("audit_flags__"):]
            sensor_col = f"{body}_avg"
            if sensor_col in wide.columns:
                target_cols = [sensor_col]
            else:
                stem = body
                target_cols = [c for c in wide.columns if c.startswith(f"{stem}__") and c.endswith("_avg")]
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


# Resolve master-build worker count from env or SLURM CPUs (cap 8).
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


# Build one node's master table, capturing stdout for the pool.
def master_build_one(kwargs: dict):
    buf = io.StringIO()
    with redirect_stdout(buf):
        n = build_node(**kwargs)
    return kwargs["hostname"], n, str(kwargs["out_path"]), buf.getvalue()


# Build per-node and infra master tables for all components.
def build_master_tables(force: bool = False) -> None:
    require_slurm_for_heavy(
        "build_master_tables", "src/preprocess/preprocess_parallel_validate.slurm"
    )
    cfg = load_config()
    audited_dir = Path(cfg["paths"]["audited"])
    raw_base = Path(cfg["paths"]["raw_parquet"])
    out_base = Path(cfg["paths"]["master"])

    apply_mask = cfg.get("master", {}).get("mask_on_audit_flags", False)
    print(
        f"\n[master] mask_on_audit_flags={apply_mask}  "
        f"({'selective: NaN spikes+bad_range, idle flatlines cleared' if apply_mask else 'keeping raw values, audit_flags passed through'})"
    )

    stage_start = time.perf_counter()
    total_rows = total_nodes = 0

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        print(f"\nComponent: {comp.upper()}")

        if comp == "infra":
            comp_start = time.perf_counter()
            source = "pdu"
            infra_audited = audited_dir / "infra" / source

            if infra_audited.exists():
                unit_ids: set = set()
                for p in sorted(infra_audited.glob("*.parquet")):
                    try:
                        df = pd.read_parquet(p, engine="pyarrow", columns=None)
                        df.columns = [c.strip().lower() for c in df.columns]
                        if "hostname" in df.columns:
                            unit_ids.update(df["hostname"].dropna().unique().tolist())
                        else:
                            nc = first_existing(
                                list(df.columns), ["nodeid", "node_id"]
                            )
                            if nc:
                                unit_ids.update(
                                    df[nc].dropna().astype(str).unique().tolist()
                                )
                    except Exception:
                        pass

                if unit_ids:
                    out_dir = out_base / "infra" / source
                    out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n  [{source.upper()}]  {len(unit_ids)} units")

                    for unit_id in sorted(unit_ids):
                        out_path = out_dir / f"{unit_id}.parquet"
                        n = build_infra_unit(
                            unit_id=str(unit_id),
                            source=source,
                            audited_dir=audited_dir,
                            out_path=out_path,
                            cfg=cfg,
                            force=force,
                            apply_mask=apply_mask,
                        )
                        if n is not None:
                            total_rows += n
                            total_nodes += 1
                        elif out_path.exists():
                            print(f"      {unit_id}: skip (exists)")
                        gc.collect()
                else:
                    print(f"  [WARN] no units found in audited/infra/{source}")

            print(f"\n  [INFRA done]  time={time.perf_counter() - comp_start:.1f}s")
            continue

        pub = load_public_tables(comp, raw_base)
        if "nodes" not in pub:
            print(f"  [WARN] no nodes table for {comp}")
            continue

        nodes_df = pub["nodes"]
        id_col = first_existing(list(nodes_df.columns), ["nodeid", "id", "node_id"])
        if id_col is None or "hostname" not in nodes_df.columns:
            print(f"  [WARN] nodes table missing id/hostname for {comp}")
            continue

        out_dir = out_base / comp
        out_dir.mkdir(parents=True, exist_ok=True)
        comp_start = time.perf_counter()

        # Per-node build is independent. workers=1 -> serial (identical); pool capped for memory (each worker holds full audited frames transiently).
        node_kwargs = [
            dict(hostname=row["hostname"], node_id=int(row[id_col]),
                 node_meta=row.to_dict(), component=comp, audited_dir=audited_dir,
                 raw_base=raw_base, out_path=out_dir / f"{row['hostname']}.parquet",
                 cfg=cfg, force=force, apply_mask=apply_mask)
            for _, row in nodes_df.iterrows()
        ]
        workers = master_workers()
        if workers == 1 or len(node_kwargs) <= 1:
            results = [master_build_one(kw) for kw in node_kwargs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(node_kwargs))) as ex:
                results = list(ex.map(master_build_one, node_kwargs))

        for hostname, n, out_path_str, log in results:
            if log.strip():
                print(log.rstrip(), flush=True)
            if n is not None:
                total_rows += n
                total_nodes += 1
            elif Path(out_path_str).exists():
                print(f"      {hostname}: skip (exists)")
        gc.collect()

        print(
            f"\n  [{comp.upper()} done]  time={time.perf_counter() - comp_start:.1f}s"
        )

    elapsed = time.perf_counter() - stage_start
    print(
        f"\nCompleted building master tables. nodes={total_nodes}  rows={total_rows:,}  time={elapsed:.1f}s"
    )


if __name__ == "__main__":
    build_master_tables(force=True)
