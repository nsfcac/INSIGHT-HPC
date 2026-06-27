from __future__ import annotations

from offline.preprocess.build_master_tables_module.constants import *
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.parsers import (
    first_existing,
    is_missing,
    parse_float_list,
    parse_int_list,
    parse_slurm_ts,
    parse_str_list,
    to_float,
    to_int,
)

from offline.preprocess.build_master_tables_module.metric_loading import *


# Expand a node's Slurm jobs into a per-minute job-requirement timeline.
def load_jobs_for_node(
    slurm_raw_dir: Path, hostname: str, node_id: int, ts_col: str, cfg: dict
) -> Optional[pd.DataFrame]:
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

    jid_col = first_existing(list(df.columns), ["jobid", "job_id", "id"])
    start_col = first_existing(list(df.columns), ["start_time", "starttime", "start"])
    end_col = first_existing(list(df.columns), ["end_time", "endtime", "end"])
    nodes_col = first_existing(
        list(df.columns), ["nodes", "nodelist", "node_list", "alloc_nodes", "nodenames"]
    )
    state_col = first_existing(list(df.columns), ["job_state", "state", "jobstate"])
    cpus_col = first_existing(
        list(df.columns),
        ["cpus", "num_cpus", "ncpus", "num_nodes_cpus", "cpus_per_node"],
    )
    # num_nodes converts total CPUs -> per-node CPU allocation for multi-node jobs.
    num_nodes_col = first_existing(
        list(df.columns), ["num_nodes", "nnodes", "node_count", "alloc_nodes_count"]
    )
    # 0 in Slurm = "not set" — treat as NaN, not 0.
    mem_per_node_col = first_existing(
        list(df.columns), ["min_mem_per_node", "mem_per_node", "mem", "memory_per_node"]
    )
    mem_per_cpu_col = first_existing(
        list(df.columns), ["mem_per_cpu", "min_mem_per_cpu", "memory_per_cpu"]
    )
    cpus_per_task_col = first_existing(
        list(df.columns), ["cpus_per_task", "cpus_per_tre", "num_cpus_per_task"]
    )
    tasks_col = first_existing(list(df.columns), ["tasks", "num_tasks", "ntasks"])
    tasks_per_node_col = first_existing(
        list(df.columns), ["tasks_per_node", "ntasks_per_node", "num_tasks_per_node"]
    )

    if any(c is None for c in [jid_col, start_col, end_col, nodes_col]):
        print(
            f"      [WARN] jobs.parquet missing required columns "
            f"(need job_id, start_time, end_time, nodes).  "
            f"Found: {list(df.columns[:10])}"
        )
        return None

    df["_start"] = parse_slurm_ts(df[start_col])
    df["_end"] = parse_slurm_ts(df[end_col])

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
        # Treat a job as having run unless every reported state is a never-ran state.
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
    node_id_str = str(node_id)
    on_node = node_lists.apply(lambda nl: hostname_str in nl or node_id_str in nl)
    df = df[on_node].copy()

    if df.empty:
        return None

    w = cfg["window"]
    window_start = pd.Timestamp(w["start"], tz="UTC")
    window_end = pd.Timestamp(w["end"], tz="UTC")

    records = []
    for _, row in df.iterrows():
        jid = to_int(row[jid_col])
        if jid is None:
            continue

        job_start = max(row["_start"], window_start)
        job_end = min(row["_end"], window_end)
        if job_end <= job_start:
            continue
        clipped_min = (job_end - job_start).total_seconds() / 60.0
        max_job_dur = float(slurm_cfg.get("max_job_duration_hours", 100)) * 60.0
        if clipped_min > max_job_dur:
            continue

        t_first = job_start.floor("60s")
        t_last = job_end.floor("60s")

        total_cpus = to_float(row[cpus_col]) if cpus_col else None
        if total_cpus is not None and num_nodes_col is not None:
            nn = to_float(row[num_nodes_col])
            if nn and nn > 1:
                total_cpus = total_cpus / nn
        n_cpus = total_cpus

        # 0 in Slurm = "not explicitly set" → NaN, not 0.
        raw_mem_node = to_float(row[mem_per_node_col]) if mem_per_node_col else None
        raw_mem_cpu = to_float(row[mem_per_cpu_col]) if mem_per_cpu_col else None
        raw_cpt = to_float(row[cpus_per_task_col]) if cpus_per_task_col else None
        raw_tasks = to_float(row[tasks_col]) if tasks_col else None
        raw_tpn = to_float(row[tasks_per_node_col]) if tasks_per_node_col else None
        nn_val = to_float(row[num_nodes_col]) if num_nodes_col else None

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
            records.append(
                {
                    ts_col: ts,
                    "job_id": jid,
                    "job_num_cpus": n_cpus,
                    "job_req_mem_mb": job_req_mem_mb,
                    "job_req_tasks_per_node": job_req_tasks_per_node,
                    "job_req_cpus_per_task": job_req_cpus_per_task,
                }
            )
            ts += pd.Timedelta(minutes=1)

    if not records:
        return None

    result = pd.DataFrame(records)
    result[ts_col] = pd.to_datetime(result[ts_col], utc=True)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")
    result["job_num_cpus"] = result["job_num_cpus"].astype("float32")
    result["job_req_mem_mb"] = result["job_req_mem_mb"].astype("float32")
    result["job_req_tasks_per_node"] = result["job_req_tasks_per_node"].astype(
        "float32"
    )
    result["job_req_cpus_per_task"] = result["job_req_cpus_per_task"].astype("float32")
    return result.sort_values([ts_col, "job_id"]).reset_index(drop=True)


# Merge expanded job windows with snapshot cpu-share polling into per-minute context.
def build_job_context(
    expanded: Optional[pd.DataFrame], snap: Optional[pd.DataFrame], ts_col: str
) -> Optional[pd.DataFrame]:
    if expanded is None and snap is None:
        return None
    if expanded is None:
        return snap

    expanded = expanded.copy()

    expanded[ts_col] = pd.to_datetime(expanded[ts_col], utc=True).astype(
        "datetime64[us, UTC]"
    )

    ts_jobs: dict = {}
    ts_ncpus: dict = {}
    ts_req_mem: dict = {}
    ts_req_tasks: dict = {}
    ts_req_cpt: dict = {}
    for _, row in expanded.iterrows():
        ts = row[ts_col]
        jid = row["job_id"]
        nc = row["job_num_cpus"]
        if ts not in ts_jobs:
            ts_jobs[ts] = []
            ts_ncpus[ts] = []
            ts_req_mem[ts] = (
                float(row["job_req_mem_mb"])
                if "job_req_mem_mb" in row.index
                and not is_missing(row.get("job_req_mem_mb"))
                else np.nan
            )
            ts_req_tasks[ts] = (
                float(row["job_req_tasks_per_node"])
                if "job_req_tasks_per_node" in row.index
                and not is_missing(row.get("job_req_tasks_per_node"))
                else np.nan
            )
            ts_req_cpt[ts] = (
                float(row["job_req_cpus_per_task"])
                if "job_req_cpus_per_task" in row.index
                and not is_missing(row.get("job_req_cpus_per_task"))
                else 1.0
            )
        if jid not in ts_jobs[ts]:
            ts_jobs[ts].append(jid)
            ts_ncpus[ts].append(nc)

    snap_lookup: dict = {}
    if snap is not None and not snap.empty:
        snap_s = snap.copy()
        snap_s[ts_col] = pd.to_datetime(snap_s[ts_col], utc=True).astype(
            "datetime64[us, UTC]"
        )
        snap_s = snap_s.sort_values(ts_col)
        exp_ts_list = [
            pd.Timestamp(t).tz_convert("UTC").as_unit("us")
            for t in sorted(ts_jobs.keys())
        ]
        exp_df = pd.DataFrame(
            {ts_col: pd.DatetimeIndex(exp_ts_list).astype("datetime64[us, UTC]")}
        )
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
                "jobs_json": mrow.get("jobs_json"),
                "cpu_shares_json": mrow.get("cpu_shares_json"),
            }

    records = []
    for ts in sorted(ts_jobs.keys()):
        job_ids = ts_jobs[ts]
        num_cpus = ts_ncpus[ts]

        # Resolve cpu_shares: prefer snap data for matching job_ids.
        cpu_shares: list = []
        snap_data = snap_lookup.get(ts, {})
        snap_jobs = parse_int_list(snap_data.get("jobs_json"))
        snap_cpus = parse_float_list(snap_data.get("cpu_shares_json"))
        snap_map = dict(zip(snap_jobs, snap_cpus))

        for jid, nc in zip(job_ids, num_cpus):
            if jid in snap_map and not is_missing(snap_map[jid]):
                cpu_shares.append(float(snap_map[jid]))
            elif not is_missing(nc):
                cpu_shares.append(float(nc))
            else:
                cpu_shares.append(np.nan)

        primary_jid = job_ids[0]
        primary_share = cpu_shares[0] if cpu_shares else np.nan
        valid_shares = [c for c in cpu_shares if not is_missing(c)]
        total_share = float(np.nansum(cpu_shares)) if valid_shares else np.nan

        records.append(
            {
                ts_col: ts,
                "jobs_json": json.dumps(job_ids),
                "cpu_shares_json": json.dumps(
                    [c if not is_missing(c) else None for c in cpu_shares]
                ),
                "active_job_count": len(job_ids),
                "primary_job_id": primary_jid,
                "primary_job_cpu_share": (
                    float(primary_share) if not is_missing(primary_share) else np.nan
                ),
                "total_job_cpu_share": (
                    float(total_share) if not is_missing(total_share) else np.nan
                ),
                "is_multi_job": float(len(job_ids) > 1),
                "job_req_mem_mb": ts_req_mem.get(ts, np.nan),
                "job_req_tasks_per_node": ts_req_tasks.get(ts, np.nan),
                "job_req_cpus_per_task": ts_req_cpt.get(ts, 1.0),
            }
        )

    if not records:
        return snap

    result = pd.DataFrame(records)
    result[ts_col] = pd.to_datetime(result[ts_col], utc=True)
    result["primary_job_id"] = pd.array(result["primary_job_id"], dtype="Int64")
    result["active_job_count"] = result["active_job_count"].astype("int16")
    if "job_req_mem_mb" in result.columns:
        result["job_req_mem_mb"] = result["job_req_mem_mb"].astype("float32")
        result["job_req_tasks_per_node"] = result["job_req_tasks_per_node"].astype(
            "float32"
        )
        result["job_req_cpus_per_task"] = result["job_req_cpus_per_task"].astype(
            "float32"
        )
    result["primary_job_cpu_share"] = result["primary_job_cpu_share"].astype("float32")
    result["total_job_cpu_share"] = result["total_job_cpu_share"].astype("float32")
    result["is_multi_job"] = result["is_multi_job"].astype("float32")
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


# Attach hostname, component, and topology columns to a node's table.
def attach_metadata(
    wide: pd.DataFrame, hostname: str, component: str, node_meta: dict
) -> pd.DataFrame:
    out = wide.copy()
    out.insert(0, "component", component)
    out.insert(0, "hostname", hostname)
    for col in TOPO_COLS:
        if col in node_meta and not is_missing(node_meta[col]):
            out[col] = node_meta[col]
    return out
