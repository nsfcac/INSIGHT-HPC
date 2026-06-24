from __future__ import annotations

from src.phase3.job_context_annotator_module.constants import *
import json, os, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, save_parquet, apply_node_limit

from src.phase3.job_context_annotator_module.indexes import *


# Annotate one node's physics anomalies with job context and explanations.
def annotate_node(
    hostname: str,
    component: str,
    scores_df: pd.DataFrame,
    constr_df: pd.DataFrame,
    seg_idx: dict,
    coh_idx: dict,
    feat_df: Optional[pd.DataFrame],
    node_cpus: int,
    cluster_power_lookup: dict = None,
    idrac_cpu_lookup: dict = None,
    slurm_job_lookup: dict = None,
) -> pd.DataFrame:
    anom = scores_df[scores_df["physics_anomaly"].fillna(False)].copy()
    if anom.empty:
        return pd.DataFrame()

    anom[TS] = pd.to_datetime(anom[TS], utc=True)

    c_flags: dict = {}
    if not constr_df.empty and TS in constr_df.columns:
        cv = constr_df[constr_df["hostname"] == hostname].copy()
        cv[TS] = pd.to_datetime(cv[TS], utc=True)
        for _, row in cv.iterrows():
            c_flags[row[TS]] = str(row.get("constraint_flags", ""))

    feat_lookup: dict = {}
    if feat_df is not None and TS in feat_df.columns:
        fd = feat_df.copy()
        fd[TS] = pd.to_datetime(fd[TS], utc=True)
        fd = fd[fd[TS].isin(pd.Index(anom[TS].unique()))]
        if not fd.empty:
            pwr_col = next(
                (
                    c
                    for c in feat_df.columns
                    if (
                        "systeminputpower" in c.lower()
                        or "systempowerconsumption" in c.lower()
                    )
                    and c.endswith("_avg")
                ),
                None,
            )
            idrac = idrac_cpu_lookup or {}
            ts_ser = fd[TS]

            if "slurm_cpu_load" in fd.columns:
                cpu = pd.to_numeric(fd["slurm_cpu_load"], errors="coerce") / node_cpus
                cpu = cpu.clip(upper=100.0).to_numpy(dtype="float64")
            else:
                cpu = np.full(len(fd), np.nan, dtype="float64")
            if idrac:
                need = np.isnan(cpu)
                if need.any():
                    fb = pd.to_numeric(ts_ser.map(idrac), errors="coerce").to_numpy(
                        dtype="float64"
                    )
                    cpu = np.where(need, fb, cpu)

            mem = (
                pd.to_numeric(fd["slurm_memoryusage"], errors="coerce").to_numpy(
                    dtype="float64"
                )
                if "slurm_memoryusage" in fd.columns
                else np.full(len(fd), np.nan, dtype="float64")
            )
            pwr = (
                pd.to_numeric(fd[pwr_col], errors="coerce").to_numpy(dtype="float64")
                if pwr_col and pwr_col in fd.columns
                else np.full(len(fd), np.nan, dtype="float64")
            )

            ts_list = ts_ser.to_list()
            feat_lookup = {
                ts_list[i]: {"cpu_load": cpu[i], "mem_usage": mem[i], "power_w": pwr[i]}
                for i in range(len(ts_list))
            }

    records = []
    for _, row in anom.iterrows():
        ts = row[TS]

        active = active_jobs_at(hostname, ts, seg_idx)
        job_ids = [s["job_id"] for s in active]
        n_jobs = len(active)
        active_str = "|".join(str(j) for j in sorted(set(job_ids))) if job_ids else ""

        dominant = -1
        req_cpu_frac = np.nan
        if active:
            req_cpus_vals = [
                (s["req_cpus"], s["job_id"])
                for s in active
                if not np.isnan(s["req_cpus"])
            ]
            if req_cpus_vals:
                req_cpu_frac = min(
                    1.0, sum(r for r, _ in req_cpus_vals) / max(node_cpus, 1)
                )
                dominant = max(req_cpus_vals, key=lambda x: x[0])[1]
            else:
                dominant = job_ids[0] if job_ids else -1

        feat = feat_lookup.get(ts, {})
        cpu_load = feat.get("cpu_load", np.nan)
        mem_usage = feat.get("mem_usage", np.nan)
        actual_power_w = feat.get("power_w", np.nan)

        flags = c_flags.get(ts, "")
        power_z = abs(float(row.get("power_residual_z", 0) or 0))
        thermal_z = abs(float(row.get("thermal_residual_z", 0) or 0))

        expected_power_w = np.nan
        if cluster_power_lookup is not None and dominant != -1:
            entry = cluster_power_lookup.get((hostname, dominant), {})
            expected_power_w = float(entry.get("pwr_mean", np.nan))

        isolation = node_isolation_label(hostname, job_ids, coh_idx)

        ctx, reason = classify(
            power_z=power_z,
            thermal_z=thermal_z,
            power_anom=bool(row.get("power_anomaly", False)),
            thermal_anom=bool(row.get("thermal_anomaly", False)),
            constraint_flags=flags,
            cpu_load=cpu_load,
            req_cpu_frac=req_cpu_frac,
            n_jobs=n_jobs,
            expected_power_w=expected_power_w,
            actual_power_w=actual_power_w,
            node_isolation=isolation,
        )

        power_excess_pct = np.nan
        if (
            not np.isnan(expected_power_w)
            and expected_power_w > 0
            and not np.isnan(actual_power_w)
        ):
            power_excess_pct = (actual_power_w - expected_power_w) / expected_power_w

        sjl_entry = (slurm_job_lookup or {}).get(dominant, {}) if dominant != -1 else {}
        dom_job_state = sjl_entry.get("job_state", "")
        dom_exit_code = sjl_entry.get("exit_code", "")
        dom_req_mem_mb = sjl_entry.get("effective_memory_mb", np.nan)

        if (
            dominant != -1
            and n_jobs > 0
            and ctx in ("JOB_EXPLAINED", "AMBIGUOUS")
            and (
                dom_job_state in FAILED_STATES
                or (dom_exit_code and dom_exit_code not in ("0:0", "", "None"))
            )
        ):
            orig_ctx = ctx
            ctx = "JOB_OVER_EXPECTATION"
            reason = (
                f"Dominant job {dominant} terminated abnormally "
                f"(state={dom_job_state!r}, exit_code={dom_exit_code!r}). "
                f"Anomalous metrics during a failed/timed-out job are a strong "
                f"signal of resource exhaustion or hardware fault. "
                f"(Original context was {orig_ctx}.)"
            )

        records.append(
            {
                TS: ts,
                "hostname": hostname,
                "component": component,
                "power_anomaly": bool(row.get("power_anomaly", False)),
                "thermal_anomaly": bool(row.get("thermal_anomaly", False)),
                "physics_anomaly": True,
                "power_residual_z": float(
                    row.get("power_residual_z", np.nan) or np.nan
                ),
                "thermal_residual_z": float(
                    row.get("thermal_residual_z", np.nan) or np.nan
                ),
                "constraint_flags": flags,
                "active_job_ids": active_str,
                "n_active_jobs": n_jobs,
                "dominant_job_id": dominant,
                "job_state": dom_job_state,
                "exit_code": dom_exit_code,
                "req_memory_mb": dom_req_mem_mb,
                "cpu_load": cpu_load,
                "mem_usage_pct": mem_usage,
                "req_cpu_frac": req_cpu_frac,
                "expected_power_w": expected_power_w,
                "power_excess_pct": power_excess_pct,
                "anomaly_context": ctx,
                "anomaly_reason": reason,
                "node_isolation": isolation,
                "split": str(row.get("split", "unknown")),
            }
        )

    return pd.DataFrame(records) if records else pd.DataFrame()


# Annotate constraint-violation rows with job context across all hosts.
def annotate_constraint_violations(
    cv_df: pd.DataFrame,
    seg_idx: dict,
    coh_idx: dict,
    feat_lookup_by_node: dict,
    node_cpus: int,
    node_cpus_map: dict = None,
    slurm_job_lookup: dict = None,
) -> pd.DataFrame:
    if cv_df.empty:
        return cv_df

    cv = cv_df.copy()
    cv[TS] = pd.to_datetime(cv[TS], utc=True)
    n = len(cv)

    active_job_ids_col = np.full(n, "", dtype=object)
    n_active_jobs_col = np.zeros(n, dtype=np.int32)
    dominant_job_id_col = np.full(n, -1, dtype=np.int64)
    job_state_col = np.full(n, "", dtype=object)
    exit_code_col = np.full(n, "", dtype=object)
    req_memory_mb_col = np.full(n, np.nan, dtype="float64")
    cpu_load_col = np.full(n, np.nan, dtype="float64")
    req_cpu_frac_col = np.full(n, np.nan, dtype="float64")
    anomaly_context_col = np.full(n, "", dtype=object)
    anomaly_reason_col = np.full(n, "", dtype=object)
    node_isolation_col = np.full(n, "N/A", dtype=object)

    flags_series = (
        cv.get("constraint_flags", pd.Series("", index=cv.index)).fillna("").astype(str)
    )
    violated_mask = flags_series.ne("").to_numpy()

    if not violated_mask.any():
        cv["active_job_ids"] = active_job_ids_col
        cv["n_active_jobs"] = n_active_jobs_col
        cv["dominant_job_id"] = dominant_job_id_col
        cv["job_state"] = job_state_col
        cv["exit_code"] = exit_code_col
        cv["req_memory_mb"] = req_memory_mb_col
        cv["cpu_load"] = cpu_load_col
        cv["req_cpu_frac"] = req_cpu_frac_col
        cv["anomaly_context"] = anomaly_context_col
        cv["anomaly_reason"] = anomaly_reason_col
        cv["node_isolation"] = node_isolation_col
        return cv

    viol_positions = np.where(violated_mask)[0]
    hostnames = cv["hostname"].astype(str).to_numpy()
    components = (
        cv["component"].astype(str).to_numpy()
        if "component" in cv.columns
        else np.full(n, "", dtype=object)
    )
    ts_list = cv[TS].iloc[viol_positions].to_list()
    flags_arr = flags_series.to_numpy()

    sjl = slurm_job_lookup or {}
    ncm = node_cpus_map or {}

    for i, pos in enumerate(viol_positions):
        hostname = hostnames[pos]
        ts = ts_list[i]
        comp = components[pos]
        row_cpus = ncm.get(comp, node_cpus)

        active = active_jobs_at(hostname, ts, seg_idx)
        job_ids = [s["job_id"] for s in active]
        n_jobs = len(active)
        active_str = "|".join(str(j) for j in sorted(set(job_ids))) if job_ids else ""

        req_cpu_frac = np.nan
        dominant = -1
        if active:
            req_cpus_vals = [
                (s["req_cpus"], s["job_id"])
                for s in active
                if not np.isnan(s["req_cpus"])
            ]
            if req_cpus_vals:
                req_cpu_frac = min(
                    1.0, sum(r for r, _ in req_cpus_vals) / max(row_cpus, 1)
                )
                dominant = max(req_cpus_vals, key=lambda x: x[0])[1]
            else:
                dominant = job_ids[0] if job_ids else -1

        feat = feat_lookup_by_node.get(hostname, {}).get(ts, {})
        cpu_load = feat.get("cpu_load", np.nan)

        flags = flags_arr[pos]
        isolation = node_isolation_label(hostname, job_ids, coh_idx)

        ctx, reason = classify(
            power_z=0.0,
            thermal_z=0.0,
            power_anom=False,
            thermal_anom=False,
            constraint_flags=flags,
            cpu_load=cpu_load,
            req_cpu_frac=req_cpu_frac,
            n_jobs=n_jobs,
            node_isolation=isolation,
        )

        sjl_entry = sjl.get(dominant, {}) if dominant != -1 else {}
        dom_job_state = sjl_entry.get("job_state", "")
        dom_exit_code = sjl_entry.get("exit_code", "")

        if (
            dominant != -1
            and n_jobs > 0
            and ctx in ("JOB_EXPLAINED", "AMBIGUOUS")
            and (
                dom_job_state in FAILED_STATES
                or (dom_exit_code and dom_exit_code not in ("0:0", "", "None"))
            )
        ):
            orig_ctx = ctx
            ctx = "JOB_OVER_EXPECTATION"
            reason = (
                f"Dominant job {dominant} terminated abnormally "
                f"(state={dom_job_state!r}, exit_code={dom_exit_code!r}). "
                f"Constraint violation during a failed/timed-out job is a strong "
                f"signal of resource exhaustion or hardware fault. "
                f"(Original context was {orig_ctx}.)"
            )

        active_job_ids_col[pos] = active_str
        n_active_jobs_col[pos] = n_jobs
        dominant_job_id_col[pos] = dominant
        job_state_col[pos] = dom_job_state
        exit_code_col[pos] = dom_exit_code
        req_memory_mb_col[pos] = sjl_entry.get("effective_memory_mb", np.nan)
        cpu_load_col[pos] = (
            cpu_load
            if not (isinstance(cpu_load, float) and np.isnan(cpu_load))
            else np.nan
        )
        req_cpu_frac_col[pos] = req_cpu_frac
        anomaly_context_col[pos] = ctx
        anomaly_reason_col[pos] = reason
        node_isolation_col[pos] = isolation

    cv["active_job_ids"] = active_job_ids_col
    cv["n_active_jobs"] = n_active_jobs_col
    cv["dominant_job_id"] = dominant_job_id_col
    cv["job_state"] = job_state_col
    cv["exit_code"] = exit_code_col
    cv["req_memory_mb"] = req_memory_mb_col
    cv["cpu_load"] = cpu_load_col
    cv["req_cpu_frac"] = req_cpu_frac_col
    cv["anomaly_context"] = anomaly_context_col
    cv["anomaly_reason"] = anomaly_reason_col
    cv["node_isolation"] = node_isolation_col
    return cv


CTX_PRIORITY = {
    "INFRA_FAULT": 3,
    "COOLING_FAULT": 2,
    "JOB_OVER_EXPECTATION": 1,
}


# Parse a node-list cell (string or list) into a list of hostnames.
def split_nodes(raw: Any) -> list:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple, set, np.ndarray)):
        return [str(x) for x in raw if x]
    s = str(raw).strip()
    if not s:
        return []
    for ch in "[]()'\"":
        s = s.replace(ch, "")
    parts = [p.strip() for p in s.replace("|", ",").split(",")]
    return [p for p in parts if p]


# Adjust phase-2 alert confidence using each node's physics context.
def enrich_attributed_alerts(annotated: pd.DataFrame, phase2_dir: Path) -> None:
    alerts_path = phase2_dir / "attributed_alerts.parquet"
    if not alerts_path.exists() or annotated.empty:
        return

    alerts = pd.read_parquet(alerts_path, engine="pyarrow")
    if alerts.empty or "attribution_confidence" not in alerts.columns:
        return

    ctx_df = annotated[["timestamp", "hostname", "anomaly_context"]].copy()
    ctx_df["timestamp"] = pd.to_datetime(ctx_df["timestamp"], utc=True).dt.floor("1min")
    physics_ctx: dict = dict(
        zip(
            zip(ctx_df["hostname"].astype(str), ctx_df["timestamp"]),
            ctx_df["anomaly_context"].astype(str),
        )
    )

    if not physics_ctx:
        return

    phys_col = []
    adj_col = []
    for _, row in alerts.iterrows():
        nodes = split_nodes(row.get("anomalous_nodes"))
        conf = float(row.get("attribution_confidence", float("nan")))
        if not nodes or pd.isna(row.get("first_flag_time")):
            phys_col.append("")
            adj_col.append(conf)
            continue

        ts_floor = pd.Timestamp(row["first_flag_time"]).floor("1min")
        best_ctx = ""
        best_pri = -1
        for h in nodes:
            ctx = physics_ctx.get((h, ts_floor), "")
            pri = CTX_PRIORITY.get(ctx, 0 if ctx else -1)
            if pri > best_pri:
                best_ctx, best_pri = ctx, pri

        if best_ctx in ("INFRA_FAULT", "COOLING_FAULT"):
            conf = max(0.0, conf - 0.15)
        elif best_ctx == "JOB_OVER_EXPECTATION":
            conf = min(1.0, conf + 0.15)
        phys_col.append(best_ctx)
        adj_col.append(conf)

    alerts["physics_context"] = phys_col
    alerts["attribution_confidence"] = pd.array(adj_col, dtype="float32")
    save_parquet(alerts, alerts_path)
    n_enriched = sum(1 for c in phys_col if c)
    print(
        f"  Enriched attributed_alerts: {n_enriched} rows got physics_context → {alerts_path}"
    )


# Resolve the job-context worker count from env / SLURM allocation.
def job_context_workers() -> int:
    env = os.environ.get("INSIGHT_HPC_PHASE3_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(10, int(slurm_cpus)))
        except ValueError:
            return 1
    return 1


PHASE3_CONTEXT: dict = {}
