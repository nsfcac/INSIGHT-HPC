from __future__ import annotations

from offline.phase3.job_context_annotator_module.constants import *
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd



# Build a job_id to {state, exit_code, memory} lookup from Slurm jobs.
def load_slurm_job_lookup(cfg: dict) -> dict:
    raw_base = Path(cfg["paths"].get("raw_parquet", "offline/data/raw_parquet"))
    lookup: dict = {}
    cols = [
        "job_id",
        "job_state",
        "exit_code",
        "memory_per_node",
        "memory_per_cpu",
        "cpus",
    ]
    for comp_cfg in cfg.get("components", []):
        comp_name = comp_cfg["name"]
        slurm_path = raw_base / comp_name / "slurm" / "jobs.parquet"
        if not slurm_path.exists():
            continue
        try:
            df = pd.read_parquet(slurm_path, engine="pyarrow")
        except Exception:
            continue

        for col in cols:
            if col not in df.columns:
                df[col] = None

        df["job_id"] = pd.to_numeric(df["job_id"], errors="coerce")
        df = df.dropna(subset=["job_id"])
        df["job_id"] = df["job_id"].astype(int)

        mpn = pd.to_numeric(df["memory_per_node"], errors="coerce").fillna(0)
        mpc = pd.to_numeric(df["memory_per_cpu"], errors="coerce").fillna(0)
        ncp = pd.to_numeric(df["cpus"], errors="coerce").fillna(1)
        df["effective_memory_mb"] = np.where(mpn > 0, mpn, mpc * ncp)

        for row in df.itertuples(index=False):
            jid = int(row.job_id)
            if jid not in lookup:
                lookup[jid] = {
                    "job_state": (
                        str(row.job_state) if row.job_state is not None else ""
                    ),
                    "exit_code": (
                        str(row.exit_code) if row.exit_code is not None else ""
                    ),
                    "effective_memory_mb": float(row.effective_memory_mb),
                }
    return lookup


# Index non-idle job segments by hostname with their requested resources.
def build_segment_index(segs: pd.DataFrame) -> dict:
    idx: dict = defaultdict(list)
    for _, row in segs.iterrows():
        if row.get("is_idle", False):
            continue
        idx[row["hostname"]].append(
            {
                "job_id": int(row["job_id"]) if pd.notna(row.get("job_id")) else -1,
                "seg_start": pd.Timestamp(row["seg_start"]).tz_convert("UTC"),
                "seg_end": pd.Timestamp(row["seg_end"]).tz_convert("UTC"),
                "req_cpus": (
                    float(row["req_cpus"])
                    if "req_cpus" in row.index and pd.notna(row["req_cpus"])
                    else np.nan
                ),
                "req_mem_mb": (
                    float(row["req_mem_mb"])
                    if "req_mem_mb" in row.index and pd.notna(row["req_mem_mb"])
                    else np.nan
                ),
            }
        )
    return idx


# Return the job segments active on a host at one timestamp.
def active_jobs_at(hostname: str, ts: pd.Timestamp, seg_idx: dict) -> list[dict]:
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return [
        seg
        for seg in seg_idx.get(hostname, [])
        if seg["seg_start"] <= ts_utc <= seg["seg_end"]
    ]


# Index per-(host, job) coherence-anomaly flags.
def build_coherence_index(coh: Optional[pd.DataFrame]) -> dict:
    if coh is None or coh.empty:
        return {}
    idx = {}
    for _, row in coh.iterrows():
        jid = int(row["job_id"]) if pd.notna(row.get("job_id")) else -1
        idx[(str(row["hostname"]), jid)] = bool(row.get("is_coherence_anomaly", False))
    return idx


# Build a (host, job) to cluster power mean/std lookup from clustered profiles.
def load_cluster_power_lookup(phase2_dir: Path) -> dict:
    clustered_path = phase2_dir / "job_profiles_clustered.parquet"
    if not clustered_path.exists():
        return {}
    try:
        df = pd.read_parquet(
            clustered_path,
            engine="pyarrow",
            columns=["hostname", "job_id", "pwr_mean", "pwr_std"],
        )
        df = df.dropna(subset=["job_id"])
        df["job_id"] = df["job_id"].astype(int)
        return {
            (str(row.hostname), int(row.job_id)): {
                "pwr_mean": float(row.pwr_mean) if pd.notna(row.pwr_mean) else np.nan,
                "pwr_std": float(row.pwr_std) if pd.notna(row.pwr_std) else np.nan,
            }
            for row in df.itertuples()
        }
    except Exception:
        return {}


# Classify a physics anomaly into a job-context label with a human-readable reason.
def classify(
    power_z: float,
    thermal_z: float,
    power_anom: bool,
    thermal_anom: bool,
    constraint_flags: str,
    cpu_load: float,
    req_cpu_frac: float,
    n_jobs: int,
    expected_power_w: float = np.nan,
    actual_power_w: float = np.nan,
    node_isolation: str = "N/A",
) -> tuple[str, str]:
    flags = set((constraint_flags or "").split("|")) - {""}

    if n_jobs == 0:
        if power_anom or thermal_anom:
            return (
                "NO_JOBS",
                "Node is idle (no SLURM jobs) but shows anomalous power/thermal "
                "— likely hardware degradation or persistent power leak.",
            )
        if flags:
            return (
                "NO_JOBS",
                f"Physics constraint violated ({', '.join(flags)}) while node is idle "
                "— infrastructure issue unrelated to workload.",
            )

    if "CROSSPLANE" in flags:
        return (
            "MEASUREMENT_DISCREPANCY",
            "Rack PDU power increased while the summed node power in that rack "
            "stayed effectively unchanged — cross-plane disagreement, unmetered "
            "rack load, or measurement inconsistency.",
        )

    cpu_load_known = not np.isnan(cpu_load)
    cpu_high = cpu_load_known and cpu_load >= CPU_LOAD_HIGH
    cpu_low = not cpu_load_known or cpu_load < CPU_LOAD_LOW
    req_heavy = not np.isnan(req_cpu_frac) and req_cpu_frac >= REQ_CPU_HEAVY

    workload_explains = (cpu_high or (req_heavy and cpu_load_known)) and n_jobs > 0
    workload_req_only = req_heavy and not cpu_load_known and n_jobs > 0

    # ISOLATED: only this node anomalous while peers on same job are normal → HW fault.
    if node_isolation == "ISOLATED" and power_anom and not workload_explains:
        return (
            "INFRA_FAULT",
            f"Node ISOLATED from job peer group (power z={power_z:.1f}) — "
            "other nodes on the same job are normal; hardware fault on this node.",
        )
    if node_isolation == "CLUSTER_WIDE" and (workload_explains or workload_req_only):
        if np.isnan(power_z) or abs(power_z) < POWER_Z_MEDIUM:
            return (
                "JOB_EXPLAINED",
                "Power elevated across all nodes of this multi-node job (CLUSTER_WIDE) — "
                "consistent with distributed job demand.",
            )

    # JOB_OVER_EXPECTATION: power anomaly far exceeds the cluster profile.
    if (
        power_anom
        and n_jobs > 0
        and not np.isnan(expected_power_w)
        and expected_power_w > 0
        and not np.isnan(actual_power_w)
    ):
        excess_pct = (actual_power_w - expected_power_w) / expected_power_w
        if excess_pct > POWER_EXCESS_THRESHOLD:
            rcf_str = f"{req_cpu_frac:.2f}" if not np.isnan(req_cpu_frac) else "N/A"
            return (
                "JOB_OVER_EXPECTATION",
                f"Actual power ({actual_power_w:.0f}W) exceeds cluster-profile expected "
                f"({expected_power_w:.0f}W) by {100 * excess_pct:.0f}% "
                f"(req_cpu_frac={rcf_str}) — job consumes far more power than its "
                "workload cluster predicts; possible thermal runaway, memory thrashing, "
                "or abnormal instruction mix.",
            )

    # Thermal anomaly
    if thermal_anom and not power_anom:
        if "TEMP_FAN" in flags:
            if workload_explains:
                return (
                    "COOLING_FAULT",
                    f"Inlet temperature rising while fan speed is low; node CPU load "
                    f"is {cpu_load:.0f}% — fans are not keeping up with compute heat "
                    f"(possible fan failure or blockage).",
                )
            return (
                "COOLING_FAULT",
                "Inlet temperature rising with low fan response and low CPU load "
                "— thermal anomaly is not workload-driven; suspect cooling system fault.",
            )
        if workload_explains:
            if abs(thermal_z) < THERMAL_Z_HIGH:
                return (
                    "JOB_EXPLAINED",
                    f"Thermal residual elevated (z={thermal_z:.1f}) with CPU load "
                    f"{cpu_load:.0f}% — expected thermal lag from active workload.",
                )
            return (
                "AMBIGUOUS",
                f"Thermal residual strongly elevated (z={thermal_z:.1f}) despite "
                f"high CPU load ({cpu_load:.0f}%) — magnitude exceeds expected "
                f"thermal lag; possible cooling degradation coinciding with job.",
            )
        return (
            "COOLING_FAULT",
            f"Thermal anomaly (z={thermal_z:.1f}) without corresponding power elevation "
            "or sufficient workload — suspect cooling system degradation or inlet "
            "recirculation.",
        )

    # Physics dynamics violation: power ramps, temp doesn't follow.
    if "DYNAMICS" in flags and not thermal_anom:
        return (
            "SENSOR_DRIFT",
            "Over the previous 10 minutes power increased significantly but inlet "
            "temperature failed to rise — likely sensor lag, firmware reporting "
            "delay, or abnormal thermal coupling.",
        )

    # Power anomaly
    if power_anom:
        if workload_explains:
            if abs(power_z) < POWER_Z_MEDIUM:
                return (
                    "JOB_EXPLAINED",
                    f"Power residual elevated (z={power_z:.1f}) with CPU load "
                    f"{cpu_load:.0f}% — power proportional to active workload; "
                    f"within expected variance.",
                )
            return (
                "AMBIGUOUS",
                f"Power residual strongly elevated (z={power_z:.1f}) despite "
                f"high CPU load ({cpu_load:.0f}%) — workload partially explains "
                f"the increase but magnitude exceeds profile prediction; possible "
                f"thermal throttling or abnormal instruction mix.",
            )
        if workload_req_only:
            rcf_str = f"{req_cpu_frac:.2f}" if not np.isnan(req_cpu_frac) else "N/A"
            if abs(power_z) < POWER_Z_MEDIUM:
                return (
                    "JOB_EXPLAINED",
                    f"Power residual elevated (z={power_z:.1f}); actual CPU load "
                    f"unavailable — inferred from req_cpu_frac={rcf_str}; power "
                    f"within expected range for this workload.",
                )
            return (
                "AMBIGUOUS",
                f"Power residual strongly elevated (z={power_z:.1f}); actual CPU "
                f"load unavailable (req_cpu_frac={rcf_str}) — cannot confirm "
                f"workload explains magnitude; manual review recommended.",
            )
        if cpu_low:
            return (
                "INFRA_FAULT",
                f"Power residual elevated (z={power_z:.1f}) with low CPU utilisation "
                f"({cpu_load:.0f}% or unknown) — power anomaly NOT explained by "
                f"active jobs; suspect hardware fault, idle power leak, or untracked "
                f"device draw.",
            )
        return (
            "AMBIGUOUS",
            f"Power anomaly (z={power_z:.1f}) with moderate CPU load ({cpu_load:.0f}%) "
            "— partial workload explanation; review job profile vs actual consumption.",
        )

    if flags:
        if "TEMP_FAN" in flags:
            return (
                "COOLING_FAULT",
                f"Temp-fan decoupling detected ({', '.join(flags)}): temperature "
                "rising without fan response. CPU load is "
                f"{'high' if cpu_high else 'normal'} ({cpu_load:.0f}%).",
            )
        return (
            "AMBIGUOUS",
            f"Constraint violation ({', '.join(flags)}) without clear power or thermal "
            "anomaly — likely transient measurement artifact or boundary condition.",
        )

    return ("AMBIGUOUS", "No clear classification; anomaly is marginal or transient.")


# Label a node ISOLATED or CLUSTER_WIDE based on peer coherence.
def node_isolation_label(hostname: str, job_ids: list[int], coh_idx: dict) -> str:
    if not job_ids or not coh_idx:
        return "N/A"
    isolated_jobs = []
    cluster_wide = []
    for jid in job_ids:
        flag = coh_idx.get((hostname, jid), None)
        if flag is None:
            continue
        if flag:
            isolated_jobs.append(jid)
        else:
            cluster_wide.append(jid)
    if cluster_wide:
        return "CLUSTER_WIDE"
    if isolated_jobs:
        return "ISOLATED"
    return "N/A"
