from __future__ import annotations

from typing import Optional

import pandas as pd

from src.phase4.operator_alert_module.data import CATEGORY_TEMPLATES
from src.phase4.operator_alert_module.phase import (
    caught_by_phases,
    detectors_for_phase,
    format_caught_by,
    format_duration,
    phase_label_short,
)


# Build the "What happened" context lines (window, jobs, job-state).
def what_happened_context(ep: pd.Series) -> list[str]:
    lines: list[str] = []
    ep_start = pd.to_datetime(ep.get("episode_start"), utc=True, errors="coerce")
    ep_end = pd.to_datetime(ep.get("episode_end"), utc=True, errors="coerce")
    if pd.notna(ep_start) and pd.notna(ep_end):
        displayed_min = (ep_end - ep_start).total_seconds() / 60.0
        lines.append(
            f"- Displayed window: `{ep_start.strftime('%Y-%m-%d %H:%M')} UTC` → "
            f"`{ep_end.strftime('%Y-%m-%d %H:%M')} UTC` ({displayed_min:.0f} min)"
        )
        persisted = float(ep.get("ep_duration_min", 0) or 0)
        if persisted > displayed_min * 1.5 and persisted > 60.0:
            lines.append(
                f"- Anomaly persistence: this node was continuously flagged "
                f"for **{format_duration(persisted)}** in the underlying "
                f"v2 episode — the figures above show only the most-anomalous "
                f"{displayed_min:.0f}-min slice."
            )

    ids = str(ep.get("active_job_ids", "") or "").strip()
    job_list = [j for j in ids.split("|") if j] if ids else []
    ctx = str(ep.get("anomaly_context", "")).upper()
    coh_z = float(ep.get("coherence_z", 0.0) or 0.0)
    peer_z = float(ep.get("phase2_peer_divergence_z", 0.0) or 0.0)
    has_p2_signal = (coh_z >= 2.0) or (peer_z >= 2.0)

    if job_list:
        if len(job_list) == 1:
            lines.append(f"- Active SLURM job during the alert window: `{job_list[0]}`")
        elif len(job_list) <= 3:
            lines.append(
                "- Active SLURM jobs during the alert window: "
                + ", ".join(f"`{j}`" for j in job_list)
            )
        else:
            lines.append(
                "- Active SLURM jobs during the alert window: "
                + ", ".join(f"`{j}`" for j in job_list[:3])
                + f" (+{len(job_list) - 3} more)"
            )
    elif has_p2_signal:
        lines.append(
            "- This node was a peer in a multi-node job during at least "
            "part of the alert window — phase-2 coherence fired on within-job "
            "feature divergence (see `13_peer_*.png` and `15_peer_active_jobs.png` "
            "for the per-minute job timeline)."
        )
    elif ctx == "NO_JOBS":
        lines.append(
            "- No SLURM job was active on this node during the alert "
            "window — the anomaly is workload-independent."
        )
    elif ctx == "JOB_EXPLAINED":
        lines.append(
            "- Phase-3 marked the local profile change as explained by a "
            "job transition (low phase-2 confidence)."
        )
    elif ctx and ctx not in ("", "NONE"):
        lines.append(f"- Phase-3 job-state label: `{ctx}`")
    return lines


# Write a plain-language summary of the anomaly and why each phase fired.
def plain_language_summary(category: str, ep: pd.Series, caught_by: list[str]) -> str:
    host = str(ep.get("hostname", ""))
    if category == "unknown":
        cat_label = "multi-phase anomaly"
    else:
        cat_label = category.replace("_", " ")
    why = ""
    if "phase1_score" in caught_by:
        p1_score = float(ep.get("phase1_score", 0.0) or 0.0)
        any_flag = (
            bool(ep.get("if_is_anomaly", False))
            or bool(ep.get("lstm_is_anomaly", False))
            or bool(ep.get("zscore_flag", False))
            or bool(ep.get("ewma_flag", False))
            or any(
                bool(ep.get(c, False))
                for c in ep.index
                if isinstance(c, str) and c.startswith("rule_")
            )
        )
        if p1_score >= 4.0 and not any_flag:
            why += (
                "The combined sensor-detector signal is at saturation across "
                "the multi-detector ensemble — sustained off-distribution "
                "behavior on this node, even though no single detector tripped "
                "its hard threshold. "
            )
        else:
            why += "Sensor readings on this node crossed a hard threshold or showed an unusual pattern. "
    if "phase2_score" in caught_by:
        peer_z = float(ep.get("phase2_peer_divergence_z", 0.0) or 0.0)
        cluster_dst = float(ep.get("phase2_cluster_dist", 0.0) or 0.0)
        coh_z = float(ep.get("coherence_z", 0.0) or 0.0)
        is_multi_node = (peer_z >= 2.0) or (coh_z >= 2.0)
        is_cluster_outlier = cluster_dst >= 2.0
        if is_multi_node and is_cluster_outlier:
            why += (
                "This node diverges both from its same-job peers AND from "
                "the workload-cluster of similar-workload nodes — peers and "
                "the cluster centroid both look normal, this node alone is "
                "the outlier. "
            )
        elif is_multi_node:
            why += (
                "This node's behavior differs from other nodes running the "
                "same job — its same-job peers look normal, but this node "
                "alone is the outlier. "
            )
        elif is_cluster_outlier:
            why += (
                "This node's workload profile diverges from its assigned "
                "workload cluster — among nodes running similar workloads "
                "(the cluster of similar-profile jobs), this node sits far "
                "from the cluster centroid. "
            )
        else:
            why += (
                "This node's job-context profile (workload features, peer / "
                "cluster reference) looks unusual on multiple fronts. "
            )
    if "phase3_score" in caught_by:
        why += (
            "The physics model expected one power/thermal pattern given the "
            "workload, and the actual measurements diverged from that expectation. "
        )
    if not why:
        why = "Telemetry from this node looked unusual on multiple fronts. "
    return f"INSIGHT-HPC flagged a likely **{cat_label}** on `{host}`. {why.strip()}"


# Summarise fusion probability, tier, and how many phases contributed.
def strength_summary(
    ep: pd.Series, norm: dict[str, float], caught_by: Optional[list[str]] = None
) -> str:
    fp = float(ep.get("fusion_prob", 0.0) or 0.0)
    tier = str(ep.get("fusion_tier", ""))
    if caught_by is not None:
        n = len(caught_by)
    else:
        n = sum(1 for v in norm.values() if v >= 0.25)
    plural = "phase" if n == 1 else "phases"
    return (
        f"fusion_prob = {fp:.3f} ({tier} tier) — "
        f"{n} of 3 {plural} contributed to the headline"
    )


# Phrase how far ahead of the GT event the alert fired.
def format_lead_time(ep: pd.Series) -> Optional[str]:
    lt = ep.get("lead_time_min")
    if lt is None or (isinstance(lt, float) and not pd.notna(lt)):
        return None
    try:
        lt = float(lt)
    except (TypeError, ValueError):
        return None
    if lt <= 0:
        return None
    if lt < 60:
        return f"alert fired **{lt:.0f} min** before the ground-truth event start"
    hrs = lt / 60.0
    return f"alert fired **{hrs:.1f} h** before the ground-truth event start"


def format_section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


# Fill a template with episode fields, falling back safely on missing keys.
def format_template_safe(template: str, ep: pd.Series, rack: str) -> str:
    metric = str(ep.get("metric_name", "")).strip()
    if not metric:
        ctx_word = str(ep.get("anomaly_context", "")).upper()
        if "FAN" in ctx_word:
            metric = "a fan channel"
        elif "POWER" in ctx_word:
            metric = "a power channel"
        elif "TEMP" in ctx_word or "THERM" in ctx_word or "COOLING" in ctx_word:
            metric = "a temperature channel"
        else:
            metric = "the affected sensor"
    n_fam = int(ep.get("ep_n_families_dark", ep.get("n_dropout_families", 2)) or 2)
    ctx = {
        "host": str(ep.get("hostname", "")),
        "rack": rack,
        "metric": metric,
        "value": float(ep.get("metric_value", 0) or 0),
        "threshold": float(ep.get("metric_threshold", 0) or 0),
        "pct": 100 * float(ep.get("metric_pct_of_baseline", 0.5) or 0.5),
        "delta_w": float(ep.get("idle_power_delta_w", 0) or 0),
        "mismatch_w": float(ep.get("pdu_mismatch_w", 0) or 0),
        "peer_nodes": str(ep.get("peer_nodes", "peer job members")),
        "n_families": n_fam,
        "family_clause": family_clause(n_fam),
        "bmc_hint": bmc_hint(n_fam, str(ep.get("hostname", ""))),
    }
    try:
        return template.format(**ctx)
    except (KeyError, ValueError):
        return template
