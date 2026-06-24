from __future__ import annotations

from typing import Optional

import pandas as pd

from src.phase4.operator_alert_module.data import (
    CATEGORY_TEMPLATES,
    FEATURE_HUMAN_NAMES,
    RULE_CATEGORY_TABLE,
    TIER_TIME_TO_ACT_MIN,
)


# Infer an alert category from fired rules, constraints, and job context.
def category_from_rules_and_context(episode_row: pd.Series) -> str:
    if (
        int(episode_row.get("ep_n_families_dark", 0) or 0) >= 2
        or float(episode_row.get("ep_pct_inlet_nan", 0.0) or 0.0) >= 0.50
    ):
        return "sensor_dropout"

    # Rule-based attribution by temporal precedence.
    fired: list[tuple] = []  # (first_fire_ts, table_index, rule, category)
    for idx, entry in enumerate(RULE_CATEGORY_TABLE):
        rule, cat, guard = entry
        if not bool(episode_row.get(rule, False)):
            continue
        if guard is not None and not guard(episode_row):
            continue
        ts = episode_row.get(f"first_fire_{rule}")
        if ts is None or pd.isna(ts):
            ts = pd.Timestamp.min.tz_localize("UTC")
        fired.append((ts, idx, rule, cat))
    if fired:
        fired.sort(key=lambda x: (x[0], x[1]))
        return fired[0][3]

    ctx = str(episode_row.get("anomaly_context", "")).upper()
    if ctx == "COOLING_FAULT":
        return "cooling_failure"
    if ctx == "MEASUREMENT_DISCREPANCY":
        return "measurement_glitch"
    if ctx == "INFRA_FAULT":
        return "hardware"
    # Phase 2 peer divergence as fallback signature.
    if float(episode_row.get("phase2_peer_divergence_z", 0.0)) > 2.0:
        return "peer_node_divergence"
    return "unknown"


# Return the top-k features by absolute SHAP contribution.
def top_shap_features(
    shap_row: pd.Series, feature_cols: list[str], k: int = 3
) -> list[tuple[str, float]]:
    contribs = []
    for c in feature_cols:
        shap_col = f"shap_{c}"
        if shap_col in shap_row.index:
            contribs.append((c, float(shap_row[shap_col])))
    if not contribs:
        return []
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:k]


# List human-readable descriptions of the threshold rules that fired.
def rule_evidence(episode_row: pd.Series) -> list[str]:
    human = {
        "rule_HIGH_SYSTEM_POWER": "system power exceeded threshold",
        "rule_HIGH_CPU_TEMP": "CPU temperature exceeded threshold",
        "rule_HIGH_INLET_TEMP": "rack inlet temperature exceeded threshold",
        "rule_HIGH_GPU_TEMP": "GPU temperature exceeded threshold",
        "rule_FAN_FAIL": "fan RPM below floor",
        "rule_FAN_RPM_PCT_DROP": "fan RPM dropped vs 7-day baseline",
        "rule_GPU_POWER_IDLE": "GPU drawing power while idle",
        "rule_HIGH_CPU_COMPONENT_POWER": "CPU component power runaway",
        "rule_HIGH_FAN_POWER": "fan power at max (thermal distress)",
        "rule_IDLE_POWER_SURGE": "idle node drew excess power",
        "rule_FLATLINE": "sensor variance collapsed (stuck)",
        "rule_DROPOUT": "sensor(s) went silent during active job",
        "rule_COLD_INLET_DROP": "inlet sensor read implausibly cold",
    }
    fired = []
    for k, label in human.items():
        if bool(episode_row.get(k, False)):
            fired.append(label)
    return fired


# List human-readable descriptions of the physics constraints violated.
def constraint_evidence(episode_row: pd.Series) -> list[str]:
    human = {
        "const1_temp_fan": "thermal/fan decoupling",
        "const2_rack_therm": "rack-wide thermal rise",
        "const3_dynamics": "power ramp without thermal response",
        "const4_crossplane": "PDU vs iDRAC power divergence",
        "const5_alloc_idle": "node allocated but idle",
    }
    fired = []
    for k, label in human.items():
        if bool(episode_row.get(k, False)):
            fired.append(label)
    return fired


# Phrase the count of affected sensor families.
def family_clause(n_fam: int) -> str:
    if n_fam <= 1:
        return "1 sensor family"
    return f"{n_fam} sensor families"


# Build a BMC/management-plane hint based on how many sensor families went dark.
def bmc_hint(n_fam: int, host: str) -> str:
    if n_fam <= 1:
        return (
            f"Sensor read path on this channel is likely stale; verify "
            f"management-plane access on {host} and cold-reset BMC if "
            "unresponsive."
        )
    return (
        "This signature (multi-family NaN onset) is typical of "
        "BMC/iDRAC instability rather than individual fan/power faults — "
        f"verify management-plane access on {host} before responding to "
        "any surface-rule recommendations; cold-reset BMC if unresponsive."
    )


# Phrase the alert tier and its time-to-act window.
def severity_clause(ep: pd.Series) -> str:
    tier = str(ep.get("fusion_tier", ""))
    win = TIER_TIME_TO_ACT_MIN.get(tier)
    if win is None:
        return ""
    if win < 60:
        return f"{tier} · Investigate within {win} min."
    return f"{tier} · Investigate within {win // 60} h."


# Phrase the jobs affected by an alert.
def jobs_clause(ep: pd.Series) -> str:
    ids = str(ep.get("active_job_ids", "") or "").strip()
    if not ids:
        return ""
    job_list = [j for j in ids.split("|") if j]
    if not job_list:
        return ""
    if len(job_list) == 1:
        return f"Affects job {job_list[0]}."
    if len(job_list) <= 3:
        return f"Affects jobs {', '.join(job_list)}."
    return f"Affects jobs {', '.join(job_list[:3])} (+{len(job_list) - 3} more)."


# Phrase the single top SHAP contributor.
def top_shap_clause(top_features: list[dict]) -> str:
    if not top_features:
        return ""
    f = top_features[0]
    val = float(f.get("shap", 0.0))
    sign = "+" if val >= 0 else "−"
    label = f.get("human") or f.get("feature") or "(feature)"
    return f"Top contributor: {label} ({sign}{abs(val):.2f} SHAP)."


# Phrase how long ago the last similar alert fired on this host.
def cooldown_clause(ep: pd.Series) -> str:
    v = ep.get("prev_similar_alert_min_ago")
    if v is None or (isinstance(v, float) and not pd.notna(v)):
        return ""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    if v <= 0:
        return ""
    if v < 60:
        return f"Last similar alert on this host: {v:.0f} min ago."
    if v < 60 * 24:
        return f"Last similar alert on this host: {v / 60:.1f} h ago."
    return f"Last similar alert on this host: {v / (60 * 24):.1f} d ago."


# Compose a message for uncategorised anomalies from the leading phase signals.
def synthesize_unknown_message(
    ep: pd.Series, top_features: Optional[list[dict]] = None
) -> str:
    norm = normalized_phase_scores(ep)
    if not norm:
        return "Anomaly detected; review supporting evidence below."

    caught, _ = caught_by_phases(ep)
    leader = caught[0] if caught else max(norm, key=norm.get)
    raw = float(ep.get(leader, 0.0) or 0.0)
    leader_label = {
        "phase1_score": "Sensor-rule",
        "phase2_score": "Job-context / peer-divergence",
        "phase3_score": "Physics-residual",
    }.get(leader, "Aggregate")

    parts = [f"{leader_label} signal leading (score {raw:.1f})."]

    co_keys = [k for k in caught if k != leader]
    if not co_keys:
        co_keys = [k for k in norm if k != leader and norm[k] >= 0.25]
    if co_keys:
        chunks = []
        for k in co_keys:
            v = float(ep.get(k, 0.0) or 0.0)
            short = {
                "phase1_score": "sensor",
                "phase2_score": "peer-divergence",
                "phase3_score": "physics",
            }[k]
            chunks.append(f"{short} (score {v:.1f})")
        parts.append(f"Co-firing: {', '.join(chunks)}.")

    shap_line = top_shap_clause(top_features or [])
    if shap_line:
        parts.append(shap_line)

    parts.append("No specific category match — inspect dashboards listed below.")
    return " ".join(parts)


# Assemble the full alert message from severity, base, jobs, and cooldown clauses.
def compose_message(
    base_message: str, ep: pd.Series, top_features: Optional[list[dict]] = None
) -> str:
    leading = " ".join(
        c for c in (severity_clause(ep), top_shap_clause(top_features or [])) if c
    )
    trailing = " ".join(c for c in (jobs_clause(ep), cooldown_clause(ep)) if c)
    pieces = [p for p in (leading, base_message, trailing) if p]
    return " ".join(pieces).strip()


# Compute minutes since the previous similar alert per host and category.
def compute_prev_similar(alerts: pd.DataFrame) -> pd.Series:
    if len(alerts) == 0 or "episode_start" not in alerts.columns:
        return pd.Series([pd.NA] * len(alerts), index=alerts.index, dtype="Float64")
    ts = pd.to_datetime(alerts["episode_start"], utc=True, errors="coerce")
    df = pd.DataFrame(
        {
            "hostname": alerts["hostname"].astype(str),
            "category": alerts["category"].astype(str),
            "ts": ts,
        },
        index=alerts.index,
    )
    df = df.sort_values(["hostname", "category", "ts"])
    delta = df.groupby(["hostname", "category"])["ts"].diff()
    minutes = delta.dt.total_seconds() / 60.0
    minutes = minutes.reindex(alerts.index)
    return minutes.astype("Float64")


# Extract the rack id from a hostname.
def infer_rack(host: str) -> str:
    if not host or "-" not in host:
        return ""
    parts = host.split("-")
    return parts[1] if len(parts) >= 2 else ""
