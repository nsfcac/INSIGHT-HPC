from __future__ import annotations

from typing import Optional

import pandas as pd

from src.phase4.operator_alert_module.data import (
    PHASE_SCALE,
    RAW_SATURATION,
    JOINT_TIE_RATIO,
)
from src.phase4.operator_alert_module.rules import (
    constraint_evidence,
    jobs_clause,
    rule_evidence,
)


# Normalise each phase score by its scale into 0-1 evidence.
def normalized_phase_scores(ep: pd.Series) -> dict[str, float]:
    out = {}
    for k, scale in PHASE_SCALE.items():
        v = float(ep.get(k, 0.0) or 0.0)
        out[k] = max(0.0, v) / scale if scale > 0 else 0.0
    return out


# Return whether any phase-1 detector or rule fired.
def phase1_has_sub_signal(ep: pd.Series) -> bool:
    detector_flags = (
        "if_is_anomaly",
        "if_is_anomaly_rel",
        "lstm_is_anomaly",
        "thr_flag",
        "zscore_flag",
        "ewma_flag",
    )
    if any(bool(ep.get(c, False)) for c in detector_flags):
        return True
    for c in ep.index:
        if isinstance(c, str) and c.startswith("rule_") and bool(ep.get(c, False)):
            return True
    return False


# Determine which phases caught the anomaly (leader plus co-leaders).
def caught_by_phases(
    ep: pd.Series, tie_ratio: float = JOINT_TIE_RATIO
) -> tuple[list[str], dict]:
    norm = normalized_phase_scores(ep)
    leader = max(norm.values()) if norm else 0.0

    raw = {k: float(ep.get(k, 0.0) or 0.0) for k in PHASE_SCALE.keys()}
    saturated: list[str] = []
    for k, v in raw.items():
        if v < RAW_SATURATION:
            continue
        if k == "phase1_score" and not phase1_has_sub_signal(ep):
            continue
        saturated.append(k)

    physics_z = abs(float(ep.get("phase3_physics_z", 0.0) or 0.0))
    n_const = int(ep.get("phase3_n_constraints", 0) or 0)
    p3_strong = (physics_z >= 4.0) or (n_const >= 2)

    if leader < 0.20 and not p3_strong:
        return ([max(norm, key=norm.get)] if norm else []), norm

    cutoff_co_leader = leader * (1.0 - tie_ratio)
    floor_active = 0.25  # below this is not a "catch"
    near_leader = [
        k for k, v in norm.items() if v >= max(cutoff_co_leader, floor_active)
    ]

    chosen = list(dict.fromkeys(saturated + near_leader))
    if p3_strong and "phase3_score" not in chosen:
        chosen.append("phase3_score")
    if not chosen:
        if any(v >= RAW_SATURATION for v in raw.values()):
            chosen = [max(raw, key=raw.get)]
        elif norm:
            chosen = [max(norm, key=norm.get)]

    def sort_key(k):
        is_sat = k in saturated
        is_p3_strong = k == "phase3_score" and p3_strong and not is_sat
        rank = -2 if is_sat else (-1 if is_p3_strong else 0)
        return (rank, -raw[k], -norm[k])

    chosen.sort(key=sort_key)
    return chosen, norm


# List the phase-1 detectors and rules that attributed to the episode.
def phase1_detector_attribution(ep: pd.Series) -> list[str]:
    rules = rule_evidence(ep)
    extras = []
    for col, label in (
        ("if_is_anomaly", "isolation-forest anomaly"),
        ("lstm_is_anomaly", "LSTM autoencoder reconstruction anomaly"),
        ("zscore_flag", "rolling z-score spike"),
        ("ewma_flag", "EWMA drift"),
    ):
        if bool(ep.get(col, False)):
            extras.append(label)
    return list(dict.fromkeys(rules + extras))


# Describe the phase-2 peer/cluster/coherence signals for the episode.
def phase2_detector_attribution(ep: pd.Series) -> list[str]:
    out = []
    peer_z = float(ep.get("phase2_peer_divergence_z", 0.0) or 0.0)
    cluster = float(ep.get("phase2_cluster_dist", 0.0) or 0.0)
    coh_z = float(ep.get("coherence_z", 0.0) or 0.0)
    peer_pm = float(ep.get("peer_z_pm", 0.0) or 0.0)
    p2_score = float(ep.get("phase2_score", 0.0) or 0.0)

    if peer_z >= 2.0:
        out.append(
            f"peer-divergence z={peer_z:.1f}σ — this node's profile is "
            f"{peer_z:.1f} stddevs from its same-job peers' median"
        )
    elif peer_z > 0:
        out.append(
            f"peer-divergence z={peer_z:.1f}σ (sub-threshold; "
            "node mostly tracks its job peers)"
        )
    if cluster >= 2.0:
        out.append(
            f"workload-cluster outlier — distance {cluster:.2f} from "
            "the assigned profile centroid (≥2.0 = outlier)"
        )
    elif cluster > 0:
        out.append(
            f"workload-cluster distance {cluster:.2f} (sub-threshold; "
            "node profile near its assigned centroid)"
        )
    if coh_z >= 2.0:
        out.append(
            f"multi-node coherence z={coh_z:.1f}σ — within-job feature "
            "divergence vs same-job peers"
        )
    if peer_pm >= 2.0 and peer_z < 2.0:
        out.append(
            f"per-minute peer divergence z={peer_pm:.1f}σ "
            "(short-window spike vs job peers)"
        )
    if not out and p2_score >= 1.0:
        out.append(
            f"aggregate phase-2 score={p2_score:.1f} "
            "(driven by job-context features below per-signal thresholds)"
        )
    return out


# Describe the phase-3 physics residual and constraint signals.
def phase3_detector_attribution(ep: pd.Series) -> list[str]:
    out = []
    physics_z = float(ep.get("phase3_physics_z", 0.0) or 0.0)
    ctx_score = float(ep.get("phase3_context_score", 1.0) or 1.0)
    n_const = int(ep.get("phase3_n_constraints", 0) or 0)

    if abs(physics_z) >= 2.0:
        magnitude = (
            "watchlist"
            if abs(physics_z) < 4.0
            else "anomalous" if abs(physics_z) < 8.0 else "severely off-model"
        )
        if ctx_score < 0.5:
            out.append(
                f"physics residual z={physics_z:.1f}σ ({magnitude}; "
                f"actual power/thermal reading is {abs(physics_z):.1f} "
                f"stddevs from the physics-model-expected envelope; "
                f"phase-3 score dampened by job-transition context, "
                f"ctx_score={ctx_score:.2f})"
            )
        else:
            out.append(
                f"physics residual z={physics_z:.1f}σ ({magnitude}; "
                f"actual reading is {abs(physics_z):.1f} stddevs from "
                f"the physics-model-expected envelope)"
            )
    if n_const > 0:
        out.append(
            f"{n_const} physics constraint(s) violated "
            "(see Constraints figure for which fired and when)"
        )
    out.extend(constraint_evidence(ep))
    if not out:
        p3_score = float(ep.get("phase3_score", 0.0) or 0.0)
        if p3_score >= 1.0:
            out.append(
                f"aggregate phase-3 score={p3_score:.1f} "
                "(constraint or context-driven, sub-threshold residual)"
            )
    return list(dict.fromkeys(out))


# Format the "Caught by" line from primary and supporting phases.
def format_caught_by(
    caught_by: list[str], norm: dict[str, float], ep: pd.Series
) -> str:
    if not caught_by:
        return "**Caught by:** (no phase clearly dominant — review supporting evidence)"

    raw = {k: float(ep.get(k, 0.0) or 0.0) for k in PHASE_SCALE}
    saturated = {k for k in caught_by if raw.get(k, 0.0) >= RAW_SATURATION}
    leader_norm = max(norm.values()) if norm else 0.0
    cutoff = max(leader_norm * (1.0 - JOINT_TIE_RATIO), 0.25)
    co_leader = {k for k in caught_by if norm.get(k, 0.0) >= cutoff}
    primary_set = saturated | co_leader
    primary = [k for k in caught_by if k in primary_set]
    if not primary:
        # Degenerate fallback — treat the first listed phase as primary.
        primary = [caught_by[0]]
    supporting = [k for k in caught_by if k not in primary]

    def short_label(k: str) -> str:
        return k.replace("phase", "Phase ").replace("_score", "")

    if len(primary) == 1:
        primary_str = phase_label(primary[0])
        if not supporting:
            return f"**Caught by:** {primary_str}"
        sup = " and ".join(short_label(s) for s in supporting)
        return f"**Caught by:** {primary_str} (supported by {sup})"
    joint = " + ".join(short_label(p) for p in primary)
    if not supporting:
        return f"**Caught by:** {joint} jointly"
    sup = " and ".join(short_label(s) for s in supporting)
    return f"**Caught by:** {joint} jointly (supported by {sup})"


# Return the full descriptive label for a phase key.
def phase_label(key: str) -> str:
    return {
        "phase1_score": "Phase 1 (sensor rules + statistical detectors)",
        "phase2_score": "Phase 2 (job-context / peer divergence)",
        "phase3_score": "Phase 3 (physics residuals + constraints)",
    }.get(key, key)


# Return the short label for a phase key.
def phase_label_short(key: str) -> str:
    return {
        "phase1_score": "Phase 1 — sensor rules + statistical detectors",
        "phase2_score": "Phase 2 — job-context / peer divergence",
        "phase3_score": "Phase 3 — physics residuals + constraints",
    }.get(key, key)


# Dispatch to the per-phase detector-attribution function.
def detectors_for_phase(phase_key: str, ep: pd.Series) -> list[str]:
    return {
        "phase1_score": phase1_detector_attribution,
        "phase2_score": phase2_detector_attribution,
        "phase3_score": phase3_detector_attribution,
    }[phase_key](ep)


def format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f} h"
    return f"{minutes / (60 * 24):.1f} days"


def sentence_case(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


# Return the plain-text supporting evidence for a phase.
def phase_supporting_text(
    phase_key: str, ep: pd.Series, master_ctx: Optional[dict] = None
) -> str:
    if phase_key == "phase1_score":
        return phase1_plain_text(ep, master_ctx)
    if phase_key == "phase2_score":
        return phase2_plain_text(ep)
    if phase_key == "phase3_score":
        return phase3_plain_text(ep, master_ctx)
    return ""


# Build phase-1 plain-text evidence (rules, detectors, channel hints).
def phase1_plain_text(ep: pd.Series, master_ctx: Optional[dict] = None) -> str:
    fired = phase1_detector_attribution(ep)
    if not fired:
        p1_score = float(ep.get("phase1_score", 0.0) or 0.0)
        if p1_score < 4.0:
            return ""
        contribs: list[str] = []
        if_z = float(ep.get("if_anomaly_score", 0.0) or 0.0)
        if_z_r = float(ep.get("if_anomaly_score_rel", 0.0) or 0.0)
        lstm_z = float(ep.get("lstm_recon_z", 0.0) or 0.0)
        if abs(if_z) > 0:
            contribs.append(f"isolation-forest score={if_z:.2f}")
        if abs(if_z_r) > 0 and abs(if_z_r - if_z) > 1e-3:
            contribs.append(f"IF rel-score={if_z_r:.2f}")
        if abs(lstm_z) > 0:
            contribs.append(f"LSTM-AE recon z={lstm_z:.2f}")
        contrib_str = (" — driven by " + "; ".join(contribs)) if contribs else ""
        return (
            f"multi-detector ensemble at saturation "
            f"(phase1_score={p1_score:.2f}/5.00; mean of robust-z "
            f"normalized detector outputs hit the cap){contrib_str}. "
            f"No single detector tripped its hard threshold — the signal "
            f"is sustained off-distribution behavior, not an isolated spike"
        )
    base = (
        f"sensor rule triggered — {fired[0]}"
        if len(fired) == 1
        else "sensor rules triggered — " + "; ".join(fired)
    )
    chan_hint = phase1_channel_hint(ep, master_ctx) if master_ctx else ""
    if chan_hint:
        base = f"{base}. Affected channel(s): {chan_hint}"
    return base


# Identify the sensor channels implicated by the earliest-firing rules.
def culprit_channels(ep: pd.Series, master_ctx: Optional[dict]) -> list[str]:
    if not master_ctx:
        return []

    cpu = master_ctx.get("peak_cpu_temp")
    inlet = master_ctx.get("peak_inlet_temp")
    fan = master_ctx.get("min_fan_rpm")
    pwr = master_ctx.get("peak_power")
    rule_to_chans: list[tuple[str, list[str]]] = []
    if bool(ep.get("rule_FLATLINE", False)):
        rule_to_chans.append(
            ("rule_FLATLINE", master_ctx.get("flatline_channels") or [])
        )
    if bool(ep.get("rule_DROPOUT", False)):
        rule_to_chans.append(("rule_DROPOUT", master_ctx.get("dropout_channels") or []))
    if bool(ep.get("rule_HIGH_CPU_TEMP", False)) and cpu and cpu.get("source"):
        rule_to_chans.append(("rule_HIGH_CPU_TEMP", [cpu["source"]]))
    if bool(ep.get("rule_HIGH_INLET_TEMP", False)) and inlet and inlet.get("source"):
        rule_to_chans.append(("rule_HIGH_INLET_TEMP", [inlet["source"]]))
    if (
        (
            bool(ep.get("rule_FAN_RPM_PCT_DROP", False))
            or bool(ep.get("rule_FAN_FAIL", False))
        )
        and fan
        and fan.get("source")
    ):
        ts_drop = ep.get("first_fire_rule_FAN_RPM_PCT_DROP")
        ts_fail = ep.get("first_fire_rule_FAN_FAIL")
        ts_pair = min((t for t in (ts_drop, ts_fail) if pd.notna(t)), default=None)
        rule_to_chans.append(
            (
                "rule_FAN_FIRST" if ts_pair is not None else "rule_FAN_FAIL",
                [fan["source"]],
            )
        )
    if (
        (
            bool(ep.get("rule_HIGH_SYSTEM_POWER", False))
            or bool(ep.get("rule_IDLE_POWER_SURGE", False))
        )
        and pwr
        and pwr.get("source")
    ):
        rule_to_chans.append(("rule_POWER_FIRST", [pwr["source"]]))

    if not rule_to_chans:
        return []

    # Return the earliest first-fire time for a rule (or rule group).
    def first_fire(name: str) -> Optional[pd.Timestamp]:
        if name == "rule_FAN_FIRST":
            members = ("first_fire_rule_FAN_RPM_PCT_DROP", "first_fire_rule_FAN_FAIL")
        elif name == "rule_POWER_FIRST":
            members = (
                "first_fire_rule_IDLE_POWER_SURGE",
                "first_fire_rule_HIGH_SYSTEM_POWER",
            )
        else:
            members = (f"first_fire_{name}",)
        ts_vals = [ep.get(m) for m in members]
        ts_vals = [t for t in ts_vals if t is not None and pd.notna(t)]
        return min(ts_vals) if ts_vals else None

    fires = [(first_fire(r), r, ch) for r, ch in rule_to_chans]
    timed = [(ts, r, ch) for ts, r, ch in fires if ts is not None and ch]
    if not timed:
        # No first-fire data → legacy behavior, in declaration order.
        out = [c for _, ch in rule_to_chans for c in ch]
        return list(dict.fromkeys(out))

    timed.sort(key=lambda x: x[0])
    earliest = timed[0][0]
    ONSET_GRACE_MIN = 5
    cutoff = earliest + pd.Timedelta(minutes=ONSET_GRACE_MIN)
    out: list[str] = []
    for ts, recall_value, ch in timed:
        if ts <= cutoff:
            out.extend(ch)
    return list(dict.fromkeys(out))


# Build a per-channel hint string for the phase-1 rules.
def phase1_channel_hint(ep: pd.Series, master_ctx: dict) -> str:
    hints: list[str] = []
    if bool(ep.get("rule_FLATLINE", False)):
        ch = master_ctx.get("flatline_channels") or []
        if ch:
            hints.append(
                f"variance collapsed on `{ch[0]}`"
                + (f" (+{len(ch)-1} more)" if len(ch) > 1 else "")
            )
    if bool(ep.get("rule_DROPOUT", False)):
        ch = master_ctx.get("dropout_channels") or []
        if ch:
            hints.append(
                f"silent on `{ch[0]}`"
                + (f" (+{len(ch)-1} more)" if len(ch) > 1 else "")
            )
    cpu = master_ctx.get("peak_cpu_temp")
    if bool(ep.get("rule_HIGH_CPU_TEMP", False)) and cpu and cpu.get("source"):
        hints.append(f"CPU temp peaked on `{cpu['source']}` at " f"{cpu['peak']:.1f}°C")
    inlet = master_ctx.get("peak_inlet_temp")
    if bool(ep.get("rule_HIGH_INLET_TEMP", False)) and inlet and inlet.get("source"):
        hints.append(
            f"inlet temp peaked on `{inlet['source']}` at " f"{inlet['peak']:.1f}°C"
        )
    fan = master_ctx.get("min_fan_rpm")
    if (
        (
            bool(ep.get("rule_FAN_RPM_PCT_DROP", False))
            or bool(ep.get("rule_FAN_FAIL", False))
        )
        and fan
        and fan.get("source")
    ):
        hints.append(f"fan RPM dropped to {fan['peak']:.0f} on " f"`{fan['source']}`")
    pwr = master_ctx.get("peak_power")
    if bool(ep.get("rule_HIGH_SYSTEM_POWER", False)) and pwr and pwr.get("source"):
        hints.append(
            f"system power peaked at {pwr['peak']:.0f} W on " f"`{pwr['source']}`"
        )
    return "; ".join(hints)


# Build phase-2 plain-text evidence (peer/cluster/coherence).
def phase2_plain_text(ep: pd.Series) -> str:
    parts: list[str] = []
    peer_z = float(ep.get("phase2_peer_divergence_z", 0.0) or 0.0)
    cluster = float(ep.get("phase2_cluster_dist", 0.0) or 0.0)
    coh_z = float(ep.get("coherence_z", 0.0) or 0.0)
    peer_pm = float(ep.get("peer_z_pm", 0.0) or 0.0)
    if peer_z >= 2.0:
        parts.append(
            f"this node's profile is {peer_z:.1f} stddevs from its same-job "
            "peers' median — peer-divergence outlier"
        )
    if cluster >= 2.0:
        parts.append(
            f"workload-cluster outlier — distance {cluster:.2f} from the "
            "assigned profile centroid (≥2.0 = outlier)"
        )
    if coh_z >= 2.0:
        parts.append(
            f"multi-node coherence anomaly at z={coh_z:.1f}σ — within-job "
            "feature divergence vs same-job peers"
        )
    if peer_pm >= 2.0 and peer_z < 2.0:
        parts.append(
            f"per-minute peer divergence at z={peer_pm:.1f}σ "
            "(short-window spike vs job peers)"
        )
    if not parts:
        if peer_z > 0:
            parts.append(
                f"peer-divergence z={peer_z:.1f}σ (sub-threshold; node "
                "mostly tracks its job peers)"
            )
        if cluster > 0:
            parts.append(
                f"workload-cluster distance {cluster:.2f} (sub-threshold; "
                "node profile near its assigned centroid)"
            )
    return "; ".join(parts)


# Build phase-3 plain-text evidence (residual plus constraints).
def phase3_plain_text(ep: pd.Series, master_ctx: Optional[dict] = None) -> str:
    parts: list[str] = []
    physics_z = float(ep.get("phase3_physics_z", 0.0) or 0.0)
    ctx_score = float(ep.get("phase3_context_score", 1.0) or 1.0)
    n_const = int(ep.get("phase3_n_constraints", 0) or 0)
    if abs(physics_z) >= 2.0:
        magnitude = (
            "watchlist"
            if abs(physics_z) < 4.0
            else "anomalous" if abs(physics_z) < 8.0 else "severely off-model"
        )
        damp = (
            (
                f" (phase-3 score dampened by job-transition context, "
                f"ctx_score={ctx_score:.2f})"
            )
            if ctx_score < 0.5
            else ""
        )
        residual_line = (
            f"power/thermal reading is {abs(physics_z):.1f} stddevs from "
            f"the physics-model-expected envelope — {magnitude}{damp}"
        )
        ratio_clause = phase3_peak_clause(master_ctx) if master_ctx else ""
        if ratio_clause:
            residual_line = f"{residual_line} ({ratio_clause})"
        parts.append(residual_line)
    const_labels = constraint_evidence(ep)
    if const_labels:
        n = max(n_const, len(const_labels))
        plural = "constraint" if n == 1 else "constraints"
        parts.append(f"{n} physics {plural} fired ({', '.join(const_labels)})")
    elif n_const > 0:
        plural = "constraint" if n_const == 1 else "constraints"
        parts.append(f"{n_const} physics {plural} fired")
    return "; ".join(parts)


# Phrase the most-deviating sensor's peak versus its baseline.
def phase3_peak_clause(master_ctx: dict) -> str:
    best = None
    best_dev = 0.0
    for c in master_ctx.get("candidates", []):
        baseline = c.get("baseline")
        peak = c.get("peak")
        if baseline is None or peak is None or abs(baseline) < 1e-9:
            continue
        diff = peak - baseline
        pct = abs(diff) / abs(baseline)
        if pct > best_dev:
            best_dev = pct
            best = c
    if best is None:
        return ""
    label = best["label"].lower()
    label = (
        label.replace(" peak", "")
        .replace("min ", "")
        .replace(" (any channel)", "")
        .strip()
    )
    peak = best["peak"]
    baseline = best["baseline"]
    unit = best["unit"]
    decimals = best["decimals"]
    if abs(baseline) <= 1e-9:
        return ""
    unit_sp = f" {unit}" if unit else ""
    if peak > baseline:
        # Peak above baseline — a multiplier reads naturally ("2.6× higher").
        ratio = peak / baseline
        magnitude = f"{ratio:.1f}× higher"
    else:
        pct_drop = 100.0 * (baseline - peak) / abs(baseline)
        magnitude = f"{pct_drop:.0f}% drop from baseline"
    return (
        f"peak {label} was {peak:.{decimals}f}{unit_sp}, vs pre-episode "
        f"baseline {baseline:.{decimals}f}{unit_sp} — {magnitude}"
    )
