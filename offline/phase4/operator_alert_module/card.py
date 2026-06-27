from __future__ import annotations

from typing import Optional

import pandas as pd

from offline.phase4.operator_alert_module.data import (
    CATEGORY_TEMPLATES,
    LIKELY_CAUSE_BY_CATEGORY,
    ROMAN_PHASE,
    TIER_TIME_TO_ACT_MIN,
)
from offline.phase4.operator_alert_module.master import (
    build_master_ctx,
    format_sensor_line,
)
from offline.phase4.operator_alert_module.phase import (
    caught_by_phases,
    format_caught_by,
    phase_label_short,
    phase_supporting_text,
    sentence_case,
)
from offline.phase4.operator_alert_module.rules import (
    category_from_rules_and_context,
    compose_message,
    constraint_evidence,
    cooldown_clause,
    infer_rack,
    jobs_clause,
    rule_evidence,
    severity_clause,
    synthesize_unknown_message,
    top_shap_clause,
)
from offline.phase4.operator_alert_module.summary import (
    format_duration,
    format_lead_time,
    format_section,
    format_template_safe,
    plain_language_summary,
    strength_summary,
    what_happened_context,
)


# Format a value as a UTC minute string.
def format_utc_minute(value) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return "Unknown"
    return ts.strftime("%Y-%m-%d %H:%M UTC")


# Describe the alert's target node (type, host, rack).
def target_text(ep: pd.Series, rack: str) -> str:
    host = str(ep.get("hostname", "")).strip() or "unknown host"
    comp = str(ep.get("component", "")).strip()
    comp_label = {
        "zen4": "CPU node",
        "h100": "GPU node",
    }.get(comp.lower(), f"{comp} node" if comp else "node")
    rack_text = f", rack {rack}" if rack else ""
    return f"{comp_label} {host}{rack_text}"


# List the SLURM jobs active during the alert window.
def associated_job_text(ep: pd.Series) -> str:
    ids = str(ep.get("active_job_ids", "") or "").strip()
    job_list = [j for j in ids.split("|") if j]
    if not job_list:
        return "None"
    if len(job_list) == 1:
        return job_list[0]
    if len(job_list) <= 4:
        return ", ".join(job_list)
    return ", ".join(job_list[:4]) + f" (+{len(job_list) - 4} more)"


# Phrase which phases detected the anomaly, primary first.
def detection_phase_text(caught_by: list[str]) -> str:
    phases = [ROMAN_PHASE.get(p, p) for p in caught_by]
    if not phases:
        return "Unattributed"
    if len(phases) == 1:
        return phases[0]
    if len(phases) == 2:
        return f"{phases[0]}, supported by {phases[1]}"
    return f"{phases[0]}, supported by " + ", ".join(phases[1:])


# Phrase the recommended response window for the alert tier.
def response_window_sentence(ep: pd.Series) -> str:
    tier = str(ep.get("fusion_tier", "")).strip().upper()
    minutes = TIER_TIME_TO_ACT_MIN.get(tier)
    if minutes is None:
        return "Review manually."
    verb = "Investigate" if tier in ("CRITICAL", "CONFIRMED") else "Review"
    if minutes < 60:
        return f"{verb} within {minutes} min."
    if minutes == 60:
        return f"{verb} within 1 hour."
    hours = minutes / 60.0
    if float(hours).is_integer():
        return f"{verb} within {int(hours)} hours."
    return f"{verb} within {hours:.1f} hours."


# Build the operator-action text (response window plus category guidance).
def operator_action_text(
    ep: pd.Series,
    category: str,
    rack: str,
    top_shap_features: Optional[list[dict]] = None,
) -> str:
    template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES["unknown"])
    if category == "unknown":
        base = synthesize_unknown_message(ep, top_shap_features or [])
    else:
        base = format_template_safe(template.get("message", ""), ep, rack)
    return " ".join(p for p in (response_window_sentence(ep), base) if p).strip()


# Return a sensor candidate's relative deviation from baseline.
def candidate_deviation(c: dict) -> float:
    try:
        baseline = float(c.get("baseline", 0.0))
        peak = float(c.get("peak", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not pd.notna(baseline) or not pd.notna(peak):
        return 0.0
    return abs(peak - baseline) / max(abs(baseline), 1e-6)


# Return the sensor candidate that deviates most from its baseline.
def best_sensor_candidate(master_ctx: Optional[dict]) -> Optional[dict]:
    candidates = (master_ctx or {}).get("candidates", []) or []
    candidates = [c for c in candidates if c.get("peak") is not None]
    if not candidates:
        return None
    return max(candidates, key=candidate_deviation)


# Phrase the episode's peak metric value versus its baseline.
def metric_peak_evidence(best: dict) -> Optional[str]:
    peak = best.get("peak")
    if peak is None:
        return None
    baseline = best.get("baseline")
    decimals = int(best.get("decimals", 1))
    unit = str(best.get("unit", ""))
    unit_sp = f" {unit}" if unit else ""
    label = str(best.get("label", "metric")).lower()
    label = label.replace(" peak", "").replace("min ", "minimum ")
    if baseline is None or not pd.notna(baseline):
        return f"The episode reached {label} {peak:.{decimals}f}{unit_sp}."
    return (
        f"The episode reached {label} {peak:.{decimals}f}{unit_sp}, "
        f"compared with a pre-episode baseline of "
        f"{float(baseline):.{decimals}f}{unit_sp}."
    )


# Phrase the physics-residual sigma and peak-vs-baseline ratio.
def sigma_ratio_evidence(ep: pd.Series, best: Optional[dict]) -> Optional[str]:
    physics_z = float(ep.get("phase3_physics_z", 0.0) or 0.0)
    if abs(physics_z) < 2.0:
        return None
    ratio = ""
    if best is not None:
        try:
            peak = float(best.get("peak"))
            baseline = float(best.get("baseline"))
            if pd.notna(peak) and pd.notna(baseline) and abs(baseline) > 1e-9:
                if peak >= baseline:
                    ratio = f" and a {peak / baseline:.1f}x increase over baseline"
                else:
                    drop = 100.0 * (baseline - peak) / abs(baseline)
                    ratio = f" and a {drop:.0f}% drop from baseline"
        except (TypeError, ValueError):
            pass
    return f"The deviation corresponds to {physics_z:.1f} sigma{ratio}."


# Assemble the supporting-evidence bullet list for the alert card.
def supporting_evidence_card(
    ep: pd.Series,
    caught_by: list[str],
    master_ctx: Optional[dict] = None,
    top_shap_features: Optional[list[dict]] = None,
) -> list[str]:
    evidence: list[str] = []
    for phase in caught_by:
        text = phase_supporting_text(phase, ep, master_ctx)
        if text:
            evidence.append(f"{ROMAN_PHASE.get(phase, phase)}: {sentence_case(text)}.")
    best = best_sensor_candidate(master_ctx)
    peak_line = metric_peak_evidence(best) if best is not None else None
    if peak_line:
        evidence.append(peak_line)
    sigma_line = sigma_ratio_evidence(ep, best)
    if sigma_line:
        evidence.append(sigma_line)
    shap_line = top_shap_clause(top_shap_features or [])
    if shap_line:
        evidence.append(shap_line)
    if not evidence:
        rules = rule_evidence(ep)
        constraints = constraint_evidence(ep)
        if rules:
            evidence.append("Phase I rules fired: " + "; ".join(rules) + ".")
        if constraints:
            evidence.append(
                "Physics constraints fired: " + "; ".join(constraints) + "."
            )
    if not evidence:
        evidence.append(
            "Review the per-phase score and sensor plots for the contributing signal."
        )
    return list(dict.fromkeys(evidence))[:6]


# Build the structured explainable-alert card (target, cause, action, evidence).
def build_explainable_alert_card(
    ep: pd.Series,
    category: str,
    rack: str,
    caught_by: Optional[list[str]] = None,
    norm: Optional[dict[str, float]] = None,
    master_ctx: Optional[dict] = None,
    top_shap_features: Optional[list[dict]] = None,
) -> dict:
    if caught_by is None or norm is None:
        caught_by, norm = caught_by_phases(ep)
    return {
        "target": target_text(ep, rack),
        "severity": str(ep.get("fusion_tier", "NONE")),
        "detection_phase": detection_phase_text(caught_by),
        "detected_at": format_utc_minute(ep.get("episode_start")),
        "associated_job": associated_job_text(ep),
        "likely_cause": LIKELY_CAUSE_BY_CATEGORY.get(
            category, LIKELY_CAUSE_BY_CATEGORY["unknown"]
        ),
        "operator_action": operator_action_text(
            ep, category, rack, top_shap_features=top_shap_features
        ),
        "supporting_evidence": supporting_evidence_card(
            ep,
            caught_by,
            master_ctx=master_ctx,
            top_shap_features=top_shap_features,
        ),
    }


# Render the explainable-alert card as a markdown section.
def format_explainable_alert_section(card: dict) -> str:
    evidence = card.get("supporting_evidence") or []
    evidence_lines = "\n".join(f"- {line}" for line in evidence)
    return "\n".join(
        [
            f"**Target:** {card.get('target', '')}",
            f"**Severity:** {card.get('severity', '')}",
            f"**Detection phase:** {card.get('detection_phase', '')}",
            f"**Detected at:** {card.get('detected_at', '')}",
            f"**Associated job:** {card.get('associated_job', '')}",
            f"**Likely cause:** {card.get('likely_cause', '')}",
            "",
            f"**Operator action.** {card.get('operator_action', '')}",
            "",
            "**Supporting evidence.**",
            evidence_lines,
        ]
    )


# Build the full markdown README for one alert episode.
def build_alert_readme(
    ep: pd.Series,
    category: Optional[str] = None,
    figures: Optional[list[tuple[str, str]]] = None,
    rack: Optional[str] = None,
    host_master_window: Optional[pd.DataFrame] = None,
    top_shap_features: Optional[list[dict]] = None,
) -> str:
    if category is None:
        category = category_from_rules_and_context(ep)
    template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES["unknown"])
    rack = rack if rack is not None else infer_rack(str(ep.get("hostname", "")))

    caught_by, norm = caught_by_phases(ep)
    title = format_template_safe(template.get("title", ""), ep, rack)

    caught_line = format_caught_by(caught_by, norm, ep)

    master_ctx = build_master_ctx(ep, host_master_window)
    alert_card = build_explainable_alert_card(
        ep,
        category=category,
        rack=rack,
        caught_by=caught_by,
        norm=norm,
        master_ctx=master_ctx,
        top_shap_features=top_shap_features or [],
    )

    evidence_rows: list[str] = []
    for k in ("phase1_score", "phase2_score", "phase3_score"):
        is_lead = k in caught_by
        marker = (
            "  ← primary signal"
            if is_lead and len(caught_by) == 1
            else ("  ← co-primary" if is_lead else "")
        )
        score = float(ep.get(k, 0.0) or 0.0)
        n = norm.get(k, 0.0)
        plain = phase_supporting_text(k, ep, master_ctx)
        # Skip silent phases that didn't make the headline.
        if not is_lead and score < 0.05 and not plain:
            continue
        if not plain:
            plain = "no individual sub-signal above threshold"
        plain = sentence_case(plain) + "."
        score_brackets = f"_(score {score:.2f}, {n:.2f}× typical-active)_"
        evidence_rows.append(
            f"- **{phase_label_short(k)}**: {plain} {score_brackets}{marker}"
        )
    if not evidence_rows:
        evidence_rows.append(
            "- _(no individual phase produced sub-signal evidence — "
            "review the per-phase score figure and SHAP contributors)_"
        )
    evidence_body = "\n".join(evidence_rows)

    sensor_lines: list[str] = []
    candidates = (master_ctx or {}).get("candidates", []) or []
    if candidates:
        deduped: dict[tuple, dict] = {}
        for c in candidates:
            key = (c["label"], round(c["peak"], c["decimals"]))
            if key in deduped:
                if c["source"] not in deduped[key]["sources"]:
                    deduped[key]["sources"].append(c["source"])
            else:
                row = {kk: vv for kk, vv in c.items() if kk != "source"}
                row["sources"] = [c["source"]] if c["source"] else []
                deduped[key] = row

        def percent_deviation(c: dict) -> float:
            try:
                base = float(c.get("baseline", 0.0))
                peak = float(c.get("peak", 0.0))
            except (TypeError, ValueError):
                return 0.0
            if not pd.notna(base) or not pd.notna(peak):
                return 0.0
            denom = max(abs(base), 1e-6)
            return abs(peak - base) / denom

        scored = [(c, percent_deviation(c)) for c in deduped.values()]
        causal = [c for c, d in scored if d >= 0.05]
        if not causal and scored:
            causal = [max(scored, key=lambda x: x[1])[0]]
        for c in causal:
            sensor_lines.append(
                format_sensor_line(
                    c["label"],
                    c["peak"],
                    c["baseline"],
                    c["unit"],
                    c["decimals"],
                    c["sources"],
                )
            )
    if not sensor_lines:
        sensor_lines = ["- _(sensor-level detail unavailable for this alert window)_"]

    ep_start = ep.get("episode_start")
    ep_end = ep.get("episode_end")
    ep_start_ts = pd.to_datetime(ep_start, utc=True, errors="coerce")
    ep_end_ts = pd.to_datetime(ep_end, utc=True, errors="coerce")
    if pd.notna(ep_start_ts) and pd.notna(ep_end_ts):
        dur = (ep_end_ts - ep_start_ts).total_seconds() / 60.0
    else:
        dur = float(ep.get("ep_duration_min", 0) or 0)
    prov = [
        f"- Displayed window: `{ep_start}` → `{ep_end}` ({dur:.0f} min)",
        f"- Host: `{ep.get('hostname','')}`  Rack: `{rack}`  Component: `{ep.get('component','')}`",
        f"- Split: `{ep.get('split','')}`  Tier: `{ep.get('fusion_tier','')}`  fusion_prob: **{float(ep.get('fusion_prob',0)):.3f}**",
        f"- Phases active (norm ≥ 0.25 of typical-active): {sum(1 for v in norm.values() if v >= 0.25)}/3",
    ]
    persisted = float(ep.get("ep_duration_min", 0) or 0)
    if persisted > dur * 1.5 and persisted > 60.0:
        prov.append(
            f"- Underlying v2 episode persistence: **{format_duration(persisted)}** "
            f"(node was continuously flagged in phase-4 for the full span; "
            f"the displayed window above is the most-anomalous slice)"
        )
    lt_line = format_lead_time(ep)
    if lt_line:
        prov.append(f"- {lt_line}")
    jobs_line = jobs_clause(ep)
    if jobs_line:
        prov.append(f"- {jobs_line}")
    if "event_id" in ep.index and pd.notna(ep.get("event_id", None)):
        prov.append(
            f"- Matched ground-truth event: `{ep['event_id']}` "
            f"(category=`{ep.get('gt_category','')}`)"
        )

    action_lines: list[str] = []
    sev = severity_clause(ep)
    if sev:
        action_lines.append(f"- {sev}")
    cd = cooldown_clause(ep)
    if cd:
        action_lines.append(f"- {cd}")
    if not action_lines:
        action_lines = ["- _(severity window unset — review tier and decide manually)_"]

    # Figures section.
    fig_lines: list[str] = []
    for fname, desc in figures or []:
        fig_lines.append(f"- `{fname}` — {desc}")
    if not fig_lines:
        fig_lines = ["- _(no figures emitted — see plot generator log)_"]

    paged_title = format_template_safe(template.get("title", ""), ep, rack)
    if category == "unknown":
        paged_base = synthesize_unknown_message(ep, top_shap_features or [])
    else:
        paged_base = format_template_safe(template.get("message", ""), ep, rack)
    paged_message = compose_message(paged_base, ep, top_shap_features or [])
    paged_block = (
        f"> **{paged_title}**\n>\n"
        f"> {paged_message}\n>\n"
        f"> _Tier:_ **{ep.get('fusion_tier', '')}** "
        f"· _Confidence:_ {float(ep.get('fusion_prob', 0)):.3f} "
        f"· _Host:_ `{ep.get('hostname', '')}` "
        f"(rack {rack}, {ep.get('component', '')})"
    )

    plain = plain_language_summary(category, ep, caught_by)
    shap_clause = top_shap_clause(top_shap_features or [])
    if shap_clause:
        plain = f"{plain} {shap_clause}"
    context_lines = what_happened_context(ep)
    if context_lines:
        plain = plain + "\n\n" + "\n".join(context_lines)

    dq_lines: list[str] = []
    for label, key in (
        ("Temperature channels", "ep_pct_temp_nan"),
        ("Inlet/exhaust channels", "ep_pct_inlet_nan"),
        ("Fan _fan_avg channels", "ep_pct_fan_nan"),
        ("Power channels", "ep_pct_power_nan"),
    ):
        v = float(ep.get(key, 0.0) or 0.0)
        if v >= 0.10:
            dq_lines.append(f"- **{label}**: {v*100:.0f}% NaN during the alert window")
    n_dark = int(ep.get("ep_n_families_dark", 0) or 0)
    if n_dark >= 2:
        dq_lines.insert(
            0,
            f"- ⚠ **{n_dark} sensor families** simultaneously degraded — typical "
            "**BMC/iDRAC sensor-pipeline collapse** signature. Treat surface-rule "
            "attribution (fan / power / sensor-stuck) as suspect; verify "
            "management-plane access first.",
        )

    # Compose final markdown.
    md_parts = [
        f"# {title}",
        "",
        f"_{strength_summary(ep, norm, caught_by)}_",
        "",
        caught_line,
        "",
        format_section(
            "Explainable Alert", format_explainable_alert_section(alert_card)
        ),
        format_section("Action priority", "\n".join(action_lines)),
        format_section("Paged to operator", paged_block),
        format_section("What happened", plain),
        format_section("Supporting evidence", evidence_body),
    ]
    if dq_lines:
        md_parts.append(
            format_section("Data quality during alert window", "\n".join(dq_lines))
        )
    md_parts += [
        format_section("Sensor-level details", "\n".join(sensor_lines)),
        format_section("Figures", "\n".join(fig_lines)),
        format_section("Provenance", "\n".join(prov)),
    ]
    return "\n".join(md_parts)
