from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from src.phase4.operator_alert_module.data import *
from src.phase4.operator_alert_module.rules import *
from src.phase4.operator_alert_module.text import *
from src.phase4.operator_alert_module.io import write_alerts


# Turn scored episodes into operator-facing alert rows (title, message, action, evidence).
def build_alerts(
    episodes: pd.DataFrame,
    shap_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    out_cols = [
        "hostname",
        "component",
        "rack",
        "episode_start",
        "episode_end",
        "fusion_prob",
        "fusion_tier",
        "category",
        "title",
        "message",
        "target",
        "severity",
        "detection_phase",
        "detected_at",
        "associated_job",
        "likely_cause",
        "operator_action",
        "supporting_evidence",
        "top_features",
        "rules_fired",
        "constraints_violated",
        "metrics_of_interest",
        "affected_jobs",
        "severity_window_min",
        "prev_similar_alert_min_ago",
    ]
    if len(episodes) == 0:
        return pd.DataFrame(columns=out_cols)

    eps = episodes.copy()
    eps["_category"] = [category_from_rules_and_context(r) for _, r in eps.iterrows()]
    similarity_frame = pd.DataFrame(
        {
            "hostname": eps["hostname"].astype(str),
            "category": eps["_category"].astype(str),
            "episode_start": eps["episode_start"],
        },
        index=eps.index,
    )
    computed_prev_similar = compute_prev_similar(similarity_frame)
    if "prev_similar_alert_min_ago" in eps.columns:
        eps["prev_similar_alert_min_ago"] = (
            eps["prev_similar_alert_min_ago"]
            .astype("Float64")
            .fillna(computed_prev_similar)
        )
    else:
        eps["prev_similar_alert_min_ago"] = computed_prev_similar

    rows = []
    for idx, ep in eps.iterrows():
        category = ep.get("_category") or category_from_rules_and_context(ep)
        template = CATEGORY_TEMPLATES.get(category, CATEGORY_TEMPLATES["unknown"])

        rack = str(ep.get("rack", "")) or infer_rack(str(ep.get("hostname", "")))

        rules = rule_evidence(ep)
        constraints = constraint_evidence(ep)

        top_features_list: list[dict] = []
        if shap_df is not None and feature_cols is not None and len(shap_df):
            m = (
                (shap_df["hostname"] == ep["hostname"])
                & (shap_df["component"] == ep["component"])
                & (
                    pd.to_datetime(shap_df["episode_start"], utc=True)
                    == pd.to_datetime(ep["episode_start"], utc=True)
                )
            )
            hits = shap_df[m]
            if len(hits) >= 1:
                top = top_shap_features(hits.iloc[0], feature_cols, k=3)
                top_features_list = [
                    {
                        "feature": c,
                        "human": FEATURE_HUMAN_NAMES.get(c, c),
                        "shap": round(v, 4),
                        "direction": "↑ anomaly" if v > 0 else "↓ anomaly",
                    }
                    for c, v in top
                ]

        n_fam = int(ep.get("ep_n_families_dark", ep.get("n_dropout_families", 2)) or 2)
        ctx = {
            "host": str(ep.get("hostname", "")),
            "rack": rack,
            "metric": "primary sensor",
            "value": float(ep.get("metric_value", 0)),
            "threshold": float(ep.get("metric_threshold", 0)),
            "pct": 100 * float(ep.get("metric_pct_of_baseline", 0.5)),
            "delta_w": float(ep.get("idle_power_delta_w", 0)),
            "mismatch_w": float(ep.get("pdu_mismatch_w", 0)),
            "peer_nodes": str(ep.get("peer_nodes", "peer job members")),
            "n_families": n_fam,
            "family_clause": family_clause(n_fam),
            "bmc_hint": bmc_hint(n_fam, str(ep.get("hostname", ""))),
        }
        try:
            title = template["title"].format(**ctx)
            base_message = template["message"].format(**ctx)
        except (KeyError, ValueError):
            title = template["title"].format(host=ctx["host"], rack=rack)
            base_message = template["message"]
        if category == "unknown":
            base_message = synthesize_unknown_message(ep, top_features_list)
        message = compose_message(base_message, ep, top_features_list)
        caught_by, norm = caught_by_phases(ep)
        alert_card = build_explainable_alert_card(
            ep,
            category=category,
            rack=rack,
            caught_by=caught_by,
            norm=norm,
            top_shap_features=top_features_list,
        )

        metrics = {
            "phase1_score": round(float(ep.get("phase1_score", 0)), 3),
            "phase3_physics_z": round(float(ep.get("phase3_physics_z", 0)), 2),
            "ep_duration_min": int(ep.get("ep_duration_min", 0)),
            "ep_consensus_phases": int(ep.get("ep_consensus_phases", 0)),
        }

        rows.append(
            {
                "hostname": ep.get("hostname", ""),
                "component": ep.get("component", ""),
                "rack": rack,
                "episode_start": ep.get("episode_start"),
                "episode_end": ep.get("episode_end"),
                "fusion_prob": round(float(ep.get("fusion_prob", 0)), 4),
                "fusion_tier": str(ep.get("fusion_tier", "NONE")),
                "category": category,
                "title": title,
                "message": message,
                "target": alert_card["target"],
                "severity": alert_card["severity"],
                "detection_phase": alert_card["detection_phase"],
                "detected_at": alert_card["detected_at"],
                "associated_job": alert_card["associated_job"],
                "likely_cause": alert_card["likely_cause"],
                "operator_action": alert_card["operator_action"],
                "supporting_evidence": json.dumps(alert_card["supporting_evidence"]),
                "top_features": json.dumps(top_features_list),
                "rules_fired": "; ".join(rules),
                "constraints_violated": "; ".join(constraints),
                "metrics_of_interest": json.dumps(metrics),
                "affected_jobs": jobs_clause(ep),
                "severity_window_min": TIER_TIME_TO_ACT_MIN.get(
                    str(ep.get("fusion_tier", "")), None
                ),
                "prev_similar_alert_min_ago": ep.get("prev_similar_alert_min_ago"),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out_cols]
    return out
