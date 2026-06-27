from __future__ import annotations

import json, math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd



# Attribution scores for one concurrent job on the anomalous node.
@dataclass
class CandidateJob:
    job_id: int
    cpu_share_frac: float
    attribution_score: float
    signal_power: Optional[float]
    signal_onset: Optional[float]
    signal_cluster: Optional[float]
    signal_coherence: Optional[float]


# Output of attribute_concurrent_jobs() for one anomaly window.
@dataclass
class AttributionResult:
    primary_job_id: int
    confidence: float
    candidates: list[CandidateJob] = field(default_factory=list)
    signals_used: list[str] = field(default_factory=list)
    confidence_reason: str = ""


POWER_KW_PREFERENCE = [
    "systeminputpower",
    "systempowerconsumption",
    "totalcpupower",
]


# Pick the preferred non-zero power "_avg" column from a window.
def detect_power_col(df: pd.DataFrame) -> Optional[str]:
    for kw in POWER_KW_PREFERENCE:
        for col in df.columns:
            if kw in col.lower() and col.endswith("_avg"):
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0 and s.mean() > 0:
                    return col
    return None


# Sum per-job CPU shares across a window from the cpu_shares/jobs JSON.
def parse_cpu_shares(window: pd.DataFrame) -> dict[int, float]:
    job_shares: dict[int, float] = {}
    if "cpu_shares_json" not in window.columns:
        return job_shares
    share_cols = ["cpu_shares_json"]
    if "jobs_json" in window.columns:
        share_cols.append("jobs_json")

    for _, row in window[share_cols].dropna(subset=["cpu_shares_json"]).iterrows():
        try:
            shares = (
                json.loads(row["cpu_shares_json"])
                if isinstance(row["cpu_shares_json"], str)
                else row["cpu_shares_json"]
            )

            if isinstance(shares, dict):
                for jid_raw, share in shares.items():
                    if jid_raw is None or share is None:
                        continue
                    jid = int(jid_raw)
                    job_shares[jid] = job_shares.get(jid, 0.0) + float(share)
                continue

            if not isinstance(shares, list) or "jobs_json" not in row:
                continue

            jobs = (
                json.loads(row["jobs_json"])
                if isinstance(row["jobs_json"], str)
                else row["jobs_json"]
            )
            if not isinstance(jobs, list):
                continue

            for jid_raw, share in zip(jobs, shares):
                if jid_raw is None or share is None:
                    continue
                jid = int(jid_raw)
                job_shares[jid] = job_shares.get(jid, 0.0) + float(share)
        except Exception:
            pass
    return job_shares


# Return the first timestamp a job appears in the window.
def job_first_seen(window: pd.DataFrame, job_id: int) -> Optional[pd.Timestamp]:
    if "jobs_json" not in window.columns or "timestamp" not in window.columns:
        return None
    for _, row in window.sort_values("timestamp").iterrows():
        try:
            jobs = (
                json.loads(row["jobs_json"])
                if isinstance(row["jobs_json"], str)
                else row["jobs_json"]
            )
            if job_id in [int(j) for j in (jobs or [])]:
                return pd.Timestamp(row["timestamp"])
        except Exception:
            pass
    return None


# Score how much a job's attributed power exceeds its expected profile.
def signal_power_residual(
    job_id: int,
    cpu_frac: float,
    p_actual: float,
    profiles: pd.DataFrame,
    hostname: str,
    component: str,
) -> Optional[float]:
    if p_actual <= 0 or cpu_frac <= 0:
        return None

    p_attributed = cpu_frac * p_actual

    prof = profiles[
        (profiles["job_id"] == job_id)
        & (profiles["hostname"] == hostname)
        & (profiles["component"] == component)
    ]
    if prof.empty:
        prof = profiles[
            (profiles["job_id"] == job_id) & (profiles["component"] == component)
        ]

    if prof.empty or "pwr_mean" not in prof.columns:
        return None

    pwr_vals = pd.to_numeric(prof["pwr_mean"], errors="coerce").dropna()
    if pwr_vals.empty:
        return None

    p_expected = float(pwr_vals.median()) * cpu_frac
    excess_frac = (p_attributed - p_expected) / p_actual
    return float(np.clip(excess_frac, 0.0, 1.0))


# Score how closely a job's start aligns with the anomaly onset.
def signal_onset_alignment(
    job_id: int,
    anomaly_onset: pd.Timestamp,
    window: pd.DataFrame,
    job_scores: pd.DataFrame,
    hostname: str,
) -> Optional[float]:
    job_start: Optional[pd.Timestamp] = None
    if not job_scores.empty and "job_id" in job_scores.columns:
        match = job_scores[
            (job_scores["job_id"] == job_id) & (job_scores["hostname"] == hostname)
        ]
        if not match.empty:
            job_start = pd.to_datetime(match["seg_start"], utc=True).min()

    if job_start is None:
        job_start = job_first_seen(window, job_id)

    if job_start is None:
        return None

    onset_delta_min = (anomaly_onset - job_start).total_seconds() / 60.0

    if onset_delta_min >= 0:
        return float(math.exp(-onset_delta_min / 15.0))
    return float(math.exp(onset_delta_min / 30.0))


# Score a job by its per-segment cluster anomaly score.
def signal_cluster_deviation(
    job_id: int, hostname: str, job_scores: pd.DataFrame
) -> Optional[float]:
    if job_scores.empty or "job_id" not in job_scores.columns:
        return None
    match = job_scores[
        (job_scores["job_id"] == job_id) & (job_scores["hostname"] == hostname)
    ]
    if match.empty:
        return None
    scores = pd.to_numeric(match["job_anomaly_score"], errors="coerce").dropna()
    if scores.empty:
        return None
    return float(np.clip(scores.max(), 0.0, 1.0))


# Score a job by its multi-node coherence z on this host.
def signal_coherence(
    job_id: int,
    hostname: str,
    component: str,
    coherence_df: pd.DataFrame,
    z_norm: float = 5.0,
) -> Optional[float]:
    if coherence_df.empty or "job_id" not in coherence_df.columns:
        return None
    match = coherence_df[
        (coherence_df["job_id"] == job_id) & (coherence_df["hostname"] == hostname)
    ]
    if not match.empty and "component" in match.columns:
        match = match[match["component"] == component]
    if match.empty:
        return None
    z = pd.to_numeric(match["coherence_z"], errors="coerce").dropna()
    if z.empty:
        return None
    return float(np.clip(float(z.max()) / z_norm, 0.0, 1.0))


DEFAULT_WEIGHTS: dict[str, float] = {
    "power": 0.30,
    "onset": 0.30,
    "cluster": 0.25,
    "coherence": 0.15,
}


# Renormalise signal weights over the signals actually present.
def active_weights(
    raw_candidates: list[dict], base_weights: dict[str, float]
) -> dict[str, float]:
    present = {k for k in base_weights if any(c[k] is not None for c in raw_candidates)}
    if not present:
        return {}
    sub = {k: v for k, v in base_weights.items() if k in present}
    total = sum(sub.values())
    return {k: v / total for k, v in sub.items()}


# Rank concurrent jobs on a node by a weighted multi-signal attribution score.
def attribute_concurrent_jobs(
    window: pd.DataFrame,
    anomaly_onset: pd.Timestamp,
    hostname: str,
    component: str,
    job_scores: pd.DataFrame,
    job_profiles: pd.DataFrame,
    coherence_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> Optional[AttributionResult]:
    base_w = dict(DEFAULT_WEIGHTS)
    if weights:
        base_w.update(weights)

    job_shares = parse_cpu_shares(window)
    if not job_shares:
        return None

    total_cpu = sum(job_shares.values()) or 1.0
    cpu_fracs = {jid: sh / total_cpu for jid, sh in job_shares.items()}

    p_col = detect_power_col(window)
    p_actual = (
        float(pd.to_numeric(window[p_col], errors="coerce").dropna().mean())
        if p_col
        else 0.0
    )

    raw: list[dict] = []
    for jid, cpu_frac in cpu_fracs.items():
        raw.append(
            {
                "job_id": jid,
                "cpu_share_frac": cpu_frac,
                "power": signal_power_residual(
                    jid, cpu_frac, p_actual, job_profiles, hostname, component
                ),
                "onset": signal_onset_alignment(
                    jid, anomaly_onset, window, job_scores, hostname
                ),
                "cluster": signal_cluster_deviation(jid, hostname, job_scores),
                "coherence": signal_coherence(jid, hostname, component, coherence_df),
            }
        )

    norm_w = active_weights(raw, base_w)
    signals_used = sorted(norm_w.keys())

    if not norm_w:
        raw.sort(key=lambda c: c["cpu_share_frac"], reverse=True)
        candidates = [
            CandidateJob(
                job_id=c["job_id"],
                cpu_share_frac=c["cpu_share_frac"],
                attribution_score=c["cpu_share_frac"],
                signal_power=None,
                signal_onset=None,
                signal_cluster=None,
                signal_coherence=None,
            )
            for c in raw
        ]
        return AttributionResult(
            primary_job_id=candidates[0].job_id,
            confidence=candidates[0].attribution_score,
            candidates=candidates,
            signals_used=["cpu_share_fallback"],
            confidence_reason="cpu_share_fallback: no attribution signals available",
        )

    for c in raw:
        c["attribution_score"] = float(
            sum(norm_w[sig] * (c[sig] if c[sig] is not None else 0.0) for sig in norm_w)
        )

    raw.sort(key=lambda c: c["attribution_score"], reverse=True)

    candidates = [
        CandidateJob(
            job_id=c["job_id"],
            cpu_share_frac=c["cpu_share_frac"],
            attribution_score=c["attribution_score"],
            signal_power=c["power"],
            signal_onset=c["onset"],
            signal_cluster=c["cluster"],
            signal_coherence=c["coherence"],
        )
        for c in raw
    ]

    return AttributionResult(
        primary_job_id=candidates[0].job_id,
        confidence=candidates[0].attribution_score,
        candidates=candidates,
        signals_used=signals_used,
        confidence_reason=build_confidence_reason(candidates, signals_used, norm_w),
    )


# Explain the attribution confidence (clear leader, onset timing, or inconclusive).
def build_confidence_reason(
    candidates: list[CandidateJob], signals_used: list[str], norm_w: dict[str, float]
) -> str:
    if len(candidates) == 1:
        return "single_job: unambiguous attribution"

    top = candidates[0]
    second = candidates[1]
    gap = top.attribution_score - second.attribution_score
    coherence_absent = "coherence" not in signals_used

    if gap > 0.20:
        dominant = max(
            [
                ("onset", top.signal_onset or 0.0),
                ("power", top.signal_power or 0.0),
                ("cluster", top.signal_cluster or 0.0),
                ("coherence", top.signal_coherence or 0.0),
            ],
            key=lambda x: x[1],
        )[0]
        suffix = " (single-node job, no coherence)" if coherence_absent else ""
        return f"{dominant}_dominant: clear leader (gap={gap:.2f}){suffix}"

    onset_w = norm_w.get("onset", 0.0)
    if onset_w >= 0.40:
        onset_top = top.signal_onset or 0.0
        onset_second = second.signal_onset or 0.0
        if abs(onset_top - onset_second) > 0.15:
            return (
                f"onset_timing_dominant: job {top.job_id} arrived first "
                f"(onset gap={abs(onset_top - onset_second):.2f})"
            )

    n_close = sum(
        1 for c in candidates if top.attribution_score - c.attribution_score <= 0.05
    )
    suffix = " (single-node job, no coherence available)" if coherence_absent else ""
    return (
        f"co_running_similar_jobs: {n_close} jobs with near-identical attribution scores "
        f"(gap={gap:.2f}){suffix} — result is inconclusive"
    )


# Serialise an AttributionResult to a JSON-friendly dict.
def attribution_to_dict(result: AttributionResult) -> dict:
    def r(v):
        return round(float(v), 4) if v is not None else None

    return {
        "primary_job_id": int(result.primary_job_id),
        "confidence": r(result.confidence),
        "confidence_reason": result.confidence_reason,
        "signals_used": result.signals_used,
        "candidates": [
            {
                "job_id": int(c.job_id),
                "cpu_share_frac": r(c.cpu_share_frac),
                "attribution_score": r(c.attribution_score),
                "signal_power": r(c.signal_power),
                "signal_onset": r(c.signal_onset),
                "signal_cluster": r(c.signal_cluster),
                "signal_coherence": r(c.signal_coherence),
            }
            for c in result.candidates
        ],
    }


# Aggregate anomalous segments into per-job multi-node / single-node alerts.
def collect_multi_node(scores: pd.DataFrame) -> pd.DataFrame:
    anomalous = scores[scores["job_if_anomaly"].fillna(False)]
    if anomalous.empty:
        return pd.DataFrame()

    records = []
    for (jid, comp), grp in anomalous.groupby(["job_id", "component"], dropna=True):
        all_job_nodes = scores[
            (scores["job_id"] == jid) & (scores["component"] == comp)
        ]["hostname"].unique()

        anom_nodes = grp["hostname"].unique().tolist()
        clean_nodes = [n for n in all_job_nodes if n not in anom_nodes]

        records.append(
            {
                "job_id": jid,
                "component": comp,
                "alert_type": "multi_node" if len(anom_nodes) > 1 else "single_node",
                "first_flag_time": grp["seg_start"].min(),
                "last_flag_time": grp["seg_end"].max(),
                "duration_flagged_min": float(grp["duration_min"].sum()),
                "max_anomaly_score": float(grp["job_anomaly_score"].max()),
                "mean_anomaly_score": float(grp["job_anomaly_score"].mean()),
                "anomalous_nodes": "|".join(sorted(anom_nodes)),
                "clean_nodes": "|".join(sorted(clean_nodes)),
                "n_anomalous_nodes": int(len(anom_nodes)),
                "n_clean_nodes": int(len(clean_nodes)),
                "split": grp["split"].mode().iloc[0] if len(grp) > 0 else "unknown",
            }
        )

    return pd.DataFrame(records) if records else pd.DataFrame()
