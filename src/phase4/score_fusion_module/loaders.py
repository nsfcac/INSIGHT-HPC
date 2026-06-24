from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
from pathlib import Path

import pandas as pd

from src.utils.io_utils import load_parquet


def load_if_scores(models_dir: Path):
    d = models_dir / "isolation_forest" / "scores"
    if not d.exists():
        return None
    frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(d.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def load_threshold_alerts(models_dir: Path):
    out_dir = models_dir / "baseline_threshold"
    frames = []
    for name in ["alerts.parquet", "pdu_alerts.parquet"]:
        df = load_parquet(out_dir / name)
        if df is not None and not df.empty:
            if "hostname" not in df.columns and "unit_id" in df.columns:
                df = df.rename(columns={"unit_id": "hostname"})
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_baseline_alerts(models_dir: Path, detector_name: str):
    return load_parquet(models_dir / "baselines" / f"{detector_name}_alerts.parquet")


def load_physics_scores(models_dir: Path):
    candidate = models_dir / "physics" / "residual_scores"
    if candidate.exists():
        frames = [
            pd.read_parquet(p, engine="pyarrow")
            for p in sorted(candidate.glob("*.parquet"))
        ]
        if frames:
            return pd.concat(frames, ignore_index=True)
    return None


def load_constraint_violations(models_dir: Path):
    p = models_dir / "physics" / "constraint_violations.parquet"
    if p.exists():
        return load_parquet(p)
    return None


def load_phase2_scores(phase2_dir: Path):
    return load_parquet(phase2_dir / "job_anomaly_scores.parquet")


def load_coherence(phase2_dir: Path):
    return load_parquet(phase2_dir / "multi_node_coherence.parquet")


def load_per_minute_coherence(phase2_dir: Path):
    return load_parquet(phase2_dir / "per_minute_coherence.parquet")


def load_streaming_coherence(phase2_dir: Path):
    return load_parquet(phase2_dir / "streaming_coherence.parquet")


def load_lstm_scores(models_dir: Path):
    scores_dir = models_dir / "lstm_ae" / "scores"
    if not scores_dir.exists():
        return None
    frames = [
        pd.read_parquet(p, engine="pyarrow")
        for p in sorted(scores_dir.glob("*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True) if frames else None


def load_job_context(phase3_dir: Path):
    p = phase3_dir / "physics" / "job_context_anomalies.parquet"
    if not p.exists():
        return None
    return load_parquet(p)
