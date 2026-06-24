from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.scoring import coerce_numeric, robust_z

DETECTOR_COLUMNS = (
    ("p2_job_anomaly", "job_if_anomaly", "cont"),
    ("cluster_dist", None, "cont"),
    ("coherence_anomaly", None, "bool"),
    ("peer_anomaly_pm", None, "bool"),
)

PERSISTENCE_THRESHOLD = 0.3
CONSENSUS_THRESHOLD = 0.0


# Pull a detector series (with fallback column), zero-filled when absent.
def extract(fused: pd.DataFrame, primary: str, fallback: Optional[str]) -> pd.Series:
    if primary in fused.columns:
        return coerce_numeric(fused, primary)
    if fallback is not None:
        return coerce_numeric(fused, fallback)
    return pd.Series(np.zeros(len(fused), dtype=np.float32), index=fused.index)


def zeros(n: int, index, dtype=np.float32) -> pd.Series:
    return pd.Series(np.zeros(n, dtype=dtype), index=index)


# Combine per-detector robust z-scores into the fused phase-2 score.
def compute_scores(
    fused: pd.DataFrame,
    train_mask: pd.Series,
    train_labels: Optional[pd.Series] = None,
    enabled: bool = True,
) -> pd.DataFrame:
    n = len(fused)

    if not enabled:
        return pd.DataFrame(
            {
                "phase2_score": zeros(n, fused.index),
                "phase2_cluster_dist": zeros(n, fused.index),
                "phase2_peer_divergence_z": zeros(n, fused.index),
                "phase2_minutes_into_job": zeros(n, fused.index, np.int32),
            }
        )

    per_detector = {
        primary: robust_z(extract(fused, primary, fb), train_mask)
        for primary, fb, detector_kind in DETECTOR_COLUMNS
    }
    det_df = pd.DataFrame(per_detector, index=fused.index)
    phase2_score = det_df.mean(axis=1).clip(lower=-5.0, upper=5.0).astype(np.float32)

    cluster_dist_z = per_detector.get("cluster_dist", zeros(n, fused.index))
    peer_z = per_detector.get("peer_anomaly_pm", zeros(n, fused.index))
    mins_into_job = extract(fused, "minutes_into_job", None).astype(np.int32)

    return pd.DataFrame(
        {
            "phase2_score": phase2_score,
            "phase2_cluster_dist": cluster_dist_z.astype(np.float32),
            "phase2_peer_divergence_z": peer_z.astype(np.float32),
            "phase2_minutes_into_job": mins_into_job,
        }
    )
