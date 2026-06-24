from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.scoring import consecutive_true_run, robust_z

DETECTOR_COLUMNS = (
    ("thr_flag", None, "bool"),
    ("if_anomaly_score", "if_is_anomaly", "cont"),
    ("zscore_flag", "zscore_max", "bool"),
    ("ewma_flag", "ewma_residual", "bool"),
    ("lstm_mae_z", "lstm_is_anomaly", "cont"),
)

PERSISTENCE_THRESHOLD = 1.0
CONSENSUS_THRESHOLD = 0.0


# Pull a detector's numeric series (with fallback column) from the fused frame, zero-filled.
def extract_detector_series(
    fused: pd.DataFrame, primary: str, fallback: Optional[str], kind: str
) -> pd.Series:
    if primary in fused.columns:
        s = fused[primary]
    elif fallback is not None and fallback in fused.columns:
        s = fused[fallback]
    else:
        return pd.Series(np.zeros(len(fused), dtype=np.float32), index=fused.index)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float32)


# Combine per-detector robust z-scores into the fused phase-1 score and firing stats.
def compute_scores(
    fused: pd.DataFrame, train_mask: pd.Series, train_labels: Optional[pd.Series] = None
) -> pd.DataFrame:
    per_detector = {
        primary: robust_z(extract_detector_series(fused, primary, fb, kind), train_mask)
        for primary, fb, kind in DETECTOR_COLUMNS
    }

    det_df = pd.DataFrame(per_detector, index=fused.index)
    phase1_score = det_df.mean(axis=1).clip(lower=-5.0, upper=5.0).astype(np.float32)

    n_firing = (det_df > CONSENSUS_THRESHOLD).sum(axis=1).astype(np.int32)
    max_det = det_df.max(axis=1).astype(np.float32)

    firing = phase1_score > PERSISTENCE_THRESHOLD
    if "hostname" in fused.columns and "component" in fused.columns:
        persistence = firing.groupby(
            [fused["hostname"], fused["component"]], dropna=False
        ).transform(lambda g: consecutive_true_run(g).to_numpy())
    else:
        persistence = consecutive_true_run(firing).to_numpy()
    persistence = pd.Series(persistence, index=fused.index).fillna(0).astype(np.int32)

    return pd.DataFrame(
        {
            "phase1_score": phase1_score,
            "phase1_n_detectors_firing": n_firing,
            "phase1_persistence_min": persistence,
            "phase1_max_detector_score": max_det,
        }
    )
