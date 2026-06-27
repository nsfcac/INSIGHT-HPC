from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.scoring import coerce_numeric, robust_z

# Aligns with job_context_annotator labels: 1.0 = fault-like, 0.0 = job-explained.
CONTEXT_SCORE_MAP = {
    "JOB_EXPLAINED": 0.0,
    "COOLING_FAULT": 1.0,
    "COOLING_DEMAND": 0.6,
    "SENSOR_DRIFT": 0.8,
    "MEASUREMENT_DISCREPANCY": 0.7,
    "INFRA_FAULT": 1.0,
    "NO_JOBS": 0.8,
    "AMBIGUOUS": 0.5,
    "JOB_OVER_EXPECTATION": 0.9,
}


# Combine physics residuals, constraint counts, and job context into the phase-3 score.
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
                "phase3_score": pd.Series(
                    np.zeros(n, dtype=np.float32), index=fused.index
                ),
                "phase3_physics_z": pd.Series(
                    np.zeros(n, dtype=np.float32), index=fused.index
                ),
                "phase3_n_constraints": pd.Series(
                    np.zeros(n, dtype=np.int32), index=fused.index
                ),
                "phase3_context_score": pd.Series(
                    np.zeros(n, dtype=np.float32), index=fused.index
                ),
            }
        )

    # physics_z: signed max-magnitude of power/thermal residuals; fall back to split cols.
    if "physics_z" in fused.columns:
        physics_z_series = coerce_numeric(fused, "physics_z")
    else:
        pz = coerce_numeric(fused, "power_residual_z")
        tz = coerce_numeric(fused, "thermal_residual_z")
        physics_z_arr = np.where(pz.abs() >= tz.abs(), pz, tz).astype(np.float32)
        physics_z_series = pd.Series(physics_z_arr, index=fused.index)

    n_constraints = coerce_numeric(fused, "n_constraints_violated").astype(np.int32)
    ctx_label = fused.get(
        "anomaly_context", pd.Series(["AMBIGUOUS"] * n, index=fused.index)
    )
    context_score = (
        ctx_label.astype(str).map(CONTEXT_SCORE_MAP).fillna(0.5).astype(np.float32)
    )

    physics_mag_norm = robust_z(physics_z_series.abs(), train_mask).clip(lower=0)
    constraint_norm = (n_constraints.astype(np.float32) / 5.0).clip(0, 1)
    evidence = (physics_mag_norm.clip(0, 3) / 3.0 + constraint_norm) / 2.0
    phase3_score = (evidence * context_score).astype(np.float32)

    const4 = fused.get("const4_crossplane", pd.Series(False, index=fused.index))
    if const4.dtype != bool:
        const4 = const4.fillna(False).astype(bool)
    if "hostname" in fused.columns and "component" in fused.columns:
        grp = const4.groupby([fused["hostname"], fused["component"]], sort=False)
        sustained = (
            const4
            & grp.shift(1).fillna(False).astype(bool)
            & grp.shift(2).fillna(False).astype(bool)
        )
    else:
        sustained = const4
    phase3_score = pd.Series(
        np.where(
            sustained.to_numpy(),
            np.maximum(phase3_score.to_numpy(), 0.8),
            phase3_score.to_numpy(),
        ),
        index=fused.index,
    ).astype(np.float32)

    return pd.DataFrame(
        {
            "phase3_score": phase3_score,
            "phase3_physics_z": physics_z_series.astype(np.float32),
            "phase3_n_constraints": n_constraints,
            "phase3_context_score": context_score,
        }
    )
