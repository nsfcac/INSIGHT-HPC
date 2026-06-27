from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
    # Phase 1 — sensor-level evidence (headline + raw detector signals).
    "phase1_score",  # raw robust-z ensemble of 5 detectors
    "phase1_n_detectors_firing",  # count of phase-1 detectors above baseline
    "phase1_persistence_min",  # consecutive minutes phase1_score > 1σ
    "if_is_anomaly_rel",  # cluster-relative Isolation-Forest flag (bool)
    "lstm_recon_z",  # LSTM-AE reconstruction z-score (raw)
    "strong_stat_consensus",  # ≥3 statistical detectors agreed (bool)
    # Phase 2 — job-level / peer context.
    "phase2_score",  # 0 if phase2 disabled
    "phase2_cluster_dist",
    "phase2_peer_divergence_z",
    "coherence_anomaly",  # multi-node job coherence flag (bool)
    # Phase 3 — physics + context.
    "phase3_score",
    "phase3_physics_z",  # combined power/thermal residual magnitude (signed)
    "phase3_n_constraints",
    "physics_anomaly",  # |physics_z| > 3 (bool)
    "const3_dynamics",  # c3: power ramp without thermal response (bool)
    "const4_crossplane",  # c4: PDU vs iDRAC divergence (bool)
    # Episode-level aggregates (set during candidate construction).
    "ep_duration_min",
    "ep_peak_phase1",
    "ep_peak_phase3",
    "ep_consensus_phases",
    "phase3_context_score",
    # Diurnal context.
    "hour_sin",
    "hour_cos",
]
