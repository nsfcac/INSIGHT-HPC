from __future__ import annotations

CATEGORY_TEMPLATES: dict[str, dict] = {
    "cooling_failure": {
        "title": "Cooling response lag on {host}",
        "message": "Inlet temperature has risen while fan RPM and system power remained near baseline. "
        "Check CRAC serving rack {rack}, verify fan health on {host}.",
        "tier_default": "CRITICAL",
    },
    "hardware": {
        "title": "Possible hardware failure on {host}",
        "message": "The node {host} stopped reporting telemetry during the alert window. "
        "Verify BMC/iDRAC accessibility; coordinate with job owner before reseating. "
        "Job-state and window are listed under What happened.",
        "tier_default": "CRITICAL",
    },
    "gpu_thermal_runaway": {
        "title": "GPU thermal excursion on {host}",
        "message": "GPU die temperature exceeded its operational ceiling while rack inlet "
        "remained normal — GPU-local issue. Check thermal paste / VRM on {host}. "
        "Peak readings are listed under Sensor-level details below.",
        "tier_default": "CRITICAL",
    },
    "measurement_glitch": {
        "title": "PDU/iDRAC disagreement for rack {rack}",
        "message": "PDU-reported rack power and summed iDRAC node power disagree — "
        "measurement-integrity issue. Check PDU calibration; confirm iDRAC "
        "readings on rack-local nodes.",
        "tier_default": "CONFIRMED",
    },
    "sensor_stuck_at_value": {
        "title": "Sensor frozen on {host}",
        "message": "{metric} on {host} has zero variance but a plausible value. "
        "iDRAC/BMC sensor read path likely stale; cold-reset BMC or power-cycle node.",
        "tier_default": "CONFIRMED",
    },
    "sensor_dropout": {
        "title": "Sensor coverage loss on {host}",
        "message": "{family_clause} went silent for ≥2 min during the alert window. {bmc_hint}",
        "tier_default": "CONFIRMED",
    },
    "fan_rpm_drop": {
        "title": "Fan RPM drop on {host}",
        "message": "Fan RPM dropped well below 7-day baseline. "
        "Inspect fan module; prepare replacement if drop persists. "
        "Min/peak RPM values are listed under Sensor-level details below.",
        "tier_default": "CONFIRMED",
    },
    "cpu_exhaustion": {
        "title": "CPU runaway on {host}",
        "message": "CPU utilization saturated at >95% with flat variance for ≥20 min. "
        "Identify owning job; check for runaway process or OOM loop.",
        "tier_default": "CRITICAL",
    },
    "idle_power_fault": {
        "title": "Idle-node power excess on {host}",
        "message": "The node {host} drew sustained power above its 7-day idle baseline with no active job. "
        "Likely parasitic process or firmware issue; schedule for reboot. "
        "System-power peak is listed under Sensor-level details below.",
        "tier_default": "CANDIDATE",
    },
    "job_excess_power": {
        "title": "Job power profile outlier on {host}",
        "message": "System power exceeded expected envelope for the workload cluster. "
        "Review job profile for efficiency drift; no node-level action required.",
        "tier_default": "CANDIDATE",
    },
    "peer_node_divergence": {
        "title": "Node diverging from job peers on {host}",
        "message": "Within a multi-node job, {host} shows different power/thermal profile than its peers {peer_nodes}. "
        "Check local workload stability; confirm no hardware slow-down on this node.",
        "tier_default": "CONFIRMED",
    },
    "memory_leak": {
        "title": "Slow memory growth on {host}",
        "message": "Memory-related metric shows sustained linear ramp over 12h+. "
        "Monitor for saturation; alert job owner to investigate.",
        "tier_default": "CANDIDATE",
    },
    "gradual_thermal_drift": {
        "title": "Slow thermal drift on {host}",
        "message": "Inlet temperature trending up by 0.05–0.5°C/hr. "
        "Pre-failure signal; verify cooling loop scheduling and coolant levels.",
        "tier_default": "CANDIDATE",
    },
    "unknown": {
        "title": "Anomaly detected on {host}",
        "message": "INSIGHT-HPC flagged unusual behavior. Review the contributing metrics listed below.",
        "tier_default": "CANDIDATE",
    },
}

FEATURE_HUMAN_NAMES: dict[str, str] = {
    "phase1_score": "sensor-level anomaly score",
    "phase1_n_detectors_firing": "number of phase-1 rules firing",
    "phase1_persistence_min": "minutes the anomaly has persisted",
    "phase2_score": "job-level anomaly score",
    "phase2_cluster_dist": "distance from expected workload profile",
    "phase2_peer_divergence_z": "divergence from multi-node job peers",
    "phase3_score": "physics + context anomaly score",
    "phase3_physics_z": "power/thermal residual (σ from expected)",
    "phase3_n_constraints": "physics constraints violated",
    "ep_duration_min": "anomaly duration",
    "ep_peak_phase1": "peak sensor evidence",
    "ep_peak_phase3": "peak physics evidence",
    "ep_consensus_phases": "phases agreeing (out of 3)",
    "hour_sin": "hour-of-day (cyclic)",
    "hour_cos": "hour-of-day (cyclic)",
}

RULE_CATEGORY_TABLE: tuple = (
    (
        "rule_FLATLINE",
        "cpu_exhaustion",
        lambda ep: bool(ep.get("cpuusage_high", False)),
    ),
    ("rule_FLATLINE", "sensor_stuck_at_value", None),
    ("rule_DROPOUT", "sensor_dropout", None),
    ("rule_HIGH_GPU_TEMP", "gpu_thermal_runaway", None),
    (
        "rule_FAN_RPM_PCT_DROP",
        "fan_rpm_drop",
        lambda ep: float(ep.get("ep_pct_fan_nan", 0.0) or 0.0) < 0.20,
    ),
    (
        "rule_FAN_FAIL",
        "fan_rpm_drop",
        lambda ep: float(ep.get("ep_pct_fan_nan", 0.0) or 0.0) < 0.20,
    ),
    ("rule_IDLE_POWER_SURGE", "idle_power_fault", None),
    ("const4_crossplane", "measurement_glitch", None),
    ("const3_dynamics", "cooling_failure", None),
)

TIER_TIME_TO_ACT_MIN: dict[str, int] = {
    "CRITICAL": 15,
    "CONFIRMED": 60,
    "CANDIDATE": 240,
}

PHASE_SCALE: dict[str, float] = {
    "phase1_score": 4.0,
    "phase2_score": 3.0,
    "phase3_score": 3.0,
}

JOINT_TIE_RATIO = 0.10

RAW_SATURATION = 4.0

ROMAN_PHASE = {
    "phase1_score": "Phase I",
    "phase2_score": "Phase II",
    "phase3_score": "Phase III",
}

LIKELY_CAUSE_BY_CATEGORY: dict[str, str] = {
    "cooling_failure": "Cooling response lag or local cooling-path issue",
    "hardware": "Node telemetry loss or management-controller failure",
    "gpu_thermal_runaway": "GPU-local thermal issue",
    "measurement_glitch": "PDU/iDRAC measurement disagreement",
    "sensor_stuck_at_value": "BMC/iDRAC sensor read path stuck at a stale value",
    "sensor_dropout": "BMC/iDRAC sensor-pipeline dropout",
    "fan_rpm_drop": "Fan module degradation or failure",
    "cpu_exhaustion": "Runaway process, OOM loop, or saturated CPU workload",
    "idle_power_fault": "Parasitic process or firmware issue",
    "job_excess_power": "Workload power profile outlier",
    "peer_node_divergence": "Local node divergence from same-job peers",
    "memory_leak": "Sustained application or job memory growth",
    "gradual_thermal_drift": "Slow cooling-loop or ambient thermal drift",
    "unknown": "Multi-phase anomaly requiring operator review",
}
