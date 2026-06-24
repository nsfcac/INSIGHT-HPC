from __future__ import annotations

import pandas as pd

INJECTION_TYPES = {
    "gradual_thermal_drift",
    "cooling_failure",
    "peer_node_divergence",
    "idle_power_fault",
    "job_excess_power",
    "measurement_glitch",
    "memory_leak",
    "cpu_exhaustion",
    "fan_rpm_drop",
    "sensor_dropout",
    "gpu_thermal_runaway",
    "sensor_stuck_at_value",
}

# Keyword patterns to locate relevant columns in master tables
INLET_KEYWORDS = ["inlettemp", "inlet_temp", "inlettemperature", "systeminlettemp"]
FAN_KEYWORDS = [
    "fanspeed",
    "fan_speed",
    "fanrpm",
    "fan_rpm",
    "tachometerreading",
    # iDRAC naming: rpmreading__fan.embedded.1a_fan_avg etc.
    "rpmreading",
    "fan.embedded",
]
POWER_KEYWORDS = [
    "systeminputpower",
    "system_input_power",
    "powerconsumption",
    "power_consumption",
    "inputpower",
]
PDU_KEYWORDS = [
    "pdu_power",
    "pdupower",
    "rack_power",
    "rackpower",
    "pdu__pdu",
    "pdu_avg",
]

JOB_CONTEXT_EXPECT: dict[str, set[str]] = {
    "idle_power_fault": {"idle"},
    "cpu_exhaustion": {"idle", "running"},  # supports either flavor
    "fan_rpm_drop": {"idle", "running"},
    "gradual_thermal_drift": {"idle", "running"},
    "sensor_dropout": {"idle", "running"},
    "sensor_stuck_at_value": {"idle", "running"},
    "cooling_failure": {"idle", "running"},
    "memory_leak": {"idle", "running", "single_host", "multi_host"},
    "job_excess_power": {"running", "single_host"},
    "peer_node_divergence": {"multi_host"},
    "measurement_glitch": {"idle", "running"},  # PDU is infra-level
    "gpu_thermal_runaway": {"idle", "running", "single_host", "multi_host"},
}


# Classify a phase-2 segment as idle, multi_host, or single_host.
def classify_segment(seg_row: pd.Series, multi_host_jobs: set[int]) -> str:
    if bool(seg_row.get("is_idle", False)):
        return "idle"
    jid = seg_row.get("job_id")
    try:
        jid_int = int(jid) if pd.notna(jid) else None
    except (TypeError, ValueError):
        jid_int = None
    if jid_int is not None and jid_int in multi_host_jobs:
        return "multi_host"
    return "single_host"
