from __future__ import annotations

TS = "timestamp"

RHO_AIR = 1.2  # kg/m³ air density at sea level (~altitude correction needed)
CP_AIR = 1005.0  # J/(kg·K) specific heat of air
POWER_Z_THRESH = 5.0
THERMAL_Z_THRESH = 4.0
POOL_MAX_ROWS_PER_NODE = 4000  # cap rows sampled per node (memory guard)
POOL_RANDOM_SEED = 42
JOB_TRANSITION_TAU_MIN = 5.0  # τ  — at t=15 min, attenuation ≈ 0.95
JOB_TRANSITION_WINDOW = 20.0  # ignore transitions older than this (minutes)
