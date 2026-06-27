from __future__ import annotations

TS = "timestamp"

WINDOW = 30
STRIDE_TRAIN = 5
STRIDE_SCORE = 1
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 30
BATCH = 256
LR = 1e-3
MIN_WINDOWS = 200
MIN_SOLO_WINDOWS = 50
ANOM_Z_THRESH = 5.0
HIGH_SCORE_THRESH = 0.70
POINT_SPIKE_SCORE = 0.30
MAX_ANOMALY_RATE = 0.20
ITER_ROUNDS_DEFAULT = 2
ITER_CLEAN_PERCENTILE_DEFAULT = 95.0

SENSOR_KEYWORDS = [
    "systeminputpower",
    "totalcpupower",
    "totalmemorypower",
    "temperaturereading",
    "rpmreading",
    "gpuusage",
    "gpumemoryusage",
    "pdu__pdu",
]
