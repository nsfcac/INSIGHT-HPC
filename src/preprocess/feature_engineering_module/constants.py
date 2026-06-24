from __future__ import annotations

import numpy as np

TS = "timestamp"

ROLL_WINDOWS = [5, 15, 60]
QUANT_WINDOWS = [15, 60]
QUANT_LEVELS = [0.05, 0.25, 0.75, 0.95]
PHYSICS_LAGS = [1, 5, 15]
DRIFT_WINDOWS = [1440, 10080]
TWO_PI = 2 * np.pi
