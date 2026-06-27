from __future__ import annotations

import numpy as np

FLAG_LABELS = ("TEMP_FAN", "RACK_THERM", "DYNAMICS", "CROSSPLANE", "ALLOC_IDLE")

FLAG_LOOKUP = np.array(
    ["|".join(l for j, l in enumerate(FLAG_LABELS) if i & (1 << j)) for i in range(32)],
    dtype=object,
)
