from __future__ import annotations

import numpy as np

AUDIT_SPLIT_MIN_ROWS = 80_000_000

FLAG_DUP = np.int8(1)
FLAG_SPIKE = np.int8(2)
FLAG_FLATLINE = np.int8(4)
FLAG_BAD_RANGE = np.int8(8)
