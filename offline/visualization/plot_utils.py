from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Create the directory (and parents) and return it as a Path.
def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# Tighten, save a matplotlib figure to path at the given dpi, then close it.
def save_fig(fig: plt.Figure, path, dpi: int = 140) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Write an object to path as indented JSON, coercing numpy/pandas/Path values.
def write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


# Coerce numpy scalars, timestamps, and Paths into JSON-serialisable values.
def _json_default(x):
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (Path,)):
        return str(x)
    raise TypeError(f"Not JSON serializable: {type(x)}")
