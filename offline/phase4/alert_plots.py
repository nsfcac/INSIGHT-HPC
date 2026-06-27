from __future__ import annotations

from typing import Iterable

import pandas as pd


# Return columns containing any needle and none of the excluded prefixes/substrings.
def matching_columns(
    df: pd.DataFrame,
    *needles: str,
    exclude: Iterable[str] = ("audit_flags__",),
) -> list:
    if df is None or len(df) == 0:
        return []
    keep = []
    for c in df.columns:
        if any(c.startswith(x) or x in c for x in exclude):
            continue
        if any(n in c for n in needles):
            keep.append(c)
    return keep
