from __future__ import annotations


# Upsert episode lifecycle rows into the insight_hpc DB.
def write_episode(engine, event):
    raise NotImplementedError("episode writer")
