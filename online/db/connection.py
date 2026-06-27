from __future__ import annotations


# Build a read-only SQLAlchemy engine to the MonSTer TimescaleDB.
def monster_engine():
    raise NotImplementedError("read-only MonSTer DB engine")


# Build a read-write engine to the separate insight_hpc DB.
def insight_engine():
    raise NotImplementedError("insight_hpc DB engine")
