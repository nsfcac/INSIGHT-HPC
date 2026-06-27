from __future__ import annotations

import gc, io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import phase1_workers
from shared.utils.maintenance import apply_maintenance_mask
from shared.utils.rack_topology import rack_id


from offline.phase1.baseline_threshold_module.threshold_loading import *
from offline.phase1.baseline_threshold_module.node_scoring import *


# Sum per-outlet PDU power into per-rack, per-minute totals.
def build_rack_pdu_power(master_dir: Path, windows: list) -> pd.DataFrame:
    pdu_dir = master_dir / "infra" / "pdu"
    if not pdu_dir.exists():
        return pd.DataFrame(columns=["timestamp", "rack_id", "pdu_power_sum_w"])

    frames = []
    for p in sorted(pdu_dir.glob("*.parquet")):
        unit_id = p.stem
        rid = rack_id(unit_id)
        if rid is None:
            continue

        df = read_threshold_frame(p)
        if df is None or df.empty or "timestamp" not in df.columns:
            continue

        df = apply_maintenance_mask(df, windows, "timestamp", None)
        pdu_cols = [c for c in df.columns if c.endswith("_avg") and "pdu" in c.lower()]
        if not pdu_cols:
            continue

        outlet_w = df[pdu_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df["timestamp"], utc=True),
                    "rack_id": rid,
                    "outlet_w": outlet_w.values.astype("float32"),
                }
            ).dropna()
        )

    if not frames:
        return pd.DataFrame(columns=["timestamp", "rack_id", "pdu_power_sum_w"])

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = combined["timestamp"].dt.floor("1min")
    rack_pdu = (
        combined.groupby(["timestamp", "rack_id"])["outlet_w"]
        .sum(min_count=1)
        .rename("pdu_power_sum_w")
        .reset_index()
    )
    del combined
    gc.collect()
    return rack_pdu


# Count how many PDU outlets each rack has.
def rack_outlet_counts(master_dir: Path) -> Dict[int, int]:
    pdu_dir = master_dir / "infra" / "pdu"
    if not pdu_dir.exists():
        return {}
    counts: Dict[int, int] = {}
    for p in sorted(pdu_dir.glob("*.parquet")):
        rid = rack_id(p.stem)
        if rid is not None:
            counts[rid] = counts.get(rid, 0) + 1
    return counts


# Scale rack-budget ratios by a rack's PDU-outlet coverage fraction.
def coverage_adjusted_ratios(
    rid: int, outlet_counts: Dict[int, int], ratio_high: float, ratio_low: float
) -> tuple:
    if not outlet_counts:
        return ratio_low, ratio_high, 1.0

    from collections import Counter

    mode_count = Counter(outlet_counts.values()).most_common(1)[0][0]
    rack_count = outlet_counts.get(rid, mode_count)
    coverage = rack_count / mode_count if mode_count > 0 else 1.0
    coverage = min(coverage, 1.0)

    return ratio_low * coverage, ratio_high * coverage, coverage


# Flag PDU outlets exceeding their p95 threshold or the rack power budget.
def score_pdu(
    master_dir: Path, thresholds: Dict[str, float], windows: list, cfg: dict
) -> Optional[pd.DataFrame]:
    pdu_dir = master_dir / "infra" / "pdu"
    if not pdu_dir.exists() or not thresholds:
        return None

    cfg_thr = cfg["phase1"]["threshold_baseline"]
    ratio_high = float(cfg_thr.get("pdu_rack_budget_ratio_high", 1.50))
    ratio_low = float(cfg_thr.get("pdu_rack_budget_ratio_low", 0.85))

    outlet_counts = rack_outlet_counts(master_dir)
    from collections import Counter

    if outlet_counts:
        mode_count = Counter(outlet_counts.values()).most_common(1)[0][0]
        partial = {r: c for r, c in outlet_counts.items() if c < mode_count}
        if partial:
            print(
                f"  PDU coverage: mode={mode_count} outlets/rack  "
                f"partial racks: { {r: f'{c}/{mode_count}' for r, c in partial.items()} }"
            )

    print("  Computing rack-level iDRAC power for PDU budget check ...")
    rack_idrac = build_rack_idrac_power(master_dir, cfg)

    print("  Computing rack-level PDU total (summing all outlets per rack) ...")
    rack_pdu = build_rack_pdu_power(master_dir, windows)

    has_budget = (not rack_idrac.empty) and (not rack_pdu.empty)
    if has_budget:
        rack_idrac["ts_floor"] = rack_idrac["timestamp"].dt.floor("1min")
        rack_pdu["ts_floor"] = rack_pdu["timestamp"].dt.floor("1min")

        rack_budget = pd.merge(
            rack_pdu[["ts_floor", "rack_id", "pdu_power_sum_w"]],
            rack_idrac[["ts_floor", "rack_id", "idrac_power_sum_w"]],
            on=["ts_floor", "rack_id"],
            how="inner",
        )

        # Flag a rack-minute whose PDU/iDRAC power ratio leaves the coverage-adjusted band.
        def flag_budget(row):
            r_low, r_high, _ = coverage_adjusted_ratios(
                int(row["rack_id"]), outlet_counts, ratio_high, ratio_low
            )
            ratio = row["pdu_power_sum_w"] / max(row["idrac_power_sum_w"], 1.0)
            return bool(ratio > r_high or ratio < r_low)

        rack_budget["is_flagged_rack_budget"] = rack_budget.apply(flag_budget, axis=1)
        rack_budget = rack_budget.set_index(["ts_floor", "rack_id"])
    else:
        rack_budget = pd.DataFrame()

    frames = []
    for p in sorted(pdu_dir.glob("*.parquet")):
        unit_id = p.stem
        thr = thresholds.get(unit_id)
        if thr is None:
            continue

        df = read_threshold_frame(p)
        if df is None or df.empty:
            continue

        rid = rack_id(unit_id)
        df = apply_maintenance_mask(df, windows, "timestamp", None)

        pdu_cols = [c for c in df.columns if c.endswith("_avg") and "pdu" in c.lower()]
        if not pdu_cols:
            continue

        vals = df[pdu_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        ts = pd.to_datetime(df["timestamp"], utc=True)

        frame = pd.DataFrame(
            {
                "timestamp": ts,
                "unit_id": unit_id,
                "rack_id": rid,
                "split": df["split"] if "split" in df.columns else "unknown",
                "maintenance_flag": df["maintenance_flag"].values,
                "pdu_value": vals.values.astype("float32"),
                "pdu_p95_threshold": float(thr),
                "is_flagged_outlet": (vals > thr).values,
            }
        )

        if has_budget and rid is not None and not rack_budget.empty:
            frame["ts_floor"] = frame["timestamp"].dt.floor("1min")
            try:
                joined = frame.join(
                    rack_budget.loc[
                        rack_budget.index.get_level_values("rack_id") == rid
                    ].droplevel("rack_id"),
                    on="ts_floor",
                    how="left",
                )
                frame["rack_pdu_power_sum"] = joined["pdu_power_sum_w"].values
                frame["idrac_power_sum_w"] = joined["idrac_power_sum_w"].values
                frame["is_flagged_rack_budget"] = (
                    joined["is_flagged_rack_budget"].eq(True).values
                )
            except Exception:
                rb_slice = rack_budget.reset_index()
                rb_slice = rb_slice[rb_slice["rack_id"] == rid][
                    [
                        "ts_floor",
                        "pdu_power_sum_w",
                        "idrac_power_sum_w",
                        "is_flagged_rack_budget",
                    ]
                ]
                frame = frame.merge(rb_slice, on="ts_floor", how="left")
                frame["rack_pdu_power_sum"] = frame["pdu_power_sum_w"]
                frame["idrac_power_sum_w"] = frame.get("idrac_power_sum_w", np.nan)
                frame["is_flagged_rack_budget"] = frame["is_flagged_rack_budget"].eq(
                    True
                )
            frame.drop(columns=["ts_floor"], errors="ignore", inplace=True)
        else:
            frame["rack_pdu_power_sum"] = np.nan
            frame["idrac_power_sum_w"] = np.nan
            frame["is_flagged_rack_budget"] = False

        frame["pdu_budget_ratio_high"] = float(ratio_high)
        frame["pdu_budget_ratio_low"] = float(ratio_low)

        if rid is not None:
            r_low_adj, r_high_adj, cov = coverage_adjusted_ratios(
                rid, outlet_counts, ratio_high, ratio_low
            )
            frame["pdu_budget_ratio_high_adj"] = float(r_high_adj)
            frame["pdu_budget_ratio_low_adj"] = float(r_low_adj)
            frame["pdu_coverage_frac"] = float(cov)
        else:
            frame["pdu_budget_ratio_high_adj"] = float(ratio_high)
            frame["pdu_budget_ratio_low_adj"] = float(ratio_low)
            frame["pdu_coverage_frac"] = 1.0

        frame["is_flagged"] = frame["is_flagged_outlet"].astype(bool) | frame[
            "is_flagged_rack_budget"
        ].astype(bool)

        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else None


THRESHOLD_WORKER_CONTEXT: dict = {}


# Score one node against thresholds in a worker process.
def threshold_score_one(comp: str, hostname: str, parquet_str: str):
    ctx = THRESHOLD_WORKER_CONTEXT
    effective_thr = ctx["thr"][comp]
    windows = ctx["windows"]

    buf = io.StringIO()
    with redirect_stdout(buf):
        df = read_threshold_frame(Path(parquet_str))
        if df is None or df.empty:
            return hostname, None, 0, 0, None
        scored = score_node(df, hostname, comp, effective_thr, windows)
        if scored is None or scored.empty:
            return hostname, None, 0, 0, None
        n_rows = len(scored)
        n_flagged = int(scored["is_flagged"].sum())
        del df
        gc.collect()
    summary = (
        f"    {hostname:20s}: {n_rows:>8,} rows  "
        f"flagged={n_flagged:>6,} ({100*n_flagged/max(n_rows,1):5.1f}%)"
    )
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    return hostname, scored, n_rows, n_flagged, log
