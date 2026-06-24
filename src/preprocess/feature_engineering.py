from __future__ import annotations

import gc, io, json, os, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.utils.parsers import first_existing, is_missing, parse_int_list, to_int
from src.utils.rack_topology import rack_id
from src.utils.regime_utils import (
    REGIME_INSIDE_MAINTENANCE,
    attach_regime_id,
    regime_safe_slope,
    rolling_by_regime,
)

from src.preprocess.feature_engineering_module.transforms import *
from src.preprocess.feature_engineering_module.normalization_pdu import *


def feature_engineering(force: bool = False) -> None:
    global ROLL_WINDOWS, QUANT_WINDOWS, QUANT_LEVELS, DRIFT_WINDOWS, FEAT_SUFFIXES

    cfg = load_config()
    master_dir = Path(cfg["paths"]["master"])
    out_base = Path(cfg["paths"]["features"])
    ts_col = "timestamp"

    s3 = cfg.get("stage3", {})
    ROLL_WINDOWS = s3.get("rolling_windows_min", ROLL_WINDOWS)
    QUANT_WINDOWS = s3.get("rolling_quantile_windows_min", QUANT_WINDOWS)
    QUANT_LEVELS = s3.get("rolling_quantiles", QUANT_LEVELS)
    DRIFT_WINDOWS = s3.get("drift_slope_windows_min", DRIFT_WINDOWS)
    FEAT_SUFFIXES = build_feat_suffixes()

    stage_start = time.perf_counter()
    all_stats = []
    all_idle_stats = []
    total_nodes = 0

    print("\n[features] Pass 1: compute features + collect norm stats")
    print(
        f"  rolling_windows={ROLL_WINDOWS}  quantile_windows={QUANT_WINDOWS}  "
        f"quantiles={QUANT_LEVELS}  drift_windows={DRIFT_WINDOWS}"
    )
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        comp_in = master_dir / comp
        if not comp_in.exists():
            continue

        out_dir = out_base / comp
        out_dir.mkdir(parents=True, exist_ok=True)

        if comp == "infra":
            parquets = [
                p for p in sorted(comp_in.rglob("*.parquet")) if "irc" not in p.parts
            ]
        else:
            parquets = sorted(comp_in.glob("*.parquet"))
        print(f"\n  {comp.upper()}  {len(parquets)} nodes")

        mem_lookup, oom_jobs = load_slurm_job_meta(cfg, comp)
        print(f"    SLURM meta: {len(mem_lookup):,} jobs, {len(oom_jobs):,} OOM kills")

        workers = fe_workers()
        args = [
            (p, comp, comp_in, out_dir, force, cfg, ts_col, oom_jobs) for p in parquets
        ]
        results = []
        if workers == 1 or not args:
            for a in args:
                r = fe_pass1_node(*a)
                print(r[3], flush=True)
                results.append(r)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass1_node, *a) for a in args]
                results = [fut.result() for fut in as_completed(futs)]
                for r in sorted(results, key=lambda r: r[0]):
                    print(r[3], flush=True)
        # deterministic reduce (independent of completion order)
        for _, stats, idle, _, processed in sorted(results, key=lambda r: r[0]):
            if processed:
                all_stats.append(stats)
                all_idle_stats.append(idle)
                total_nodes += 1
        gc.collect()

    if not all_stats:
        print("[features] No data processed")
        return

    norm_stats = pd.concat(all_stats, ignore_index=True)
    norm_path = out_base / "norm_stats.parquet"
    save_parquet(norm_stats, norm_path)
    print(f"\n  norm_stats saved: {len(norm_stats)} entries → {norm_path}")

    if all_idle_stats:
        idle_stats = pd.concat(all_idle_stats, ignore_index=True)
        idle_path = out_base / "norm_stats_idle.parquet"
        save_parquet(idle_stats, idle_path)
        n_with_idle = int((idle_stats["count"] > 0).sum())
        print(
            f"  norm_stats_idle saved: {len(idle_stats)} entries "
            f"({n_with_idle} with idle rows) → {idle_path}"
        )

    print("\n[features] Pass 2: apply z-score normalization (clean train stats)")
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        out_dir = out_base / comp
        if not out_dir.exists():
            continue
        comp_stats = norm_stats[norm_stats["component"] == comp]
        parquets = (
            sorted(out_dir.rglob("*.parquet"))
            if comp == "infra"
            else sorted(out_dir.glob("*.parquet"))
        )
        parquets = [p for p in parquets if "irc" not in p.parts]  # no-IRC
        workers = fe_workers()
        if workers == 1 or not parquets:
            for p in parquets:
                print(fe_pass2_node(p, comp_stats)[1], flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass2_node, p, comp_stats) for p in parquets]
                results = [fut.result() for fut in as_completed(futs)]
                for _, log in sorted(results, key=lambda r: r[0]):
                    print(log, flush=True)
        gc.collect()

    attach_pdu_pass(cfg, force=force)

    elapsed = time.perf_counter() - stage_start
    print(f"\nFeature Engineering completed. nodes={total_nodes}  time={elapsed:.1f}s")
    print(f"Norm stats: {norm_path}")


if __name__ == "__main__":
    feature_engineering(force=True)
