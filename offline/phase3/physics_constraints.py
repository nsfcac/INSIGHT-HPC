from __future__ import annotations

import os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from shared.utils.io_utils import load_config, save_parquet, apply_node_limit

from offline.phase3.physics_constraints_module.node_checks import *
from offline.phase3.physics_constraints_module.rack_jobs import *


# Run node-level and rack-level physics constraint checks and write violations.
def run_physics_constraints(force: bool = False) -> None:
    cfg = load_config()
    feat_aligned = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))
    feat_dir = feat_aligned if feat_aligned.exists() else Path(cfg["paths"]["features"])
    models_dir = Path(cfg["phase3"]["output_dir"])

    master_dir = Path(cfg["paths"]["master"])
    out_dir = models_dir / "physics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constraint_violations.parquet"

    p2_base = Path(cfg.get("phase2", {}).get("output_dir", "offline/data/phase2"))
    seg_path = p2_base / "job_segments.parquet"
    seg_idx = load_segments_index(seg_path)
    if seg_idx:
        print(f"\n[constraints] Loaded job segments for {len(seg_idx)} nodes")
    else:
        print("\n[constraints] No job segments found — active_job_ids will be empty")

    print(f"\n[constraints] Output: {out_dir}")

    if out_path.exists() and not force:
        print("[constraints] constraint_violations.parquet exists — loading")
        return

    constraints_cfg = cfg.get("phase3", {}).get("constraints", {}) or {}
    c1_cfg = constraints_cfg.get("c1_temp_fan", {}) or {}
    c2_cfg = constraints_cfg.get("c2_rack_thermal", {}) or {}
    c3_cfg = constraints_cfg.get("c3_dynamics", {}) or {}
    c4_cfg = constraints_cfg.get("c4_crossplane", {}) or {}
    # Allowed kwargs per constraint function (match keyword names).
    c1_params = {
        k: v
        for k, v in c1_cfg.items()
        if k in {"temp_rise_rate", "fan_slack", "window_min", "enabled"}
    }
    c2_params = {
        k: v
        for k, v in c2_cfg.items()
        if k
        in {"min_node_frac", "pdu_stable_pct", "temp_rise_min", "window_min", "enabled"}
    }
    c3_params = {
        k: v
        for k, v in c3_cfg.items()
        if k
        in {
            "power_ramp_pct",
            "lag_window_min",
            "min_temp_rise",
            "temp_rise_per_min",
            "power_stable_min",
            "power_stable_w",
            "min_power_rise_w",
            "min_temp_rise_abs",
        }
    }
    c4_params = {
        k: v
        for k, v in c4_cfg.items()
        if k
        in {
            "pdu_rise_pct",
            "node_stable",
            "window_min",
            "min_node_frac",
            "min_pdu_rise_w",
            "min_mismatch_w",
        }
    }
    print(f"[constraints] c1 params: {c1_params}")
    print(f"[constraints] c2 params: {c2_params}")
    print(f"[constraints] c3 params: {c3_params}")
    print(f"[constraints] c4 params: {c4_params}")

    t0 = time.perf_counter()
    print(f"\n[constraints] Running physics constraint checks")

    env_workers = os.environ.get("INSIGHT_HPC_PHASE3_WORKERS")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        available_cpus = int(slurm_cpus)
    elif hasattr(os, "sched_getaffinity"):
        available_cpus = len(os.sched_getaffinity(0))
    else:
        available_cpus = os.cpu_count() or 4
    cpu_budget = max(1, available_cpus - 2)

    all_results = []

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = feat_dir / comp
        if not comp_dir.exists():
            continue

        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))
        n_workers = (
            int(env_workers) if env_workers else max(1, min(len(parquets), cpu_budget))
        )
        print(
            f"\n  [{comp.upper()}]  {len(parquets)} nodes  "
            f"(parallel: {n_workers} workers)"
        )

        rack_groups: dict = defaultdict(dict)  # {rack_id: {hostname: slim_df}}
        node_results: dict = {}
        print_buffer: list = []
        total_viol = 0

        if n_workers <= 1 or len(parquets) <= 1:
            # Fallback: in-process (useful for single-node test or debugging).
            worker_init(seg_idx, comp, c1_params, c3_params)
            for p in parquets:
                hostname, result, slim_df, rid = worker_process_node(str(p))
                if result is not None:
                    node_results[hostname] = result
                    n_viol = int(result["n_constraints_violated"].gt(0).sum())
                    total_viol += n_viol
                    if n_viol > 0:
                        print_buffer.append((hostname, n_viol, len(result)))
                if slim_df is not None and rid is not None:
                    rack_groups[rid][hostname] = slim_df
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=worker_init,
                initargs=(seg_idx, comp, c1_params, c3_params),
            ) as ex:
                futures = [ex.submit(worker_process_node, str(p)) for p in parquets]
                for fut in as_completed(futures):
                    hostname, result, slim_df, rid = fut.result()
                    if result is not None:
                        node_results[hostname] = result
                        n_viol = int(result["n_constraints_violated"].gt(0).sum())
                        total_viol += n_viol
                        if n_viol > 0:
                            print_buffer.append((hostname, n_viol, len(result)))
                    if slim_df is not None and rid is not None:
                        rack_groups[rid][hostname] = slim_df

        for hostname, n_viol, n_rows in sorted(print_buffer):
            print(
                f"    {hostname:20s}: {n_viol:>5,} violations "
                f"({100*n_viol/max(n_rows,1):.2f}%)"
            )

        # Run rack-level constraints C2 (thermal mismatch) and C4 (cross-plane).
        pdu_master_dir = master_dir / "infra" / "pdu"
        pdu_start_time = time.perf_counter()
        pdu_by_rack = preload_pdu_by_rack(pdu_master_dir)
        print(
            f"  [{comp.upper()}] PDU preload: {len(pdu_by_rack)} racks in "
            f"{time.perf_counter()-pdu_start_time:.1f}s"
        )

        for rid, nodes in rack_groups.items():
            rack_dfs = {h: df for h, df in nodes.items() if df is not None}
            if not rack_dfs:
                continue
            rack_start_time = time.perf_counter()

            # pdu_feeders: {pdu_hostname: per-feeder series}, or None if no PDU data.
            pdu_feeders, pdu_ts_by_hostname = pdu_by_rack.get(rid, (None, {}))

            if pdu_feeders:
                pdu_sum = pd.concat(list(pdu_feeders.values())).sort_index()
                pdu_sum = pdu_sum.groupby(level=0).sum()
            else:
                pdu_sum = None

            c2_start_time = time.perf_counter()
            c2_enabled = bool(c2_params.get("enabled", False))
            c2_runtime_params = {
                k: v for k, v in (c2_params or {}).items() if k != "enabled"
            }
            if c2_enabled:
                rack_c2 = check_rack_thermal_mismatch(
                    rack_dfs,
                    pdu_sum,
                    rid,
                    seg_idx=seg_idx,
                    **c2_runtime_params,
                )
            else:
                # Build an empty Series at rack timeline for downstream alignment.
                ts_idx = pdu_sum.index if pdu_sum is not None else pd.DatetimeIndex([])
                rack_c2 = pd.Series(False, index=ts_idx, dtype=bool)
            c2_check_seconds = time.perf_counter() - c2_start_time

            c2_writeback_start_time = time.perf_counter()
            if rack_c2.any():
                n_c2 = int(rack_c2.sum())
                print(f"    rack-{rid:02d}: RACK_THERMAL_MISMATCH = {n_c2} minutes")
                for hostname in rack_dfs.keys():
                    if hostname in node_results:
                        aligned = align_rack_to_node(
                            rack_c2, node_results[hostname][TS]
                        )
                        node_results[hostname]["const2_rack_therm"] = aligned
                        node_results[hostname][
                            "n_constraints_violated"
                        ] += aligned.astype(np.int8)
            c2_writeback_seconds = time.perf_counter() - c2_writeback_start_time

            c4_start_time = time.perf_counter()
            rack_c4 = pd.Series(dtype=bool)
            per_feeder_c4: dict = {}
            if pdu_feeders:
                for pdu_hostname, feeder_series in pdu_feeders.items():
                    c4_feeder_flags = check_rack_cross_plane_disagreement(
                        rack_dfs,
                        feeder_series,
                        rid,
                        **c4_params,
                    )
                    if c4_feeder_flags is None or c4_feeder_flags.empty:
                        continue
                    per_feeder_c4[pdu_hostname] = c4_feeder_flags
                    if rack_c4.empty:
                        rack_c4 = c4_feeder_flags.copy()
                    else:
                        # Align on union of ts, OR
                        rack_c4 = rack_c4.reindex(
                            rack_c4.index.union(c4_feeder_flags.index), fill_value=False
                        )
                        aligned_c4_feeder_flags = c4_feeder_flags.reindex(
                            rack_c4.index, fill_value=False
                        )
                        rack_c4 = rack_c4 | aligned_c4_feeder_flags
            else:
                # No PDU data — produce an empty rack_c4 for back-compat.
                rack_c4 = check_rack_cross_plane_disagreement(
                    rack_dfs,
                    None,
                    rid,
                    **c4_params,
                )
            c4_check_seconds = time.perf_counter() - c4_start_time

            c4_writeback_start_time = time.perf_counter()
            if not rack_c4.empty and rack_c4.any():
                n_c4 = int(rack_c4.sum())
                feeders_fired_count = sum(1 for s in per_feeder_c4.values() if s.any())
                print(
                    f"    rack-{rid:02d}: CROSSPLANE = {n_c4} minutes "
                    f"(across {feeders_fired_count} feeder(s))"
                )
                for hostname in rack_dfs.keys():
                    if hostname in node_results:
                        aligned = align_rack_to_node(
                            rack_c4, node_results[hostname][TS]
                        )
                        existing = node_results[hostname]["const4_crossplane"].to_numpy(
                            dtype=bool
                        )
                        node_results[hostname]["const4_crossplane"] = existing | aligned
                        node_results[hostname][
                            "n_constraints_violated"
                        ] += aligned.astype(np.int8)

                for pdu_hostname, c4_feeder_flags_for_pdu in per_feeder_c4.items():
                    if not c4_feeder_flags_for_pdu.any():
                        continue
                    pdu_timestamps = pdu_ts_by_hostname.get(pdu_hostname)
                    if pdu_timestamps is None:
                        continue
                    c4_true_flags = c4_feeder_flags_for_pdu[c4_feeder_flags_for_pdu]
                    c4_merge_source = build_merge_src(c4_true_flags, "_rc4")
                    pdu_target_frame = pd.DataFrame(
                        {TS: to_ns(pdu_timestamps)}
                    ).sort_values(TS)
                    c4_pdu_flags = (
                        pd.merge_asof(
                            pdu_target_frame,
                            c4_merge_source,
                            on=TS,
                            direction="nearest",
                            tolerance=pd.Timedelta(minutes=2),
                        )["_rc4"]
                        .fillna(False)
                        .astype(bool)
                    )
                    if c4_pdu_flags.any():
                        all_results.append(
                            pd.DataFrame(
                                {
                                    TS: pdu_timestamps.values,
                                    "hostname": pdu_hostname,
                                    "component": "infra",
                                    "const1_temp_fan": False,
                                    "const2_rack_therm": False,
                                    "const3_dynamics": False,
                                    "const4_crossplane": c4_pdu_flags.values,
                                    "const5_alloc_idle": False,
                                    "n_constraints_violated": c4_pdu_flags.astype(
                                        "int8"
                                    ).values,
                                    "constraint_flags": np.where(
                                        c4_pdu_flags.values, "CROSSPLANE", ""
                                    ),
                                    "active_job_ids": "",
                                }
                            )
                        )
            c4_writeback_seconds = time.perf_counter() - c4_writeback_start_time

            rack_total_seconds = time.perf_counter() - rack_start_time
            print(
                f"    rack-{rid:02d} [timing]: c2={c2_check_seconds:.1f}s "
                f"c2_wb={c2_writeback_seconds:.1f}s c4={c4_check_seconds:.1f}s "
                f"c4_wb={c4_writeback_seconds:.1f}s total={rack_total_seconds:.1f}s "
                f"(nodes={len(rack_dfs)})"
            )

        # Apply persistence filter — parallelized across nodes.
        persist_min = int(
            cfg.get("phase3", {}).get("constraint_min_persistence_min", 3)
        )
        if persist_min > 1 and node_results:
            apply_persistence_parallel(node_results, persist_min, n_workers)

        all_results.extend(node_results.values())
        print(f"  [{comp.upper()}] total violations: {total_viol:,}")

    if not all_results:
        print("[constraints] No results produced.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    save_parquet(combined, out_path)

    elapsed = time.perf_counter() - t0
    total_viol = int(combined["n_constraints_violated"].gt(0).sum())
    print(f"\n[constraints] Done in {elapsed:.1f}s")
    print(
        f"  Total rows with violations: {total_viol:,} "
        f"({100*total_viol/max(len(combined),1):.2f}%)"
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    run_physics_constraints(force=True)
