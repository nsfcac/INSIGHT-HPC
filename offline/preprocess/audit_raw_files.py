from __future__ import annotations

import gc, time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from shared.utils.io_utils import (
    load_config,
    load_public_tables,
    save_parquet,
    build_lookup_dicts,
)
from offline.preprocess.audit_raw_files_module.checks import *


# Audit every raw metric parquet, flagging duplicates/spikes/flatlines and writing summaries.
def audit_raw_files(force: bool = False, verbose: bool = True) -> None:
    cfg = load_config()
    raw_base = Path(cfg["paths"]["raw_parquet"])
    out_base = Path(cfg["paths"]["audited"])
    dynamic_thresholds = load_thresholds(cfg)
    if dynamic_thresholds is not None:
        print(f"Loaded {len(dynamic_thresholds)} dynamic thresholds for audit.")

    stage_start = time.perf_counter()
    total_files = total_rows = 0

    audit_stats: dict = {}

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        print(f"\nComponent: {comp.upper()}")

        audit_stats.setdefault(comp, {})

        pub_tables = load_public_tables(comp, raw_base)
        print(f"  Public tables loaded: {list(pub_tables.keys())}")

        # Decode the threshold index with this component's fqdd/source maps so per-sensor thresholds match the decoded _avg column names.
        if dynamic_thresholds is not None:
            _, fqdd_map, source_map = build_lookup_dicts(pub_tables)
            comp_thresholds = make_sensor_thresholds(
                dynamic_thresholds, fqdd_map, source_map
            )
            if verbose and comp_thresholds is not None:
                print(f"  Decoded {len(comp_thresholds)} threshold entries for {comp}")
        else:
            comp_thresholds = None

        for source in ["idrac", "slurm", "pdu"]:
            source_dir = raw_base / comp / source
            if not source_dir.exists():
                continue

            source_cfg = cfg.get("sources", {}).get(source, {})
            timeseries_only = source_cfg.get("timeseries_tables", None)

            all_parquets = sorted(source_dir.glob("*.parquet"))
            if timeseries_only is not None:
                parquets = [
                    p
                    for p in all_parquets
                    if p.stem.lower() in [t.lower() for t in timeseries_only]
                ]
                skipped = [p.stem for p in all_parquets if p not in parquets]
                if skipped:
                    print(f"  [{source.upper()}] skipping non-timeseries: {skipped}")
            else:
                parquets = all_parquets

            if not parquets:
                continue

            print(f"\n  [{source.upper()}]  {len(parquets)} files")
            (out_base / comp / source).mkdir(parents=True, exist_ok=True)

            audit_stats[comp].setdefault(source, {})

            source_rows = 0
            source_start = time.perf_counter()

            # Per-file audit is independent (workers=1 is serial and identical). Giant idrac files can be split into per-node-group chunks via INSIGHT_HPC_AUDIT_SPLIT so one huge file isn't the wall-time floor.
            workers = audit_workers()
            split = audit_split()
            node_ids_all = (
                sorted(build_lookup_dicts(pub_tables)[0].keys()) if split > 1 else []
            )
            if split > 1 and source == "idrac":
                print(
                    f"  [{source.upper()}] workers={workers} split={split} "
                    f"node_groups={min(split, len(node_ids_all))}",
                    flush=True,
                )

            whole_tasks, chunk_tasks, merge_specs = [], [], []
            for p in parquets:
                metric = p.stem.lower()
                out_path = out_base / comp / source / f"{metric}.parquet"
                do_split = (
                    split > 1
                    and source == "idrac"
                    and (force or not out_path.exists())
                    and len(node_ids_all) >= split
                    and parquet_num_rows(p) >= AUDIT_SPLIT_MIN_ROWS
                )
                if do_split:
                    partials = []
                    for gi in range(split):
                        g = node_ids_all[gi::split]  # round-robin balances nodes
                        if not g:
                            continue
                        pp = out_base / comp / source / f".{metric}.part{gi}.parquet"
                        partials.append(pp)
                        chunk_tasks.append(
                            (
                                p,
                                comp,
                                source,
                                source_dir,
                                pp,
                                cfg,
                                pub_tables,
                                comp_thresholds,
                                set(g),
                            )
                        )
                    merge_specs.append((metric, out_path, partials))
                else:
                    whole_tasks.append(
                        (
                            p,
                            comp,
                            source,
                            source_dir,
                            out_path,
                            force,
                            cfg,
                            pub_tables,
                            comp_thresholds,
                            verbose,
                        )
                    )

            results, chunk_done = [], {}
            n_tasks = len(whole_tasks) + len(chunk_tasks)
            defer_whole_logs = False
            if workers == 1 or n_tasks <= 1:
                for t in whole_tasks:
                    r = audit_one_file(*t)
                    print(r[3], flush=True)
                    results.append(r)
                for t in chunk_tasks:
                    pp, n, log = audit_chunk(*t)
                    if log.strip():
                        print(log.rstrip(), flush=True)
                    chunk_done[pp] = n
            else:
                defer_whole_logs = True
                result_order = {t[0].stem.lower(): i for i, t in enumerate(whole_tasks)}
                with ProcessPoolExecutor(max_workers=min(workers, n_tasks)) as ex:
                    fkind = {}
                    for t in whole_tasks:
                        fkind[ex.submit(audit_one_file, *t)] = "whole"
                    for t in chunk_tasks:
                        fkind[ex.submit(audit_chunk, *t)] = "chunk"

                    pending = set(fkind)
                    done_count = 0
                    heartbeat = log_heartbeat_seconds()
                    while pending:
                        if heartbeat > 0:
                            done, pending = wait(
                                pending,
                                timeout=heartbeat,
                                return_when=FIRST_COMPLETED,
                            )
                            if not done:
                                elapsed = time.perf_counter() - source_start
                                print(
                                    f"    ... audit running  done={done_count}/{n_tasks}  "
                                    f"pending={len(pending)}  elapsed={elapsed:.0f}s",
                                    flush=True,
                                )
                                continue
                        else:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)

                        for fut in done:
                            done_count += 1
                            if fkind[fut] == "whole":
                                results.append(fut.result())
                            else:
                                pp, n, worker_log = fut.result()
                                chunk_done[pp] = n

                for r in sorted(results, key=lambda x: result_order.get(x[0], 10**9)):
                    print(r[3], flush=True)

            for metric, stats, processed, _ in results:
                if stats:
                    audit_stats[comp][source][metric] = stats
                    source_rows += stats["rows"]
                    if processed:
                        total_rows += stats["rows"]
                        total_files += 1

            # Merge split files: concat partials -> final. Absent-sensor rows get audit_flags 0 (matches whole-file semantics); value cols stay NaN.
            for metric, final_out, partials in merge_specs:
                frames = [
                    pd.read_parquet(pp, engine="pyarrow")
                    for pp in partials
                    if pp.exists()
                ]
                frames = [f for f in frames if not f.empty]
                if not frames:
                    print(f"    [WARN] {metric}: no chunk output", flush=True)
                    continue
                full = pd.concat(frames, ignore_index=True)
                for c in [c for c in full.columns if c.startswith("audit_flags__")]:
                    full[c] = full[c].fillna(0).astype(np.int8)
                save_parquet(full, final_out)
                stats = extract_metric_stats(full)
                if stats:
                    audit_stats[comp][source][metric] = stats
                    source_rows += stats["rows"]
                    total_rows += stats["rows"]
                    total_files += 1
                n_rows = len(full)
                del full, frames
                gc.collect()
                for pp in partials:
                    try:
                        pp.unlink()
                    except OSError:
                        pass
                print(
                    f"    DONE {metric:30s}  {n_rows:>12,} rows  "
                    f"(merged {len(partials)} chunks)",
                    flush=True,
                )
            gc.collect()

            source_elapsed = time.perf_counter() - source_start
            print(
                f"\n  [{source.upper()} done]  "
                f"rows={source_rows:,}  time={source_elapsed:.1f}s"
            )

    total_elapsed = time.perf_counter() - stage_start
    print("\nAudit complete")
    print(
        f" - files={total_files}  rows={total_rows:,}  "
        f"time={total_elapsed:.1f}s  "
        f"({total_rows/max(total_elapsed,0.001)/1e3:.0f}K rows/s overall)"
    )

    write_audit_summary(audit_stats, cfg)


if __name__ == "__main__":
    audit_raw_files(force=True)
