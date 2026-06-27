from __future__ import annotations

import json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import load_config, save_parquet, apply_node_limit

from offline.phase3.job_context_annotator_module.indexes import *
from offline.phase3.job_context_annotator_module.annotation import *


# Worker entry: load a node's features/master and annotate its physics anomalies.
def annotate_one_node(score_path_str: str):
    ctx = PHASE3_CONTEXT
    score_path = Path(score_path_str)
    hostname = score_path.stem

    scores = pd.read_parquet(score_path, engine="pyarrow")
    if scores.empty:
        return hostname, None, None

    comp = str(scores["component"].iloc[0]) if "component" in scores.columns else "zen4"
    node_cpus = ctx["node_cpus_map"].get(comp, 128)

    feat_base = ctx["feat_base"]
    master_base = ctx["master_base"]
    FEAT_COLS = ctx["FEAT_COLS"]
    SLURM_CPU_COL, SLURM_MEM_COL, IDRAC_CPU_COL = ctx["cols"]
    cv_df = ctx["cv_df"]
    host_viol_ts = ctx["cv_violated_ts_by_host"].get(hostname, set())

    feat_path = feat_base / comp / f"{hostname}.parquet"
    feat_df: Optional[pd.DataFrame] = None
    if feat_path.exists():
        try:
            feat_df_full = pd.read_parquet(feat_path, engine="pyarrow")
            pwr_extra = [
                c
                for c in feat_df_full.columns
                if (
                    "systeminputpower" in c.lower()
                    or "systempowerconsumption" in c.lower()
                )
                and c.endswith("_avg")
            ]
            keep = [c for c in FEAT_COLS if c in feat_df_full.columns] + pwr_extra[:1]
            feat_df = feat_df_full[keep] if keep else None
            del feat_df_full
        except Exception:
            feat_df = None

    idrac_cpu_lookup: dict = {}
    master_path = master_base / comp / f"{hostname}.parquet"
    if master_path.exists():
        try:
            raw_df = pd.read_parquet(
                master_path,
                columns=["timestamp", IDRAC_CPU_COL],
                engine="pyarrow",
            )
            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True)
            idrac_cpu_lookup = dict(
                zip(
                    raw_df["timestamp"],
                    pd.to_numeric(raw_df[IDRAC_CPU_COL], errors="coerce"),
                )
            )
        except Exception:
            pass

    feat_entry = None
    if feat_df is not None and TS in feat_df.columns and host_viol_ts:
        ts_arr = pd.to_datetime(feat_df[TS], utc=True)

        if "slurm_cpu_load" in feat_df.columns:
            slurm_cpu = pd.to_numeric(feat_df["slurm_cpu_load"], errors="coerce") / max(
                node_cpus, 1
            )
            slurm_cpu = slurm_cpu.clip(upper=100.0)
        else:
            slurm_cpu = pd.Series(np.nan, index=feat_df.index)
        cpu_load_arr = slurm_cpu.to_numpy(dtype="float64", copy=True)

        if idrac_cpu_lookup:
            need_fallback = np.isnan(cpu_load_arr)
            if need_fallback.any():
                fallback = ts_arr.map(idrac_cpu_lookup)
                fallback_arr = pd.to_numeric(fallback, errors="coerce").to_numpy(
                    dtype="float64"
                )
                cpu_load_arr = np.where(need_fallback, fallback_arr, cpu_load_arr)

        if "slurm_memoryusage" in feat_df.columns:
            mem_arr = pd.to_numeric(
                feat_df["slurm_memoryusage"], errors="coerce"
            ).to_numpy(dtype="float64")
        else:
            mem_arr = np.full(len(feat_df), np.nan, dtype="float64")

        ts_list = ts_arr.to_list()
        feat_entry = {
            ts_list[i]: {"cpu_load": cpu_load_arr[i], "mem_usage": mem_arr[i]}
            for i in range(len(ts_list))
            if ts_list[i] in host_viol_ts
        }

    constr_node = (
        cv_df[cv_df["hostname"] == hostname]
        if cv_df is not None and not cv_df.empty and "hostname" in cv_df.columns
        else pd.DataFrame()
    )

    anom_df = annotate_node(
        hostname=hostname,
        component=comp,
        scores_df=scores,
        constr_df=constr_node,
        seg_idx=ctx["seg_idx"],
        coh_idx=ctx["coh_idx"],
        feat_df=feat_df,
        node_cpus=node_cpus,
        cluster_power_lookup=ctx["cluster_power_lookup"],
        idrac_cpu_lookup=idrac_cpu_lookup,
        slurm_job_lookup=ctx["slurm_job_lookup"],
    )
    anom_out = anom_df if anom_df is not None and not anom_df.empty else None
    return hostname, anom_out, feat_entry


# Label each physics anomaly with its job context (fault vs job-explained) across all nodes.
def run_job_context_annotator(force: bool = False) -> None:
    cfg = load_config()
    models_dir = Path(cfg["phase3"]["output_dir"])
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "offline/data/phase2"))

    physics_dir = models_dir / "physics"
    scores_dir = physics_dir / "residual_scores"
    cv_path = physics_dir / "constraint_violations.parquet"
    out_anom = physics_dir / "job_context_anomalies.parquet"
    out_cv = physics_dir / "constraint_violations_annotated.parquet"
    rep_dir = Path(cfg["phase3"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[job_ctx] Physics dir : {physics_dir}")
    print(f"          Phase 2 dir : {phase2_dir}")

    if out_anom.exists() and out_cv.exists() and not force:
        print("[job_ctx] Outputs exist — skipping (use --force to recompute)")
        return

    if not scores_dir.exists():
        print("[job_ctx] residual_scores/ not found — run physics_models first")
        return

    t0 = time.perf_counter()

    seg_path = phase2_dir / "job_segments.parquet"
    if seg_path.exists():
        segs = pd.read_parquet(seg_path, engine="pyarrow")
        segs["seg_start"] = pd.to_datetime(segs["seg_start"], utc=True)
        segs["seg_end"] = pd.to_datetime(segs["seg_end"], utc=True)
        print(f"  Segments: {len(segs):,} loaded from {seg_path}")
    else:
        print(f"  [WARN] {seg_path} not found — active_job_ids will be empty")
        segs = pd.DataFrame()

    seg_idx = build_segment_index(segs) if not segs.empty else {}

    coh_path = phase2_dir / "multi_node_coherence.parquet"
    if coh_path.exists():
        coh_df = pd.read_parquet(coh_path, engine="pyarrow")
        print(f"  Coherence: {len(coh_df):,} node-job pairs loaded")
    else:
        coh_df = None
        print("  [WARN] multi_node_coherence.parquet not found")
    coh_idx = build_coherence_index(coh_df)

    cluster_power_lookup = load_cluster_power_lookup(phase2_dir)
    if cluster_power_lookup:
        print(
            f"  Cluster power lookup: {len(cluster_power_lookup):,} (hostname, job_id) entries"
        )
    else:
        print(
            "  [WARN] job_profiles_clustered.parquet not found — JOB_OVER_EXPECTATION disabled"
        )

    slurm_job_lookup = load_slurm_job_lookup(cfg)
    print(f"  SLURM job lookup: {len(slurm_job_lookup):,} job entries")

    if cv_path.exists():
        cv_df = pd.read_parquet(cv_path, engine="pyarrow")
        print(f"  Constraint violations: {len(cv_df):,} rows loaded")
    else:
        cv_df = pd.DataFrame()
        print("  [WARN] constraint_violations.parquet not found")

    node_cpus_map: dict = {}
    for comp_cfg in cfg.get("components", []):
        node_cpus_map[comp_cfg["name"]] = int(comp_cfg.get("cpu_count", 128))

    feat_base = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))
    if not feat_base.exists():
        feat_base = Path(cfg["paths"]["features"])

    master_base = Path(cfg["paths"].get("master", "offline/data/master"))

    SLURM_CPU_COL = "slurm_cpu_load"
    IDRAC_CPU_COL = "cpuusage__systemusage_avg"
    SLURM_MEM_COL = "slurm_memoryusage"
    FEAT_COLS = ["timestamp", SLURM_CPU_COL, SLURM_MEM_COL]

    cv_violated_ts_by_host: dict = {}
    if (
        not cv_df.empty
        and "constraint_flags" in cv_df.columns
        and "hostname" in cv_df.columns
    ):
        viol = cv_df[cv_df["constraint_flags"].fillna("").astype(str).ne("")]
        if not viol.empty:
            viol = viol.assign(__ts=pd.to_datetime(viol[TS], utc=True))
            for h, grp in viol.groupby("hostname"):
                cv_violated_ts_by_host[str(h)] = set(grp["__ts"])

    print(f"\n[job_ctx] Annotating physics anomalies ...")
    score_paths = apply_node_limit(sorted(scores_dir.glob("*.parquet")))
    workers = job_context_workers()

    PHASE3_CONTEXT.clear()
    PHASE3_CONTEXT.update(
        node_cpus_map=node_cpus_map,
        feat_base=feat_base,
        master_base=master_base,
        FEAT_COLS=FEAT_COLS,
        cols=(SLURM_CPU_COL, SLURM_MEM_COL, IDRAC_CPU_COL),
        cv_df=cv_df,
        cv_violated_ts_by_host=cv_violated_ts_by_host,
        seg_idx=seg_idx,
        coh_idx=coh_idx,
        cluster_power_lookup=cluster_power_lookup,
        slurm_job_lookup=slurm_job_lookup,
    )

    if workers == 1 or len(score_paths) <= 1:
        results = [annotate_one_node(str(p)) for p in score_paths]
    else:
        print(f"  [job_ctx] per-node pool: workers={workers}  nodes={len(score_paths)}")
        with ProcessPoolExecutor(max_workers=min(workers, len(score_paths))) as ex:
            results = list(ex.map(annotate_one_node, [str(p) for p in score_paths]))

    # Deterministic reduce: order by hostname (== original sorted-glob order).
    all_anom_records = []
    feat_lookup_by_node: dict = {}
    for hostname, anom_df, feat_entry in sorted(results, key=lambda r: r[0]):
        if feat_entry:
            feat_lookup_by_node[hostname] = feat_entry
        if anom_df is not None and not anom_df.empty:
            all_anom_records.append(anom_df)

    if all_anom_records:
        annotated = pd.concat(all_anom_records, ignore_index=True)
        annotated["power_residual_z"] = annotated["power_residual_z"].astype("float32")
        annotated["thermal_residual_z"] = annotated["thermal_residual_z"].astype(
            "float32"
        )
        annotated["cpu_load"] = annotated["cpu_load"].astype("float32")
        annotated["mem_usage_pct"] = annotated["mem_usage_pct"].astype("float32")
        annotated["req_cpu_frac"] = annotated["req_cpu_frac"].astype("float32")
        annotated["expected_power_w"] = annotated["expected_power_w"].astype("float32")
        annotated["power_excess_pct"] = annotated["power_excess_pct"].astype("float32")
        annotated["dominant_job_id"] = pd.array(
            annotated["dominant_job_id"], dtype="Int64"
        )
        annotated["req_memory_mb"] = annotated["req_memory_mb"].astype("float32")
        annotated["job_state"] = annotated["job_state"].astype("string")
        annotated["exit_code"] = annotated["exit_code"].astype("string")
        save_parquet(annotated, out_anom)
        print(f"  Saved job_context_anomalies: {len(annotated):,} rows → {out_anom}")
    else:
        print("  [WARN] No anomalous rows found to annotate.")
        annotated = pd.DataFrame()

    if not cv_df.empty:
        default_cpus = node_cpus_map.get(
            "zen4", node_cpus_map.get(next(iter(node_cpus_map), "zen4"), 128)
        )
        cv_annotated = annotate_constraint_violations(
            cv_df=cv_df,
            seg_idx=seg_idx,
            coh_idx=coh_idx,
            feat_lookup_by_node=feat_lookup_by_node,
            node_cpus_map=node_cpus_map,
            node_cpus=default_cpus,
            slurm_job_lookup=slurm_job_lookup,
        )
        save_parquet(cv_annotated, out_cv)
        print(
            f"  Saved constraint_violations_annotated: {len(cv_annotated):,} rows → {out_cv}"
        )
    else:
        cv_annotated = pd.DataFrame()

    elapsed = time.perf_counter() - t0
    summary: dict = {"elapsed_s": round(elapsed, 1)}

    if not annotated.empty:
        ctx_counts = annotated["anomaly_context"].value_counts().to_dict()
        n_total = len(annotated)
        summary["anomaly_context_counts"] = ctx_counts
        summary["anomaly_context_pct"] = {
            k: round(100 * v / n_total, 2) for k, v in ctx_counts.items()
        }
        summary["n_physics_anomaly_rows"] = n_total
        summary["n_job_explained"] = int(ctx_counts.get("JOB_EXPLAINED", 0))
        summary["n_job_over_expectation"] = int(
            ctx_counts.get("JOB_OVER_EXPECTATION", 0)
        )
        summary["n_infra_fault"] = int(ctx_counts.get("INFRA_FAULT", 0))
        summary["n_cooling_fault"] = int(ctx_counts.get("COOLING_FAULT", 0))
        summary["n_no_jobs"] = int(ctx_counts.get("NO_JOBS", 0))
        summary["n_isolated"] = int((annotated["node_isolation"] == "ISOLATED").sum())
        summary["n_cluster_wide"] = int(
            (annotated["node_isolation"] == "CLUSTER_WIDE").sum()
        )
        fault_nodes = (
            annotated[
                annotated["anomaly_context"].isin(
                    ["INFRA_FAULT", "COOLING_FAULT", "NO_JOBS", "JOB_OVER_EXPECTATION"]
                )
            ]
            .groupby("hostname")["physics_anomaly"]
            .count()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )
        summary["top_fault_nodes"] = fault_nodes

    if not cv_annotated.empty and "anomaly_context" in cv_annotated.columns:
        cv_ctx = cv_annotated["anomaly_context"].value_counts().to_dict()
        summary["constraint_context_counts"] = cv_ctx

    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if not annotated.empty:
        enrich_attributed_alerts(annotated, phase2_dir)

    print(f"\n[job_ctx] Done in {elapsed:.1f}s")
    if not annotated.empty:
        print(f"\n  Context breakdown ({len(annotated):,} anomalous rows):")
        for ctx, count in sorted(ctx_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(annotated)
            print(f"    {ctx:30s}: {count:>6,}  ({pct:.1f}%)")
        print(f"\n  Top fault nodes:")
        for node, cnt in list(fault_nodes.items())[:5]:
            print(f"    {node:22s}: {cnt:>5,} anomalous minutes")
    print(f"  Report: {rep_dir}/summary.json")


if __name__ == "__main__":
    run_job_context_annotator(force=True)
