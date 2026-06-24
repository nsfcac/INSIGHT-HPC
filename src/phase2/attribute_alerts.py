from __future__ import annotations

import json, math, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet


from src.phase2.attribute_alerts_module.attribution import *


# Attribute each multi-job anomalous segment to its most-likely culprit jobs.
def attribute_multi_job_segments(
    scores: pd.DataFrame, master_dir: Path, cfg: dict, phase2_dir: Path
) -> pd.DataFrame:
    multi_anom = scores[
        scores["job_if_anomaly"].fillna(False) & scores["is_multi_job"].fillna(False)
    ]
    if multi_anom.empty:
        return pd.DataFrame()

    scores_path = phase2_dir / "job_anomaly_scores.parquet"
    profiles_path = phase2_dir / "job_profiles_clustered.parquet"
    coherence_path = phase2_dir / "multi_node_coherence.parquet"

    job_scores = (
        pd.read_parquet(scores_path, engine="pyarrow")
        if scores_path.exists()
        else pd.DataFrame()
    )
    job_profiles = (
        pd.read_parquet(profiles_path, engine="pyarrow")
        if profiles_path.exists()
        else pd.DataFrame()
    )
    coherence_df = (
        pd.read_parquet(coherence_path, engine="pyarrow")
        if coherence_path.exists()
        else pd.DataFrame()
    )

    for ts_col in ("seg_start", "seg_end"):
        if ts_col in job_scores.columns:
            job_scores[ts_col] = pd.to_datetime(job_scores[ts_col], utc=True)

    records = []
    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_segs = multi_anom[multi_anom["component"] == comp]
        if comp_segs.empty:
            continue

        for hostname, node_segs in comp_segs.groupby("hostname"):
            master_path = master_dir / comp / f"{hostname}.parquet"
            if not master_path.exists():
                continue

            master = load_parquet(master_path)
            if master is None or "cpu_shares_json" not in master.columns:
                continue
            master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)

            for _, seg in node_segs.iterrows():
                seg_start = pd.Timestamp(seg["seg_start"])
                seg_end = pd.Timestamp(seg["seg_end"])
                window = master[
                    (master["timestamp"] >= seg_start)
                    & (master["timestamp"] <= seg_end)
                ]
                if window.empty:
                    continue

                result = attribute_concurrent_jobs(
                    window=window,
                    anomaly_onset=seg_start,
                    hostname=str(hostname),
                    component=comp,
                    job_scores=job_scores,
                    job_profiles=job_profiles,
                    coherence_df=coherence_df,
                )

                if result is None:
                    continue

                attr_dict = attribution_to_dict(result)
                top_suspects = [
                    f"{c['job_id']}({c['attribution_score']:.2f})"
                    for c in attr_dict.get("candidates", [])
                    if float(c.get("attribution_score", 0)) > 0.2
                ][:3]
                records.append(
                    {
                        "job_id": result.primary_job_id,
                        "component": comp,
                        "alert_type": "multi_job_attributed",
                        "first_flag_time": seg_start,
                        "last_flag_time": seg_end,
                        "duration_flagged_min": float(seg["duration_min"]),
                        "max_anomaly_score": float(seg["job_anomaly_score"]),
                        "mean_anomaly_score": float(seg["job_anomaly_score"]),
                        "anomalous_nodes": hostname,
                        "clean_nodes": "",
                        "n_anomalous_nodes": 1,
                        "n_clean_nodes": 0,
                        "split": seg["split"],
                        "attribution_confidence": float(result.confidence),
                        "attribution_signals_used": ",".join(result.signals_used),
                        "top_suspect_jobs": "|".join(top_suspects),
                        "attribution_candidates_json": json.dumps(
                            attr_dict, default=str
                        ),
                    }
                )

    return pd.DataFrame(records) if records else pd.DataFrame()


# Collect and attribute phase-2 anomaly alerts, then write the ranked alert table.
def run_attribute_alerts(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "data/phase2"))
    master_dir = Path(cfg["paths"]["master"])
    scores_path = phase2_dir / "job_anomaly_scores.parquet"
    out_path = phase2_dir / "attributed_alerts.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[attribute] attributed_alerts.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    if not scores_path.exists():
        raise FileNotFoundError("Run train_cluster_models first.")

    scores = pd.read_parquet(scores_path, engine="pyarrow")
    scores["seg_start"] = pd.to_datetime(scores["seg_start"], utc=True)
    scores["seg_end"] = pd.to_datetime(scores["seg_end"], utc=True)

    t0 = time.perf_counter()
    print(
        f"\n[attribute] Attributing alerts from "
        f"{int(scores['job_if_anomaly'].sum()):,} anomalous segments ..."
    )

    main_alerts = collect_multi_node(scores)
    print(
        f"  Path A/B: {len(main_alerts):,} job-level alerts "
        f"({int((main_alerts['alert_type']=='multi_node').sum()) if not main_alerts.empty else 0} multi-node)"
    )

    multi_job_alerts = attribute_multi_job_segments(scores, master_dir, cfg, phase2_dir)
    print(
        f"  Path C: {len(multi_job_alerts):,} multi-job attributed segments"
        f"  (multi-signal confidence scoring)"
    )

    frames = [f for f in [main_alerts, multi_job_alerts] if not f.empty]
    if not frames:
        print("[attribute] No alerts to attribute.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")
    result["max_anomaly_score"] = result["max_anomaly_score"].astype("float32")
    result["mean_anomaly_score"] = result["mean_anomaly_score"].astype("float32")
    result["duration_flagged_min"] = result["duration_flagged_min"].astype("float32")
    result["n_anomalous_nodes"] = result["n_anomalous_nodes"].astype("int16")
    result["n_clean_nodes"] = result["n_clean_nodes"].astype("int16")

    if "attribution_confidence" not in result.columns:
        result["attribution_confidence"] = float("nan")
        result["attribution_signals_used"] = ""
        result["top_suspect_jobs"] = ""
        result["attribution_candidates_json"] = ""
    elif "top_suspect_jobs" not in result.columns:
        result["top_suspect_jobs"] = ""
    result["attribution_confidence"] = pd.to_numeric(
        result["attribution_confidence"], errors="coerce"
    ).astype("float32")

    type_rank = {"multi_node": 0, "multi_job_attributed": 1, "single_node": 2}
    result["_rank"] = result["alert_type"].map(type_rank).fillna(3)
    result = (
        result.sort_values(["_rank", "max_anomaly_score"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    save_parquet(result, out_path)

    attr_rows = result[result["alert_type"] == "multi_job_attributed"]
    summary = {
        "total_alerts": len(result),
        "multi_node_alerts": int((result["alert_type"] == "multi_node").sum()),
        "single_node_alerts": int((result["alert_type"] == "single_node").sum()),
        "multi_job_attributed": int(len(attr_rows)),
        "unique_jobs": int(result["job_id"].nunique()),
        "attribution_confidence_p50": (
            float(attr_rows["attribution_confidence"].median())
            if not attr_rows.empty
            else None
        ),
        "attribution_confidence_p90": (
            float(attr_rows["attribution_confidence"].quantile(0.9))
            if not attr_rows.empty
            else None
        ),
        "top_10_by_score": result.head(10)[
            [
                "job_id",
                "component",
                "alert_type",
                "max_anomaly_score",
                "n_anomalous_nodes",
                "anomalous_nodes",
                "attribution_confidence",
            ]
        ].to_dict(orient="records"),
    }
    (out_path.parent / "attributed_alerts_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    elapsed = time.perf_counter() - t0
    print(f"\n[attribute] Done in {elapsed:.1f}s")
    print(
        f"  Total alerts: {summary['total_alerts']:,}  "
        f"multi_node={summary['multi_node_alerts']}  "
        f"multi_job_attributed={summary['multi_job_attributed']}  "
        f"unique_jobs={summary['unique_jobs']}"
    )
    if summary["attribution_confidence_p50"] is not None:
        print(
            f"  Attribution confidence: "
            f"p50={summary['attribution_confidence_p50']:.2f}  "
            f"p90={summary['attribution_confidence_p90']:.2f}"
        )
    print(f"  Saved: {out_path}")
    return result


if __name__ == "__main__":
    run_attribute_alerts(force=True)
