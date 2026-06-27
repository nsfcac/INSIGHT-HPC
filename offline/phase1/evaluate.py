from __future__ import annotations

import json, time
from pathlib import Path


from shared.utils.io_utils import load_config, save_parquet
from offline.visualization.plot_utils import ensure_dir, write_json
from offline.phase1.evaluate_module.model_loading import *


# Merge phase-1 detector outputs, compute alert/agreement metrics, and write the report.
def run_evaluation(force: bool = False) -> dict:
    cfg = load_config()
    model_dir = Path(cfg["phase1"]["output_dir"])
    eval_cfg = cfg["phase1"].get("evaluation", {})
    out_dir = ensure_dir(eval_cfg.get("output_dir", "offline/reports/phase1_eval"))
    dpi = cfg.get("visualization", {}).get("dpi", 140)

    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not force:
        print("[eval] Report exists — skipping (use force=True to rerun)")
        with open(summary_path) as f:
            return json.load(f)

    t0 = time.perf_counter()
    print("\n[eval] Phase I Evaluation")

    thr_df = load_threshold_alerts(model_dir)
    if_df = load_if_scores(model_dir)
    pdu_df = load_pdu_alerts(model_dir)

    if thr_df is None and if_df is None:
        print("[eval] ERROR: No model outputs found.  Run baseline and IF first.")
        return {}

    results: dict = {}

    if thr_df is not None and if_df is not None:
        print(f"  Merging: threshold={len(thr_df):,} rows  IF={len(if_df):,} rows")
        merged = merge_models(thr_df, if_df)
        print(f"  Merged: {len(merged):,} rows")
    elif thr_df is not None:
        merged = thr_df.rename(columns={"is_flagged": "thr_is_flagged"})
        merged["if_is_anomaly"] = False
        merged["if_anomaly_score"] = 0.0
        merged["both_flag"] = False
        merged["either_flag"] = merged["thr_is_flagged"]
    else:
        merged = if_df.copy()
        merged["thr_is_flagged"] = False
        merged["both_flag"] = False
        merged["either_flag"] = merged["if_is_anomaly"]

    results["alert_rates"] = compute_alert_rates(merged)
    results["agreement"] = compute_agreement(merged)
    results["clustering"] = compute_clustering(merged)

    window_min = int(eval_cfg.get("suspicious_window_min", 60))
    multi_node_thr = float(eval_cfg.get("multi_node_threshold", 0.25))

    suspicious = find_suspicious_periods(merged, window_min, multi_node_thr)
    if not suspicious.empty:
        suspicious.to_csv(out_dir / "top_suspicious_periods.csv", index=False)
        results["top_suspicious_periods"] = suspicious.head(5).to_dict(orient="records")
        print(f"  Top suspicious periods saved: {len(suspicious)} rows")

    node_sum = node_summary(merged)
    if not node_sum.empty:
        save_parquet(node_sum, out_dir / "node_summary.parquet")
        results["node_count"] = len(node_sum)

    if pdu_df is not None:
        n_pdu = int(pdu_df["is_flagged"].sum())
        results["pdu_flagged_rows"] = n_pdu
        results["pdu_flag_rate"] = float(pdu_df["is_flagged"].mean())
        print(
            f"  PDU: {n_pdu:,} flagged rows  rate={results['pdu_flag_rate']*100:.2f}%"
        )

    plot_alert_rates(results["alert_rates"], out_dir / "alert_rates.png", dpi)
    plot_score_distributions(merged, out_dir / "score_distributions.png", dpi)
    plot_temporal_alert_rate(
        merged, window_min, out_dir / "temporal_alert_rate.png", dpi
    )
    if thr_df is not None and if_df is not None:
        plot_agreement(results["agreement"], out_dir / "model_agreement.png", dpi)

    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    write_json(results, summary_path)
    write_readme(results, out_dir)

    print(f"\n[eval] Complete in {results['elapsed_seconds']:.1f}s")
    print(
        f"  Threshold alert rate: {results['alert_rates']['all']['threshold_alert_rate']*100:.2f}%"
    )
    print(f"  IF alert rate: {results['alert_rates']['all']['if_alert_rate']*100:.2f}%")
    print(
        f"  Model agreement: {results['agreement']['agreement_of_flagged']:.1f}% of flagged rows"
    )
    print(f"  Report: {out_dir}")
    return results


if __name__ == "__main__":
    run_evaluation(force=True)
