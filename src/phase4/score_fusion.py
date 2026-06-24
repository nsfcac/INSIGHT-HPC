from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
import json, time
from typing import Optional

import pandas as pd

from src.utils.io_utils import load_config, load_parquet, save_parquet
from src.phase1.unified_scorer import compute_scores as compute_phase1_scores
from src.phase2.unified_scorer import compute_scores as compute_phase2_scores
from src.phase3.unified_scorer import compute_scores as compute_phase3_scores
from src.phase4.fusion_gbdt import (
    attach_episode_features,
    apply_fusion_gbdt,
    build_candidate_episodes,
    build_fused_index,
    emit_auditability_outputs,
    label_episodes,
    load_bundle,
    save_bundle,
    train_fusion_gbdt,
)
from src.phase4.operator_alert import (
    build_alerts,
    category_from_rules_and_context,
    write_alerts,
)
from src.phase4.score_fusion_module.eval import (
    evaluate_gbdt_as_detector,
    evaluate_v2,
    per_detector_eval,
)
from src.phase4.score_fusion_module.labels import (
    load_gt_events,
    resolve_paths,
    row_level_labels,
    split_of_episode,
)
from src.phase4.score_fusion_module.report import (
    candidate_coverage,
    episode_rule_summary,
    print_coverage,
    type_attribution,
    write_ablation_csv,
)
from src.phase4.score_fusion_module.table import run_fused_table


# Orchestrate phase-4: fuse phase scores, train/score the GBDT, and emit metrics and alerts.
def run(cfg: Optional[dict] = None, force_retrain: bool = True) -> dict:
    if cfg is None:
        cfg = load_config()

    phase2_enabled = cfg.get("phase2", {}).get("enabled", True)
    phase3_enabled = cfg.get("phase3", {}).get("enabled", True)

    p4_dir, fused_in, reports_dir = resolve_paths(cfg)
    if not fused_in.exists() or force_retrain:
        print(f"[score_fusion] Building per-minute fused table → {fused_in}")
        run_fused_table(force=force_retrain)

    print(f"[score_fusion] Loading fused table from {fused_in}")
    t0 = time.time()
    fused = load_parquet(fused_in)
    if "timestamp" not in fused.columns:
        raise RuntimeError(f"fused parquet has no timestamp column: {fused.columns}")
    fused["timestamp"] = pd.to_datetime(fused["timestamp"], utc=True)
    fused = fused.sort_values(["hostname", "component", "timestamp"]).reset_index(
        drop=True
    )
    print(f"[fusion_v2] Loaded {len(fused):,} rows in {time.time() - t0:.1f}s")

    gt_events = load_gt_events(cfg)

    # Row-level labels for isotonic calibration inside each phase's unified scorer.
    print("[fusion_v2] Building row-level labels for calibration")
    row_labels = row_level_labels(fused, gt_events)
    train_mask = fused.get("split", "train").astype(str) == "train"

    # Unified phase scorers.
    print("[fusion_v2] Computing phase-1 unified scores")
    p1 = compute_phase1_scores(fused, train_mask, row_labels)
    print("[fusion_v2] Computing phase-2 unified scores (enabled=%s)" % phase2_enabled)
    p2 = compute_phase2_scores(fused, train_mask, row_labels, enabled=phase2_enabled)
    print("[fusion_v2] Computing phase-3 unified scores (enabled=%s)" % phase3_enabled)
    p3 = compute_phase3_scores(fused, train_mask, row_labels, enabled=phase3_enabled)

    # Attach phase outputs to fused.
    for col in p1.columns:
        fused[col] = p1[col].values
    for col in p2.columns:
        fused[col] = p2[col].values
    for col in p3.columns:
        fused[col] = p3[col].values

    # Persist row-level phase scores for downstream / debugging.
    phase_scores_path = p4_dir / "phase_scores.parquet"
    keep = [
        "timestamp",
        "hostname",
        "component",
        "split",
        "phase1_score",
        "phase1_n_detectors_firing",
        "phase1_persistence_min",
        "phase1_max_detector_score",
        "phase2_score",
        "phase2_cluster_dist",
        "phase2_peer_divergence_z",
        "phase2_minutes_into_job",
        "phase3_score",
        "phase3_physics_z",
        "phase3_n_constraints",
        "phase3_context_score",
    ]
    keep = [c for c in keep if c in fused.columns]
    save_parquet(fused[keep].copy(), phase_scores_path)
    print(f"[fusion_v2] Wrote per-row phase scores to {phase_scores_path}")

    # Build candidate episodes.
    print("[fusion_v2] Building candidate episodes")
    candidates = build_candidate_episodes(fused)
    print(f"[fusion_v2] {len(candidates):,} candidate episodes formed")
    if len(candidates) == 0:
        print("[fusion_v2] No candidate episodes — skipping GBDT")
        return {"candidates": 0}

    print("[fusion_v2] Building shared fused row-index for O(log N) lookups")
    start_time = time.time()
    fused_index = build_fused_index(fused)
    print(f"[fusion_v2]   index built in {time.time() - start_time:.1f}s")

    # Attach episode features, labels, split — all use the shared index.
    candidates = attach_episode_features(candidates, fused, fused_index=fused_index)
    candidates["ep_label"] = label_episodes(candidates, gt_events).values
    candidates["split"] = split_of_episode(
        fused, candidates, fused_index=fused_index
    ).values
    print(
        f"[fusion_v2] Positive episodes: {int(candidates['ep_label'].sum())} / {len(candidates)}"
    )

    coverage = candidate_coverage(candidates, gt_events)
    print_coverage(coverage)
    # Stash coverage for the precision_recall_v2.json summary below.
    coverage_report = coverage

    # Train on train+val, calibrate on val.
    tv_mask = candidates["split"].isin(["train", "val"])
    bundle_path = p4_dir / "fusion_gbdt.pkl"

    incumbent_bundle = None
    if bundle_path.exists():
        try:
            incumbent_bundle = load_bundle(bundle_path)
            incumbent_threshold = getattr(incumbent_bundle, "t_confirmed", "?")
            print(
                f"[fusion_v2] Incumbent bundle loaded for monotone calibration "
                f"(t_confirmed={incumbent_threshold})"
            )
        except Exception as e:
            print(
                f"[fusion_v2] Incumbent bundle unreadable ({e}); cold-start calibration"
            )
            incumbent_bundle = None

    if force_retrain or not bundle_path.exists():
        print("[fusion_v2] Training GBDT on %d train+val episodes" % int(tv_mask.sum()))
        cal_cfg = cfg.get("phase4", {}).get("v2", {}).get("calibration", {}) or {}
        bundle = train_fusion_gbdt(
            candidates[tv_mask],
            candidates.loc[tv_mask, "ep_label"],
            candidates.loc[tv_mask, "split"],
            gt_events=gt_events,
            incumbent_bundle=incumbent_bundle,
            floor_p=float(cal_cfg.get("min_val_precision", 0.85)),
            floor_r=float(cal_cfg.get("min_val_recall", 0.0)),
            floor_f1=float(cal_cfg.get("min_val_f1", 0.0)),
            use_incumbent_ratchet=bool(cal_cfg.get("use_incumbent_ratchet", True)),
            crit_target_p=float(cal_cfg.get("crit_target_p", 0.90)),
            crit_fallback=float(cal_cfg.get("crit_fallback", 0.85)),
            cand_target_r=float(cal_cfg.get("cand_target_r", 0.85)),
            cand_floor_p=float(cal_cfg.get("cand_floor_p", 0.50)),
            cand_fallback=float(cal_cfg.get("cand_fallback", 0.30)),
        )
        save_bundle(bundle, p4_dir)
        print(f"[fusion_v2] Saved bundle to {bundle_path}")
    else:
        print(f"[fusion_v2] Loading existing bundle from {bundle_path}")
        bundle = incumbent_bundle

    # Score all candidates (including test).
    scored = apply_fusion_gbdt(candidates, bundle)

    # Auditability outputs on test episodes (reviewers only inspect test).
    test_mask = scored["split"] == "test"
    audit_out_dir = reports_dir
    audit_out_dir.mkdir(parents=True, exist_ok=True)
    audit = emit_auditability_outputs(
        bundle,
        scored[test_mask],
        candidates.loc[test_mask, "ep_label"] if test_mask.any() else None,
        audit_out_dir,
    )
    print(f"[fusion_v2] Auditability outputs: {audit}")

    # Write v2 episodes parquet.
    fused_v2_path = p4_dir / "fused_alerts_v2.parquet"
    keep_v2 = [
        "hostname",
        "component",
        "episode_start",
        "episode_end",
        "split",
        "ep_duration_min",
        "ep_peak_phase1",
        "ep_peak_phase3",
        "ep_consensus_phases",
        "phase1_score",
        "phase2_score",
        "phase3_score",
        "phase2_peer_divergence_z",
        "phase2_cluster_dist",
        "phase3_physics_z",
        "phase3_n_constraints",
        "phase3_context_score",
        "hour_sin",
        "hour_cos",
        "ep_label",
        "fusion_prob",
        "fusion_tier",
    ]
    keep_v2 = [c for c in keep_v2 if c in scored.columns]
    save_parquet(scored[keep_v2].copy(), fused_v2_path)
    print(f"[fusion_v2] Wrote scored episodes to {fused_v2_path}")

    rule_cols = [
        c for c in fused.columns if c.startswith("rule_") or c.startswith("const")
    ]
    ctx_cols = [
        c for c in ("anomaly_context", "anomaly_reason", "rack") if c in fused.columns
    ]
    if rule_cols or ctx_cols:
        print("[fusion_v2] Enriching episodes with rule flags for operator alerts")
        enriched = episode_rule_summary(
            fused, scored, rule_cols + ctx_cols, fused_index=fused_index
        )
        scored_enriched = scored.merge(
            enriched,
            on=["hostname", "component", "episode_start"],
            how="left",
        )
    else:
        scored_enriched = scored

    # Operator alerts: read SHAP output if it was emitted.
    shap_parquet = audit_out_dir / "shap_per_event.parquet"
    shap_df = load_parquet(shap_parquet) if shap_parquet.exists() else None

    alerts = build_alerts(
        scored_enriched[test_mask] if test_mask.any() else scored_enriched,
        shap_df=shap_df,
        feature_cols=bundle.feature_columns,
    )
    alert_paths = write_alerts(alerts, audit_out_dir)
    print(f"[fusion_v2] Operator alerts: {alert_paths}")

    print("[fusion_v2] Inferring per-episode category for type-attribution")
    scored_enriched["inferred_category"] = scored_enriched.apply(
        category_from_rules_and_context, axis=1
    )

    print("[fusion_v2] Computing precision/recall on test + pooled")
    pr = evaluate_v2(scored_enriched, gt_events, bundle)
    # Include the candidate coverage report (ceiling on recall).
    pr["candidate_coverage"] = coverage_report
    print("[fusion_v2] Computing type-attribution confusion matrix on test/CONFIRMED")
    t_confirmed = bundle.t_confirmed
    test_scored = scored_enriched[
        (scored_enriched["split"] == "test")
        & (scored_enriched["fusion_prob"] >= t_confirmed)
    ]
    gt_test = (
        gt_events[gt_events["split"] == "test"]
        if "split" in gt_events.columns
        else gt_events
    )
    pr["type_attribution"] = type_attribution(test_scored, gt_test)

    # Per-detector / per-cascade / per-component ablation table for the paper.
    print("[fusion_v2] Computing per-detector ablation table on test split")
    pr["per_detector_table"] = per_detector_eval(fused, gt_events)

    gbdt_row = evaluate_gbdt_as_detector(scored_enriched, gt_events, bundle)
    if gbdt_row:
        for split_name in ("train", "val", "test", "pooled"):
            for comp in ("all", "zen4", "h100"):
                row = gbdt_row.get((split_name, comp))
                if row is not None:
                    pr["per_detector_table"].setdefault(split_name, {}).setdefault(
                        comp, []
                    ).insert(0, row)

    pr_path = audit_out_dir / "precision_recall.json"
    with open(pr_path, "w") as f:
        json.dump(pr, f, indent=2, default=str)
    print(f"[fusion_v2] Wrote PR to {pr_path}")

    # Also emit the per-detector table as CSV for paper ingestion.
    ablation_csv = audit_out_dir / "per_detector_ablation.csv"
    write_ablation_csv(pr["per_detector_table"], ablation_csv)
    print(f"[fusion_v2] Wrote per-detector ablation CSV to {ablation_csv}")

    return {
        "phase_scores": str(phase_scores_path),
        "fused_v2": str(fused_v2_path),
        "audit": audit,
        "alerts": alert_paths,
        "precision_recall": str(pr_path),
        "n_candidates": int(len(candidates)),
        "n_positives": int(candidates["ep_label"].sum()),
        "bundle": str(bundle_path),
        "summary": pr.get("summary", {}),
    }
