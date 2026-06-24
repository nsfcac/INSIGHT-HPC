from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.phase4.score_fusion_module.event_eval import event_level_match


# Compute per-tier, per-split event-level P/R/F1 and AUROC/AUPRC for scored episodes.
def evaluate_v2(scored: pd.DataFrame, gt_events: pd.DataFrame, bundle) -> dict:
    if len(scored) == 0 or len(gt_events) == 0:
        return {"summary": {}, "per_category": {}, "per_split": {}}

    out: dict = {}
    for split_name in ("train", "val", "test", "pooled"):
        if split_name == "pooled":
            sub = scored
        else:
            sub = scored[scored["split"] == split_name]
        if len(sub) == 0:
            continue

        gt_for_split = gt_events.copy()
        if "split" in gt_for_split.columns and split_name != "pooled":
            gt_for_split = gt_for_split[gt_for_split["split"] == split_name]

        # For each tier, compute event-level P/R.
        per_tier = {}
        for tier_name, t in [
            ("fused_critical_v2", bundle.t_critical),
            ("fused_high_v2", bundle.t_confirmed),
            ("fused_candidate_v2", bundle.t_candidate),
        ]:
            pos_episodes = sub[sub["fusion_prob"] >= t]
            tp, fp, fn, matched_events, per_cat = event_level_match(
                pos_episodes, gt_for_split
            )
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            per_tier[tier_name] = {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "recall_by_category": per_cat,
            }

        # AUROC / AUPRC on the full score distribution (any candidate vs label).
        if "ep_label" in sub.columns and sub["ep_label"].nunique() == 2:
            auroc = float(roc_auc_score(sub["ep_label"], sub["fusion_prob"]))
            auprc = float(average_precision_score(sub["ep_label"], sub["fusion_prob"]))
            per_tier["auroc_episode"] = round(auroc, 4)
            per_tier["auprc_episode"] = round(auprc, 4)

        out[split_name] = per_tier

    # Headline summary from test fused_high_v2.
    test = out.get("test", {})
    high = test.get("fused_high_v2", {})
    summary = {
        "test_P": high.get("precision", 0.0),
        "test_R": high.get("recall", 0.0),
        "test_F1": high.get("f1", 0.0),
        "test_AUROC_episode": test.get("auroc_episode", 0.0),
        "test_AUPRC_episode": test.get("auprc_episode", 0.0),
        "t_critical": bundle.t_critical,
        "t_confirmed": bundle.t_confirmed,
        "t_candidate": bundle.t_candidate,
    }

    return {"per_split": out, "summary": summary}
