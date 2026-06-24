from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.phase4.fusion_gbdt import build_fused_index, window_slice
from src.phase4.score_fusion_module.event_eval import event_level_match


# Flatten the per-detector ablation table into a CSV for the paper.
def write_ablation_csv(per_detector_table: dict, out_csv: Path) -> None:
    rows = []
    for split, by_comp in per_detector_table.items():
        for comp, detectors in by_comp.items():
            for det in detectors:
                rows.append(
                    {
                        "split": split,
                        "component": comp,
                        "detector": det.get("name", det.get("detector", "?")),
                        "flag_col": det.get("detector", ""),
                        "tp": det.get("tp", 0),
                        "fp": det.get("fp", 0),
                        "fn": det.get("fn", 0),
                        "precision": det.get("precision", 0.0),
                        "recall": det.get("recall", 0.0),
                        "f1": det.get("f1", 0.0),
                        "n_deduped_alerts": det.get("n_deduped_alerts", 0),
                        "auroc_episode": det.get("auroc_episode", ""),
                        "auprc_episode": det.get("auprc_episode", ""),
                        "auroc_pointwise": det.get("auroc_pointwise", ""),
                        "auprc_pointwise": det.get("auprc_pointwise", ""),
                        "precision_at_10": det.get("precision_at_10", ""),
                        "precision_at_50": det.get("precision_at_50", ""),
                        "precision_at_100": det.get("precision_at_100", ""),
                        "precision_at_200": det.get("precision_at_200", ""),
                        "auc_score_mode": det.get("auc_score_mode", ""),
                    }
                )
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# Compute the category confusion matrix between alerts and matched GT events.
def type_attribution(
    scored_episodes: pd.DataFrame,
    gt_events: pd.DataFrame,
    lead_min: int = 60,
    lag_min: int = 30,
) -> dict:
    if len(scored_episodes) == 0 or len(gt_events) == 0:
        return {
            "n_matched_events": 0,
            "accuracy": 0.0,
            "confusion": {},
            "per_category": {},
        }

    gt = gt_events.copy()
    gt["w_start"] = pd.to_datetime(gt["event_start"], utc=True) - pd.Timedelta(
        minutes=lead_min
    )
    gt["w_end"] = pd.to_datetime(gt["event_end"], utc=True) + pd.Timedelta(
        minutes=lag_min
    )

    # Per-host index of scored alerts for overlap lookups.
    eps = scored_episodes.reset_index(drop=True).copy()
    eps["_s"] = pd.to_datetime(eps["episode_start"], utc=True)
    eps["_e"] = pd.to_datetime(eps["episode_end"], utc=True)

    alerts_by_host: dict[str, list[tuple]] = {}
    for i, a in eps.iterrows():
        alerts_by_host.setdefault(a["hostname"], []).append(
            (
                a["_s"],
                a["_e"],
                float(a.get("fusion_prob", 0.0)),
                str(a.get("inferred_category", "unknown")),
            )
        )
    all_alerts = [v for lst in alerts_by_host.values() for v in lst]

    confusion: dict[str, dict[str, int]] = {}
    per_cat_matched: dict[str, int] = {}
    per_cat_correct: dict[str, int] = {}  # strict
    per_cat_correct_lenient: dict[str, int] = {}  # lenient: any overlap matched
    per_cat_misses: dict[str, list[str]] = {}
    n_total = 0
    n_correct = 0
    n_correct_lenient = 0

    for _, g in gt.iterrows():
        gh = g.get("hostname") or "*"
        gc = str(g.get("category", "unknown"))
        candidates = alerts_by_host.get(gh, []) if gh != "*" else all_alerts
        overlapping = [
            (s, e, p, c)
            for (s, e, p, c) in candidates
            if s <= g["w_end"] and e >= g["w_start"]
        ]
        if not overlapping:
            continue
        # Representative = highest fusion_prob alert (strict view).
        overlapping.sort(key=lambda x: -x[2])
        inferred_strict = overlapping[0][3]
        # Lenient view: any overlapping alert's category matches GT.
        inferred_any = {cat for (_, _, _, cat) in overlapping}

        confusion.setdefault(gc, {}).setdefault(inferred_strict, 0)
        confusion[gc][inferred_strict] += 1
        per_cat_matched[gc] = per_cat_matched.get(gc, 0) + 1

        is_strict_match = inferred_strict == gc
        if is_strict_match:
            per_cat_correct[gc] = per_cat_correct.get(gc, 0) + 1
            n_correct += 1
        else:
            per_cat_misses.setdefault(gc, []).append(inferred_strict)

        is_lenient_match = gc in inferred_any
        if is_lenient_match:
            per_cat_correct_lenient[gc] = per_cat_correct_lenient.get(gc, 0) + 1
            n_correct_lenient += 1
        n_total += 1

    per_category = {}
    from collections import Counter

    for gc, n in per_cat_matched.items():
        correct = per_cat_correct.get(gc, 0)
        correct_len = per_cat_correct_lenient.get(gc, 0)
        misses = per_cat_misses.get(gc, [])
        mc_miss = Counter(misses).most_common(1)
        per_category[gc] = {
            "matched": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "correct_lenient": correct_len,
            "accuracy_lenient": round(correct_len / n, 4) if n else 0.0,
            "most_common_miss": mc_miss[0][0] if mc_miss else None,
            "most_common_miss_count": mc_miss[0][1] if mc_miss else 0,
        }

    return {
        "n_matched_events": n_total,
        "accuracy": round(n_correct / n_total, 4) if n_total else 0.0,
        "accuracy_lenient": round(n_correct_lenient / n_total, 4) if n_total else 0.0,
        "confusion": confusion,
        "per_category": per_category,
    }


# Compute the fraction of GT events covered by candidate episodes (the recall ceiling).
def candidate_coverage(
    candidates: pd.DataFrame,
    gt_events: pd.DataFrame,
    lead_min: int = 60,
    lag_min: int = 30,
) -> dict:
    if len(gt_events) == 0:
        return {"by_split": {}, "by_category": {}, "uncovered_events": []}

    gt = gt_events.copy()
    gt["w_start"] = pd.to_datetime(gt["event_start"], utc=True) - pd.Timedelta(
        minutes=lead_min
    )
    gt["w_end"] = pd.to_datetime(gt["event_end"], utc=True) + pd.Timedelta(
        minutes=lag_min
    )

    # Index candidates per host for fast lookup.
    cands_by_host: dict[str, list[tuple]] = {}
    for _, c in candidates.iterrows():
        host = c["hostname"]
        cs = pd.to_datetime(c["episode_start"], utc=True)
        ce = pd.to_datetime(c["episode_end"], utc=True)
        cands_by_host.setdefault(host, []).append((cs, ce))

    cands_by_rack: dict[str, list[tuple]] = {}
    for host, lst in cands_by_host.items():
        if isinstance(host, str) and (
            host.startswith("rpc-") or host.startswith("rpg-")
        ):
            parts = host.split("-")
            if len(parts) >= 2:
                cands_by_rack.setdefault(parts[1], []).extend(lst)

    covered_mask = []
    uncovered = []
    for i, g in gt.iterrows():
        gh = g.get("hostname") or "*"
        hits = []
        if gh == "*":
            # Cluster GT — covered if ANY candidate episode overlaps the window
            hits = [
                1
                for lst in cands_by_host.values()
                for cs, ce in lst
                if cs <= g["w_end"] and ce >= g["w_start"]
            ]
        else:
            for cs, ce in cands_by_host.get(gh, []):
                if cs <= g["w_end"] and ce >= g["w_start"]:
                    hits.append(1)
                    break
            if not hits and isinstance(gh, str) and gh.startswith("pdu-"):
                parts = gh.split("-")
                if len(parts) >= 2:
                    for cs, ce in cands_by_rack.get(parts[1], []):
                        if cs <= g["w_end"] and ce >= g["w_start"]:
                            hits.append(1)
                            break
        is_covered = len(hits) > 0
        covered_mask.append(is_covered)
        if not is_covered:
            uncovered.append(
                {
                    "event_id": g.get("event_id", ""),
                    "category": g.get("category", ""),
                    "hostname": gh,
                    "split": g.get("split", ""),
                    "event_start": str(g.get("event_start", "")),
                    "duration_min": float(g.get("duration_min", 0) or 0),
                }
            )
    gt["_covered"] = covered_mask

    by_split: dict = {}
    if "split" in gt.columns:
        for s, sub in gt.groupby("split"):
            n = int(len(sub))
            k = int(sub["_covered"].sum())
            by_split[str(s)] = {
                "gt": n,
                "covered": k,
                "rate": round(k / n, 4) if n else 0.0,
            }
    else:
        n = int(len(gt))
        k = int(gt["_covered"].sum())
        by_split = {
            "all": {"gt": n, "covered": k, "rate": round(k / n, 4) if n else 0.0}
        }

    by_category: dict = {}
    if "category" in gt.columns:
        for cat, sub in gt.groupby("category"):
            n = int(len(sub))
            k = int(sub["_covered"].sum())
            by_category[str(cat)] = {
                "gt": n,
                "covered": k,
                "rate": round(k / n, 4) if n else 0.0,
            }

    return {
        "by_split": by_split,
        "by_category": by_category,
        "uncovered_events": uncovered,
    }


# Print candidate coverage by split and category, with uncovered events.
def print_coverage(coverage: dict) -> None:
    print(
        "[fusion_v2] Candidate coverage (ceiling on recall — irrecoverable if < 100%):"
    )
    for split, s in coverage.get("by_split", {}).items():
        bar_ok = "OK" if s["rate"] >= 0.95 else "LOW"
        print(
            f"  {split:8s}: {s['covered']:3d}/{s['gt']:3d}  "
            f"= {100*s['rate']:.1f}%  [{bar_ok}]"
        )
    print("  per-category coverage:")
    for cat, c in sorted(
        coverage.get("by_category", {}).items(), key=lambda kv: kv[1]["rate"]
    ):
        print(
            f"    {cat:26s}  {c['covered']:3d}/{c['gt']:3d}  " f"= {100*c['rate']:.1f}%"
        )
    n_missed = len(coverage.get("uncovered_events", []))
    if n_missed:
        print(f"  {n_missed} uncovered GT events (ceiling loss) — first 5:")
        for u in coverage["uncovered_events"][:5]:
            print(
                f"    {u['event_id']:20s}  cat={u['category']:20s}  "
                f"split={u['split']:6s}  host={u['hostname']:14s}  "
                f"dur={u['duration_min']:.1f}m"
            )


# Summarise rule flags and context labels within each episode's window.
def episode_rule_summary(
    fused: pd.DataFrame,
    episodes: pd.DataFrame,
    flag_cols: list[str],
    fused_index: Optional[dict] = None,
) -> pd.DataFrame:
    if fused_index is None:
        # Keep only the columns we need so the index builds quickly.
        fused_subset = fused[["hostname", "component", "timestamp"] + flag_cols].copy()
        fused_index = build_fused_index(fused_subset)
    sorted_fused = fused_index["sorted"]
    # Cache column arrays up front.
    col_arrays = {
        c: sorted_fused[c].to_numpy() for c in flag_cols if c in sorted_fused.columns
    }
    col_dtypes = {c: sorted_fused[c].dtype for c in col_arrays}

    rows = []
    for _, ep in episodes.iterrows():
        lo, hi = window_slice(
            fused_index,
            ep["hostname"],
            ep["component"],
            ep["episode_start"],
            ep["episode_end"],
        )
        agg = {
            "hostname": ep["hostname"],
            "component": ep["component"],
            "episode_start": ep["episode_start"],
        }
        if hi <= lo:
            for c in flag_cols:
                if c not in col_arrays:
                    agg[c] = ""
                elif col_dtypes[c] == "bool":
                    agg[c] = False
                else:
                    agg[c] = ""
            rows.append(agg)
            continue
        for c in flag_cols:
            if c not in col_arrays:
                agg[c] = ""
                continue
            window = col_arrays[c][lo:hi]
            if col_dtypes[c] == "bool":
                agg[c] = bool(np.any(window))
            else:
                window_str = pd.Series(window).fillna("").astype(str).to_numpy()
                window_ne = window_str[window_str != ""]
                if len(window_ne) == 0:
                    agg[c] = ""
                else:
                    vals, counts = np.unique(window_ne, return_counts=True)
                    agg[c] = str(vals[int(np.argmax(counts))])
        rows.append(agg)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    res = run()
    print(f"\n[fusion_v2] done: {res}", file=sys.stderr)
