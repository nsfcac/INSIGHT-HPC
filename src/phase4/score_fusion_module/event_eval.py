from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
import pandas as pd


# Match scored episodes to GT events (with lead/lag) for event-level P/R counts.
def event_level_match(
    episodes: pd.DataFrame,
    gt_events: pd.DataFrame,
    lead_min: int = 60,
    lag_min: int = 30,
) -> tuple[int, int, int, set, dict]:
    if len(episodes) == 0 and len(gt_events) == 0:
        return 0, 0, 0, set(), {}

    gt = gt_events.copy()
    gt["w_start"] = pd.to_datetime(gt["event_start"], utc=True) - pd.Timedelta(
        minutes=lead_min
    )
    gt["w_end"] = pd.to_datetime(gt["event_end"], utc=True) + pd.Timedelta(
        minutes=lag_min
    )

    matched_events: set = set()
    matched_episodes = [False] * len(episodes)

    eps = episodes.reset_index(drop=True)
    host_to_events: dict[str, list[tuple]] = {}
    for i, g in gt.iterrows():
        key = g.get("hostname", "*") or "*"
        host_to_events.setdefault(key, []).append((i, g["w_start"], g["w_end"]))
    wildcard = host_to_events.get("*", [])

    for ei, ep in eps.iterrows():
        host = ep["hostname"]
        s = pd.to_datetime(ep["episode_start"], utc=True)
        e = pd.to_datetime(ep["episode_end"], utc=True)
        for gi, gs, ge in host_to_events.get(host, []) + wildcard:
            if s <= ge and e >= gs:
                matched_events.add(gi)
                matched_episodes[ei] = True

    tp_alerts = int(sum(1 for m in matched_episodes if m))  # alert-count
    fp_alerts = int(sum(1 for m in matched_episodes if not m))
    fn_events = int(len(gt) - len(matched_events))  # event-count

    per_cat: dict = {}
    if "category" in gt.columns:
        for cat in gt["category"].unique():
            cat_gt = gt[gt["category"] == cat]
            cat_matched = sum(1 for i, _ in cat_gt.iterrows() if i in matched_events)
            per_cat[cat] = round(cat_matched / max(1, len(cat_gt)), 3)

    return tp_alerts, fp_alerts, fn_events, matched_events, per_cat
