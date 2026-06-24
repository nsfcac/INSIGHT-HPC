from __future__ import annotations

from src.phase4.score_fusion_module.constants import *
from typing import Optional

import numpy as np
import pandas as pd

from src.phase4.fusion_gbdt import FEATURE_COLUMNS
from src.phase4.score_fusion_module.event_eval import event_level_match


# Collapse consecutive firing rows into episodes split by a time gap.
def dedup_rows_to_episodes(rows: pd.DataFrame, gap_min: int = 60) -> pd.DataFrame:
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "hostname",
                "component",
                "episode_start",
                "episode_end",
                "max_score",
                "n_firings",
                "duration_min",
                "split",
            ]
        )
    r = (
        rows[["timestamp", "hostname", "component", "_score", "split"]].copy()
        if "_score" in rows.columns and "split" in rows.columns
        else rows.copy()
    )
    r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
    if "component" not in r.columns:
        r["component"] = ""
    if "_score" not in r.columns:
        r["_score"] = 1.0
    if "split" not in r.columns:
        r["split"] = ""
    r = r.sort_values(["hostname", "component", "timestamp"]).reset_index(drop=True)

    gap = pd.Timedelta(minutes=gap_min)
    prev_ts = r.groupby(["hostname", "component"])["timestamp"].shift(1)
    time_gap = r["timestamp"] - prev_ts
    host_change = (r["hostname"] != r["hostname"].shift(1)) | (
        r["component"] != r["component"].shift(1)
    )
    # new_episode is True at row 0 (NaT prev_ts propagates to True via host_change)
    new_episode = host_change | (time_gap > gap)
    new_episode.iloc[0] = True
    r["_ep_id"] = new_episode.cumsum()

    # One aggregation produces all episode columns.
    g = r.groupby("_ep_id", sort=False)
    out = g.agg(
        hostname=("hostname", "first"),
        component=("component", "first"),
        episode_start=("timestamp", "min"),
        episode_end=("timestamp", "max"),
        max_score=("_score", "max"),
        n_firings=("timestamp", "size"),
        split=("split", "first"),
    ).reset_index(drop=True)
    out["duration_min"] = (
        (out["episode_end"] - out["episode_start"]).dt.total_seconds() / 60
    ).astype(int) + 1
    return out[
        [
            "hostname",
            "component",
            "episode_start",
            "episode_end",
            "max_score",
            "n_firings",
            "duration_min",
            "split",
        ]
    ]


# Mark each row that falls inside a GT event window (per host plus wildcard).
def ground_truth_row_membership_mask(
    s_all: pd.DataFrame, gt_sub: pd.DataFrame
) -> pd.Series:
    pos_mask = pd.Series(False, index=s_all.index)
    if len(gt_sub) == 0 or len(s_all) == 0:
        return pos_mask
    ts = (
        pd.to_datetime(s_all["timestamp"], utc=True)
        .dt.tz_convert(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    hosts = s_all["hostname"].to_numpy()
    gt2 = gt_sub.copy()
    gt2["_s"] = pd.to_datetime(gt2["event_start"], utc=True).dt.tz_convert(None)
    gt2["_e"] = pd.to_datetime(gt2["event_end"], utc=True).dt.tz_convert(None)

    order = np.argsort(hosts, kind="stable")
    hosts_sorted = hosts[order]
    ts_sorted = ts[order]
    uniq, first = np.unique(hosts_sorted, return_index=True)
    host_ranges = {
        h: (first[i], first[i + 1] if i + 1 < len(first) else len(hosts_sorted))
        for i, h in enumerate(uniq)
    }

    mask_arr = np.zeros(len(s_all), dtype=bool)
    for _, g in gt2.iterrows():
        host = g.get("hostname") or "*"
        s = np.datetime64(g["_s"].to_datetime64())
        e = np.datetime64(g["_e"].to_datetime64())
        if host == "*":
            # Wildcard: all hosts, window-scoped mask over the unsorted frame.
            mask_arr |= (ts >= s) & (ts <= e)
            continue
        rng = host_ranges.get(host)
        if rng is None:
            continue
        lo, hi = rng
        # host-sliced timestamps are sorted → O(log N) window lookup.
        start = np.searchsorted(ts_sorted[lo:hi], s, side="left") + lo
        stop = np.searchsorted(ts_sorted[lo:hi], e, side="right") + lo
        if stop > start:
            mask_arr[order[start:stop]] = True
    pos_mask[:] = mask_arr
    return pos_mask


# Evaluate one detector's episode-level P/R/F1 and AUROC/AUPRC against GT.
def evaluate_single_detector(
    fused: pd.DataFrame,
    gt_events: pd.DataFrame,
    flag_col: str,
    score_col: Optional[str] = None,
    split: Optional[str] = None,
    component: Optional[str] = None,
    shared: Optional[dict] = None,
) -> dict:
    if flag_col not in fused.columns:
        return {
            "detector": flag_col,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_deduped_alerts": 0,
            "note": "column missing",
        }

    if shared is not None and "sub_frame" in shared:
        sub = shared["sub_frame"]
    else:
        sub = fused
        if split is not None and "split" in fused.columns:
            sub = sub[sub["split"] == split]
        if component is not None and "component" in fused.columns:
            sub = sub[sub["component"] == component]

    # Rows where the detector fires.
    firing = sub[sub[flag_col].astype(bool)][
        ["timestamp", "hostname", "component", "split"]
    ].copy()
    if score_col is not None and score_col in sub.columns:
        firing["_score"] = pd.to_numeric(
            sub.loc[firing.index, score_col], errors="coerce"
        ).fillna(0)
    else:
        firing["_score"] = 1.0

    if len(firing) == 0:
        gt_sub = gt_events
        if split is not None and "split" in gt_events.columns:
            gt_sub = gt_events[gt_events["split"] == split]
        return {
            "detector": flag_col,
            "tp": 0,
            "fp": 0,
            "fn": int(len(gt_sub)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_deduped_alerts": 0,
        }

    episodes = dedup_rows_to_episodes(firing)

    if shared is not None and "gt_sub" in shared:
        gt_sub = shared["gt_sub"]
    else:
        gt_sub = gt_events.copy()
        if split is not None and "split" in gt_sub.columns:
            gt_sub = gt_sub[gt_sub["split"] == split]
        if component is not None and "hostname" in gt_sub.columns and len(gt_sub) > 0:
            host_comp = fused[["hostname", "component"]].drop_duplicates()
            hosts_in_comp = set(
                host_comp[host_comp["component"] == component]["hostname"]
            )
            gt_sub = gt_sub[
                gt_sub["hostname"].isin(hosts_in_comp)
                | (gt_sub["hostname"].isna())
                | (gt_sub["hostname"] == "*")
            ]

    tp_alerts, fp_alerts, fn_events, matched_events, per_cat = event_level_match(
        episodes.rename(
            columns={"episode_start": "episode_start", "episode_end": "episode_end"}
        ),
        gt_sub,
    )
    p = tp_alerts / (tp_alerts + fp_alerts) if (tp_alerts + fp_alerts) else 0.0
    r = tp_alerts / (tp_alerts + fn_events) if (tp_alerts + fn_events) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    result = {
        "detector": flag_col,
        "tp": tp_alerts,
        "fp": fp_alerts,
        "fn": fn_events,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "n_deduped_alerts": int(len(episodes)),
    }
    if split is not None:
        result["split"] = split
    if component is not None:
        result["component"] = component

    if score_col is not None and score_col in sub.columns and len(gt_sub) > 0:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            if shared is not None and "pos_mask" in shared:
                pos_mask = shared["pos_mask"]
                s_all = shared.get("sub_frame", sub)
            else:
                s_all = sub[["timestamp", "hostname", score_col]].copy()
                s_all["timestamp"] = pd.to_datetime(s_all["timestamp"], utc=True)
                pos_mask = ground_truth_row_membership_mask(s_all, gt_sub)

            n_pos = int(pos_mask.sum())
            if n_pos >= 50:
                rng = np.random.default_rng(42)
                pos_idx_arr = np.flatnonzero(
                    pos_mask.to_numpy() if hasattr(pos_mask, "to_numpy") else pos_mask
                )
                neg_idx_arr = np.flatnonzero(
                    ~(
                        pos_mask.to_numpy()
                        if hasattr(pos_mask, "to_numpy")
                        else pos_mask
                    )
                )
                if len(neg_idx_arr) > n_pos:
                    neg_sample = rng.choice(neg_idx_arr, size=n_pos, replace=False)
                else:
                    neg_sample = neg_idx_arr
                sample_pos = pd.Index(s_all.index[pos_idx_arr])
                sample_neg = pd.Index(s_all.index[neg_sample])
                scores_pw = (
                    pd.to_numeric(
                        sub.loc[sample_pos.append(sample_neg), score_col],
                        errors="coerce",
                    )
                    .fillna(0)
                    .to_numpy()
                )
                labels_pw = np.concatenate(
                    [
                        np.ones(len(sample_pos), dtype=np.int8),
                        np.zeros(len(sample_neg), dtype=np.int8),
                    ]
                )
                result["auroc_pointwise"] = round(
                    float(roc_auc_score(labels_pw, scores_pw)), 4
                )
                result["auprc_pointwise"] = round(
                    float(average_precision_score(labels_pw, scores_pw)), 4
                )
                result["pointwise_n_pos"] = n_pos
                result["pointwise_n_neg"] = int(len(sample_neg))
        except Exception:
            pass

    if len(episodes) > 5 and len(gt_sub) > 0:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            if shared is not None and "host_to_events" in shared:
                host_to_events = shared["host_to_events"]
            else:
                gt2 = gt_sub.copy()
                gt2["_w_start"] = pd.to_datetime(
                    gt2["event_start"], utc=True
                ) - pd.Timedelta(minutes=60)
                gt2["_w_end"] = pd.to_datetime(
                    gt2["event_end"], utc=True
                ) + pd.Timedelta(minutes=30)
                host_to_events = {}
                for _, g in gt2.iterrows():
                    key = g.get("hostname", "*") or "*"
                    host_to_events.setdefault(key, []).append(
                        (g["_w_start"], g["_w_end"])
                    )
            wild = host_to_events.get("*", [])
            ep_labels = np.zeros(len(episodes), dtype=np.int8)
            ep_starts = (
                pd.to_datetime(episodes["episode_start"], utc=True)
                .dt.tz_convert(None)
                .to_numpy(dtype="datetime64[ns]")
            )
            ep_ends = (
                pd.to_datetime(episodes["episode_end"], utc=True)
                .dt.tz_convert(None)
                .to_numpy(dtype="datetime64[ns]")
            )
            hosts = episodes["hostname"].to_numpy()

            def naive64(t):
                t = pd.Timestamp(t)
                if t.tz is not None:
                    t = t.tz_convert(None)
                return t.to_datetime64()

            h2e_naive = {
                h: [(naive64(gs), naive64(ge)) for gs, ge in lst]
                for h, lst in host_to_events.items()
            }
            wild_naive = h2e_naive.get("*", [])
            for i in range(len(episodes)):
                s = ep_starts[i]
                e = ep_ends[i]
                cands = h2e_naive.get(hosts[i], []) + wild_naive
                for gs, ge in cands:
                    if s <= ge and e >= gs:
                        ep_labels[i] = 1
                        break
            if ep_labels.sum() > 0 and ep_labels.sum() < len(ep_labels):
                if score_col is not None:
                    scores = episodes["max_score"].fillna(0).to_numpy()
                    score_mode = "max_raw_score"
                else:
                    scores = episodes["n_firings"].fillna(0).to_numpy().astype(float)
                    score_mode = "n_firings"
                result["auroc_episode"] = round(
                    float(roc_auc_score(ep_labels, scores)), 4
                )
                result["auprc_episode"] = round(
                    float(average_precision_score(ep_labels, scores)), 4
                )
                result["auc_score_mode"] = score_mode
                result["n_positive_episodes"] = int(ep_labels.sum())
                result["n_total_episodes"] = int(len(ep_labels))
                result["positive_rate"] = round(float(ep_labels.mean()), 4)

                order = np.argsort(-scores, kind="stable")
                labels_sorted = ep_labels[order]
                for k in (10, 50, 100, 200):
                    if k <= len(labels_sorted):
                        result[f"precision_at_{k}"] = round(
                            float(labels_sorted[:k].sum()) / k, 4
                        )
        except Exception:
            pass

    return result


DETECTOR_SPEC: list[tuple[str, str, Optional[str]]] = [
    # (display_name, flag_col, score_col)
    ("threshold", "thr_flag", None),
    ("isolation_forest", "if_is_anomaly", "if_anomaly_score"),
    ("zscore", "zscore_flag", None),
    ("ewma", "ewma_flag", None),
    ("lstm_ae", "lstm_is_anomaly", "lstm_recon_z"),
    ("p3_physics", "physics_anomaly", "physics_z"),
    ("p3_const3_dyn", "const3_dynamics", None),
    ("p3_const4_xplane", "const4_crossplane", None),
    ("phase2_job", "p2_job_anomaly", None),
    ("p2_coherence", "coherence_anomaly", "coherence_z"),
    ("peer_minute", "peer_anomaly_pm", "peer_z_pm"),
]


# Build per-row cascade and ensemble-vote flags across the three phases.
def compute_cascade_flags(fused: pd.DataFrame) -> dict[str, pd.Series]:
    p1_any = pd.Series(False, index=fused.index)
    for c in (
        "thr_flag",
        "if_is_anomaly",
        "zscore_flag",
        "ewma_flag",
        "lstm_is_anomaly",
    ):
        if c in fused.columns:
            p1_any = p1_any | fused[c].astype(bool)

    p2_any = pd.Series(False, index=fused.index)
    for c in ("p2_job_anomaly", "coherence_anomaly", "peer_anomaly_pm"):
        if c in fused.columns:
            p2_any = p2_any | fused[c].astype(bool)

    p3_any = pd.Series(False, index=fused.index)
    for c in ("physics_anomaly", "const3_dynamics", "const4_crossplane"):
        if c in fused.columns:
            p3_any = p3_any | fused[c].astype(bool)

    return {
        "cascade_p1_any": p1_any,
        "cascade_p2_any": p2_any,
        "cascade_p3_any": p3_any,
        "cascade_p1_p2": (p1_any & p2_any),
        "cascade_p1_p2_p3": (p1_any & p2_any & p3_any),
        "ensemble_vote_any": (p1_any | p2_any | p3_any),
        "ensemble_vote_2of3": (
            (p1_any.astype(int) + p2_any.astype(int) + p3_any.astype(int)) >= 2
        ),
        "ensemble_vote_all3": (p1_any & p2_any & p3_any),
    }


# Evaluate every detector, cascade, and ensemble per split and component.
def per_detector_eval(fused: pd.DataFrame, gt_events: pd.DataFrame) -> dict:
    # Add cascade flags to a working copy.
    work = fused.copy()
    cascades = compute_cascade_flags(work)
    for k, v in cascades.items():
        work[k] = v

    # Detector list: individual + cascades + ensembles.
    all_detectors = [
        *DETECTOR_SPEC,
        ("cascade_p1_any", "cascade_p1_any", None),
        ("cascade_p2_any", "cascade_p2_any", None),
        ("cascade_p3_any", "cascade_p3_any", None),
        ("cascade_p1_p2", "cascade_p1_p2", None),
        ("cascade_p1_p2_p3", "cascade_p1_p2_p3", None),
        ("ensemble_any", "ensemble_vote_any", None),
        ("ensemble_2of3", "ensemble_vote_2of3", None),
        ("ensemble_all3", "ensemble_vote_all3", None),
    ]

    components = [None, "zen4", "h100"]  # None = "all"
    splits = ["train", "val", "test", None]  # None = "pooled"
    out: dict = {}
    split_label = {"train": "train", "val": "val", "test": "test", None: "pooled"}
    comp_label = {None: "all", "zen4": "zen4", "h100": "h100"}

    # Pre-cache the (host, component) index so we don't recompute it 12x.
    host_comp_all = (
        work[["hostname", "component"]].drop_duplicates()
        if "component" in work.columns
        else None
    )

    for split in splits:
        sk = split_label[split]
        out[sk] = {}
        for comp in components:
            ck = comp_label[comp]
            sub = work
            if split is not None and "split" in work.columns:
                sub = sub[sub["split"] == split]
            if comp is not None and "component" in work.columns:
                sub = sub[sub["component"] == comp]

            gt_sub = gt_events.copy()
            if split is not None and "split" in gt_sub.columns:
                gt_sub = gt_sub[gt_sub["split"] == split]
            if comp is not None and host_comp_all is not None and len(gt_sub) > 0:
                hosts_in_comp = set(
                    host_comp_all[host_comp_all["component"] == comp]["hostname"]
                )
                gt_sub = gt_sub[
                    gt_sub["hostname"].isin(hosts_in_comp)
                    | gt_sub["hostname"].isna()
                    | (gt_sub["hostname"] == "*")
                ]

            # Per-row GT mask (for pointwise AUROC) — vectorized binary-search.
            s_all_frame = sub[["timestamp", "hostname"]].copy()
            s_all_frame["timestamp"] = pd.to_datetime(
                s_all_frame["timestamp"], utc=True
            )
            pos_mask = ground_truth_row_membership_mask(s_all_frame, gt_sub)

            # host_to_events (for episode-level AUROC).
            gt2 = gt_sub.copy()
            gt2["_w_start"] = pd.to_datetime(
                gt2["event_start"], utc=True
            ) - pd.Timedelta(minutes=60)
            gt2["_w_end"] = pd.to_datetime(gt2["event_end"], utc=True) + pd.Timedelta(
                minutes=30
            )
            host_to_events: dict = {}
            for _, g in gt2.iterrows():
                key = g.get("hostname", "*") or "*"
                host_to_events.setdefault(key, []).append((g["_w_start"], g["_w_end"]))

            shared = {
                "sub_frame": sub,
                "pos_mask": pos_mask,
                "host_to_events": host_to_events,
                "gt_sub": gt_sub,
            }

            rows = []
            for name, flag_col, score_col in all_detectors:
                row = evaluate_single_detector(
                    work,
                    gt_events,
                    flag_col,
                    score_col,
                    split=split,
                    component=comp,
                    shared=shared,
                )
                row["name"] = name
                rows.append(row)
            out[sk][ck] = rows
    return out


# Evaluate the fused GBDT as a single detector per split and component.
def evaluate_gbdt_as_detector(
    scored_enriched: pd.DataFrame, gt_events: pd.DataFrame, bundle
) -> dict:
    if len(scored_enriched) == 0:
        return {}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        return {}

    rows: dict = {}
    for split in ("train", "val", "test", None):
        sk = "pooled" if split is None else split
        for comp in (None, "zen4", "h100"):
            ck = "all" if comp is None else comp
            sub = scored_enriched
            if split is not None and "split" in sub.columns:
                sub = sub[sub["split"] == split]
            if comp is not None and "component" in sub.columns:
                sub = sub[sub["component"] == comp]
            if len(sub) < 5:
                continue

            gt_sub = gt_events
            if split is not None and "split" in gt_events.columns:
                gt_sub = gt_events[gt_events["split"] == split]
            if comp is not None and "hostname" in gt_sub.columns:
                host_comp = scored_enriched[["hostname", "component"]].drop_duplicates()
                hosts_in = set(host_comp[host_comp["component"] == comp]["hostname"])
                gt_sub = gt_sub[
                    gt_sub["hostname"].isin(hosts_in)
                    | (gt_sub["hostname"].isna())
                    | (gt_sub["hostname"] == "*")
                ]
            if len(gt_sub) == 0:
                continue

            # Build AUROC-ready tuples (score = fusion_prob, label = overlap GT).
            ep = sub[["hostname", "fusion_prob", "episode_start", "episode_end"]].copy()
            ep["episode_start"] = pd.to_datetime(ep["episode_start"], utc=True)
            ep["episode_end"] = pd.to_datetime(ep["episode_end"], utc=True)
            gt2 = gt_sub.copy()
            gt2["_w_start"] = pd.to_datetime(
                gt2["event_start"], utc=True
            ) - pd.Timedelta(minutes=60)
            gt2["_w_end"] = pd.to_datetime(gt2["event_end"], utc=True) + pd.Timedelta(
                minutes=30
            )
            host_to_events: dict = {}
            for _, g in gt2.iterrows():
                key = g.get("hostname", "*") or "*"
                host_to_events.setdefault(key, []).append((g["_w_start"], g["_w_end"]))
            wild = host_to_events.get("*", [])

            labels = np.zeros(len(ep), dtype=np.int8)
            scores = ep["fusion_prob"].fillna(0).to_numpy()
            for i, r in enumerate(ep.itertuples(index=False)):
                cands = host_to_events.get(r.hostname, []) + wild
                for gs, ge in cands:
                    if r.episode_start <= ge and r.episode_end >= gs:
                        labels[i] = 1
                        break

            row: dict = {
                "detector": "fused_high_v2",
                "name": "fused_high_v2 (GBDT)",
                "n_deduped_alerts": int(len(ep)),
            }
            if labels.sum() > 0 and labels.sum() < len(labels):
                row["auroc_episode"] = round(float(roc_auc_score(labels, scores)), 4)
                row["auprc_episode"] = round(
                    float(average_precision_score(labels, scores)), 4
                )
                row["auc_score_mode"] = "fusion_prob"
                row["n_positive_episodes"] = int(labels.sum())
                row["n_total_episodes"] = int(len(labels))
                row["positive_rate"] = round(float(labels.mean()), 4)
                order = np.argsort(-scores, kind="stable")
                labels_sorted = labels[order]
                for k in (10, 50, 100, 200):
                    if k <= len(labels_sorted):
                        row[f"precision_at_{k}"] = round(
                            float(labels_sorted[:k].sum()) / k, 4
                        )

            # TP/FP/FN at the CONFIRMED tier.
            t = bundle.t_confirmed
            pos_ep = sub[sub["fusion_prob"] >= t].copy()
            pos_ep["episode_start"] = pd.to_datetime(pos_ep["episode_start"], utc=True)
            pos_ep["episode_end"] = pd.to_datetime(pos_ep["episode_end"], utc=True)
            tp, fp, fn, _, _ = event_level_match(pos_ep, gt_sub)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            rr = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
            row.update(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(p, 4),
                    "recall": round(rr, 4),
                    "f1": round(f1, 4),
                }
            )
            if split is not None:
                row["split"] = split
            if comp is not None:
                row["component"] = comp

            rows[(sk, ck)] = row
    return rows
