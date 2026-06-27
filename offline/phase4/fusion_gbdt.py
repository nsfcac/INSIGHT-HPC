from __future__ import annotations

from offline.phase4.fusion_gbdt_module.constants import *
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Trained GBDT plus its tier thresholds and calibration/training metadata.
@dataclass
class FusionBundle:
    clf: "HistGradientBoostingClassifier"
    t_critical: float
    t_confirmed: float
    t_candidate: float
    feature_columns: list[str]
    calibration_stats: dict  # thresholds' val-set P/R/F1 at selection time
    training_stats: dict  # n positives, n negatives, CV-F1, etc.


#   Candidate episode generation

CANDIDATE_MIN_PEAK = 0.7  # the *top* phase_score must exceed this
CANDIDATE_MIN_RUN = 2  # 3→2: short hardware failures (1-4 min telemetry-stops)
CANDIDATE_MERGE_GAP = 10  # 5→10: short noisy events bridge into episodes


# Form candidate anomaly episodes from runs of high phase scores per node.
def build_candidate_episodes(fused: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in fused.columns or len(fused) == 0:
        return pd.DataFrame(
            columns=[
                "hostname",
                "component",
                "episode_start",
                "episode_end",
                "ep_duration_min",
                "ep_peak_phase1",
                "ep_peak_phase3",
                "ep_consensus_phases",
            ]
        )

    df = fused.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["hostname", "component", "timestamp"]).reset_index(drop=True)

    p1 = df.get("phase1_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    p2 = df.get("phase2_score", pd.Series(0.0, index=df.index)).fillna(0.0)
    p3 = df.get("phase3_score", pd.Series(0.0, index=df.index)).fillna(0.0)

    active = (
        np.maximum.reduce([p1.to_numpy(), p2.to_numpy(), p3.to_numpy()])
        >= CANDIDATE_MIN_PEAK
    )

    # Group into consecutive-run segments per (host, component).
    grp = df.groupby(["hostname", "component"], sort=False)
    episodes = []
    for (host, comp), idx in grp.indices.items():
        idx = np.sort(idx)
        if len(idx) == 0:
            continue
        active_slice = active[idx]
        if not active_slice.any():
            continue

        # Find runs of True.
        in_run = False
        run_start = None
        runs = []
        for i, a in enumerate(active_slice):
            if a and not in_run:
                in_run = True
                run_start = i
            elif not a and in_run:
                in_run = False
                if i - run_start >= CANDIDATE_MIN_RUN:
                    runs.append((run_start, i - 1))
        if in_run and (len(active_slice) - run_start) >= CANDIDATE_MIN_RUN:
            runs.append((run_start, len(active_slice) - 1))

        # Merge runs separated by <= CANDIDATE_MERGE_GAP minutes.
        merged = []
        for s, e in runs:
            if merged and s - merged[-1][1] <= CANDIDATE_MERGE_GAP:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        for s_local, e_local in merged:
            s_idx, e_idx = idx[s_local], idx[e_local]
            ts_start = df["timestamp"].iloc[s_idx]
            ts_end = df["timestamp"].iloc[e_idx]
            # Episode aggregates: peak phase scores and phase consensus.
            window = df.iloc[s_idx : e_idx + 1]
            peak_1 = (
                float(window["phase1_score"].max()) if "phase1_score" in window else 0.0
            )
            peak_3 = (
                float(window["phase3_score"].max()) if "phase3_score" in window else 0.0
            )
            peak_2 = (
                float(window["phase2_score"].max()) if "phase2_score" in window else 0.0
            )
            consensus = int((peak_1 >= 1.0) + (peak_2 >= 1.0) + (peak_3 >= 0.3))
            duration = int((ts_end - ts_start).total_seconds() // 60) + 1
            episodes.append(
                {
                    "hostname": host,
                    "component": comp,
                    "episode_start": ts_start,
                    "episode_end": ts_end,
                    "ep_duration_min": duration,
                    "ep_peak_phase1": peak_1,
                    "ep_peak_phase3": peak_3,
                    "ep_consensus_phases": consensus,
                }
            )

    if not episodes:
        return pd.DataFrame(
            columns=[
                "hostname",
                "component",
                "episode_start",
                "episode_end",
                "ep_duration_min",
                "ep_peak_phase1",
                "ep_peak_phase3",
                "ep_consensus_phases",
            ]
        )
    return (
        pd.DataFrame(episodes)
        .sort_values(["hostname", "component", "episode_start"])
        .reset_index(drop=True)
    )


# Build a sorted index of the fused table for O(log N) per-episode window lookups.
def build_fused_index(fused: pd.DataFrame) -> dict:
    f = fused.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
    f = f.sort_values(["hostname", "component", "timestamp"]).reset_index(drop=True)
    # Contiguous (host, component) ranges: detect transitions in one pass.
    hosts = f["hostname"].to_numpy()
    comps = f["component"].to_numpy()
    n = len(f)
    ranges: dict[tuple[str, str], tuple[int, int]] = {}
    if n == 0:
        return {
            "sorted": f,
            "ranges": ranges,
            "ts_values": np.array([], dtype="datetime64[ns]"),
        }
    change = np.ones(n, dtype=bool)
    change[1:] = (hosts[1:] != hosts[:-1]) | (comps[1:] != comps[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    ranges = {
        (hosts[s], comps[s]): (int(s), int(e))
        for s, e in zip(starts.tolist(), ends.tolist())
    }
    # Convert once so searchsorted on ts is on a contiguous datetime64 array.
    ts_values = f["timestamp"].to_numpy().astype("datetime64[ns]")
    return {"sorted": f, "ranges": ranges, "ts_values": ts_values}


# Return the row-index range of one episode's window within the fused index.
def window_slice(
    index: dict, host: str, comp: str, ts_start, ts_end
) -> tuple[int, int]:
    key = (host, comp)
    rng = index["ranges"].get(key)
    if rng is None:
        return (0, 0)
    s, e = rng
    ts = index["ts_values"][s:e]
    # Accept pandas Timestamps or numpy datetime64 inputs.
    ts_start_n = (
        np.datetime64(pd.Timestamp(ts_start).tz_convert("UTC").tz_localize(None), "ns")
        if hasattr(pd.Timestamp(ts_start), "tz_convert")
        and pd.Timestamp(ts_start).tz is not None
        else np.datetime64(pd.Timestamp(ts_start), "ns")
    )
    ts_end_n = (
        np.datetime64(pd.Timestamp(ts_end).tz_convert("UTC").tz_localize(None), "ns")
        if hasattr(pd.Timestamp(ts_end), "tz_convert")
        and pd.Timestamp(ts_end).tz is not None
        else np.datetime64(pd.Timestamp(ts_end), "ns")
    )
    lo_rel = np.searchsorted(ts, ts_start_n, side="left")
    hi_rel = np.searchsorted(ts, ts_end_n, side="right")
    return (s + lo_rel, s + hi_rel)


# Aggregate per-window fused columns into the GBDT feature set for each episode.
def attach_episode_features(
    candidates: pd.DataFrame, fused: pd.DataFrame, fused_index: Optional[dict] = None
) -> pd.DataFrame:
    if len(candidates) == 0:
        return pd.DataFrame(
            columns=["hostname", "component", "episode_start", "episode_end"]
            + FEATURE_COLUMNS
        )

    if fused_index is None:
        fused_index = build_fused_index(fused)
    sorted_fused = fused_index["sorted"]

    out = candidates.copy()
    # Diurnal from episode_start.
    hr = pd.to_datetime(out["episode_start"], utc=True).dt.hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hr / 24.0).astype(np.float32)
    out["hour_cos"] = np.cos(2 * np.pi * hr / 24.0).astype(np.float32)

    peak_cols = {"phase1_score", "phase2_score", "phase3_score", "lstm_recon_z"}
    mean_cols = {
        "phase1_n_detectors_firing",
        "phase1_persistence_min",
        "phase2_cluster_dist",
        "phase2_peer_divergence_z",
        "phase3_n_constraints",
        "phase3_context_score",
    }
    any_cols = {
        "if_is_anomaly_rel",
        "strong_stat_consensus",
        "coherence_anomaly",
        "physics_anomaly",
        "const3_dynamics",
        "const4_crossplane",
    }
    max_mag_cols = {"phase3_physics_z"}

    want = [
        c
        for c in FEATURE_COLUMNS
        if c in sorted_fused.columns
        and c
        not in (
            "hour_sin",
            "hour_cos",
            "ep_duration_min",
            "ep_peak_phase1",
            "ep_peak_phase3",
            "ep_consensus_phases",
        )
    ]
    col_arrays = {c: sorted_fused[c].to_numpy() for c in want}

    feat_rows = []
    for _, row in out.iterrows():
        lo, hi = window_slice(
            fused_index,
            row["hostname"],
            row["component"],
            row["episode_start"],
            row["episode_end"],
        )
        if hi <= lo:
            feat_rows.append({c: 0.0 for c in want})
            continue
        agg = {}
        for c in want:
            arr = col_arrays[c][lo:hi]
            if len(arr) == 0:
                agg[c] = 0.0
            elif c in peak_cols:
                agg[c] = float(np.max(arr))
            elif c in any_cols:
                agg[c] = float(np.any(arr.astype(bool)))
            elif c in max_mag_cols:
                abs_arr = np.abs(arr)
                if len(abs_arr) == 0:
                    agg[c] = 0.0
                else:
                    idx = int(np.argmax(abs_arr))
                    agg[c] = float(arr[idx])
            elif c in mean_cols:
                agg[c] = float(np.mean(arr))
            else:
                agg[c] = float(np.mean(arr))
        feat_rows.append(agg)
    feat_df = pd.DataFrame(feat_rows, index=out.index)
    for c in want:
        out[c] = feat_df[c].astype(np.float32)

    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            default = 0.5 if c.startswith("phase2") else 0.0
            out[c] = pd.Series(
                np.full(len(out), default, dtype=np.float32), index=out.index
            )

    return out


#   Episode labeling (GT matching)

EVENT_LEAD_MIN = 60
EVENT_LAG_MIN = 30


# Label episodes positive when they overlap a GT event (with lead/lag and rack fan-out).
def label_episodes(episodes: pd.DataFrame, gt_events: pd.DataFrame) -> pd.Series:
    if len(episodes) == 0:
        return pd.Series([], dtype=np.int8)
    if gt_events is None or len(gt_events) == 0:
        return pd.Series(np.zeros(len(episodes), dtype=np.int8), index=episodes.index)

    gt = gt_events.copy()
    gt["event_start"] = pd.to_datetime(gt["event_start"], utc=True) - pd.Timedelta(
        minutes=EVENT_LEAD_MIN
    )
    gt["event_end"] = pd.to_datetime(gt["event_end"], utc=True) + pd.Timedelta(
        minutes=EVENT_LAG_MIN
    )

    labels = np.zeros(len(episodes), dtype=np.int8)
    eps = episodes.reset_index(drop=True)

    # Build a per-host lookup for speed.
    host_to_events: dict[str, list[tuple]] = {}
    rack_fanout: dict[str, list[tuple]] = {}
    for _, g in gt.iterrows():
        key = g.get("hostname", "*") or "*"
        host_to_events.setdefault(key, []).append((g["event_start"], g["event_end"]))
        if isinstance(key, str) and key.startswith("pdu-"):
            # Extract rack id from "pdu-XX-Y" → "XX"
            parts = key.split("-")
            if len(parts) >= 2:
                rack = parts[1]
                rack_fanout.setdefault(rack, []).append(
                    (g["event_start"], g["event_end"])
                )

    wildcard_events = host_to_events.get("*", [])

    for i, row in eps.iterrows():
        host = row["hostname"]
        s = pd.to_datetime(row["episode_start"], utc=True)
        e = pd.to_datetime(row["episode_end"], utc=True)
        candidates = host_to_events.get(host, []) + wildcard_events
        # If this episode's host is in a rack with PDU-host GT events, pick those up too.
        if isinstance(host, str) and (
            host.startswith("rpc-") or host.startswith("rpg-")
        ):
            parts = host.split("-")
            if len(parts) >= 2:
                rack = parts[1]
                if rack in rack_fanout:
                    candidates = candidates + rack_fanout[rack]
        for gs, ge in candidates:
            if s <= ge and e >= gs:
                labels[i] = 1
                break
    return pd.Series(labels, index=eps.index, dtype=np.int8)


# Train the HistGBDT episode classifier and calibrate its tier thresholds on the val split.
def train_fusion_gbdt(
    episodes_train_val: pd.DataFrame,
    labels_train_val: pd.Series,
    split_tag: pd.Series,
    gt_events: Optional[pd.DataFrame] = None,
    incumbent_bundle=None,
    rng_seed: int = 42,
    floor_p: float = 0.85,
    floor_r: float = 0.0,
    floor_f1: float = 0.0,
    use_incumbent_ratchet: bool = True,
    crit_target_p: float = 0.90,
    crit_fallback: float = 0.85,
    cand_target_r: float = 0.85,
    cand_floor_p: float = 0.50,
    cand_fallback: float = 0.30,
) -> FusionBundle:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for fusion_gbdt")

    X = episodes_train_val[FEATURE_COLUMNS].astype(np.float32).to_numpy()
    y = labels_train_val.astype(np.int8).to_numpy()

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 5 or n_neg < 5:
        raise RuntimeError(
            f"[fusion_gbdt] Too few train+val episodes to fit a classifier: "
            f"positives={n_pos}, negatives={n_neg}. Check candidate generation "
            f"(fusion_gbdt.CANDIDATE_MIN_PEAK) and GT label coverage."
        )

    use_early_stop = n_pos >= 30

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400 if use_early_stop else 200,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=use_early_stop,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=rng_seed,
    )
    t0 = time.time()
    clf.fit(X, y)
    train_time = time.time() - t0

    # Calibrate tier thresholds on val episodes.
    val_mask = split_tag.astype(str).to_numpy() == "val"
    val_idx = np.where(val_mask)[0]
    if len(val_idx) < 10:
        # Not enough val; fall back to train for calibration. Log it.
        val_idx = np.arange(len(X))

    val_probs = clf.predict_proba(X[val_idx])[:, 1]
    val_y = y[val_idx]

    use_nab = (
        gt_events is not None
        and "hostname" in episodes_train_val.columns
        and "episode_start" in episodes_train_val.columns
    )

    calibration_stats: dict = {
        "val": {"n_val_episodes": int(len(val_idx)), "val_positives": int(val_y.sum())}
    }

    if use_nab:
        val_eps_df = episodes_train_val.iloc[val_idx].reset_index(drop=True).copy()
        val_eps_df["fusion_prob"] = val_probs
        gt_val = (
            gt_events[gt_events["split"] == "val"]
            if "split" in gt_events.columns
            else gt_events.copy()
        )

        incumbent_metrics = incumbent_validation_metrics(
            incumbent_bundle, val_eps_df, gt_val
        )
        eff_floor_p = floor_p
        eff_floor_r = floor_r
        eff_floor_f1 = floor_f1
        if incumbent_metrics is not None and use_incumbent_ratchet:
            old_threshold, precision_value, recall_value, f1_value = incumbent_metrics
            eff_floor_p = max(eff_floor_p, precision_value)
            eff_floor_r = max(eff_floor_r, recall_value)
            eff_floor_f1 = max(eff_floor_f1, f1_value)
            calibration_stats["val"]["incumbent"] = {
                "t": old_threshold,
                "val_P": round(precision_value, 4),
                "val_R": round(recall_value, 4),
                "val_F1": round(f1_value, 4),
                "ratchet_applied": True,
            }
        elif incumbent_metrics is not None:
            old_threshold, precision_value, recall_value, f1_value = incumbent_metrics
            calibration_stats["val"]["incumbent"] = {
                "t": old_threshold,
                "val_P": round(precision_value, 4),
                "val_R": round(recall_value, 4),
                "val_F1": round(f1_value, 4),
                "ratchet_applied": False,
            }

        t_conf, conf_info = find_threshold_confirmed_nab(
            val_eps_df,
            gt_val,
            floor_p=eff_floor_p,
            floor_r=eff_floor_r,
            floor_f1=eff_floor_f1,
            incumbent_t=bundle_attribute(incumbent_bundle, "t_confirmed"),
        )
        t_crit = find_threshold_precision_nab(
            val_eps_df,
            gt_val,
            target_p=crit_target_p,
            fallback=crit_fallback,
        )
        t_cand = find_threshold_recall_nab(
            val_eps_df,
            gt_val,
            target_r=cand_target_r,
            floor_p=cand_floor_p,
            fallback=cand_fallback,
        )

        _, _, _, p_conf, r_conf, f_conf = nab_precision_recall_at(
            val_eps_df, gt_val, t_conf
        )
        _, _, _, p_crit, r_crit, f_crit = nab_precision_recall_at(
            val_eps_df, gt_val, t_crit
        )
        _, _, _, p_cand, r_cand, f_cand = nab_precision_recall_at(
            val_eps_df, gt_val, t_cand
        )

        calibration_stats["val"]["calibration_mode"] = "nab_event_level"
        calibration_stats["val"]["floors"] = {
            "val_P": round(eff_floor_p, 4),
            "val_R": round(eff_floor_r, 4),
            "val_F1": round(eff_floor_f1, 4),
        }
        calibration_stats["val"]["t_confirmed_reason"] = conf_info["reason"]
    else:
        # Cold start / no GT — legacy raw-ep calibration.
        t_conf = find_threshold_best_f1(val_probs, val_y)
        t_crit = find_threshold_for_precision(
            val_probs, val_y, target_p=crit_target_p, fallback=crit_fallback
        )
        t_cand = find_threshold_for_recall(
            val_probs, val_y, target_r=cand_target_r, fallback=cand_fallback
        )

        p_conf, r_conf, f_conf = precision_recall_at(val_probs, val_y, t_conf)
        p_crit, r_crit, f_crit = precision_recall_at(val_probs, val_y, t_crit)
        p_cand, r_cand, f_cand = precision_recall_at(val_probs, val_y, t_cand)
        calibration_stats["val"]["calibration_mode"] = "legacy_raw_ep"

    calibration_stats["val"]["t_critical"] = (t_crit, p_crit, r_crit, f_crit)
    calibration_stats["val"]["t_confirmed"] = (t_conf, p_conf, r_conf, f_conf)
    calibration_stats["val"]["t_candidate"] = (t_cand, p_cand, r_cand, f_cand)

    training_stats = {
        "n_episodes": int(len(y)),
        "n_positives": int(y.sum()),
        "n_negatives": int((1 - y).sum()),
        "train_time_s": round(train_time, 2),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    return FusionBundle(
        clf=clf,
        t_critical=t_crit,
        t_confirmed=t_conf,
        t_candidate=t_cand,
        feature_columns=list(FEATURE_COLUMNS),
        calibration_stats=calibration_stats,
        training_stats=training_stats,
    )


from offline.phase4.fusion_gbdt_module.thresholds_audit import *
