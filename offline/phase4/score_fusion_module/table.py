from __future__ import annotations

from offline.phase4.score_fusion_module.constants import *
import os, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import bool_series, load_config, save_parquet
from offline.phase4.score_fusion_module.loaders import (
    load_baseline_alerts,
    load_coherence,
    load_constraint_violations,
    load_if_scores,
    load_job_context,
    load_lstm_scores,
    load_per_minute_coherence,
    load_phase2_scores,
    load_physics_scores,
    load_streaming_coherence,
    load_threshold_alerts,
)

JOB_CONTEXT_SCORE = {
    "JOB_EXPLAINED": 0.0,
    "AMBIGUOUS": 0.5,
    "COOLING_DEMAND": 0.6,
    "MEASUREMENT_DISCREPANCY": 0.7,
    "SENSOR_DRIFT": 0.8,
    "JOB_OVER_EXPECTATION": 0.9,  # power>>profile OR failed/timeout job + anomaly
    "COOLING_FAULT": 1.0,
    "INFRA_FAULT": 1.0,
    "NO_JOBS": 0.8,
}


# Merge all detector outputs into one per-minute fused table with vote features.
def build_fusion_table(
    if_df,
    thr_df,
    phys_df,
    const_df,
    p2_df,
    coh_df,
    lstm_df=None,
    jctx_df=None,
    zscore_df=None,
    ewma_df=None,
    pm_coh_df=None,
    streaming_coh_df=None,
):
    # Expand phase-2 job segments into per-minute rows.
    def expand_phase2_minutes(p2_src: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if p2_src is None or p2_src.empty:
            return None
        seg_cols = [
            "hostname",
            "seg_start",
            "seg_end",
            "cluster_dist",
            "job_if_anomaly",
            "duration_min",
            "is_multi_job",
        ]
        for extra in ["component", "split"]:
            if extra in p2_src.columns:
                seg_cols.append(extra)
        p2s = p2_src[seg_cols].copy()
        p2s["seg_start"] = pd.to_datetime(p2s["seg_start"], utc=True).dt.floor("1min")
        p2s["seg_end"] = pd.to_datetime(p2s["seg_end"], utc=True).dt.floor("1min")
        dur_minutes = (
            (p2s["seg_end"] - p2s["seg_start"])
            .dt.total_seconds()
            .div(60)
            .add(1)
            .clip(lower=1)
            .astype(int)
        )
        idx = np.repeat(np.arange(len(p2s)), dur_minutes.values)
        p2m_local = p2s.iloc[idx].reset_index(drop=True)
        offsets = np.concatenate([np.arange(n) for n in dur_minutes.values])
        p2m_local[TS] = p2m_local["seg_start"] + pd.to_timedelta(offsets, unit="min")
        p2m_local["minutes_into_job"] = offsets.astype(np.float32)
        p2m_local = p2m_local.rename(
            columns={"job_if_anomaly": "p2_job_anomaly", "duration_min": "job_duration"}
        )
        p2m_local["cluster_dist"] = p2m_local["cluster_dist"].astype("float32")
        p2m_local["p2_job_anomaly"] = bool_series(
            p2m_local["p2_job_anomaly"], index=p2m_local.index
        )
        p2m_local["job_duration"] = p2m_local["job_duration"].astype("float32")
        p2m_local["is_multi_job"] = bool_series(
            p2m_local["is_multi_job"], index=p2m_local.index
        )
        p2m_local = p2m_local.sort_values(
            "cluster_dist", ascending=False
        ).drop_duplicates(subset=[TS, "hostname"], keep="first")
        return p2m_local

    # Reduce a detector frame to its key columns (timestamp, host, component, split).
    def key_frame(
        src: Optional[pd.DataFrame], ts_col: str = TS
    ) -> Optional[pd.DataFrame]:
        if (
            src is None
            or src.empty
            or ts_col not in src.columns
            or "hostname" not in src.columns
        ):
            return None
        cols = [ts_col, "hostname"] + [
            c for c in ["component", "split", "maintenance_flag"] if c in src.columns
        ]
        out = src[cols].copy()
        out = out.rename(columns={ts_col: TS})
        out[TS] = pd.to_datetime(out[TS], utc=True).dt.floor("1min")
        if "component" not in out.columns:
            out["component"] = pd.NA
        if "split" not in out.columns:
            out["split"] = pd.NA
        if "maintenance_flag" not in out.columns:
            out["maintenance_flag"] = False
        return out[[TS, "hostname", "component", "split", "maintenance_flag"]]

    p2m = expand_phase2_minutes(p2_df)
    base_parts = [
        key_frame(if_df),
        key_frame(thr_df),
        key_frame(zscore_df),
        key_frame(ewma_df),
        key_frame(phys_df),
        key_frame(const_df),
        key_frame(lstm_df),
        key_frame(jctx_df),
        key_frame(p2m),
    ]
    base_parts = [b for b in base_parts if b is not None and not b.empty]
    if not base_parts:
        print("[fusion] No detector sources available — cannot build fusion table")
        return pd.DataFrame()

    base = pd.concat(base_parts, ignore_index=True)
    base = base.groupby([TS, "hostname"], as_index=False, sort=False).agg(
        {
            "component": "first",
            "split": "first",
            "maintenance_flag": "max",
        }
    )

    # Left-merge selected detector columns onto the base table.
    def merge_left(src, extra_cols, rename=None):
        nonlocal base
        if src is None:
            return
        src_cols = [c for c in extra_cols if c in src.columns]
        if not src_cols:
            return
        s = src[[TS, "hostname"] + src_cols].copy()
        s[TS] = pd.to_datetime(s[TS], utc=True).dt.floor("1min")
        if rename:
            s = s.rename(columns=rename)
        base = base.merge(s, on=[TS, "hostname"], how="left")

    if if_df is not None:
        extra_base = [
            c
            for c in ["if_anomaly_score_rel", "if_is_anomaly_rel"]
            if c in if_df.columns
        ]
        merge_left(if_df, ["if_anomaly_score", "if_is_anomaly"] + extra_base)
    if "if_anomaly_score" not in base.columns:
        base["if_anomaly_score"] = np.nan
    if "if_is_anomaly" not in base.columns:
        base["if_is_anomaly"] = False
    if "maintenance_flag" not in base.columns:
        base["maintenance_flag"] = False

    threshold_extra_columns = [
        c
        for c in (thr_df.columns if thr_df is not None else [])
        if c.startswith("rule_")
    ]
    merge_left(
        thr_df,
        ["is_flagged"] + threshold_extra_columns,
        rename={"is_flagged": "thr_flag"},
    )
    if "thr_flag" not in base.columns:
        base["thr_flag"] = False

    # Phase-1 statistical baselines (z-score / EWMA)
    merge_left(zscore_df, ["is_flagged"], rename={"is_flagged": "zscore_flag"})
    if "zscore_flag" not in base.columns:
        base["zscore_flag"] = False
    merge_left(ewma_df, ["is_flagged"], rename={"is_flagged": "ewma_flag"})
    if "ewma_flag" not in base.columns:
        base["ewma_flag"] = False

    if os.environ.get("INSIGHT_HPC_ABLATE_ZSCORE_EWMA", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        print(
            "  [ablate] INSIGHT_HPC_ABLATE_ZSCORE_EWMA=1 — "
            "zeroing zscore_flag and ewma_flag for the full fusion run"
        )
        base["zscore_flag"] = False
        base["ewma_flag"] = False

    # D3 — physics residuals
    if phys_df is not None:
        phys_cols = [
            TS,
            "hostname",
            "power_residual_z",
            "thermal_residual_z",
            "physics_anomaly",
        ]
        # Carry through job-transition attenuation factor if available
        if "job_transition_atten" in phys_df.columns:
            phys_cols.append("job_transition_atten")
        p = phys_df[phys_cols].copy()
        p[TS] = pd.to_datetime(p[TS], utc=True).dt.floor("1min")
        p["physics_z"] = p[["power_residual_z", "thermal_residual_z"]].abs().max(axis=1)
        merge_cols = [TS, "hostname", "physics_z", "physics_anomaly"]
        if "job_transition_atten" in p.columns:
            merge_cols.append("job_transition_atten")
        base = base.merge(p[merge_cols], on=[TS, "hostname"], how="left")
    else:
        base["physics_z"] = np.nan
        base["physics_anomaly"] = False

    constraint_extra_columns = [
        c
        for c in [
            "const1_temp_fan",
            "const2_rack_therm",
            "const3_dynamics",
            "const4_crossplane",
            "const5_alloc_idle",
        ]
        if const_df is not None and c in const_df.columns
    ]
    merge_left(const_df, ["n_constraints_violated"] + constraint_extra_columns)
    if "n_constraints_violated" not in base.columns:
        base["n_constraints_violated"] = 0
    for c in [
        "const1_temp_fan",
        "const2_rack_therm",
        "const3_dynamics",
        "const4_crossplane",
        "const5_alloc_idle",
    ]:
        if c not in base.columns:
            base[c] = False

    # D5a — Phase II per-cluster IF (expanded segment → minutes)
    if p2m is not None and not p2m.empty:
        merge_cols = [
            TS,
            "hostname",
            "cluster_dist",
            "p2_job_anomaly",
            "job_duration",
            "is_multi_job",
            "minutes_into_job",
        ]
        merge_cols = [c for c in merge_cols if c in p2m.columns]
        base = base.merge(p2m[merge_cols], on=[TS, "hostname"], how="left")
    for c, v in [
        ("cluster_dist", np.nan),
        ("p2_job_anomaly", False),
        ("job_duration", np.nan),
        ("is_multi_job", False),
        ("minutes_into_job", np.nan),
    ]:
        if c not in base.columns:
            base[c] = v

    base["coherence_anomaly"] = False
    base["coherence_z"] = np.nan

    streaming_loaded = False
    if streaming_coh_df is not None and not streaming_coh_df.empty:
        sc_cols = [
            "hostname",
            "timestamp",
            "streaming_coherence_z",
            "is_streaming_anomaly",
        ]
        sc = streaming_coh_df[
            [c for c in sc_cols if c in streaming_coh_df.columns]
        ].copy()
        if {"hostname", "timestamp"}.issubset(sc.columns):
            sc["timestamp"] = pd.to_datetime(sc["timestamp"], utc=True)
            sc = sc.drop_duplicates(subset=["hostname", "timestamp"])
            base = base.merge(sc, on=["hostname", "timestamp"], how="left")
            if "streaming_coherence_z" in base.columns:
                has_stream = base["streaming_coherence_z"].notna()
                base.loc[has_stream, "coherence_z"] = base.loc[
                    has_stream, "streaming_coherence_z"
                ].astype("float32")
                if "is_streaming_anomaly" in base.columns:
                    base.loc[has_stream, "coherence_anomaly"] = (
                        base.loc[has_stream, "is_streaming_anomaly"]
                        .fillna(False)
                        .astype(bool)
                    )
                n_pop = int(has_stream.sum())
                print(
                    f"  [fusion] streaming coherence: {n_pop:,} per-minute rows "
                    f"populated (per-segment broadcast SKIPPED)"
                )
                streaming_loaded = True
            base = base.drop(
                columns=["streaming_coherence_z", "is_streaming_anomaly"],
                errors="ignore",
            )

    if not streaming_loaded and coh_df is not None and not coh_df.empty:
        anom = coh_df[
            coh_df.get("is_coherence_anomaly", pd.Series(True, index=coh_df.index)).eq(
                True
            )
        ].copy()
        if not anom.empty and "job_id" in anom.columns:
            # Prefer coherence-native segment timing when available.
            has_native_timing = {"seg_start", "seg_end"}.issubset(anom.columns)
            if has_native_timing:
                anom_joined = anom.copy()
                anom_joined["seg_start"] = pd.to_datetime(
                    anom_joined["seg_start"], utc=True
                )
                anom_joined["seg_end"] = pd.to_datetime(
                    anom_joined["seg_end"], utc=True
                )
            elif p2_df is not None and not p2_df.empty:
                # Otherwise recover job windows from Phase II segment metadata.
                job_times = p2_df[["hostname", "job_id", "seg_start", "seg_end"]].copy()
                job_times["seg_start"] = pd.to_datetime(
                    job_times["seg_start"], utc=True
                )
                job_times["seg_end"] = pd.to_datetime(job_times["seg_end"], utc=True)
                anom_joined = anom.merge(
                    job_times, on=["hostname", "job_id"], how="left"
                )
                for col in ["seg_start", "seg_end"]:
                    if col not in anom_joined.columns:
                        left = f"{col}_x"
                        right = f"{col}_y"
                        if left in anom_joined.columns:
                            anom_joined[col] = anom_joined[left]
                        elif right in anom_joined.columns:
                            anom_joined[col] = anom_joined[right]
                    if col in anom_joined.columns:
                        anom_joined[col] = pd.to_datetime(
                            anom_joined[col], utc=True, errors="coerce"
                        )
            else:
                anom_joined = pd.DataFrame()

            if not anom_joined.empty:
                anom_joined = anom_joined.dropna(subset=["seg_start", "seg_end"])

                if not anom_joined.empty:
                    WINDOW = pd.Timedelta(minutes=60)
                    anom_joined["_win_start"] = anom_joined["seg_start"] - WINDOW
                    anom_joined["_win_end"] = anom_joined["seg_end"] + WINDOW
                    anom_joined["_coh_z"] = pd.to_numeric(
                        anom_joined.get("coherence_z", 2.0), errors="coerce"
                    ).fillna(2.0)
                    # For each hostname, flag base rows that fall within any window.
                    for host, grp in anom_joined.groupby("hostname"):
                        host_mask = base["hostname"] == host
                        if not host_mask.any():
                            continue
                        ts_vals = base.loc[host_mask, TS].values  # numpy datetime64
                        in_any_window = np.zeros(ts_vals.shape[0], dtype=bool)
                        max_z = np.full(ts_vals.shape[0], np.nan, dtype=np.float32)
                        for ws, we, cz in zip(
                            grp["_win_start"].values,
                            grp["_win_end"].values,
                            grp["_coh_z"].values,
                        ):
                            hit = (ts_vals >= ws) & (ts_vals <= we)
                            in_any_window |= hit
                            max_z[hit] = np.fmax(max_z[hit], cz)
                        base.loc[host_mask, "coherence_anomaly"] = in_any_window
                        base.loc[host_mask, "coherence_z"] = max_z
            else:
                # No timing metadata available — fall back to hostname-level broadcast.
                print(
                    "  [fusion] WARN: no coherence timing metadata available "
                    "— falling back to hostname broadcast (may inflate D5b coverage)"
                )
                coh_z = anom.groupby("hostname")["coherence_z"].max()
                coh_hosts = set(anom["hostname"].unique())
                base["coherence_anomaly"] = base["hostname"].isin(coh_hosts)
                base["coherence_z"] = base["hostname"].map(coh_z).astype("float32")

    base["coherence_z"] = base["coherence_z"].astype("float32")

    base["peer_anomaly_pm"] = False
    base["peer_z_pm"] = np.nan
    if pm_coh_df is not None and not pm_coh_df.empty:
        pm = pm_coh_df[[TS, "hostname", "is_peer_anomaly", "pwr_z"]].copy()
        pm[TS] = pd.to_datetime(pm[TS], utc=True).dt.floor("1min")
        pm = pm.rename(
            columns={"is_peer_anomaly": "peer_anomaly_pm", "pwr_z": "peer_z_pm"}
        )
        pm["peer_anomaly_pm"] = bool_series(pm["peer_anomaly_pm"], index=pm.index)
        # Keep the strongest |z| when a minute has multiple rows for a host
        pm["_abs_z"] = pm["peer_z_pm"].abs()
        pm = (
            pm.sort_values("_abs_z", ascending=False, na_position="last")
            .drop_duplicates([TS, "hostname"], keep="first")
            .drop(columns=["_abs_z"])
        )
        base = base.drop(columns=["peer_anomaly_pm", "peer_z_pm"])
        base = base.merge(pm, on=[TS, "hostname"], how="left")
        base["peer_anomaly_pm"] = bool_series(base["peer_anomaly_pm"], index=base.index)
        # Fold per-minute peer anomaly into coherence_anomaly (same family).
        base["coherence_anomaly"] = base["coherence_anomaly"] | base["peer_anomaly_pm"]
        # Populate coherence_z when missing: use peer_z_pm magnitude.
        base["coherence_z"] = base["coherence_z"].where(
            base["coherence_z"].notna(),
            base["peer_z_pm"].abs().astype("float32"),
        )
    base["peer_z_pm"] = base["peer_z_pm"].astype("float32")

    # D6 — LSTM-AE temporal reconstruction error
    if lstm_df is not None and not lstm_df.empty:
        lstm_cols = [
            c for c in ["lstm_recon_z", "lstm_is_anomaly"] if c in lstm_df.columns
        ]
        if lstm_cols:
            merge_left(lstm_df, lstm_cols)
    if "lstm_recon_z" not in base.columns:
        base["lstm_recon_z"] = np.nan
    if "lstm_is_anomaly" not in base.columns:
        base["lstm_is_anomaly"] = False

    if jctx_df is not None and not jctx_df.empty:
        jc = jctx_df[[TS, "hostname", "anomaly_context"]].copy()
        jc[TS] = pd.to_datetime(jc[TS], utc=True).dt.floor("1min")
        base = base.merge(jc, on=[TS, "hostname"], how="left")
    if "anomaly_context" not in base.columns:
        base["anomaly_context"] = pd.NA
    base["job_context_score"] = (
        base["anomaly_context"]
        .map(JOB_CONTEXT_SCORE)
        .fillna(0.5)  # rows without physics annotation → neutral
        .astype("float32")
    )

    bin_timestamps = base[TS].dt.floor("5min")
    base["_bin_ts"] = bin_timestamps

    data_quality_fire = pd.Series(False, index=base.index)
    for rc in ["rule_FLATLINE", "rule_DROPOUT", "rule_COLD_INLET_DROP"]:
        if rc in base.columns:
            data_quality_fire = data_quality_fire | bool_series(
                base[rc], index=base.index
            )

    vote_families = {
        "thr": bool_series(
            base.get("thr_flag", pd.Series(False, index=base.index)), index=base.index
        ),
        "if": bool_series(
            base.get("if_is_anomaly", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "zsc": bool_series(
            base.get("zscore_flag", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "ewma": bool_series(
            base.get("ewma_flag", pd.Series(False, index=base.index)), index=base.index
        ),
        "lstm": bool_series(
            base.get("lstm_is_anomaly", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "phys": bool_series(
            base.get("physics_anomaly", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "coh": bool_series(
            base.get("coherence_anomaly", pd.Series(False, index=base.index)),
            index=base.index,
        )
        | bool_series(
            base.get("peer_anomaly_pm", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "p2j": bool_series(
            base.get("p2_job_anomaly", pd.Series(False, index=base.index)),
            index=base.index,
        ),
        "dq": data_quality_fire,  # data-quality family (L1)
    }
    for fam_name, fire in vote_families.items():
        tmp = pd.DataFrame(
            {
                "hostname": base["hostname"],
                "_bin_ts": bin_timestamps,
                "_fire": fire.astype(bool),
            }
        )
        # True if ANY row in this (host, bin) has the family firing
        any_in_bin = (
            tmp.groupby(["hostname", "_bin_ts"], sort=False)["_fire"]
            .transform("any")
            .astype("int8")
        )
        base[f"vote5_{fam_name}"] = any_in_bin
    vote_cols = [f"vote5_{n}" for n in vote_families]
    base["votes_5min"] = base[vote_cols].sum(axis=1).astype("int8")

    family_weights = {
        "thr": 1,
        "if": 1,
        "zsc": 1,
        "ewma": 1,
        "lstm": 1,
        "phys": 2,
        "coh": 2,
        "p2j": 2,
        "dq": 2,
    }
    wvote = pd.Series(0, index=base.index, dtype="int8")
    for fam_name, weight in family_weights.items():
        wvote = wvote + base[f"vote5_{fam_name}"].astype("int8") * int(weight)
    base["votes_5min_weighted"] = wvote.astype("int8")

    base = base.drop(columns=["_bin_ts"])

    return base


# Load every detector's output and build/save the fused per-minute table.
def run_fused_table(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    models_dir = Path(cfg["phase1"]["output_dir"])
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "offline/data/phase2"))
    phase3_dir = Path(cfg.get("phase3", {}).get("output_dir", "offline/data/phase3"))
    out_dir = Path(cfg["phase4"].get("output_dir", "offline/data/phase4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fused_alerts.parquet"

    if out_path.exists() and not force:
        print(f"[fused_table] {out_path.name} exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    t0 = time.perf_counter()
    print(f"[fused_table] models_dir={models_dir}")

    phase3_enabled = cfg.get("phase3", {}).get("enabled", True)

    if_df = load_if_scores(models_dir)
    thr_df = load_threshold_alerts(models_dir)
    zscore_df = load_baseline_alerts(models_dir, "zscore")
    ewma_df = load_baseline_alerts(models_dir, "ewma")
    if phase3_enabled:
        phys_df = load_physics_scores(phase3_dir)
        const_df = load_constraint_violations(phase3_dir)
        jctx_df = load_job_context(phase3_dir)
    else:
        phys_df = const_df = jctx_df = None
    p2_df = load_phase2_scores(phase2_dir)
    coh_df = load_coherence(phase2_dir)
    pm_coh_df = load_per_minute_coherence(phase2_dir)
    lstm_df = load_lstm_scores(models_dir)
    stream_df = load_streaming_coherence(phase2_dir)

    for name, df in [
        ("D1 IF", if_df),
        ("D2 Threshold", thr_df),
        ("D1 z-score", zscore_df),
        ("D1 EWMA", ewma_df),
        ("D3 Physics", phys_df),
        ("D4 Constraints", const_df),
        ("D5 Phase2", p2_df),
        ("D5b Coherence", coh_df),
        ("D5c CohPerMin", pm_coh_df),
        ("D6 LSTM-AE", lstm_df),
        ("P3 JobContext", jctx_df),
        ("D5d StreamCoh", stream_df),
    ]:
        n = len(df) if df is not None else 0
        print(f"  {name:<20s}: {n:>10,} rows")

    base = build_fusion_table(
        if_df,
        thr_df,
        phys_df,
        const_df,
        p2_df,
        coh_df,
        lstm_df=lstm_df,
        jctx_df=jctx_df,
        zscore_df=zscore_df,
        ewma_df=ewma_df,
        pm_coh_df=pm_coh_df,
        streaming_coh_df=stream_df,
    )
    save_parquet(base, out_path)
    elapsed = time.perf_counter() - t0
    print(f"[fused_table] wrote {len(base):,} rows → {out_path}  ({elapsed:.1f}s)")
    return base
