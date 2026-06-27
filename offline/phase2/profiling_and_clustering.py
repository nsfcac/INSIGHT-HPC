from __future__ import annotations

from offline.phase2.profiling_and_clustering_module.profiles import *
from offline.phase2.profiling_and_clustering_module.clustering_features import *
from offline.phase2.profiling_and_clustering_module.training import *

try:
    NpEncoder.__module__ = __name__
except NameError:
    pass
try:
    PduCache.__module__ = __name__
except NameError:
    pass


# Build per-segment job profiles (power/thermal/GPU/PDU features) from master tables.
def run_build_profiles(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "offline/data/phase2"))
    master_dir = Path(cfg["paths"]["master"])
    seg_path = phase2_dir / "job_segments.parquet"
    out_path = phase2_dir / "job_profiles.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[profiles] job_profiles.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    if not seg_path.exists():
        raise FileNotFoundError("Run segment_jobs first.")

    phase2_cfg = cfg.get("phase2", {})
    min_duration = int(phase2_cfg.get("min_segment_duration_min", 5))
    efficiency_enabled = bool(phase2_cfg.get("efficiency_metrics", True))

    segments = pd.read_parquet(seg_path, engine="pyarrow")
    segs = segments[
        (~segments["is_idle"]) & (segments["duration_min"] >= min_duration)
    ].copy()
    print(f"\n[profiles] {len(segs):,} job segments >= {min_duration} min to profile")
    if efficiency_enabled:
        print("  Energy efficiency metrics: ON")

    rack_map = build_rack_map_from_paths(master_dir)
    pdu_cache = PduCache(master_dir, rack_map)

    t0 = time.perf_counter()
    all_profiles = []

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue

        comp_segs = segs[segs["component"] == comp]
        nodes = comp_segs["hostname"].unique()
        print(
            f"\n  [{comp.upper()}]  {len(comp_segs):,} segments across {len(nodes)} nodes"
        )

        for hostname in sorted(nodes):
            node_segs = comp_segs[comp_segs["hostname"] == hostname]
            p = comp_dir / f"{hostname}.parquet"
            if not p.exists():
                continue

            master_df = load_parquet(p)
            if master_df is None or master_df.empty:
                continue

            master_df[TS] = pd.to_datetime(master_df[TS], utc=True)
            rid = rack_map.node_to_rack.get(hostname)

            node_profiles = []
            for _, seg in node_segs.iterrows():
                profile = profile_segment(seg, master_df, comp, pdu_cache, rid)
                if not profile.get("profile_valid", False):
                    continue
                row = {
                    "hostname": hostname,
                    "component": comp,
                    "job_id": seg["job_id"],
                    "seg_start": seg["seg_start"],
                    "seg_end": seg["seg_end"],
                    "duration_min": seg["duration_min"],
                    "is_multi_job": seg["is_multi_job"],
                    "split": seg["split"],
                    "req_cpus": (
                        float(seg["req_cpus"])
                        if "req_cpus" in seg.index and pd.notna(seg.get("req_cpus"))
                        else np.nan
                    ),
                }
                row.update(profile)
                node_profiles.append(row)

            if node_profiles:
                all_profiles.extend(node_profiles)
                print(
                    f"    {hostname:20s}: {len(node_segs):>4} segs  "
                    f"profiled={len(node_profiles):>4}"
                )

    if not all_profiles:
        print("[profiles] No profiles produced.")
        return pd.DataFrame()

    result = pd.DataFrame(all_profiles)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")

    float_cols = [c for c in result.columns if result[c].dtype in (float, np.float64)]
    for c in float_cols:
        result[c] = result[c].astype("float32")

    save_parquet(result, out_path)
    elapsed = time.perf_counter() - t0
    valid = int(result["profile_valid"].sum())
    print(f"\n[profiles] Done in {elapsed:.1f}s  profiles={valid:,}  saved: {out_path}")

    for feat in ["pdu_idrac_efficiency", "joules_per_job"]:
        if feat in result.columns:
            coverage = 100 * result[feat].notna().mean()
            print(f"  {feat}: {coverage:.0f}% coverage")

    return result


# Cluster job profiles per component with K-means/HDBSCAN and tag each segment.
def run_cluster_jobs(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_cfg = cfg.get("phase2", {})
    k_values = phase2_cfg.get("cluster_k_values", [4, 8, 12])
    min_cluster = phase2_cfg.get("min_cluster_size", 10)
    random_state = phase2_cfg.get("random_state", 42)
    use_hdbscan = bool(phase2_cfg.get("use_hdbscan", True))

    phase2_dir = Path(phase2_cfg.get("output_dir", "offline/data/phase2"))
    profiles_path = phase2_dir / "job_profiles.parquet"
    cluster_dir = phase2_dir / "clusters"
    out_path = phase2_dir / "job_profiles_clustered.parquet"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[cluster] job_profiles_clustered.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    if not profiles_path.exists():
        raise FileNotFoundError("Run build_job_profiles first.")

    profiles = pd.read_parquet(profiles_path, engine="pyarrow")
    print(
        f"\n[cluster] Clustering {len(profiles):,} job profiles  "
        f"use_hdbscan={use_hdbscan}"
    )

    FAILED_STATES = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}
    failed_job_ids: set = set()
    raw_base = Path(cfg["paths"].get("raw_parquet", "offline/data/raw_parquet"))
    for comp_cfg in cfg.get("components", []):
        slurm_path = raw_base / comp_cfg["name"] / "slurm" / "jobs.parquet"
        if not slurm_path.exists():
            continue
        try:
            sdf = pd.read_parquet(
                slurm_path, engine="pyarrow", columns=["job_id", "job_state"]
            )
            sdf["job_id"] = pd.to_numeric(sdf["job_id"], errors="coerce")
            sdf = sdf.dropna(subset=["job_id"])
            bad = sdf[sdf["job_state"].isin(FAILED_STATES)]["job_id"].astype(int)
            failed_job_ids.update(bad.tolist())
        except Exception as e:
            print(f"  [WARN] Could not load SLURM jobs for {comp_cfg['name']}: {e}")
    if failed_job_ids:
        print(
            f"[cluster] Excluding {len(failed_job_ids):,} failed/timeout job IDs "
            f"from training baseline"
        )

    profile_job_ids = (
        pd.to_numeric(profiles["job_id"], errors="coerce").fillna(-1).astype(int)
    )

    profiles["cluster_id"] = -1
    profiles["cluster_dist"] = np.nan
    profiles["cluster_algo"] = "none"

    for comp in profiles["component"].unique():
        comp_mask = profiles["component"] == comp
        comp_df = profiles[comp_mask].copy()
        comp_job_ids = profile_job_ids[comp_mask]
        print(f"\n  [{comp.upper()}]  {len(comp_df):,} profiles")

        res = cluster_component(
            comp,
            comp_df,
            comp_job_ids,
            failed_job_ids,
            cluster_dir,
            k_values,
            min_cluster,
            random_state,
            use_hdbscan,
        )
        if res is None:
            continue
        assigned_ids, min_dists, chosen_algo = res

        idx = profiles.index[comp_mask]
        profiles.loc[idx, "cluster_id"] = assigned_ids.astype("int16")
        profiles.loc[idx, "cluster_dist"] = min_dists.astype("float32")
        profiles.loc[idx, "cluster_algo"] = chosen_algo

    profiles["cluster_id"] = profiles["cluster_id"].astype("int16")
    profiles["cluster_dist"] = profiles["cluster_dist"].astype("float32")
    save_parquet(profiles, out_path)
    print(f"\n[cluster] Saved: {out_path}")
    return profiles


# Train per-cluster Isolation Forests and score every job segment for anomalies.
def run_train_cluster_models(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_cfg = cfg.get("phase2", {})
    repro_enabled = bool(phase2_cfg.get("reproducibility_score", True))
    dist_quantile = float(phase2_cfg.get("distance_alert_quantile", 0.97))
    dist_min_duration = float(phase2_cfg.get("distance_alert_min_duration_min", 30))
    dist_single_host_only = bool(
        phase2_cfg.get("distance_alert_single_host_only", True)
    )

    feat_aligned = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))
    feat_dir_base = Path(cfg["paths"]["features"])
    feat_dir = feat_aligned if feat_aligned.exists() else feat_dir_base
    phase2_dir = Path(phase2_cfg.get("output_dir", "offline/data/phase2"))
    print(f"\n[cluster_if] Feature source: {feat_dir}")

    model_dir = phase2_dir / "cluster_models"
    out_path = phase2_dir / "job_anomaly_scores.parquet"
    repro_path = phase2_dir / "job_reproducibility.parquet"
    model_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[cluster_if] job_anomaly_scores.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    clustered_path = phase2_dir / "job_profiles_clustered.parquet"
    if not clustered_path.exists():
        raise FileNotFoundError(f"Run cluster_jobs first (expected: {clustered_path})")

    profiles = pd.read_parquet(clustered_path, engine="pyarrow")
    windows = load_maintenance_windows(cfg)
    if_cfg = cfg.get("phase1", {}).get("isolation_forest", {})
    p2_if_cfg = cfg.get("phase2", {}).get("cluster_if", {})
    n_est = int(p2_if_cfg.get("n_estimators", if_cfg.get("n_estimators", 200)))
    contam = float(p2_if_cfg.get("contamination", if_cfg.get("contamination", 0.01)))
    n_jobs = int(p2_if_cfg.get("n_jobs", if_cfg.get("n_jobs", -1)))

    print(
        f"[cluster_if] Per-cluster IF  n_estimators={n_est}  n_jobs={n_jobs}  "
        f"contam={contam}  reproducibility={repro_enabled}"
    )
    print(
        f"[cluster_if] Supplemental distance alerts  q={dist_quantile:.3f}  "
        f"min_duration={dist_min_duration:g}min  "
        f"single_host_only={dist_single_host_only}"
    )

    t0 = time.perf_counter()
    score_records = []

    job_host_counts = (
        profiles.groupby(["component", "job_id"])["hostname"].nunique().to_dict()
    )

    for comp in profiles["component"].unique():
        comp_df = profiles[profiles["component"] == comp]

        feature_cols = discover_features(feat_dir, comp)
        if not feature_cols:
            print(f"  [WARN] No IF features for {comp}, skipping")
            continue

        cluster_ids = sorted(
            int(c) for c in comp_df["cluster_id"].dropna().unique() if c >= 0
        )
        print(
            f"\n  [{comp.upper()}]  {len(comp_df):,} segments  "
            f"{len(cluster_ids)} clusters  {len(feature_cols)} features"
        )

        cluster_models: dict = {}
        cluster_thresholds: dict = {}
        cluster_distance_thresholds: dict = {}

        for cid in cluster_ids:
            model_path = model_dir / f"{comp}_cluster{cid}.pkl"
            thr_path = model_dir / f"{comp}_cluster{cid}_threshold.json"

            if model_path.exists() and thr_path.exists() and not force:
                with open(model_path, "rb") as f:
                    cluster_models[cid] = pickle.load(f)
                with open(thr_path) as f:
                    thr_payload = json.load(f)
                cluster_thresholds[cid] = float(thr_payload["threshold"])
                cluster_distance_thresholds[cid] = float(
                    thr_payload.get("distance_threshold", np.inf)
                )
                print(f"    C{cid:2d}: loaded")
                continue

            train_segs = comp_df[
                (comp_df["cluster_id"] == cid)
                & (comp_df["split"] == "train")
                & (~comp_df["is_multi_job"].fillna(False))
            ]
            if len(train_segs) < 5:
                print(f"    C{cid:2d}: only {len(train_segs)} train segs, skipping")
                continue

            X = load_cluster_train_matrix(
                feat_dir, comp, feature_cols, train_segs, windows
            )
            if X is None or len(X) < 50:
                print(
                    f"    C{cid:2d}: insufficient rows "
                    f"({len(X) if X is not None else 0}), skipping"
                )
                continue

            print(f"    C{cid:2d}: training on {len(X):,} rows ...")
            clf = IsolationForest(
                n_estimators=n_est, contamination=contam, random_state=42, n_jobs=n_jobs
            )
            clf.fit(X)

            raw = clf.decision_function(X)
            s_min, s_max = raw.min(), raw.max()
            norm = (
                1.0 - (raw - s_min) / (s_max - s_min)
                if s_max > s_min
                else np.full_like(raw, 0.5)
            )

            train_seg_scores = []
            for _, train_seg in train_segs.iterrows():
                seg_score, _ = score_segment(
                    feat_dir,
                    comp,
                    feature_cols,
                    clf,
                    train_seg["hostname"],
                    pd.Timestamp(train_seg["seg_start"]),
                    pd.Timestamp(train_seg["seg_end"]),
                )
                if not np.isnan(seg_score):
                    train_seg_scores.append(float(seg_score))

            threshold = (
                float(np.percentile(train_seg_scores, 99))
                if train_seg_scores
                else float(np.percentile(norm, 99))
            )
            train_dist = pd.to_numeric(
                train_segs["cluster_dist"], errors="coerce"
            ).dropna()
            distance_threshold = (
                float(train_dist.quantile(dist_quantile))
                if not train_dist.empty
                else float(np.inf)
            )

            cluster_models[cid] = clf
            cluster_thresholds[cid] = threshold
            cluster_distance_thresholds[cid] = distance_threshold

            with open(model_path, "wb") as f:
                pickle.dump(clf, f)
            with open(thr_path, "w") as f:
                json.dump(
                    {
                        "threshold": threshold,
                        "distance_threshold": distance_threshold,
                        "distance_quantile": dist_quantile,
                        "distance_min_duration_min": dist_min_duration,
                        "distance_single_host_only": dist_single_host_only,
                        "n_train_rows": len(X),
                        "n_train_segments": len(train_seg_scores),
                    },
                    f,
                )
            del X
            gc.collect()
            print(
                f"    C{cid:2d}: trained  score_thr={threshold:.3f}  "
                f"dist_thr={distance_threshold:.3f}  "
                f"(train_segments={len(train_seg_scores)})"
            )

        if feature_cache_enabled():
            print(
                f"  Feature cache active: {len(FEATURE_MATRIX_CACHE)} node matrix/matrices loaded"
            )

        print(f"  Scoring {len(comp_df):,} segments ...")
        for _, seg in comp_df.iterrows():
            cid = int(seg["cluster_id"])
            if cid not in cluster_models:
                continue

            score, n_rows = score_segment(
                feat_dir,
                comp,
                feature_cols,
                cluster_models[cid],
                seg["hostname"],
                pd.Timestamp(seg["seg_start"]),
                pd.Timestamp(seg["seg_end"]),
            )
            thr = cluster_thresholds.get(cid, 0.9)
            dist_thr = cluster_distance_thresholds.get(cid, float(np.inf))
            score_flag = (not np.isnan(score)) and (score > thr)
            cluster_dist = float(seg["cluster_dist"])
            duration_min = float(seg["duration_min"])
            job_key = (comp, seg["job_id"])
            n_job_hosts = int(job_host_counts.get(job_key, 0))
            is_single_host = n_job_hosts == 1
            dist_gate = is_single_host if dist_single_host_only else True
            dist_flag = (
                dist_gate
                and np.isfinite(cluster_dist)
                and np.isfinite(dist_thr)
                and (cluster_dist > dist_thr)
                and (duration_min >= dist_min_duration)
            )
            score_records.append(
                {
                    "hostname": seg["hostname"],
                    "component": comp,
                    "job_id": seg["job_id"],
                    "seg_start": seg["seg_start"],
                    "seg_end": seg["seg_end"],
                    "duration_min": seg["duration_min"],
                    "split": seg["split"],
                    "is_multi_job": seg["is_multi_job"],
                    "cluster_id": cid,
                    "cluster_dist": cluster_dist,
                    "cluster_dist_threshold": (
                        float(dist_thr) if np.isfinite(dist_thr) else np.nan
                    ),
                    "job_anomaly_score": score,
                    "job_if_score_anomaly": score_flag,
                    "job_dist_anomaly": dist_flag,
                    "job_if_anomaly": score_flag or dist_flag,
                    "n_job_hosts": n_job_hosts,
                    "n_scored_rows": n_rows,
                    "req_cpus": (
                        float(seg["req_cpus"])
                        if "req_cpus" in seg.index and pd.notna(seg.get("req_cpus"))
                        else np.nan
                    ),
                }
            )

        clear_feature_cache(feat_dir, comp)

    if not score_records:
        msg = (
            "[cluster_if] No scores produced. Refusing to continue because "
            "downstream Phase 2 steps would otherwise reuse stale artifacts."
        )
        if force:
            if out_path.exists():
                out_path.unlink()
            raise RuntimeError(msg)
        print(msg)
        return pd.DataFrame()

    result = pd.DataFrame(score_records)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")
    result["job_anomaly_score"] = result["job_anomaly_score"].astype("float32")
    result["cluster_dist"] = result["cluster_dist"].astype("float32")
    result["cluster_dist_threshold"] = result["cluster_dist_threshold"].astype(
        "float32"
    )
    result["n_job_hosts"] = result["n_job_hosts"].astype("int16")
    result["n_scored_rows"] = result["n_scored_rows"].astype("int32")
    result["job_if_score_anomaly"] = result["job_if_score_anomaly"].astype(bool)
    result["job_dist_anomaly"] = result["job_dist_anomaly"].astype(bool)
    result["job_if_anomaly"] = result["job_if_anomaly"].astype(bool)
    save_parquet(result, out_path)

    n_anom = int(result["job_if_anomaly"].sum())
    print(
        f"\n[cluster_if] Done in {time.perf_counter()-t0:.1f}s  "
        f"segments={len(result):,}  anomalous={n_anom:,} "
        f"({100*n_anom/max(len(result),1):.1f}%)"
    )
    print(f"  Scores: {out_path}")

    if repro_enabled:
        print("\n[cluster_if] Computing job reproducibility scores ...")
        all_profiles = pd.read_parquet(clustered_path, engine="pyarrow")
        repro_df = compute_reproducibility(all_profiles)
        if not repro_df.empty:
            save_parquet(repro_df, repro_path)
            n_multi = int((repro_df["n_runs"] >= 2).sum())
            n_irrepro = int((~repro_df["is_reproducible"]).sum())
            print(
                f"  Jobs with >=2 runs: {n_multi}  "
                f"non-reproducible: {n_irrepro}  saved: {repro_path}"
            )
        else:
            print("  No multi-run jobs found for reproducibility analysis.")

    return result


if __name__ == "__main__":
    run_build_profiles(force=True)
    run_cluster_jobs(force=True)
