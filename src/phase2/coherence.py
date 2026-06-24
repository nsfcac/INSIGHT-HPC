from __future__ import annotations

from src.phase2.coherence_module.batch import *
from src.phase2.coherence_module.streaming import *


# Score multi-node jobs for cross-node profile coherence anomalies.
def run_multi_node_coherence(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_cfg = cfg.get("phase2", {})
    z_thresh = float(phase2_cfg.get("coherence_z_thresh", 2.0))
    min_peers = int(phase2_cfg.get("coherence_min_peer_nodes", 2))

    phase2_dir = Path(phase2_cfg.get("output_dir", "data/phase2"))
    cluster_dir = phase2_dir / "clusters"
    profiles_path = phase2_dir / "job_profiles_clustered.parquet"
    out_path = phase2_dir / "multi_node_coherence.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[coherence] multi_node_coherence.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    if not profiles_path.exists():
        raise FileNotFoundError(
            "Run cluster_jobs first (need job_profiles_clustered.parquet)."
        )

    profiles = pd.read_parquet(profiles_path, engine="pyarrow")

    t0 = time.perf_counter()
    print(f"\n[coherence] Multi-node job coherence scoring")
    print(f"  z_thresh={z_thresh}  min_peer_nodes={min_peers}")
    print(f"  {len(profiles):,} job profiles loaded")

    all_records = []
    total_jobs_scored = 0
    total_anomalies = 0

    for comp in profiles["component"].unique():
        comp_df = profiles[profiles["component"] == comp].copy()

        scaler = load_scaler(cluster_dir, comp)
        feature_cols = load_feature_list(cluster_dir, comp) or COMPONENT_FEATURES.get(
            comp, BASE_FEATURES
        )

        medians_path = cluster_dir / f"{comp}_medians.json"
        medians = json.load(open(medians_path)) if medians_path.exists() else {}

        if scaler is None:
            print(
                f"  [{comp.upper()}] No scaler found — run cluster_jobs first, skipping"
            )
            continue

        job_node_counts = comp_df.groupby("job_id")["hostname"].nunique()
        multi_node_jids = job_node_counts[job_node_counts >= min_peers].index

        n_multi = len(multi_node_jids)
        print(
            f"\n  [{comp.upper()}]  {len(comp_df):,} profiles  "
            f"{n_multi:,} multi-node jobs (>={min_peers} nodes)"
        )

        comp_jobs_scored = 0
        comp_anomalies = 0

        for jid in multi_node_jids:
            job_df = comp_df[comp_df["job_id"] == jid]

            node_profiles = (
                job_df.sort_values("seg_start")
                .groupby("hostname", sort=False)
                .last()
                .reset_index()
            )

            if len(node_profiles) < min_peers:
                continue

            recs = coherence_for_job(
                node_profiles, feature_cols, scaler, medians, z_thresh
            )

            for r, (_, prof_row) in zip(recs, node_profiles.iterrows()):
                r["job_id"] = jid
                r["component"] = comp
                r["seg_start"] = prof_row.get("seg_start")
                r["seg_end"] = prof_row.get("seg_end")

            all_records.extend(recs)
            n_anom_recs = sum(r["is_coherence_anomaly"] for r in recs)
            total_jobs_scored += 1
            total_anomalies += n_anom_recs
            comp_jobs_scored += 1
            comp_anomalies += n_anom_recs

        print(
            f"  [{comp.upper()}] Scored {comp_jobs_scored} multi-node jobs  "
            f"anomalous node-job pairs: {comp_anomalies}"
        )

    if not all_records:
        print("[coherence] No multi-node jobs found.")
        return pd.DataFrame()

    result = pd.DataFrame(all_records)
    result["job_id"] = pd.array(result["job_id"], dtype="Int64")

    for col in ["coherence_score", "peer_median_dist", "peer_mad_dist", "coherence_z"]:
        if col in result.columns:
            result[col] = result[col].astype("float32")
    for col in ["n_peer_nodes", "n_valid_profiles", "cluster_id"]:
        if col in result.columns:
            result[col] = result[col].astype("int16")

    result = result.sort_values("coherence_z", ascending=False, na_position="last")
    save_parquet(result, out_path)

    n_anom = int(result["is_coherence_anomaly"].sum())
    n_jobs = int(result["job_id"].nunique())
    elapsed = time.perf_counter() - t0
    print(f"\n[coherence] Done in {elapsed:.1f}s")
    print(f"  Jobs scored   : {n_jobs:,}")
    print(f"  Node-job pairs: {len(result):,}")
    print(f"  Anomalous     : {n_anom:,} ({100*n_anom/max(len(result),1):.1f}%)")
    print(f"  Saved: {out_path}")

    if n_anom > 0:
        print("\n  Top coherence anomalies (all components):")
        for comp_name in result["component"].unique():
            comp_anom = result[
                (result["component"] == comp_name) & result["is_coherence_anomaly"]
            ]
            if comp_anom.empty:
                print(f"    [{comp_name.upper()}]  no anomalies")
                continue
            print(
                f"    [{comp_name.upper()}]  {len(comp_anom)} anomalous node-job pairs:"
            )
            for _, row in comp_anom.head(10).iterrows():
                print(
                    f"      job={row['job_id']:>10}  node={row['hostname']:20s}  "
                    f"z={row['coherence_z']:.2f}  peers={row['n_peer_nodes']}"
                )

    return result


# Score each node's per-minute power deviation against its job peers.
def run_per_minute_coherence(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "data/phase2"))
    seg_path = phase2_dir / "job_segments.parquet"
    out_path = phase2_dir / "per_minute_coherence.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print("[coh_pm] per_minute_coherence.parquet exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")
    if not seg_path.exists():
        print(f"[coh_pm] {seg_path} missing — run segment_jobs first. Skipping.")
        return pd.DataFrame()

    t0 = time.perf_counter()

    segs = pd.read_parquet(seg_path, engine="pyarrow")
    segs = segs.dropna(subset=["job_id", "seg_start", "seg_end", "hostname"]).copy()
    segs["seg_start"] = pd.to_datetime(segs["seg_start"], utc=True).dt.floor("1min")
    segs["seg_end"] = pd.to_datetime(segs["seg_end"], utc=True).dt.floor("1min")

    host_per_job = segs.groupby("job_id")["hostname"].nunique()
    multi_jids = host_per_job[host_per_job >= PM_MIN_PEERS + 1].index
    segs = segs[segs["job_id"].isin(multi_jids)].copy()
    if segs.empty:
        print("[coh_pm] No multi-host jobs found.")
        return pd.DataFrame()
    print(f"[coh_pm] {len(multi_jids):,} multi-host jobs  ({len(segs):,} segments)")

    master_dir = Path(cfg["paths"]["master"])
    print("[coh_pm] Loading per-minute power from master parquets ...")
    pwr = load_per_minute_power(master_dir)
    if pwr.empty:
        print("[coh_pm] No master power data — cannot compute per-minute coherence.")
        return pd.DataFrame()
    print(f"[coh_pm] Loaded {len(pwr):,} per-minute power rows")

    rows = []
    for job_id, grp in segs.groupby("job_id"):
        parts = []
        for _, s in grp.iterrows():
            dur = int((s["seg_end"] - s["seg_start"]).total_seconds() // 60) + 1
            if dur <= 0:
                continue
            ts = pd.date_range(s["seg_start"], periods=dur, freq="1min", tz="UTC")
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "hostname": s["hostname"],
                        "job_id": job_id,
                    }
                )
            )
        if parts:
            rows.append(pd.concat(parts, ignore_index=True))
    if not rows:
        return pd.DataFrame()
    job_minutes = pd.concat(rows, ignore_index=True)
    job_minutes = job_minutes.drop_duplicates(["timestamp", "hostname", "job_id"])

    merged = job_minutes.merge(pwr, on=["timestamp", "hostname"], how="inner")
    if merged.empty:
        print("[coh_pm] No power rows matched job minutes — aborting.")
        return pd.DataFrame()

    g = merged.groupby(["timestamp", "job_id"])["power_w"]
    merged["peer_count"] = g.transform("count").astype("int16")
    merged["peer_sum"] = g.transform("sum").astype("float32")
    merged["peer_sq"] = g.transform(lambda s: (s**2).sum()).astype("float32")
    n = merged["peer_count"] - 1
    peer_mean = (merged["peer_sum"] - merged["power_w"]) / n.replace(0, np.nan)
    peer_var = (
        (merged["peer_sq"] - merged["power_w"] ** 2) / n.replace(0, np.nan)
        - peer_mean**2
    ).clip(lower=0)
    peer_std = np.sqrt(peer_var).replace(0, np.nan)
    merged["peer_mean_w"] = peer_mean.astype("float32")
    merged["peer_std_w"] = peer_std.astype("float32")
    merged["pwr_z"] = ((merged["power_w"] - peer_mean) / peer_std).astype("float32")

    merged = merged[merged["peer_count"] >= PM_MIN_PEERS + 1].copy()

    merged = merged.sort_values(["hostname", "job_id", "timestamp"]).reset_index(
        drop=True
    )
    hot = merged["pwr_z"].abs() > PM_MIN_Z
    keys = merged["hostname"].astype(str) + "|" + merged["job_id"].astype(str)
    run_id = (hot != hot.shift()).cumsum()
    run_key = run_id.astype(str) + ":" + keys
    run_len = hot.groupby(run_key).transform("sum")
    merged["is_peer_anomaly"] = (hot & (run_len >= PM_MIN_RUN)).astype(bool)

    out = merged[
        [
            "timestamp",
            "hostname",
            "job_id",
            "peer_count",
            "peer_mean_w",
            "pwr_z",
            "is_peer_anomaly",
        ]
    ].copy()
    out["job_id"] = pd.array(out["job_id"], dtype="Int64")

    save_parquet(out, out_path)
    n_anom = int(out["is_peer_anomaly"].sum())
    elapsed = time.perf_counter() - t0
    print(
        f"[coh_pm] Done in {elapsed:.1f}s — {len(out):,} rows  "
        f"is_peer_anomaly={n_anom:,}"
    )
    print(f"[coh_pm] Saved: {out_path}")
    return out


# Replay multi-node segments minute-by-minute to detect emerging peer divergence.
def run_streaming_coherence(
    force: bool = False,
    fingerprint_min: int = 30,
    update_every_min: int = 1,
    z_thresh: float = 2.0,
    min_peers: int = 2,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    cfg = load_config()
    phase2_dir = Path(cfg.get("phase2", {}).get("output_dir", "data/phase2"))
    cluster_dir = phase2_dir / "clusters"
    out_path = phase2_dir / "streaming_coherence.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[streaming_coherence] {out_path} exists — loading")
        return pd.read_parquet(out_path, engine="pyarrow")

    base_path = phase2_dir / "multi_node_coherence.parquet"
    if not base_path.exists():
        print(f"[streaming_coherence] {base_path} missing — run phase2 first; skip")
        return pd.DataFrame()
    coh = pd.read_parquet(base_path, engine="pyarrow")
    coh["seg_start"] = pd.to_datetime(coh["seg_start"], utc=True, errors="coerce")
    coh["seg_end"] = pd.to_datetime(coh["seg_end"], utc=True, errors="coerce")

    grouped = coh.groupby(["job_id", "component", "seg_start", "seg_end"])
    n_groups = grouped.ngroups
    print(
        f"[streaming_coherence] {n_groups:,} multi-node-job-segments to score "
        f"(fingerprint_min={fingerprint_min}, update_every={update_every_min} min)"
    )

    master_root = Path(
        cfg.get("paths", {}).get("master_injected", "data/master_injected")
    )
    master_cache: dict = {}

    # Load and cache a host's relevant master sensor columns.
    def load_master(host: str, component: str) -> Optional[pd.DataFrame]:
        key = (host, component)
        if key in master_cache:
            return master_cache[key]
        p = master_root / component / f"{host}.parquet"
        if not p.exists():
            master_cache[key] = None
            return None
        try:
            schema_cols = pq.read_schema(p).names
            keep = ["timestamp"]
            for c in schema_cols:
                cl = c.lower()
                if not c.endswith("_avg") and c not in {
                    "slurm_cpu_load",
                    "slurm_memory_usage",
                }:
                    continue
                if (
                    "systeminputpower" in cl
                    or "systempowerconsumption" in cl
                    or "totalcpupower" in cl
                    or "totalmemorypower" in cl
                    or "temperaturereading" in cl
                    or "rpmreading" in cl
                    or "fanspeed" in cl
                    or "gpuusage" in cl
                    or "gpumemoryusage" in cl
                    or ("powerconsumption" in cl and "systempowerconsumption" not in cl)
                    or c in {"slurm_cpu_load", "slurm_memory_usage"}
                ):
                    keep.append(c)
            keep = [
                c for i, c in enumerate(keep) if c in schema_cols and c not in keep[:i]
            ]
            df = pd.read_parquet(p, engine="pyarrow", columns=keep)
        except Exception:
            df = pd.read_parquet(p, engine="pyarrow")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        master_cache[key] = df
        return df

    art_cache: dict = {}

    # Load and cache a component's scaler, feature list, and medians.
    def load_art(component: str):
        if component in art_cache:
            return art_cache[component]
        scaler = load_scaler(cluster_dir, component)
        feature_cols = load_feature_list(
            cluster_dir, component
        ) or COMPONENT_FEATURES.get(component, BASE_FEATURES)
        medians_path = cluster_dir / f"{component}_medians.json"
        medians = json.load(open(medians_path)) if medians_path.exists() else {}
        art_cache[component] = (scaler, feature_cols, medians)
        return art_cache[component]

    if n_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit():
            n_workers = max(1, int(slurm_cpus))
        else:
            n_workers = max(1, (os.cpu_count() or 2) - 1)

    work_units: list = []
    needed_hosts: set = set()
    skipped_no_scaler = 0
    for (jid, comp, seg_start, seg_end), grp in grouped:
        peer_hosts = tuple(sorted(grp["hostname"].dropna().unique().tolist()))
        work_units.append((jid, comp, seg_start, seg_end, peer_hosts))
        for h in peer_hosts:
            needed_hosts.add((h, comp))

    print(
        f"[streaming_coherence] {len(work_units):,} work units; "
        f"{len(needed_hosts):,} unique (host, comp) parquets to preload"
    )

    for comp in {comp for _, comp, *_ in work_units}:
        GLOBAL_ART_CACHE[comp] = load_art(comp)
        if GLOBAL_ART_CACHE[comp][0] is None:
            skipped_no_scaler += sum(1 for _, c, *_ in work_units if c == comp)

    t_load = time.perf_counter()
    for h, comp in needed_hosts:
        GLOBAL_MASTER_CACHE[(h, comp)] = load_master(h, comp)
    print(
        f"[streaming_coherence] master preload done in {time.perf_counter() - t_load:.1f}s "
        f"({sum(1 for v in GLOBAL_MASTER_CACHE.values() if v is not None)} loaded)"
    )

    GLOBAL_PARAMS.clear()
    GLOBAL_PARAMS.update(
        {
            "fingerprint_min": int(fingerprint_min),
            "update_every_min": int(update_every_min),
            "z_thresh": float(z_thresh),
            "min_peers": int(min_peers),
        }
    )

    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(v, "1")

    all_records: list = []
    completed = 0
    t0 = time.perf_counter()

    if n_workers <= 1:
        for wu in work_units:
            recs = process_segment_worker(wu)
            all_records.extend(recs)
            completed += 1
            if completed % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[streaming_coherence]   {completed}/{len(work_units)} groups, "
                    f"{len(all_records):,} records ({elapsed:.0f}s)"
                )
    else:
        ctx = mp.get_context("fork")
        chunksize = max(1, len(work_units) // (n_workers * 4))
        print(
            f"[streaming_coherence] parallel: n_workers={n_workers}, chunksize={chunksize}"
        )
        with ctx.Pool(processes=n_workers) as pool:
            for recs in pool.imap_unordered(
                process_segment_worker, work_units, chunksize=chunksize
            ):
                all_records.extend(recs)
                completed += 1
                if completed % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[streaming_coherence]   {completed}/{len(work_units)} groups, "
                        f"{len(all_records):,} records ({elapsed:.0f}s)"
                    )

    elapsed = time.perf_counter() - t0
    print(
        f"[streaming_coherence] Done: {completed}/{len(work_units)} groups scored "
        f"({skipped_no_scaler} skipped — no scaler), "
        f"{len(all_records):,} per-minute records in {elapsed:.1f}s"
    )

    if not all_records:
        msg = (
            "[streaming_coherence] No records produced. Refusing to keep a "
            "stale streaming_coherence.parquet for a forced run."
        )
        if force and len(work_units) > 0:
            if out_path.exists():
                out_path.unlink()
            raise RuntimeError(msg)
        print(msg)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    save_parquet(df, out_path)
    print(f"[streaming_coherence] Wrote {len(df):,} rows → {out_path}")
    return df


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    run_multi_node_coherence(force=force)
    run_per_minute_coherence(force=force)
    run_streaming_coherence(force=force)
