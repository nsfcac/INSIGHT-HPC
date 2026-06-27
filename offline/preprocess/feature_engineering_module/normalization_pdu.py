from __future__ import annotations

from offline.preprocess.feature_engineering_module.constants import *
import gc, io, os, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from shared.utils.io_utils import load_parquet, save_parquet
from shared.utils.rack_topology import rack_id
from shared.utils.regime_utils import attach_regime_id

from offline.preprocess.feature_engineering_module.transforms import *


# Compute per-feature mean/std from clean train rows for z-score normalization.
def compute_norm_stats(
    df: pd.DataFrame,
    feat_cols: list,
    hostname: str,
    comp: str,
    quiet_only: bool = False,
) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    if "audit_any" in train.columns:
        train = train[~train["audit_any"].fillna(False)]

    if quiet_only:
        if "is_running_job" in train.columns:
            train = train[train["is_running_job"].fillna(1.0).eq(0.0)]
        elif "active_job_count" in train.columns:
            train = train[train["active_job_count"].fillna(1).eq(0)]
        if len(train) < 60:
            print(
                f"    [WARN] {hostname}: only {len(train)} idle train rows "
                f"for norm stats — z-scores may be unreliable"
            )

    # Streak values above this are treated as anomalous silence and excluded from normalization stats; routine collection gaps stay in.
    STREAK_NORMAL_MAX = 30

    records = []
    for col in feat_cols:
        s = pd.to_numeric(train[col], errors="coerce").dropna()
        if col.endswith("_nan_streak"):
            s = s[s <= STREAK_NORMAL_MAX]
        records.append(
            {
                "component": comp,
                "hostname": hostname,
                "feature": col,
                "mean": float(s.mean()) if len(s) > 0 else 0.0,
                "std": float(s.std()) if len(s) > 1 else 1.0,
                "count": int(len(s)),
            }
        )
    return pd.DataFrame(records)


# Z-score each feature column using the host's precomputed mean/std.
def normalize(
    df: pd.DataFrame, feat_cols: list, norm_stats: pd.DataFrame, hostname: str
) -> pd.DataFrame:
    STD_FLOOR = 1e-3
    out = df.copy()
    idx = norm_stats[norm_stats["hostname"] == hostname].set_index("feature")
    for col in feat_cols:
        if col not in idx.index:
            continue
        mean = float(idx.loc[col, "mean"])
        std = float(idx.loc[col, "std"])
        if std < STD_FLOOR:
            std = 1.0
        out[col] = (
            ((pd.to_numeric(out[col], errors="coerce") - mean) / std)
            .clip(-10, 10)
            .astype("float32")
        )
    return out


# Build the tuple of feature-column suffixes produced by the transforms.
def build_feat_suffixes() -> tuple:
    suffixes = ["_avg", "_roc1"]
    for w in ROLL_WINDOWS:
        suffixes += [f"_rmean{w}", f"_rstd{w}"]
    for w in QUANT_WINDOWS:
        for q in QUANT_LEVELS:
            suffixes.append(f"_rp{int(q * 100):02d}_{w}")
    for lag in PHYSICS_LAGS:
        suffixes.append(f"_lag{lag}")
    suffixes.append("_enc")
    window_labels = {1440: "1d", 10080: "7d"}
    for w in DRIFT_WINDOWS:
        label = window_labels.get(w, f"{w}m")
        suffixes.append(f"_slope{label}")
    suffixes.append("_nan_streak")
    return tuple(suffixes)


FEAT_SUFFIXES = build_feat_suffixes()


# Aggregate per-outlet PDU parquets into rack-level power features per minute.
def load_pdu_rack_features(pdu_paths: List[Path]) -> Optional[pd.DataFrame]:
    rows: List[pd.DataFrame] = []

    for p in pdu_paths:
        df = load_parquet(p)
        if df is None or df.empty or TS not in df.columns:
            continue

        df[TS] = pd.to_datetime(df[TS], utc=True).dt.floor("60s")
        df = df.drop_duplicates(TS, keep="last").sort_values(TS).reset_index(drop=True)

        avg_c = [
            c for c in df.columns if c.endswith("_avg") and not c.startswith("audit")
        ]
        rmean_c = [c for c in df.columns if c.endswith("_avg_rmean15")]
        rstd_c = [c for c in df.columns if c.endswith("_avg_rstd15")]
        audit_col = "audit_any" if "audit_any" in df.columns else None

        outlet = pd.DataFrame({TS: df[TS]})
        outlet["_avg"] = (
            df[avg_c].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            if avg_c
            else np.nan
        )
        outlet["_rmean15"] = (
            df[rmean_c].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            if rmean_c
            else np.nan
        )
        outlet["_rstd15"] = (
            np.sqrt(
                (df[rstd_c].apply(pd.to_numeric, errors="coerce") ** 2).mean(axis=1)
            )
            if rstd_c
            else np.nan
        )
        outlet["_audit"] = (
            df[audit_col].fillna(False).astype(bool) if audit_col else False
        )
        rows.append(outlet)

    if not rows:
        return None

    long = pd.concat(rows, ignore_index=True)
    result = (
        long.groupby(TS, sort=True)
        .agg(
            pdu__rack_total_avg=("_avg", "sum"),
            pdu__rack_rmean15=("_rmean15", "mean"),
            pdu__rack_rstd15=("_rstd15", "mean"),
            pdu__rack_audit_any=("_audit", "any"),
        )
        .reset_index()
    )
    result["pdu__rack_total_avg"] = result["pdu__rack_total_avg"].astype("float32")
    result["pdu__rack_rmean15"] = result["pdu__rack_rmean15"].astype("float32")
    result["pdu__rack_rstd15"] = result["pdu__rack_rstd15"].astype("float32")
    result["pdu__rack_audit_any"] = result["pdu__rack_audit_any"].astype("int8")
    return result


# Merge rack-level PDU features onto a node table by timestamp.
def attach_pdu(node_df: pd.DataFrame, pdu_paths: List[Path]) -> pd.DataFrame:
    if not pdu_paths:
        return node_df
    pdu_df = load_pdu_rack_features(pdu_paths)
    if pdu_df is None or pdu_df.empty:
        return node_df

    n_before = len(node_df)
    node_df[TS] = pd.to_datetime(node_df[TS], utc=True).dt.floor("60s")
    pdu_df[TS] = pd.to_datetime(pdu_df[TS], utc=True).dt.floor("60s")
    merged = node_df.merge(pdu_df, on=TS, how="left")

    assert (
        len(merged) == n_before
    ), f"PDU join changed row count {n_before} -> {len(merged)}"
    return merged


# Add the node-vs-rack system-power deviation feature.
def rack_power_delta(df: pd.DataFrame) -> pd.DataFrame:
    sys_power_col = next(
        (c for c in df.columns if "systeminputpower" in c and c.endswith("_avg")), None
    )
    if not sys_power_col or "pdu__rack_rmean15" not in df.columns:
        return df
    return pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "node_vs_rack_power_z_avg": df[sys_power_col]
                    .sub(df["pdu__rack_rmean15"])
                    .astype("float32")
                },
                index=df.index,
            ),
        ],
        axis=1,
    )


# Pass 3: attach PDU rack features to every node and copy PDU tables across.
def attach_pdu_pass(cfg: dict, force: bool) -> None:
    feat_dir = Path(cfg["paths"]["features"])
    out_base = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))

    pdu_feat_dir = feat_dir / "infra" / "pdu"
    all_pdu_ids: List[str] = (
        [p.stem for p in sorted(pdu_feat_dir.glob("*.parquet"))]
        if pdu_feat_dir.exists()
        else []
    )
    n_racks = len({rack_id(u) for u in all_pdu_ids if rack_id(u)})
    print(
        f"\n[features] Pass 3: attach PDU rack features  "
        f"({len(all_pdu_ids)} outlets across {n_racks} racks)"
    )

    pdu_attached = pdu_missing = 0

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue

        comp_feat = feat_dir / comp
        if not comp_feat.exists():
            continue

        (out_base / comp).mkdir(parents=True, exist_ok=True)
        # Resolve PDU paths + counters in the parent (deterministic); the I/O + compute per node runs serially or in a pool.
        tasks = []
        for feat_path in sorted(comp_feat.glob("*.parquet")):
            hostname = feat_path.stem
            out_path = out_base / comp / feat_path.name
            node_rack = rack_id(hostname)
            pdu_paths_node: List[Path] = []
            if node_rack is not None:
                units = [u for u in all_pdu_ids if rack_id(u) == node_rack]
                pdu_paths_node = [pdu_feat_dir / f"{u}.parquet" for u in units]
                if pdu_paths_node:
                    pdu_attached += 1
                else:
                    pdu_missing += 1
                    print(f"    [WARN] {hostname}: no PDU for rack {node_rack}")
            tasks.append((feat_path, out_path, pdu_paths_node, force))

        workers = fe_workers()
        if workers == 1 or not tasks:
            for t in tasks:
                log = fe_pass3_node(*t)[1]
                if log:
                    print(log, flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(fe_pass3_node, *t) for t in tasks]
                results = [fut.result() for fut in as_completed(futs)]
                for _, log, _ in sorted(results, key=lambda r: r[0]):
                    if log:
                        print(log, flush=True)

    # Copy infra/pdu tables across unchanged so downstream sees a complete tree.
    src_pdu = feat_dir / "infra" / "pdu"
    if src_pdu.exists():
        dst_pdu = out_base / "infra" / "pdu"
        dst_pdu.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_pdu.glob("*.parquet")):
            d = dst_pdu / p.name
            if not d.exists() or force:
                shutil.copy2(p, d)

    print(f"  [pdu] attached={pdu_attached}  missing={pdu_missing}  → {out_base}")


# Resolve the feature-engineering worker count from env / SLURM allocation.
def fe_workers() -> int:
    env = os.environ.get("INSIGHT_HPC_FE_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, min(12, int(slurm_cpus)))
        except ValueError:
            return 1
    return 1


# Pass 1: compute all features for one node and collect its norm stats.
def fe_pass1_node(
    p: Path,
    comp: str,
    comp_in: Path,
    out_dir: Path,
    force: bool,
    cfg: dict,
    ts_col: str,
    oom_jobs: dict,
):
    hostname = p.stem
    rel = p.relative_to(comp_in)
    out_path = out_dir / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return hostname, None, None, f"    skip {hostname}", False

    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = pd.read_parquet(p, engine="pyarrow")
        if ts_col not in df.columns or "split" not in df.columns:
            detail = buf.getvalue().rstrip()
            msg = f"    [WARN] {hostname}: missing {ts_col} or split"
            return hostname, None, None, f"{detail}\n{msg}" if detail else msg, False

        df = df.sort_values(ts_col).reset_index(drop=True)
        val_cols = [c for c in df.columns if c.endswith("_avg")]
        if not val_cols:
            detail = buf.getvalue().rstrip()
            msg = f"    [WARN] {hostname}: no _avg cols"
            return hostname, None, None, f"{detail}\n{msg}" if detail else msg, False

        df = attach_regime_id(df, cfg=cfg, ts_col=ts_col, exclude_split=True)
        regime_arr = df["regime_id"].to_numpy(dtype=np.int64)

        df = rolling_features(df, val_cols, regime_id=regime_arr)
        df = lag_features(df, physics_cols(val_cols))
        df = drift_features(df, val_cols, regime_id=regime_arr)
        df = silence_features(df, val_cols)
        df = job_features(df)

        fex = pd.Series(False, index=df.index)
        if oom_jobs and "primary_job_id" in df.columns:
            pjid = pd.to_numeric(df["primary_job_id"], errors="coerce")
            ts_utc = pd.to_datetime(df[ts_col], utc=True)
            window = pd.Timedelta("30min")
            for jid, end_ts in oom_jobs.items():
                mask = (
                    (pjid == jid) & (ts_utc >= (end_ts - window)) & (ts_utc <= end_ts)
                )
                fex |= mask
        df["failed_job_exclusion"] = fex.fillna(False).astype("bool")

        n_req_resource = getattr(df, "n_req_resource_count", 0)
        df = time_features(df, ts_col)

        feat_cols = [c for c in df.columns if c.endswith(FEAT_SUFFIXES)]
        stats = compute_norm_stats(df, feat_cols, hostname, comp)
        idle = compute_norm_stats(df, feat_cols, hostname, comp, quiet_only=True)
        save_parquet(df, out_path)
    summary = (
        f"    {hostname:20s}: {len(df):>8,} rows  {len(feat_cols)} features  "
        f"({n_req_resource} req-resource)  {time.perf_counter()-t0:.1f}s"
    )
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    del df
    gc.collect()
    return hostname, stats, idle, log, True


# Pass 2: apply z-score normalization to one node's feature table.
def fe_pass2_node(p: Path, comp_stats: pd.DataFrame):
    hostname = p.stem
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = pd.read_parquet(p, engine="pyarrow")
        feat_cols = [c for c in df.columns if c.endswith(FEAT_SUFFIXES)]
        df = normalize(df, feat_cols, comp_stats, hostname)
        save_parquet(df, p)
    summary = f"    normalized {hostname:20s}  {time.perf_counter()-t0:.1f}s"
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    del df
    gc.collect()
    return hostname, log


# Pass 3: attach PDU features and the rack-power delta for one node.
def fe_pass3_node(
    feat_path: Path, out_path: Path, pdu_paths_node: List[Path], force: bool
):
    hostname = feat_path.stem
    if out_path.exists() and not force:
        return hostname, f"    {hostname}: skip (exists)", 0
    t0 = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        df = load_parquet(feat_path)
        if df is None or df.empty:
            return hostname, buf.getvalue().rstrip(), 0
        df = attach_pdu(df, pdu_paths_node)
        df = rack_power_delta(df)
        pdu_n = len([c for c in df.columns if c.startswith("pdu__")])
        save_parquet(df, out_path)
    summary = (
        f"    {hostname:20s}: {len(df):>8,} rows  +pdu={pdu_n}  "
        f"{time.perf_counter() - t0:.1f}s"
    )
    detail = buf.getvalue().rstrip()
    log = f"{detail}\n{summary}" if detail else summary
    return hostname, log, pdu_n
