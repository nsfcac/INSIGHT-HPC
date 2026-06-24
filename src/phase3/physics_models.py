from __future__ import annotations

import gc, json, pickle, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit
from src.utils.maintenance import load_maintenance_windows, apply_maintenance_mask

import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*divide by zero encountered in matmul.*",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*overflow encountered in matmul.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*invalid value encountered in matmul.*"
)


from src.phase3.physics_models_module.features import *


# Ridge regression of inlet temperature on power (and regime) for thermal residuals.
class ThermalModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.use_regression = False
        self.regression_use_regime = False
        self.inlet_col = None
        self.power_col = None
        self.train_resid_std = 1.0
        self.fitted = False
        # Per-host thermal residual calibration.
        self.host_resid_mean: dict = {}
        self.host_resid_std: dict = {}

    # Fit the thermal Ridge model on clean train rows; return whether it fitted.
    def fit(self, df: pd.DataFrame, component: str) -> bool:
        cols = list(df.columns)
        self.inlet_col = next(
            (
                c
                for c in cols
                if "temperaturereading" in c.lower()
                and "inlet" in c.lower()
                and c.endswith("_avg")
            ),
            None,
        ) or next(
            (c for c in cols if "inlet" in c.lower() and c.endswith("_avg")),
            None,
        )
        self.power_col = find_avg(cols, "systeminputpower")

        if not self.inlet_col or not self.power_col:
            return False

        train = df[df["split"] == "train"].copy()
        if "audit_any" in train.columns:
            train = train[~train["audit_any"].eq(True)]
        if "maintenance_flag" in train.columns:
            train = train[~train["maintenance_flag"].fillna(False)]
        if "if_is_anomaly" in train.columns:
            train = train[~train["if_is_anomaly"].fillna(False)]
        try:
            from src.utils.gt_mask import in_gt_window

            gt_mask = in_gt_window(train)
            if gt_mask.any():
                n_excluded = int(gt_mask.sum())
                train = train[~gt_mask]
                print(f"    [thermal] excluded {n_excluded:,} rows in GT windows")
        except Exception as e:
            print(f"    [thermal] GT exclusion skipped: {e}")
        if len(train) < 100:
            return False

        inlet = pd.to_numeric(train[self.inlet_col], errors="coerce")
        power = pd.to_numeric(train[self.power_col], errors="coerce")

        X_parts = [power.fillna(0).values.reshape(-1, 1)]
        use_regime = "regime_id" in train.columns
        if use_regime:
            rid_vec = (
                pd.to_numeric(train["regime_id"], errors="coerce").fillna(0).values
            )
            X_parts.append(rid_vec.reshape(-1, 1))
        self.regression_use_regime = use_regime

        X = np.hstack(X_parts).astype(np.float32)
        y = inlet.values
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < 100:
            return False

        X_s = self.scaler.fit_transform(X[valid])
        self.model.fit(X_s, y[valid])

        y_pred = self.model.predict(X_s)
        resid = y[valid] - y_pred
        self.train_resid_std = float(resid.std()) if resid.std() > 0.01 else 1.0
        self.use_regression = True
        self.fitted = True
        return True

    @property
    def use_physics(self) -> bool:
        return False

    # Predict thermal residuals and per-host-calibrated z-scores for every row.
    def predict_residual(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        empty = pd.DataFrame(
            {
                "thermal_residual_c": np.full(n, np.nan, dtype=np.float32),
                "thermal_residual_z": np.full(n, np.nan, dtype=np.float32),
                "thermal_anomaly": np.zeros(n, dtype=bool),
            }
        )

        if not self.fitted or not self.inlet_col or not self.power_col:
            return empty

        inlet = pd.to_numeric(df[self.inlet_col], errors="coerce").values
        power = pd.to_numeric(df[self.power_col], errors="coerce").fillna(0).values

        X_parts = [np.nan_to_num(power.reshape(-1, 1).astype(np.float32), nan=0.0)]
        if self.regression_use_regime:
            if "regime_id" in df.columns:
                rid_vec = (
                    pd.to_numeric(df["regime_id"], errors="coerce").fillna(0).values
                )
            else:
                rid_vec = np.zeros(len(df), dtype=np.float32)
            X_parts.append(rid_vec.reshape(-1, 1).astype(np.float32))
        X = np.nan_to_num(np.hstack(X_parts), nan=0.0)
        X_s = self.scaler.transform(X)
        expected = self.model.predict(X_s)
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)

        residual = (inlet - expected).astype(np.float32)

        resid_std = self.train_resid_std
        resid_mean = 0.0
        power_slope = 0.0  # per-node power→thermal residual slope
        power_resid_vals = None

        if "split" in df.columns and self.power_col and self.power_col in df.columns:
            train_mask = df["split"].values == "train"
            n_train = int(train_mask.sum())
            if n_train >= 50:
                # Compute power residual on training rows for slope correction
                p_vals = (
                    pd.to_numeric(df[self.power_col], errors="coerce").fillna(0).values
                )
                p_tr = p_vals[train_mask].astype(np.float64)
                t_tr = residual[train_mask].astype(np.float64)
                valid = np.isfinite(p_tr) & np.isfinite(t_tr)
                if valid.sum() >= 50:
                    # OLS: t_tr = α + β*p_tr
                    p_v = p_tr[valid]
                    t_v = t_tr[valid]
                    p_c = p_v - p_v.mean()
                    denom = float((p_c**2).sum())
                    if denom > 1e-6:
                        power_slope = float((p_c * (t_v - t_v.mean())).sum() / denom)
                    resid_corrected_tr = t_v - (t_v.mean() + power_slope * p_c)
                    residual_std_candidate = float(np.std(resid_corrected_tr))
                    if residual_std_candidate > 0.01:
                        resid_mean = float(t_v.mean())
                        resid_std = residual_std_candidate
                    # Store full-dataset power residual for correction below
                    power_resid_vals = (p_vals - p_vals[train_mask].mean()).astype(
                        np.float32
                    )

        # Apply slope correction to full dataset residual
        if power_resid_vals is not None and abs(power_slope) > 1e-6:
            residual_corrected = (
                residual.astype(np.float64)
                - resid_mean
                - power_slope * power_resid_vals.astype(np.float64)
            )
            resid_z_raw = (residual_corrected / resid_std).astype(np.float32)
            resid_z = adaptive_resid_z(
                np.asarray(residual_corrected, dtype=np.float64), df, resid_std
            )
        else:
            shifted_residual = residual.astype(np.float64) - resid_mean
            resid_z_raw = (shifted_residual / resid_std).astype(np.float32)
            resid_z = adaptive_resid_z(
                np.asarray(shifted_residual, dtype=np.float64), df, resid_std
            )

        return pd.DataFrame(
            {
                "thermal_residual_c": residual,
                "thermal_residual_z_raw": resid_z_raw,
                "thermal_residual_z": resid_z,
                "thermal_anomaly": np.abs(resid_z) > THERMAL_Z_THRESH,
            }
        )


# Load a node's IF anomaly flags for training-row filtering.
def load_if_scores_for_node(
    if_scores_dir: Path, hostname: str, ts_col: str = TS
) -> Optional[pd.DataFrame]:
    p = if_scores_dir / f"{hostname}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow", columns=[ts_col, "if_is_anomaly"])
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        return df[[ts_col, "if_is_anomaly"]]
    except Exception:
        return None


# Merge IF anomaly flags onto a node frame by timestamp.
def merge_if_flags(
    df: pd.DataFrame, if_df: Optional[pd.DataFrame], ts_col: str = TS
) -> pd.DataFrame:
    if if_df is None or if_df.empty:
        return df
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.merge(if_df.rename(columns={ts_col: ts_col}), on=ts_col, how="left")
    return df


# Train pooled power/thermal models per component and score every node's residuals.
def run_physics_models(force: bool = False) -> None:
    cfg = load_config()
    feat_aligned = Path(cfg["paths"].get("features_aligned", "data/features_aligned"))
    feat_dir = feat_aligned if feat_aligned.exists() else Path(cfg["paths"]["features"])
    models_dir = Path(cfg["phase3"]["output_dir"])

    model_dir = models_dir / "physics"
    scores_dir = model_dir / "residual_scores"
    model_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    if_scores_dir = Path(cfg["phase1"]["output_dir"]) / "isolation_forest" / "scores"
    has_if = if_scores_dir.exists()

    maint_windows = load_maintenance_windows(cfg)

    p2_base = Path(cfg.get("phase2", {}).get("output_dir", "data/phase2"))
    job_segs = load_job_segments(p2_base / "job_segments.parquet")
    has_job_segs = job_segs is not None and not job_segs.empty

    t0 = time.perf_counter()
    print(f"\n[physics] Training expected power + thermal models")
    print(f"  Feature source: {feat_dir}")
    print(f"  Model output  : {model_dir}")
    print(
        f"  IF score filter: {'enabled' if has_if else 'IF scores not found — skipped'}"
    )
    print(
        f"  Job-start grace: {'enabled' if has_job_segs else 'disabled (no job_segments.parquet)'}"
        f"  τ={JOB_TRANSITION_TAU_MIN}min  window={JOB_TRANSITION_WINDOW}min"
    )

    total_power_anom = 0
    total_thermal_anom = 0
    total_rows = 0

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        comp_dir = feat_dir / comp
        if not comp_dir.exists():
            continue

        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))
        print(f"\n  [{comp.upper()}]  {len(parquets)} nodes")

        power_model = PowerModel()
        thermal_model = ThermalModel()
        pool_parts: list = []
        host_train_samples: dict = {}

        rng = np.random.default_rng(POOL_RANDOM_SEED)
        for p in parquets:
            df = load_parquet(p)
            if df is None or df.empty:
                continue
            apply_maintenance_mask(df, maint_windows, TS, "hostname")
            if has_if:
                if_df_node = load_if_scores_for_node(if_scores_dir, p.stem)
                df = merge_if_flags(df, if_df_node)

            # Pre-filter to clean training rows (same guards as model.fit())
            if "split" not in df.columns:
                del df
                continue
            tr = df[df["split"] == "train"].copy()
            if "audit_any" in tr.columns:
                tr = tr[~tr["audit_any"].eq(True)]
            if "maintenance_flag" in tr.columns:
                tr = tr[~tr["maintenance_flag"].fillna(False)]
            if "if_is_anomaly" in tr.columns:
                tr = tr[~tr["if_is_anomaly"].fillna(False)]
            if len(tr) < 50:
                del df, tr
                continue

            row_count = len(tr)
            shuf_idx = rng.permutation(row_count)
            n_pool = min(POOL_MAX_ROWS_PER_NODE, max(50, int(0.8 * row_count)))
            pool_idx = shuf_idx[:n_pool]
            cal_idx = shuf_idx[n_pool : n_pool + 20000]
            tr_pool = tr.iloc[pool_idx]
            tr_cal = (
                tr.iloc[cal_idx] if len(cal_idx) >= 50 else tr_pool
            )  # tiny hosts fall back
            pool_parts.append(tr_pool)
            host_train_samples[p.stem] = tr_cal.copy()
            del df, tr, tr_pool, tr_cal
            gc.collect()

        power_fitted = False
        thermal_fitted = False
        if pool_parts:
            pooled = pd.concat(pool_parts, ignore_index=True)
            pooled["split"] = "train"  # all rows are training rows
            n_nodes_pooled = len(pool_parts)

            # Find a cpu_load column for coverage reporting
            cpu_diagnostic_column = (
                "slurm_cpu_load"
                if "slurm_cpu_load" in pooled.columns
                else next(
                    (
                        c
                        for c in pooled.columns
                        if c.startswith("cpuusage") and c.endswith("_avg")
                    ),
                    None,
                )
            )
            load_vals = (
                pd.to_numeric(pooled[cpu_diagnostic_column], errors="coerce").dropna()
                if cpu_diagnostic_column
                else pd.Series(dtype=float)
            )
            load_p95 = float(load_vals.quantile(0.95)) if len(load_vals) > 0 else 0.0
            load_std = float(load_vals.std()) if len(load_vals) > 0 else 0.0

            print(
                f"    Pooled training: {len(pooled):,} rows from {n_nodes_pooled} nodes  "
                f"cpu_load p95={load_p95:.1f}%  std={load_std:.1f}%"
            )
            power_fitted = power_model.fit(pooled, comp)
            thermal_fitted = thermal_model.fit(pooled, comp)
            if power_fitted:
                try:
                    power_model.fit_host_stats(host_train_samples)
                    n_cal = len(power_model.host_resid_mean)
                    if n_cal > 0:
                        mus = list(power_model.host_resid_mean.values())
                        print(
                            f"    [power] per-host calibration: {n_cal} hosts "
                            f"bias μ range [{min(mus):.1f}, {max(mus):.1f}]W"
                        )
                except Exception as err:
                    print(f"    [power] per-host calibration skipped: {err}")
            del pooled, pool_parts, host_train_samples
            gc.collect()
        else:
            print(f"    [WARN] No poolable training rows for {comp} — models unfitted")

        print(
            f"    Power model fitted: {power_fitted}  "
            f"Thermal model fitted: {thermal_fitted}  "
            f"(thermal physics: {thermal_model.use_physics})  "
            f"load_coverage: p95={power_model.max_train_cpu_load:.1f}%  "
            f"std={power_model.train_cpu_load_std:.1f}%"
        )

        # Save models
        with open(model_dir / f"power_model_{comp}.pkl", "wb") as f:
            pickle.dump(power_model, f)
        with open(model_dir / f"thermal_model_{comp}.pkl", "wb") as f:
            pickle.dump(thermal_model, f)

        # Score all nodes
        for p in parquets:
            hostname = p.stem
            score_path = scores_dir / f"{hostname}.parquet"
            if score_path.exists() and not force:
                continue

            df = load_parquet(p)
            if df is None or df.empty:
                continue

            apply_maintenance_mask(df, maint_windows, TS, "hostname")

            power_scores = power_model.predict_residual(df, hostname=hostname)
            thermal_scores = thermal_model.predict_residual(df)

            result = pd.DataFrame(
                {
                    TS: (
                        pd.to_datetime(df[TS], utc=True) if TS in df.columns else pd.NaT
                    ),
                    "hostname": hostname,
                    "component": comp,
                    "split": df["split"].values if "split" in df.columns else "unknown",
                    "maintenance_flag": df["maintenance_flag"].values,
                }
            )
            result = pd.concat([result, power_scores, thermal_scores], axis=1)

            if has_job_segs:
                trans = build_transition_times(job_segs, hostname)
                if len(trans) > 0:
                    ts_vals = result[TS].values.astype("datetime64[ns]")
                    atten = job_transition_attenuation(ts_vals, trans)
                    # Store raw z-scores for diagnostics, apply attenuation
                    result["power_residual_z_raw"] = result["power_residual_z"].copy()
                    result["thermal_residual_z_raw"] = result[
                        "thermal_residual_z"
                    ].copy()
                    result["job_transition_atten"] = atten
                    result["power_residual_z"] = (
                        result["power_residual_z"] * atten
                    ).astype(np.float32)
                    result["thermal_residual_z"] = (
                        result["thermal_residual_z"] * atten
                    ).astype(np.float32)
                    # Re-threshold with attenuated z-scores
                    result["power_anomaly"] = (
                        np.abs(result["power_residual_z"]) > POWER_Z_THRESH
                    )
                    result["thermal_anomaly"] = (
                        np.abs(result["thermal_residual_z"]) > THERMAL_Z_THRESH
                    )

            result.loc[result["maintenance_flag"].eq(True), "power_anomaly"] = False
            result.loc[result["maintenance_flag"].eq(True), "thermal_anomaly"] = False

            result["physics_anomaly"] = (
                result["power_anomaly"] | result["thermal_anomaly"]
            )
            result["physics_anomaly_strong"] = (
                result["power_anomaly"] & result["thermal_anomaly"]
            )

            n_power = int(result["power_anomaly"].sum())
            n_thermal = int(result["thermal_anomaly"].sum())
            total_power_anom += n_power
            total_thermal_anom += n_thermal
            total_rows += len(result)

            save_parquet(result, score_path)
            print(
                f"    {hostname:20s}: power_anom={n_power:>5,} "
                f"({100*n_power/max(len(result),1):.2f}%)  "
                f"thermal_anom={n_thermal:>5,} "
                f"({100*n_thermal/max(len(result),1):.2f}%)"
            )
            del df, result
            gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"\n[physics] Done in {elapsed:.1f}s")
    print(f"  Total rows: {total_rows:,}")
    print(
        f"  Power anomalies: {total_power_anom:,}  "
        f"({100*total_power_anom/max(total_rows,1):.2f}%)"
    )
    print(
        f"  Thermal anomalies: {total_thermal_anom:,}  "
        f"({100*total_thermal_anom/max(total_rows,1):.2f}%)"
    )
    print(f"  Scores: {scores_dir}")


if __name__ == "__main__":
    run_physics_models(force=True)
