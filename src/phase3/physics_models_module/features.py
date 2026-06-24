from __future__ import annotations

from src.phase3.physics_models_module.constants import *
import gc, json, pickle, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit
from src.utils.maintenance import load_maintenance_windows, apply_maintenance_mask
from src.utils.parsers import find_avg, findany

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


# Load non-idle job segments from phase-2 for transition grace periods.
def load_job_segments(seg_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    p = seg_path or Path("data/phase2/job_segments.parquet")
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(
            p, engine="pyarrow", columns=["hostname", "seg_start", "seg_end", "is_idle"]
        )
        df["seg_start"] = pd.to_datetime(df["seg_start"], utc=True)
        df["seg_end"] = pd.to_datetime(df["seg_end"], utc=True)
        return df[~df["is_idle"]].drop(columns=["is_idle"])
    except Exception:
        return None


# Build the sorted array of job start/end transition times for a host.
def build_transition_times(job_segs: pd.DataFrame, hostname: str) -> np.ndarray:
    node_segs = job_segs[job_segs["hostname"] == hostname]
    if node_segs.empty:
        return np.array([], dtype="datetime64[ns]")
    starts = node_segs["seg_start"].values
    ends = node_segs["seg_end"].values
    transitions = np.concatenate([starts, ends])
    transitions.sort()
    return transitions


# Attenuate residual scores near job start/end transitions.
def job_transition_attenuation(
    timestamps: np.ndarray, transition_times: np.ndarray
) -> np.ndarray:
    n = len(timestamps)
    if len(transition_times) == 0:
        return np.ones(n, dtype=np.float32)

    idx = np.searchsorted(transition_times, timestamps, side="right")

    # Candidates: transition just before (idx-1) and just after (idx)
    min_delta = np.full(n, np.inf, dtype=np.float64)
    for offset in [0, -1]:
        j = idx + offset
        valid = (j >= 0) & (j < len(transition_times))
        where_valid = np.where(valid)[0]
        if len(where_valid) == 0:
            continue
        delta_ns = np.abs(
            timestamps[where_valid].astype(np.int64)
            - transition_times[j[where_valid]].astype(np.int64)
        )
        delta_min = delta_ns / 6e10  # nanoseconds → minutes
        min_delta[where_valid] = np.minimum(min_delta[where_valid], delta_min)

    # Beyond the window → no attenuation
    min_delta = np.clip(min_delta, 0.0, JOB_TRANSITION_WINDOW)

    # Exponential ramp: 0 at transition, ~1 after a few τ
    attenuation = 1.0 - np.exp(-min_delta / JOB_TRANSITION_TAU_MIN)
    return attenuation.astype(np.float32)


ADAPTIVE_WINDOW_DAYS = 14  # rolling window length (days)
ADAPTIVE_MIN_PERIODS = 500  # minimum rows before rolling kicks in
ADAPTIVE_STD_FLOOR_F = 0.5
ADAPTIVE_STD_CAP_F = 5.0  # never go above 5× train_resid_std (avoid too-wide threshold)


# Z-score residuals using a robust rolling-IQR standard deviation.
def adaptive_resid_z(
    residual: np.ndarray, df: "pd.DataFrame", train_resid_std: float
) -> np.ndarray:
    n = len(residual)
    if n < ADAPTIVE_MIN_PERIODS:
        return (residual / train_resid_std).astype(np.float32)

    resid_s = pd.Series(residual.astype(np.float64))

    # Sort by timestamp if available so the window is truly past-only.
    if TS in df.columns:
        order = pd.to_datetime(df[TS], utc=True).values.argsort()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        resid_sorted = pd.Series(resid_s.values[order])
    else:
        order = None
        resid_sorted = resid_s

    win = ADAPTIVE_WINDOW_DAYS * 1440  # minutes per day
    q75 = resid_sorted.rolling(win, min_periods=ADAPTIVE_MIN_PERIODS).quantile(0.75)
    q25 = resid_sorted.rolling(win, min_periods=ADAPTIVE_MIN_PERIODS).quantile(0.25)
    iqr_std = (q75 - q25) / 1.349  # robust σ estimate

    floor = train_resid_std * ADAPTIVE_STD_FLOOR_F
    cap = train_resid_std * ADAPTIVE_STD_CAP_F
    eff_std = iqr_std.clip(lower=floor, upper=cap).fillna(train_resid_std)

    z_sorted = (resid_sorted.values / eff_std.values).astype(np.float32)

    if order is not None:
        return z_sorted[inv]
    return z_sorted


# Ridge regression predicting expected system power from load features.
class PowerModel:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.feature_cols: list = []
        self.power_col: Optional[str] = None
        self.train_resid_std: float = 1.0
        # Load-range diagnostics (set in fit, used for extrapolation warnings)
        self.max_train_cpu_load: float = 100.0  # p95 cpu_load seen in training
        self.train_cpu_load_std: float = 0.0  # std of cpu_load in training
        self.fitted = False
        self.host_resid_mean: dict = {}
        self.host_resid_std: dict = {}

    # Assemble the load/power feature matrix for the power model.
    def build_X(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        cols = list(df.columns)
        cpu_column = (
            "slurm_cpu_load"
            if "slurm_cpu_load" in cols
            else next(
                (c for c in cols if c.startswith("cpuusage") and c.endswith("_avg")),
                None,
            )
        )

        gpu_util_columns = [
            c for c in cols if "gpuusage" in c.lower() and c.endswith("_avg")
        ]
        single_gpu_util_column = (
            gpu_util_columns[0] if len(gpu_util_columns) == 1 else None
        )
        aggregated_gpu_util = (
            df[gpu_util_columns]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .fillna(0)
            .values
            if len(gpu_util_columns) > 1
            else None
        )

        feat_candidates = {
            "cpu_load": cpu_column,
            "mem_usage": "slurm_memoryusage",
            "active_jobs": "active_job_count",
            "cpu_power": find_avg(cols, "totalcpupower"),
            "mem_power": find_avg(cols, "totalmemorypower"),
            "gpu_util": single_gpu_util_column,  # None when multi-slot (handled below)
            "regime_id": "regime_id" if "regime_id" in cols else None,
        }
        used = []
        X_parts = []
        for name, col in feat_candidates.items():
            if col and col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                X_parts.append(s.reshape(-1, 1))
                used.append(name)

        # Append averaged multi-slot GPU util (H100 with 4 GPU slots)
        if aggregated_gpu_util is not None:
            X_parts.append(aggregated_gpu_util.reshape(-1, 1))
            used.append("gpu_util")

        if not used:
            return None
        if not self.feature_cols:
            self.feature_cols = used
        return np.hstack(X_parts).astype(np.float32)

    # Fit the power Ridge model on clean train rows; return whether it fitted.
    def fit(self, df: pd.DataFrame, component: str) -> bool:
        cols = list(df.columns)
        self.power_col = find_avg(cols, "systeminputpower") or findany(
            cols, "systempowerconsumption"
        )
        if not self.power_col:
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
                print(f"    [power] excluded {n_excluded:,} rows in GT windows")
        except Exception as e:
            print(f"    [power] GT exclusion skipped: {e}")
        if len(train) < 100:
            return False

        X = self.build_X(train)
        if X is None or X.shape[0] < 100:
            return False

        y = pd.to_numeric(train[self.power_col], errors="coerce").fillna(0).values
        valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
        if valid.sum() < 100:
            return False

        X_s = self.scaler.fit_transform(X[valid])
        self.model.fit(X_s, y[valid])

        # Compute training residual std for z-scoring
        y_pred = self.model.predict(X_s)
        resid = y[valid] - y_pred
        self.train_resid_std = float(resid.std()) if resid.std() > 0.1 else 1.0

        # Track load coverage so we can detect extrapolation at score time
        cpu_fit_column = (
            "slurm_cpu_load"
            if "slurm_cpu_load" in train.columns
            else next(
                (
                    c
                    for c in train.columns
                    if c.startswith("cpuusage") and c.endswith("_avg")
                ),
                None,
            )
        )
        if cpu_fit_column:
            load_v = pd.to_numeric(train[cpu_fit_column], errors="coerce").dropna()
            self.max_train_cpu_load = (
                float(load_v.quantile(0.95)) if len(load_v) > 0 else 100.0
            )
            self.train_cpu_load_std = float(load_v.std()) if len(load_v) > 0 else 0.0
        else:
            self.max_train_cpu_load = 100.0
            self.train_cpu_load_std = 0.0

        self.fitted = True
        return True

    # Compute per-host residual mean/std for power-bias calibration.
    def fit_host_stats(self, host_train_dfs: dict) -> None:
        if not self.fitted or not self.power_col:
            return
        for host, tr_df in host_train_dfs.items():
            try:
                X = self.build_X(tr_df)
                if X is None or X.shape[0] < 50:
                    continue
                y = (
                    pd.to_numeric(tr_df[self.power_col], errors="coerce")
                    .fillna(0)
                    .values
                )
                valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
                if valid.sum() < 50:
                    continue
                X_s = self.scaler.transform(np.nan_to_num(X[valid], nan=0.0))
                y_pred = self.model.predict(X_s)
                resid = y[valid] - y_pred
                mu = float(resid.mean())
                sd = float(resid.std())
                # Guard against degenerate hosts (flat idle): keep pooled stats.
                if sd < 0.1:
                    continue
                self.host_resid_mean[host] = mu
                self.host_resid_std[host] = sd
            except Exception:
                continue  # skip this host; use pooled fallback

    # Predict per-row power residuals and host-calibrated z-scores.
    def predict_residual(
        self, df: pd.DataFrame, hostname: Optional[str] = None
    ) -> pd.DataFrame:
        if not self.fitted or not self.power_col:
            n = len(df)
            return pd.DataFrame(
                {
                    "power_residual_w": np.nan * np.ones(n, dtype=np.float32),
                    "power_residual_z": np.nan * np.ones(n, dtype=np.float32),
                    "power_anomaly": np.zeros(n, dtype=bool),
                }
            )

        X = self.build_X(df)
        if X is None:
            n = len(df)
            return pd.DataFrame(
                {
                    "power_residual_w": np.nan * np.ones(n, dtype=np.float32),
                    "power_residual_z": np.nan * np.ones(n, dtype=np.float32),
                    "power_anomaly": np.zeros(n, dtype=bool),
                }
            )

        # Fill NaN features with 0 (idle state)
        X = np.nan_to_num(X, nan=0.0)
        X_s = self.scaler.transform(X)
        y_pred = np.nan_to_num(
            self.model.predict(X_s).astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        y_actual = pd.to_numeric(df[self.power_col], errors="coerce").values.astype(
            np.float32
        )
        residual = y_actual - y_pred

        if hostname is not None and hostname in self.host_resid_mean:
            host_bias = np.float32(self.host_resid_mean[hostname])
            residual = residual - host_bias
            host_std = self.host_resid_std.get(hostname, self.train_resid_std)
        else:
            host_std = self.train_resid_std

        resid_z = adaptive_resid_z(residual, df, host_std)

        return pd.DataFrame(
            {
                "power_residual_w": residual,
                "power_residual_z_raw": (residual / host_std).astype(np.float32),
                "power_residual_z": resid_z.astype(np.float32),
                "power_anomaly": np.abs(resid_z) > POWER_Z_THRESH,
            }
        )
