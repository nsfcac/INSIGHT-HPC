from __future__ import annotations

import numpy as np
import pandas as pd

from shared.utils.io_utils import save_parquet
from offline.utils.synthetic_injector_module.constants import (
    INLET_KEYWORDS,
    POWER_KEYWORDS,
    PDU_KEYWORDS,
)


class SyntheticWorkloadInjections:
    # Linearly ramp inlet temperatures upward across the window.
    def gradual_thermal_drift(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 48))
        rate = float(spec.get("drift_rate_c_per_hr", 0.05))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)

        # Find matching window
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in [{start}, +{duration_hr}h] for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "gradual_thermal_drift")

        # Ramp: 0 at start → rate * duration_hr at end
        minutes_elapsed = (ts[mask] - start).dt.total_seconds().to_numpy() / 60.0
        delta_c = (minutes_elapsed / 60.0) * rate  # linear ramp in °C

        # Apply to all inlet temperature columns
        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in INLET_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                df.loc[mask, c] = (
                    pd.to_numeric(df.loc[mask, c], errors="coerce") + delta_c
                ).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No inlet temperature columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"drift_rate_c_per_hr": rate, "total_delta_c": rate * duration_hr},
            target_detectors=["IF_slope7d", "IF_slope1d", "thermal_residual"],
        )

    # Ramp inlet temperatures up while leaving fan RPM unchanged.
    def cooling_failure(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 1.0))
        delta_c = float(spec.get("delta_c", 5.0))
        ramp_min = float(spec.get("ramp_min", 30.0))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "cooling_failure")

        minutes_elapsed = (ts[mask] - start).dt.total_seconds().to_numpy() / 60.0

        # Temperature: ramp up then stay flat
        temp_delta = np.where(
            minutes_elapsed <= ramp_min,
            (minutes_elapsed / ramp_min) * delta_c,
            delta_c,
        )

        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in INLET_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                df.loc[mask, c] = (
                    pd.to_numeric(df.loc[mask, c], errors="coerce") + temp_delta
                ).astype("float32")
                cols_modified.append(c)
        # Fan RPM is intentionally NOT modified (simulating fan not responding)

        if not cols_modified:
            raise ValueError(f"No inlet temperature columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"delta_c": delta_c, "ramp_min": ramp_min},
            target_detectors=["TEMP_FAN_DECOUPLING_constraint"],
        )

    # Scale one node's system power up so it diverges from its peers.
    def peer_node_divergence(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 2.0))
        factor = float(spec.get("power_factor", 1.4))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "peer_node_divergence")

        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in POWER_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce")
                df.loc[mask, c] = (orig * factor).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No system power columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_factor": factor},
            target_detectors=["multi_node_coherence"],
        )

    # Add a fixed power offset during an idle window.
    def idle_power_fault(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 4.0))
        power_delta_w = float(spec.get("power_delta_w", 150))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "idle_power_fault")

        # Warn if jobs are running during the window
        job_col = next(
            (
                c
                for c in df.columns
                if "active_job_count" in c.lower() or "is_running_job" in c.lower()
            ),
            None,
        )
        if job_col is not None:
            n_busy = int(
                (
                    pd.to_numeric(df.loc[mask, job_col], errors="coerce").fillna(0) > 0
                ).sum()
            )
            if n_busy > 0:
                print(
                    f"  [WARN] {hostname}: {n_busy} busy rows in idle_power_fault window — "
                    "anomaly context may not be NO_JOBS"
                )

        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in POWER_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce")
                df.loc[mask, c] = (orig + power_delta_w).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No system power columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_delta_w": power_delta_w},
            target_detectors=["IF", "power_residual", "NO_JOBS_context"],
        )

    # Scale system power up during a running-job window.
    def job_excess_power(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 2.0))
        factor = float(spec.get("power_factor", 1.3))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "job_excess_power")

        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in POWER_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce")
                df.loc[mask, c] = (orig * factor).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No system power columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_factor": factor},
            target_detectors=["IF", "power_residual", "JOB_OVER_EXPECTATION_context"],
        )

    # Briefly scale PDU power to simulate a measurement glitch.
    def measurement_glitch(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_min = float(spec.get("duration_min", 15.0))
        factor = float(spec.get("pdu_factor", 1.25))
        end = start + pd.Timedelta(minutes=duration_min)

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < end)
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "measurement_glitch")

        cols_modified = []
        for c in df.columns:
            if any(k in c.lower() for k in PDU_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce")
                df.loc[mask, c] = (orig * factor).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(
                f"No PDU columns found for {hostname}. "
                "This injection type requires PDU columns in the master table. "
                "Consider targeting an infra/pdu node instead."
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            end,
            modified_cols=cols_modified,
            params={"pdu_factor": factor, "duration_min": duration_min},
            target_detectors=["CROSSPLANE_DISAGREEMENT_constraint"],
        )
