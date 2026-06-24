from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io_utils import save_parquet
from src.utils.synthetic_injector_module.constants import FAN_KEYWORDS, INLET_KEYWORDS


class SyntheticSensorInjections:
    # Inject a gradually growing memory-usage leak across the window.
    def memory_leak(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 12.0))
        target_delta = float(spec.get("target_delta_pct", 80.0))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "memory_leak")

        minutes_elapsed = (ts[mask] - start).dt.total_seconds().to_numpy() / 60.0
        total_min = duration_hr * 60.0
        delta_pct = (minutes_elapsed / total_min) * target_delta  # 0 → target_delta

        cols_modified = []
        for c in df.columns:
            cl = c.lower()
            if "memoryusage" in cl:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                df.loc[mask, c] = np.clip(orig + delta_pct, 0.0, 100.0).astype(
                    "float32"
                )
                cols_modified.append(c)
            elif "freememory" in cl:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                drop_frac = (minutes_elapsed / total_min) * (target_delta / 100.0)
                reduced = orig * (1.0 - drop_frac)
                df.loc[mask, c] = np.clip(reduced, 0.0, None).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No memoryusage/freememory columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_delta_pct": target_delta, "duration_hr": duration_hr},
            target_detectors=["IF_slope1d", "IF_slope7d", "IF_rmean15"],
        )

    # Pin CPU usage and power to near-maximum across the window.
    def cpu_exhaustion(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 3.0))
        target_pct = float(spec.get("target_cpu_pct", 99.0))
        target_pwr_w = float(spec.get("target_cpu_power_w", 500.0))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "cpu_exhaustion")

        cols_modified = []
        for c in df.columns:
            cl = c.lower()
            if "gpu" in cl:
                continue
            if "cpuusage" in cl:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                # max with target floor; preserve any value already above target
                new_vals = np.where(
                    np.isnan(orig), target_pct, np.maximum(orig, target_pct)
                )
                df.loc[mask, c] = np.clip(new_vals, 0.0, 100.0).astype("float32")
                cols_modified.append(c)
            elif "cpupower" in cl or "totalcpupower" in cl:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                new_vals = np.where(
                    np.isnan(orig), target_pwr_w, np.maximum(orig, target_pwr_w)
                )
                df.loc[mask, c] = new_vals.astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No cpuusage/cpupower columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_cpu_pct": target_pct, "target_cpu_power_w": target_pwr_w},
            target_detectors=[
                "IF",
                "power_residual",
                "NO_JOBS_context",
                "JOB_OVER_EXPECTATION_context",
            ],
        )

    # Cut fan RPM, optionally raising inlet temperature, across the window.
    def fan_rpm_drop(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 1.0))
        rpm_factor = float(spec.get("rpm_factor", 0.2))
        rpm_floor = float(spec.get("rpm_floor", 0.0))
        temp_bump_c = float(spec.get("temp_bump_c", 0.0))
        ramp_min = float(spec.get("ramp_min", 15.0))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "fan_rpm_drop")

        cols_modified = []
        # Reduce fan RPM
        for c in df.columns:
            if any(k in c.lower() for k in FAN_KEYWORDS):
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                reduced = np.maximum(orig * rpm_factor, rpm_floor)
                df.loc[mask, c] = reduced.astype("float32")
                cols_modified.append(c)

        # Optional coupled inlet-temperature rise
        if temp_bump_c > 0.0:
            minutes_elapsed = (ts[mask] - start).dt.total_seconds().to_numpy() / 60.0
            temp_delta = np.where(
                minutes_elapsed <= ramp_min,
                (minutes_elapsed / max(ramp_min, 1e-6)) * temp_bump_c,
                temp_bump_c,
            )
            for c in df.columns:
                if any(k in c.lower() for k in INLET_KEYWORDS):
                    if not pd.api.types.is_float_dtype(df[c]):
                        df[c] = df[c].astype("float32")
                    orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                    df.loc[mask, c] = (orig + temp_delta).astype("float32")
                    cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No fan/rpm columns found for {hostname}")

        targets = ["fan_rpm_low_baseline"]
        if temp_bump_c > 0.0:
            targets.append("TEMP_FAN_DECOUPLING_constraint")
        targets.append("IF")

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={
                "rpm_factor": rpm_factor,
                "rpm_floor": rpm_floor,
                "temp_bump_c": temp_bump_c,
                "ramp_min": ramp_min,
            },
            target_detectors=targets,
        )

    # Blank matching sensors to NaN across the window.
    def sensor_dropout(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 2.0))
        target_keywords = list(
            spec.get(
                "target_keywords",
                ["systeminputpower", "inlet", "rpmreading"],
            )
        )
        target_keywords = [k.lower() for k in target_keywords]

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "sensor_dropout")

        cols_modified = []
        for c in df.columns:
            if not c.endswith("_avg"):
                continue
            cl = c.lower()
            if any(k in cl for k in target_keywords):
                # Cast to float to allow NaN assignment (int columns fail)
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                df.loc[mask, c] = np.nan
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(
                f"No columns matched target_keywords={target_keywords} "
                f"for {hostname}"
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_keywords": target_keywords, "duration_hr": duration_hr},
            target_detectors=["nan_streak_feature", "IF", "baseline_min_coverage"],
        )

    # Ramp GPU temperatures up to a peak delta across the window.
    def gpu_thermal_runaway(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 1.0))
        peak_delta_c = float(spec.get("peak_delta_c", 15.0))
        ramp_min = float(spec.get("ramp_min", 15.0))
        target_gpus = [s.lower() for s in spec.get("target_gpus", [])] or None

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "gpu_thermal_runaway")

        minutes_elapsed = (ts[mask] - start).dt.total_seconds().to_numpy() / 60.0
        temp_delta = np.where(
            minutes_elapsed <= ramp_min,
            (minutes_elapsed / max(ramp_min, 1e-6)) * peak_delta_c,
            peak_delta_c,
        )

        cols_modified = []
        for c in df.columns:
            cl = c.lower()
            if "gputemp" not in cl:
                continue
            if target_gpus and not any(t in cl for t in target_gpus):
                continue
            if not pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float32")
            orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
            df.loc[mask, c] = (orig + temp_delta).astype("float32")
            cols_modified.append(c)

        if not cols_modified:
            raise ValueError(
                f"No gputemp columns found for {hostname}. "
                "This injection requires an h100 node with GPU temperature sensors."
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={
                "peak_delta_c": peak_delta_c,
                "ramp_min": ramp_min,
                "target_gpus": target_gpus,
            },
            target_detectors=[
                "IF_gputemp",
                "thermal_residual",
                "baseline_gpu_temp_high",
            ],
        )

    # Freeze matching sensors at a fixed or look-back-median value across the window.
    def sensor_stuck_at_value(self, spec: dict) -> dict:
        hostname = spec["hostname"]
        start = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 2.0))
        target_keywords = [k.lower() for k in spec.get("target_keywords", [])]
        if not target_keywords:
            raise ValueError(
                "sensor_stuck_at_value requires non-empty 'target_keywords' "
                "(e.g. ['inlettemp']). There's no safe default."
            )
        stuck_value_spec = spec.get("stuck_value", None)

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "sensor_stuck_at_value")

        # Look-back window for default stuck_value (24 h before start)
        lb_start = start - pd.Timedelta(hours=24)
        lb_mask = (ts >= lb_start) & (ts < start)

        cols_modified = []
        value_used: dict[str, float] = {}
        for c in df.columns:
            if not c.endswith("_avg"):
                continue
            cl = c.lower()
            if not any(k in cl for k in target_keywords):
                continue
            if stuck_value_spec is None:
                lb_vals = pd.to_numeric(df.loc[lb_mask, c], errors="coerce").dropna()
                if lb_vals.empty:
                    # Skip columns with no look-back data
                    continue
                val = float(lb_vals.median())
            else:
                val = float(stuck_value_spec)
            if not pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype("float32")
            df.loc[mask, c] = val
            cols_modified.append(c)
            value_used[c] = val

        if not cols_modified:
            raise ValueError(
                f"No columns matched target_keywords={target_keywords} "
                f"for {hostname} (or no look-back data for median)."
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec,
            comp,
            hostname,
            start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={
                "target_keywords": target_keywords,
                "stuck_value": stuck_value_spec,
                "values_used": value_used,
                "duration_hr": duration_hr,
            },
            target_detectors=[
                "IF_slope1d",
                "IF_rstd15",
                "baseline_stuck_sensor",
                "thermal_residual",
                "power_residual",
            ],
        )
