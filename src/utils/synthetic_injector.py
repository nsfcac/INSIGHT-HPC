from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, save_parquet


INJECTION_TYPES = {
    "gradual_thermal_drift",
    "cooling_failure",
    "peer_node_divergence",
    "idle_power_fault",
    "job_excess_power",
    "measurement_glitch",
    "memory_leak",
    "cpu_exhaustion",
    "fan_rpm_drop",
    "sensor_dropout",
    "gpu_thermal_runaway",
    "sensor_stuck_at_value",
}

# Keyword patterns to locate relevant columns in master tables
INLET_KEYWORDS   = ["inlettemp", "inlet_temp", "inlettemperature", "systeminlettemp"]
FAN_KEYWORDS     = ["fanspeed", "fan_speed", "fanrpm", "fan_rpm",
                     "tachometerreading",
                     # iDRAC naming: rpmreading__fan.embedded.1a_fan_avg etc.
                     "rpmreading", "fan.embedded"]
POWER_KEYWORDS   = ["systeminputpower", "system_input_power", "powerconsumption",
                     "power_consumption", "inputpower"]
PDU_KEYWORDS     = ["pdu_power", "pdupower", "rack_power", "rackpower",
                     "pdu__pdu", "pdu_avg"]

JOB_CONTEXT_EXPECT: dict[str, set[str]] = {
    "idle_power_fault":       {"idle"},
    "cpu_exhaustion":         {"idle", "running"},   # supports either flavor
    "fan_rpm_drop":           {"idle", "running"},
    "gradual_thermal_drift":  {"idle", "running"},
    "sensor_dropout":         {"idle", "running"},
    "sensor_stuck_at_value":  {"idle", "running"},
    "cooling_failure":        {"idle", "running"},
    "memory_leak":            {"idle", "running", "single_host", "multi_host"},
    "job_excess_power":       {"running", "single_host"},
    "peer_node_divergence":   {"multi_host"},
    "measurement_glitch":     {"idle", "running"},    # PDU is infra-level
    "gpu_thermal_runaway":    {"idle", "running", "single_host", "multi_host"},
}

# Classify a job segment as idle/single-host/multi-host.
def classify_segment(seg_row: pd.Series, multi_host_jobs: set[int]) -> str:
    if bool(seg_row.get("is_idle", False)):
        return "idle"
    jid = seg_row.get("job_id")
    try:
        jid_int = int(jid) if pd.notna(jid) else None
    except (TypeError, ValueError):
        jid_int = None
    if jid_int is not None and jid_int in multi_host_jobs:
        return "multi_host"
    return "single_host"

class SyntheticInjector:

    def __init__(self, config_path: str = "configs/config.yaml", injected_master_dir: str | None = None, manifest_path: str | None = None) -> None:
        self.cfg = load_config(config_path)
        master_base = Path(self.cfg["paths"]["master"])

        if injected_master_dir is not None:
            self.dst_dir = Path(injected_master_dir)
            self.src_dir = self.derive_clean_source(master_base, self.dst_dir)
        elif master_base.name.endswith("_injected"):
            # paths.master was already redirected → dst is that, src is the sibling
            self.dst_dir = master_base
            self.src_dir = master_base.parent / master_base.name.removesuffix("_injected")
        else:
            self.src_dir = master_base
            self.dst_dir = master_base.parent / (master_base.name + "_injected")

        # Manifest: prefer config's ground_truth_dir so INSIGHT_HPC_RUN_SUFFIX flows through; fall back to phase4.output_dir/ground_truth.
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            # Use the top-level ground_truth_dir (canonical, never suffixed). Fall back to legacy phase4.ground_truth_dir for backwards compat.
            gt_dir = self.cfg.get("ground_truth_dir") or \
                     self.cfg.get("phase4", {}).get("ground_truth_dir") or \
                     "data/ground_truth"
            self.manifest_path = Path(gt_dir) / "injected_events.json"
        self.events: list[dict] = []

    @staticmethod
    def derive_clean_source(master_base: Path, dst_dir: Path) -> Path:
        if dst_dir.name.endswith("_injected"):
            return dst_dir.parent / dst_dir.name.removesuffix("_injected")
        return master_base

    def run(self, specs: list[dict]) -> list[dict]:
        self.events = []
        self.validate_job_context(specs)
        self.prepare_destination()

        for i, spec in enumerate(specs):
            itype = spec.get("type")
            if itype is None:
                continue  # comment-only separator entry
            if itype not in INJECTION_TYPES:
                raise ValueError(
                    f"Unknown injection type '{itype}'. "
                    f"Valid types: {sorted(INJECTION_TYPES)}"
                )
            print(f"[inject] {i+1}/{len(specs)}  type={itype}  "
                  f"host={spec.get('hostname', '?')}")
            try:
                event = self.dispatch_injection(spec)
                self.events.append(event)
            except Exception as exc:
                print(f"  [WARN] injection {i+1} failed: {exc}")

        self.save_manifest()
        print(f"\n[inject] Done. {len(self.events)} events written to "
              f"{self.manifest_path}")
        return self.events


    def load_job_segments(self) -> pd.DataFrame | None:
        p2 = self.cfg.get("phase2", {}).get("output_dir", "data/phase2")
        seg_path = Path(p2) / "job_segments.parquet"
        if not seg_path.exists():
            # Try no_irc variant as fallback
            alt = Path(str(p2) + "_no_irc") / "job_segments.parquet"
            if alt.exists():
                seg_path = alt
            else:
                return None
        try:
            segs = pd.read_parquet(seg_path)
        except Exception as exc:
            print(f"  [WARN] could not load {seg_path}: {exc}")
            return None
        segs["seg_start"] = pd.to_datetime(segs["seg_start"], utc=True)
        segs["seg_end"]   = pd.to_datetime(segs["seg_end"],   utc=True)
        return segs

    def validate_job_context(self, specs: list[dict]) -> None:
        segs = self.load_job_segments()
        if segs is None:
            print("[inject] job_segments.parquet not found — skipping "
                  "job-context validation. Run phase 2 first for stricter checks.")
            return

        print("[inject] Validating job context against data/phase2/job_segments.parquet")
        # Derive multi-host jobs once: any job_id with >1 distinct hostname in segs
        hosts_per_job = (
            segs.dropna(subset=["job_id"])
                .groupby("job_id")["hostname"].nunique()
        )
        multi_host_jobs = set(int(j) for j in hosts_per_job[hosts_per_job > 1].index)
        print(f"[inject]   {len(multi_host_jobs)} multi-host jobs detected in phase2")
        n_warn = 0
        for i, spec in enumerate(specs):
            itype = spec.get("type")
            if itype is None:
                continue  # comment-only separator entry
            expect = JOB_CONTEXT_EXPECT.get(itype, set())
            if not expect:
                continue
            host = spec.get("hostname", "")
            if str(host).startswith("pdu-"):
                # PDU injections: check rack is active, not per-host segment class
                rack = host.split("-")[1] if "-" in host else None
                if rack is None:
                    continue
                start = pd.Timestamp(spec["start"], tz="UTC")
                dur   = float(spec.get("duration_min", spec.get("duration_hr", 0) * 60) or 0)
                end   = start + pd.Timedelta(minutes=dur)
                rack_host_prefix = f"rpc-{rack}-"
                rack_segs = segs[
                    segs["hostname"].astype(str).str.startswith(rack_host_prefix) &
                    (segs["seg_start"] < end) &
                    (segs["seg_end"]   > start) &
                    (~segs["is_idle"])
                ]
                if rack_segs.empty:
                    print(f"  [WARN] spec {i+1} ({itype}, {host}): no running "
                          f"rack-{rack} jobs overlap the window — "
                          f"CROSSPLANE_DISAGREEMENT may not trigger cleanly.")
                    n_warn += 1
                else:
                    n_nodes = rack_segs["hostname"].nunique()
                    job_ids = sorted({int(j) for j in rack_segs["job_id"].dropna().unique()})
                    print(f"  [ok]   spec {i+1} ({itype}, {host}): rack-{rack} "
                          f"has {n_nodes} busy node(s), jobs={job_ids[:3]}")
                continue

            start = pd.Timestamp(spec["start"], tz="UTC")
            dur_hr = float(spec.get("duration_hr", 0))
            dur_min = float(spec.get("duration_min", 0))
            end = start + pd.Timedelta(hours=dur_hr, minutes=dur_min)

            host_segs = segs[
                (segs["hostname"] == host) &
                (segs["seg_start"] < end) &
                (segs["seg_end"]   > start)
            ].copy()

            if host_segs.empty:
                print(f"  [WARN] spec {i+1} ({itype}, {host}): NO phase-2 segment "
                      f"covers [{start}, {end}] — hostname/window may be wrong.")
                n_warn += 1
                continue

            # Compute overlap duration per segment and pick dominant class
            host_segs["ov_start"] = host_segs["seg_start"].where(
                host_segs["seg_start"] > start, start
            )
            host_segs["ov_end"] = host_segs["seg_end"].where(
                host_segs["seg_end"] < end, end
            )
            host_segs["ov_sec"] = (
                host_segs["ov_end"] - host_segs["ov_start"]
            ).dt.total_seconds().clip(lower=0)
            host_segs["klass"] = host_segs.apply(
                lambda r: classify_segment(r, multi_host_jobs), axis=1
            )
            by_class = host_segs.groupby("klass")["ov_sec"].sum().sort_values(
                ascending=False
            )
            dominant = by_class.index[0]

            if dominant not in expect:
                # Special case: "running" is satisfied by single_host or multi_host
                if ("running" in expect) and dominant in ("single_host", "multi_host"):
                    pass
                else:
                    class_mix = dict(by_class.astype(int))
                    print(f"  [WARN] spec {i+1} ({itype}, {host}): dominant "
                          f"segment class is '{dominant}', expected one of "
                          f"{sorted(expect)}. Overlap mix (sec): {class_mix}")
                    n_warn += 1
                    continue

            # Emit an INFO line on success so reviewers can confirm
            job_ids = [
                int(j) for j in host_segs.loc[host_segs["klass"] == dominant, "job_id"]
                .dropna().unique().tolist()
            ]
            print(f"  [ok]   spec {i+1} ({itype}, {host}): landed in "
                  f"'{dominant}' segment" +
                  (f" (job_ids={job_ids[:3]})" if job_ids else " (idle)"))

        if n_warn > 0:
            print(f"[inject] {n_warn} spec(s) flagged by job-context validator. "
                  "Proceeding — fix the specs if the warnings matter to the paper.")
        else:
            print("[inject] all specs passed job-context validation.")


    def dispatch_injection(self, spec: dict) -> dict:
        itype = spec["type"]
        if itype == "gradual_thermal_drift":
            return self.gradual_thermal_drift(spec)
        if itype == "cooling_failure":
            return self.cooling_failure(spec)
        if itype == "peer_node_divergence":
            return self.peer_node_divergence(spec)
        if itype == "idle_power_fault":
            return self.idle_power_fault(spec)
        if itype == "job_excess_power":
            return self.job_excess_power(spec)
        if itype == "measurement_glitch":
            return self.measurement_glitch(spec)
        if itype == "memory_leak":
            return self.memory_leak(spec)
        if itype == "cpu_exhaustion":
            return self.cpu_exhaustion(spec)
        if itype == "fan_rpm_drop":
            return self.fan_rpm_drop(spec)
        if itype == "sensor_dropout":
            return self.sensor_dropout(spec)
        if itype == "gpu_thermal_runaway":
            return self.gpu_thermal_runaway(spec)
        if itype == "sensor_stuck_at_value":
            return self.sensor_stuck_at_value(spec)
        raise ValueError(f"Unhandled type: {itype}")

    @staticmethod
    def warn_if_train_split(df: pd.DataFrame, mask: pd.Series, hostname: str, itype: str) -> None:
        if "split" not in df.columns:
            return
        n_train = int((df.loc[mask, "split"].astype(str) == "train").sum())
        if n_train > 0:
            print(f"  [WARN] {hostname}: {n_train} rows in {itype} window "
                  f"are split='train' — will contaminate model training. "
                  f"Shift the injection window past window.train_end.")


    def gradual_thermal_drift(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 48))
        rate         = float(spec.get("drift_rate_c_per_hr", 0.05))

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
                df.loc[mask, c] = (pd.to_numeric(df.loc[mask, c], errors="coerce")
                                   + delta_c).astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(f"No inlet temperature columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"drift_rate_c_per_hr": rate, "total_delta_c": rate * duration_hr},
            target_detectors=["IF_slope7d", "IF_slope1d", "thermal_residual"],
        )

    def cooling_failure(self, spec: dict) -> dict:
        hostname    = spec["hostname"]
        start       = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 1.0))
        delta_c     = float(spec.get("delta_c", 5.0))
        ramp_min    = float(spec.get("ramp_min", 30.0))

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
                df.loc[mask, c] = (pd.to_numeric(df.loc[mask, c], errors="coerce")
                                   + temp_delta).astype("float32")
                cols_modified.append(c)
        # Fan RPM is intentionally NOT modified (simulating fan not responding)

        if not cols_modified:
            raise ValueError(f"No inlet temperature columns found for {hostname}")

        save_parquet(df, dst_path)
        return self.build_event(
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"delta_c": delta_c, "ramp_min": ramp_min},
            target_detectors=["TEMP_FAN_DECOUPLING_constraint"],
        )

    def peer_node_divergence(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 2.0))
        factor       = float(spec.get("power_factor", 1.4))

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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_factor": factor},
            target_detectors=["multi_node_coherence"],
        )

    def idle_power_fault(self, spec: dict) -> dict:
        hostname      = spec["hostname"]
        start         = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr   = float(spec.get("duration_hr", 4.0))
        power_delta_w = float(spec.get("power_delta_w", 150))

        df, src_path, dst_path, comp = self.load_node(hostname)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        mask = (ts >= start) & (ts < start + pd.Timedelta(hours=duration_hr))
        if not mask.any():
            raise ValueError(f"No rows in window for {hostname}")
        self.warn_if_train_split(df, mask, hostname, "idle_power_fault")

        # Warn if jobs are running during the window
        job_col = next((c for c in df.columns
                        if "active_job_count" in c.lower() or "is_running_job" in c.lower()),
                       None)
        if job_col is not None:
            n_busy = int((pd.to_numeric(df.loc[mask, job_col], errors="coerce")
                          .fillna(0) > 0).sum())
            if n_busy > 0:
                print(f"  [WARN] {hostname}: {n_busy} busy rows in idle_power_fault window — "
                      "anomaly context may not be NO_JOBS")

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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_delta_w": power_delta_w},
            target_detectors=["IF", "power_residual", "NO_JOBS_context"],
        )

    def job_excess_power(self, spec: dict) -> dict:
        hostname    = spec["hostname"]
        start       = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr = float(spec.get("duration_hr", 2.0))
        factor      = float(spec.get("power_factor", 1.3))

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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"power_factor": factor},
            target_detectors=["IF", "power_residual", "JOB_OVER_EXPECTATION_context"],
        )

    def measurement_glitch(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_min = float(spec.get("duration_min", 15.0))
        factor       = float(spec.get("pdu_factor", 1.25))
        end          = start + pd.Timedelta(minutes=duration_min)

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
            spec, comp, hostname, start, end,
            modified_cols=cols_modified,
            params={"pdu_factor": factor, "duration_min": duration_min},
            target_detectors=["CROSSPLANE_DISAGREEMENT_constraint"],
        )

    def memory_leak(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 12.0))
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
                df.loc[mask, c] = np.clip(orig + delta_pct, 0.0, 100.0).astype("float32")
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
            raise ValueError(
                f"No memoryusage/freememory columns found for {hostname}"
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_delta_pct": target_delta,
                    "duration_hr": duration_hr},
            target_detectors=["IF_slope1d", "IF_slope7d", "IF_rmean15"],
        )

    def cpu_exhaustion(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 3.0))
        target_pct   = float(spec.get("target_cpu_pct", 99.0))
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
                new_vals = np.where(np.isnan(orig), target_pct,
                                    np.maximum(orig, target_pct))
                df.loc[mask, c] = np.clip(new_vals, 0.0, 100.0).astype("float32")
                cols_modified.append(c)
            elif "cpupower" in cl or "totalcpupower" in cl:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].astype("float32")
                orig = pd.to_numeric(df.loc[mask, c], errors="coerce").to_numpy()
                new_vals = np.where(np.isnan(orig), target_pwr_w,
                                    np.maximum(orig, target_pwr_w))
                df.loc[mask, c] = new_vals.astype("float32")
                cols_modified.append(c)

        if not cols_modified:
            raise ValueError(
                f"No cpuusage/cpupower columns found for {hostname}"
            )

        save_parquet(df, dst_path)
        return self.build_event(
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_cpu_pct": target_pct,
                    "target_cpu_power_w": target_pwr_w},
            target_detectors=["IF", "power_residual",
                              "NO_JOBS_context", "JOB_OVER_EXPECTATION_context"],
        )

    def fan_rpm_drop(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 1.0))
        rpm_factor   = float(spec.get("rpm_factor", 0.2))
        rpm_floor    = float(spec.get("rpm_floor", 0.0))
        temp_bump_c  = float(spec.get("temp_bump_c", 0.0))
        ramp_min     = float(spec.get("ramp_min", 15.0))

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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"rpm_factor": rpm_factor, "rpm_floor": rpm_floor,
                    "temp_bump_c": temp_bump_c, "ramp_min": ramp_min},
            target_detectors=targets,
        )

    def sensor_dropout(self, spec: dict) -> dict:
        hostname        = spec["hostname"]
        start           = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr     = float(spec.get("duration_hr", 2.0))
        target_keywords = list(spec.get(
            "target_keywords",
            ["systeminputpower", "inlet", "rpmreading"],
        ))
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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_keywords": target_keywords,
                    "duration_hr": duration_hr},
            target_detectors=["nan_streak_feature", "IF",
                              "baseline_min_coverage"],
        )

    def gpu_thermal_runaway(self, spec: dict) -> dict:
        hostname     = spec["hostname"]
        start        = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr  = float(spec.get("duration_hr", 1.0))
        peak_delta_c = float(spec.get("peak_delta_c", 15.0))
        ramp_min     = float(spec.get("ramp_min", 15.0))
        target_gpus  = [s.lower() for s in spec.get("target_gpus", [])] or None

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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"peak_delta_c": peak_delta_c, "ramp_min": ramp_min,
                    "target_gpus": target_gpus},
            target_detectors=["IF_gputemp", "thermal_residual",
                              "baseline_gpu_temp_high"],
        )

    def sensor_stuck_at_value(self, spec: dict) -> dict:
        hostname        = spec["hostname"]
        start           = pd.Timestamp(spec["start"], tz="UTC")
        duration_hr     = float(spec.get("duration_hr", 2.0))
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
            spec, comp, hostname, start,
            start + pd.Timedelta(hours=duration_hr),
            modified_cols=cols_modified,
            params={"target_keywords": target_keywords,
                    "stuck_value": stuck_value_spec,
                    "values_used": value_used,
                    "duration_hr": duration_hr},
            target_detectors=["IF_slope1d", "IF_rstd15",
                              "baseline_stuck_sensor",
                              "thermal_residual", "power_residual"],
        )


    def prepare_destination(self) -> None:
        if self.dst_dir.exists():
            print(f"[inject] Removing stale injected dir → {self.dst_dir}")
            shutil.rmtree(self.dst_dir)
        t0 = time.perf_counter()
        print(f"[inject] Copying master → {self.dst_dir} …")
        shutil.copytree(self.src_dir, self.dst_dir, dirs_exist_ok=False)
        print(f"  Done in {time.perf_counter()-t0:.1f}s")

    def locate_parquet(self, hostname: str) -> tuple[Path, str]:
        for comp_cfg in self.cfg["components"]:
            comp = comp_cfg["name"]
            comp_dir = self.src_dir / comp
            if not comp_dir.exists():
                continue
            pattern = "**/*.parquet" if comp == "infra" else "*.parquet"
            for p in comp_dir.glob(pattern):
                if p.stem == hostname:
                    return p, comp
        raise FileNotFoundError(
            f"No master parquet found for hostname '{hostname}' under {self.src_dir}"
        )

    def load_node(self, hostname: str) -> tuple[pd.DataFrame, Path, Path, str]:
        src_path, comp = self.locate_parquet(hostname)
        rel      = src_path.relative_to(self.src_dir)
        dst_path = self.dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(dst_path, engine="pyarrow")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, src_path, dst_path, comp

    @staticmethod
    def build_event(spec: dict, comp: str, hostname: str, start: pd.Timestamp, end: pd.Timestamp, modified_cols: list[str], params: dict, target_detectors: list[str]) -> dict:
        return {
            "type":             spec["type"],
            "component":        comp,
            "hostname":         hostname,
            "start_time":       start.isoformat(),
            "end_time":         end.isoformat(),
            "modified_columns": modified_cols,
            "injection_params": params,
            "target_detectors": target_detectors,
            "spec":             spec,
        }

    def save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)
        print(f"[inject] Manifest → {self.manifest_path}")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Inject synthetic anomalies into PHASE master tables."
    )
    parser.add_argument(
        "--spec-file", required=True,
        help="JSON file containing a list of injection spec dicts."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dst-dir", default=None,
                        help="Override destination directory for injected parquets.")
    parser.add_argument("--manifest", default=None,
                        help="Override manifest output path.")
    args = parser.parse_args()

    with open(args.spec_file) as f:
        specs: list[dict] = json.load(f)

    injector = SyntheticInjector(
        config_path=args.config,
        injected_master_dir=args.dst_dir,
        manifest_path=args.manifest,
    )
    events = injector.run(specs)
    print(f"\nInjected {len(events)} events.")
