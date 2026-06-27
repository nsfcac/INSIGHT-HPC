from __future__ import annotations

import json, shutil, time
from pathlib import Path

import pandas as pd

from shared.utils.io_utils import load_config
from offline.utils.synthetic_injector_module.constants import (
    INJECTION_TYPES,
    JOB_CONTEXT_EXPECT,
    classify_segment,
)
from offline.utils.synthetic_injector_module.sensor_injections import (
    SyntheticSensorInjections,
)
from offline.utils.synthetic_injector_module.workload_injections import (
    SyntheticWorkloadInjections,
)


class SyntheticInjector(SyntheticSensorInjections, SyntheticWorkloadInjections):
    def __init__(
        self,
        config_path: str = "shared/configs/config.yaml",
        injected_master_dir: str | None = None,
        manifest_path: str | None = None,
    ) -> None:
        self.cfg = load_config(config_path)
        master_base = Path(self.cfg["paths"]["master"])

        if injected_master_dir is not None:
            self.dst_dir = Path(injected_master_dir)
            self.src_dir = self.derive_clean_source(master_base, self.dst_dir)
        elif master_base.name.endswith("_injected"):
            # paths.master was already redirected → dst is that, src is the sibling
            self.dst_dir = master_base
            self.src_dir = master_base.parent / master_base.name.removesuffix(
                "_injected"
            )
        else:
            self.src_dir = master_base
            self.dst_dir = master_base.parent / (master_base.name + "_injected")

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            gt_dir = (
                self.cfg.get("ground_truth_dir")
                or self.cfg.get("phase4", {}).get("ground_truth_dir")
                or "offline/data/ground_truth"
            )
            self.manifest_path = Path(gt_dir) / "injected_events.json"
        self.events: list[dict] = []

    # Derive the clean source dir corresponding to an injected destination dir.
    @staticmethod
    def derive_clean_source(master_base: Path, dst_dir: Path) -> Path:
        if dst_dir.name.endswith("_injected"):
            return dst_dir.parent / dst_dir.name.removesuffix("_injected")
        return master_base

    # Validate, inject each spec into the copied master, and write the manifest.
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
            print(
                f"[inject] {i+1}/{len(specs)}  type={itype}  "
                f"host={spec.get('hostname', '?')}"
            )
            try:
                event = self.dispatch_injection(spec)
                self.events.append(event)
            except Exception as exc:
                print(f"  [WARN] injection {i+1} failed: {exc}")

        self.save_manifest()
        print(
            f"\n[inject] Done. {len(self.events)} events written to "
            f"{self.manifest_path}"
        )
        return self.events

    # Load phase-2 job segments, falling back to the no_irc variant.
    def load_job_segments(self) -> pd.DataFrame | None:
        p2 = self.cfg.get("phase2", {}).get("output_dir", "offline/data/phase2")
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
        segs["seg_end"] = pd.to_datetime(segs["seg_end"], utc=True)
        return segs

    # Warn when injection specs do not land in their expected phase-2 job context.
    def validate_job_context(self, specs: list[dict]) -> None:
        segs = self.load_job_segments()
        if segs is None:
            print(
                "[inject] job_segments.parquet not found — skipping "
                "job-context validation. Run phase 2 first for stricter checks."
            )
            return

        print(
            "[inject] Validating job context against data/phase2/job_segments.parquet"
        )
        # Derive multi-host jobs once: any job_id with >1 distinct hostname in segs
        hosts_per_job = (
            segs.dropna(subset=["job_id"]).groupby("job_id")["hostname"].nunique()
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
                dur = float(
                    spec.get("duration_min", spec.get("duration_hr", 0) * 60) or 0
                )
                end = start + pd.Timedelta(minutes=dur)
                rack_host_prefix = f"rpc-{rack}-"
                rack_segs = segs[
                    segs["hostname"].astype(str).str.startswith(rack_host_prefix)
                    & (segs["seg_start"] < end)
                    & (segs["seg_end"] > start)
                    & (~segs["is_idle"])
                ]
                if rack_segs.empty:
                    print(
                        f"  [WARN] spec {i+1} ({itype}, {host}): no running "
                        f"rack-{rack} jobs overlap the window — "
                        f"CROSSPLANE_DISAGREEMENT may not trigger cleanly."
                    )
                    n_warn += 1
                else:
                    n_nodes = rack_segs["hostname"].nunique()
                    job_ids = sorted(
                        {int(j) for j in rack_segs["job_id"].dropna().unique()}
                    )
                    print(
                        f"  [ok]   spec {i+1} ({itype}, {host}): rack-{rack} "
                        f"has {n_nodes} busy node(s), jobs={job_ids[:3]}"
                    )
                continue

            start = pd.Timestamp(spec["start"], tz="UTC")
            dur_hr = float(spec.get("duration_hr", 0))
            dur_min = float(spec.get("duration_min", 0))
            end = start + pd.Timedelta(hours=dur_hr, minutes=dur_min)

            host_segs = segs[
                (segs["hostname"] == host)
                & (segs["seg_start"] < end)
                & (segs["seg_end"] > start)
            ].copy()

            if host_segs.empty:
                print(
                    f"  [WARN] spec {i+1} ({itype}, {host}): NO phase-2 segment "
                    f"covers [{start}, {end}] — hostname/window may be wrong."
                )
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
                (host_segs["ov_end"] - host_segs["ov_start"])
                .dt.total_seconds()
                .clip(lower=0)
            )
            host_segs["klass"] = host_segs.apply(
                lambda r: classify_segment(r, multi_host_jobs), axis=1
            )
            by_class = (
                host_segs.groupby("klass")["ov_sec"].sum().sort_values(ascending=False)
            )
            dominant = by_class.index[0]

            if dominant not in expect:
                # Special case: "running" is satisfied by single_host or multi_host
                if ("running" in expect) and dominant in ("single_host", "multi_host"):
                    pass
                else:
                    class_mix = dict(by_class.astype(int))
                    print(
                        f"  [WARN] spec {i+1} ({itype}, {host}): dominant "
                        f"segment class is '{dominant}', expected one of "
                        f"{sorted(expect)}. Overlap mix (sec): {class_mix}"
                    )
                    n_warn += 1
                    continue

            # Emit an INFO line on success so reviewers can confirm
            job_ids = [
                int(j)
                for j in host_segs.loc[host_segs["klass"] == dominant, "job_id"]
                .dropna()
                .unique()
                .tolist()
            ]
            print(
                f"  [ok]   spec {i+1} ({itype}, {host}): landed in "
                f"'{dominant}' segment"
                + (f" (job_ids={job_ids[:3]})" if job_ids else " (idle)")
            )

        if n_warn > 0:
            print(
                f"[inject] {n_warn} spec(s) flagged by job-context validator. "
                "Proceeding — fix the specs if the warnings matter to the paper."
            )
        else:
            print("[inject] all specs passed job-context validation.")

    # Dispatch a spec to its injection method by type.
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

    # Warn if any injected rows fall in the train split and would contaminate training.
    @staticmethod
    def warn_if_train_split(
        df: pd.DataFrame, mask: pd.Series, hostname: str, itype: str
    ) -> None:
        if "split" not in df.columns:
            return
        n_train = int((df.loc[mask, "split"].astype(str) == "train").sum())
        if n_train > 0:
            print(
                f"  [WARN] {hostname}: {n_train} rows in {itype} window "
                f"are split='train' — will contaminate model training. "
                f"Shift the injection window past window.train_end."
            )

    # Remove any stale injected dir and copy the clean master into place.
    def prepare_destination(self) -> None:
        if self.dst_dir.exists():
            print(f"[inject] Removing stale injected dir → {self.dst_dir}")
            shutil.rmtree(self.dst_dir)
        t0 = time.perf_counter()
        print(f"[inject] Copying master → {self.dst_dir} …")
        shutil.copytree(self.src_dir, self.dst_dir, dirs_exist_ok=False)
        print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Find the master parquet for a hostname across all components.
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

    # Load a node's already-copied injected parquet, sorted by timestamp.
    def load_node(self, hostname: str) -> tuple[pd.DataFrame, Path, Path, str]:
        src_path, comp = self.locate_parquet(hostname)
        rel = src_path.relative_to(self.src_dir)
        dst_path = self.dst_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(dst_path, engine="pyarrow")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df, src_path, dst_path, comp

    # Assemble a ground-truth event record describing one injection.
    @staticmethod
    def build_event(
        spec: dict,
        comp: str,
        hostname: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        modified_cols: list[str],
        params: dict,
        target_detectors: list[str],
    ) -> dict:
        return {
            "type": spec["type"],
            "component": comp,
            "hostname": hostname,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "modified_columns": modified_cols,
            "injection_params": params,
            "target_detectors": target_detectors,
            "spec": spec,
        }

    # Write the collected injection events to the manifest JSON.
    def save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)
        print(f"[inject] Manifest → {self.manifest_path}")
