from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, save_parquet


# Parse a timestamp series to UTC microsecond dtype.
def parse_timestamp_series(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    except Exception:
        parsed = pd.to_datetime(series, errors="coerce").dt.tz_localize("UTC")
    return parsed.astype("datetime64[us, UTC]")


# Parse local timestamps (tz-aware or naive) into UTC using the facility timezone.
def parse_local_timestamp_series_to_utc(
    series: pd.Series, facility_tz: str
) -> pd.Series:
    s = series.astype(str).str.strip()
    # Rows with explicit tz offset (contain + or - in the time portion after T)
    has_tz = s.str.contains(
        r"T\d{2}:\d{2}:\d{2}[+\-]\d{2}:?\d{2}$", regex=True, na=False
    )

    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[us, UTC]")

    if has_tz.any():
        with_tz = pd.to_datetime(s[has_tz], utc=True, errors="coerce")
        out.loc[has_tz] = with_tz.astype("datetime64[us, UTC]")

    naive_mask = ~has_tz
    if naive_mask.any():
        naive_parsed = pd.to_datetime(s[naive_mask], errors="coerce")
        localised = naive_parsed.dt.tz_localize(
            facility_tz, nonexistent="shift_forward", ambiguous="NaT"
        )
        out.loc[naive_mask] = localised.dt.tz_convert("UTC").astype(
            "datetime64[us, UTC]"
        )

    return out


# Return the configured facility timezone name.
def facility_timezone(cfg: Optional[dict] = None) -> str:
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            cfg = {}
    return (
        cfg.get("facility_timezone")
        or cfg.get("time", {}).get("facility_tz")
        or "America/Chicago"
    )


# Load tickets.csv into per-host ground-truth event rows.
def load_tickets(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        print(f"  [WARN] tickets.csv is empty — skipping")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"start_time", "end_time", "hostnames"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] tickets.csv missing columns: {missing}")
        return None

    records = []
    for idx, row in df.iterrows():
        # Explode pipe-delimited hostnames to one row per node
        hosts = [h.strip() for h in str(row["hostnames"]).split("|") if h.strip()]
        if not hosts:
            hosts = ["unknown"]
        for host in hosts:
            records.append(
                {
                    "event_id": f"ticket-{idx}",
                    "tier": "ticket",
                    "category": normalise_category(row.get("category", "unknown")),
                    "severity": parse_severity(row.get("severity", 2)),
                    "hostname": host,
                    "job_id": safe_int(row.get("job_id")),
                    "event_start": row["start_time"],
                    "event_end": row["end_time"],
                    "description": str(row.get("description", "")),
                }
            )

    if not records:
        return None
    out = pd.DataFrame(records)
    out["event_start"] = parse_timestamp_series(out["event_start"])
    out["event_end"] = parse_timestamp_series(out["event_end"])
    out["duration_min"] = (
        ((out["event_end"] - out["event_start"]).dt.total_seconds() / 60)
        .clip(lower=0)
        .astype("float32")
    )
    return out


# Load node_failures.csv into ground-truth event rows.
def load_node_failures(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    has_data = False
    if path.stat().st_size > 0:
        try:
            has_data = len(pd.read_csv(path, nrows=1)) > 0
        except Exception:
            has_data = False
    if not has_data:
        print(f"  [WARN] node_failures.csv is empty — skipping")
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"timestamp", "hostname", "failure_type"}
    if required - set(df.columns):
        print(f"  [WARN] node_failures.csv missing: {required - set(df.columns)}")
        return None

    out_rows = []
    for idx, row in df.iterrows():
        dur = (
            float(row["duration_min"])
            if "duration_min" in df.columns and not pd.isna(row.get("duration_min"))
            else 60.0
        )
        start = row["timestamp"]
        ftype = str(row.get("failure_type", "")).strip()
        reason = str(row.get("reason", "") or "").strip()
        category_src = ftype if ftype else reason
        description = f"{ftype} — {reason}" if reason and reason != ftype else ftype
        out_rows.append(
            {
                "event_id": f"failure-{idx}",
                "tier": "node_failure",
                "category": normalise_category(category_src or "hardware"),
                "severity": 3,  # node failures are always critical
                "hostname": str(row["hostname"]).strip(),
                "job_id": None,  # node failures are host-scoped, not job-scoped
                "event_start": start,
                "event_end": start,  # will be offset below
                "duration_min": dur,
                "description": description,
            }
        )

    if not out_rows:
        return None
    out = pd.DataFrame(out_rows)
    facility_tz = facility_timezone()
    out["event_start"] = parse_local_timestamp_series_to_utc(
        out["event_start"], facility_tz
    )
    out["event_end"] = out["event_start"] + pd.to_timedelta(
        out["duration_min"], unit="m"
    )
    out["event_end"] = out["event_end"].astype("datetime64[us, UTC]")
    return out


# Load slurm_failures.csv (non-zero exits / node fails) into event rows.
def load_slurm_failures(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        print(f"  [WARN] slurm_failures.csv is empty — skipping")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"job_id", "node", "start_time", "end_time", "exit_code"}
    if required - set(df.columns):
        print(f"  [WARN] slurm_failures.csv missing: {required - set(df.columns)}")
        return None

    df["exit_code"] = (
        pd.to_numeric(df["exit_code"], errors="coerce").fillna(0).astype(int)
    )
    reason_lower = (
        df.get("failure_reason", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    is_node_fail = reason_lower.str.contains("node_fail")
    df = df[(df["exit_code"] != 0) | is_node_fail]
    hardware_exits = {
        1,
        2,
        126,
        127,
    }  # 9=OOM, 143=SIGTERM, 130=SIGINT excluded from severity=2

    out_rows = []
    for idx, row in df.iterrows():
        ec = int(row["exit_code"])
        reason = str(row.get("failure_reason", "") or "")
        reason_lc = reason.lower()
        if "node_fail" in reason_lc:
            sev = 3  # Slurm-attributed node failure — critical
            cat = normalise_category("node_fail")
        elif ec in hardware_exits:
            sev = 2
            cat = "software"
        else:
            sev = 1
            cat = "software"
        out_rows.append(
            {
                "event_id": f"slurm-{idx}",
                "tier": "slurm_failure",
                "category": cat,
                "severity": sev,
                "hostname": str(row["node"]).strip(),
                "job_id": safe_int(row.get("job_id")),
                "event_start": row["start_time"],
                "event_end": row["end_time"],
                "duration_min": 0.0,  # filled after parse
                "description": f"exit_code={ec} reason={reason}",
            }
        )

    if not out_rows:
        return None
    out = pd.DataFrame(out_rows)
    out["event_start"] = parse_timestamp_series(out["event_start"])
    out["event_end"] = parse_timestamp_series(out["event_end"])
    out["duration_min"] = (
        ((out["event_end"] - out["event_start"]).dt.total_seconds() / 60)
        .clip(lower=0)
        .astype("float32")
    )
    return out


#   Helpers

CATEGORY_MAP = {
    "thermal": ["thermal", "temp", "heat", "fan", "cooling", "overheat", "irc"],
    "power": ["power", "pdu", "psu", "voltage", "current", "watt"],
    "hardware": [
        "hardware",
        "disk",
        "memory",
        "dimm",
        "cpu",
        "gpu",
        "nic",
        "dram",
        "nvme",
        "ssd",
        "failure",
        "fault",
        "rma",
        "drain",
        "node_fail",
        "not responding",
        "invalid_config",
        "system_board",
        "realmemory",
    ],
    "software": [
        "software",
        "slurm",
        "driver",
        "kernel",
        "crash",
        "oom",
        "timeout",
        "exit",
        "prolog",
        "kill_task",
    ],
    "network": ["network", "ib", "infiniband", "ethernet", "link"],
}


# Map a raw category/keyword string to a canonical category.
def normalise_category(raw: str) -> str:
    raw = str(raw).lower().strip()
    for cat, keywords in CATEGORY_MAP.items():
        if any(k in raw for k in keywords):
            return cat
    return "unknown"


# Parse a severity value (numeric or label) to 1-3.
def parse_severity(raw) -> int:
    try:
        v = int(raw)
        return max(1, min(3, v))
    except Exception:
        raw = str(raw).lower()
        if raw in ("critical", "high", "p1", "sev1"):
            return 3
        if raw in ("warning", "med", "p2", "sev2"):
            return 2
        return 1


def safe_int(x) -> Optional[int]:
    try:
        v = int(float(x))
        return v if v > 0 else None
    except Exception:
        return None


# Write blank GT CSV templates (tickets, node failures, slurm failures).
def write_templates(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "hostnames",
            "category",
            "severity",
            "description",
            "job_id",
        ]
    ).to_csv(out_dir / "tickets.csv", index=False)

    pd.DataFrame(
        columns=[
            "timestamp",
            "hostname",
            "failure_type",
            "duration_min",
            "ticket_id",
            "job_id",
        ]
    ).to_csv(out_dir / "node_failures.csv", index=False)

    pd.DataFrame(
        columns=[
            "job_id",
            "node",
            "start_time",
            "end_time",
            "exit_code",
            "failure_reason",
        ]
    ).to_csv(out_dir / "slurm_failures.csv", index=False)

    print(f"[gt] Templates written to {out_dir}")
    print("  Fill in and re-run: python -m src.phase3.ground_truth_loader")


# Load injected_events.json into ground-truth event rows.
def load_injected_events(gt_dir: Path) -> Optional[pd.DataFrame]:
    path = gt_dir / "injected_events.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            events = json.load(f)
    except Exception as exc:
        print(f"  [WARN] could not load {path}: {exc}")
        return None
    if not events:
        return None
    records = []
    for i, ev in enumerate(events):
        records.append(
            {
                "event_id": f"inject-{i}-{ev.get('hostname', '?')}",
                "tier": "injected",
                "category": ev.get("type", "unknown"),
                "severity": 3,
                "hostname": ev.get("hostname", "unknown"),
                "job_id": None,
                "event_start": ev.get("start_time"),
                "event_end": ev.get("end_time"),
                "description": (
                    f"{ev.get('type','')} " f"params={ev.get('injection_params', {})}"
                ),
            }
        )
    out = pd.DataFrame(records)
    out["event_start"] = parse_timestamp_series(out["event_start"])
    out["event_end"] = parse_timestamp_series(out["event_end"])
    out["duration_min"] = (
        ((out["event_end"] - out["event_start"]).dt.total_seconds() / 60)
        .clip(lower=0)
        .astype("float32")
    )
    return out


# Load and merge all GT tiers into events.parquet within the study window.
def run_load_ground_truth(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    gt_dir = Path(
        cfg.get(
            "ground_truth_dir",
            cfg.get("phase4", {}).get("ground_truth_dir", "data/ground_truth"),
        )
    )
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_path = gt_dir / "events.parquet"

    if out_path.exists() and not force:
        print("[gt] events.parquet exists — loading (use --force to reload)")
        return pd.read_parquet(out_path, engine="pyarrow")

    t0 = time.perf_counter()
    print("\n[gt] Loading ground truth from all available tiers ...")

    frames = []
    for loader, fname, label in [
        (load_tickets, "tickets.csv", "Tier A: tickets"),
        (load_node_failures, "node_failures.csv", "Tier B: node failures"),
        (load_slurm_failures, "slurm_failures.csv", "Tier C: slurm failures"),
    ]:
        path = gt_dir / fname
        df = loader(path)
        if df is not None and not df.empty:
            print(f"  {label}: {len(df)} events")
            frames.append(df)
        else:
            print(f"  {label}: not found or empty — skipping")

    inj_df = load_injected_events(gt_dir)
    if inj_df is not None and not inj_df.empty:
        print(f"  Tier I (injected): {len(inj_df)} events")
        frames.append(inj_df)
    else:
        print("  Tier I (injected): injected_events.json not found — skipping")

    if not frames:
        print("\n[gt] No ground truth available.")
        print("  Run with --templates to create blank CSV templates.")
        print("  Fill them in with data from your facility operators and re-run.")
        # Return empty with correct schema so downstream code doesn't crash
        return pd.DataFrame(
            columns=[
                "event_id",
                "tier",
                "category",
                "severity",
                "hostname",
                "job_id",
                "event_start",
                "event_end",
                "duration_min",
                "description",
            ]
        )

    # Restrict to study window
    w = cfg["window"]
    win_start = pd.Timestamp(w["start"], tz="UTC")
    win_end = pd.Timestamp(w["end"], tz="UTC")

    result = pd.concat(frames, ignore_index=True)
    result = result[
        (result["event_start"] >= win_start) & (result["event_start"] <= win_end)
    ].copy()

    result["job_id"] = pd.array(result["job_id"], dtype="Int64")
    result["severity"] = result["severity"].astype("int8")
    result["duration_min"] = result["duration_min"].astype("float32")

    save_parquet(result, out_path)

    # Summary
    summary = {
        "total_events": len(result),
        "by_tier": result["tier"].value_counts().to_dict(),
        "by_category": result["category"].value_counts().to_dict(),
        "by_severity": result["severity"].value_counts().to_dict(),
        "unique_nodes": int(result["hostname"].nunique()),
        "date_range": [
            str(result["event_start"].min()),
            str(result["event_start"].max()),
        ],
    }
    (gt_dir / "events_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(
        f"\n[gt] Loaded {len(result)} ground-truth events in "
        f"{time.perf_counter()-t0:.1f}s"
    )
    print(f"  Nodes affected: {summary['unique_nodes']}")
    print(f"  Categories: {summary['by_category']}")
    print(f"  Saved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.templates:
        config = load_config()
        write_templates(
            Path(
                config.get(
                    "ground_truth_dir",
                    config.get("phase4", {}).get(
                        "ground_truth_dir", "data/ground_truth"
                    ),
                )
            )
        )
    else:
        run_load_ground_truth(force=args.force)
