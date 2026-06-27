from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from shared.utils.io_utils import load_config

# Schema columns expected by ground_truth_loader.py
TICKETS_COLUMNS = [
    "start_time",
    "end_time",
    "hostnames",
    "category",
    "severity",
    "description",
    "job_id",
]
FAILURES_COLUMNS = [
    "timestamp",
    "hostname",
    "failure_type",
    "duration_min",
    "ticket_id",
    "job_id",
]


# Build ground-truth ticket events from configured maintenance windows.
def maintenance_events(cfg: dict) -> list[dict]:
    events = []
    for mw in cfg.get("maintenance_windows", []):
        start = mw.get("start") or mw.get("start_time")
        end = mw.get("end") or mw.get("end_time")
        if not start or not end:
            continue
        events.append(
            {
                "start_time": pd.Timestamp(start, tz="UTC").isoformat(),
                "end_time": pd.Timestamp(end, tz="UTC").isoformat(),
                "hostnames": "*",  # cluster-wide
                "category": "maintenance",
                "severity": 2,
                "description": (
                    f"Planned cluster maintenance. "
                    f"Source: shared/configs/config.yaml maintenance_windows entry. "
                    f"Window: {start} → {end} UTC."
                ),
                "job_id": "",
            }
        )
    return events


# Seed ground-truth CSVs and provenance from config-confirmed maintenance events.
def generate(force: bool = False) -> None:
    cfg = load_config()
    gt_dir = Path(
        cfg.get(
            "ground_truth_dir",
            cfg.get("phase4", {}).get("ground_truth_dir", "offline/data/ground_truth"),
        )
    )
    gt_dir.mkdir(parents=True, exist_ok=True)

    tickets_path = gt_dir / "tickets.csv"
    failures_path = gt_dir / "node_failures.csv"

    if tickets_path.exists() and failures_path.exists() and not force:
        existing = (
            pd.read_csv(tickets_path)
            if tickets_path.stat().st_size > 0
            else pd.DataFrame()
        )
        print(
            f"[synth_gt] Ground truth CSVs already exist "
            f"({len(existing)} tickets). Use --force to regenerate from scratch."
        )
        return

    print("\n[synth_gt] Seeding ground truth from config-confirmed events ...")

    #   Anchor events: maintenance windows from config
    maint_events = maintenance_events(cfg)
    if maint_events:
        tickets = pd.DataFrame(maint_events, columns=TICKETS_COLUMNS)
        print(f"  Maintenance windows from config: {len(maint_events)}")
    else:
        # No maintenance windows in config — create empty file with correct schema
        tickets = pd.DataFrame(columns=TICKETS_COLUMNS)
        print(
            "  [WARN] No maintenance_windows found in config.yaml — "
            "tickets.csv seeded empty."
        )

    tickets.to_csv(tickets_path, index=False)
    print(f"  tickets.csv: {len(tickets)} event(s) → {tickets_path}")

    existing_rows = 0
    if failures_path.exists() and failures_path.stat().st_size > 0:
        try:
            existing_rows = len(pd.read_csv(failures_path))
        except Exception:
            existing_rows = 0
    if existing_rows == 0:
        failures = pd.DataFrame(columns=FAILURES_COLUMNS)
        failures.to_csv(failures_path, index=False)
        print(f"  node_failures.csv: 0 events (schema placeholder) → {failures_path}")
    else:
        print(
            f"  node_failures.csv: preserving {existing_rows} curated rows → {failures_path}"
        )

    #   Provenance
    study_window = cfg.get("window", {})
    provenance = {
        "dataset": "2026 PHASE production telemetry",
        "study_start": study_window.get("start", "2026-01-01"),
        "study_end": study_window.get("end", "2026-03-20"),
        "anchor_events": maint_events,
        "source": "config-confirmed maintenance windows only",
        "why_not_more": (
            "Retroactive labelling from IF/audit agreement is circular — IF is "
            "also being evaluated against the same GT.  Confirmed real events "
            "must come from operator knowledge or the GT review workflow."
        ),
        "extend_with_real_events": (
            "Run: python -m offline.phase4.build_gt_review  "
            "→ produces GT_REVIEW.md with the 15-20 highest-confidence "
            "multi-signal episodes for manual confirmation."
        ),
    }
    prov_path = gt_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"  provenance.json → {prov_path}")

    print("\n[synth_gt] Done.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing tickets.csv with fresh seed.",
    )
    args = p.parse_args()
    generate(force=args.force)
