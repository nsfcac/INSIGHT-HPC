from __future__ import annotations

import argparse, json, os
from pathlib import Path

from src.utils.phase_runner import add_common_args, run_stages, select_stages

STAGE_ORDER = ["convert", "audit", "master", "inject", "feat"]
STAGE_LABELS = {
    "convert": "Stage 1 — convert_to_parquet",
    "audit": "Stage 2 — audit_raw_files",
    "master": "Stage 3 — build_master_tables",
    "inject": "Stage 4 — synthetic_injector",
    "feat": "Stage 5 — feature_engineering",
}


def convert(force: bool) -> None:
    from src.preprocess.convert_to_parquet import convert_raw_to_parquet

    convert_raw_to_parquet(force=force)


def audit(force: bool) -> None:
    from src.preprocess.audit_raw_files import audit_raw_files

    audit_raw_files(force=force)


def master(force: bool) -> None:
    from src.preprocess.build_master_tables import build_master_tables

    build_master_tables(force=force)


# Inject synthetic anomalies when a spec file is configured, then redirect to the injected run.
def inject(force: bool) -> None:
    spec_env = os.environ.get("INSIGHT_HPC_INJECT_SPEC", "")
    if not spec_env:
        print("[inject] INSIGHT_HPC_INJECT_SPEC not set — skipping.")
        return
    spec_path = Path(spec_env)
    if not spec_path.exists():
        print(f"[inject] spec file missing: {spec_path} — skipping.")
        return
    from src.utils.synthetic_injector import SyntheticInjector

    specs = json.loads(spec_path.read_text())
    SyntheticInjector().run(specs)
    os.environ["INSIGHT_HPC_RUN_SUFFIX"] = "_injected"


def feat(force: bool) -> None:
    from src.preprocess.feature_engineering import feature_engineering

    feature_engineering(force=force)


RUNNERS = {
    "convert": convert,
    "audit": audit,
    "master": master,
    "inject": inject,
    "feat": feat,
}


# Parse CLI args and run this module.
def main() -> None:
    parser = argparse.ArgumentParser(description="INSIGHT-HPC preprocessing pipeline")
    add_common_args(parser)
    args = parser.parse_args()
    stages = select_stages(STAGE_ORDER, args.only_stage, args.from_stage)
    run_stages(
        "Preprocessing",
        STAGE_ORDER,
        STAGE_LABELS,
        RUNNERS,
        stages=stages,
        force=args.force,
    )


if __name__ == "__main__":
    main()
