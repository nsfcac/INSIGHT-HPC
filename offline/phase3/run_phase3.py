from __future__ import annotations

import argparse, sys

from shared.utils.io_utils import load_config
from offline.utils.phase_runner import add_common_args, run_stages

STAGE_ORDER = ["physics", "constraints", "jctx"]
STAGE_LABELS = {
    "physics": "Step 1 — physics_models (power + thermal residual)",
    "constraints": "Step 2 — physics_constraints (4 engineered rules)",
    "jctx": "Step 3 — job_context_annotator (job vs fault)",
}


def run_physics_stage(force: bool) -> None:
    from offline.phase3.physics_models import run_physics_models

    run_physics_models(force=force)


def run_constraints_stage(force: bool) -> None:
    from offline.phase3.physics_constraints import run_physics_constraints

    run_physics_constraints(force=force)


def run_job_context_stage(force: bool) -> None:
    from offline.phase3.job_context_annotator import run_job_context_annotator

    run_job_context_annotator(force=force)


RUNNERS = {
    "physics": run_physics_stage,
    "constraints": run_constraints_stage,
    "jctx": run_job_context_stage,
}


# Parse CLI args and run this module.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PHASE Phase III — Physics-informed models"
    )
    add_common_args(parser)
    parser.add_argument(
        "--step",
        nargs="+",
        type=str,
        default=None,
        help="Run only these stages (e.g. --step physics constraints)",
    )
    args = parser.parse_args()

    # phase3.enabled=false → no-op (e.g. Prodigy dataset without physics pipeline).
    try:
        cfg = load_config()
        if not cfg.get("phase3", {}).get("enabled", True):
            print("\n[run_phase3] phase3.enabled=false — skipping all steps")
            sys.exit(0)
    except Exception:
        pass

    only_set = set(args.step) if args.step else None
    stages = [s for s in STAGE_ORDER if not only_set or s in only_set]
    if args.only_stage:
        stages = [args.only_stage]

    results = run_stages(
        "Phase III",
        STAGE_ORDER,
        STAGE_LABELS,
        RUNNERS,
        stages=stages,
        force=args.force,
        capture_errors=True,
    )
    failures = [n for n, (s, _) in results.items() if s != "ok"]
    if failures:
        print(f"\n  FAILED steps: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
