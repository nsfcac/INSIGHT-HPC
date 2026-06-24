from __future__ import annotations

import argparse

from src.utils.phase_runner import add_common_args, run_stages, select_stages

STAGE_ORDER = ["threshold", "iforest", "lstm", "baseline", "eval"]
STAGE_LABELS = {
    "threshold": "Step 1 — baseline_threshold",
    "iforest": "Step 2 — isolation_forest",
    "lstm": "Step 3 — LSTM Autoencoder",
    "baseline": "Step 4 — Baseline Comparison",
    "eval": "Step 5 — evaluate",
}


def threshold(force: bool) -> None:
    from src.phase1.baseline_threshold import run_baseline_threshold

    run_baseline_threshold(force=force)


def iforest(force: bool) -> None:
    from src.phase1.isolation_forest import run_isolation_forest

    run_isolation_forest(force=force)


def lstm(force: bool) -> None:
    from src.phase1.lstm_autoencoder import run_lstm_autoencoder

    run_lstm_autoencoder(force=force)


def baseline(force: bool) -> None:
    from src.phase1.baseline_comparison import run_baseline_comparison

    run_baseline_comparison(force=force)


def evaluate(force: bool) -> None:
    from src.phase1.evaluate import run_evaluation

    run_evaluation(force=force)


RUNNERS = {
    "threshold": threshold,
    "iforest": iforest,
    "lstm": lstm,
    "baseline": baseline,
    "eval": evaluate,
}


# Parse CLI args and run this module.
def main() -> None:
    parser = argparse.ArgumentParser(description="INSIGHT-HPC Phase I")
    add_common_args(parser)
    args = parser.parse_args()
    stages = select_stages(STAGE_ORDER, args.only_stage, args.from_stage)
    run_stages(
        "Phase I", STAGE_ORDER, STAGE_LABELS, RUNNERS, stages=stages, force=args.force
    )


if __name__ == "__main__":
    main()
