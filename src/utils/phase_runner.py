from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import Callable, Iterable, Optional

# Print a stage banner.
def banner(label: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {label}\n{bar}\n")

# Run a stage function, catching and reporting errors.
def try_run(label: str, fn: Callable[[bool], None], force: bool) -> bool:
    try:
        fn(force)
        return True
    except Exception as e:
        print(f"\n[run] ERROR in {label}: {e}")
        traceback.print_exc()
        return False

# Select which stages to run from only/from args.
def select_stages(order: list, only: Optional[str] = None, from_stage: Optional[str] = None) -> list:
    if only:
        return [only]
    if from_stage:
        try:
            return order[order.index(from_stage):]
        except ValueError:
            print(f"Unknown stage '{from_stage}'. Valid: {order}")
            sys.exit(1)
    return list(order)

# Add the shared CLI args (force/from/only).
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Recompute outputs")
    parser.add_argument("--from", dest="from_stage", help="Start from this stage")
    parser.add_argument("--only", dest="only_stage", help="Run only this stage")

# Execute the selected stages with timing and status.
def run_stages(title: str, order: list, labels: dict, runners: dict, *, stages: Iterable[str], force: bool, capture_errors: bool = False) -> dict:
    t0 = time.perf_counter()
    results: dict = {}
    for key in stages:
        if key not in runners:
            continue
        banner(labels.get(key, key))
        ts = time.perf_counter()
        if capture_errors:
            ok = try_run(labels.get(key, key), runners[key], force)
        else:
            runners[key](force)
            ok = True
        results[key] = ("ok" if ok else "FAILED",
                        round(time.perf_counter() - ts, 1))
    print(f"\n  {title} complete  total={time.perf_counter() - t0:.1f}s\n")
    return results
