from __future__ import annotations

import argparse

from offline.utils.phase_runner import add_common_args, run_stages, select_stages

STAGE_ORDER = [
    "segment",
    "profiles",
    "cluster",
    "train",
    "attribute",
    "multi",
    "coh_pm",
    "streaming",
]
STAGE_LABELS = {
    "segment": "Step 1 — segment_jobs",
    "profiles": "Step 2 — profiling_and_clustering.run_build_profiles",
    "cluster": "Step 3 — profiling_and_clustering.run_cluster_jobs (K-means + HDBSCAN)",
    "train": "Step 4 — profiling_and_clustering.run_train_cluster_models + reproducibility",
    "attribute": "Step 5 — attribute_alerts",
    "multi": "Step 6 — coherence.run_multi_node_coherence",
    "coh_pm": "Step 7 — coherence.run_per_minute_coherence",
    "streaming": "Step 8 — coherence.run_streaming_coherence",
}


def segment(force: bool) -> None:
    from offline.phase2.segment_jobs import run_segment_jobs

    run_segment_jobs(force=force)


def profiles(force: bool) -> None:
    from offline.phase2.profiling_and_clustering import run_build_profiles

    run_build_profiles(force=force)


def cluster(force: bool) -> None:
    from offline.phase2.profiling_and_clustering import run_cluster_jobs

    run_cluster_jobs(force=force)


def train(force: bool) -> None:
    from offline.phase2.profiling_and_clustering import run_train_cluster_models

    run_train_cluster_models(force=force)


def attribute(force: bool) -> None:
    from offline.phase2.attribute_alerts import run_attribute_alerts

    run_attribute_alerts(force=force)


def multi(force: bool) -> None:
    from offline.phase2.coherence import run_multi_node_coherence

    run_multi_node_coherence(force=force)


def coh_pm(force: bool) -> None:
    from offline.phase2.coherence import run_per_minute_coherence

    run_per_minute_coherence(force=force)


def streaming(force: bool) -> None:
    from offline.phase2.coherence import run_streaming_coherence

    run_streaming_coherence(force=force)


RUNNERS = {
    "segment": segment,
    "profiles": profiles,
    "cluster": cluster,
    "train": train,
    "attribute": attribute,
    "multi": multi,
    "coh_pm": coh_pm,
    "streaming": streaming,
}


# Parse CLI args and run this module.
def main() -> None:
    parser = argparse.ArgumentParser(description="INSIGHT-HPC Phase II")
    add_common_args(parser)
    args = parser.parse_args()
    stages = select_stages(STAGE_ORDER, args.only_stage, args.from_stage)
    run_stages(
        "Phase II", STAGE_ORDER, STAGE_LABELS, RUNNERS, stages=stages, force=args.force
    )


if __name__ == "__main__":
    main()
