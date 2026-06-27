from __future__ import annotations

import shutil
from pathlib import Path

from shared.utils.io_utils import load_config


# Map each phase to its trained-artifact source dir and the glob patterns to snapshot.
def artifact_specs(cfg: dict) -> list[tuple[str, Path, list[str]]]:
    paths = cfg["paths"]
    return [
        ("phase1", Path(cfg["phase1"]["output_dir"]), [
            "isolation_forest/*_model.pkl",
            "isolation_forest/*_features.json",
            "lstm_ae/*_model.pt",
            "lstm_ae/*_stats.json",
        ]),
        ("phase2", Path(cfg["phase2"]["output_dir"]), [
            "cluster_models/*.pkl",
            "cluster_models/*_threshold.json",
            "clusters/*",
        ]),
        ("phase3", Path(cfg["phase3"]["output_dir"]), [
            "physics/power_model_*.pkl",
            "physics/thermal_model_*.pkl",
        ]),
        ("phase4", Path(cfg["phase4"]["output_dir"]), [
            "fusion_gbdt.pkl",
        ]),
        ("features", Path(paths["features"]), [
            "norm_stats.parquet",
            "norm_stats_idle.parquet",
        ]),
    ]


# Copy one glob's matches into dst_base as flat files named by their basename.
def copy_pattern(src_dir: Path, pattern: str, dst_base: Path) -> int:
    n = 0
    for src in sorted(src_dir.glob(pattern)):
        if not src.is_file():
            continue
        dst = dst_base / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
        print(f"  [snapshot] {src} -> {dst}")
    return n


# Copy every trained artifact from each phase output dir into offline/models/<phase>/.
def snapshot_models() -> int:
    cfg = load_config()
    models_root = Path(cfg["paths"].get("models", "offline/models"))
    total = 0
    for name, src_dir, patterns in artifact_specs(cfg):
        dst_base = models_root / name
        if not src_dir.exists():
            print(f"[snapshot] WARNING: source dir missing for {name}: {src_dir}")
            continue
        for pattern in patterns:
            matched = copy_pattern(src_dir, pattern, dst_base)
            if matched == 0:
                print(f"[snapshot] WARNING: no files matched {name} pattern '{pattern}' in {src_dir}")
            total += matched
    print(f"[snapshot] copied {total} artifact(s) into {models_root}")
    return total


if __name__ == "__main__":
    snapshot_models()
