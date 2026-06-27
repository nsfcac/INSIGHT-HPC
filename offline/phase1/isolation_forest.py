from __future__ import annotations

import gc, json, pickle, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sklearn.ensemble import IsolationForest

from shared.utils.io_utils import load_config, apply_node_limit
from shared.utils.maintenance import load_maintenance_windows
from offline.phase1.isolation_forest_module.features_scoring import *


# Train per-component Isolation Forests, score every node, then add cluster-relative scores.
def run_isolation_forest(force: bool = False) -> None:
    cfg = load_config()
    feat_aligned = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))
    feat_dir_base = Path(cfg["paths"]["features"])
    feat_dir = feat_aligned if feat_aligned.exists() else feat_dir_base
    out_base = Path(cfg["phase1"]["output_dir"]) / "isolation_forest"
    out_base.mkdir(parents=True, exist_ok=True)

    if_cfg = cfg["phase1"]["isolation_forest"]
    windows = load_maintenance_windows(cfg)

    only_under_load = bool(if_cfg.get("only_under_load", False))
    min_train_rows = int(if_cfg.get("min_train_rows", 500))

    src_label = "aligned+pdu" if feat_dir == feat_aligned else "features only"
    print(f"\n[iforest] Feature source : {feat_dir}  ({src_label})")
    print(f"[iforest] Model output   : {out_base}")
    print(
        f"[iforest] n_estimators={if_cfg['n_estimators']}  "
        f"n_jobs={int(if_cfg.get('n_jobs', -1))}  "
        f"contamination={if_cfg['contamination']}  "
        f"only_under_load={only_under_load}"
    )

    models: dict = {}
    feature_map: dict = {}

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue

        model_path = out_base / f"{comp}_model.pkl"
        feat_path = out_base / f"{comp}_features.json"

        if model_path.exists() and feat_path.exists() and not force:
            print(f"\n  [{comp.upper()}] Loading existing model from {model_path}")
            with open(model_path, "rb") as f:
                models[comp] = pickle.load(f)
            with open(feat_path) as f:
                feature_map[comp] = json.load(f)
            print(f"  [{comp.upper()}] {len(feature_map[comp])} features loaded")
            continue

        print(f"\n  [{comp.upper()}] Discovering features ...")
        feature_cols = discover_features(feat_dir, comp)
        if not feature_cols:
            print(f"  [WARN] No features found for {comp}, skipping")
            continue
        print(f"  [{comp.upper()}] {len(feature_cols)} features discovered")

        t0 = time.perf_counter()
        print(f"  [{comp.upper()}] Loading training data ...")
        X = load_train_matrix(feat_dir, comp, feature_cols, windows, only_under_load)

        if X is None or len(X) < min_train_rows:
            n = len(X) if X is not None else 0
            print(
                f"  [WARN] {comp}: only {n} training rows (need {min_train_rows}), skipping"
            )
            continue

        print(
            f"  [{comp.upper()}] Training on {len(X):,} rows x {X.shape[1]} features  "
            f"({time.perf_counter()-t0:.1f}s loading)"
        )

        t_train = time.perf_counter()
        clf = IsolationForest(
            n_estimators=int(if_cfg["n_estimators"]),
            contamination=float(if_cfg["contamination"]),
            max_features=float(if_cfg.get("max_features", 1.0)),
            random_state=int(if_cfg.get("random_state", 42)),
            n_jobs=int(if_cfg.get("n_jobs", -1)),
        )
        clf.fit(X)
        print(f"  [{comp.upper()}] Trained in {time.perf_counter()-t_train:.1f}s")

        models[comp] = clf
        feature_map[comp] = feature_cols

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        print(f"  [{comp.upper()}] Model saved: {model_path}")
        del X
        gc.collect()

    if not models:
        print("[iforest] No models trained -- check feature tables exist.")
        return

    print("\n[iforest] Pass 2: scoring all nodes ...")
    scores_dir = out_base / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = load_norm_stats(feat_dir)
    if norm_stats is not None:
        print(
            f"[iforest] norm_stats loaded: {len(norm_stats)} entries — "
            f"top-3 feature attribution enabled"
        )
    else:
        print(
            "[iforest] norm_stats not found — attribution will use anomalous-row stats"
        )

    t0 = time.perf_counter()
    total_rows = anomaly_rows = 0

    workers = phase1_workers()
    IF_WORKER_CONTEXT.clear()
    IF_WORKER_CONTEXT.update(
        models={c: {"clf": models[c], "feature_cols": feature_map[c]} for c in models},
        windows=windows,
        norm_stats=norm_stats,
    )

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp not in models:
            continue

        feature_cols = feature_map[comp]
        comp_dir = feat_dir / comp
        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))

        print(f"\n  [{comp.upper()}]  {len(parquets)} nodes")

        tasks = [
            (comp, p.stem, str(p), str(scores_dir / f"{p.stem}.parquet"), force)
            for p in parquets
        ]
        if workers == 1 or len(tasks) <= 1:
            results = [score_one_node(*t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
                results = list(ex.map(score_one_node, *zip(*tasks)))

        for hostname, log, n_rows, n_anomaly in sorted(results, key=lambda r: r[0]):
            if log:
                print(log, flush=True)
            total_rows += n_rows
            anomaly_rows += n_anomaly

    IF_WORKER_CONTEXT.clear()
    gc.collect()
    elapsed = time.perf_counter() - t0
    print(
        f"\n[iforest] Scoring complete  "
        f"rows={total_rows:,}  anomaly={anomaly_rows:,}  "
        f"({100*anomaly_rows/max(total_rows,1):.2f}%)  "
        f"time={elapsed:.1f}s"
    )
    print(f"  Scores: {scores_dir}")

    print("\n[iforest] Pass 3: computing cluster-relative anomaly scores ...")
    add_cluster_relative_scores(scores_dir, total_rows=total_rows)


if __name__ == "__main__":
    run_isolation_forest(force=True)
