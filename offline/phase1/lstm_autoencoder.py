from __future__ import annotations

import gc, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

try:

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from shared.utils.io_utils import load_config, save_parquet, apply_node_limit
from shared.utils.maintenance import load_maintenance_windows, apply_maintenance_mask
from offline.phase1.lstm_autoencoder_module.windows_model import *


# Train per-tier LSTM autoencoders on clean windows and score every node's reconstruction error.
def run_lstm_autoencoder(force: bool = False) -> None:
    only_comp = os.environ.get("INSIGHT_HPC_LSTM_ONLY_COMP")
    if os.environ.get("INSIGHT_HPC_LSTM_CONCURRENT", "0") == "1" and not only_comp:
        return run_lstm_concurrent(force)

    set_deterministic_seeds(42)

    cfg = load_config()
    maint_windows = load_maintenance_windows(cfg)
    feat_aligned = Path(cfg["paths"].get("features_aligned", "offline/data/features_aligned"))
    feat_dir = feat_aligned if feat_aligned.exists() else Path(cfg["paths"]["features"])
    models_dir = Path(cfg["phase1"]["output_dir"])

    model_dir = models_dir / "lstm_ae"
    scores_dir = model_dir / "scores"
    model_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    lstm_cfg = cfg.get("phase1", {}).get("lstm_ae", {})
    ITER_ROUNDS = int(lstm_cfg.get("iterative_rounds", ITER_ROUNDS_DEFAULT))
    ITER_CLEAN_PERCENTILE = float(
        lstm_cfg.get("iter_clean_percentile", ITER_CLEAN_PERCENTILE_DEFAULT)
    )
    epochs = int(lstm_cfg.get("epochs", EPOCHS))
    max_train_windows = int(lstm_cfg.get("max_train_windows", 50000))

    training_log: dict = {}
    device = require_cuda()
    print(
        f"\n[lstm_ae] device={device}  window={WINDOW}min  hidden={HIDDEN}  "
        f"layers={LAYERS}  epochs={epochs}  iter_rounds={ITER_ROUNDS}"
    )
    print(f"  Feature source: {feat_dir}")
    print(f"  Model output  : {model_dir}")

    t0 = time.perf_counter()

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue
        if only_comp and comp != only_comp:
            continue
        comp_dir = feat_dir / comp
        if not comp_dir.exists():
            continue

        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))
        print(f"\n  [{comp.upper()}]  {len(parquets)} nodes")

        feature_cols: list = []
        for p in parquets[:5]:
            try:
                feature_cols = select_lstm_features(list(pq.read_schema(p).names))
            except Exception:
                feature_cols = []
            if feature_cols:
                break

        if not feature_cols:
            print(f"    [WARN] No valid feature columns for {comp}")
            continue
        print(f"    Feature cols: {len(feature_cols)} — {feature_cols[:5]} ...")
        n_features = len(feature_cols)

        node_meta: dict = {}
        tier_windows: dict = {}
        n_train_nodes = 0

        for p in parquets:
            hostname = p.stem
            try:
                node_feat_cols = select_lstm_features(list(pq.read_schema(p).names))
            except Exception:
                node_feat_cols = []
            if not node_feat_cols:
                continue
            df = read_lstm_frame(p, node_feat_cols)
            if df is None or df.empty or "split" not in df.columns:
                if df is not None:
                    del df
                continue

            df = df.sort_values(TS).reset_index(drop=True)
            apply_maintenance_mask(df, maint_windows, TS, "hostname")

            pwr_col = next(
                (
                    c
                    for c in node_feat_cols
                    if "systeminputpower" in c.lower() or "totalpower" in c.lower()
                ),
                None,
            )
            med_pwr = (
                float(df[pwr_col].median())
                if pwr_col and pwr_col in df.columns
                else 0.0
            )

            if "req_power_class" in df.columns:
                tier = int(
                    pd.to_numeric(df["req_power_class"], errors="coerce")
                    .fillna(0)
                    .median()
                )
                tier = max(0, min(4, tier))
            else:
                tier = -1

            node_meta[hostname] = {
                "feat_cols": node_feat_cols,
                "tier": tier,
                "med_pwr": med_pwr,
                "df": df,
            }

        missing_tier = [h for h, m in node_meta.items() if m["tier"] == -1]
        if missing_tier:
            powers = sorted(missing_tier, key=lambda h: node_meta[h]["med_pwr"])
            n = len(powers)
            for rank, hostname in enumerate(powers):
                node_meta[hostname]["tier"] = min(int(rank / n * 5), 4)

        tier_groups = group_nodes_by_tier_and_schema(node_meta)
        schema_counts = {}
        for tk, hns in tier_groups.items():
            schema_key = tk.split("_", 1)[1] if "_" in tk else "unknown"
            schema_counts[schema_key] = schema_counts.get(schema_key, 0) + len(hns)
        if len(schema_counts) > 1:
            print(f"    {len(schema_counts)} distinct feature schemas detected:")
            for sk, cnt in schema_counts.items():
                print(f"           schema={sk}: {cnt} nodes")

        for tier_key, hostnames in tier_groups.items():
            for hostname in hostnames:
                meta = node_meta[hostname]
                df = meta["df"]
                node_feat = meta["feat_cols"]

                train = df[df["split"] == "train"].copy()
                if "audit_any" in train.columns:
                    train = train[~train["audit_any"].eq(True)]
                if "maintenance_flag" in train.columns:
                    train = train[~train["maintenance_flag"].eq(True)]
                try:
                    from offline.utils.gt_mask import in_gt_window

                    gt = in_gt_window(train)
                    if gt.any():
                        train = train[~gt]
                except Exception:
                    pass
                train = train.reset_index(drop=True)

                if len(train) < WINDOW + 10:
                    continue

                if_scores = load_if_scores_for_node(hostname, cfg)
                mat, excluded = build_clean_training_matrix(train, node_feat, if_scores)

                n_excl = int(excluded.sum())
                if n_excl > 0:
                    pct = 100 * n_excl / max(len(train), 1)
                    if pct > 1.0:
                        print(
                            f"      {hostname}: excluded {n_excl} sustained-anomaly "
                            f"rows from training windows ({pct:.1f}%)"
                        )

                windows = extract_windows_safe(mat, excluded, stride=STRIDE_TRAIN)
                if len(windows) == 0:
                    continue

                n_train_nodes += 1
                tier_windows.setdefault(tier_key, []).extend(list(windows))

                del train, mat, windows
                gc.collect()

        all_tier_wins = [w for ws in tier_windows.values() for w in ws]

        tier_models: dict = {}

        for tier_key, win_list in tier_windows.items():
            if len(win_list) < MIN_WINDOWS:
                print(
                    f"    Tier {tier_key}: only {len(win_list)} windows "
                    f"(need {MIN_WINDOWS}) — will use global fallback"
                )
                continue

            hostnames_in_tier = tier_groups.get(tier_key, [])
            if not hostnames_in_tier:
                continue
            tier_feat = node_meta[hostnames_in_tier[0]]["feat_cols"]
            nf = len(tier_feat)

            t_arr = np.stack(win_list[:max_train_windows], axis=0)
            print(f"    Tier {tier_key}: {len(t_arr):,} windows — round 1 training ...")
            model, mae_mean, mae_std = train_model(
                t_arr, nf, epochs=epochs, device=device
            )

            if model is not None and ITER_ROUNDS >= 2:
                keep_mask = filter_windows_by_recon_error(
                    t_arr, model, device, percentile=ITER_CLEAN_PERCENTILE
                )
                n_kept = int(keep_mask.sum())
                n_dropped = len(t_arr) - n_kept
                if n_dropped > 0 and n_kept >= MIN_WINDOWS:
                    print(
                        f"      round 2: dropped {n_dropped} high-error windows "
                        f"({100*n_dropped/len(t_arr):.1f}%) — retraining on "
                        f"{n_kept:,} clean windows ..."
                    )
                    del model
                    gc.collect()
                    t_arr_clean = t_arr[keep_mask]
                    model, mae_mean, mae_std = train_model(
                        t_arr_clean, nf, epochs=epochs, device=device
                    )
                    del t_arr_clean
                    gc.collect()
                else:
                    print(
                        f"      round 2: {n_dropped} windows dropped "
                        f"— below MIN_WINDOWS floor, keeping round-1 model"
                    )

            if model is not None:
                tier_models[tier_key] = (model, mae_mean, mae_std, tier_feat)
                torch.save(
                    model.state_dict(), model_dir / f"{comp}_{tier_key}_model.pt"
                )
                (model_dir / f"{comp}_{tier_key}_stats.json").write_text(
                    json.dumps(
                        {
                            "feature_cols": tier_feat,
                            "tier_key": tier_key,
                            "n_features": nf,
                            "train_mae_mean": mae_mean,
                            "train_mae_std": mae_std,
                            "n_windows": len(t_arr),
                        }
                    )
                )
            del t_arr
            gc.collect()

        global_model = None
        global_mae_mean = 0.0
        global_mae_std = 1.0

        if len(all_tier_wins) >= MIN_WINDOWS:
            all_arr = np.stack(all_tier_wins[:max_train_windows], axis=0)
            print(
                f"    Global fallback model: {len(all_arr):,} windows — round 1 training ..."
            )
            global_model, global_mae_mean, global_mae_std = train_model(
                all_arr, n_features, epochs=epochs, device=device
            )

            if global_model is not None and ITER_ROUNDS >= 2:
                keep_mask = filter_windows_by_recon_error(
                    all_arr, global_model, device, percentile=ITER_CLEAN_PERCENTILE
                )
                n_kept = int(keep_mask.sum())
                n_dropped = len(all_arr) - n_kept
                if n_dropped > 0 and n_kept >= MIN_WINDOWS:
                    print(
                        f"      global round 2: dropped {n_dropped} high-error windows "
                        f"({100*n_dropped/len(all_arr):.1f}%) — retraining ..."
                    )
                    del global_model
                    gc.collect()
                    all_arr_clean = all_arr[keep_mask]
                    global_model, global_mae_mean, global_mae_std = train_model(
                        all_arr_clean, n_features, epochs=epochs, device=device
                    )
                    del all_arr_clean
                    gc.collect()

            if global_model is not None:
                torch.save(
                    global_model.state_dict(), model_dir / f"{comp}_global_model.pt"
                )
                (model_dir / f"{comp}_global_stats.json").write_text(
                    json.dumps(
                        {
                            "feature_cols": feature_cols,
                            "n_features": n_features,
                            "train_mae_mean": global_mae_mean,
                            "train_mae_std": global_mae_std,
                            "n_windows": len(all_arr),
                        }
                    )
                )
            del all_arr
            gc.collect()

        if not tier_models and global_model is None:
            print(f"    [WARN] No model trained for {comp} — insufficient data")
            continue

        total_anom = 0
        total_rows = 0
        n_collapsed = 0

        for p in parquets:
            hostname = p.stem
            score_path = scores_dir / f"{hostname}.parquet"
            if score_path.exists() and not force:
                continue

            meta = node_meta.get(hostname, {})
            df = meta.get("df")
            if df is None or df.empty:
                node_feat_for_read = meta.get("feat_cols", feature_cols)
                df = read_lstm_frame(p, node_feat_for_read)
            if df is None or df.empty:
                continue

            df = df.sort_values(TS).reset_index(drop=True)
            T = len(df)

            node_feat = meta.get("feat_cols", feature_cols)
            nf_node = len(node_feat)

            mat = np.zeros((T, nf_node), dtype=np.float32)
            for j, col in enumerate(node_feat):
                if col in df.columns:
                    v = pd.to_numeric(df[col], errors="coerce").fillna(0).values
                    mat[:, j] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

            windows = extract_windows(mat, stride=STRIDE_SCORE)
            if len(windows) == 0:
                del df
                continue

            node_tier_key = None
            if hostname in node_meta:
                node_tier_key = (
                    f"T{node_meta[hostname]['tier']}_"
                    f"{feature_schema_key(node_meta[hostname]['feat_cols'])}"
                )

            m = global_model
            if node_tier_key and node_tier_key in tier_models:
                m, _, _, _ = tier_models[node_tier_key]
            if m is None:
                del df
                continue

            starts = list(range(0, T - WINDOW + 1, STRIDE_SCORE))
            ts_tier = np.zeros(T, dtype=np.int8)
            if "req_power_class" in df.columns:
                ts_tier = (
                    pd.to_numeric(df["req_power_class"], errors="coerce")
                    .fillna(0)
                    .values.astype(np.int8)
                )
            window_tier = np.array(
                [
                    max(0, min(4, int(ts_tier[s : s + WINDOW].mean().round())))
                    for s in starts
                ],
                dtype=np.int8,
            )

            window_scores_arr = score_windows(m, windows, device=device)

            if "split" in df.columns:
                is_train = df["split"].values == "train"
                train_mask = np.array([is_train[s : s + WINDOW].all() for s in starts])
                train_maes = window_scores_arr[train_mask]
                if len(train_maes) >= 20:
                    node_mu = float(train_maes.mean())
                    node_sg = float(train_maes.std()) + 1e-6
                else:
                    node_mu, node_sg = global_mae_mean, global_mae_std
            else:
                node_mu, node_sg = global_mae_mean, global_mae_std

            window_z = (window_scores_arr - node_mu) / node_sg

            ts_recon_error = window_scores_to_timestep(window_scores_arr, T)
            ts_recon_z = window_scores_to_timestep(window_z, T)
            ts_tier_out = window_scores_to_timestep(
                window_tier.astype(np.float32), T
            ).astype(np.int8)

            is_anomaly = np.abs(ts_recon_z) > ANOM_Z_THRESH

            n_anom = int(is_anomaly.sum())
            anom_rate = n_anom / max(T, 1)
            collapsed = False

            if 0.15 < anom_rate <= MAX_ANOMALY_RATE:
                is_anomaly = np.abs(ts_recon_z) > 7.0
                n_anom = int(is_anomaly.sum())
                anom_rate = n_anom / max(T, 1)

            if anom_rate > MAX_ANOMALY_RATE:
                print(
                    f"    [WARN] {hostname}: LSTM anomaly rate {anom_rate:.1%} "
                    f"> {MAX_ANOMALY_RATE:.0%} threshold — model collapsed. "
                    f"Suppressing LSTM flags for this node."
                )
                is_anomaly[:] = False
                ts_recon_z[:] = 0.0
                n_anom = 0
                collapsed = True
                n_collapsed += 1

            result = pd.DataFrame(
                {
                    TS: pd.to_datetime(df[TS], utc=True),
                    "hostname": hostname,
                    "component": comp,
                    "split": df["split"].values if "split" in df.columns else "unknown",
                    "lstm_recon_error": ts_recon_error,
                    "lstm_recon_z": ts_recon_z.astype(np.float32),
                    "lstm_is_anomaly": is_anomaly,
                    "lstm_tier": ts_tier_out,
                }
            )

            training_log[hostname] = {
                "tier_key": node_tier_key or "global",
                "n_rows": T,
                "n_anomalous": n_anom,
                "anomaly_rate": round(float(anom_rate), 4),
                "collapsed": collapsed,
                "node_mae_mean": round(float(node_mu), 6),
                "node_mae_std": round(float(node_sg), 6),
            }

            total_anom += n_anom
            total_rows += T
            save_parquet(result, score_path)

            status = " [COLLAPSED→suppressed]" if collapsed else ""
            print(
                f"    {hostname:20s}: lstm_anom={n_anom:>5,} "
                f"({100*n_anom/max(T,1):.2f}%){status}"
            )

            del df, mat, windows, result
            gc.collect()

        elapsed = time.perf_counter() - t0
        print(
            f"\n[lstm_ae] {comp.upper()} done.  "
            f"total_lstm_anom={total_anom:,}/{total_rows:,} "
            f"({100*total_anom/max(total_rows,1):.2f}%)  "
            f"elapsed={elapsed:.1f}s"
        )
        if n_collapsed:
            print(
                f"  {n_collapsed} node(s) collapsed and suppressed. "
                f"See {model_dir / 'training_log.json'}"
            )

        for _, (m, _, _, _) in tier_models.items():
            del m
        if global_model is not None:
            del global_model
        del tier_models
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    log_name = f"training_log__{only_comp}.json" if only_comp else "training_log.json"
    (model_dir / log_name).write_text(json.dumps(training_log, indent=2))

    print(f"\n[lstm_ae] Done.  Scores: {scores_dir}")
    print(f"  Training log: {model_dir / log_name}")


if __name__ == "__main__":
    force_run = os.environ.get("INSIGHT_HPC_LSTM_FORCE", "1") == "1"
    run_lstm_autoencoder(force=force_run)
