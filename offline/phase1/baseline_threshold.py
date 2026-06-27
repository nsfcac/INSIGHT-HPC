from __future__ import annotations

from offline.phase1.baseline_threshold_module.threshold_loading import *
from offline.phase1.baseline_threshold_module.node_scoring import *
from offline.phase1.baseline_threshold_module.pdu_scoring import *


# Score every node against per-rule thresholds and write node and PDU alerts.
def run_baseline_threshold(force: bool = False) -> None:
    cfg = load_config()
    master_dir = Path(cfg["paths"]["master"])
    out_dir = Path(cfg["phase1"]["output_dir"]) / "baseline_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)

    alerts_path = out_dir / "alerts.parquet"
    pdu_path = out_dir / "pdu_alerts.parquet"

    if alerts_path.exists() and not force:
        print("[threshold] Alerts exist — skipping (use force=True to rerun)")
        return

    cfg_thr = cfg["phase1"]["threshold_baseline"]
    windows = load_maintenance_windows(cfg)

    print(
        f"\n[threshold] Threshold-only baseline model  (dynamic thresholds from CSV when available)"
    )
    print(
        f"  Config fallbacks: "
        f"sys_pwr>{cfg_thr.get('system_power_w', 1800)}W  "
        f"cpu_temp>{cfg_thr.get('cpu_temp_c', 85)}°C  "
        f"inlet>{cfg_thr.get('inlet_temp_c', 35)}°C  "
        f"fan_rpm<{cfg_thr.get('fan_rpm_low', 1500)}  "
        f"fan_temp_xchk>{cfg_thr.get('fan_temp_crosscheck', 70)}°C"
    )

    print("\n  Computing PDU training thresholds ...")
    pdu_thr = compute_pdu_thresholds(
        master_dir, cfg_thr.get("pdu_percentile", 95), windows
    )
    print(f"  PDU outlets with thresholds: {len(pdu_thr)}")

    t0 = time.perf_counter()
    node_frames = []
    total_rows = flagged_rows = 0

    workers = phase1_workers()
    THRESHOLD_WORKER_CONTEXT.clear()
    THRESHOLD_WORKER_CONTEXT["windows"] = windows
    THRESHOLD_WORKER_CONTEXT["thr"] = {}

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        if comp == "infra":
            continue

        comp_dir = master_dir / comp
        if not comp_dir.exists():
            continue

        dyn_thr = load_dynamic_thresholds(cfg, comp)
        THRESHOLD_WORKER_CONTEXT["thr"][comp] = {**cfg_thr, **dyn_thr}

        parquets = apply_node_limit(sorted(comp_dir.glob("*.parquet")))
        print(f"\n  [{comp.upper()}]  {len(parquets)} nodes")

        tasks = [(comp, p.stem, str(p)) for p in parquets]
        if workers == 1 or len(tasks) <= 1:
            results = [threshold_score_one(*t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
                results = list(ex.map(threshold_score_one, *zip(*tasks)))

        for hostname, scored, n_rows, n_flagged, log in sorted(
            results, key=lambda r: r[0]
        ):
            if log:
                print(log, flush=True)
            if scored is not None:
                total_rows += n_rows
                flagged_rows += n_flagged
                node_frames.append(scored)

    THRESHOLD_WORKER_CONTEXT.clear()
    gc.collect()

    if not node_frames:
        print("[threshold] No node data found — nothing to save.")
        return

    alerts = pd.concat(node_frames, ignore_index=True)
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], utc=True)
    save_parquet(alerts, alerts_path)

    elapsed = time.perf_counter() - t0
    print(
        f"\n  [threshold] Total rows={total_rows:,}  "
        f"flagged={flagged_rows:,} ({100*flagged_rows/max(total_rows,1):.2f}%)  "
        f"time={elapsed:.1f}s"
    )
    print(f"  Alerts saved: {alerts_path}")

    print("\n  Per-rule flag rates (across all components):")
    for r in RULES:
        col = f"rule_{r}"
        if col in alerts.columns:
            rate = 100 * alerts[col].sum() / max(len(alerts), 1)
            print(f"    {r:25s}: {rate:5.2f}%")

    print("\n  Scoring PDU outlets (outlet p95 + rack budget check) ...")
    pdu_alerts = score_pdu(master_dir, pdu_thr, windows, cfg)
    if pdu_alerts is not None:
        n_outlet = int(pdu_alerts["is_flagged_outlet"].fillna(False).sum())
        n_budget = int(pdu_alerts["is_flagged_rack_budget"].fillna(False).sum())
        n_either = int(pdu_alerts["is_flagged"].fillna(False).sum())
        save_parquet(pdu_alerts, pdu_path)
        print(
            f"  PDU: {len(pdu_alerts):,} rows  "
            f"outlet_flagged={n_outlet:,}  "
            f"rack_budget_flagged={n_budget:,}  "
            f"either={n_either:,}  saved: {pdu_path}"
        )


if __name__ == "__main__":
    run_baseline_threshold(force=True)
