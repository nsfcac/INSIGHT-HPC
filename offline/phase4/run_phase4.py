from __future__ import annotations

import argparse, sys, time, traceback
from pathlib import Path

from shared.utils.io_utils import load_config


def header(step: int, name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Step {step} — {name}")
    print(f"{'=' * 70}")


def try_run(label: str, fn, force: bool) -> bool:
    try:
        fn(force)
        return True
    except Exception as e:
        print(f"\n[run_phase4] ERROR in {label}: {e}")
        traceback.print_exc()
        return False


# Parse CLI args and run this module.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "INSIGHT-HPC Phase IV — Scoring, Fusion, Precision/Recall, Alerts\n\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute all outputs even if they exist"
    )
    parser.add_argument(
        "--step",
        nargs="+",
        type=int,
        default=None,
        help="Run only these step numbers (e.g. --step 1 3)",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compatibility flag; the lean Phase IV runner only has "
        "the core scoring/evaluation steps.",
    )
    args = parser.parse_args()
    force = args.force
    only = set(args.step) if args.step else None
    if args.metrics_only and only is None:
        only = {1, 2, 3}

    cfg = load_config()
    p4_cfg = cfg.get("phase4", {})
    gt_dir = Path(
        cfg.get("ground_truth_dir", p4_cfg.get("ground_truth_dir", "offline/data/ground_truth"))
    )
    gt_path = gt_dir / "events.parquet"
    gt_csv = gt_dir / "tickets.csv"

    tpl_files = ["tickets.csv", "node_failures.csv", "slurm_failures.csv"]
    if not any((gt_dir / f).exists() for f in tpl_files):
        from offline.phase4.ground_truth_loader import write_templates

        write_templates(gt_dir)
        print("  [gt] Blank GT templates created — fill them in and re-run.")

    if not gt_path.exists() and not gt_csv.exists():
        print("=" * 70)
        print("  WARNING: No ground truth found.")
        print(f"  Expected: {gt_csv}")
        print("=" * 70)
        print("  Continuing — steps 1+2 (GT load + synthetic GT) can still run.")
        print("  Step 3 will train/evaluate only if confirmed events are available.")
        print("")

    steps: list = []

    def step1(f):
        from offline.phase4.ground_truth_loader import run_load_ground_truth

        run_load_ground_truth(force=f)

    steps.append((1, "ground_truth_loader  ", step1))

    def step2(f):
        from offline.phase4.synthetic_ground_truth import generate

        generate(force=f)
        from offline.phase4.ground_truth_loader import run_load_ground_truth

        run_load_ground_truth(force=True)

    steps.append((2, "synthetic_ground_truth ", step2))

    def step3(f):
        from offline.phase4.score_fusion import run as run_score_fusion

        run_score_fusion(force_retrain=f)
        import json

        rep_dir = Path(cfg["phase4"].get("reports_dir", "offline/reports/phase4_eval"))
        pr_path = rep_dir / "precision_recall.json"
        if pr_path.exists():
            try:
                d = json.loads(pr_path.read_text())
                t = d.get("per_split", {}).get("test", {})
                hi = t.get("fused_high_v2", {})
                print(
                    f"[run_phase4] test headline: "
                    f"P={float(hi.get('precision', 0.0)):.4f}  "
                    f"R={float(hi.get('recall', 0.0)):.4f}  "
                    f"F1={float(hi.get('f1', 0.0)):.4f}  "
                    f"AUROC={float(t.get('auroc_episode', 0.0)):.4f}"
                )
            except Exception as e:
                print(f"[run_phase4] headline read failed: {e}")

    steps.append((3, "score_fusion  ", step3))

    t_total = time.perf_counter()
    results: dict = {}

    for num, name, fn in steps:
        if only and num not in only:
            continue
        header(num, name)
        t0 = time.perf_counter()
        ok = try_run(name, fn, force)
        results[name] = ("ok" if ok else "FAILED", round(time.perf_counter() - t0, 1))

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Phase IV complete  total={elapsed:.1f}s")
    print(f"{'=' * 70}")
    for name, (status, t) in results.items():
        flag = "OK  " if status == "ok" else "FAIL"
        print(f"  [{flag}]  {name:55s}  {status:6s}  {t:.1f}s")

    failures = [n for n, (s, _) in results.items() if s != "ok"]
    if failures:
        print(f"\n  FAILED steps: {', '.join(failures)}")
        sys.exit(1)

    out_dir = p4_cfg.get("output_dir", "offline/data/phase4")
    rep_dir = p4_cfg.get("reports_dir", "offline/reports/phase4_eval")
    print(f"""
  Key outputs:
    {out_dir}/fused_alerts.parquet            <- per-minute merged detectors
    {out_dir}/fused_alerts_v2.parquet         <- per-episode scored alerts
    {out_dir}/phase_scores.parquet            <- per-row phase 1/2/3 scores
    {out_dir}/fusion_gbdt.pkl                 <- trained GBDT classifier
    {rep_dir}/precision_recall.json           <- canonical P/R/F1/AUROC/AUPRC
    {rep_dir}/operator_alerts.{{csv,json}}    <- per-alert actionable messages
""")


if __name__ == "__main__":
    main()
