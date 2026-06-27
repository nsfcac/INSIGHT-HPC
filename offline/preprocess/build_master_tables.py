from __future__ import annotations

from offline.preprocess.build_master_tables_module.metric_loading import *
from offline.preprocess.build_master_tables_module.job_context import *
from offline.preprocess.build_master_tables_module.builders import *


# Build per-node and infra master tables from the audited parquet trees.
def build_master_tables(force: bool = False) -> None:
    require_slurm_for_heavy(
        "build_master_tables", "offline/scripts/run_preprocess.slurm"
    )
    cfg = load_config()
    audited_dir = Path(cfg["paths"]["audited"])
    raw_base = Path(cfg["paths"]["raw_parquet"])
    out_base = Path(cfg["paths"]["master"])

    apply_mask = cfg.get("master", {}).get("mask_on_audit_flags", False)
    print(
        f"\n[master] mask_on_audit_flags={apply_mask}  "
        f"({'selective: NaN spikes+bad_range, idle flatlines cleared' if apply_mask else 'keeping raw values, audit_flags passed through'})"
    )

    stage_start = time.perf_counter()
    total_rows = total_nodes = 0

    for comp_cfg in cfg["components"]:
        comp = comp_cfg["name"]
        print(f"\nComponent: {comp.upper()}")

        if comp == "infra":
            comp_start = time.perf_counter()
            source = "pdu"
            infra_audited = audited_dir / "infra" / source

            if infra_audited.exists():
                unit_ids: set = set()
                for p in sorted(infra_audited.glob("*.parquet")):
                    try:
                        df = pd.read_parquet(p, engine="pyarrow", columns=None)
                        df.columns = [c.strip().lower() for c in df.columns]
                        if "hostname" in df.columns:
                            unit_ids.update(df["hostname"].dropna().unique().tolist())
                        else:
                            nc = first_existing(list(df.columns), ["nodeid", "node_id"])
                            if nc:
                                unit_ids.update(
                                    df[nc].dropna().astype(str).unique().tolist()
                                )
                    except Exception:
                        pass

                if unit_ids:
                    out_dir = out_base / "infra" / source
                    out_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n  [{source.upper()}]  {len(unit_ids)} units")

                    for unit_id in sorted(unit_ids):
                        out_path = out_dir / f"{unit_id}.parquet"
                        n = build_infra_unit(
                            unit_id=str(unit_id),
                            source=source,
                            audited_dir=audited_dir,
                            out_path=out_path,
                            cfg=cfg,
                            force=force,
                            apply_mask=apply_mask,
                        )
                        if n is not None:
                            total_rows += n
                            total_nodes += 1
                        elif out_path.exists():
                            print(f"      {unit_id}: skip (exists)")
                        gc.collect()
                else:
                    print(f"  [WARN] no units found in audited/infra/{source}")

            print(f"\n  [INFRA done]  time={time.perf_counter() - comp_start:.1f}s")
            continue

        pub = load_public_tables(comp, raw_base)
        if "nodes" not in pub:
            print(f"  [WARN] no nodes table for {comp}")
            continue

        nodes_df = pub["nodes"]
        id_col = first_existing(list(nodes_df.columns), ["nodeid", "id", "node_id"])
        if id_col is None or "hostname" not in nodes_df.columns:
            print(f"  [WARN] nodes table missing id/hostname for {comp}")
            continue

        out_dir = out_base / comp
        out_dir.mkdir(parents=True, exist_ok=True)
        comp_start = time.perf_counter()

        # Per-node build is independent. workers=1 -> serial (identical); pool capped for memory (each worker holds full audited frames transiently).
        node_kwargs = [
            dict(
                hostname=row["hostname"],
                node_id=int(row[id_col]),
                node_meta=row.to_dict(),
                component=comp,
                audited_dir=audited_dir,
                raw_base=raw_base,
                out_path=out_dir / f"{row['hostname']}.parquet",
                cfg=cfg,
                force=force,
                apply_mask=apply_mask,
            )
            for _, row in nodes_df.iterrows()
        ]
        workers = master_workers()
        if workers == 1 or len(node_kwargs) <= 1:
            results = [master_build_one(kw) for kw in node_kwargs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(node_kwargs))) as ex:
                results = list(ex.map(master_build_one, node_kwargs))

        for hostname, n, out_path_str, log in results:
            if log.strip():
                print(log.rstrip(), flush=True)
            if n is not None:
                total_rows += n
                total_nodes += 1
            elif Path(out_path_str).exists():
                print(f"      {hostname}: skip (exists)")
        gc.collect()

        print(
            f"\n  [{comp.upper()} done]  time={time.perf_counter() - comp_start:.1f}s"
        )

    elapsed = time.perf_counter() - stage_start
    print(
        f"\nCompleted building master tables. nodes={total_nodes}  rows={total_rows:,}  time={elapsed:.1f}s"
    )


if __name__ == "__main__":
    build_master_tables(force=True)
