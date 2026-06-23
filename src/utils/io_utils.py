from __future__ import annotations

import gc
import shutil
import tempfile
import time
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


# Load YAML config and apply env path overrides / run-suffix.
def load_config(path: str = "configs/config.yaml") -> dict:
    import os
    with open(path) as f:
        cfg = yaml.safe_load(f)

    PATH_ENV_OVERRIDES = [
        ("INSIGHT_HPC_AUDITED_OVERRIDE",          "audited"),
        ("INSIGHT_HPC_MASTER_OVERRIDE",           "master"),
        ("INSIGHT_HPC_FEATURES_OVERRIDE",         "features"),
        ("INSIGHT_HPC_FEATURES_ALIGNED_OVERRIDE", "features_aligned"),
    ]
    for env_key, cfg_key in PATH_ENV_OVERRIDES:
        val = os.environ.get(env_key)
        if val:
            cfg.setdefault("paths", {})[cfg_key] = val

    run_suffix = os.environ.get("INSIGHT_HPC_RUN_SUFFIX", "")
    if run_suffix:
        paths    = cfg.setdefault("paths", {})
        explicit = {
            "master":           "INSIGHT_HPC_MASTER_OVERRIDE",
            "features":         "INSIGHT_HPC_FEATURES_OVERRIDE",
            "features_aligned": "INSIGHT_HPC_FEATURES_ALIGNED_OVERRIDE",
        }
        for key in explicit:
            if key in paths and not paths[key].endswith(run_suffix):
                if not os.environ.get(explicit[key]):
                    paths[key] = paths[key] + run_suffix

        for phase in ("phase1", "phase2", "phase3", "phase4"):
            node = cfg.get(phase)
            if not isinstance(node, dict):
                continue
            for subkey in ("output_dir", "reports_dir"):
                val = node.get(subkey)
                if isinstance(val, str) and not val.endswith(run_suffix):
                    node[subkey] = val + run_suffix

    return cfg


# Load a component's public lookup tables.
def load_public_tables(component: str, raw_base: Path) -> Dict[str, pd.DataFrame]:
    pub_dir = raw_base / component / "public"
    tables  = {}
    for name in ["nodes", "fqdd", "source", "metrics_definition"]:
        p = pub_dir / f"{name}.parquet"
        if p.exists():
            try:
                tables[name] = pd.read_parquet(p, engine="pyarrow")
                tables[name].columns = [c.strip().lower()
                                        for c in tables[name].columns]
            except Exception as e:
                print(f"  [WARN] {name}.parquet: {e}")
    return tables


# Build id->name maps for node/fqdd/source.
def build_lookup_dicts(pub: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
    def to_int_keys(d: dict) -> dict:
        try:
            return {int(k): v for k, v in d.items()}
        except (ValueError, TypeError):
            return d

    node_map = fqdd_map = source_map = {}

    if "nodes" in pub:
        n      = pub["nodes"]
        id_col = next((c for c in ["nodeid", "id"] if c in n.columns), None)
        h_col  = "hostname" if "hostname" in n.columns else None
        if id_col and h_col:
            node_map = to_int_keys(dict(zip(n[id_col], n[h_col])))

    if "fqdd" in pub:
        f  = pub["fqdd"]
        nc = next((c for c in ["fqdd", "fqddname", "name"] if c in f.columns), None)
        if "id" in f.columns and nc:
            fqdd_map = to_int_keys(dict(zip(f["id"], f[nc])))

    if "source" in pub:
        s  = pub["source"]
        nc = next((c for c in ["source", "sourcename", "name"] if c in s.columns), None)
        if "id" in s.columns and nc:
            source_map = to_int_keys(dict(zip(s["id"], s[nc])))

    return node_map, fqdd_map, source_map


# Pivot a long sensor frame to wide per-sensor columns.
def long_to_wide(df: pd.DataFrame, ts_col: str, id_col: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    try:
        key = [ts_col, id_col, "sensor_name"]
        if df.duplicated(subset=key).any():
            df = df.drop_duplicates(subset=key, keep="last")

        wide = (
            df.set_index(key)["_avg"]
            .unstack("sensor_name")
        )
        wide.columns = [f"{c}_avg" for c in wide.columns]
        wide.columns.name = None
        wide = wide.reset_index()
        return wide
    except Exception as e:
        print(f"    [WARN] pivot failed: {e}")
        return None


BATCH_ROWS  = 2_000_000
FLUSH_EVERY = 5

# Write a batch of aggregated rows to a spill parquet.
def write_spill(batches: List[pd.DataFrame], spill_dir: Path, spill_idx: int, key: List[str]) -> int:
    if not batches:
        return spill_idx
    combined = pd.concat(batches, ignore_index=True)
    agg = (combined
           .groupby(key, sort=False)
           .agg(_sum=("_sum", "sum"), _count=("_count", "sum"))
           .reset_index())
    del combined
    out = spill_dir / f"spill_{spill_idx:04d}.parquet"
    agg.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    del agg
    return spill_idx + 1


# Write a batch of duplicate-count rows to a spill parquet.
def write_dup_spill(batches: List[pd.DataFrame], spill_dir: Path, spill_idx: int, key: List[str]) -> int:
    if not batches:
        return spill_idx
    combined = pd.concat(batches, ignore_index=True)
    agg = (combined
           .groupby(key, sort=False)["_raw_count"]
           .sum().reset_index())
    del combined
    out = spill_dir / f"dup_{spill_idx:04d}.parquet"
    agg.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    del agg
    return spill_idx + 1


# Stream-load and aggregate one raw metric parquet to a wide frame.
def load_metric_parquet(component: str, source_dir, metric_file: Path, public_tables: Dict[str, pd.DataFrame], cfg: dict, verbose: bool = True, node_ids=None) -> Optional[pd.DataFrame]:
    t_total = time.perf_counter()

    source_name  = Path(source_dir).name
    use_fqdd_src = (source_name == "idrac")
    source_cfg   = cfg.get("sources", {}).get(source_name,
                   cfg["sources"].get("idrac", {}))
    schema       = source_cfg["schema"]

    ts_col   = schema["timestamp_col"]
    node_col = schema["node_col"]
    val_col  = schema["value_col"]
    fqdd_col = schema.get("fqdd_col",   "fqdd")
    src_col  = schema.get("source_col", "source")

    needed = [ts_col, node_col, val_col]
    if use_fqdd_src:
        needed += [fqdd_col, src_col]

    try:
        file_cols  = set(pq.read_schema(metric_file).names)
        read_cols  = [c for c in needed if c in file_cols]
        pf         = pq.ParquetFile(metric_file)
        total_rows = pf.metadata.num_rows
    except Exception as e:
        print(f"    [WARN] Cannot open {metric_file.name}: {e}")
        return None

    node_map, fqdd_map, source_map = build_lookup_dicts(public_tables)

    if verbose:
        if use_fqdd_src and len(fqdd_map) == 0:
            print("    [WARN] fqdd_map is empty — all fqdd IDs will map to 'unk'")
        if use_fqdd_src and len(source_map) == 0:
            print("    [WARN] source_map is empty — all source IDs will map to 'unk'")

    spill_dir = Path(tempfile.mkdtemp(prefix="insight_spill_"))
    try:
        return load_with_spill(
            pf, read_cols, total_rows,
            ts_col, node_col, val_col, fqdd_col, src_col,
            use_fqdd_src, metric_file,
            node_map, fqdd_map, source_map,
            spill_dir, verbose, t_total,
            node_ids=node_ids,
        )
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)


# Batch-aggregate a metric file to disk spills, then re-aggregate.
def load_with_spill(pf, read_cols, total_rows, ts_col, node_col, val_col, fqdd_col, src_col, use_fqdd_src, metric_file, node_map, fqdd_map, source_map, spill_dir, verbose, t_total, node_ids=None) -> Optional[pd.DataFrame]:

    key     = [ts_col, "hostname", "sensor_name"]
    dup_key = [ts_col, "hostname"]

    val_batches = []
    dup_batches = []
    val_spill   = 0
    dup_spill   = 0

    t_read = t_decode = t_agg = t_flush = 0.0
    n_batches = rows_read = 0

    batch_iter = None
    node_filter_pushed = False
    if node_ids is not None and node_col in read_cols:
        try:
            node_values = sorted(node_ids)
            scanner = ds.dataset(metric_file, format="parquet").scanner(
                columns=read_cols,
                filter=ds.field(node_col).isin(node_values),
                batch_size=BATCH_ROWS,
            )
            batch_iter = scanner.to_batches()
            node_filter_pushed = True
        except Exception as e:
            if verbose:
                print(f"      [WARN] pyarrow node filter failed; falling back to pandas filter: {e}")
    if batch_iter is None:
        batch_iter = pf.iter_batches(batch_size=BATCH_ROWS, columns=read_cols)

    for batch in batch_iter:
        t0    = time.perf_counter()
        chunk = batch.to_pandas()
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        if node_ids is not None and not node_filter_pushed and node_col in chunk.columns:
            chunk = chunk[chunk[node_col].isin(node_ids)]
        t_read += time.perf_counter() - t0
        rows_read += len(chunk)
        if node_ids is not None and chunk.empty:
            continue

        t0 = time.perf_counter()
        chunk[val_col] = pd.to_numeric(chunk[val_col], errors="coerce").astype("float32")

        if node_col in chunk.columns:
            chunk["hostname"] = chunk[node_col].map(node_map).astype(object)
            unmap = chunk["hostname"].isna()
            if unmap.any():
                chunk.loc[unmap, "hostname"] = chunk.loc[unmap, node_col].astype(str)
            chunk.drop(columns=[node_col], inplace=True)
        else:
            chunk["hostname"] = "unknown"

        if not pd.api.types.is_datetime64_any_dtype(chunk[ts_col]):
            if pd.api.types.is_integer_dtype(chunk[ts_col]):
                chunk[ts_col] = pd.to_datetime(chunk[ts_col], unit="ns", utc=True)
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chunk[ts_col] = pd.to_datetime(chunk[ts_col], utc=True)
        chunk[ts_col] = chunk[ts_col].dt.floor("60s")

        if use_fqdd_src:
            fnames = (chunk[fqdd_col].map(fqdd_map).fillna("unk")
                      if fqdd_col in chunk.columns
                      else pd.Series("unk", index=chunk.index))
            snames = (chunk[src_col].map(source_map).fillna("unk")
                      if src_col in chunk.columns
                      else pd.Series("unk", index=chunk.index))
            same   = fnames.astype(str).str.lower() == snames.astype(str).str.lower()
            chunk["sensor_name"] = np.where(
                same,
                fnames.astype(str).str.lower(),
                fnames.astype(str).str.lower() + "_" + snames.astype(str).str.lower()
            )
            chunk.drop(columns=[c for c in [fqdd_col, src_col]
                                 if c in chunk.columns], inplace=True)
        else:
            chunk["sensor_name"] = metric_file.stem.lower()
        t_decode += time.perf_counter() - t0

        t0 = time.perf_counter()
        dup = (chunk.groupby(dup_key, sort=False)[val_col]
                    .size().reset_index(name="_raw_count"))
        dup_batches.append(dup)

        valid = chunk[val_col].notna()
        if valid.any():
            agg = (chunk.loc[valid]
                        .groupby(key, sort=False)[val_col]
                        .agg(_sum="sum", _count="count")
                        .reset_index())
            val_batches.append(agg)
        t_agg += time.perf_counter() - t0

        n_batches += 1
        del chunk, dup
        if valid.any():
            del agg

        if n_batches == 1 and use_fqdd_src and val_batches and verbose:
            sample_sensors = val_batches[0]["sensor_name"].unique()
            n_unk = sum(1 for s in sample_sensors if "unk" in str(s))
            if n_unk == len(sample_sensors):
                print(f"    [WARN] ALL sensors mapped to 'unk' — "
                      f"fqdd_map keys: {list(fqdd_map.keys())[:5]}  "
                      f"chunk fqdd sample: {val_batches[0].get('fqdd_col', 'N/A')}")

        if n_batches % FLUSH_EVERY == 0:
            t0        = time.perf_counter()
            val_spill = write_spill(val_batches, spill_dir, val_spill, key)
            dup_spill = write_dup_spill(dup_batches, spill_dir, dup_spill, dup_key)
            val_batches = []
            dup_batches = []
            gc.collect()
            t_flush += time.perf_counter() - t0

    if val_batches:
        val_spill = write_spill(val_batches, spill_dir, val_spill, key)
        del val_batches; gc.collect()
    if dup_batches:
        dup_spill = write_dup_spill(dup_batches, spill_dir, dup_spill, dup_key)
        del dup_batches; gc.collect()

    if val_spill == 0:
        print(f"    [WARN] {metric_file.name}: no valid values")
        return None

    t0 = time.perf_counter()
    spill_files = sorted(spill_dir.glob("spill_*.parquet"))

    dataset   = ds.dataset(spill_files, format="parquet")
    final_tbl = dataset.to_table()
    final_df  = final_tbl.to_pandas()
    del final_tbl; gc.collect()

    final_long = (final_df
                  .groupby(key, sort=False)
                  .agg(_sum=("_sum", "sum"), _count=("_count", "sum"))
                  .reset_index())
    del final_df; gc.collect()

    final_long["_avg"] = (final_long["_sum"] / final_long["_count"]).astype("float32")
    final_long.drop(columns=["_sum", "_count"], inplace=True)
    t_reagg = time.perf_counter() - t0

    t0   = time.perf_counter()
    wide = long_to_wide(final_long, ts_col, "hostname")
    del final_long; gc.collect()
    t_pivot = time.perf_counter() - t0

    if wide is None or wide.empty:
        return None

    t0        = time.perf_counter()
    dup_files = sorted(spill_dir.glob("dup_*.parquet"))
    dup_tbl   = ds.dataset(dup_files, format="parquet").to_table()
    dup_df    = dup_tbl.to_pandas()
    del dup_tbl; gc.collect()

    dup_agg = (dup_df
               .groupby(dup_key, sort=False)["_raw_count"]
               .sum().reset_index()
               .rename(columns={"_raw_count": "_dup_count"}))
    del dup_df; gc.collect()

    wide = wide.merge(dup_agg, on=dup_key, how="left")
    del dup_agg; gc.collect()
    wide["_dup_count"] = wide["_dup_count"].fillna(1).astype("int32")
    t_dup = time.perf_counter() - t0

    return wide


# Atomically write a dataframe to parquet.
def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")


# Read a parquet file, returning None on failure.
def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        print(f"  [WARN] load_parquet {path}: {e}")
        return None