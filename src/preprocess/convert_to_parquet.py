from __future__ import annotations

try:
    import cudf.pandas

    cudf.pandas.install()
    print("GPU Acceleration via cudf.pandas enabled.")
except ImportError:
    print("cudf.pandas not found. Running on CPU via standard pandas.")

import re, time, warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.io_utils import load_config

BYTEA_RE = re.compile(r"^\{(\d+(?:,\d+)*)\}$")

NUMERIC_NAMES = re.compile(
    r"(^id$|_id$|nodeid|jobid|count|num_|^n_|ncpu|ngpu|nnode|nnodes|ncores"
    r"|memory|mem_|^size$|port|rank|slot|index|seq|^flag|code|exit|signal"
    r"|priority|weight|limit|quota|step|offset|version|epoch|unix|_at$)",
    re.IGNORECASE,
)

TS_NAMES = re.compile(
    r"(^timestamp$|_time$|^time$|created_at|updated_at|submit|^start$|^end$|eligible)",
    re.IGNORECASE,
)


# Decode a Postgres bytea {int,...} string back to text when it is valid UTF-8.
def decode_one(val) -> str:
    if not isinstance(val, str):
        return val
    m = BYTEA_RE.match(val.strip())
    if not m:
        return val
    try:
        ints = [int(x) for x in m.group(1).split(",")]
    except ValueError:
        return val
    if any(i > 255 for i in ints):
        return val
    try:
        decoded = bytes(ints).decode("utf-8")
        if decoded and all(c.isprintable() or c in "\t\n\r " for c in decoded):
            return decoded
    except ValueError:
        pass
    return val


# Decide whether a column holds encodable bytea arrays worth decoding.
def needs_decode(col: str, series: pd.Series) -> tuple[bool, str]:
    if NUMERIC_NAMES.search(col):
        return False, "numeric col name"

    str_series = series.dropna().astype(str).head(200)
    if str_series.empty:
        return False, "empty"

    bytea_rows = str_series[str_series.str.match(r"^\{\d")]
    if bytea_rows.empty:
        return False, "no {int,...} values"

    for val in bytea_rows.head(20):
        m = BYTEA_RE.match(val.strip())
        if m:
            try:
                ints = [int(x) for x in m.group(1).split(",")]
                if any(i > 255 for i in ints):
                    return False, "integer array (value > 255)"
            except ValueError:
                pass

    return True, "ok"


# Return whether a column name and sample values look like a parseable timestamp.
def is_ts_col(col: str, series: pd.Series) -> bool:
    if not TS_NAMES.search(col):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    sample = series.dropna().astype(str).head(5)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(sample, utc=True)
        return True
    except Exception:
        return False


# Convert a column to UTC microsecond timestamps (Slurm integer epochs are seconds).
def to_utc_us(series: pd.Series, col: str) -> pd.Series:
    # Slurm submit/start/end are Unix epoch SECONDS, not nanoseconds.
    if pd.api.types.is_integer_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True).astype("datetime64[us, UTC]")

    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            return series.dt.tz_localize("UTC").astype("datetime64[us, UTC]")
        return series.dt.tz_convert("UTC").astype("datetime64[us, UTC]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, utc=True, errors="raise")
    return parsed.astype("datetime64[us, UTC]")


# Decode bytea columns, coerce value to float, and normalise timestamps in one chunk.
def transform_chunk(
    chunk: pd.DataFrame, decode_cols: set[str], ts_cols: set[str]
) -> pd.DataFrame:
    if "value" in chunk.columns:
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce").astype(
            "float64"
        )

    for col in decode_cols:
        if col in chunk.columns:
            chunk[col] = (
                chunk[col]
                .astype(str)
                .where(chunk[col].notna(), other=None)
                .apply(decode_one)
            )

    for col in ts_cols:
        if col in chunk.columns:
            try:
                chunk[col] = to_utc_us(chunk[col], col)
            except Exception as exc:
                raise ValueError(
                    f"Timestamp conversion failed for column '{col}' "
                    f"in chunk (first value: {chunk[col].iloc[0]!r}): {exc}"
                ) from exc

    return chunk


# Build an Arrow schema, forcing timestamp columns to us-UTC and value to float64.
def build_pa_schema(chunk: pd.DataFrame, ts_cols: set[str]) -> pa.Schema:
    fields = []
    inferred = pa.Schema.from_pandas(chunk, preserve_index=False)
    for field in inferred:
        if field.name in ts_cols:
            fields.append(pa.field(field.name, pa.timestamp("us", tz="UTC")))
        elif field.name == "value":
            fields.append(pa.field(field.name, pa.float64()))
        else:
            fields.append(field)
    return pa.schema(fields)


# Stream a CSV to parquet in chunks, decoding and normalising as it goes.
def csv_to_parquet(src: Path, dst: Path, chunksize: int = 1_000_000) -> None:
    time_to_convert = time.perf_counter()
    src_size_gb = src.stat().st_size / 1e9
    print(f"\nStarting to convert {src.name}  ({src_size_gb:.2f} GB)", flush=True)

    with open(src, "rb") as f:
        csv_lines = sum(1 for _ in f) - 1

    try:
        df_sample = pd.read_csv(
            src, nrows=2000, engine="c", dtype=str, keep_default_na=False
        )
    except Exception:
        df_sample = pd.read_csv(
            src, nrows=2000, engine="python", dtype=str, keep_default_na=False
        )

    df_sample.columns = [c.strip().lower() for c in df_sample.columns]
    decode_cols: set[str] = set()
    skip_reasons: dict[str, str] = {}
    for col in df_sample.columns:
        ok, reason = needs_decode(col, df_sample[col])
        if ok:
            decode_cols.add(col)
        elif reason not in ("no {int,...} values", "empty"):
            str_s = df_sample[col].dropna().astype(str).head(50)
            if str_s.str.match(r"^\{\d").any():
                skip_reasons[col] = reason

    df_sample_decoded = df_sample.copy()
    for col in decode_cols:
        df_sample_decoded[col] = df_sample_decoded[col].apply(decode_one)

    ts_cols = {
        c for c in df_sample_decoded.columns if is_ts_col(c, df_sample_decoded[c])
    }

    if decode_cols:
        print(f" - bytea cols: {sorted(decode_cols)}", flush=True)
    else:
        print(" - bytea cols: none", flush=True)

    if ts_cols:
        print(f" - timestamp cols: {sorted(ts_cols)}", flush=True)

    if skip_reasons:
        for col, reason in skip_reasons.items():
            print(f" - kept as-is: {col} ({reason})", flush=True)

    try:
        reader = pd.read_csv(
            src, chunksize=chunksize, engine="c", low_memory=False, on_bad_lines="warn"
        )
    except Exception:
        reader = pd.read_csv(
            src, chunksize=chunksize, engine="python", on_bad_lines="warn"
        )

    writer = None
    rows_in = 0
    rows_out = 0
    t_read = t_decode = t_write = 0.0

    with tqdm(
        total=csv_lines,
        unit=" rows",
        unit_scale=True,
        desc=f"    {src.stem[:28]}",
        leave=False,
    ) as pbar:
        for chunk in reader:
            t1 = time.perf_counter()
            chunk.columns = [c.strip().lower() for c in chunk.columns]
            rows_in += len(chunk)
            t_read += time.perf_counter() - t1

            t1 = time.perf_counter()
            chunk = transform_chunk(chunk, decode_cols, ts_cols)
            t_decode += time.perf_counter() - t1

            t1 = time.perf_counter()
            table = pa.Table.from_pandas(chunk, preserve_index=False)

            if writer is None:
                schema = build_pa_schema(chunk, ts_cols)
                table = table.cast(schema)
                writer = pq.ParquetWriter(dst, schema, compression="snappy")
            else:
                table = table.cast(writer.schema)

            writer.write_table(table)
            t_write += time.perf_counter() - t1
            rows_out += len(chunk)
            pbar.update(len(chunk))

    if writer:
        writer.close()

    written_metadata = pq.read_metadata(dst)
    if written_metadata.num_rows != rows_out:
        raise RuntimeError(
            f"Row count mismatch after writing {dst.name}: "
            f"expected {rows_out:,} but parquet metadata reports "
            f"{written_metadata.num_rows:,}. File may be corrupt."
        )

    elapsed = time.perf_counter() - time_to_convert
    dst_size_gb = dst.stat().st_size / 1e9
    skipped_by_pandas = csv_lines - rows_in
    dropped = rows_in - rows_out

    status_parts = []
    if skipped_by_pandas:
        status_parts.append(f"{skipped_by_pandas:,} rows skipped by parser")
    if dropped:
        status_parts.append(f"{dropped:,} rows dropped")
    status = f" - {', '.join(status_parts)}" if status_parts else " -"

    print(
        f"{status} rows={rows_out:,}  out={dst_size_gb:.3f}GB  "
        f"read={t_read:.1f}s  decode={t_decode:.1f}s  write={t_write:.1f}s  "
        f"total={elapsed:.1f}s  ({rows_out / max(elapsed, 0.001) / 1e6:.1f}M rows/s)",
        flush=True,
    )

    final_schema = pq.read_schema(dst)
    schema_str = "  ".join(f"{f.name}:{f.type}" for f in final_schema)
    print(f" - schema: {schema_str}", flush=True)


# Convert every raw CSV under the configured directory to parquet, skipping up-to-date outputs.
def convert_raw_to_parquet(
    config_path: str = "configs/config.yaml", force: bool = False
) -> None:
    cfg = load_config(str(config_path))
    raw_base = Path(cfg["paths"]["raw_csv"]).resolve()
    out_base = Path(cfg["paths"]["raw_parquet"]).resolve()

    csv_files = sorted(raw_base.rglob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files under {raw_base}\n")

    conversion_start = time.perf_counter()
    n_converted = n_skipped = n_failed = 0

    for src in csv_files:
        rel = src.relative_to(raw_base)
        dst = (out_base / rel).with_suffix(".parquet")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            print(f"This file already exists. Skipping {src.name}")
            n_skipped += 1
            continue

        try:
            csv_to_parquet(src, dst)
            n_converted += 1
        except Exception as e:
            import traceback

            print(f"Failed to convert {src.name} : {e}")
            traceback.print_exc()
            n_failed += 1

    elapsed = time.perf_counter() - conversion_start
    print(
        f"\nConversion complete. converted={n_converted}  skipped={n_skipped}  "
        f"failed={n_failed}  total={elapsed:.1f}s"
    )


if __name__ == "__main__":
    convert_raw_to_parquet(force=False)
