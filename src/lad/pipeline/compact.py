"""Compacts finalized JSONL output into partitioned Parquet for downstream
querying (e.g. with DuckDB) without running a database server."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lad.pipeline.parquet_utils import json_encode_dict_fields
from lad.storage import writer


def compact_source(source_name: str, run_date: date | None = None, needs_review: bool = False) -> Path | None:
    run_date = run_date or date.today()
    jsonl_path = writer.processed_path(source_name, needs_review=needs_review)
    rows = writer.read_jsonl(jsonl_path)
    if not rows:
        return None

    table = pa.Table.from_pylist(json_encode_dict_fields(rows))
    partition = "needs_review" if needs_review else "records"
    out_dir = writer.PARQUET_DIR / f"source={source_name}" / f"date={run_date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{partition}.parquet"
    pq.write_table(table, out_path)
    return out_path
