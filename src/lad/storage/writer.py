"""Append-safe JSONL writer plus raw-response caching and checkpointing.

Layout:
  data/raw/<source>/<yyyy-mm-dd>/page_00001.json  -- cached raw responses
  data/raw/<source>/checkpoint.json                -- resume state
  data/processed/<source>/records.jsonl            -- normalized, rights-clean
  data/processed/<source>/needs_review.jsonl        -- normalized, rights-unclear
  data/logs/run_summary.jsonl                       -- one line per connector run
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_DIR = DATA_DIR / "processed_parquet"
LOGS_DIR = DATA_DIR / "logs"


def raw_page_path(source_name: str, run_date: date, page_index: int) -> Path:
    day_dir = RAW_DIR / source_name / run_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"page_{page_index:05d}.json"


def write_raw_page(source_name: str, run_date: date, page_index: int, raw_text: str) -> Path:
    path = raw_page_path(source_name, run_date, page_index)
    path.write_text(raw_text, encoding="utf-8")
    return path


def processed_path(source_name: str, needs_review: bool = False) -> Path:
    source_dir = PROCESSED_DIR / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    filename = "needs_review.jsonl" if needs_review else "records.jsonl"
    return source_dir / filename


def append_record(source_name: str, record: BaseModel, needs_review: bool = False) -> None:
    path = processed_path(source_name, needs_review=needs_review)
    with path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def checkpoint_path(source_name: str) -> Path:
    source_dir = RAW_DIR / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    return source_dir / "checkpoint.json"


def load_checkpoint(source_name: str) -> dict[str, Any]:
    path = checkpoint_path(source_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(source_name: str, state: dict[str, Any]) -> None:
    checkpoint_path(source_name).write_text(json.dumps(state), encoding="utf-8")


def append_run_summary(summary: dict[str, Any]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "run_summary.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
