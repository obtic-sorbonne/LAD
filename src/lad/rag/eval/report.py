"""Serializes eval results (rag/eval/run_eval.py's output) to JSON/CSV --
the actual on-disk artifacts behind "raw evaluation results," a project
deliverable. Every number reported in PROJECT_STATUS.md this project has
produced so far came from one-off scripts printing to stdout, hand-
transcribed into markdown tables -- nothing was ever saved as a
reusable, reanalyzable file. This module (plus run_eval.py's raw_rows
output) closes that gap for good, not just for this one deliverable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(results: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Writes `rows` (list of flat dicts, all sharing the same keys -- true
    of run_eval.py's retrieval_raw/synthesis_raw lists) as a CSV. An empty
    `rows` still creates the file (empty), rather than raising, so a
    downstream script that expects the path to exist doesn't need a
    special case for "the run produced zero rows"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
