"""Shared helpers for converting JSONL rows to Parquet -- used by both
compact.py and publish_hf.py, so the fix below only has to exist once."""

from __future__ import annotations

import json
from typing import Any


def json_encode_dict_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """pyarrow infers a Parquet struct schema from dict-valued fields --
    when a field is an empty dict `{}` on every row (e.g. Getty AAT never
    populates VocabularyTerm.alt_labels/scope_note/source_concept_ids),
    it infers a zero-field struct, which Parquet's writer can't serialize
    at all ("Cannot write struct type with no child field"). JSON-encoding
    every dict-valued field to a string sidesteps that fragile inference
    entirely -- costs a `json_extract`-style call to read a language back
    out in DuckDB, but never breaks regardless of which fields happen to
    be empty across an entire source."""
    encoded = []
    for row in rows:
        encoded.append({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in row.items()})
    return encoded
