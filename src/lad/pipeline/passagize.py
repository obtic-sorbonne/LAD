"""Chunks harvested records into retrieval-unit Passages (150-200 tokens,
20-token overlap, per LAD paper §3.1) for the RAG system to index.

Draws from BOTH records.jsonl and needs_review.jsonl (decided: corpus needs
volume for a meaningful v1 prototype -- see plan Part A3). rights_statement/
reuse_risk are denormalized onto every Passage at chunk time so a passage
built from a needs_review row carries that status through retrieval and
synthesis instead of it being silently dropped.

Two source shapes need different extraction (see schema.py):
  - HeritageRecord rows (Europeana/UNESDOC/WDL): one language per row,
    chunk `title` and `description` separately.
  - VocabularyTerm rows (UNESCO Thesaurus/Getty AAT): all languages on one
    row, chunk `scope_note[lang]` (falling back to `pref_label[lang]` when
    there's no scope note) per language present.

Arabic passages are normalized via pipeline/arabic_normalize.normalize()
for the indexed `text` field; `text_raw` keeps the original for citation/
display. Only applied when language_code cleanly identifies as Arabic
(ar/ara/arabic) -- UNESDOC's compound codes like "eng,ara,rus" are left
unnormalized since the row is genuinely multi-language and can't be cleanly
attributed to one language's text.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from lad.pipeline import arabic_normalize
from lad.schema import Passage, ReuseRisk
from lad.storage.writer import DATA_DIR, PROCESSED_DIR, read_jsonl

PASSAGES_DIR = DATA_DIR / "passages"

_TOKEN_RE = re.compile(r"\S+")
_CHUNK_SIZE = 180
_CHUNK_OVERLAP = 20


def _is_arabic(lang_code: str | None) -> bool:
    return (lang_code or "").strip().lower() in {"ar", "ara", "arabic"}


def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[tuple[str, int, int]]:
    """Splits `text` into (chunk, char_start, char_end) tuples of up to
    `chunk_size` whitespace-delimited tokens, overlapping by `overlap`
    tokens. Char offsets are real spans into `text` (regex match positions,
    not reconstructed from a lossy .split()), so citations can point back
    to an exact substring of the original field."""
    tokens = list(_TOKEN_RE.finditer(text))
    if not tokens:
        return []

    chunks: list[tuple[str, int, int]] = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        span_start = tokens[start_idx].start()
        span_end = tokens[end_idx - 1].end()
        chunks.append((text[span_start:span_end], span_start, span_end))
        if end_idx == len(tokens):
            break
        start_idx = end_idx - overlap
    return chunks


def _make_passage(
    row: dict[str, Any],
    source_name: str,
    language_code: str,
    field_source: str,
    passage_index: int,
    chunk_str: str,
    span_start: int,
    span_end: int,
) -> Passage:
    is_ar = _is_arabic(language_code)
    text = arabic_normalize.normalize(chunk_str) if is_ar else chunk_str
    return Passage(
        source_name=source_name,
        source_url=row.get("source_url", ""),
        source_record_id=row.get("source_record_id", ""),
        retrieval_date=date.today(),
        rights_statement=row.get("rights_statement"),
        license_note=row.get("license_note"),
        reuse_risk=ReuseRisk(row.get("reuse_risk", "unknown")),
        passage_id=f"{source_name}:{row.get('source_record_id', '')}:{language_code}:{field_source}:{passage_index}",
        language_code=language_code,
        passage_index=passage_index,
        field_source=field_source,
        text=text,
        text_raw=chunk_str,
        token_count=len(chunk_str.split()),
        char_offset_start=span_start,
        char_offset_end=span_end,
    )


def _passages_from_heritage_record(row: dict[str, Any], source_name: str) -> list[Passage]:
    language_code = row.get("language_code") or "und"
    passages: list[Passage] = []
    for field_source in ("title", "description"):
        text = row.get(field_source)
        if not text:
            continue
        for passage_index, (chunk_str, start, end) in enumerate(chunk_text(text)):
            passages.append(
                _make_passage(row, source_name, language_code, field_source, passage_index, chunk_str, start, end)
            )
    return passages


def _passages_from_vocab_term(row: dict[str, Any], source_name: str) -> list[Passage]:
    pref_label = row.get("pref_label") or {}
    scope_note = row.get("scope_note") or {}
    languages = set(pref_label) | set(scope_note)

    passages: list[Passage] = []
    for language_code in languages:
        text = scope_note.get(language_code) or pref_label.get(language_code)
        if not text:
            continue
        field_source = "scope_note" if scope_note.get(language_code) else "pref_label"
        for passage_index, (chunk_str, start, end) in enumerate(chunk_text(text)):
            passages.append(
                _make_passage(row, source_name, language_code, field_source, passage_index, chunk_str, start, end)
            )
    return passages


def passagize_source(source_name: str) -> Path:
    rows = read_jsonl(PROCESSED_DIR / source_name / "records.jsonl") + read_jsonl(
        PROCESSED_DIR / source_name / "needs_review.jsonl"
    )

    passages: list[Passage] = []
    for row in rows:
        if "pref_label" in row:  # VocabularyTerm-shaped
            passages.extend(_passages_from_vocab_term(row, source_name))
        else:  # HeritageRecord-shaped
            passages.extend(_passages_from_heritage_record(row, source_name))

    PASSAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PASSAGES_DIR / f"{source_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for passage in passages:
            f.write(passage.model_dump_json() + "\n")
    return out_path
