"""Parses the real Louvre Abu Dhabi Termbase export (Kalcium TMS "Termbase
export" sheet) into VocabularyTerm rows -- see project plan Part A1.

This is the REAL counterpart to the paper's public-data substitute
(pipeline/build_termbase.py, source_name="lad_termbase_interim", derived
from UNESCO Thesaurus + Getty AAT -- 53,200 of whose 53,355 entries have
zero Arabic because they come from Getty AAT, which has no Arabic labels
at all). The real resource lives at data/Export from Kalcium.xlsx: 4,652
real concepts (one stray repeated header row filtered out, see below),
99.5% with all three languages populated (4,629/4,652) -- essentially
closing the Arabic-coverage gap the interim termbase couldn't.

Export layout (verified against the live file, not assumed from a spec):
two header rows (a language-group row, then a field-name row), then one
row per CONCEPT -- not one row per concept+language, not one row per
synonym. Per data row:
  columns 0-10:  concept-level fields (Concept ID, audit fields, Subject
                 field, ...) -- see the *_COL constants below.
  columns 11-31: Arabic block (21 columns)
  columns 32-52: English block (21 columns)
  columns 53-73: French block (21 columns)
Each language block repeats the same 21-field layout (see the offset
constants below) and holds exactly ONE term per language per concept --
verified live: no concept ID repeats across rows, so this termbase does
not carry multiple synonyms per language the way VocabularyTerm.alt_labels
could hold them. Cross-lingual synonym enrichment (gilt/gilded/water
gilding, etc.) comes from Getty AAT/WordNet in the RAG lexical-enrichment
step, not from this resource -- alt_labels is left empty here, not
force-populated from something that isn't in the data.

One stray repeated header row appears mid-sheet (a printed-header
artifact: its "Concept" cell is the literal string "Concept", not an int)
-- filtered by requiring an int concept id, not by row position, since the
row number isn't a stable property of the export.

Broader/narrower/related concepts are exported as <br/>-joined display
LABELS (e.g. "Plastic arts<br />Sculptors<br />Visual arts"), not concept
IDs -- Kalcium's export doesn't expose the linked concept's ID here.
Stored as label strings in VocabularyTerm.broader_ids/narrower_ids/
related_ids despite the field name (the schema types them as list[str],
not as an ID format) -- still directly useful as lexical-enrichment
signal, just not resolvable back to a concept row without a separate
lookup.

Subject field is kept as the export's own fine-grained category (29
distinct values, e.g. "Ceramics and Pottery", "Liturgical and Ritual
Objects") rather than collapsed into the LAD paper's 4 coarse gold-set
buckets (materials_and_techniques/museography/object_typology/provenance)
-- that collapse is lossy and debatable (the real termbase's categories
don't map onto "provenance" cleanly at all), so it's left to the eval-set
builder to do that mapping for reporting purposes, not baked into the
stored record.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import openpyxl

from lad.schema import ReuseRisk, VocabularyTerm
from lad.storage.writer import DATA_DIR

KALCIUM_EXPORT_PATH = DATA_DIR / "Export from Kalcium.xlsx"
REAL_TERMBASE_DIR = DATA_DIR / "termbase"
REAL_TERMBASE_PATH = REAL_TERMBASE_DIR / "real_termbase.jsonl"

SHEET_NAME = "Termbase export"
HEADER_ROWS = 2  # skip both header rows; data starts at row 3 (1-indexed)

LANGS = ("ar", "en", "fr")
_LANG_BLOCK_START = {"ar": 11, "en": 32, "fr": 53}

# Offsets *within* a language block (21 columns wide).
_STATUS = 0
_CONCEPT_NOTE = 1
_CONTEXT = 2
_SOURCE_CONTEXT = 3
_DEFINITION = 4
_SOURCE_DEFINITION = 5
_BROADER = 6
_NARROWER = 7
_RELATED = 8
_TERM_STAMP = 9
_TERM = 10
_TERM_CR_USER = 11
_TERM_CR_DATE = 12
_TERM_CH_USER = 13
_TERM_CH_DATE = 14
_TERM_NOTE = 15
_TERM_CONTEXT = 16
_PART_OF_SPEECH = 17
_USAGE_STATUS = 18
_TERM_SOURCE = 19
_GRAMMATICAL_GENDER = 20

_CONCEPT_ID_COL = 0
_CH_DATE_COL = 4
_SUBJECT_FIELD_COL = 10

_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _get(row: tuple[Any, ...], index: int) -> Any:
    return row[index] if index < len(row) else None


def _split_labels(cell: Any) -> list[str]:
    """Splits a <br/>-joined display-label cell into a clean list, dropping
    empty strings and stray non-breaking spaces (the export has several,
    e.g. a lone '\\xa0' entry with nothing else on its line)."""
    if not cell:
        return []
    parts = _BR_TAG_RE.split(str(cell))
    cleaned = [p.replace("\xa0", "").strip() for p in parts]
    return [p for p in cleaned if p]


def _parse_kalcium_date(value: Any) -> date | None:
    """Kalcium's audit-date format is '2025.07.24. 02:10:38'. Best-effort --
    used only to backdate retrieval_date to the concept's last-changed date
    instead of "today" (which would misleadingly imply the underlying
    terminology was just curated). Falls back to None (caller defaults to
    today) rather than raising -- a provenance nicety, not worth failing
    the whole parse over."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y.%m.%d. %H:%M:%S").date()
    except ValueError:
        return None


def _iter_data_rows(path: Path) -> list[tuple[Any, ...]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[SHEET_NAME]
    rows = []
    for row in ws.iter_rows(min_row=HEADER_ROWS + 1, values_only=True):
        concept_id = _get(row, _CONCEPT_ID_COL)
        if isinstance(concept_id, int):  # excludes the stray repeated header row
            rows.append(row)
    return rows


def _term_from_row(row: tuple[Any, ...]) -> VocabularyTerm:
    concept_id = _get(row, _CONCEPT_ID_COL)
    subject_field = _get(row, _SUBJECT_FIELD_COL)

    pref_label: dict[str, str] = {}
    scope_note: dict[str, str] = {}
    broader: list[str] = []
    narrower: list[str] = []
    related: list[str] = []

    for lang in LANGS:
        start = _LANG_BLOCK_START[lang]
        term = _get(row, start + _TERM)
        if term and str(term).strip():
            pref_label[lang] = str(term).strip()

        definition = _get(row, start + _DEFINITION)
        context = _get(row, start + _CONTEXT)
        note = definition or context  # definition preferred; context is a fallback, not appended
        if note and str(note).strip():
            scope_note[lang] = str(note).strip()

        broader.extend(_split_labels(_get(row, start + _BROADER)))
        narrower.extend(_split_labels(_get(row, start + _NARROWER)))
        related.extend(_split_labels(_get(row, start + _RELATED)))

    retrieval_date = _parse_kalcium_date(_get(row, _CH_DATE_COL)) or date.today()

    return VocabularyTerm(
        source_name="lad_termbase_real",
        source_url=f"internal://lad-termbase/kalcium/concept/{concept_id}",
        source_record_id=str(concept_id),
        retrieval_date=retrieval_date,
        rights_statement="Louvre Abu Dhabi Termbase (Kalcium export) -- internal institutional resource",
        reuse_risk=ReuseRisk.RESTRICTED,
        term_id=f"kalcium:{concept_id}",
        concept_scheme="LAD Termbase (real, Kalcium export)",
        pref_label=pref_label,
        alt_labels={},
        scope_note=scope_note,
        broader_ids=broader,
        narrower_ids=narrower,
        related_ids=related,
        subject_field=str(subject_field).strip() if subject_field else None,
        source_concept_ids={"kalcium": str(concept_id)},
    )


def build_real_termbase(path: Path = KALCIUM_EXPORT_PATH, out_path: Path = REAL_TERMBASE_PATH) -> Path:
    """Parses the Kalcium export and writes data/termbase/real_termbase.jsonl
    -- one VocabularyTerm per concept, source_name="lad_termbase_real" (the
    substitute-vs-real tag, mirroring pipeline/build_termbase.py's
    "lad_termbase_interim"). Concepts with no term in *any* language are
    skipped -- they carry no retrieval value and would only pollute
    downstream lexical-enrichment lookups with empty-label entries.
    `out_path` is overridable (tests use a tmp_path) without needing to
    monkeypatch module globals."""
    rows = _iter_data_rows(path)
    entries = [_term_from_row(row) for row in rows]
    entries = [e for e in entries if e.pref_label]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")
    return out_path
