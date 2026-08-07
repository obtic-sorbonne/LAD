"""Builds the interim LAD Termbase substitute (see README / plan Part A2).

We don't have the real ~1,000-entry Louvre Abu Dhabi Termbase, so this
derives a stand-in from data already collected -- no new fetching:
  - UNESCO Thesaurus concepts (data/processed/unesco_thesaurus/records.jsonl,
    already 100% trilingual) filtered to heritage/museum-relevant concepts
    by keyword match against a subject_field vocabulary.
  - Getty AAT concepts (data/processed/getty_aat/records.jsonl) taken as-is
    -- already scoped to museum-relevant facets by the connector itself,
    English/French only (AAT has no Arabic).

Every output row's source_name is "lad_termbase_interim" -- this IS the
substitute-vs-real tag (see schema.py VocabularyTerm docstring); the
original vocabulary + concept id is preserved in source_concept_ids for
traceability. Not deduplicated/cross-linked between UNESCO Thesaurus and
Getty AAT in this pass -- that needs embedding-based entity resolution,
which is RAG-system territory (Part B), not this stage.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from lad.schema import ReuseRisk, VocabularyTerm
from lad.storage.writer import DATA_DIR, PROCESSED_DIR, read_jsonl

TERMBASE_DIR = DATA_DIR / "termbase"
TERMBASE_PATH = TERMBASE_DIR / "interim_termbase.jsonl"

# Keyword -> subject_field, checked in this priority order (first match
# wins). Deliberately a simple, inspectable heuristic rather than a
# black-box classifier -- documented for a v1 substitute, not a claim of
# precision. Matches the LAD paper's 4 gold-set subfields.
_SUBJECT_FIELD_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "provenance",
        ["provenance", "acquisition", "repatriation", "looted", "illicit traffic", "ownership"],
    ),
    (
        "materials_and_techniques",
        ["material", "technique", "craft", "pigment", "textile", "ceramic", "metalwork",
         "restoration", "conservation treatment"],
    ),
    (
        "object_typology",
        ["artefact", "artifact", "monument", "sculpture", "manuscript", "antiquities",
         "archaeological object"],
    ),
    (
        "art_historical_period",
        ["period", "ancient", "medieval", "renaissance", "prehistoric", "antiquity"],
    ),
    (
        "museography",
        ["museum", "museolog", "exhibit", "curator", "collection management", "heritage site",
         "cultural heritage", "cultural property", "intangible heritage"],
    ),
]

# A UNESCO Thesaurus concept must match one of these to be included at all.
_INCLUDE_KEYWORDS = {kw for _, kws in _SUBJECT_FIELD_KEYWORDS for kw in kws} | {
    "heritage",
    "archaeolog",
    "art history",
    "curatorial",
    "world heritage",
}


def _classify_subject_field(text: str) -> str | None:
    lowered = text.lower()
    for subject_field, keywords in _SUBJECT_FIELD_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return subject_field
    return None


def _is_heritage_relevant(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _INCLUDE_KEYWORDS)


def _term_from_row(row: dict[str, Any], source_concept_ids: dict[str, str], subject_field: str | None) -> VocabularyTerm:
    return VocabularyTerm(
        source_name="lad_termbase_interim",
        source_url=row["source_url"],
        source_record_id=row["source_record_id"],
        retrieval_date=date.today(),
        rights_statement=row.get("rights_statement"),
        license_note=row.get("license_note"),
        reuse_risk=ReuseRisk(row.get("reuse_risk", "unknown")),
        term_id=row["term_id"],
        concept_scheme="LAD Termbase (interim)",
        pref_label=row.get("pref_label", {}),
        alt_labels=row.get("alt_labels", {}),
        scope_note=row.get("scope_note", {}),
        broader_ids=row.get("broader_ids", []),
        narrower_ids=row.get("narrower_ids", []),
        related_ids=row.get("related_ids", []),
        subject_field=subject_field,
        source_concept_ids=source_concept_ids,
    )


def _from_unesco_thesaurus() -> list[VocabularyTerm]:
    rows = read_jsonl(PROCESSED_DIR / "unesco_thesaurus" / "records.jsonl")
    entries = []
    for row in rows:
        en_label = row.get("pref_label", {}).get("en", "")
        en_note = row.get("scope_note", {}).get("en", "")
        text = f"{en_label} {en_note}"
        if not _is_heritage_relevant(text):
            continue
        subject_field = _classify_subject_field(text) or "museography"
        entries.append(
            _term_from_row(row, {"unesco_thesaurus": row["term_id"]}, subject_field)
        )
    return entries


def _from_getty_aat() -> list[VocabularyTerm]:
    rows = read_jsonl(PROCESSED_DIR / "getty_aat" / "records.jsonl")
    return [
        _term_from_row(row, {"getty_aat": row["term_id"]}, row.get("subject_field"))
        for row in rows
    ]


def build_termbase() -> Path:
    entries = _from_unesco_thesaurus() + _from_getty_aat()
    TERMBASE_DIR.mkdir(parents=True, exist_ok=True)
    with TERMBASE_PATH.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")
    return TERMBASE_PATH
