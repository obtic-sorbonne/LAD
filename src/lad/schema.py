"""Normalized record schemas shared by every connector and pipeline stage.

Three shapes, all built on ProvenanceFields so every row is independently
traceable back to its source and rights status:
- HeritageRecord: descriptive/bibliographic items (WDL, UNESDOC, Europeana)
- VocabularyTerm: controlled-vocabulary concepts (UNESCO Thesaurus, Getty AAT,
  and the interim termbase -- see pipeline/build_termbase.py)
- Passage: chunked retrieval units derived from HeritageRecord/VocabularyTerm
  text fields (see pipeline/passagize.py) -- what the RAG system actually
  indexes and retrieves, not the records themselves.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class ReuseRisk(str, Enum):
    CLEAR = "clear"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class HumanReviewStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ProvenanceFields(BaseModel):
    source_name: str
    source_url: str
    source_record_id: str
    retrieval_date: date
    rights_statement: str | None = None
    license_note: str | None = None
    reuse_risk: ReuseRisk = ReuseRisk.UNKNOWN


class HeritageRecord(ProvenanceFields):
    # language
    language_code: str | None = None
    original_language: str | None = None
    translation_language: str | None = None
    alignment_group_id: str | None = None

    # object / document
    title: str | None = None
    description: str | None = None
    object_type: str | None = None
    collection: str | None = None
    period: str | None = None
    date_text: str | None = None
    creator: str | None = None
    place: str | None = None
    material: str | None = None
    technique: str | None = None

    # terminology (populated later by Tier 2 enrichment; present now so the
    # schema doesn't change shape when that enrichment step is added)
    subject_terms: list[str] = Field(default_factory=list)
    controlled_vocabulary_source: str | None = None
    unesco_id: str | None = None
    getty_id: str | None = None
    wikidata_qid: str | None = None

    # heritage context
    heritage_category: str | None = None
    culture_region: str | None = None
    inscription_status: str | None = None
    institution: str | None = None
    provenance_note: str | None = None

    # quality control
    confidence_score: float | None = None
    human_review_status: HumanReviewStatus = HumanReviewStatus.NOT_REVIEWED
    notes: str | None = None


class VocabularyTerm(ProvenanceFields):
    term_id: str
    concept_scheme: str | None = None
    pref_label: dict[str, str] = Field(default_factory=dict)  # lang_code -> label
    alt_labels: dict[str, list[str]] = Field(default_factory=dict)
    scope_note: dict[str, str] = Field(default_factory=dict)
    broader_ids: list[str] = Field(default_factory=list)
    narrower_ids: list[str] = Field(default_factory=list)
    related_ids: list[str] = Field(default_factory=list)

    # Used by the interim LAD Termbase substitute (pipeline/build_termbase.py):
    # subject_field buckets entries into the LAD paper's 4 gold-set subfields
    # (materials_and_techniques / museography / object_typology /
    # art_historical_period / provenance); source_concept_ids traces a merged
    # termbase entry back to the concept(s) it was built from, e.g.
    # {"unesco_thesaurus": "http://vocabularies.unesco.org/thesaurus/concept460"}.
    # Left empty by connectors that don't need them (UNESCO Thesaurus, Getty AAT).
    subject_field: str | None = None
    source_concept_ids: dict[str, str] = Field(default_factory=dict)


class Passage(ProvenanceFields):
    """A retrieval-unit chunk derived from a HeritageRecord or VocabularyTerm.

    ProvenanceFields already carries source_name/source_url/source_record_id
    (the parent record this was chunked from) and rights_statement/reuse_risk,
    denormalized at chunk time -- not looked up via join -- so a passage built
    from a needs_review record carries that status forward through retrieval
    and synthesis rather than it getting silently dropped.
    """

    passage_id: str
    language_code: str
    passage_index: int
    field_source: str  # which field this was chunked from: title/description/scope_note
    text: str
    text_raw: str | None = None  # pre-normalization text, kept for citation/display
    token_count: int
    char_offset_start: int
    char_offset_end: int
