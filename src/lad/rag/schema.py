"""RAG-specific output schemas. Kept separate from the top-level
`lad.schema` (HeritageRecord/VocabularyTerm/Passage, all ProvenanceFields-
based -- one per harvested/derived row) since these describe *synthesized*
output derived from multiple passages, not a single traceable source row.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """One passage retrieved for one query variant, with its similarity score."""

    passage_id: str
    language_code: str
    text: str
    score: float
    query_variant: str
    source_name: str
    source_record_id: str
    rights_statement: str | None = None
    reuse_risk: str = "unknown"


class AttestedEquivalent(BaseModel):
    """One candidate term equivalent, grounded in the passages it was
    actually observed in -- never a bare LLM claim without a citation."""

    label: str
    language_code: str
    attestation_count: int
    passage_ids: list[str] = Field(default_factory=list)


class TerminologyRecord(BaseModel):
    """The structured output of one term query -- matches the LAD paper's
    Figure 2 sample record: source term, per-language equivalents with
    attesting passages, a usage note, and generation metadata."""

    source_term: str
    source_language: str
    equivalents: dict[str, list[AttestedEquivalent]] = Field(default_factory=dict)
    usage_note: str | None = None
    rights_caveat: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    llm_model: str
    embedding_model: str
