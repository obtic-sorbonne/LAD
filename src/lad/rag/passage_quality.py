"""Passage-quality filtering for the retrieval index -- see
PROJECT_STATUS.md's "Phase 5": 75% of the pre-existing museum-subset
index (72,112/96,552 passages) is <=3-token bare labels, mostly Getty
AAT/UNESCO Thesaurus falling back to a bare pref_label because neither
has real scope_note coverage in English or French (passagize.py's
documented fallback rule: chunk scope_note, falling back to pref_label
when there's none -- for Getty AAT specifically, there's *never* one).
Demonstrated directly to actively harm retrieval, not just add noise:
top-5 results for the French query "dorure" were five duplicate,
unrelated Getty AAT entries reading "tuile" (tile), while genuinely
relevant passages ranked 8,632nd/12,257th out of 14,267.

Two independent filters, both applied at INDEX BUILD TIME, not at
passagize.py's chunking stage -- the underlying data/passages/*.jsonl
files stay complete and unedited; this only governs what gets embedded
into the retrieval index, so nothing downstream that reads raw passage
files (stats, HF export, etc.) loses data:

1. field_source exclusion: a VocabularyTerm-derived passage (Getty AAT,
   UNESCO Thesaurus) with field_source == "pref_label" exists ONLY
   because there was no real scope_note for that concept in that
   language -- a direct, principled signal ("no real definition
   existed"), not an inferred one from length alone.
2. Minimum token count: a general safety net (default 4 tokens) catching
   short/degenerate chunks regardless of source -- including
   HeritageRecord title fragments and even a scope_note that happens to
   be unusually short, where there's no field_source signal this clean.

Deduplication: many surviving passages are still exact duplicates across
concepts (e.g. 28 separate Getty AAT concepts share the literal French
label "tuile") -- collapsed to one representative per exact normalized
text before embedding, so a common short label doesn't occupy N nearly-
identical FAISS slots.
"""

from __future__ import annotations

from typing import Any

MIN_TOKENS = 4
LOW_QUALITY_FIELD_SOURCES = {"pref_label"}


def is_low_quality_passage(passage: dict[str, Any], min_tokens: int = MIN_TOKENS) -> bool:
    """True if this passage should be excluded from the retrieval index
    (still kept in the underlying passages/*.jsonl file -- this is an
    index-build-time filter, not a corpus edit)."""
    if passage.get("field_source") in LOW_QUALITY_FIELD_SOURCES:
        return True
    token_count = passage.get("token_count")
    if token_count is None:
        token_count = len((passage.get("text") or "").split())
    return token_count < min_tokens


def filter_and_dedupe(passages: list[dict[str, Any]], min_tokens: int = MIN_TOKENS) -> list[dict[str, Any]]:
    """Drops low-quality passages, then collapses exact-duplicate text
    (case-insensitive, whitespace-stripped) among what survives, keeping
    the first occurrence. Order-preserving for what's kept."""
    seen_text: set[str] = set()
    kept: list[dict[str, Any]] = []
    for passage in passages:
        if is_low_quality_passage(passage, min_tokens=min_tokens):
            continue
        key = (passage.get("text") or "").strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        kept.append(passage)
    return kept
