"""Evaluation metrics (Part C of the project plan): retrieval (P@k, R@k,
MRR, semantic_relevance_at_k) and synthesis (attestation accuracy,
equivalence correctness). Numbers from these are not directly comparable
to the LAD paper's reported figures -- both the corpus and the termbase
are interim substitutes here, not the real, larger, human-validated
resources the paper used. Treat as a smoke test / relative signal for
iteration, not a publishable result.

`precision_at_k`/`recall_at_k`/`mean_reciprocal_rank` are exact
case-insensitive substring matches against the reference label -- strict
and auditable, but brittle: a passage can genuinely attest a term without
containing its literal string (a different inflection, a synonym, an
institutionally-equivalent phrasing). That brittleness surfaced concretely
when a cross-encoder reranking A/B (see PROJECT_STATUS.md "Phase 1.6")
showed a P@5 regression under these metrics with no way to tell whether
retrieval had actually gotten worse, or had surfaced more relevant-but-
differently-worded passages that these metrics can't credit.
`semantic_relevance_at_k` is a companion metric for exactly that -- not a
replacement, since substring match still has real, auditable value the
embedding-based score doesn't."""

from __future__ import annotations

from typing import Protocol

from lad.rag.schema import RetrievalHit, TerminologyRecord


class _TextEncoder(Protocol):
    """Structural type for whatever encodes text to L2-normalized vectors --
    satisfied by rag.embeddings.Embedder and by test fakes alike (see
    test_retrieval.py's _FakeEmbedder), without importing the real Embedder
    here just for a type hint."""

    def encode(self, texts: list[str]): ...


def semantic_relevance_at_k(
    hits: list[RetrievalHit], reference_label: str, embedder: _TextEncoder, k: int = 5
) -> float:
    """Cosine similarity between the reference label and the single
    closest-matching passage among the top-k -- a graded, less brittle
    companion to precision_at_k's exact substring match (see module
    docstring for why this exists). Assumes L2-normalized embeddings (true
    of rag.embeddings.Embedder), so cosine similarity reduces to a plain
    dot product. Returns 0.0 for an empty hit list, matching the other
    retrieval metrics' convention."""
    top_k = hits[:k]
    if not top_k:
        return 0.0
    ref_vector = embedder.encode([reference_label])[0]
    hit_vectors = embedder.encode([h.text for h in top_k])
    similarities = hit_vectors @ ref_vector
    return float(similarities.max())


def precision_at_k(hits: list[RetrievalHit], reference_label: str, k: int = 5) -> float:
    """Fraction of the top-k retrieved passages whose text contains the
    reference label (substring match, case-insensitive)."""
    top_k = hits[:k]
    if not top_k:
        return 0.0
    hit_count = sum(1 for h in top_k if reference_label.lower() in h.text.lower())
    return hit_count / len(top_k)


def recall_at_k(hits: list[RetrievalHit], reference_label: str, k: int = 10) -> float:
    """1.0 if the reference label appears anywhere in the top-k, else 0.0."""
    top_k = hits[:k]
    return 1.0 if any(reference_label.lower() in h.text.lower() for h in top_k) else 0.0


def mean_reciprocal_rank(hits: list[RetrievalHit], reference_label: str) -> float:
    for rank, hit in enumerate(hits, start=1):
        if reference_label.lower() in hit.text.lower():
            return 1.0 / rank
    return 0.0


def equivalence_correctness(record: TerminologyRecord, reference_equivalents: dict[str, str]) -> float:
    """Fraction of reference-language equivalents where the synthesized
    output's proposed label matches (case-insensitive substring, either
    direction) the gold reference."""
    if not reference_equivalents:
        return 0.0
    correct = 0
    for lang, reference_label in reference_equivalents.items():
        candidates = record.equivalents.get(lang, [])
        if any(
            reference_label.lower() in c.label.lower() or c.label.lower() in reference_label.lower()
            for c in candidates
        ):
            correct += 1
    return correct / len(reference_equivalents)


def attestation_accuracy(record: TerminologyRecord, passage_text_by_id: dict[str, str]) -> float:
    """Fraction of proposed equivalents whose cited passage(s) actually
    contain the proposed label text -- catches synthesis hallucination
    (a citation to a passage that doesn't actually say what's claimed)."""
    all_equivalents = [e for equivs in record.equivalents.values() for e in equivs]
    if not all_equivalents:
        return 0.0
    attested = 0
    for equivalent in all_equivalents:
        cited_texts = [passage_text_by_id.get(pid, "") for pid in equivalent.passage_ids]
        if any(equivalent.label.lower() in text.lower() for text in cited_texts):
            attested += 1
    return attested / len(all_equivalents)
