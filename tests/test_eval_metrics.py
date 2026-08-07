import numpy as np
import pytest

from lad.rag.eval.metrics import (
    attestation_accuracy,
    equivalence_correctness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    semantic_relevance_at_k,
)
from lad.rag.schema import AttestedEquivalent, RetrievalHit, TerminologyRecord


class _FixedVectorEmbedder:
    """Maps known strings to fixed, pre-normalized vectors so cosine
    similarity is exactly predictable in tests -- unlike a hash-based fake
    (used elsewhere for retrieval tests), which gives arbitrary but
    deterministic vectors, not ones with a controllable relationship to
    each other."""

    _RAW_VECTORS = {
        "dorure": [1.0, 0.0],
        "exact match text": [1.0, 0.0],  # same direction as "dorure" -> cosine sim = 1.0
        "unrelated text": [0.0, 1.0],  # orthogonal -> cosine sim = 0.0
        "partial match text": [0.6, 0.8],  # cosine sim to "dorure" = 0.6
    }

    def encode(self, texts):
        vectors = []
        for text in texts:
            raw = np.array(self._RAW_VECTORS[text])
            vectors.append(raw / np.linalg.norm(raw))
        return np.stack(vectors)


def _hit(passage_id, text, score=1.0):
    return RetrievalHit(
        passage_id=passage_id,
        language_code="fr",
        text=text,
        score=score,
        query_variant="x",
        source_name="test",
        source_record_id="rec1",
    )


def test_precision_at_k_counts_matching_passages():
    hits = [_hit("p1", "la dorure est une technique"), _hit("p2", "unrelated text"), _hit("p3", "autre chose")]

    assert precision_at_k(hits, "dorure", k=3) == 1 / 3


def test_precision_at_k_empty_hits_is_zero():
    assert precision_at_k([], "dorure", k=5) == 0.0


def test_recall_at_k_finds_match_anywhere_in_top_k():
    hits = [_hit("p1", "unrelated"), _hit("p2", "la dorure")]

    assert recall_at_k(hits, "dorure", k=2) == 1.0
    assert recall_at_k(hits, "nonexistent", k=2) == 0.0


def test_mean_reciprocal_rank():
    hits = [_hit("p1", "unrelated"), _hit("p2", "la dorure"), _hit("p3", "also dorure")]

    assert mean_reciprocal_rank(hits, "dorure") == 1 / 2
    assert mean_reciprocal_rank(hits, "nonexistent") == 0.0


def test_equivalence_correctness_matches_case_insensitive_substring():
    record = TerminologyRecord(
        source_term="gilding",
        source_language="en",
        equivalents={"fr": [AttestedEquivalent(label="Dorure", language_code="fr", attestation_count=1)]},
        llm_model="test",
        embedding_model="test",
    )

    assert equivalence_correctness(record, {"fr": "dorure"}) == 1.0
    assert equivalence_correctness(record, {"fr": "dorure", "ar": "تذهيب"}) == 0.5
    assert equivalence_correctness(record, {}) == 0.0


def test_semantic_relevance_at_k_returns_max_similarity_among_top_k():
    hits = [_hit("p1", "unrelated text"), _hit("p2", "exact match text")]
    embedder = _FixedVectorEmbedder()

    assert semantic_relevance_at_k(hits, "dorure", embedder, k=2) == pytest.approx(1.0)


def test_semantic_relevance_at_k_gives_partial_credit_unlike_substring_match():
    hits = [_hit("p1", "partial match text")]
    embedder = _FixedVectorEmbedder()

    # substring match would score this 0.0 (the string "dorure" never
    # appears literally) -- this is exactly the brittleness the metric
    # exists to give partial credit for.
    assert precision_at_k(hits, "dorure", k=1) == 0.0
    assert semantic_relevance_at_k(hits, "dorure", embedder, k=1) == pytest.approx(0.6)


def test_semantic_relevance_at_k_respects_k_cutoff():
    hits = [_hit("p1", "unrelated text"), _hit("p2", "exact match text")]
    embedder = _FixedVectorEmbedder()

    # only the first hit is within k=1, so the k=2 best match is excluded
    assert semantic_relevance_at_k(hits, "dorure", embedder, k=1) == pytest.approx(0.0)


def test_semantic_relevance_at_k_empty_hits_is_zero():
    embedder = _FixedVectorEmbedder()
    assert semantic_relevance_at_k([], "dorure", embedder, k=5) == 0.0


def test_attestation_accuracy_catches_hallucinated_citation():
    record = TerminologyRecord(
        source_term="gilding",
        source_language="en",
        equivalents={
            "fr": [
                AttestedEquivalent(label="dorure", language_code="fr", attestation_count=1, passage_ids=["p1"]),
                AttestedEquivalent(label="peinture", language_code="fr", attestation_count=1, passage_ids=["p2"]),
            ]
        },
        llm_model="test",
        embedding_model="test",
    )
    passage_text_by_id = {"p1": "la dorure est appliquée", "p2": "this passage never mentions the word at all"}

    # only 1 of 2 equivalents is actually attested in its cited passage
    assert attestation_accuracy(record, passage_text_by_id) == 0.5
