from lad.rag.rerank import Reranker
from lad.rag.schema import RetrievalHit


class _FakeCrossEncoder:
    """Deterministic stand-in for sentence_transformers.CrossEncoder --
    scores a pair higher the more word-overlap the query and text share,
    so tests don't need network/GPU access to a real model."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def _hit(passage_id, query_variant, text, score=0.5):
    return RetrievalHit(
        passage_id=passage_id,
        language_code="en",
        text=text,
        score=score,
        query_variant=query_variant,
        source_name="test_source",
        source_record_id="rec1",
    )


def test_rerank_reorders_by_cross_encoder_score_not_original_score():
    reranker = Reranker.__new__(Reranker)  # skip __init__'s real model load
    reranker._model = _FakeCrossEncoder()

    hits = [
        _hit("p1", "gilding technique", "a passage about pottery glazing", score=0.9),  # high embed score, low overlap
        _hit("p2", "gilding technique", "gilding technique applied to wood", score=0.1),  # low embed score, high overlap
    ]

    reranked = reranker.rerank(hits)

    assert [h.passage_id for h in reranked] == ["p2", "p1"]  # cross-encoder overrides embedding order


def test_rerank_scores_each_hit_against_its_own_query_variant():
    reranker = Reranker.__new__(Reranker)
    reranker._model = _FakeCrossEncoder()

    hits = [
        _hit("p1", "museum", "a museum in the city"),  # variant "museum" overlaps text
        _hit("p2", "متحف", "a museum in the city"),  # variant is Arabic, no overlap with English text
    ]

    reranked = reranker.rerank(hits)

    assert reranked[0].passage_id == "p1"
    assert reranked[0].score > reranked[1].score


def test_rerank_replaces_score_field_with_cross_encoder_score():
    reranker = Reranker.__new__(Reranker)
    reranker._model = _FakeCrossEncoder()

    hits = [_hit("p1", "gilding", "gilding technique", score=0.9999)]
    reranked = reranker.rerank(hits)

    assert reranked[0].score == 1.0  # "gilding" overlaps "gilding" -> 1 shared word, not the original 0.9999


def test_rerank_empty_hits_returns_empty():
    reranker = Reranker.__new__(Reranker)
    reranker._model = _FakeCrossEncoder()

    assert reranker.rerank([]) == []
