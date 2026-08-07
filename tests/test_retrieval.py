import hashlib
import json

import numpy as np
import pytest

from lad.rag import index as index_module
from lad.rag import lexical_enrichment
from lad.rag import lexical_index as lexical_index_module
from lad.rag.lexical_index import LexicalIndex, build_lexical_index
from lad.rag.rerank import Reranker
from lad.rag.retrieval import retrieve
from lad.storage import writer


class _FakeEmbedder:
    model_name = "fake-embedder"

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()[:8]
            vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            vectors.append(vec)
        return np.stack(vectors) if vectors else np.zeros((0, 8), dtype=np.float32)


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(index_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(index_module, "EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr(lexical_enrichment, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lexical_index_module, "DATA_DIR", tmp_path)
    lexical_enrichment._termbase_lookup.cache_clear()
    yield
    lexical_enrichment._termbase_lookup.cache_clear()


def _passage_row(passage_id, lang, text, reuse_risk="clear", rights_statement="CC0"):
    return {
        "source_name": "test_source",
        "source_url": "https://example.org",
        "source_record_id": "rec1",
        "retrieval_date": "2026-07-21",
        "rights_statement": rights_statement,
        "reuse_risk": reuse_risk,
        "passage_id": passage_id,
        "language_code": lang,
        "passage_index": 0,
        "field_source": "description",
        "text": text,
        "text_raw": text,
        "token_count": len(text.split()),
        "char_offset_start": 0,
        "char_offset_end": len(text),
    }


def test_retrieve_returns_hits_per_language_with_rights_carried_through(tmp_path):
    passages_path = tmp_path / "passages" / "test_source.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_passage_row("p1", "en", "gilding technique")) + "\n")
        f.write(json.dumps(_passage_row("p2", "fr", "technique de dorure", reuse_risk="unknown", rights_statement=None)) + "\n")

    embedder = _FakeEmbedder()
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="fake-embedder")
    index = index_module.PassageIndex(model_name="fake-embedder")

    hits = retrieve("gilding", "en", index, embedder, top_k=5)

    assert "en" in hits
    assert hits["en"][0].passage_id == "p1"
    if "fr" in hits:
        fr_hit = next(h for h in hits["fr"] if h.passage_id == "p2")
        assert fr_hit.reuse_risk == "unknown"  # not silently upgraded to clear


def test_retrieve_dedupes_across_query_variants(tmp_path):
    """A passage matched by two different expanded-query variants should
    appear once in the results, keeping its best score."""
    passages_path = tmp_path / "passages" / "test_source.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_passage_row("p1", "en", "museum")) + "\n")

    embedder = _FakeEmbedder()
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="fake-embedder")
    index = index_module.PassageIndex(model_name="fake-embedder")

    hits = retrieve("museum", "en", index, embedder, top_k=5)

    passage_ids = [h.passage_id for h in hits["en"]]
    assert passage_ids.count("p1") == 1


class _WordOverlapCrossEncoder:
    def predict(self, pairs):
        return [
            float(len(set(q.lower().split()) & set(t.lower().split())))
            for q, t in pairs
        ]


def test_retrieve_with_reranker_can_change_final_ranking(tmp_path):
    """A passage that ranks low by embedding similarity but is a strong
    cross-encoder match should be able to outrank one that only wins on
    embedding similarity -- requires fetch_k to widen past top_k so the
    reranker actually has both candidates to choose from."""
    passages_path = tmp_path / "passages" / "test_source.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as f:
        # _FakeEmbedder scores are hash-based (arbitrary but deterministic);
        # what matters here is only that both passages are retrievable
        # within fetch_k=top_k*3, so the reranker's word-overlap score is
        # what decides final order, not embedding similarity.
        f.write(json.dumps(_passage_row("p_weak_overlap", "en", "a passage about pottery glazing")) + "\n")
        f.write(json.dumps(_passage_row("p_strong_overlap", "en", "gilding technique applied to wood")) + "\n")

    embedder = _FakeEmbedder()
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="fake-embedder")
    index = index_module.PassageIndex(model_name="fake-embedder")

    reranker = Reranker.__new__(Reranker)
    reranker._model = _WordOverlapCrossEncoder()

    hits = retrieve("gilding technique", "en", index, embedder, top_k=2, reranker=reranker)

    assert hits["en"][0].passage_id == "p_strong_overlap"


def test_retrieve_without_reranker_is_unchanged_default_behavior(tmp_path):
    passages_path = tmp_path / "passages" / "test_source.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_passage_row("p1", "en", "gilding technique")) + "\n")

    embedder = _FakeEmbedder()
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="fake-embedder")
    index = index_module.PassageIndex(model_name="fake-embedder")

    hits = retrieve("gilding", "en", index, embedder, top_k=5)  # reranker defaults to None
    hits_again = retrieve("gilding", "en", index, embedder, top_k=5, reranker=None)

    assert hits["en"][0].passage_id == "p1"
    # explicit reranker=None must be identical to the omitted-argument default
    assert [h.score for h in hits["en"]] == [h.score for h in hits_again["en"]]


class _ControllableEmbedder:
    """Maps specific known texts to fixed, controllable vectors, unlike
    _FakeEmbedder's hash-based (arbitrary) ones -- needed to reproduce the
    actual "tuile"/"dorure" failure mode: a passage with zero lexical
    overlap to the query scoring HIGHER on dense similarity than one that
    genuinely contains the query term."""

    model_name = "controllable-embedder"

    def __init__(self, vectors: dict[str, list[float]], default: list[float] | None = None):
        self._vectors = {k.lower(): v for k, v in vectors.items()}
        self._default = default or [0.0, 0.0]

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        vecs = []
        for text in texts:
            # expand_query() can pull in extra variants (e.g. a differently-
            # cased WordNet form) this test doesn't care about -- look up
            # case-insensitively, falling back to a neutral vector for any
            # variant not explicitly set up rather than requiring every
            # possible expansion to be enumerated.
            raw = np.array(self._vectors.get(text.lower(), self._default), dtype=np.float32)
            vecs.append(raw / (np.linalg.norm(raw) + 1e-9))
        return np.stack(vecs) if vecs else np.zeros((0, 2), dtype=np.float32)


def _build_lexical_and_dense(tmp_path, rows, vectors):
    passages_path = tmp_path / "passages" / "test_source.jsonl"
    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    embedder = _ControllableEmbedder(vectors)
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="controllable-embedder")
    index = index_module.PassageIndex(model_name="controllable-embedder")
    build_lexical_index(sources=["test_source"], out_dir=tmp_path / "embeddings" / "lexical")
    lexical_index = LexicalIndex(base_dir=tmp_path / "embeddings" / "lexical")
    return embedder, index, lexical_index


def test_fusion_fixes_the_tuile_dorure_failure_mode(tmp_path):
    """Reproduces the real failure mode found in PROJECT_STATUS.md Phase
    4/5/6: a short, lexically-unrelated passage ("tuile") scores higher
    dense similarity to the query ("dorure") than a passage that actually
    contains the query term, because embedding similarity is unreliable
    for short strings. Fusion with BM25 should fix this -- "tuile" has
    zero lexical overlap, "dorure technique" has exact overlap."""
    # BM25's IDF is ~0 for a term in exactly half a tiny corpus (its
    # break-even point, see test_lexical_index.py) -- filler passages keep
    # "dorure" at a realistic fraction of the corpus instead of hitting
    # that edge case.
    rows = [
        _passage_row("p_wrong", "fr", "tuile"),
        _passage_row("p_right", "fr", "dorure technique"),
        _passage_row("p_filler1", "fr", "un vase en porcelaine ancienne"),
        _passage_row("p_filler2", "fr", "une exposition sur l'histoire de l'art"),
    ]
    vectors = {
        "dorure": [1.0, 0.0],
        "tuile": [0.99, 0.01],  # spuriously close to the query despite being unrelated
        "dorure technique": [0.3, 0.95],  # genuinely relevant but embeds far from the bare query
        "un vase en porcelaine ancienne": [-0.9, -0.4],
        "une exposition sur l'histoire de l'art": [-0.4, -0.9],
    }
    embedder, index, lexical_index = _build_lexical_and_dense(tmp_path, rows, vectors)

    dense_only = retrieve("dorure", "fr", index, embedder, top_k=2)
    assert dense_only["fr"][0].passage_id == "p_wrong"  # confirms the failure mode is reproduced

    fused = retrieve("dorure", "fr", index, embedder, top_k=2, lexical_index=lexical_index)
    assert fused["fr"][0].passage_id == "p_right"  # fusion corrects it


def test_retrieve_without_lexical_index_is_unchanged_default_behavior(tmp_path):
    rows = [_passage_row("p1", "fr", "dorure technique")]
    vectors = {"dorure": [1.0, 0.0], "dorure technique": [0.9, 0.1]}
    embedder, index, lexical_index = _build_lexical_and_dense(tmp_path, rows, vectors)

    hits = retrieve("dorure", "fr", index, embedder, top_k=5)  # lexical_index defaults to None
    hits_again = retrieve("dorure", "fr", index, embedder, top_k=5, lexical_index=None)

    assert [h.score for h in hits["fr"]] == [h.score for h in hits_again["fr"]]


def test_fusion_with_zero_lexical_matches_falls_back_to_dense_ranking(tmp_path):
    rows = [_passage_row("p1", "fr", "un texte totalement different")]
    vectors = {"dorure": [1.0, 0.0], "un texte totalement different": [0.5, 0.5]}
    embedder, index, lexical_index = _build_lexical_and_dense(tmp_path, rows, vectors)

    hits = retrieve("dorure", "fr", index, embedder, top_k=5, lexical_index=lexical_index)

    assert hits["fr"][0].passage_id == "p1"  # still returned, dense-only ranking preserved
