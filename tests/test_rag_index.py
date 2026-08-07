import hashlib
import json

import numpy as np
import pytest

from lad.rag import index as index_module
from lad.storage import writer


class _FakeEmbedder:
    """Deterministic, tiny (8-dim) fake embedder -- avoids loading real
    LaBSE (network + GPU + ~15s) for a unit test. Same text always maps to
    the same vector, and different texts map to different vectors, which
    is all these tests need."""

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


def _write_passages(tmp_path, source_name, rows):
    path = tmp_path / "passages" / f"{source_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _passage_row(passage_id, lang, text, source_name="test_source"):
    return {
        "source_name": source_name,
        "source_url": "https://example.org",
        "source_record_id": "rec1",
        "retrieval_date": "2026-07-21",
        "rights_statement": "CC0",
        "reuse_risk": "clear",
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


def test_resolve_lang_bucket_handles_all_source_encodings():
    assert index_module.resolve_lang_bucket("ar") == "ar"
    assert index_module.resolve_lang_bucket("ara") == "ar"
    assert index_module.resolve_lang_bucket("arabic") == "ar"
    assert index_module.resolve_lang_bucket("english") == "en"
    assert index_module.resolve_lang_bucket("fre") == "fr"
    assert index_module.resolve_lang_bucket("russian") is None
    assert index_module.resolve_lang_bucket(None) is None
    assert index_module.resolve_lang_bucket("eng,ara,rus") is None  # compound codes excluded


def test_build_index_partitions_by_language_and_excludes_out_of_scope(tmp_path):
    _write_passages(
        tmp_path,
        "test_source",
        [
            _passage_row("p1", "en", "museum collection"),
            _passage_row("p2", "fr", "collection du musée"),
            _passage_row("p3", "ar", "مجموعة المتحف"),
            _passage_row("p4", "russian", "музей"),  # out of scope, must be excluded
        ],
    )

    counts = index_module.build_index(sources=["test_source"], embedder=_FakeEmbedder(), model_name="fake-embedder")

    assert counts == {"ar": 1, "en": 1, "fr": 1}
    for lang in ("ar", "en", "fr"):
        index_path, meta_path = index_module.index_paths(lang, "fake-embedder")
        assert index_path.exists()
        assert meta_path.exists()


def test_build_index_filter_low_quality_excludes_bare_labels_by_default_off(tmp_path):
    rows = [
        _passage_row("p1", "en", "a genuinely useful real passage here"),
        {**_passage_row("p2", "en", "tuile"), "field_source": "pref_label"},  # bare AAT-style label
    ]
    _write_passages(tmp_path, "test_source", rows)

    # default (filter_low_quality=False) keeps both, unchanged prior behavior
    counts = index_module.build_index(sources=["test_source"], embedder=_FakeEmbedder(), model_name="fake-embedder")
    assert counts["en"] == 2


def test_build_index_filter_low_quality_true_drops_bare_labels(tmp_path):
    rows = [
        _passage_row("p1", "en", "a genuinely useful real passage here"),
        {**_passage_row("p2", "en", "tuile"), "field_source": "pref_label"},
    ]
    _write_passages(tmp_path, "test_source", rows)

    counts = index_module.build_index(
        sources=["test_source"], embedder=_FakeEmbedder(), model_name="fake-embedder-filtered",
        filter_low_quality=True,
    )
    assert counts["en"] == 1

    loaded = index_module.PassageIndex(model_name="fake-embedder-filtered")
    assert loaded._meta["en"][0]["passage_id"] == "p1"


def test_passage_index_search_returns_best_match(tmp_path):
    _write_passages(
        tmp_path,
        "test_source",
        [
            _passage_row("p1", "en", "museum collection"),
            _passage_row("p2", "en", "archaeological excavation"),
        ],
    )
    embedder = _FakeEmbedder()
    index_module.build_index(sources=["test_source"], embedder=embedder, model_name="fake-embedder")

    loaded = index_module.PassageIndex(model_name="fake-embedder")
    assert loaded.available_languages() == ["en"]

    query_vector = embedder.encode(["museum collection"])[0]
    hits = loaded.search("en", query_vector, top_k=2)

    assert hits[0][0]["passage_id"] == "p1"
    assert hits[0][1] > hits[1][1]  # exact-text match scores higher than the other passage
