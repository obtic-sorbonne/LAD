import json

import pytest

from lad.rag import lexical_index as lexical_index_module
from lad.rag.lexical_index import LexicalIndex, build_lexical_index, reciprocal_rank_fusion
from lad.storage import writer


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lexical_index_module, "DATA_DIR", tmp_path)


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


def test_build_and_search_finds_lexical_overlap(tmp_path):
    lex_dir = tmp_path / "lexical"
    # BM25's IDF is ~0 for a term appearing in exactly half a tiny corpus
    # (its break-even point) -- a realistic-sized corpus (here: "dorure"
    # in 1 of 4 docs) avoids that edge case rather than working around it.
    _write_passages(
        tmp_path,
        "test_source",
        [
            _passage_row("p1", "fr", "la dorure est une technique de decoration"),
            _passage_row("p2", "fr", "les tuiles de drainage couvrent le toit"),
            _passage_row("p3", "fr", "un vase en porcelaine ancienne"),
            _passage_row("p4", "fr", "une exposition sur l'histoire de l'art"),
        ],
    )

    build_lexical_index(sources=["test_source"], out_dir=lex_dir)
    index = LexicalIndex(base_dir=lex_dir)

    assert index.available_languages() == ["fr"]
    hits = index.search("fr", "dorure", top_k=5)

    assert len(hits) == 1
    assert hits[0][0]["passage_id"] == "p1"
    assert hits[0][1] > 0


def test_search_no_overlap_returns_nothing(tmp_path):
    lex_dir = tmp_path / "lexical"
    _write_passages(
        tmp_path,
        "test_source",
        [_passage_row("p1", "fr", "les tuiles de drainage couvrent le toit")],
    )
    build_lexical_index(sources=["test_source"], out_dir=lex_dir)
    index = LexicalIndex(base_dir=lex_dir)

    hits = index.search("fr", "dorure", top_k=5)

    assert hits == []  # zero lexical overlap -- exactly the "tuile"/"dorure" case


def test_search_unknown_language_returns_empty(tmp_path):
    lex_dir = tmp_path / "lexical"
    _write_passages(tmp_path, "test_source", [_passage_row("p1", "fr", "la dorure")])
    build_lexical_index(sources=["test_source"], out_dir=lex_dir)
    index = LexicalIndex(base_dir=lex_dir)

    assert index.search("ar", "dorure", top_k=5) == []


def test_search_empty_query_returns_empty(tmp_path):
    lex_dir = tmp_path / "lexical"
    _write_passages(tmp_path, "test_source", [_passage_row("p1", "fr", "la dorure")])
    build_lexical_index(sources=["test_source"], out_dir=lex_dir)
    index = LexicalIndex(base_dir=lex_dir)

    assert index.search("fr", "   ", top_k=5) == []


def test_build_lexical_index_excludes_out_of_scope_languages(tmp_path):
    lex_dir = tmp_path / "lexical"
    _write_passages(
        tmp_path,
        "test_source",
        [_passage_row("p1", "en", "museum"), _passage_row("p2", "russian", "музей")],
    )

    counts = build_lexical_index(sources=["test_source"], out_dir=lex_dir)

    assert counts == {"ar": 0, "en": 1, "fr": 0}


# ---- reciprocal_rank_fusion -------------------------------------------


def test_rrf_boosts_passages_ranked_well_in_both_lists():
    dense = ["p_dense_only", "p_both", "p_dense_low"]
    lexical = ["p_both", "p_lexical_only"]

    scores = reciprocal_rank_fusion(dense, lexical)

    assert scores["p_both"] > scores["p_dense_only"]
    assert scores["p_both"] > scores["p_lexical_only"]


def test_rrf_absent_from_one_list_still_contributes_from_the_other():
    dense = ["p1"]
    lexical: list[str] = []

    scores = reciprocal_rank_fusion(dense, lexical)

    assert scores["p1"] > 0


def test_rrf_the_tuile_dorure_case():
    """Simulates the actual failure mode this module fixes: a spurious
    dense-only top result ("tuile") vs. a passage found by both dense
    (lower rank) and lexical search -- fusion should prefer the genuinely
    relevant one."""
    dense_ranked = ["tuile_passage", "real_gilding_passage"]  # tuile wins on dense similarity alone
    lexical_ranked = ["real_gilding_passage"]  # only the real passage has lexical overlap

    scores = reciprocal_rank_fusion(dense_ranked, lexical_ranked)

    assert scores["real_gilding_passage"] > scores["tuile_passage"]
