import json

import pytest

from lad.rag import lexical_enrichment
from lad.storage import writer


@pytest.fixture(autouse=True)
def _isolate_termbase(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(lexical_enrichment, "DATA_DIR", tmp_path)
    lexical_enrichment._termbase_lookup.cache_clear()
    yield
    lexical_enrichment._termbase_lookup.cache_clear()


def _write_termbase(tmp_path, entries):
    path = tmp_path / "termbase" / "interim_termbase.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_termbase_expansion_pulls_cross_lingual_labels(tmp_path):
    _write_termbase(
        tmp_path,
        [
            {
                "term_id": "1",
                "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"},
                "alt_labels": {"en": ["gilt"]},
            }
        ],
    )

    result = lexical_enrichment.expand_query("gilding", "en")

    assert "gilding" in result["en"]
    assert "gilt" in result["en"]
    assert "dorure" in result["fr"]
    assert any("تذهيب" in v or v == "تذهيب" for v in result.get("ar", []))


def test_expand_query_always_includes_the_original_term():
    result = lexical_enrichment.expand_query("some_unknown_term_xyz", "en")

    assert result["en"] == ["some_unknown_term_xyz"]


def test_expand_query_falls_back_to_bare_term_cross_lingually_when_uncovered(tmp_path):
    """A term with zero termbase/WordNet coverage must still produce a
    variant for EVERY target language (the bare term itself, relying on
    LaBSE's cross-lingual embedding alignment at retrieval time) -- not be
    silently absent from the returned dict, which would make retrieve()
    skip searching that language's index entirely. See the Phase 1.7 fix
    in expand_query's docstring."""
    result = lexical_enrichment.expand_query("some_unknown_term_xyz", "en")

    assert set(result.keys()) == {"en", "fr", "ar"}
    assert result["fr"] == ["some_unknown_term_xyz"]
    assert result["ar"] == ["some_unknown_term_xyz"]  # unaffected by Arabic normalization (no Arabic chars)


def test_expand_query_rejects_unsupported_language():
    with pytest.raises(ValueError):
        lexical_enrichment.expand_query("term", "de")


def test_arabic_query_gets_normalized_variant_added():
    result = lexical_enrichment.expand_query("مُتْحَف", "ar")

    assert "مُتْحَف" in result["ar"]
    assert "متحف" in result["ar"]  # normalized (dediacritized) form also present


def test_real_and_interim_termbase_entries_are_both_pooled(tmp_path):
    # Real termbase entry for "gilding" (has Arabic -- what the real
    # termbase adds over the AAT/UNESCO-derived interim substitute).
    real_path = tmp_path / "termbase" / "real_termbase.jsonl"
    real_path.parent.mkdir(parents=True, exist_ok=True)
    with real_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "term_id": "kalcium:1",
            "pref_label": {"en": "gilding", "ar": "تذهيب"},
            "alt_labels": {},
        }, ensure_ascii=False) + "\n")

    # Interim termbase entry for the same term, contributing a French label
    # the real termbase doesn't have for this concept.
    _write_termbase(tmp_path, [
        {
            "term_id": "interim:1",
            "pref_label": {"en": "gilding", "fr": "dorure"},
            "alt_labels": {},
        }
    ])

    result = lexical_enrichment.expand_query("gilding", "en")

    assert any("تذهيب" in v for v in result.get("ar", []))  # from the real termbase
    assert "dorure" in result["fr"]  # from the interim termbase
