import json

import numpy as np
import pytest

from lad.rag import lexical_enrichment
from lad.rag.generate_expand import (
    _parse_candidates,
    augment_with_generated_variants,
    generate_candidates,
    verify_candidates,
)
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


class _StubGenerator:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.prompts_sent = []

    def generate(self, prompt: str) -> str:
        self.prompts_sent.append(prompt)
        return self.response_text


class _RaisingGenerator:
    """Used to prove a language was never generated for -- calling
    .generate() at all is a test failure."""

    def generate(self, prompt: str) -> str:
        raise AssertionError("generator should not have been called for a statically-covered language")


class _FakeEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 1), dtype=np.float32)


class _SequentialFakeIndex:
    """Returns one preset (meta, score) response per .search() call, in
    call order -- lets tests control which candidates pass
    verify_candidates' threshold without real embedding math."""

    def __init__(self, languages, responses):
        self._languages = languages
        self._responses = list(responses)
        self._call_index = 0

    def available_languages(self):
        return self._languages

    def search(self, lang, query_vector, top_k=10):
        response = self._responses[self._call_index]
        self._call_index += 1
        return [response] if response is not None else []


# ---- _parse_candidates ----------------------------------------------------


def test_parse_candidates_clean_json_array():
    assert _parse_candidates('["dorure", "doré"]') == ["dorure", "doré"]


def test_parse_candidates_strips_markdown_code_fence():
    raw = "```json\n" + json.dumps(["dorure"]) + "\n```"
    assert _parse_candidates(raw) == ["dorure"]


def test_parse_candidates_malformed_json_returns_empty_list():
    assert _parse_candidates("not json at all") == []


def test_parse_candidates_non_list_json_returns_empty_list():
    assert _parse_candidates('{"not": "a list"}') == []


def test_parse_candidates_drops_blank_entries():
    assert _parse_candidates('["dorure", "", "  "]') == ["dorure"]


# ---- generate_candidates ---------------------------------------------------


def test_generate_candidates_sends_expected_prompt_and_parses_response():
    generator = _StubGenerator(json.dumps(["تذهيب", "تمويه بالذهب"]))

    result = generate_candidates("gilding", "en", "ar", generator, n=5)

    assert result == ["تذهيب", "تمويه بالذهب"]
    sent_prompt = generator.prompts_sent[0]
    assert "gilding" in sent_prompt
    assert "English" in sent_prompt
    assert "Arabic" in sent_prompt


# ---- verify_candidates ------------------------------------------------------


def test_verify_candidates_keeps_only_those_above_threshold():
    index = _SequentialFakeIndex(
        languages=["ar"],
        responses=[({}, 0.8), ({}, 0.2)],  # first candidate passes, second doesn't
    )
    embedder = _FakeEmbedder()

    result = verify_candidates(["تذهيب", "بعيد جدا"], "ar", index, embedder, min_similarity=0.5)

    assert result == ["تذهيب"]


def test_verify_candidates_empty_list_short_circuits():
    index = _SequentialFakeIndex(languages=["ar"], responses=[])
    assert verify_candidates([], "ar", index, _FakeEmbedder()) == []


def test_verify_candidates_language_not_in_index_returns_empty():
    index = _SequentialFakeIndex(languages=["fr"], responses=[({}, 0.9)])
    assert verify_candidates(["تذهيب"], "ar", index, _FakeEmbedder()) == []


def test_verify_candidates_no_hits_for_a_candidate_excludes_it():
    index = _SequentialFakeIndex(languages=["ar"], responses=[None])
    assert verify_candidates(["تذهيب"], "ar", index, _FakeEmbedder()) == []


# ---- augment_with_generated_variants ---------------------------------------


def test_augment_skips_generation_for_statically_covered_language(tmp_path):
    # Only French is statically covered here -- Arabic is genuinely
    # uncovered and SHOULD trigger generation, so a plain call-recording
    # generator (not a raise-on-any-call one) is what actually isolates
    # "French is skipped" without conflating it with "Arabic is attempted".
    _write_termbase(tmp_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure"}, "alt_labels": {}},
    ])
    query_variants = {"en": ["gilding"], "fr": ["gilding", "dorure"], "ar": ["gilding"]}
    index = _SequentialFakeIndex(languages=["ar"], responses=[None])
    generator = _StubGenerator("[]")

    result = augment_with_generated_variants(
        query_variants, "gilding", "en", generator, index, _FakeEmbedder()
    )

    assert len(generator.prompts_sent) == 1  # only one generation call was made...
    assert "Arabic" in generator.prompts_sent[0]  # ...and it was for Arabic, not French
    assert result["fr"] == ["dorure", "gilding"]  # untouched, no generation attempted


def test_augment_adds_verified_candidates_for_uncovered_language(tmp_path):
    query_variants = {"en": ["deaccessioning"], "fr": ["deaccessioning"], "ar": ["deaccessioning"]}
    generator = _StubGenerator(json.dumps(["تخريج", "شطب"]))
    index = _SequentialFakeIndex(languages=["ar"], responses=[({}, 0.8), ({}, 0.8)])

    result = augment_with_generated_variants(
        query_variants, "deaccessioning", "en", generator, index, _FakeEmbedder()
    )

    assert "تخريج" in result["ar"]
    assert "شطب" in result["ar"]
    assert "deaccessioning" in result["ar"]  # bare-term fallback preserved, not replaced


def test_augment_never_worse_off_when_nothing_passes_verification(tmp_path):
    query_variants = {"en": ["deaccessioning"], "fr": ["deaccessioning"], "ar": ["deaccessioning"]}
    generator = _StubGenerator(json.dumps(["a bad guess"]))
    index = _SequentialFakeIndex(languages=["ar"], responses=[None])  # nothing verifies

    result = augment_with_generated_variants(
        query_variants, "deaccessioning", "en", generator, index, _FakeEmbedder()
    )

    assert result["ar"] == ["deaccessioning"]  # exactly what expand_query already had


def test_augment_never_generates_for_the_source_language_itself(tmp_path):
    # fr and ar both statically covered too, so _RaisingGenerator would
    # catch a call for ANY of the three languages -- isolating specifically
    # whether "en" (the source language) gets skipped by the source-language
    # check, not conflated with "uncovered languages get generated for".
    _write_termbase(tmp_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}, "alt_labels": {}},
    ])
    query_variants = {"en": ["gilding"], "fr": ["gilding", "dorure"], "ar": ["gilding", "تذهيب"]}
    index = _SequentialFakeIndex(languages=[], responses=[])

    result = augment_with_generated_variants(
        query_variants, "gilding", "en", _RaisingGenerator(), index, _FakeEmbedder()
    )

    assert result["en"] == ["gilding"]  # unchanged -- generator never called at all
