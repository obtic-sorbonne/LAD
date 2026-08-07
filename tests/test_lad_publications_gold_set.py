import json

from lad.rag.eval.lad_publications_gold_set import build_lad_publications_gold_set


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _passage(passage_id, lang, text):
    return {"passage_id": passage_id, "language_code": lang, "text": text}


def test_triple_attested_entry_included_with_all_reference_equivalents(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}},
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "a passage discussing gilding techniques"),
        _passage("p2", "fr", "un passage sur la dorure"),
        _passage("p3", "ar", "نص عن تقنية التذهيب"),
    ])

    path, stats = build_lad_publications_gold_set(termbase_path, passages_path, out_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert stats["n_triple_attested_ar_en_fr"] == 1
    assert set(rows[0]["attested_languages"]) == {"en", "fr", "ar"}
    assert rows[0]["source_language"] == "en"  # SOURCE_LANG_PRIORITY prefers en
    assert set(rows[0]["reference_equivalents"]) == {"fr", "ar"}


def test_partial_attestation_only_includes_attested_target_languages(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}},
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "a passage discussing gilding techniques"),
        _passage("p2", "fr", "un passage sur la dorure"),
        # no Arabic passage attests "تذهيب" at all
    ])

    path, stats = build_lad_publications_gold_set(termbase_path, passages_path, out_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert stats["n_triple_attested_ar_en_fr"] == 0
    assert rows[0]["attested_languages"] == ["en", "fr"]
    assert rows[0]["reference_equivalents"] == {"fr": "dorure"}  # ar excluded -- not attested


def test_single_language_attestation_is_excluded_by_default(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}},
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "a passage discussing gilding techniques"),
        # neither fr nor ar attested
    ])

    path, stats = build_lad_publications_gold_set(termbase_path, passages_path, out_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows == []
    assert stats["n_total"] == 0


def test_non_trilingual_termbase_entry_is_skipped_entirely(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "1", "pref_label": {"en": "gilding", "fr": "dorure"}},  # no Arabic label at all
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "a passage discussing gilding techniques"),
        _passage("p2", "fr", "un passage sur la dorure"),
    ])

    path, stats = build_lad_publications_gold_set(termbase_path, passages_path, out_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows == []


def test_short_labels_below_min_length_are_never_counted_as_attested(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "1", "pref_label": {"en": "Art", "fr": "Art", "ar": "فن"}},  # all shorter than MIN_LABEL_LENGTH
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "the history of art in this region"),
        _passage("p2", "fr", "l'histoire de l'art dans cette region"),
        _passage("p3", "ar", "تاريخ الفن في هذه المنطقة"),
    ])

    path, stats = build_lad_publications_gold_set(termbase_path, passages_path, out_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert rows == []  # short labels filtered even though they'd technically substring-match


def test_min_attested_languages_can_require_all_three(tmp_path):
    termbase_path = tmp_path / "termbase.jsonl"
    passages_path = tmp_path / "passages.jsonl"
    out_path = tmp_path / "gold.jsonl"

    _write_jsonl(termbase_path, [
        {"term_id": "triple", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}},
        {"term_id": "partial", "pref_label": {"en": "museum", "fr": "musee", "ar": "متحف"}},
    ])
    _write_jsonl(passages_path, [
        _passage("p1", "en", "a passage discussing gilding techniques"),
        _passage("p2", "fr", "un passage sur la dorure"),
        _passage("p3", "ar", "نص عن تقنية التذهيب"),
        _passage("p4", "en", "the museum collection"),
        _passage("p5", "fr", "la collection du musee"),
        # no Arabic attestation for "museum"
    ])

    path, stats = build_lad_publications_gold_set(
        termbase_path, passages_path, out_path, min_attested_languages=3
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [r["term_id"] for r in rows] == ["triple"]
