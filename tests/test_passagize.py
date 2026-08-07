from lad.pipeline.passagize import (
    _is_arabic,
    _passages_from_heritage_record,
    _passages_from_vocab_term,
    chunk_text,
)


def test_chunk_text_short_text_is_one_chunk():
    text = "a short passage of text"
    chunks = chunk_text(text, chunk_size=180, overlap=20)

    assert len(chunks) == 1
    chunk_str, start, end = chunks[0]
    assert chunk_str == text
    assert text[start:end] == text


def test_chunk_text_long_text_overlaps():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=180, overlap=20)

    assert len(chunks) > 1
    # consecutive chunks share the last `overlap` tokens of the previous one
    first_tokens = chunks[0][0].split()
    second_tokens = chunks[1][0].split()
    assert first_tokens[-20:] == second_tokens[:20]
    # offsets are real spans into the original text
    for chunk_str, start, end in chunks:
        assert text[start:end] == chunk_str


def test_chunk_text_empty_returns_nothing():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_is_arabic_recognizes_known_codes_only():
    assert _is_arabic("ar") is True
    assert _is_arabic("ara") is True
    assert _is_arabic("arabic") is True
    assert _is_arabic("ARABIC") is True
    assert _is_arabic("eng,ara,rus") is False  # compound code, deliberately not treated as Arabic
    assert _is_arabic("fr") is False
    assert _is_arabic(None) is False


def test_heritage_record_passages_carry_rights_and_language():
    row = {
        "source_url": "https://example.org/item/1",
        "source_record_id": "item-1",
        "rights_statement": None,
        "reuse_risk": "unknown",
        "language_code": "ar",
        "title": "متحف الفن الإسلامي",
        "description": None,
    }

    passages = _passages_from_heritage_record(row, "test_source")

    assert len(passages) == 1
    p = passages[0]
    assert p.field_source == "title"
    assert p.language_code == "ar"
    assert p.reuse_risk == "unknown"
    assert p.text != p.text_raw  # Arabic normalization applied
    assert p.text_raw == row["title"]


def test_vocab_term_passages_one_per_language():
    row = {
        "source_url": "https://vocab.example.org/concept1",
        "source_record_id": "concept1",
        "rights_statement": "CC BY-SA 3.0 IGO",
        "reuse_risk": "clear",
        "pref_label": {"en": "Communication", "fr": "Communication", "ar": "اتصالات"},
        "scope_note": {"en": "An interactive social process."},
    }

    passages = _passages_from_vocab_term(row, "test_vocab")

    by_lang = {p.language_code: p for p in passages}
    assert set(by_lang) == {"en", "fr", "ar"}
    assert by_lang["en"].field_source == "scope_note"
    assert by_lang["en"].text_raw == "An interactive social process."
    assert by_lang["fr"].field_source == "pref_label"  # no FR scope_note, falls back
    # "اتصالات" has no diacritics/alef-variants to strip, so normalization is
    # a no-op here -- the Arabic-normalization-changes-text case is already
    # covered by test_heritage_record_passages_carry_rights_and_language.
    assert by_lang["ar"].text == by_lang["ar"].text_raw
