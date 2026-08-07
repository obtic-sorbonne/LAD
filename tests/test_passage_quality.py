from lad.rag.passage_quality import filter_and_dedupe, is_low_quality_passage


def _passage(text, field_source="description", token_count=None):
    return {
        "text": text,
        "field_source": field_source,
        "token_count": token_count if token_count is not None else len(text.split()),
    }


def test_pref_label_field_source_is_always_low_quality_even_if_long():
    # Rare in practice (pref_label fallback passages are almost always
    # short), but the field_source signal should win regardless of length
    # -- it means "no real definition existed", which length alone can't
    # fully capture.
    p = _passage("a surprisingly long bare label with many words in it", field_source="pref_label")
    assert is_low_quality_passage(p) is True


def test_scope_note_field_source_short_text_is_still_filtered_by_length():
    p = _passage("very short", field_source="scope_note")  # 2 tokens
    assert is_low_quality_passage(p) is True


def test_scope_note_field_source_long_enough_text_survives():
    p = _passage("a real definition with enough words", field_source="scope_note")
    assert is_low_quality_passage(p) is False


def test_min_tokens_is_configurable():
    p = _passage("three word text", field_source="description")  # 3 tokens
    assert is_low_quality_passage(p, min_tokens=4) is True
    assert is_low_quality_passage(p, min_tokens=3) is False


def test_missing_token_count_falls_back_to_counting_whitespace():
    p = {"text": "one two", "field_source": "description"}  # no token_count key at all
    assert is_low_quality_passage(p, min_tokens=3) is True


def test_filter_and_dedupe_drops_low_quality_and_duplicates():
    passages = [
        _passage("tuile", field_source="pref_label"),  # low quality: pref_label
        _passage("tuile", field_source="pref_label"),  # low quality + duplicate
        _passage("ok", field_source="description"),  # low quality: too short
        _passage("a genuinely useful real passage here", field_source="scope_note"),
        _passage("a genuinely useful real passage here", field_source="scope_note"),  # duplicate of survivor
        _passage("another distinct real passage of real length", field_source="description"),
    ]

    kept = filter_and_dedupe(passages)

    assert [p["text"] for p in kept] == [
        "a genuinely useful real passage here",
        "another distinct real passage of real length",
    ]


def test_filter_and_dedupe_dedup_is_case_and_whitespace_insensitive():
    passages = [
        _passage("A Genuinely Useful Passage Here", field_source="description"),
        _passage("  a genuinely useful passage here  ", field_source="description"),
    ]

    kept = filter_and_dedupe(passages)

    assert len(kept) == 1


def test_filter_and_dedupe_preserves_order_of_survivors():
    passages = [
        _passage("first real passage of real length", field_source="description"),
        _passage("x", field_source="pref_label"),
        _passage("second real passage of real length", field_source="description"),
    ]

    kept = filter_and_dedupe(passages)

    assert [p["text"] for p in kept] == [
        "first real passage of real length",
        "second real passage of real length",
    ]
