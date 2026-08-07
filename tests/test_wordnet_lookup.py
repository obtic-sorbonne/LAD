from lad.pipeline.wordnet_lookup import lookup_synonyms


def test_english_query_returns_cross_lingual_synonyms():
    result = lookup_synonyms("museum", "en")

    assert "museum" in result.get("en", [])
    assert "musée" in result.get("fr", [])
    assert "متحف" in result.get("ar", [])


def test_french_query_returns_same_concept():
    result = lookup_synonyms("musée", "fr")

    assert "museum" in result.get("en", [])


def test_arabic_query_without_diacritics_still_matches():
    """The OMW Arabic lemma is diacritized ('متْحف'); a plain-text query
    ('متحف') must still resolve via the dediacritized reverse index."""
    result = lookup_synonyms("متحف", "ar")

    assert "museum" in result.get("en", [])
    assert "musée" in result.get("fr", [])


def test_unknown_term_returns_empty_dict():
    assert lookup_synonyms("xyzzynotarealword", "en") == {}
