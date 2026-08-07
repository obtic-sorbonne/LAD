from lad.pipeline.build_termbase import _classify_subject_field, _is_heritage_relevant


def test_unrelated_unesco_concept_is_excluded():
    assert _is_heritage_relevant("Nuclear explosions") is False
    assert _is_heritage_relevant("Educational programmes") is False


def test_museum_concept_is_included():
    assert _is_heritage_relevant("Museums and museology") is True


def test_classify_subject_field_priority_order():
    # provenance keywords should win even when a later-bucket word is also present
    assert _classify_subject_field("Provenance of museum objects") == "provenance"
    assert _classify_subject_field("Ceramic restoration technique") == "materials_and_techniques"
    assert _classify_subject_field("Archaeological monument") == "object_typology"
    assert _classify_subject_field("Renaissance period") == "art_historical_period"
    assert _classify_subject_field("Museum curatorial practice") == "museography"


def test_classify_subject_field_no_match_returns_none():
    assert _classify_subject_field("Nuclear explosions") is None
