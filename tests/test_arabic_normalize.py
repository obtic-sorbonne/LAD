from lad.pipeline.arabic_normalize import dediacritize, normalize


def test_dediacritize_strips_diacritics():
    assert dediacritize("مُتْحَفٌ") == "متحف"


def test_normalize_folds_alef_variants():
    # Both spellings should collapse to the same normalized form.
    assert normalize("إسلامي") == normalize("اسلامي")


def test_normalize_is_idempotent():
    text = "مُتْحَفٌ إِسْلامِيٌّ"
    once = normalize(text)
    twice = normalize(once)
    assert once == twice
