"""Cross-lingual synonym lookup via NLTK WordNet + Open Multilingual Wordnet.

Not a harvested source -- WordNet is ~120K synsets, too large to export
wholesale into the termbase. Used on-demand by pipeline/build_termbase.py
and later RAG query enrichment. Requires the `wordnet` + `omw-2.0` NLTK
data packages (scripts/00_setup.sh downloads them).

Arabic OMW lemmas carry diacritics (verified live: the synset for "museum"
has the Arabic lemma "متْحف", not the plain-text "متحف" a user would type)
that won't match undiacritized input directly, so Arabic lookups go through
a dediacritized reverse index built once via camel_tools' dediac_ar, rather
than NLTK's own lang= parameter on synsets() (which does exact-string
matching and misses undiacritized queries -- confirmed live).
"""

from __future__ import annotations

from functools import lru_cache

from nltk.corpus import wordnet as wn

from lad.pipeline.arabic_normalize import dediacritize as dediac_ar

_OMW_LANG = {"en": "eng", "fr": "fra", "ar": "arb"}


@lru_cache(maxsize=1)
def _arabic_dediac_index() -> dict[str, list[str]]:
    """dediacritized Arabic lemma -> original diacritized lemma form(s)."""
    index: dict[str, list[str]] = {}
    for lemma in wn.all_lemma_names(lang="arb"):
        key = dediac_ar(lemma.replace("_", " "))
        index.setdefault(key, []).append(lemma)
    return index


def lookup_synonyms(term: str, lang: str) -> dict[str, list[str]]:
    """Return synonym labels for `term` (given in `lang`, one of en/fr/ar)
    across all three languages, keyed by language code. Empty dict if the
    term isn't found."""
    if lang == "ar":
        candidates = _arabic_dediac_index().get(dediac_ar(term), [])
        synsets = [s for candidate in candidates for s in wn.synsets(candidate, lang="arb")]
    else:
        omw_lang = _OMW_LANG.get(lang, "eng")
        synsets = wn.synsets(term) if omw_lang == "eng" else wn.synsets(term, lang=omw_lang)

    if not synsets:
        return {}

    result: dict[str, set[str]] = {code: set() for code in _OMW_LANG}
    for synset in synsets:
        for code, omw_code in _OMW_LANG.items():
            for lemma in synset.lemma_names(omw_code):
                label = lemma.replace("_", " ")
                if code == "ar":
                    label = dediac_ar(label)
                result[code].add(label)

    return {code: sorted(names) for code, names in result.items() if names}
