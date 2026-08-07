"""Arabic text normalization, centralized so every caller (passagization,
WordNet lookup, later RAG query enrichment) normalizes the same way.

Uses camel_tools' lightweight normalization functions -- no pretrained
model download required, just diacritic/character-variant folding (verified
live: pip-installable, works standalone). Matches the LAD paper's stated
preprocessing (§3.1): diacritic stripping, alef normalization, light
stemming -- stemming is deliberately left out here, since it's lossy and
only wanted at retrieval-index time (passagize.py), not for general-purpose
normalization used for matching/lookup.
"""

from __future__ import annotations

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)


def dediacritize(text: str) -> str:
    """Strip diacritics only -- the minimal normalization needed to match
    plain-text Arabic input against diacritized lexical resources (e.g.
    Open Multilingual Wordnet's Arabic lemmas)."""
    return dediac_ar(text)


def normalize(text: str) -> str:
    """Full normalization: diacritic stripping + alef/alef-maksura/teh-marbuta
    variant folding. Use for indexing/matching where spelling variation
    (not just diacritics) needs to collapse to a canonical form."""
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text
