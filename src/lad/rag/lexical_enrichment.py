"""Term-centric query expansion -- static-only for Phase 1 (LAD paper's
C2, sub-components 1+2: termbase/WordNet lookup + Arabic morphological
normalization). tRAG-style generate-then-rank (component 3) and
cross-encoder reranking (component 4) are Phase 2/3, not part of this
first pass -- see the project plan.

Reuses Phase 0 modules directly rather than reimplementing lookup logic:
pipeline/wordnet_lookup.py for cross-lingual synonyms, pipeline/
arabic_normalize.py for Arabic variant forms, and both termbases for
museum-domain synonyms/labels:
- data/termbase/real_termbase.jsonl -- the REAL Louvre Abu Dhabi Termbase
  (parsed from the Kalcium export, see pipeline/build_termbase_from_kalcium.py):
  4,652 entries, 100% Arabic label coverage. This is what actually closes
  the Arabic query-expansion gap (see plan Part A1) -- the interim
  substitute below is 53,200/53,355 Getty-AAT-derived entries with zero
  Arabic labels at all.
- data/termbase/interim_termbase.jsonl -- the public-data substitute
  (UNESCO Thesaurus + Getty AAT), kept as a fallback for broader EN/FR
  coverage beyond the curated ~4,652 real concepts.
Both are merged into one lookup index; a term matching in both simply
contributes candidates from each (more expansion candidates, not a
conflict -- there's no field where one termbase's answer should override
the other's).
"""

from __future__ import annotations

from functools import lru_cache

from lad.pipeline.arabic_normalize import normalize as normalize_arabic
from lad.pipeline.wordnet_lookup import lookup_synonyms
from lad.storage.writer import DATA_DIR, read_jsonl

LANGS = ("ar", "en", "fr")


def _termbase_paths() -> tuple:
    # Computed from the module-level DATA_DIR *at call time*, not frozen at
    # import time -- tests monkeypatch lexical_enrichment.DATA_DIR to an
    # isolated tmp_path, which a module-level constant built at import
    # would silently ignore.
    return (
        DATA_DIR / "termbase" / "real_termbase.jsonl",
        DATA_DIR / "termbase" / "interim_termbase.jsonl",
    )


@lru_cache(maxsize=1)
def _termbase_lookup() -> dict[tuple[str, str], list[dict]]:
    """(lang_code, normalized_label) -> list of termbase entries (from
    either termbase) whose pref_label or an alt_label matches, in that
    language. Built once per process -- the combined termbase has tens of
    thousands of entries, not something to linear-scan per query."""
    index: dict[tuple[str, str], list[dict]] = {}

    for path in _termbase_paths():
        if not path.exists():
            continue
        for entry in read_jsonl(path):
            for lang in LANGS:
                labels = []
                pref = (entry.get("pref_label") or {}).get(lang)
                if pref:
                    labels.append(pref)
                labels.extend((entry.get("alt_labels") or {}).get(lang, []))
                for label in labels:
                    key = (lang, label.strip().lower())
                    index.setdefault(key, []).append(entry)
    return index


def _termbase_expansion(term: str, lang: str) -> dict[str, set[str]]:
    expansion: dict[str, set[str]] = {code: set() for code in LANGS}
    matches = _termbase_lookup().get((lang, term.strip().lower()), [])
    for entry in matches:
        pref_label = entry.get("pref_label") or {}
        alt_labels = entry.get("alt_labels") or {}
        for code in LANGS:
            if pref_label.get(code):
                expansion[code].add(pref_label[code])
            for alt in alt_labels.get(code, []):
                expansion[code].add(alt)
    return expansion


def _wordnet_expansion(term: str, lang: str) -> dict[str, set[str]]:
    synonyms = lookup_synonyms(term, lang)
    return {code: set(synonyms.get(code, [])) for code in LANGS}


def static_translation_coverage(term: str, lang: str) -> dict[str, set[str]]:
    """Returns {lang_code: variants} from the termbase/WordNet ONLY --
    excluding the bare-term cross-lingual fallback expand_query() always
    adds (see that function's docstring, "Phase 1.7"). Used by
    rag/generate_expand.py to decide which target languages have no real
    translation and are candidates for LLM-based generation. Kept out of
    expand_query() itself so that function stays simple and dependency-
    light (no FAISS/LLM access needed) for callers that don't need
    generation."""
    expansion: dict[str, set[str]] = {code: set() for code in LANGS}
    for source_expansion in (_termbase_expansion(term, lang), _wordnet_expansion(term, lang)):
        for code, variants in source_expansion.items():
            expansion[code] |= variants
    return expansion


def expand_query(term: str, lang: str) -> dict[str, list[str]]:
    """Returns {lang_code: [query variant strings]} for ar/en/fr -- always
    non-empty for all three, never just the source language.

    Fixed bug (see PROJECT_STATUS.md "Phase 1.7"): this used to add the
    bare term only to its *own* language's bucket, leaving a target
    language with no termbase/WordNet cross-lingual match completely
    ABSENT from the returned dict -- not weaker retrieval into that
    language, but retrieval.py's `retrieve()` never calling
    `index.search()` for it at all, since it only iterates the languages
    this function returns. That defeated the actual point of using LaBSE
    (rag/embeddings.py): it's a cross-lingual embedding model *precisely*
    so an English query can be encoded and searched directly against the
    Arabic FAISS index without needing a termbase/WordNet translation
    first (LAD paper §3.3: "enabling a French query to retrieve Arabic and
    English passages attesting the equivalent concept without explicit
    translation"). Now the bare term is added to every language's bucket
    unconditionally, termbase/WordNet-covered or not -- cheap (one more
    embedding + FAISS search per language) and strictly additive to the
    existing pooled-variants/keep-best-score dedup in retrieval.py, so it
    can only add recall, never remove a hit that static expansion would
    have found on its own.

    Static lookup only beyond that -- matches only what the termbase/
    WordNet already contain; a term neither resource covers still gets no
    *translated* cross-lingual variant, just the raw cross-lingual
    embedding match on the bare term. Generating better candidate
    variants for such terms (not just relying on the bare term) is what
    Phase 2's generate-then-rank targets, not this module."""
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")

    expansion: dict[str, set[str]] = {code: set() for code in LANGS}
    for code in LANGS:
        expansion[code].add(term)

    for source_expansion in (_termbase_expansion(term, lang), _wordnet_expansion(term, lang)):
        for code, variants in source_expansion.items():
            expansion[code] |= variants

    if lang == "ar":
        expansion["ar"].add(normalize_arabic(term))
    # Also normalize any Arabic variants pulled in from termbase/WordNet,
    # so retrieval matches passages that were themselves normalized at
    # index time (pipeline/passagize.py).
    expansion["ar"] = {normalize_arabic(v) for v in expansion["ar"]} | expansion["ar"]

    return {code: sorted(variants) for code, variants in expansion.items() if variants}
