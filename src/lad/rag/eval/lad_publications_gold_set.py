"""Builds a small, REAL gold-standard set from the LAD Publications corpus
(see pipeline/ingest_lad_publications.py) -- unlike the main gold set
(gold_set.py), which is auto-sampled *from* the termbase (so every entry
trivially has a translation by construction, see PROJECT_STATUS.md
"Phase 1.7"), this one only includes a real termbase entry if its label in
a given language is independently CONFIRMED attested (case-insensitive
substring match -- the same convention rag/eval/metrics.py's
precision_at_k already uses, not a new untested technique) somewhere in
the actual LAD Publications text for that language. This is real, if
small, evidence the concept is genuinely discussed trilingually (or
bilingually) in real institutional documentation, not a public-data
substitute -- exactly the "final test" scenario the project's biggest
open limitation (no real institutional documentation) has been missing.

Corpus imbalance shows up here too, honestly, not hidden: only the
Architecture Book has any Arabic content at all (39 passages total, vs.
360 English / 371 French across both publications), so full AR+EN+FR
triple-attestation is rare -- 18 terms out of 4,628 fully-trilingual real
termbase entries, checked directly. EN+FR attestation alone is far more
common (632 terms, source_language chosen from whichever language is
itself attested). Both counts are reported by build_lad_publications_gold_set,
not silently collapsed into one number.

Substring matching on short/common words (e.g. "canon", "ideal") can
produce a spurious match by coincidence -- MIN_LABEL_LENGTH filters the
shortest, highest-risk cases, but this is not a substitute for a human
spot-check on final output, which is explicitly recommended, not assumed
done here.
"""

from __future__ import annotations

import json
from pathlib import Path

from lad.storage.writer import DATA_DIR, read_jsonl

GOLD_SET_PATH = DATA_DIR / "eval" / "lad_publications_gold_set.jsonl"
REAL_TERMBASE_PATH = DATA_DIR / "termbase" / "real_termbase.jsonl"
PASSAGES_PATH = DATA_DIR / "passages" / "lad_publications.jsonl"

LANGS = ("ar", "en", "fr")
MIN_LABEL_LENGTH = 4
MAX_ATTESTING_PASSAGES = 3
SOURCE_LANG_PRIORITY = ("en", "fr", "ar")


def _load_passages_by_lang(passages_path: Path) -> dict[str, list[dict]]:
    by_lang: dict[str, list[dict]] = {lang: [] for lang in LANGS}
    for passage in read_jsonl(passages_path):
        lang = passage.get("language_code")
        if lang in by_lang:
            by_lang[lang].append(passage)
    return by_lang


def _attesting_passage_ids(label: str, lang: str, passages_by_lang: dict[str, list[dict]]) -> list[str]:
    label_lower = label.strip().lower()
    if len(label_lower) < MIN_LABEL_LENGTH:
        return []
    matches = [p["passage_id"] for p in passages_by_lang[lang] if label_lower in p["text"].lower()]
    return matches[:MAX_ATTESTING_PASSAGES]


def build_lad_publications_gold_set(
    termbase_path: Path = REAL_TERMBASE_PATH,
    passages_path: Path = PASSAGES_PATH,
    out_path: Path = GOLD_SET_PATH,
    min_attested_languages: int = 2,
) -> tuple[Path, dict[str, int]]:
    """Writes out_path and returns (path, stats) where stats reports how
    many entries achieved each attestation tier -- callers should surface
    both the triple-attested count and the total, not just the total,
    since they mean different things (see module docstring)."""
    termbase = read_jsonl(termbase_path)
    passages_by_lang = _load_passages_by_lang(passages_path)

    rows = []
    n_triple = 0
    for entry in termbase:
        pref_label = entry.get("pref_label") or {}
        if not all(lang in pref_label for lang in LANGS):
            continue

        attestations = {lang: _attesting_passage_ids(pref_label[lang], lang, passages_by_lang) for lang in LANGS}
        attested_langs = [lang for lang in LANGS if attestations[lang]]
        if len(attested_langs) >= 3:
            n_triple += 1
        if len(attested_langs) < min_attested_languages:
            continue

        source_lang = next((lang for lang in SOURCE_LANG_PRIORITY if lang in attested_langs), attested_langs[0])
        reference_equivalents = {
            lang: pref_label[lang] for lang in attested_langs if lang != source_lang
        }
        if not reference_equivalents:
            continue  # only the source language itself was attested -- nothing to evaluate against

        rows.append(
            {
                "term_id": entry["term_id"],
                "source_term": pref_label[source_lang],
                "source_language": source_lang,
                "reference_equivalents": reference_equivalents,
                "subject_field": entry.get("subject_field"),
                "attested_languages": attested_langs,
                "attesting_passage_ids": {lang: ids for lang, ids in attestations.items() if ids},
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {"n_total": len(rows), "n_triple_attested_ar_en_fr": n_triple}
    return out_path, stats
