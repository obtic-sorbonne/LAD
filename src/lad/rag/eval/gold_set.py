"""Builds a gold-standard evaluation set from the termbase (project plan
Part A2) -- auto-sampled, NOT terminologist-validated, so still not
directly comparable to the LAD paper's real 120-entry, human-validated
set. What changed from the original interim-only version:

- **Drawn from the real termbase first** (data/termbase/real_termbase.jsonl,
  parsed from the Kalcium export -- see pipeline/build_termbase_from_kalcium.py),
  falling back to the interim substitute only if the real termbase hasn't
  been built yet. This matters a lot here specifically: the interim
  substitute is 53,200/53,355 Getty-AAT-derived entries with zero Arabic,
  so an eval set sampled from it before now had almost no real AR<->EN/FR
  gold pairs to test against at all. The real termbase has 4,628 entries
  with all three languages populated.
- **Target size raised to 120** (was 50), matching the paper's real
  gold-set size -- the real termbase has ample entries to support this.
- **Source language now rotates round-robin across en/fr/ar** (was always
  "en" when available, i.e. every single ago row before now only tested
  EN-as-source retrieval -- AR-as-source and FR-as-source, the paper's
  other two eval conditions, were never exercised by the gold set at all,
  regardless of what run_eval.py did with the rows).
- **No forced 4/5-bucket subject-field classification.** The real
  termbase's own subject_field is populated on only ~740/4,628 fully-
  trilingual entries (the rest are untagged at the term level, not
  miscategorized) -- reusing build_termbase.py's keyword classifier here
  dumps ~93% of candidates into a "museography" catch-all and produces
  literally zero "provenance" matches, which is a false precision, not a
  useful stratification. Instead: stratify by the real, sparser
  subject_field where a concept has one (its 28 native categories, e.g.
  "Ceramics and Pottery", "Liturgical and Ritual Objects"), then backfill
  the remainder from the untagged pool by plain random sampling --
  reported honestly (subject_field is None on backfilled rows), not
  disguised as a 4th/5th bucket.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from lad.storage.writer import DATA_DIR, read_jsonl

GOLD_SET_PATH = DATA_DIR / "eval" / "gold_set.jsonl"
TARGET_SIZE = 120

LANGS = ("en", "fr", "ar")


def _load_candidates() -> tuple[list[dict], bool]:
    """Returns (candidates, from_real). Prefers the real termbase, and
    within it prefers concepts with all three languages populated (there's
    no shortage -- 4,628 qualify) so every gold row can test both
    non-source target languages, matching the paper's eval design (each
    source term evaluated against the two remaining languages).

    Paths are resolved from the module-level DATA_DIR at call time, not
    frozen into constants at import time -- tests monkeypatch
    gold_set.DATA_DIR to an isolated tmp_path, which an import-time
    constant would silently ignore."""
    real_path = DATA_DIR / "termbase" / "real_termbase.jsonl"
    if real_path.exists():
        entries = read_jsonl(real_path)
        trilingual = [e for e in entries if all(l in (e.get("pref_label") or {}) for l in LANGS)]
        if trilingual:
            return trilingual, True
        # Real termbase exists but (unexpectedly) has no fully-trilingual
        # rows -- fall back to its own >=2-language entries rather than
        # silently switching to the interim substitute.
        return [e for e in entries if len(e.get("pref_label") or {}) >= 2], True

    interim_path = DATA_DIR / "termbase" / "interim_termbase.jsonl"
    entries = read_jsonl(interim_path)
    return [e for e in entries if len(e.get("pref_label") or {}) >= 2], False


def _stratified_selection(candidates: list[dict], target_size: int, rng: random.Random) -> list[dict]:
    tagged = [e for e in candidates if e.get("subject_field")]
    untagged = [e for e in candidates if not e.get("subject_field")]

    by_field: dict[str, list[dict]] = defaultdict(list)
    for entry in tagged:
        by_field[entry["subject_field"]].append(entry)
    for field_entries in by_field.values():
        rng.shuffle(field_entries)

    per_field = max(1, min(target_size, len(tagged)) // max(len(by_field), 1)) if by_field else 0
    selected: list[dict] = []
    for field_entries in by_field.values():
        selected.extend(field_entries[:per_field])
    selected = selected[:target_size]

    if len(selected) < target_size:
        remaining = target_size - len(selected)
        shuffled_untagged = untagged[:]
        rng.shuffle(shuffled_untagged)
        selected.extend(shuffled_untagged[:remaining])

    return selected


def _pick_source_language(pref_label: dict[str, str], rotation_index: int) -> str:
    """Round-robins the source language across en/fr/ar by row position,
    falling back to whichever language the entry actually has if the
    rotated choice is missing (only reachable when candidates came from
    the interim-substitute fallback, which allows <3-language entries)."""
    for offset in range(len(LANGS)):
        lang = LANGS[(rotation_index + offset) % len(LANGS)]
        if lang in pref_label:
            return lang
    return next(iter(pref_label))


def build_gold_set(target_size: int = TARGET_SIZE, seed: int = 42) -> Path:
    candidates, from_real = _load_candidates()
    rng = random.Random(seed)
    selected = _stratified_selection(candidates, target_size, rng)

    gold_rows = []
    for i, entry in enumerate(selected):
        pref_label = entry.get("pref_label") or {}
        source_lang = _pick_source_language(pref_label, rotation_index=i)
        gold_rows.append(
            {
                "term_id": entry["term_id"],
                "source_term": pref_label[source_lang],
                "source_language": source_lang,
                "reference_equivalents": {lang: label for lang, label in pref_label.items() if lang != source_lang},
                "subject_field": entry.get("subject_field"),
                "termbase_source": "real" if from_real else "interim",
            }
        )

    GOLD_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_SET_PATH.open("w", encoding="utf-8") as f:
        for row in gold_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return GOLD_SET_PATH
