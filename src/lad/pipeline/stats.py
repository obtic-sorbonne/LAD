"""Generates a plain-text collection statistics report (stats.txt) from
whatever is currently in data/processed/ -- regenerated on demand rather
than hand-maintained, so it never goes stale like a committed snapshot would.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from pathlib import Path

from lad.config import load_sources
from lad.connectors import REGISTRY
from lad.storage import writer

# Language field is encoded differently per source: Europeana uses ISO
# 639-1 (ar/fr/en), UNESDOC uses ISO 639-2 (ara/fre/eng, sometimes comma-
# joined for multilingual documents), WDL uses full English words
# (arabic/french/english). Substring matching catches all three uniformly.
_LANG_MARKERS = {
    "ar": ("ar", "ara", "arabic"),
    "fr": ("fr", "fre", "fra", "french"),
    "en": ("en", "eng", "english"),
}


def _count_lang(records: list[dict], markers: tuple[str, ...]) -> int:
    return sum(1 for r in records if any(m in (r.get("language_code") or "").lower() for m in markers))


def generate_stats() -> str:
    out: list[str] = []

    def w(line: str = "") -> None:
        out.append(line)

    w("=" * 70)
    w("LAD HERITAGE DATA PIPELINE -- COLLECTION STATISTICS")
    w(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    w("=" * 70)

    grand_total = grand_clean = grand_flagged = 0

    for source_name in REGISTRY:
        records = writer.read_jsonl(writer.processed_path(source_name, needs_review=False))
        flagged = writer.read_jsonl(writer.processed_path(source_name, needs_review=True))
        total = len(records) + len(flagged)
        if total == 0:
            continue
        grand_total += total
        grand_clean += len(records)
        grand_flagged += len(flagged)
        all_records = records + flagged

        w("")
        w("-" * 70)
        w(source_name.upper().replace("_", " "))
        w("-" * 70)
        w(f"Total records: {total}  (clean: {len(records)}, flagged for review: {len(flagged)})")

        if all_records and "pref_label" in all_records[0]:
            has_ar = sum(1 for r in all_records if r.get("pref_label", {}).get("ar"))
            has_en = sum(1 for r in all_records if r.get("pref_label", {}).get("en"))
            has_fr = sum(1 for r in all_records if r.get("pref_label", {}).get("fr"))
            w(f"Arabic label coverage:  {has_ar}/{total}")
            w(f"English label coverage: {has_en}/{total}")
            w(f"French label coverage:  {has_fr}/{total}")
        else:
            for code, label in (("ar", "Arabic"), ("fr", "French"), ("en", "English")):
                n = _count_lang(all_records, _LANG_MARKERS[code])
                w(f"{label} (any mention): {n} ({n / total * 100:.1f}%)")

            reuse = collections.Counter(r.get("reuse_risk") for r in all_records)
            w("Reuse-risk: " + ", ".join(f"{k}={v}" for k, v in reuse.most_common()))

    w("")
    w("=" * 70)
    w("GRAND TOTALS")
    w("=" * 70)
    w(f"Total records collected: {grand_total}")
    if grand_total:
        w(f"  Clean (rights-clear): {grand_clean} ({grand_clean / grand_total * 100:.1f}%)")
        w(f"  Flagged for review:   {grand_flagged} ({grand_flagged / grand_total * 100:.1f}%)")

    termbase_path = writer.DATA_DIR / "termbase" / "interim_termbase.jsonl"
    if termbase_path.exists():
        n = sum(1 for _ in termbase_path.open(encoding="utf-8"))
        w("")
        w(f"Interim termbase (public-data substitute): {n} entries ({termbase_path})")

    real_termbase_path = writer.DATA_DIR / "termbase" / "real_termbase.jsonl"
    if real_termbase_path.exists():
        rows = writer.read_jsonl(real_termbase_path)
        has_ar = sum(1 for r in rows if r.get("pref_label", {}).get("ar"))
        has_en = sum(1 for r in rows if r.get("pref_label", {}).get("en"))
        has_fr = sum(1 for r in rows if r.get("pref_label", {}).get("fr"))
        w("")
        w(f"Real LAD Termbase (Kalcium export, institutional -- restricted): {len(rows)} entries ({real_termbase_path})")
        w(f"  Arabic label coverage:  {has_ar}/{len(rows)}")
        w(f"  English label coverage: {has_en}/{len(rows)}")
        w(f"  French label coverage:  {has_fr}/{len(rows)}")

    passages_dir = writer.DATA_DIR / "passages"
    if passages_dir.exists():
        w("")
        w("Passages (RAG retrieval units):")
        for path in sorted(passages_dir.glob("*.jsonl")):
            n = sum(1 for _ in path.open(encoding="utf-8"))
            w(f"  {path.stem}: {n}")

    w("")
    w("-" * 70)
    w("SOURCE STATUS (config/sources.yaml)")
    w("-" * 70)
    for name, config in load_sources().items():
        w(f"  {name}: enabled={config.get('enabled', True)} connector={'yes' if name in REGISTRY else 'no'}")

    w("=" * 70)
    return "\n".join(out) + "\n"


def write_stats(path: Path | None = None) -> Path:
    path = path or (writer.DATA_DIR.parent / "stats.txt")
    path.write_text(generate_stats(), encoding="utf-8")
    return path
