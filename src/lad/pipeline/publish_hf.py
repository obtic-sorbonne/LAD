"""Builds a local Hugging Face Hub-ready export of the collected dataset.

Does NOT push anything itself or touch HF credentials -- that's
scripts/06_push_to_hf.sh's job, which calls huggingface_hub directly after
this module has written plain Parquet files to disk. Kept separate so the
export can be inspected/tested without any network access or auth.

Scope decided with the user: include everything (harvested records +
interim termbase + passages), not just rights-clear content, because 84%
of harvested records have *unverified* rights (mostly UNESDOC/WDL, which
don't expose per-record rights in their APIs at all) rather than
*confirmed-restricted* rights -- excluding them would misrepresent
"unknown" as "bad". Every row keeps its reuse_risk field, and the
generated dataset card states the 84% figure plainly rather than leaving
it implicit.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lad.connectors import REGISTRY
from lad.pipeline.parquet_utils import json_encode_dict_fields
from lad.storage import writer

EXPORT_DIR = writer.DATA_DIR / "hf_export"

# Per-source license/rights summary for the dataset card -- kept here
# rather than re-derived from config/sources.yaml at export time, since
# the card needs prose explanation, not just the raw license_note string.
_SOURCE_LICENSE_NOTES: dict[str, str] = {
    "unesco_thesaurus": "CC BY-SA 3.0 IGO. 100% rights-clear.",
    "getty_aat": "ODC-BY. 100% rights-clear.",
    "europeana": (
        "Mixed -- per-item `rights_statement` governs each row; roughly half "
        "clear (CC0/CC BY/CC BY-SA/public domain/NoC), the rest restricted "
        "(In Copyright) or unknown. Filtered to French-language content only "
        "at harvest time (Europeana has no Arabic-tagged content at all)."
    ),
    "unesdoc": (
        "UNESCO Open Access policy applies in general, but the source API does "
        "not expose a per-record machine-readable rights statement, so every "
        "row here is reuse_risk=unknown, not confirmed-restricted."
    ),
    "world_digital_library": (
        "Varies by item; Library of Congress rights statements apply, but the "
        "source API does not expose a per-record machine-readable rights "
        "statement, so every row here is reuse_risk=unknown, not "
        "confirmed-restricted."
    ),
}


def _write_parquet(rows: list[dict], out_path: Path) -> int:
    if not rows:
        return 0
    table = pa.Table.from_pylist(json_encode_dict_fields(rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return len(rows)


def _build_records(counts: dict[str, int]) -> None:
    for source_name in REGISTRY:
        for needs_review, label in ((False, "records"), (True, "needs_review")):
            rows = writer.read_jsonl(writer.processed_path(source_name, needs_review=needs_review))
            n = _write_parquet(rows, EXPORT_DIR / "records" / source_name / f"{label}.parquet")
            if n:
                counts[f"records/{source_name}/{label}"] = n


def _build_termbase(counts: dict[str, int]) -> None:
    termbase_path = writer.DATA_DIR / "termbase" / "interim_termbase.jsonl"
    if not termbase_path.exists():
        return
    rows = writer.read_jsonl(termbase_path)
    n = _write_parquet(rows, EXPORT_DIR / "termbase" / "interim_termbase.parquet")
    if n:
        counts["termbase"] = n


def _build_passages(counts: dict[str, int]) -> None:
    passages_dir = writer.DATA_DIR / "passages"
    if not passages_dir.exists():
        return
    for path in sorted(passages_dir.glob("*.jsonl")):
        rows = writer.read_jsonl(path)
        n = _write_parquet(rows, EXPORT_DIR / "passages" / f"{path.stem}.parquet")
        if n:
            counts[f"passages/{path.stem}"] = n


def _dataset_card(counts: dict[str, int]) -> str:
    total_records = sum(v for k, v in counts.items() if k.startswith("records/"))
    clean_records = sum(v for k, v in counts.items() if k.startswith("records/") and k.endswith("/records"))
    flagged_records = sum(v for k, v in counts.items() if k.startswith("records/") and k.endswith("/needs_review"))
    clean_pct = clean_records / total_records * 100 if total_records else 0.0
    flagged_pct = flagged_records / total_records * 100 if total_records else 0.0

    lines = [
        "---",
        "license: other",
        "language:",
        "- ar",
        "- en",
        "- fr",
        "pretty_name: LAD Collected Dataset",
        "tags:",
        "- cultural-heritage",
        "- museum",
        "- multilingual",
        "- terminology",
        "---",
        "",
        "# LAD Collected Dataset",
        "",
        "Multilingual (Arabic/French/English) cultural-heritage text and "
        "terminology, collected from public sources as the data foundation "
        "for a term-centric multilingual RAG system for museum terminology "
        "discovery. See the source pipeline repository for full collection "
        "methodology.",
        "",
        "## Rights status -- read before use",
        "",
        f"Of {total_records:,} harvested records, **{clean_records:,} "
        f"({clean_pct:.1f}%) have a confirmed-open "
        f"rights statement** (`reuse_risk=clear` -- e.g. CC0, CC BY, CC BY-SA, "
        f"ODC-BY, public domain). The remaining **{flagged_records:,} "
        f"({flagged_pct:.1f}%) are `reuse_risk=unknown` "
        "or `restricted`**, included here for completeness and research use, "
        "not because their rights status is confirmed open. UNESDOC and World "
        "Digital Library in particular are ~100% `unknown` because neither "
        "source API exposes a per-record machine-readable rights statement at "
        "all -- this is a gap in the source data, not a claim that the "
        "content is restricted. **Every row carries its own `reuse_risk` "
        "field (`clear` / `restricted` / `unknown`) -- filter on it before "
        "any redistribution or production use.**",
        "",
        "## Contents",
        "",
        "- `records/<source>/records.parquet` -- rights-clear harvested records",
        "- `records/<source>/needs_review.parquet` -- rights-unverified/restricted harvested records",
        "- `termbase/interim_termbase.parquet` -- an **interim substitute** for "
        "the real Louvre Abu Dhabi Termbase (not available to this pipeline), "
        "derived by keyword-filtering UNESCO Thesaurus + Getty AAT concepts. "
        "Not a claim of precision -- see the source repository's README for "
        "known misclassifications.",
        "- `passages/<source>.parquet` -- chunked retrieval-unit passages "
        "(150-200 tokens, 20-token overlap) derived from the records above, "
        "the format a downstream RAG system would actually index.",
        "",
        "## Per-source license notes",
        "",
    ]
    for source_name, note in _SOURCE_LICENSE_NOTES.items():
        lines.append(f"- **{source_name}**: {note}")

    lines += [
        "",
        "## Row counts",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]:,}")

    return "\n".join(lines) + "\n"


def build_export() -> Path:
    counts: dict[str, int] = {}
    _build_records(counts)
    _build_termbase(counts)
    _build_passages(counts)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "README.md").write_text(_dataset_card(counts), encoding="utf-8")
    return EXPORT_DIR
