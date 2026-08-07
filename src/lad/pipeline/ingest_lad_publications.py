"""Ingests the LAD Publications PDFs (real Louvre Abu Dhabi publications,
provided directly -- not fetched by any connector, must already be
present at `source_dir`) into HeritageRecord rows, one per page, so the
existing passagize_source()/build-index pipeline can process them
completely unchanged: passagize.py dispatches on `"pref_label" in row` to
tell VocabularyTerm-shaped rows from HeritageRecord-shaped ones, and a
HeritageRecord-shaped row here needs no new chunking logic at all.

This is the first REAL institutional documentation this project has
indexed -- everything else (Getty AAT, UNESCO Thesaurus, Europeana,
UNESDOC, World Digital Library) is public-data substitute, explicitly
labeled as such throughout this project. The corpus-content gap (not the
termbase, not retrieval quality) has been the single most consistently
identified bottleneck this project has measured -- see PROJECT_STATUS.md.

Direct text extraction (pypdf), not OCR -- verified live before writing
this: all 5 known PDFs have real, directly extractable text layers (not
scanned images), so the OCR/correction pipeline in the sibling
`extraction` project (old LAD) isn't needed for this specific corpus.
Kept as its own explicit ingestion path rather than routed through that
pipeline -- routing already-clean-text PDFs through an OCR pipeline built
for noisy scans would be pure overhead, not a real requirement.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pypdf

from lad.schema import HeritageRecord, ReuseRisk
from lad.storage.writer import PROCESSED_DIR

SOURCE_NAME = "lad_publications"
DEFAULT_SOURCE_DIR = Path("data/raw_pdfs/lad_publications")

# Explicit, inspectable filename -> language mapping -- not pattern-guessed.
# There are only 5 known files; guessing wrong here would silently
# mislabel real institutional content, so every file this module knows
# about is named explicitly. A file present in source_dir but absent from
# this mapping is skipped, not guessed at (see ingest_lad_publications).
_FILENAME_LANGUAGE = {
    "LAD_Architecture_Book_AR_4WEB.pdf": "ar",
    "LAD_Architecture_Book_EN_4WEB.pdf": "en",
    "LAD_Architecture_Book_FR_4WEB.pdf": "fr",
    "LAD_LUXE_BAT_ENgr.pdf": "en",
    "LAD_LUXE_BAT_FRgr.pdf": "fr",
}

# Explicit collection grouping, for citation/provenance -- the LAD paper's
# "source metadata (type, institution, date)" (§3.4) needs something more
# specific than just the shared source_name.
_FILENAME_COLLECTION = {
    "LAD_Architecture_Book_AR_4WEB.pdf": "LAD Architecture Book",
    "LAD_Architecture_Book_EN_4WEB.pdf": "LAD Architecture Book",
    "LAD_Architecture_Book_FR_4WEB.pdf": "LAD Architecture Book",
    "LAD_LUXE_BAT_ENgr.pdf": "LAD LUXE",
    "LAD_LUXE_BAT_FRgr.pdf": "LAD LUXE",
}

RIGHTS_STATEMENT = (
    "Louvre Abu Dhabi publication -- provided directly for internal RAG "
    "testing; public redistribution rights not separately confirmed"
)


def _extract_pages(pdf_path: Path) -> list[str]:
    reader = pypdf.PdfReader(str(pdf_path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _record_for_page(filename: str, lang: str, page_num: int, text: str) -> HeritageRecord:
    return HeritageRecord(
        source_name=SOURCE_NAME,
        source_url=f"internal://lad-publications/{filename}#page={page_num}",
        source_record_id=f"{Path(filename).stem}:p{page_num}",
        retrieval_date=date.today(),
        rights_statement=RIGHTS_STATEMENT,
        reuse_risk=ReuseRisk.UNKNOWN,
        language_code=lang,
        description=text,
        collection=_FILENAME_COLLECTION[filename],
        institution="Louvre Abu Dhabi",
        object_type="publication_page",
    )


def ingest_lad_publications(source_dir: Path = DEFAULT_SOURCE_DIR) -> Path:
    """Extracts every page of every known LAD Publications PDF found in
    `source_dir` into data/processed/lad_publications/needs_review.jsonl
    -- needs_review, not records.jsonl, since redistribution rights
    aren't confirmed (same convention already used for UNESDOC/WDL, whose
    APIs likewise expose no machine-readable rights field). A file in
    source_dir not in this module's known filename mapping is skipped, not
    guessed at. Pages with no extractable text (e.g. a pure-image cover)
    are skipped too, not written as empty records."""
    out_dir = PROCESSED_DIR / SOURCE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "needs_review.jsonl"

    records: list[HeritageRecord] = []
    for filename, lang in _FILENAME_LANGUAGE.items():
        pdf_path = source_dir / filename
        if not pdf_path.exists():
            continue
        for page_num, text in enumerate(_extract_pages(pdf_path), start=1):
            if not text or not text.strip():
                continue
            records.append(_record_for_page(filename, lang, page_num, text))

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    return out_path
