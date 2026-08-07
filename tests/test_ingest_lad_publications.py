import json

import pytest

from lad.pipeline import ingest_lad_publications as mod
from lad.storage import writer


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path / "processed")


def test_record_for_page_maps_fields_correctly():
    record = mod._record_for_page("LAD_LUXE_BAT_ENgr.pdf", "en", 5, "Some page text.")

    assert record.source_name == "lad_publications"
    assert record.language_code == "en"
    assert record.description == "Some page text."
    assert record.collection == "LAD LUXE"
    assert record.institution == "Louvre Abu Dhabi"
    assert record.object_type == "publication_page"
    assert record.source_record_id == "LAD_LUXE_BAT_ENgr:p5"
    assert record.source_url == "internal://lad-publications/LAD_LUXE_BAT_ENgr.pdf#page=5"
    assert record.reuse_risk.value == "unknown"
    assert "not separately confirmed" in record.rights_statement


def test_ingest_skips_pages_with_no_extractable_text(tmp_path, monkeypatch):
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    (source_dir / "LAD_LUXE_BAT_ENgr.pdf").write_bytes(b"fake pdf bytes")

    monkeypatch.setattr(mod, "_extract_pages", lambda path: ["Real content here.", "", "   ", "More content."])

    out_path = mod.ingest_lad_publications(source_dir=source_dir)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2  # the two blank/whitespace-only pages were skipped
    assert rows[0]["description"] == "Real content here."
    assert rows[1]["description"] == "More content."
    assert rows[0]["source_record_id"] == "LAD_LUXE_BAT_ENgr:p1"
    assert rows[1]["source_record_id"] == "LAD_LUXE_BAT_ENgr:p4"  # original page number preserved, not renumbered


def test_ingest_skips_files_not_in_the_known_filename_mapping(tmp_path, monkeypatch):
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    (source_dir / "LAD_LUXE_BAT_ENgr.pdf").write_bytes(b"fake pdf bytes")
    (source_dir / "some_other_publication.pdf").write_bytes(b"fake pdf bytes")

    monkeypatch.setattr(mod, "_extract_pages", lambda path: ["text"])

    out_path = mod.ingest_lad_publications(source_dir=source_dir)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["source_record_id"] == "LAD_LUXE_BAT_ENgr:p1"


def test_ingest_writes_to_needs_review_not_records(tmp_path, monkeypatch):
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    (source_dir / "LAD_LUXE_BAT_ENgr.pdf").write_bytes(b"fake pdf bytes")
    monkeypatch.setattr(mod, "_extract_pages", lambda path: ["text"])

    out_path = mod.ingest_lad_publications(source_dir=source_dir)

    assert out_path.name == "needs_review.jsonl"
    assert not (out_path.parent / "records.jsonl").exists()


def test_ingest_correctly_tags_language_per_file(tmp_path, monkeypatch):
    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    for filename in ("LAD_Architecture_Book_AR_4WEB.pdf", "LAD_Architecture_Book_EN_4WEB.pdf", "LAD_Architecture_Book_FR_4WEB.pdf"):
        (source_dir / filename).write_bytes(b"fake pdf bytes")
    monkeypatch.setattr(mod, "_extract_pages", lambda path: ["text"])

    out_path = mod.ingest_lad_publications(source_dir=source_dir)
    rows = {json.loads(line)["source_record_id"]: json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()}

    assert rows["LAD_Architecture_Book_AR_4WEB:p1"]["language_code"] == "ar"
    assert rows["LAD_Architecture_Book_EN_4WEB:p1"]["language_code"] == "en"
    assert rows["LAD_Architecture_Book_FR_4WEB:p1"]["language_code"] == "fr"
    assert all(r["collection"] == "LAD Architecture Book" for r in rows.values())
