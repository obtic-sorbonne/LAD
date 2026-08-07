"""Tests for the local HF export builder. No network access, no HF
credentials involved -- publish_hf.py deliberately only writes local
Parquet + a README; the actual push lives in scripts/06_push_to_hf.sh."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from lad.pipeline import publish_hf
from lad.storage import writer


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(publish_hf, "EXPORT_DIR", tmp_path / "hf_export")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_dataset_card_states_rights_percentages_correctly():
    counts = {
        "records/unesco_thesaurus/records": 80,
        "records/unesdoc/needs_review": 20,
    }

    card = publish_hf._dataset_card(counts)

    assert "100 " in card or "100," in card  # total appears somewhere formatted
    assert "80" in card
    assert "20" in card
    assert "reuse_risk" in card
    assert "license: other" in card  # honest about mixed licensing, not a single SPDX license


def test_dataset_card_lists_every_source_license_note():
    card = publish_hf._dataset_card({"records/unesco_thesaurus/records": 1})

    for source_name in publish_hf._SOURCE_LICENSE_NOTES:
        assert source_name in card


def test_build_export_handles_all_empty_dict_fields_without_crashing(tmp_path, monkeypatch):
    """Regression check: the exact bug that broke `compact` for Getty AAT
    (a dict field that's {} on every row) must not break the HF export
    either -- termbase rows have this same shape for most Getty-derived
    entries (alt_labels/scope_note always empty)."""
    processed = writer.PROCESSED_DIR
    _write_jsonl(
        processed / "getty_aat" / "records.jsonl",
        [
            {
                "source_name": "getty_aat",
                "source_url": "https://example.org/1",
                "source_record_id": "1",
                "retrieval_date": "2026-07-21",
                "rights_statement": "ODC-BY",
                "reuse_risk": "clear",
                "term_id": "1",
                "pref_label": {"en": "test"},
                "alt_labels": {},
                "scope_note": {},
                "broader_ids": [],
                "narrower_ids": [],
                "related_ids": [],
                "source_concept_ids": {},
            }
        ],
    )
    monkeypatch.setattr(publish_hf, "REGISTRY", {"getty_aat": object})

    export_path = publish_hf.build_export()  # must not raise

    table = pq.read_table(export_path / "records" / "getty_aat" / "records.parquet")
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert json.loads(row["alt_labels"]) == {}  # JSON-encoded, not a broken struct


def test_build_export_writes_readme(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_hf, "REGISTRY", {})

    export_path = publish_hf.build_export()

    assert (export_path / "README.md").exists()
