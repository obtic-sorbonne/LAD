import json

import httpx

from lad.connectors.world_digital_library import WorldDigitalLibraryConnector
from lad.pipeline.rights import gate_rights
from lad.schema import HeritageRecord, ReuseRisk


def _connector() -> WorldDigitalLibraryConnector:
    return WorldDigitalLibraryConnector({"license_note": "Varies by item"})


def test_parse_returns_results(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "world_digital_library" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)

    items = connector.parse(response)

    assert len(items) == 2


def test_normalize_handles_list_and_string_typed_fields(fixtures_dir):
    """loc.gov mixes plain strings and single-element lists for the same
    logical field across item types (see connector module docstring) --
    this must not raise a pydantic validation error either way."""
    connector = _connector()
    payload = json.loads((fixtures_dir / "world_digital_library" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)

    for item in connector.parse(response):
        record = connector.normalize(item)
        assert isinstance(record, HeritageRecord)
        assert record.source_record_id == item.get("id", "")


def test_missing_rights_statement_is_flagged_for_review(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "world_digital_library" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)
    needs_review = gate_rights(record)

    assert record.reuse_risk == ReuseRisk.UNKNOWN
    assert needs_review is True
