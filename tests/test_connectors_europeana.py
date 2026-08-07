import json

import httpx

from lad.connectors.europeana import EuropeanaConnector
from lad.pipeline.rights import gate_rights
from lad.schema import HeritageRecord, ReuseRisk


def _connector() -> EuropeanaConnector:
    return EuropeanaConnector(
        {"license_note": "Metadata mostly CC0; per-item edmRights governs media reuse"}
    )


def test_parse_returns_items(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "europeana" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)

    items = connector.parse(response)

    assert len(items) == 2


def test_normalize_maps_core_fields(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "europeana" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)

    assert isinstance(record, HeritageRecord)
    assert record.source_name == "europeana"
    assert record.source_record_id == item["id"]
    assert record.title  # title list was flattened to a single string
    assert isinstance(record.title, str)


def test_restrictive_rights_statement_is_flagged_for_review(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "europeana" / "sample.json").read_text())
    # Force a known-restrictive rightsstatements.org value regardless of fixture content.
    payload["items"][0]["rights"] = ["http://rightsstatements.org/vocab/InC-EDU/1.0/"]
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)
    needs_review = gate_rights(record)

    assert record.reuse_risk == ReuseRisk.RESTRICTED
    assert needs_review is True


def test_open_rights_statement_passes_gate(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "europeana" / "sample.json").read_text())
    payload["items"][0]["rights"] = ["http://creativecommons.org/publicdomain/mark/1.0/"]
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)
    needs_review = gate_rights(record)

    assert record.reuse_risk == ReuseRisk.CLEAR
    assert needs_review is False
