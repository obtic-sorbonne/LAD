import json

import httpx

from lad.connectors.unesdoc import UnesdocConnector
from lad.pipeline.rights import gate_rights
from lad.schema import HeritageRecord, ReuseRisk


def _connector() -> UnesdocConnector:
    return UnesdocConnector({"license_note": "UNESCO Open Access / CC BY-SA 3.0 IGO (verify per record)"})


def test_discover_yields_once_then_stops_after_checkpoint():
    connector = _connector()
    assert list(connector.discover({})) == [{}]
    assert list(connector.discover({"last_page_index": 1})) == []


def test_parse_returns_records(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "unesdoc" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)

    items = connector.parse(response)

    assert len(items) == 2


def test_normalize_splits_comma_separated_subjects(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "unesdoc" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)

    assert isinstance(record, HeritageRecord)
    assert record.source_name == "unesdoc"
    raw_subject = item["fields"].get("subject", "")
    if raw_subject:
        assert len(record.subject_terms) == len(raw_subject.split(","))
        assert all(term == term.strip() for term in record.subject_terms)


def test_missing_rights_statement_is_flagged_for_review(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "unesdoc" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    item = connector.parse(response)[0]

    record = connector.normalize(item)
    needs_review = gate_rights(record)

    assert record.rights_statement is None
    assert record.reuse_risk == ReuseRisk.UNKNOWN
    assert needs_review is True
