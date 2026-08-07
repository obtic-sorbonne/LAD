import json

import httpx

from lad.connectors.getty_aat import GettyAatConnector
from lad.pipeline.rights import gate_rights
from lad.schema import ReuseRisk, VocabularyTerm


def _connector() -> GettyAatConnector:
    return GettyAatConnector({"license_note": "ODC-BY"})


def test_parse_returns_bindings(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "getty_aat" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)

    items = connector.parse(response)

    assert len(items) == 2


def test_normalize_handles_missing_and_present_french_label(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "getty_aat" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    # subject_field is normally stashed by fetch_page(cursor) before parse()/
    # normalize() run; set it directly here since this test skips fetch_page.
    connector._current_subject_field = "materials_and_techniques"
    items = connector.parse(response)

    no_fr, with_fr = (connector.normalize(item) for item in items)

    assert isinstance(no_fr, VocabularyTerm)
    assert no_fr.pref_label == {"en": "balk (timber material)"}
    assert with_fr.pref_label == {"en": "Materials (hierarchy name)", "fr": "Matériaux (hiearchy name)"}
    assert with_fr.subject_field == "materials_and_techniques"


def test_odc_by_passes_rights_gate(fixtures_dir):
    connector = _connector()
    payload = json.loads((fixtures_dir / "getty_aat" / "sample.json").read_text())
    response = httpx.Response(200, json=payload)
    connector._current_subject_field = "materials_and_techniques"
    item = connector.parse(response)[0]

    record = connector.normalize(item)
    needs_review = gate_rights(record)

    assert record.reuse_risk == ReuseRisk.CLEAR
    assert needs_review is False
