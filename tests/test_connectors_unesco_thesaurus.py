import httpx

from lad.connectors.unesco_thesaurus import UnescoThesaurusConnector
from lad.pipeline.rights import gate_rights
from lad.schema import ReuseRisk, VocabularyTerm


def _connector() -> UnescoThesaurusConnector:
    return UnescoThesaurusConnector({"license_note": "CC BY-SA 3.0 IGO"})


def test_discover_yields_once_then_stops_after_checkpoint():
    connector = _connector()
    assert list(connector.discover({})) == [{}]
    assert list(connector.discover({"last_page_index": 1})) == []


def test_parse_and_normalize_extracts_multilingual_labels(fixtures_dir):
    connector = _connector()
    ttl_text = (fixtures_dir / "unesco_thesaurus" / "sample.ttl").read_text()
    response = httpx.Response(200, text=ttl_text)

    items = connector.parse(response)
    assert len(items) == 2

    records = [connector.normalize(item) for item in items]
    concept460 = next(r for r in records if r.term_id.endswith("concept460"))

    assert isinstance(concept460, VocabularyTerm)
    assert concept460.pref_label == {
        "ar": "اتصالات",
        "en": "Communication",
        "fr": "Communication",
    }
    assert concept460.alt_labels == {"en": ["Communications"]}
    assert concept460.scope_note["en"] == "Interactive social process."
    assert concept460.narrower_ids == ["http://vocabularies.unesco.org/thesaurus/concept461"]


def test_open_license_passes_rights_gate(fixtures_dir):
    connector = _connector()
    ttl_text = (fixtures_dir / "unesco_thesaurus" / "sample.ttl").read_text()
    response = httpx.Response(200, text=ttl_text)
    item = connector.parse(response)[0]
    record = connector.normalize(item)

    needs_review = gate_rights(record)

    assert record.reuse_risk == ReuseRisk.CLEAR
    assert needs_review is False
