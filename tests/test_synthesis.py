import json
from types import SimpleNamespace

from lad.rag.schema import RetrievalHit
from lad.rag.synthesis import synthesize


class _StubMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_call_kwargs = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self._response_text)])


class _StubClient:
    def __init__(self, response_text: str):
        self.messages = _StubMessages(response_text)


def _hit(passage_id, lang, text, reuse_risk="clear"):
    return RetrievalHit(
        passage_id=passage_id,
        language_code=lang,
        text=text,
        score=0.9,
        query_variant="gilding",
        source_name="test_source",
        source_record_id="rec1",
        rights_statement="CC0" if reuse_risk == "clear" else None,
        reuse_risk=reuse_risk,
    )


def test_synthesize_parses_clean_json_response():
    response_json = json.dumps(
        {
            "equivalents": {
                "fr": [{"label": "dorure", "passage_ids": ["p_fr_1"]}],
            },
            "usage_note": "French institutional documentation prefers dorure.",
        }
    )
    client = _StubClient(response_json)
    hits_by_lang = {
        "en": [_hit("p_en_1", "en", "gilding is a technique")],
        "fr": [_hit("p_fr_1", "fr", "la dorure est une technique")],
    }

    record = synthesize("gilding", "en", hits_by_lang, embedding_model="fake-embedder", client=client)

    assert record.source_term == "gilding"
    assert record.equivalents["fr"][0].label == "dorure"
    assert record.equivalents["fr"][0].attestation_count == 1
    assert record.usage_note == "French institutional documentation prefers dorure."
    assert record.rights_caveat is None  # all cited passages were rights-clear


def test_synthesize_strips_markdown_code_fence_if_present():
    response_text = "```json\n" + json.dumps({"equivalents": {}, "usage_note": None}) + "\n```"
    client = _StubClient(response_text)

    record = synthesize("term", "en", {"en": [_hit("p1", "en", "term text")]}, "fake-embedder", client=client)

    assert record.equivalents == {}


def test_synthesize_flags_rights_caveat_for_unverified_citations():
    response_json = json.dumps({"equivalents": {"fr": [{"label": "dorure", "passage_ids": ["p_fr_1"]}]}})
    client = _StubClient(response_json)
    hits_by_lang = {"fr": [_hit("p_fr_1", "fr", "la dorure", reuse_risk="unknown")]}

    record = synthesize("gilding", "en", hits_by_lang, "fake-embedder", client=client)

    assert record.rights_caveat is not None
    assert "1 of 1" in record.rights_caveat


def test_synthesize_sends_expected_prompt_content():
    response_json = json.dumps({"equivalents": {}})
    client = _StubClient(response_json)
    hits_by_lang = {"en": [_hit("p1", "en", "gilding is applied to wood")]}

    synthesize("gilding", "en", hits_by_lang, "fake-embedder", client=client)

    sent_prompt = client.messages.last_call_kwargs["messages"][0]["content"]
    assert "gilding" in sent_prompt
    assert "p1" in sent_prompt
    assert "gilding is applied to wood" in sent_prompt
