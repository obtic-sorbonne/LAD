"""B4: RAG Synthesis (LAD paper §3.4) -- structured, attested-equivalents-
only synthesis via Claude. Requires ANTHROPIC_API_KEY (not set in this
environment -- same pattern as EUROPEANA_API_KEY: provide your own key via
.env or an exported env var; anthropic.Anthropic() reads it automatically).
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from lad.rag.schema import AttestedEquivalent, RetrievalHit, TerminologyRecord

DEFAULT_LLM_MODEL = "claude-sonnet-5"
PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesis.md"
MAX_PASSAGES_PER_LANG = 5
LANG_NAMES = {"en": "English", "fr": "French", "ar": "Arabic"}


def _cap(hits: list[RetrievalHit], max_n: int = MAX_PASSAGES_PER_LANG) -> list[RetrievalHit]:
    # Passages are already deduplicated by passage_id in retrieval.py and
    # sorted by score -- capping to the top N per language is the LAD
    # paper's "top-5 passages/language" evidence-aggregation step (§3.4).
    return hits[:max_n]


def _format_passages_block(hits_by_lang: dict[str, list[RetrievalHit]]) -> str:
    lines = []
    for lang, hits in hits_by_lang.items():
        lines.append(f"### {LANG_NAMES.get(lang, lang)}")
        for hit in _cap(hits):
            rights_note = "" if hit.reuse_risk == "clear" else " [rights: unverified]"
            lines.append(f"- passage_id={hit.passage_id} (source: {hit.source_name}{rights_note}): {hit.text}")
        lines.append("")
    return "\n".join(lines)


def _extract_text(response: anthropic.types.Message) -> str:
    """response.content[0] isn't reliably the text block -- when extended
    thinking is active, a ThinkingBlock (`.thinking`, no `.text`) comes
    first. Find the actual text block by type instead of assuming
    position. Live bug, not hypothetical: 467/635 synthesis calls in the
    LAD-publications eval run crashed on `content[0].text` being a
    ThinkingBlock (see PROJECT_STATUS.md)."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    block_types = [getattr(b, "type", type(b).__name__) for b in response.content]
    raise ValueError(f"No text block in Claude response (content block types: {block_types})")


def _parse_response(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Be lenient about the model wrapping JSON in a code fence despite
        # being told not to -- strip and retry once before giving up.
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())


def synthesize(
    term: str,
    lang: str,
    hits_by_lang: dict[str, list[RetrievalHit]],
    embedding_model: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    client: anthropic.Anthropic | None = None,
) -> TerminologyRecord:
    client = client or anthropic.Anthropic()

    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        term=term,
        language=LANG_NAMES.get(lang, lang),
        passages_block=_format_passages_block(hits_by_lang),
    )

    response = client.messages.create(
        model=llm_model,
        # 1500 was too tight in practice: this model spends part of its
        # budget on extended thinking before answering (see _extract_text),
        # and 63/635 live calls in the LAD-publications eval either ran out
        # of budget entirely mid-think (no text block at all) or got cut off
        # partway through the JSON output ("Unterminated string..."). 4096
        # gives real headroom for both thinking + a full structured response.
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = _parse_response(_extract_text(response))

    equivalents: dict[str, list[AttestedEquivalent]] = {}
    for lang_code, entries in (parsed.get("equivalents") or {}).items():
        equivalents[lang_code] = [
            AttestedEquivalent(
                label=entry["label"],
                language_code=lang_code,
                attestation_count=len(entry.get("passage_ids", [])),
                passage_ids=entry.get("passage_ids", []),
            )
            for entry in entries
        ]

    all_hits = {hit.passage_id: hit for hits in hits_by_lang.values() for hit in hits}
    cited_ids = {pid for eqs in equivalents.values() for e in eqs for pid in e.passage_ids}
    unclear_count = sum(1 for pid in cited_ids if pid in all_hits and all_hits[pid].reuse_risk != "clear")
    rights_caveat = None
    if unclear_count:
        rights_caveat = f"{unclear_count} of {len(cited_ids)} cited passages have unverified/restricted rights status."

    return TerminologyRecord(
        source_term=term,
        source_language=lang,
        equivalents=equivalents,
        usage_note=parsed.get("usage_note"),
        rights_caveat=rights_caveat,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )
