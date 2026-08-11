"""tRAG's generate-then-rank / collective verification (LAD paper's C2,
sub-component 3) -- Phase 2, deliberately deferred until now (see the
project plan). For a target language where the termbase/WordNet provide
NO real translation for a term (only the bare-term cross-lingual
embedding fallback added by lexical_enrichment.expand_query()'s "Phase
1.7" fix), an LLM generates candidate terminology variants, which are
then verified against the actual passage corpus before being trusted --
not free generation. This specifically targets the Arabic query-expansion
gap: a sample of 500 real Getty AAT English terms showed 96% had zero
Arabic termbase/WordNet coverage (see PROJECT_STATUS.md "Phase 1.7"), so
this is the step meant to close that gap with something better than just
the bare source term alone.

Two interchangeable generator backends, both behind the same minimal
`TextGenerator` protocol (one `.generate(prompt) -> str` method):
- ClaudeGenerator: needs ANTHROPIC_API_KEY (same credential dependency as
  rag/synthesis.py).
- Jais2Generator: needs a Hugging Face token with the gated
  inceptionai/Jais-2-8B-Chat license accepted
  (huggingface.co/inceptionai/Jais-2-8B-Chat) -- an Arabic-centric
  bilingual (ar/en) 8B model, chosen specifically for Arabic generation
  quality over a general multilingual model, and small enough to run
  comfortably on one of this project's GPUs once access is granted.

**Neither credential is present in this development environment** --
verified directly: no ANTHROPIC_API_KEY, no HF_TOKEN, and
inceptionai/Jais-2-8B-Chat's repo returns HTTP 401 without an accepted-
license token. This module is built and unit-tested against a stub
generator (see tests/test_generate_expand.py), matching the exact pattern
rag/synthesis.py already uses for its own credential-gated dependency.
Neither generator has been exercised live end-to-end yet --
the generation/verification *logic* is tested, not real Jais 2 or Claude
output quality for this specific task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from lad.rag.embeddings import Embedder
from lad.rag.index import PassageIndex
from lad.rag.lexical_enrichment import static_translation_coverage

LANGS = ("ar", "en", "fr")
LANG_NAMES = {"ar": "Arabic", "en": "English", "fr": "French"}
PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_variants.md"
DEFAULT_N_CANDIDATES = 5
DEFAULT_MIN_SIMILARITY = 0.5


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class ClaudeGenerator:
    """Wraps anthropic.Anthropic() -- same credential pattern as
    rag/synthesis.py (ANTHROPIC_API_KEY). Import deferred to __init__ so
    importing this module doesn't require the anthropic package to be
    configured just to use Jais2Generator or a stub instead."""

    def __init__(self, model: str = "claude-sonnet-5", client=None):
        import anthropic

        self.model = model
        self._client = client or anthropic.Anthropic()

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            # 500 proved too tight for rag/synthesis.py's identical call
            # shape once extended thinking eats into the same budget (see
            # that module's max_tokens comment) -- matching its headroom
            # here pre-emptively rather than waiting to hit it live.
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        # response.content[0] isn't reliably the text block -- a
        # ThinkingBlock can come first when extended thinking is active
        # (see rag/synthesis.py's _extract_text, same live bug).
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        block_types = [getattr(b, "type", type(b).__name__) for b in response.content]
        raise ValueError(f"No text block in Claude response (content block types: {block_types})")


class Jais2Generator:
    """Wraps inceptionai/Jais-2-8B-Chat (transformers) -- a gated repo,
    needs a Hugging Face token with the license accepted at
    huggingface.co/inceptionai/Jais-2-8B-Chat. Chosen specifically for
    Arabic generation over a general multilingual model. Not exercised
    live yet (see module docstring) -- verified only that the
    repo exists, is bilingual ar/en, 8B parameters (comfortably fits this
    project's GPUs), and is gated (HTTP 401 without a token); not that its
    generation quality is actually good for this specific task versus
    ClaudeGenerator -- that comparison is exactly what Phase 3's
    comparison harness (project plan) is for, once access is available.
    Imports deferred to __init__ -- torch/transformers cost real time and
    memory to import, and most callers (including tests) never construct
    this class."""

    def __init__(self, model_name: str = "inceptionai/Jais-2-8B-Chat", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(self._device)

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._device)
        output = self._model.generate(input_ids, max_new_tokens=500, do_sample=False)
        generated_tokens = output[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True)


def _parse_candidates(raw_text: str) -> list[str]:
    """Prompt asks for a JSON array; lenient about markdown code-fence
    wrapping despite being told not to -- same pattern as
    rag/synthesis.py's _parse_response. Malformed output degrades to an
    empty candidate list rather than raising, since a generation failure
    for one language shouldn't break retrieval for the others."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    try:
        candidates = json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(candidates, list):
        return []
    return [str(c).strip() for c in candidates if str(c).strip()]


def generate_candidates(
    term: str,
    source_lang: str,
    target_lang: str,
    generator: TextGenerator,
    n: int = DEFAULT_N_CANDIDATES,
) -> list[str]:
    """Prompts `generator` for up to n candidate museum/heritage
    terminology equivalents of `term` in target_lang. Raw output, NOT yet
    verified against the corpus -- see verify_candidates."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        term=term,
        source_lang=LANG_NAMES.get(source_lang, source_lang),
        target_lang=LANG_NAMES.get(target_lang, target_lang),
        n=n,
    )
    raw = generator.generate(prompt)
    return _parse_candidates(raw)


def verify_candidates(
    candidates: list[str],
    target_lang: str,
    index: PassageIndex,
    embedder: Embedder,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[str]:
    """tRAG's "collective verification": keeps only generated candidates
    with at least one passage in the target-language corpus scoring above
    min_similarity -- an LLM can propose a plausible-sounding term that
    isn't actually attested anywhere in this project's corpus; this is
    the check that stops such a term from being trusted just because it
    was generated, rather than free generation. min_similarity's default
    (0.5) is a coarse, not empirically tuned, threshold -- calibrating it
    against real generator output is follow-up work once Jais 2/Claude
    credentials are available to run this live."""
    if not candidates or target_lang not in index.available_languages():
        return []

    vectors = embedder.encode(candidates)
    verified = []
    for candidate, vector in zip(candidates, vectors):
        hits = index.search(target_lang, vector, top_k=1)
        if hits and hits[0][1] >= min_similarity:
            verified.append(candidate)
    return verified


def augment_with_generated_variants(
    query_variants: dict[str, list[str]],
    term: str,
    lang: str,
    generator: TextGenerator,
    index: PassageIndex,
    embedder: Embedder,
    n: int = DEFAULT_N_CANDIDATES,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> dict[str, list[str]]:
    """Adds LLM-generated, corpus-verified variants to `query_variants`
    (expand_query()'s output) for any target language where the
    termbase/WordNet provided no real translation -- only languages that
    actually need it, not every language on every call, since this costs
    one LLM round-trip per under-served language. Purely additive: a
    language with real termbase/WordNet coverage is left untouched, and a
    language where generation produces nothing verifiable is left exactly
    as expand_query() already had it (the bare-term fallback), never
    worse off."""
    static_coverage = static_translation_coverage(term, lang)
    augmented = {code: set(variants) for code, variants in query_variants.items()}

    for target_lang in LANGS:
        if target_lang == lang or static_coverage.get(target_lang):
            continue  # source language itself, or already has a real translation

        candidates = generate_candidates(term, lang, target_lang, generator, n=n)
        verified = verify_candidates(candidates, target_lang, index, embedder, min_similarity=min_similarity)
        if verified:
            augmented.setdefault(target_lang, set()).update(verified)

    return {code: sorted(variants) for code, variants in augmented.items() if variants}
