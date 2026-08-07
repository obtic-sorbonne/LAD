"""Builds and loads the FAISS passage index.

One index **per language** (ar/en/fr), not per source and not a single
mixed index. The LAD paper's retrieval spec (C3) is "top-k passages *per
language*" -- a single combined index across languages can't guarantee
that (FAISS returns globally-nearest neighbors regardless of language, so
a query could get back 10 English hits and zero Arabic ones even when
relevant Arabic passages exist further down the ranking). Splitting by
language guarantees each language bucket is actually searched
independently. Still combined *across sources* within each language index
-- retrieval shouldn't have to hit N per-source indices and merge.

Passages whose language isn't one of ar/en/fr (e.g. World Digital
Library's Russian/Italian/etc. items -- WDL is multilingual well beyond
this project's scope) are excluded from the index entirely, not kept in
an unused fourth bucket.

Decided in Phase 1: the first index covers the museum-specific
subset only (Getty AAT + UNESCO Thesaurus + Europeana + WDL = 96,552
passages pre-language-filter) -- UNESDOC's 370,990 administrative/policy
passages are excluded for now, added back in Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from lad.rag.embeddings import DEFAULT_MODEL, Embedder
from lad.rag.passage_quality import filter_and_dedupe
from lad.storage.writer import DATA_DIR, read_jsonl

MUSEUM_SOURCES = ["getty_aat", "unesco_thesaurus", "europeana", "world_digital_library"]
LANGS = ("ar", "en", "fr")

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

_LANG_BUCKET_MARKERS = {
    "ar": {"ar", "ara", "arabic"},
    "en": {"en", "eng", "english"},
    "fr": {"fr", "fre", "fra", "french"},
}


def resolve_lang_bucket(language_code: str | None) -> str | None:
    """Maps a raw Passage.language_code (varies by source: ISO 639-1,
    ISO 639-2, or full English words) to ar/en/fr, or None if it's outside
    this project's trilingual scope. Exact match, not substring -- the
    museum-subset sources never use compound codes (unlike UNESDOC)."""
    normalized = (language_code or "").strip().lower()
    for bucket, markers in _LANG_BUCKET_MARKERS.items():
        if normalized in markers:
            return bucket
    return None


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def index_paths(lang: str, model_name: str = DEFAULT_MODEL) -> tuple[Path, Path]:
    out_dir = EMBEDDINGS_DIR / _model_slug(model_name)
    return out_dir / f"index_{lang}.faiss", out_dir / f"meta_{lang}.jsonl"


def build_index(
    sources: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    embedder: Embedder | None = None,
    batch_size: int = 64,
    filter_low_quality: bool = False,
) -> dict[str, int]:
    """Builds one FAISS index per language. Returns {lang: passage_count}.

    `filter_low_quality` (default False, preserving prior behavior exactly
    for anyone not opting in) applies rag/passage_quality.py's field_source
    + minimum-token-count filter and exact-duplicate dedup before
    embedding -- see that module's docstring and PROJECT_STATUS.md's
    "Phase 5" for why this exists (75% of the museum-subset index was
    bare, frequently-duplicated short labels actively harming retrieval,
    not just adding noise) and the measured effect of turning it on."""
    sources = sources if sources is not None else MUSEUM_SOURCES
    embedder = embedder or Embedder(model_name)

    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in LANGS}
    for source in sources:
        path = DATA_DIR / "passages" / f"{source}.jsonl"
        for passage in read_jsonl(path):
            bucket = resolve_lang_bucket(passage.get("language_code"))
            if bucket:
                by_lang[bucket].append(passage)

    if filter_low_quality:
        by_lang = {lang: filter_and_dedupe(passages) for lang, passages in by_lang.items()}

    counts: dict[str, int] = {}
    for lang, passages in by_lang.items():
        if not passages:
            counts[lang] = 0
            continue

        texts = [p["text"] for p in passages]
        embeddings = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)

        dim = embeddings.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        ids = np.arange(len(passages), dtype=np.int64)
        index.add_with_ids(embeddings, ids)

        index_path, meta_path = index_paths(lang, model_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))

        with meta_path.open("w", encoding="utf-8") as f:
            for faiss_id, passage in zip(ids, passages):
                row = dict(passage)
                row["faiss_id"] = int(faiss_id)
                row["embedding_model"] = model_name
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts[lang] = len(passages)

    return counts


class PassageIndex:
    """Loaded FAISS indices (one per language) + passage metadata, ready
    for per-language search."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._indices: dict[str, faiss.Index] = {}
        self._meta: dict[str, list[dict[str, Any]]] = {}
        for lang in LANGS:
            index_path, meta_path = index_paths(lang, model_name)
            if index_path.exists():
                self._indices[lang] = faiss.read_index(str(index_path))
                # meta files are written in the same order faiss_ids were
                # assigned (0..N-1), so list position == faiss_id.
                self._meta[lang] = read_jsonl(meta_path)

    def available_languages(self) -> list[str]:
        return sorted(self._indices)

    def search(self, lang: str, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        if lang not in self._indices:
            return []
        scores, ids = self._indices[lang].search(query_vector.reshape(1, -1), top_k)
        meta = self._meta[lang]
        hits = []
        for faiss_id, score in zip(ids[0], scores[0]):
            if faiss_id == -1:
                continue
            hits.append((meta[faiss_id], float(score)))
        return hits
