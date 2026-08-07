"""BM25 lexical retrieval, combined with dense FAISS retrieval via
Reciprocal Rank Fusion -- see PROJECT_STATUS.md "Phase 7". Built because
Phases 5 and 6 both showed the short-passage problem (short, context-free
passages scoring spuriously high dense-embedding similarity to unrelated
queries -- e.g. "tuile" ranking near the top for the query "dorure",
completely unrelated) is NOT fixable by filtering, deduplication, or
swapping the embedding model -- it's a property of dense sentence
embeddings applied to short, context-free strings in general, demonstrated
directly across two different models (LaBSE and multilingual-e5-large).

The fix that actually targets the mechanism: a passage's dense similarity
score isn't the only signal available -- exact/near-exact lexical overlap
is a second, independent signal that's much more reliable for exactly the
cases where dense similarity is unreliable (short strings). "tuile" has
ZERO lexical overlap with "dorure" (verified directly: BM25 score 0.0 --
see test_lexical_index.py), so a lexical signal correctly excludes it,
regardless of what its embedding similarity says.

Fusion via Reciprocal Rank Fusion (RRF), not raw score blending -- BM25
scores and cosine similarities are on incomparable scales (BM25 is
unbounded and corpus-dependent; cosine similarity is bounded [-1,1]), so
combining RANKS rather than raw scores avoids having to calibrate a blend
weight between two different measurement systems. Standard constant
k=60, from the original RRF paper (Cormack, Clarke & Buettcher, 2009).

Tokenization is intentionally minimal (lowercase + \\w+ matching, which
covers Arabic Unicode word characters, Latin letters, and digits) -- not
stemmed or otherwise processed, since passage text is already normalized
upstream at passagize.py's indexing time (Arabic diacritic/alef folding)
and query variants are already normalized the same way by
lexical_enrichment.expand_query() before reaching this module.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from lad.storage.writer import DATA_DIR, read_jsonl

LEXICAL_DIR = DATA_DIR / "embeddings" / "lexical"
LANGS = ("ar", "en", "fr")
RRF_K = 60

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def lexical_index_paths(lang: str, base_dir: Path = LEXICAL_DIR) -> tuple[Path, Path]:
    return base_dir / f"bm25_{lang}.pkl", base_dir / f"meta_{lang}.jsonl"


def build_lexical_index(sources: list[str] | None = None, out_dir: Path = LEXICAL_DIR) -> dict[str, int]:
    """Builds one BM25 index per language from the same passages/*.jsonl
    files the dense FAISS index reads -- no new data pipeline needed.
    Returns {lang: passage_count}."""
    from lad.rag.index import MUSEUM_SOURCES, resolve_lang_bucket  # deferred: avoid a module-load cycle

    sources = sources if sources is not None else MUSEUM_SOURCES
    by_lang: dict[str, list[dict[str, Any]]] = {lang: [] for lang in LANGS}
    for source in sources:
        path = DATA_DIR / "passages" / f"{source}.jsonl"
        for passage in read_jsonl(path):
            bucket = resolve_lang_bucket(passage.get("language_code"))
            if bucket:
                by_lang[bucket].append(passage)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for lang, passages in by_lang.items():
        if not passages:
            counts[lang] = 0
            continue
        tokenized_corpus = [_tokenize(p["text"]) for p in passages]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_path, meta_path = lexical_index_paths(lang, out_dir)
        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        with meta_path.open("w", encoding="utf-8") as f:
            for passage in passages:
                f.write(json.dumps(passage, ensure_ascii=False) + "\n")
        counts[lang] = len(passages)

    return counts


class LexicalIndex:
    """Loaded BM25 indices (one per language) + passage metadata, ready
    for per-language lexical search -- the lexical-side counterpart to
    rag/index.py's PassageIndex."""

    def __init__(self, base_dir: Path = LEXICAL_DIR):
        self._bm25: dict[str, BM25Okapi] = {}
        self._meta: dict[str, list[dict[str, Any]]] = {}
        for lang in LANGS:
            bm25_path, meta_path = lexical_index_paths(lang, base_dir)
            if bm25_path.exists():
                with bm25_path.open("rb") as f:
                    self._bm25[lang] = pickle.load(f)
                self._meta[lang] = read_jsonl(meta_path)

    def available_languages(self) -> list[str]:
        return sorted(self._bm25)

    def search(self, lang: str, query_text: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Returns up to top_k (passage_meta, bm25_score) pairs, scores
        descending, excluding zero-score (no lexical overlap at all)
        matches -- a BM25 score of exactly 0 means no query token appears
        in the passage, which shouldn't count as "found", just unranked."""
        if lang not in self._bm25:
            return []
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25[lang].get_scores(tokens)
        meta = self._meta[lang]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(meta[i], float(scores[i])) for i in ranked_idx if scores[i] > 0]


def reciprocal_rank_fusion(
    dense_ranked_ids: list[str], lexical_ranked_ids: list[str], k: int = RRF_K
) -> dict[str, float]:
    """Returns {passage_id: fused_score}, higher is better. Ranks are
    1-indexed positions within each input list; a passage_id absent from
    one list contributes 0 from that side, not a penalty -- so a passage
    strong in only one signal still surfaces, just not as strongly as one
    strong in both."""
    scores: dict[str, float] = {}
    for rank, passage_id in enumerate(dense_ranked_ids, start=1):
        scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (k + rank)
    for rank, passage_id in enumerate(lexical_ranked_ids, start=1):
        scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (k + rank)
    return scores
