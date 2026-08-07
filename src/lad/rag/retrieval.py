"""B3: Cross-Lingual Semantic Retrieval (LAD paper §3.3).

Encodes the lexically-enriched query set and searches each language's
FAISS index independently (see index.py's docstring for why per-language,
not one mixed index), returning top-k passages per language deduplicated
across query variants (the same passage can match more than one variant;
keep its best score).
"""

from __future__ import annotations

from lad.rag.embeddings import Embedder
from lad.rag.generate_expand import TextGenerator, augment_with_generated_variants
from lad.rag.index import PassageIndex
from lad.rag.lexical_enrichment import expand_query
from lad.rag.lexical_index import LexicalIndex, reciprocal_rank_fusion
from lad.rag.rerank import Reranker
from lad.rag.schema import RetrievalHit


def retrieve(
    term: str,
    lang: str,
    index: PassageIndex,
    embedder: Embedder,
    top_k: int = 10,
    reranker: Reranker | None = None,
    fetch_k: int | None = None,
    generator: TextGenerator | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, list[RetrievalHit]]:
    """Returns {lang_code: [RetrievalHit, ...]} for every language the
    query was expanded into (ar/en/fr, whichever the index has data for),
    each list sorted by score descending, length <= top_k.

    `reranker` is optional (see rag/rerank.py, LAD paper's C2 cross-encoder
    step) -- when given, FAISS is searched with a wider net (`fetch_k`,
    default 3x top_k) so the cross-encoder has more than top_k candidates
    to actually rerank among. Reranking a set already truncated to top_k
    by embedding similarity alone would only reorder it, not let the
    cross-encoder change *which* passages survive. `reranker=None` (the
    default) reproduces the exact pre-reranking behavior, unchanged.

    `generator` is optional (see rag/generate_expand.py, tRAG's generate-
    then-rank step) -- when given, any target language with no real
    termbase/WordNet translation gets LLM-generated, corpus-verified
    candidate variants added on top of the bare-term fallback. `None` (the
    default) skips this entirely, unchanged from before this existed.

    `lexical_index` is optional (see rag/lexical_index.py, Phase 7's fix
    for the short-passage problem found in Phases 4-6) -- when given,
    each target language is ALSO searched via BM25, and the dense/lexical
    rankings are combined with Reciprocal Rank Fusion before reranking or
    truncation. This is what actually fixes the "tuile" outranking
    "dorure"-relevant passages problem: a passage with zero lexical
    overlap to the query gets zero contribution from the lexical side,
    pulling its fused rank down regardless of a spuriously high dense
    score. `None` (the default) reproduces pre-Phase-7 behavior exactly."""
    fetch_k = fetch_k if fetch_k is not None else (top_k * 3 if (reranker or lexical_index) else top_k)
    query_variants = expand_query(term, lang)
    if generator is not None:
        query_variants = augment_with_generated_variants(query_variants, term, lang, generator, index, embedder)

    results: dict[str, list[RetrievalHit]] = {}
    for target_lang, variants in query_variants.items():
        if target_lang not in index.available_languages():
            continue

        vectors = embedder.encode(variants)
        best_by_passage: dict[str, RetrievalHit] = {}

        for variant, vector in zip(variants, vectors):
            for meta, score in index.search(target_lang, vector, top_k=fetch_k):
                passage_id = meta["passage_id"]
                existing = best_by_passage.get(passage_id)
                if existing is not None and existing.score >= score:
                    continue
                best_by_passage[passage_id] = RetrievalHit(
                    passage_id=passage_id,
                    language_code=meta["language_code"],
                    text=meta["text"],
                    score=score,
                    query_variant=variant,
                    source_name=meta["source_name"],
                    source_record_id=meta["source_record_id"],
                    rights_statement=meta.get("rights_statement"),
                    reuse_risk=meta.get("reuse_risk", "unknown"),
                )

        ranked = sorted(best_by_passage.values(), key=lambda h: h.score, reverse=True)

        if lexical_index is not None and target_lang in lexical_index.available_languages():
            ranked = _fuse_with_lexical(ranked, best_by_passage, variants, target_lang, lexical_index, fetch_k)

        if reranker is not None and ranked:
            ranked = reranker.rerank(ranked)
        if ranked:
            results[target_lang] = ranked[:top_k]

    return results


def _fuse_with_lexical(
    dense_ranked: list[RetrievalHit],
    dense_by_passage: dict[str, RetrievalHit],
    variants: list[str],
    target_lang: str,
    lexical_index: LexicalIndex,
    fetch_k: int,
) -> list[RetrievalHit]:
    """Searches `lexical_index` with every query variant (pooling across
    variants the same way the dense side does: keep each passage's best
    lexical score and the variant that produced it), then combines the
    dense and lexical rankings with RRF. Returns a new list of
    RetrievalHits sorted by fused score -- `.score` is overwritten with
    the fused score (same convention rag/rerank.py already uses), since
    the original dense cosine similarity and the fused RRF score aren't
    on comparable scales and mixing them in one field would be
    misleading."""
    lexical_best: dict[str, tuple[dict, float, str]] = {}
    for variant in variants:
        for meta, score in lexical_index.search(target_lang, variant, top_k=fetch_k):
            passage_id = meta["passage_id"]
            existing = lexical_best.get(passage_id)
            if existing is not None and existing[1] >= score:
                continue
            lexical_best[passage_id] = (meta, score, variant)

    if not lexical_best:
        return dense_ranked  # nothing lexically found -- fusion would be a no-op, skip it

    dense_ranked_ids = [h.passage_id for h in dense_ranked]
    lexical_ranked_ids = [
        pid for pid, _ in sorted(lexical_best.items(), key=lambda item: item[1][1], reverse=True)
    ]
    fused_scores = reciprocal_rank_fusion(dense_ranked_ids, lexical_ranked_ids)

    hits_by_id: dict[str, RetrievalHit] = dict(dense_by_passage)
    for passage_id, (meta, score, variant) in lexical_best.items():
        if passage_id in hits_by_id:
            continue  # already have a RetrievalHit for it from the dense side
        hits_by_id[passage_id] = RetrievalHit(
            passage_id=passage_id,
            language_code=meta["language_code"],
            text=meta["text"],
            score=score,
            query_variant=variant,
            source_name=meta["source_name"],
            source_record_id=meta["source_record_id"],
            rights_statement=meta.get("rights_statement"),
            reuse_risk=meta.get("reuse_risk", "unknown"),
        )

    fused_hits = [
        hit.model_copy(update={"score": fused_scores.get(hit.passage_id, 0.0)}) for hit in hits_by_id.values()
    ]
    return sorted(fused_hits, key=lambda h: h.score, reverse=True)
