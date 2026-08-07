"""Cross-encoder reranking -- LAD paper's C2 "re-ranked by a cross-encoder
model (mMiniLM-L6)" step, present in both source papers' pipelines
(mMiniLM-L6 / ms-marco-MiniLM-L-12-v2) and missing from this codebase
until now (see project plan Part B1; PROJECT_STATUS.md's ablation table
shows the paper measuring -5.3 P@5 / -4.5 correctness when this step is
removed -- i.e. free precision being left on the table).

Model choice: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, trained on
mMARCO (machine-translated MS MARCO into ~14 languages including Arabic)
-- picked over a purely-English cross-encoder specifically because this
system needs to score Arabic and French pairs too, not just English ones.

Verified live (not assumed): this cross-encoder scores SAME-language pairs
reliably (Arabic query "تذهيب" vs. a relevant Arabic passage: +0.63; vs. an
irrelevant one: -4.33) but is NOT reliably cross-lingual (the English query
"gilding" against that same relevant Arabic passage scores -2.04, wrongly
negative) -- mMARCO's per-language training data is monolingual pairs, not
cross-lingual ones. This is exactly why `rerank_hits` scores each hit
against its own `query_variant` (already carried on every RetrievalHit,
set by retrieval.py to whichever same-language expanded-query string
matched it) rather than the original, possibly different-language, source
term -- a cross-lingual query would give the cross-encoder an unreliable
signal to rerank by.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from lad.rag.schema import RetrievalHit

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Re-scores and re-sorts hits by cross-encoder relevance against
        each hit's own query_variant (same language as the hit -- see
        module docstring for why not the original source term). Returns
        new RetrievalHit copies with `score` replaced by the cross-encoder
        score -- the original embedding-similarity score is not preserved
        alongside it, since the two scores aren't on a comparable scale
        and mixing them would need a fusion strategy this pass doesn't
        implement (see project plan for a future comparison-harness
        phase)."""
        if not hits:
            return hits

        pairs = [(h.query_variant, h.text) for h in hits]
        scores = self._model.predict(pairs)

        rescored = [h.model_copy(update={"score": float(score)}) for h, score in zip(hits, scores)]
        return sorted(rescored, key=lambda h: h.score, reverse=True)
