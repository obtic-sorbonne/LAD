"""Orchestrates running the gold-standard set (eval/gold_set.py) through
the full lexical-enrichment -> retrieval -> (optional) synthesis pipeline
and computing Part C metrics. Builds the gold set first if missing.

Returns both an aggregate `summary` (what the CLI prints) AND per-row
`retrieval_raw`/`synthesis_raw` lists -- the raw, reanalyzable results a
project deliverable, not just a printed average. See eval/report.py for
serializing these to actual JSON/CSV files.
"""

from __future__ import annotations

import statistics
from typing import Any

from pathlib import Path

from lad.rag.embeddings import Embedder
from lad.rag.eval.gold_set import GOLD_SET_PATH, build_gold_set
from lad.rag.eval.metrics import (
    attestation_accuracy,
    equivalence_correctness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    semantic_relevance_at_k,
)
from lad.rag.index import PassageIndex
from lad.rag.lexical_index import LexicalIndex
from lad.rag.rerank import Reranker
from lad.rag.retrieval import retrieve
from lad.rag.synthesis import synthesize
from lad.storage.writer import read_jsonl


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def run_eval(
    run_synthesis: bool = True,
    use_reranker: bool = False,
    generator=None,
    index: PassageIndex | None = None,
    embedder: Embedder | None = None,
    run_label: str | None = None,
    gold_set_path: Path | None = None,
    lexical_index: LexicalIndex | None = None,
) -> dict[str, Any]:
    """`index`/`embedder` are injectable so the same eval can be pointed at
    a different passage index (e.g. the museum-subset baseline vs. the
    lad_publications-expanded variant, see PROJECT_STATUS.md) without
    duplicating this whole function -- both default to the standard
    museum-subset index/LaBSE embedder if not given. `run_label` is a
    free-text tag carried into the output (e.g. "baseline",
    "with_lad_publications") purely for downstream result-file naming/
    identification, not used by this function itself. `gold_set_path`
    defaults to the standard 120-term gold set (auto-built if missing);
    an explicit path (e.g. eval/lad_publications_gold_set.jsonl) is used
    as-is and NOT auto-built if missing -- auto-building only makes sense
    for the one canonical gold set gold_set.py knows how to construct."""
    gold_set_path = gold_set_path or GOLD_SET_PATH
    if gold_set_path == GOLD_SET_PATH and not gold_set_path.exists():
        build_gold_set()
    gold_rows = read_jsonl(gold_set_path)

    embedder = embedder or Embedder()
    index = index or PassageIndex()
    reranker = Reranker() if use_reranker else None

    retrieval_scores: dict[str, list[float]] = {"p_at_5": [], "r_at_10": [], "mrr": [], "semantic_relevance_at_5": []}
    synthesis_scores: dict[str, list[float]] = {"equivalence_correctness": [], "attestation_accuracy": []}
    retrieval_raw: list[dict[str, Any]] = []
    synthesis_raw: list[dict[str, Any]] = []
    errors = 0
    error_details: list[dict[str, Any]] = []

    for row in gold_rows:
        term = row["source_term"]
        lang = row["source_language"]
        reference_equivalents = row["reference_equivalents"]

        hits_by_lang = retrieve(
            term, lang, index, embedder, reranker=reranker, generator=generator, lexical_index=lexical_index
        )

        for ref_lang, ref_label in reference_equivalents.items():
            hits = hits_by_lang.get(ref_lang, [])
            p5 = precision_at_k(hits, ref_label, k=5)
            r10 = recall_at_k(hits, ref_label, k=10)
            mrr = mean_reciprocal_rank(hits, ref_label)
            sem5 = semantic_relevance_at_k(hits, ref_label, embedder, k=5)

            retrieval_scores["p_at_5"].append(p5)
            retrieval_scores["r_at_10"].append(r10)
            retrieval_scores["mrr"].append(mrr)
            retrieval_scores["semantic_relevance_at_5"].append(sem5)

            retrieval_raw.append(
                {
                    "term_id": row.get("term_id"),
                    "source_term": term,
                    "source_language": lang,
                    "target_language": ref_lang,
                    "reference_label": ref_label,
                    "n_hits_retrieved": len(hits),
                    "p_at_5": p5,
                    "r_at_10": r10,
                    "mrr": mrr,
                    "semantic_relevance_at_5": sem5,
                }
            )

        if run_synthesis and hits_by_lang:
            try:
                record = synthesize(term, lang, hits_by_lang, embedding_model=embedder.model_name)
            except Exception as exc:
                errors += 1
                error_details.append(
                    {
                        "term_id": row.get("term_id"),
                        "source_term": term,
                        "source_language": lang,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            passage_text_by_id = {h.passage_id: h.text for hits in hits_by_lang.values() for h in hits}
            eq = equivalence_correctness(record, reference_equivalents)
            att = attestation_accuracy(record, passage_text_by_id)
            synthesis_scores["equivalence_correctness"].append(eq)
            synthesis_scores["attestation_accuracy"].append(att)
            synthesis_raw.append(
                {
                    "term_id": row.get("term_id"),
                    "source_term": term,
                    "source_language": lang,
                    "equivalence_correctness": eq,
                    "attestation_accuracy": att,
                }
            )

    return {
        "run_label": run_label,
        "n_terms": len(gold_rows),
        "index_model_name": index.model_name,
        "embedding_model": embedder.model_name,
        "reranker": reranker.model_name if reranker else None,
        "generator": type(generator).__name__ if generator else None,
        "lexical_fusion": bool(lexical_index),
        "summary": {
            "retrieval": {k: _avg(v) for k, v in retrieval_scores.items()},
            "synthesis": {k: _avg(v) for k, v in synthesis_scores.items()} if run_synthesis else None,
            "synthesis_errors": errors if run_synthesis else None,
            "synthesis_error_details": error_details if run_synthesis else None,
        },
        "retrieval_raw": retrieval_raw,
        "synthesis_raw": synthesis_raw,
    }
