"""CLI entrypoint: `lad harvest`, `lad compact`, `lad validate`, `lad status`."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from lad.config import get_source_config, load_sources
from lad.connectors import REGISTRY
from lad.pipeline.build_termbase import build_termbase
from lad.pipeline.build_termbase_from_kalcium import build_real_termbase
from lad.pipeline.ingest_lad_publications import DEFAULT_SOURCE_DIR as LAD_PUBLICATIONS_DIR
from lad.pipeline.ingest_lad_publications import ingest_lad_publications
from lad.pipeline.compact import compact_source
from lad.pipeline.passagize import passagize_source
from lad.pipeline.publish_hf import build_export
from lad.pipeline.stats import write_stats
from lad.storage import writer

app = typer.Typer(help="Multilingual heritage data collection pipeline")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs the full request URL at INFO -- fine for short URLs, but
    # Getty AAT's SPARQL queries URL-encode into unreadable multi-hundred-
    # character walls of text. Our own connector/pipeline loggers stay at
    # INFO; only httpx's own logger is turned down.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _harvest_one(name: str, refresh: bool, page_limit: Optional[int]) -> None:
    config = get_source_config(name)
    if not config.get("enabled", True):
        typer.echo(f"[skip] {name}: disabled (check config/sources.yaml and .env)")
        return
    connector_cls = REGISTRY.get(name)
    if connector_cls is None:
        typer.echo(f"[skip] {name}: no connector implemented yet")
        return

    connector = connector_cls(config)
    summary = connector.run(refresh=refresh, page_limit=page_limit)
    typer.echo(
        f"[done] {name}: fetched={summary.records_fetched} "
        f"normalized={summary.records_normalized} "
        f"flagged={summary.flagged_for_review} errors={summary.errors}"
    )


@app.command()
def harvest(
    source: Optional[str] = typer.Option(None, help="Source name, e.g. europeana. Omit with --all."),
    all_sources: bool = typer.Option(False, "--all", help="Harvest every enabled source."),
    refresh: bool = typer.Option(False, help="Ignore checkpoint and re-harvest from the start."),
    page_limit: Optional[int] = typer.Option(None, help="Stop after this many new pages (useful for smoke tests)."),
) -> None:
    """Run one or all connectors."""
    _configure_logging()
    if not source and not all_sources:
        typer.echo("Specify --source <name> or --all", err=True)
        raise typer.Exit(code=1)

    names = list(REGISTRY.keys()) if all_sources else [source]
    for name in names:
        _harvest_one(name, refresh=refresh, page_limit=page_limit)


@app.command()
def compact(
    source: Optional[str] = typer.Option(None, help="Source name. Omit with --all."),
    all_sources: bool = typer.Option(False, "--all", help="Compact every source."),
) -> None:
    """Convert finalized JSONL output into partitioned Parquet."""
    names = list(REGISTRY.keys()) if all_sources else [source]
    for name in names:
        for needs_review in (False, True):
            out_path = compact_source(name, needs_review=needs_review)
            if out_path:
                typer.echo(f"[compacted] {name} ({'needs_review' if needs_review else 'records'}) -> {out_path}")


@app.command(name="build-termbase")
def build_termbase_cmd() -> None:
    """Build the interim LAD Termbase substitute from already-harvested
    UNESCO Thesaurus + Getty AAT data (see README: Termbase)."""
    path = build_termbase()
    count = sum(1 for _ in path.open(encoding="utf-8"))
    typer.echo(f"[done] wrote {count} termbase entries -> {path}")


@app.command(name="build-real-termbase")
def build_real_termbase_cmd() -> None:
    """Parse the real Louvre Abu Dhabi Termbase (data/Export from Kalcium.xlsx)
    into data/termbase/real_termbase.jsonl -- replaces the interim
    substitute's near-total Arabic blindness with 100% real Arabic label
    coverage. See pipeline/build_termbase_from_kalcium.py. Requires the
    Kalcium export file to be present (institutional data, not fetched by
    any connector)."""
    path = build_real_termbase()
    count = sum(1 for _ in path.open(encoding="utf-8"))
    typer.echo(f"[done] wrote {count} real termbase entries -> {path}")


@app.command(name="ingest-lad-publications")
def ingest_lad_publications_cmd(
    source_dir: Path = typer.Option(
        LAD_PUBLICATIONS_DIR, help="Directory containing the LAD Publications PDFs."
    ),
) -> None:
    """Extract the real LAD Publications PDFs (page-level text, pypdf) into
    data/processed/lad_publications/needs_review.jsonl -- the first REAL
    institutional documentation this project indexes, vs. every other
    source's public-data substitute. Run `lad passagize --source
    lad_publications` afterward to chunk it like any other source."""
    path = ingest_lad_publications(source_dir=source_dir)
    count = sum(1 for _ in path.open(encoding="utf-8"))
    typer.echo(f"[done] wrote {count} page records -> {path}")


@app.command()
def passagize(
    source: Optional[str] = typer.Option(None, help="Source name. Omit with --all."),
    all_sources: bool = typer.Option(False, "--all", help="Passagize every source."),
) -> None:
    """Chunk harvested records into retrieval-unit passages for the RAG index."""
    names = list(REGISTRY.keys()) if all_sources else [source]
    for name in names:
        out_path = passagize_source(name)
        count = sum(1 for _ in out_path.open(encoding="utf-8"))
        typer.echo(f"[done] {name}: wrote {count} passages -> {out_path}")


@app.command()
def validate(source: str) -> None:
    """Report schema/rights health for one source's processed output."""
    records = writer.read_jsonl(writer.processed_path(source, needs_review=False))
    flagged = writer.read_jsonl(writer.processed_path(source, needs_review=True))
    typer.echo(f"{source}: {len(records)} clean records, {len(flagged)} flagged for rights review")
    if flagged:
        sample_ids = [r.get("source_record_id") for r in flagged[:5]]
        typer.echo(f"  sample flagged ids: {sample_ids}")


@app.command()
def status() -> None:
    """Per-source counts and last-run summary."""
    sources = load_sources()
    for name, config in sources.items():
        records = writer.read_jsonl(writer.processed_path(name, needs_review=False))
        flagged = writer.read_jsonl(writer.processed_path(name, needs_review=True))
        enabled = config.get("enabled", True)
        has_connector = name in REGISTRY
        typer.echo(
            f"{name}: enabled={enabled} connector={'yes' if has_connector else 'no'} "
            f"records={len(records)} flagged={len(flagged)}"
        )

    summary_path = writer.LOGS_DIR / "run_summary.jsonl"
    if summary_path.exists():
        typer.echo("\nLast 5 runs:")
        lines = summary_path.read_text(encoding="utf-8").splitlines()[-5:]
        for line in lines:
            typer.echo(f"  {line}")


@app.command(name="build-hf-export")
def build_hf_export_cmd() -> None:
    """Build a local Hugging Face Hub-ready export (data/hf_export/) --
    does not push anything or touch HF credentials, see scripts/06_push_to_hf.sh."""
    path = build_export()
    typer.echo(f"[done] export ready at {path} (see {path}/README.md for contents)")


@app.command()
def stats() -> None:
    """Regenerate stats.txt from current data/processed/ contents."""
    path = write_stats()
    typer.echo(f"[done] wrote {path}")


# ---- RAG commands (Phase 1) -------------------------------------------
# Imports of lad.rag.* are deliberately deferred into each command body,
# not hoisted to the top of this file -- they pull in torch/sentence-
# transformers/faiss, which cost real time and memory to import. Every
# other command above (harvest/status/stats/...) should stay fast without
# paying that cost.


@app.command(name="build-index")
def build_index_cmd(
    sources: Optional[str] = typer.Option(
        None, help="Comma-separated source names. Defaults to the museum-specific subset (see rag/index.py)."
    ),
) -> None:
    """Build the FAISS passage index (one per language: ar/en/fr) from
    already-passagized data. Requires a GPU-capable torch for reasonable
    speed on the full subset -- see scripts/00_setup.sh."""
    from lad.rag.index import build_index

    source_list = sources.split(",") if sources else None
    counts = build_index(sources=source_list)
    for lang, count in counts.items():
        typer.echo(f"[done] {lang}: {count} passages indexed")


@app.command(name="build-lexical-index")
def build_lexical_index_cmd(
    sources: Optional[str] = typer.Option(
        None, help="Comma-separated source names. Defaults to the museum-specific subset (see rag/index.py)."
    ),
) -> None:
    """Build the BM25 lexical index (one per language) from the same
    passages the FAISS index reads -- see rag/lexical_index.py, Phase 7's
    fix for the short-passage problem (PROJECT_STATUS.md). Use alongside
    --lexical on `lad query`/`lad eval`/`lad repl` to fuse dense + lexical
    retrieval via Reciprocal Rank Fusion."""
    from lad.rag.lexical_index import build_lexical_index

    source_list = sources.split(",") if sources else None
    counts = build_lexical_index(sources=source_list)
    for lang, count in counts.items():
        typer.echo(f"[done] {lang}: {count} passages indexed (lexical)")


def _build_generator(name: Optional[str]):
    """Constructs a rag/generate_expand.py TextGenerator from a CLI choice
    string, or None if generation isn't requested. Shared by query/eval/
    repl so the claude-vs-jais2 branching isn't triplicated. Neither
    backend has a working credential in this development environment
    (see generate_expand.py's module docstring) -- constructing either
    will raise if its credential is actually missing at call time, not
    silently no-op, since a user explicitly asking for --generator should
    get a clear error, not a quiet skip."""
    if name is None:
        return None
    from lad.rag.generate_expand import ClaudeGenerator, Jais2Generator

    if name == "claude":
        return ClaudeGenerator()
    if name == "jais2":
        return Jais2Generator()
    raise typer.BadParameter(f"--generator must be 'claude' or 'jais2', got {name!r}")


_GENERATOR_HELP = (
    "Enable tRAG generate-then-rank (rag/generate_expand.py) for target languages the "
    "termbase/WordNet don't cover: 'claude' (needs ANTHROPIC_API_KEY) or 'jais2' (needs a "
    "Hugging Face token with the gated inceptionai/Jais-2-8B-Chat license accepted). Omit to skip."
)


@app.command()
def query(
    term: str,
    lang: str = typer.Option(..., "--lang", help="Source term's language: ar, en, or fr"),
    top_k: int = typer.Option(10, help="Passages retrieved per language before synthesis"),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Retrieval only, skip the Claude synthesis call"),
    rerank: bool = typer.Option(False, "--rerank", help="Apply cross-encoder reranking (rag/rerank.py) after retrieval"),
    generator: Optional[str] = typer.Option(None, "--generator", help=_GENERATOR_HELP),
    lexical: bool = typer.Option(
        False, "--lexical",
        help="Fuse dense + BM25 lexical retrieval via RRF (rag/lexical_index.py, Phase 7). "
        "Needs `lad build-lexical-index` run first.",
    ),
) -> None:
    """Run one term through lexical enrichment -> retrieval -> synthesis,
    print the structured result. Synthesis needs ANTHROPIC_API_KEY set."""
    from lad.rag.embeddings import Embedder
    from lad.rag.index import PassageIndex
    from lad.rag.lexical_index import LexicalIndex
    from lad.rag.rerank import Reranker
    from lad.rag.retrieval import retrieve

    embedder = Embedder()
    index = PassageIndex()
    reranker = Reranker() if rerank else None
    lexical_index = LexicalIndex() if lexical else None
    gen = _build_generator(generator)
    hits_by_lang = retrieve(
        term, lang, index, embedder, top_k=top_k, reranker=reranker, generator=gen, lexical_index=lexical_index
    )

    for hit_lang, hits in hits_by_lang.items():
        typer.echo(f"\n=== {hit_lang}: {len(hits)} passages ===")
        for hit in hits[:5]:
            typer.echo(f"  [{hit.score:.3f}] ({hit.source_name}) {hit.text[:100]}")

    if no_synthesis:
        return
    if not hits_by_lang:
        typer.echo("\nNo passages retrieved -- skipping synthesis.")
        return

    from lad.rag.synthesis import synthesize

    record = synthesize(term, lang, hits_by_lang, embedding_model=embedder.model_name)
    typer.echo("\n=== Synthesis ===")
    typer.echo(record.model_dump_json(indent=2))


@app.command()
def eval(
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Retrieval metrics only, skip Claude calls"),
    rerank: bool = typer.Option(False, "--rerank", help="Apply cross-encoder reranking (rag/rerank.py) after retrieval"),
    generator: Optional[str] = typer.Option(None, "--generator", help=_GENERATOR_HELP),
    run_label: Optional[str] = typer.Option(None, "--run-label", help="Free-text tag saved into the output, e.g. 'baseline'"),
    gold_set_path: Optional[Path] = typer.Option(
        None, "--gold-set-path", help="Use an alternate gold set (default: the standard 120-term one, auto-built if missing)"
    ),
    index_model_name: Optional[str] = typer.Option(
        None, "--index-model-name",
        help="Load an alternate FAISS index variant by its embeddings/<name>/ directory "
        "(default: the standard museum-subset LaBSE index). E.g. 'labse-plus-lad-publications'.",
    ),
    output_json: Optional[Path] = typer.Option(
        None, "--output-json", help="Write full results (summary + raw per-row rows) to this JSON path"
    ),
    output_csv_dir: Optional[Path] = typer.Option(
        None, "--output-csv-dir",
        help="Write retrieval_raw.csv (and synthesis_raw.csv, if synthesis ran) into this directory",
    ),
    lexical: bool = typer.Option(
        False, "--lexical",
        help="Fuse dense + BM25 lexical retrieval via RRF (rag/lexical_index.py, Phase 7). "
        "Needs `lad build-lexical-index` run first.",
    ),
) -> None:
    """Run a gold-standard set through the pipeline and print Part C
    metrics. Builds the standard gold set first if missing. Use
    --output-json/--output-csv-dir to save raw, reanalyzable results as
    files instead of only printing the averaged summary; use
    --gold-set-path/--index-model-name to run the same pipeline against an
    alternate gold set or FAISS index variant (e.g. to reproduce a
    corpus-expansion A/B -- see PROJECT_STATUS.md)."""
    from lad.rag.eval.report import write_csv, write_json
    from lad.rag.eval.run_eval import run_eval as run_eval_fn
    from lad.rag.index import PassageIndex
    from lad.rag.lexical_index import LexicalIndex

    gen = _build_generator(generator)
    index_obj = PassageIndex(model_name=index_model_name) if index_model_name else None
    lexical_index_obj = LexicalIndex() if lexical else None
    results = run_eval_fn(
        run_synthesis=not no_synthesis,
        use_reranker=rerank,
        generator=gen,
        run_label=run_label,
        gold_set_path=gold_set_path,
        index=index_obj,
        lexical_index=lexical_index_obj,
    )
    summary = results["summary"]
    typer.echo(
        f"Evaluated {results['n_terms']} gold terms "
        f"(reranker: {results['reranker'] or 'off'}, generator: {results['generator'] or 'off'}, "
        f"lexical fusion: {results['lexical_fusion']})\n"
    )
    typer.echo("Retrieval:")
    for metric, value in summary["retrieval"].items():
        typer.echo(f"  {metric}: {value:.3f}")
    if summary["synthesis"] is not None:
        typer.echo("\nSynthesis:")
        for metric, value in summary["synthesis"].items():
            typer.echo(f"  {metric}: {value:.3f}")
        if summary["synthesis_errors"]:
            typer.echo(f"  ({summary['synthesis_errors']} terms failed synthesis and were skipped)")

    if output_json:
        path = write_json(results, output_json)
        typer.echo(f"\n[written] {path}")
    if output_csv_dir:
        retrieval_path = write_csv(results["retrieval_raw"], output_csv_dir / "retrieval_raw.csv")
        typer.echo(f"[written] {retrieval_path}")
        if results["synthesis_raw"]:
            synthesis_path = write_csv(results["synthesis_raw"], output_csv_dir / "synthesis_raw.csv")
            typer.echo(f"[written] {synthesis_path}")


@app.command()
def repl(
    top_k: int = typer.Option(10, help="Passages retrieved per language before synthesis"),
    no_synthesis: bool = typer.Option(False, "--no-synthesis", help="Retrieval only for the whole session"),
    rerank: bool = typer.Option(False, "--rerank", help="Apply cross-encoder reranking (rag/rerank.py) after retrieval"),
    generator: Optional[str] = typer.Option(None, "--generator", help=_GENERATOR_HELP),
    lexical: bool = typer.Option(
        False, "--lexical",
        help="Fuse dense + BM25 lexical retrieval via RRF (rag/lexical_index.py, Phase 7). "
        "Needs `lad build-lexical-index` run first.",
    ),
) -> None:
    """Interactive live-experiment loop: loads LaBSE + the FAISS index
    ONCE, then lets you query as many terms as you want without paying the
    ~15s model-load cost per query the way `lad query` does on every
    invocation. 'quit' or Ctrl-D to exit."""
    from lad.rag.embeddings import Embedder
    from lad.rag.index import PassageIndex
    from lad.rag.lexical_index import LexicalIndex
    from lad.rag.rerank import Reranker
    from lad.rag.retrieval import retrieve

    typer.echo("Loading LaBSE + FAISS index (one-time)...")
    embedder = Embedder()
    index = PassageIndex()
    reranker = None
    if rerank:
        typer.echo("Loading cross-encoder reranker (one-time)...")
        reranker = Reranker()
    lexical_index = None
    if lexical:
        typer.echo("Loading lexical (BM25) index (one-time)...")
        lexical_index = LexicalIndex()
    gen = None
    if generator:
        typer.echo(f"Loading {generator} generator (one-time)...")
        gen = _build_generator(generator)
    typer.echo(f"Ready. Languages indexed: {index.available_languages()}")
    typer.echo("Enter a term, then its language when prompted. 'quit' or Ctrl-D to exit.\n")

    while True:
        try:
            term = input("term> ").strip()
        except EOFError:
            typer.echo("")
            break
        if not term:
            continue
        if term.lower() in ("quit", "exit", "q"):
            break

        try:
            lang = input("lang (ar/en/fr)> ").strip().lower()
        except EOFError:
            typer.echo("")
            break
        if lang not in ("ar", "en", "fr"):
            typer.echo("  language must be ar, en, or fr\n")
            continue

        hits_by_lang = retrieve(
            term, lang, index, embedder, top_k=top_k, reranker=reranker, generator=gen, lexical_index=lexical_index
        )
        if not hits_by_lang:
            typer.echo("  no passages retrieved\n")
            continue

        for hit_lang, hits in hits_by_lang.items():
            typer.echo(f"\n  === {hit_lang}: {len(hits)} passages ===")
            for hit in hits[:5]:
                typer.echo(f"    [{hit.score:.3f}] ({hit.source_name}) {hit.text[:100]}")

        if not no_synthesis:
            from lad.rag.synthesis import synthesize

            try:
                record = synthesize(term, lang, hits_by_lang, embedding_model=embedder.model_name)
                typer.echo("\n  === Synthesis ===")
                for line in record.model_dump_json(indent=2).splitlines():
                    typer.echo(f"  {line}")
            except Exception as exc:
                typer.echo(f"\n  [synthesis failed: {exc}]")

        typer.echo("\n" + "-" * 60 + "\n")

    typer.echo("Goodbye.")


if __name__ == "__main__":
    app()
