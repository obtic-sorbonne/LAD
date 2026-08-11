# LAD — Multilingual Heritage Data Pipeline + RAG System

A data pipeline that collects Arabic/French/English cultural-heritage text and terminology, and a term-centric RAG system built on top of it for museum terminology discovery: lexical query expansion, cross-lingual retrieval (FAISS + LaBSE, with an optional BM25 lexical fusion layer), and structured synthesis via Claude that only returns attested, citable equivalents.

Everything runs locally against files on disk — JSONL, Parquet, FAISS indices — with no database server required.

This README covers setup and day-to-day usage. For the project's development history, experimental findings, and current known issues, see `PROJECT_STATUS.md`.

## Setup

```bash
./scripts/00_setup.sh
```

This creates `.venv`, installs dependencies (including a GPU-matched build of torch), and downloads the WordNet/Open Multilingual Wordnet data used for cross-lingual synonym lookup. Every other script calls this automatically if `.venv` doesn't exist yet, so you don't normally need to run it by hand.

The setup script prefers `python3.11` specifically if it's available, and falls back to `python3` otherwise. This matters for GPU support: on the deployment machine (CUDA 12.4 driver), PyTorch's official Python 3.14 wheels only support CUDA 12.6+, so there's no compatible GPU build for that Python version — only CPU-only ones. Python 3.11 has matching cu118/cu121/cu124 wheels. If your machine doesn't have `python3.11` installed, everything still works on CPU, just slower for embedding generation.

**API keys.** Both are optional — the affected feature is skipped or disabled if the key is missing, nothing crashes:

- `EUROPEANA_API_KEY` — free key from europeana.eu, needed for the Europeana harvest.
- `ANTHROPIC_API_KEY` — needed for RAG synthesis (stages 8/9's Claude calls). Retrieval-only mode (`--no-synthesis`) works without it.

Put either in `.env` (see `.env.example`) or export them directly.

## Pipeline stages

Each stage is a standalone numbered script in `scripts/`. They read from stage 1's output independently rather than forming a strict chain, so you only need to re-run a stage when its actual inputs changed.

```bash
./scripts/00_setup.sh                    # environment + WordNet data
./scripts/01_harvest.sh                  # pull data from every source
./scripts/02_compact.sh                  # JSONL -> partitioned Parquet
./scripts/03_build_termbase.sh           # derive the interim (public-data) termbase
./scripts/03b_build_real_termbase.sh     # parse the real institutional termbase
./scripts/04_passagize.sh                # chunk records into RAG passages
./scripts/05_stats.sh                    # regenerate stats.txt
./scripts/06_push_to_hf.sh               # push a Parquet export to the Hugging Face Hub
./scripts/07_build_index.sh              # build the dense FAISS index
./scripts/08_query.sh "term" en          # query one term end to end
./scripts/09_eval.sh                     # run the gold-standard evaluation
./scripts/10_experiment.sh               # interactive REPL for repeated queries
./scripts/11_ingest_lad_publications.sh  # ingest the real LAD publication PDFs
./scripts/12_build_lad_publications_gold_set.sh  # build the publications-derived gold set
```

### Stage 1 — Harvest

Pulls from five sources through a shared connector interface (`src/lad/connectors/base.py`) that handles retry/backoff, rate limiting, checkpointed resume, and rights classification uniformly, so each source connector only implements `discover → fetch_page → parse → normalize`.

| Source | Kind | Access | Notes |
|---|---|---|---|
| UNESCO Thesaurus | vocabulary | one-shot SKOS/Turtle download | 4,499 concepts, fully trilingual |
| Getty AAT | vocabulary | SPARQL, paginated per facet | English/French only, no Arabic labels |
| UNESDOC | records | bulk JSON download | UNESCO Digital Library catalogue, 285,433 records |
| World Digital Library | records | loc.gov JSON API, paginated | Library of Congress, a growing live collection |
| Europeana | records | REST search, paginated | requires an API key; query is French-scoped, since Europeana carries no Arabic content |

```bash
./scripts/01_harvest.sh                            # all sources
./scripts/01_harvest.sh --source europeana          # one source
./scripts/01_harvest.sh --source europeana --clean --refresh   # wipe and re-harvest
```

`PAGE_LIMIT` (env var, default 5000) caps how many pages a single run fetches. Re-running resumes from a checkpoint rather than starting over, so raising the cap incrementally is safe. `--clean` wipes a source's prior output first, useful after changing its query scope.

Europeana's free tier caps result offsets around 1000; harvesting stops cleanly there rather than erroring. Full-corpus Europeana harvesting would need cursor-based pagination, which isn't implemented.

### Stage 2 — Compact

Repackages the harvested JSONL into partitioned Parquet (`data/processed_parquet/source=<x>/date=<y>/*.parquet`), queryable with DuckDB or similar without a database server. No filtering or transformation — same rows, same fields.

### Stage 3 — Interim termbase

Builds `data/termbase/interim_termbase.jsonl` by filtering already-harvested UNESCO Thesaurus and Getty AAT concepts to museum-relevant subject fields. This is a public-data stand-in for the real Louvre Abu Dhabi Termbase, tagged `source_name=lad_termbase_interim` so it's never confused with the real thing downstream.

### Stage 3b — Real termbase

Parses the actual Louvre Abu Dhabi Termbase — an institutional Kalcium export at `data/Export from Kalcium.xlsx`, which must already be present locally (it isn't fetched by any connector and isn't committed to the repository). 4,652 concepts, effectively full trilingual coverage. Output is tagged `source_name=lad_termbase_real`, marked `reuse_risk=restricted`, and deliberately excluded from the public Hugging Face export.

`rag/lexical_enrichment.py` merges both termbases at query time — the real one first, the interim one filling gaps outside its 4,652 curated concepts.

### Stage 4 — Passagize

Chunks harvested records into `data/passages/<source>.jsonl` — the actual unit the RAG index searches over, not the raw records. 150–200 token chunks with 20-token overlap. Draws from both rights-clear and needs-review records (a v1 corpus needs the volume), with each passage carrying its parent record's rights status forward so nothing downstream has to re-derive it.

Arabic text is normalized (diacritic stripping, alef/teh-marbuta folding) before indexing, with the original preserved separately for citation and display.

### Stage 5 — Stats

Regenerates `stats.txt` from whatever is currently on disk in `data/`.

### Stage 6 — Push to Hugging Face

Builds a local Parquet export and pushes it to the private dataset repository `SorbonneUniversity/LAD-Collected-Dataset`. Requires Hugging Face auth already set up (`huggingface-cli login` or `HF_TOKEN`) — this repo's code never reads or writes a token itself.

```bash
huggingface-cli login
./scripts/06_push_to_hf.sh
```

### Stage 7 — Build index

```bash
./scripts/07_build_index.sh
./scripts/07_build_index.sh --sources getty_aat,unesco_thesaurus
```

Builds one FAISS index per language (Arabic/English/French), not a single mixed index — a combined index can't guarantee top-k results per language, since it returns globally-nearest neighbors regardless of language. Passages in any other language are excluded. Uses `sentence-transformers/LaBSE` on GPU when available.

`lad build-lexical-index` builds the companion BM25 index used by `--lexical` (see below).

### Stage 8 — Query

```bash
./scripts/08_query.sh "manuscript" en
./scripts/08_query.sh "gilding" en --no-synthesis --lexical
```

Runs one term through the full pipeline: lexical enrichment (termbase + WordNet + Arabic normalization), retrieval (dense, optionally fused with BM25 lexical search via `--lexical`), and synthesis (a Claude call that returns only attested, citable equivalents). Also accepts `--rerank`, `--generator claude|jais2`, `--index-model-name`, and `--lexical-index-name` — see `lad query --help`.

### Stage 9 — Eval

```bash
./scripts/09_eval.sh
./scripts/09_eval.sh --no-synthesis
```

Runs the gold-standard set through the full pipeline and reports retrieval metrics (P@5, R@10, MRR) and, unless `--no-synthesis`, synthesis metrics (equivalence correctness, attestation accuracy). The set is auto-sampled from the real termbase rather than terminologist-validated, so treat results as a consistency signal across runs, not as a benchmark figure. `--output-json`/`--output-csv-dir` persist full per-row results, not just the averaged summary.

### Stage 10 — Live REPL

```bash
./scripts/10_experiment.sh
./scripts/10_experiment.sh --no-synthesis
```

`08_query.sh` reloads the embedding model and the full index on every call. This loads everything once and drops into a loop instead, which is much better suited to iterating on queries by hand.

### Stage 11 — Ingest LAD Publications

```bash
./scripts/11_ingest_lad_publications.sh
```

Extracts and passagizes real Louvre Abu Dhabi publication PDFs (provided directly, not harvested — must already exist at `data/raw_pdfs/lad_publications` or a path passed via `--source-dir`). This is the first real institutional documentation the project indexes, as opposed to public-data substitutes.

### Stage 12 — Build the LAD Publications gold set

```bash
./scripts/12_build_lad_publications_gold_set.sh
```

Builds a second, independently-constructed gold set: an entry only qualifies if its termbase label is confirmed present in the actual publications text, rather than assumed present by sampling from the termbase itself.

## RAG system

```
src/lad/rag/
  schema.py              RetrievalHit, AttestedEquivalent, TerminologyRecord
  embeddings.py          LaBSE wrapper (GPU-aware, L2-normalized output)
  index.py               per-language FAISS index build + load/search
  lexical_enrichment.py  query expansion: termbases + WordNet + Arabic normalization
  lexical_index.py       BM25 index + Reciprocal Rank Fusion (--lexical)
  rerank.py              cross-encoder reranking (--rerank)
  generate_expand.py     generate-then-rank query expansion (--generator claude|jais2)
  retrieval.py           orchestrates enrichment -> search -> dedup -> rerank -> generate
  synthesis.py           Claude call -> structured TerminologyRecord
  prompts/               versioned prompt templates
  eval/
    gold_set.py                    main gold-standard set (120 terms, real termbase)
    lad_publications_gold_set.py   gold set independently confirmed in real LAD text
    metrics.py                     retrieval + synthesis metrics
    run_eval.py                    orchestrates gold set -> pipeline -> metrics
    report.py                      serializes results to JSON/CSV
```

`--lexical`, `--rerank`, and `--generator` are all opt-in and off by default on `lad query`/`lad eval`/`lad repl`. See `PROJECT_STATUS.md` for what each one measurably does to retrieval quality — reranking currently hurts results and generate-then-rank hasn't been validated live, so neither is a safe default yet; hybrid lexical retrieval is a clear win but still opt-in pending a synthesis-inclusive validation pass.

RAG imports (`torch`, `sentence-transformers`, `faiss`) are deferred inside individual CLI command bodies rather than loaded at startup, so commands that don't touch the RAG system — `lad status`, `lad validate`, and so on — stay fast.

## Cross-lingual lookup: WordNet

`pipeline/wordnet_lookup.py` wraps NLTK's WordNet and Open Multilingual Wordnet for on-demand English/French/Arabic synonym lookup. It isn't harvested into a JSONL file — WordNet is roughly 120K synsets, too large to export wholesale, so it's queried live instead.

Arabic OMW lemmas carry diacritics that plain-text queries won't match directly (the lemma for "museum" is `متْحف`, not `متحف`), so Arabic lookups go through a dediacritized reverse index built once at load time rather than NLTK's own language matching, which does exact-string comparison.

```python
from lad.pipeline.wordnet_lookup import lookup_synonyms
lookup_synonyms("museum", "en")  # {"en": ["museum"], "fr": ["musée"], "ar": ["متحف"]}
```

## Repository layout

```
config/sources.yaml          per-source config: URLs, auth, rate limits, license info
scripts/                     numbered stage runners (00-12)
src/lad/
  schema.py                  HeritageRecord, VocabularyTerm, Passage
  config.py                  loads sources.yaml, resolves env-based auth
  cli.py                     the `lad` command-line entrypoint
  connectors/
    base.py                  shared retry/rate-limit/checkpoint/rights-gate logic
    <source>.py               one file per source
  pipeline/
    rights.py                 rights_statement -> clear/restricted/unknown
    parquet_utils.py           shared JSONL -> Parquet helpers
    compact.py                 JSONL -> Parquet
    build_termbase.py           interim termbase builder
    build_termbase_from_kalcium.py  real termbase parser
    ingest_lad_publications.py   LAD Publications PDF ingestion
    passagize.py                 record chunking + Arabic normalization
    arabic_normalize.py          centralized Arabic text normalization
    wordnet_lookup.py             cross-lingual synonym lookup
    publish_hf.py                  builds the local HF Hub export
    stats.py                       stats.txt generator
  rag/                        the RAG system, see above
  storage/writer.py           shared JSONL append/checkpoint/raw-cache helpers
tests/                        one test file per connector/pipeline/RAG module
data/                         all gitignored except data/eval/ — regenerate via the scripts above
```

## Rights handling

Every record is classified by `pipeline/rights.py` from its actual rights-statement text, not merely by whether a rights field is present:

- **clear** — a recognized open license (CC0, CC BY, CC BY-SA, ODC-BY, public domain marks, the rightsstatements.org NoC family). Goes to `records.jsonl`.
- **restricted** — a recognized closed statement (the rightsstatements.org InC family, "all rights reserved"). Goes to `needs_review.jsonl`.
- **unknown** — anything else, including real restrictions like CC BY-NC/ND and sources with no machine-readable rights field at all (UNESDOC, WDL). Also goes to `needs_review.jsonl`.

An earlier version of this classifier treated any present rights field as clear, which misclassified in-copyright Europeana items as reusable. The current regex-based classifier and its test suite (`tests/test_rights.py`) exist to keep that from happening silently again.

## Tests

```bash
.venv/bin/pytest -q
```

One test file per connector/pipeline/RAG module, run against fixtures and stub clients — real API responses for the data layer, a deterministic fake embedder and a stub Anthropic client for the RAG layer. No network access required except one deliberate smoke test (`test_embeddings.py`) that loads the real LaBSE model. 155 tests as of this writing.

## Known limitations

See `PROJECT_STATUS.md` for the full, current list with measurements. Briefly:

- Arabic-as-target retrieval is bottlenecked by corpus size, not lexical enrichment — the Arabic passage index is a fraction the size of the English one.
- The main gold set is auto-sampled from the termbase rather than terminologist-validated.
- Cross-encoder reranking is built but measurably hurts results on the current corpus; kept opt-in.
- Generate-then-rank query expansion is built and unit-tested but has never been run against a real model.
- UNESDOC is excluded from the RAG index for now, to keep retrieval signal clean.
- No cross-source entity resolution between the termbase sources — safe by construction (extra recall, no false merges), but leaves some redundancy unresolved.
