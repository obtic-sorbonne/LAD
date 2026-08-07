# LAD — Multilingual Heritage Data Pipeline + RAG System

Collects and prepares Arabic/French/English cultural-heritage text and
terminology, and — Phase 1, built and verified against real data — a
term-centric multilingual RAG system for museum terminology discovery on
top of it: lexical enrichment, cross-lingual retrieval (FAISS + LaBSE),
and structured, attested-equivalents-only synthesis (Claude). Local
pipeline, no database server: everything is JSONL/Parquet/FAISS on disk.

Full design rationale (why each source/schema/decision is what it is) lives
in the project plan; this README documents what the pipeline *is* and how
to run it.

---

## Setup

```bash
./scripts/00_setup.sh
```

Creates `.venv` (with **Python 3.11 specifically**, not whatever `python3`
resolves to — see below), installs dependencies including a GPU-compatible
torch build, and downloads the WordNet + Open Multilingual Wordnet data NLTK
needs for cross-lingual synonym lookup. Every other script calls this
automatically if `.venv` is missing, so you rarely need to run it directly.

**Why Python 3.11 specifically:** verified live on a 4x NVIDIA
A40 box (driver 550.127.05, CUDA 12.4) — PyTorch's official CUDA wheels for
Python 3.14 are only built against CUDA >=12.6, which this driver can't
satisfy at all (no compatible GPU wheel exists for 3.14, only CPU-only).
Python 3.11 has cu118/cu121/cu124 wheels, which do match. If `00_setup.sh`
can't find `python3.11` on your machine it falls back to generic `python3`
with a CPU-only torch — slower for embedding generation, everything still
works.

**API keys**, same pattern for both — a source/feature is skipped or
disabled without one, nothing crashes:
- Europeana harvesting: free key at europeana.eu, paste into
  `scripts/01_harvest.sh`'s `EUROPEANA_API_KEY` line (or export it / `.env`).
- RAG synthesis (Stage 8/9's Claude calls): key at console.anthropic.com,
  export `ANTHROPIC_API_KEY` or put it in `.env`. Not set in this repo's
  dev environment — retrieval-only (`--no-synthesis`) works without it.

---

## Pipeline stages

Each stage is a numbered script in `scripts/`, runnable standalone or as
part of the sequence. Run them in order for a first pass:

```bash
./scripts/00_setup.sh                    # environment + WordNet data
./scripts/01_harvest.sh                  # pull data from every source
./scripts/02_compact.sh                  # JSONL -> partitioned Parquet
./scripts/03_build_termbase.sh           # derive the interim termbase
./scripts/04_passagize.sh                # chunk records into RAG passages
./scripts/05_stats.sh                    # regenerate stats.txt
./scripts/06_push_to_hf.sh               # push a Parquet export to Hugging Face Hub (needs HF auth, see below)
./scripts/07_build_index.sh              # build the FAISS passage index (Phase 1: RAG)
./scripts/08_query.sh "term" en          # query one term end to end (Phase 1: RAG)
./scripts/09_eval.sh                     # evaluate against the interim gold set (Phase 1: RAG)
./scripts/10_experiment.sh               # interactive live-query REPL (Phase 1: RAG)
```

Stages 2-10 all read from Stage 1's output independently — they're parallel
branches, not a strict chain. Only re-run one when its specific inputs have
changed (e.g. `04_passagize.sh` after a fresh harvest, `07_build_index.sh`
after `04_passagize.sh` changes the museum-subset passage counts).

### Stage 1 — Harvest (`01_harvest.sh`)

Pulls from five sources via `src/lad/connectors/`, each implementing a
shared `discover → fetch_page → parse → normalize` interface
(`connectors/base.py`) that handles retry/backoff, rate limiting, raw-
response caching, checkpointed resume, rights classification, and JSONL
output uniformly.

Pagination stops on either an empty page (200, zero items) **or** an HTTP
404 — some paginated APIs (e.g. loc.gov) 404 a page number past the true
last page instead of returning an empty 200; both are treated as "reached
the end," not an error, so a real end-of-collection isn't logged as a
scary traceback with `errors += 1`. A non-404 HTTP error still counts as a
real failure. Covered by `tests/test_connector_base.py`.

| Source | Kind | Access | Notes |
|---|---|---|---|
| UNESCO Thesaurus | vocabulary | one-shot SKOS/Turtle download | 100% trilingual (ar/en/fr), 4,499 concepts |
| Getty AAT | vocabulary | SPARQL, paginated per facet | EN/FR only — AAT has no Arabic labels at all (verified) |
| UNESDOC | records | bulk JSON download | UNESCO Digital Library catalogue, 285,433 records |
| World Digital Library | records | loc.gov JSON API, paginated | Library of Congress; grows over time (live collection) |
| Europeana | records | REST search, paginated | Needs API key; query filtered to French only (Europeana has zero Arabic content — verified) |

```bash
./scripts/01_harvest.sh                              # all sources, PAGE_LIMIT pages each
./scripts/01_harvest.sh --source europeana            # one source
./scripts/01_harvest.sh --source europeana --clean --refresh   # wipe + full re-harvest
```

`PAGE_LIMIT` (env var or edit the script, default 5000) caps pages fetched
*this run* — re-running resumes from checkpoint rather than restarting, so
raising the cap or removing it (`PAGE_LIMIT=""`) is safe to do incrementally.
`--clean` wipes a source's prior output first — use when a source's query
config changed and old output would otherwise mix two different scopes.

**Known limitation:** Europeana's free-tier `start` parameter caps around
1000 results (`400 Bad Request` past that offset) — full-corpus Europeana
harvesting needs cursor-based pagination (`cursor=*`), not yet implemented.
Harvesting stops cleanly at that point rather than crashing; not something
this pass fixed.

### Stage 2 — Compact (`02_compact.sh`)

Converts finalized JSONL into partitioned Parquet
(`data/processed_parquet/source=<x>/date=<y>/*.parquet`) for querying with
DuckDB or similar, without running a database server. Pure repackaging —
same rows, same fields, no filtering or transformation.

Dict-valued fields (`pref_label`, `alt_labels`, `scope_note`,
`source_concept_ids` on `VocabularyTerm`) are JSON-encoded to strings before
the Arrow conversion, not left as native structs: pyarrow infers a Parquet
struct schema from the data, and when a dict field is `{}` on every row of
a source (Getty AAT never populates `alt_labels`/`scope_note`/
`source_concept_ids` at all), it infers a zero-field struct, which Parquet's
writer can't serialize (`ArrowNotImplementedError: Cannot write struct type
with no child field`) — hit live, fixed generically rather than
special-cased for that one field. Covered by `tests/test_compact.py`.

### Stage 3 — Termbase (`03_build_termbase.sh`)

Builds `data/termbase/interim_termbase.jsonl` — **not a new fetch**, a
transform of already-harvested UNESCO Thesaurus + Getty AAT data.

We don't have the real LAD Termbase (Louvre Abu Dhabi's curated ~1,000-term
trilingual resource); this is an explicit, documented stand-in:
- UNESCO Thesaurus concepts filtered to heritage/museum relevance by keyword
  match, classified into one of 5 subject fields (`materials_and_techniques`
  / `museography` / `object_typology` / `art_historical_period` /
  `provenance`) via a simple, inspectable keyword-priority heuristic (see
  `pipeline/build_termbase.py` — not a black-box classifier).
- Getty AAT concepts taken as-is (already scoped to relevant facets by the
  connector itself).
- Every entry's `source_name` is `lad_termbase_interim` — that field value
  *is* the substitute-vs-real tag. `source_concept_ids` traces each entry
  back to its origin vocabulary + concept ID.
- Not deduplicated/cross-linked between the two source vocabularies (needs
  embedding-based entity resolution — that's RAG-system work, not this
  stage).

### Stage 3b — Build the REAL termbase (`03b_build_real_termbase.sh`)

```bash
./scripts/03b_build_real_termbase.sh
```

Parses the real Louvre Abu Dhabi Termbase -- `data/Export from Kalcium.xlsx`,
an institutional Kalcium TMS export (not fetched by any connector; must
already be present) -- into `data/termbase/real_termbase.jsonl`. 4,652
concepts, 100% Arabic label coverage (verified), vs. Stage 3's interim
substitute where 53,200/53,355 entries have zero Arabic (Getty-AAT-derived,
and AAT has no Arabic labels at all).

The export's layout (verified against the live file): two header rows,
then one row per concept -- not one row per concept+language, not one row
per synonym. 74 columns: 11 concept-level fields + three 21-column
language blocks (ar/en/fr), each holding exactly one term per language per
concept (no concept ID repeats across rows in the real data -- this
termbase doesn't carry multiple synonyms per language the way
`alt_labels` could hold them; that enrichment still comes from Getty
AAT/WordNet). One stray repeated header row appears mid-sheet and is
filtered by requiring an int concept ID, not by row position. Broader/
narrower/related concepts are exported as `<br/>`-joined display labels,
not concept IDs -- stored as label strings despite the field name (still
useful lexical-enrichment signal, just not resolvable back to a concept
row). See `pipeline/build_termbase_from_kalcium.py`'s module docstring
for the full column-offset reference. Covered by
`tests/test_build_termbase_from_kalcium.py` (7 tests against a synthetic
fixture mirroring the real layout, `tests/fixtures/kalcium_termbase/`).

Output rows use `source_name="lad_termbase_real"` and
`reuse_risk="restricted"` (institutional, not public) -- deliberately
**not** wired into `publish_hf.py`'s public HF export, which still only
reads the interim substitute.

`rag/lexical_enrichment.py` merges this with the interim substitute (real
termbase first, interim as a broader-coverage fallback) -- a term
matching in both simply gets candidates pooled from each. `rag/eval/
gold_set.py` now prefers this real termbase for building the gold-standard
eval set too (see Stage 9).

### Stage 4 — Passagize (`04_passagize.sh`)

Chunks harvested records into `data/passages/<source>.jsonl` — the actual
retrieval unit a RAG system indexes, not the records themselves. 150–200
whitespace-token chunks with 20-token overlap, real character offsets
(regex-span based, not reconstructed from a lossy split).

Draws from **both** `records.jsonl` and `needs_review.jsonl` (decided: a v1
corpus needs volume — restricting to rights-clear-only shrinks the corpus by
~98%). Every passage carries its parent record's `rights_statement`/
`reuse_risk` forward, denormalized at chunk time, so this is enforceable
downstream without a join — a passage from an unknown-rights record is never
silently presented as equivalent to a confirmed-open one.

Two source shapes need different chunking (see `schema.py`):
- **HeritageRecord** rows (Europeana/UNESDOC/WDL): one language per row,
  `title` and `description` chunked separately.
- **VocabularyTerm** rows (UNESCO Thesaurus/Getty AAT): all languages on one
  row, chunked per language from `scope_note` (falling back to `pref_label`
  when there's no scope note).

Arabic passages are normalized (`pipeline/arabic_normalize.py` — diacritic
stripping + alef/teh-marbuta variant folding via `camel_tools`, no
pretrained-model download needed) for the indexed `text` field;
`text_raw` keeps the original for citation/display. Only applied when
`language_code` unambiguously identifies as Arabic (`ar`/`ara`/`arabic`) —
UNESDOC's compound codes like `eng,ara,rus` are left as-is since the row is
genuinely multi-language.

### Stage 5 — Stats (`05_stats.sh`)

Regenerates `stats.txt` from whatever's currently in `data/`. Not a
hand-maintained snapshot — always reflects current state, so it never goes
stale the way a committed report would.

### Stage 6 — Push to Hugging Face Hub (`06_push_to_hf.sh`)

Builds a local Parquet export (`data/hf_export/`, via `lad build-hf-export`)
and pushes it to the private dataset repo
`SorbonneUniversity/LAD-Collected-Dataset`. Two independent steps:

1. **Export build** (`pipeline/publish_hf.py`) — no network access, no HF
   credentials touched. Converts harvested records + termbase + passages to
   Parquet (reusing `pipeline/parquet_utils.py`'s dict-field JSON-encoding —
   the same fix Stage 2 needed, since termbase rows have the identical
   all-empty-dict shape for most Getty AAT-derived entries) and writes a
   dataset card (`README.md`) documenting per-source licensing and, most
   importantly, the **rights composition of the export**: every row keeps
   its `reuse_risk` field, and the card states plainly what fraction of
   records are confirmed-open vs. unverified/restricted — this was a
   deliberate decision (include everything, labeled, not just rights-clear
   content) rather than silently dropping 84% of the harvest.
2. **Push** (inline in the script, via `huggingface_hub`) — requires auth
   already set up: `huggingface-cli login`, or `export HF_TOKEN=...` (get
   one at huggingface.co/settings/tokens). Not handled by this repo's code
   at all — no token is read from or written to any file here.

```bash
huggingface-cli login   # or: export HF_TOKEN=hf_...
./scripts/06_push_to_hf.sh
```

---

## Phase 1 — RAG system (`src/lad/rag/`)

A term-centric multilingual RAG pipeline on top of the passages from
Stage 4: give it a term in Arabic, French, or English, and it returns
attested cross-lingual equivalents grounded in retrieved passages, not
generated freely. Combines ideas from three sources (full rationale in the
project plan): the LAD paper's 4-component architecture (corpus → lexical
enrichment → cross-lingual retrieval → synthesis), tRAG's term-as-
retrieval-unit framing, and a Spanish legal-RAG paper's practical
terminology-driven query expansion pattern.

**Scope decision:** first index covers the **museum-specific
subset** (Getty AAT + UNESCO Thesaurus + Europeana + WDL) — UNESDOC's
370,990 administrative/policy passages are excluded for now to keep
retrieval-quality signal clean (added back in a later phase). Lexical
enrichment is **static-only** (termbase + WordNet lookup + Arabic
morphological normalization) — the tRAG-style dynamic generate-then-rank
step and cross-encoder reranking are explicitly deferred, not part of this
first pass.

**Verified live** (not just unit-tested): built the real
index (78,679 passages after language filtering, ~90s on GPU), ran real
queries — an English "museum" query correctly cross-lingually retrieved
Arabic museum passages (متاحف, عماره المتاحف, ...) and French ones (musée);
an Arabic "متحف" query retrieved the same cluster from the other
direction. Ran the full interim eval set (45 terms): P@5=0.507,
R@10=0.803, MRR=0.795 (retrieval-only; synthesis wasn't run live since no
`ANTHROPIC_API_KEY` is set in this environment — the code path is unit
tested against a stub client instead).

### Stage 7 — Build index (`07_build_index.sh`)

```bash
./scripts/07_build_index.sh                                    # museum subset (default)
./scripts/07_build_index.sh --sources getty_aat,unesco_thesaurus  # custom subset
```

Builds **one FAISS index per language** (`ar`/`en`/`fr`), not one mixed
index and not one per source (`rag/index.py`'s docstring has the full
reasoning) — the LAD paper's retrieval spec is "top-k passages *per
language*", which a single mixed index can't guarantee since FAISS returns
globally-nearest neighbors regardless of language. Passages whose language
isn't ar/en/fr (WDL has plenty — Russian, Italian, etc.) are excluded from
the index entirely.

Embeddings: `sentence-transformers/LaBSE`, GPU if available (falls back to
CPU automatically). Output: `data/embeddings/labse/index_{ar,en,fr}.faiss`
+ `meta_{ar,en,fr}.jsonl` (passage metadata including `rights_statement`/
`reuse_risk`, aligned to FAISS IDs).

### Stage 8 — Query (`08_query.sh`)

```bash
./scripts/08_query.sh "manuscript" en
./scripts/08_query.sh --no-synthesis "gilding" en   # retrieval only, no Claude call/API key needed
```

Runs one term through the full pipeline:
1. **Lexical enrichment** (`rag/lexical_enrichment.py`) — expands the query
   using the interim termbase + `pipeline/wordnet_lookup.py` cross-lingual
   synonyms + `pipeline/arabic_normalize.py` for Arabic variants. Reuses
   Phase 0 modules directly rather than reimplementing lookup logic.
2. **Retrieval** (`rag/retrieval.py`) — encodes every expanded query variant,
   searches each language's index, deduplicates hits across variants
   (same passage matched by two variants keeps its best score).
3. **Synthesis** (`rag/synthesis.py`) — Claude call with a structured,
   attested-equivalents-only prompt (`rag/prompts/synthesis.md`); output is
   a `TerminologyRecord` (source term, per-language equivalents with
   citing `passage_id`s, a usage note, and a `rights_caveat` if any cited
   passage's rights aren't confirmed-clear — never silently presented as
   equivalent to a clear source).

### Stage 9 — Eval (`09_eval.sh`)

```bash
./scripts/09_eval.sh
./scripts/09_eval.sh --no-synthesis   # retrieval metrics only
```

Builds the gold-standard set (`rag/eval/gold_set.py` — 120 terms, matching
the LAD paper's real gold-set size) if it doesn't exist yet, runs every
term through the full pipeline, and prints retrieval metrics (P@5, R@10,
MRR) and, unless `--no-synthesis`, synthesis metrics (equivalence
correctness, attestation accuracy — see `rag/eval/metrics.py`). Still
auto-sampled, not terminologist-validated — numbers are a smoke-test
signal for iteration, not comparable to the paper's reported figures.

As of Phase 1.5 (see PROJECT_STATUS.md), the gold set is drawn from the
**real** termbase (`data/termbase/real_termbase.jsonl`, falling back to
the interim substitute only if that doesn't exist) and rotates source
language round-robin across en/fr/ar, so it now exercises all three of
the paper's eval directions — previously the gold set always defaulted to
English-as-source, so AR-as-source and FR-as-source retrieval were never
tested regardless of what this script did with the rows.

**Known real result worth knowing before trusting the blended average**:
retrieval quality is not uniform across languages or directions. A
controlled A/B (real termbase merged in vs. hidden, same gold set, same
passage index — see PROJECT_STATUS.md's "Phase 1.5" section for the full
table) found the real termbase roughly **doubles blended P@5** (0.018 →
0.033), driven by a **5-7x jump in Arabic-as-source retrieval**
(ar→en 0.005→0.037, ar→fr 0.010→0.053) — but Arabic-as-**target**
retrieval stayed flat (en→ar, fr→ar unchanged), because that depends on
the Arabic passage index itself (still 5,038 vs. 59,745 English passages),
which a termbase fix doesn't touch. Mirrors the real LAD paper's own
central finding (Arabic underperforms EN/FR) and narrows down *which*
part of the pipeline (corpus size, not lexical enrichment) still needs to
close that gap.

### Stage 10 — Live experiment REPL (`10_experiment.sh`)

```bash
./scripts/10_experiment.sh                  # interactive, needs ANTHROPIC_API_KEY for synthesis
./scripts/10_experiment.sh --no-synthesis   # retrieval only, no key needed
```

`08_query.sh` reloads LaBSE + the full index on every single call (~90s+,
mostly index metadata loading — `meta_en.jsonl` alone is 36MB). This loads
everything **once**, then drops into a loop prompting for `term>` and
`lang (ar/en/fr)>`, printing retrieval hits (and, unless `--no-synthesis`,
the full synthesis result) after each — much better suited to actually
iterating on queries. `quit`/`exit`/`q` or Ctrl-D to leave.

### Stage 11 — Ingest LAD Publications (`11_ingest_lad_publications.sh`)

```bash
./scripts/11_ingest_lad_publications.sh
./scripts/11_ingest_lad_publications.sh --source-dir "/path/to/pdfs"
```

Extracts and passagizes the real LAD Publications PDFs (5 real Louvre Abu
Dhabi publications provided directly, not fetched by any connector -- must
already exist at the source dir, default `data/raw_pdfs/lad_publications`
relative to the repo root). See `pipeline/ingest_lad_publications.py`'s module
docstring and PROJECT_STATUS.md's "Phase 4" section for the full story --
short version: this is the first REAL institutional documentation this
project has indexed, direct `pypdf` text extraction (no OCR needed, all 5
PDFs verified to have real text layers), reusing the standard
`passagize_source()` pipeline unchanged since the ingestion output is
plain `HeritageRecord` rows.

### Stage 12 — Build the LAD Publications gold set (`12_build_lad_publications_gold_set.sh`)

```bash
./scripts/12_build_lad_publications_gold_set.sh
```

Builds a gold set from the real termbase + the real LAD Publications text
together (`rag/eval/lad_publications_gold_set.py`) -- unlike the main gold
set (auto-sampled *from* the termbase, so a "translation" trivially exists
for every entry by construction), this one only includes an entry if its
termbase label is independently confirmed attested (substring match) in
the actual publications text, per language. Run `lad eval --gold-set-path
data/eval/lad_publications_gold_set.jsonl [--index-model-name
labse-plus-lad-publications]` to evaluate against it.

### RAG package layout

```
src/lad/rag/
  schema.py               RetrievalHit, AttestedEquivalent, TerminologyRecord
  embeddings.py            LaBSE wrapper (GPU-aware, L2-normalized output)
  index.py                 per-language FAISS index build + load/search
  lexical_enrichment.py     query expansion: termbases + WordNet + Arabic
                            normalization + bare-term cross-lingual fallback
                            (Phase 1.7 -- see PROJECT_STATUS.md)
  rerank.py                 cross-encoder reranking (Phase 1.6, opt-in --rerank,
                            measured to hurt on the current eval -- see docs)
  generate_expand.py         tRAG generate-then-rank (Phase 2, opt-in
                            --generator claude|jais2, unit-tested only --
                            no live credential in this environment)
  lexical_index.py           BM25 + Reciprocal Rank Fusion (Phase 7, opt-in
                            --lexical -- the fix that actually solved the
                            short-passage problem: +83%/+157% P@5)
  retrieval.py              query encode -> per-language FAISS search -> dedup
                            -> optional rerank -> optional LLM-generated augment
  synthesis.py               Claude call -> structured TerminologyRecord
  prompts/
    synthesis.md               the synthesis prompt template (versioned, not inline)
    generate_variants.md        the generate-then-rank candidate-generation prompt
  eval/
    gold_set.py               main gold-standard set builder (real termbase, 120 terms)
    lad_publications_gold_set.py  gold set from terms independently confirmed
                                  attested in the real LAD Publications text (Phase 4)
    metrics.py                 P@k/R@k/MRR/semantic_relevance_at_k + attestation/equivalence metrics
    run_eval.py                 orchestrates gold set -> pipeline -> metrics,
                                returns raw per-row results, not just an average
    report.py                   serializes eval results to JSON/CSV (Phase 4)
```

All `lad.rag.*` imports are deferred inside CLI command bodies, not hoisted
to the top of `cli.py` — they pull in torch/sentence-transformers/faiss,
which cost real time and memory to import; verified live that `lad status`
and other non-RAG commands never load them.

---

## Cross-lingual lookup: WordNet (not a harvest stage)

`pipeline/wordnet_lookup.py` wraps NLTK's WordNet + Open Multilingual
Wordnet for on-demand EN/FR/AR synonym lookup — used by the termbase
builder and later RAG query enrichment, **not** harvested into a JSONL file
(WordNet is ~120K synsets, too large to export wholesale; looked up live
instead).

Arabic OMW lemmas carry diacritics that plain-text input won't match
directly (e.g. the lemma for "museum" is `متْحف`, not `متحف`) — Arabic
lookups go through a dediacritized reverse index built once via
`arabic_normalize.dediacritize`, not NLTK's own `lang=` parameter (which
does exact-string matching and misses undiacritized queries — verified).

```python
from lad.pipeline.wordnet_lookup import lookup_synonyms
lookup_synonyms("museum", "en")  # {"en": ["museum"], "fr": ["musée"], "ar": ["متحف"]}
```

---

## Repo layout

```
config/sources.yaml          per-source config: URLs, auth, rate limits, license info
scripts/                     numbered stage runners (00-10, see above)
src/lad/
  schema.py                  HeritageRecord, VocabularyTerm, Passage -- all ProvenanceFields-based
  config.py                  loads sources.yaml, resolves env-based auth
  cli.py                     `lad harvest/compact/build-termbase/build-real-termbase/passagize/build-hf-export/stats/validate/status/build-index/query/eval/repl`
  connectors/
    base.py                  shared Connector: retry, rate limit, checkpoint, rights gate, JSONL output
    <source>.py               one file per source, ~100-200 lines each
  pipeline/
    rights.py                classifies rights_statement -> clear/restricted/unknown (see below)
    parquet_utils.py          shared JSONL->Parquet helpers (dict-field encoding fix, see Stage 2)
    compact.py                JSONL -> Parquet
    build_termbase.py         interim termbase builder (public-data substitute)
    build_termbase_from_kalcium.py  REAL termbase parser (Kalcium export, see Stage 3b)
    ingest_lad_publications.py  REAL LAD Publications PDF ingestion (see Stage 11)
    passagize.py               record chunking + Arabic normalization hookup
    arabic_normalize.py        centralized Arabic text normalization (camel_tools)
    wordnet_lookup.py          cross-lingual synonym lookup (NLTK + OMW)
    publish_hf.py               builds the local HF Hub export (no network/credentials)
    stats.py                   stats.txt generator
  rag/                        Phase 1 RAG system -- see "Phase 1" section above
  storage/writer.py           JSONL append/checkpoint/raw-cache helpers shared by everything
tests/                        one test file per connector/pipeline module, fixtures from real API responses
data/
  raw/<source>/                cached raw API responses + checkpoint.json (gitignored)
  processed/<source>/          records.jsonl (rights-clear) + needs_review.jsonl (gitignored)
  processed_parquet/            compacted output (gitignored)
  termbase/interim_termbase.jsonl   public-data substitute (gitignored)
  termbase/real_termbase.jsonl      REAL termbase, from Export from Kalcium.xlsx (gitignored)
  passages/<source>.jsonl       (gitignored)
  embeddings/labse/             FAISS indices + metadata sidecars, per language (gitignored)
  eval/gold_set.jsonl            interim gold-standard set (gitignored)
  hf_export/                    local HF Hub export staging (gitignored)
  logs/run_summary.jsonl        (gitignored)
```

Data directories are gitignored (regenerate by running the scripts) — only
code, config, tests, and fixtures are tracked.

---

## Rights gate

Every record gets classified by `pipeline/rights.py` from its actual rights
statement text, not just "does a rights field exist":
- **clear**: recognized open licenses (CC0, CC BY, CC BY-SA, ODC-BY, public
  domain marks, rightsstatements.org's NoC family) — goes to `records.jsonl`
- **restricted**: recognized closed statements (rightsstatements.org's InC
  family, "all rights reserved") — goes to `needs_review.jsonl`
- **unknown**: anything else, including CC BY-NC/ND (real restrictions, not
  guessed as open) and sources with no machine-readable rights field at all
  (UNESDOC, WDL) — also goes to `needs_review.jsonl`

This classification was wrong once during development (a version that
treated "any rights field present" as clear, misclassifying Europeana's
In-Copyright items) — the regex-based approach in `rights.py` and its test
suite (`tests/test_rights.py`) exist specifically to keep that from
recurring silently.

---

## Tests

```bash
.venv/bin/pytest -q
```

One file per connector/pipeline/RAG module, run against fixtures/stub
clients (real API responses for data-layer tests, a deterministic fake
embedder + stub Anthropic client for RAG tests) — no network access needed
except the one deliberate LaBSE smoke test (`test_embeddings.py`), which
loads the real model. 154 tests as of this pass.

---

## Known simplifications / limitations (documented, not silent)

- **75% of the museum-subset FAISS index (72,112/96,552 passages) is <=3-token
  bare labels, not real passages** -- found in Phase 4 (see PROJECT_STATUS.md),
  the single largest known retrieval-quality limitation right now. Getty AAT
  has 0% `scope_note` coverage in English or French, so `passagize.py`'s
  documented scope_note-with-pref_label-fallback rule falls back to the bare
  label for literally every one of its 60,436 passages, many duplicated many
  times over. These appear to embed into a degenerate region of LaBSE's
  vector space and can dominate top-ranked results regardless of actual
  relevance -- demonstrated directly, not inferred: the top-5 results for the
  French query "dorure" (gilding) were five *different* Getty AAT entries all
  reading "tuile" (tile), an unrelated word, all scoring identically.
  **Two candidate fixes tried (Phase 5, `rag/passage_quality.py`) and both
  failed**: filtering short/label-only passages out of the index is a clear
  regression (removes far more correct signal than noise -- Getty AAT's short
  labels are frequently the *correct* answer for simple terms, not just
  noise); deduplicating exact-duplicate passages is measurably neutral
  (duplicate count wasn't the actual mechanism). A newer embedding model
  (Phase 6, `multilingual-e5-large`) was only a marginal +2-7% gain. **What
  actually worked (Phase 7, `rag/lexical_index.py`)**: fusing dense retrieval
  with BM25 lexical search via Reciprocal Rank Fusion -- +83% P@5 on the main
  gold set, +157% P@5 on the publications gold set. Opt-in via `--lexical`,
  not yet the default -- see PROJECT_STATUS.md's Next Steps item 0.
- **Europeana** free-tier pagination caps around `start=1000`; full-corpus
  harvesting needs the API's cursor-based pagination, not implemented.
- **UNESDOC and WDL** don't expose per-record machine-readable rights in
  their APIs, so their records are 100% `needs_review` by design, not error.
- **Getty AAT** connector pulls `skos:prefLabel` only (en/fr) — no
  `altLabel`/broader/narrower, to avoid an expensive per-concept N+1 query
  pattern against a shared public SPARQL endpoint.
- **Interim termbase** is a keyword-heuristic filter over UNESCO Thesaurus +
  Getty AAT, not the real Louvre Abu Dhabi Termbase — as of Phase 1.5, the
  real termbase (4,652 entries, parsed from `data/Export from
  Kalcium.xlsx`) is integrated and merged into lexical enrichment
  alongside this substitute; see Stage 3b and PROJECT_STATUS.md's "Phase
  1.5" section for what changed and its measured effect.
- **No cross-source entity resolution** across any of the three termbase
  sources (UNESCO Thesaurus, Getty AAT, and now the real Kalcium termbase
  — concepts for the same real-world thing aren't linked across them) —
  needs embedding-based matching, out of scope for this pass. Currently
  safe by construction: unlinked entries just contribute independent
  lexical-enrichment candidates rather than being incorrectly merged.
- **Institutional documentation** (actual Louvre/museum catalogs, object
  labels, curatorial text) isn't sourced at all — no public equivalent at
  comparable quality; requires the Louvre partnership.
- **RAG Phase 1 is retrieval + static lexical enrichment only** — no
  tRAG-style dynamic generate-then-rank for terms the termbase/WordNet miss
  (expected to disproportionately affect Arabic), no cross-encoder
  reranking, no comparison harness (single embedding model, single LLM).
  All explicitly deferred to later phases, not silently skipped.
- **UNESDOC is excluded from the RAG index** for now (370,990 passages,
  mostly administrative/policy text) to keep early retrieval-quality signal
  clean — a deliberate scope decision, not an oversight.
- **Gold-standard eval set** (`rag/eval/gold_set.py`) is auto-sampled, not
  terminologist-validated — metrics from it are a relative/smoke-test
  signal, not comparable to the LAD paper's reported numbers on its real,
  human-validated 120-entry set. It's now the same *size* (120) and drawn
  from the *real* termbase (Phase 1.5), which closes part of the gap, but
  auto-sampling vs. human validation is still a different thing.
