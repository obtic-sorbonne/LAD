# Reproducing the Experiments

This document gives exact commands to reproduce every experiment reported
in `PROJECT_STATUS.md`. It assumes the setup in the main `README.md` is
done (`./scripts/00_setup.sh`, then harvest/compact/build-termbase/
passagize/build-index have already been run once — see README.md's
"Pipeline stages" section for the full first-time sequence).

All commands below are run from the repo root. `--no-synthesis` is used
throughout because `ANTHROPIC_API_KEY` is not set in the environment this
was developed in — omit it if you have a key, to also get synthesis
metrics (equivalence correctness, attestation accuracy).

## 1. Real termbase integration (PROJECT_STATUS.md "Phase 1.5")

```bash
./scripts/03b_build_real_termbase.sh          # parse data/Export from Kalcium.xlsx
.venv/bin/python -c "from lad.rag.eval.gold_set import build_gold_set; build_gold_set()"
.venv/bin/lad eval --no-synthesis --run-label real_termbase \
  --output-json data/eval/results/real_termbase.json
```

To reproduce the *controlled A/B* specifically (real termbase merged in
vs. hidden, same gold set, same index): temporarily move
`data/termbase/real_termbase.jsonl` aside, clear
`lad.rag.lexical_enrichment._termbase_lookup`'s cache, rerun `lad eval`,
then restore the file. (This isolates the termbase's effect from the
corpus, which the CLI's `--gold-set-path`/`--index-model-name` flags don't
cover on their own since the termbase isn't an index/gold-set variant —
there's no single-flag way to do this one; it's a small Python snippet,
see PROJECT_STATUS.md's Phase 1.5 section for the exact methodology.)

## 2. Cross-encoder reranking (Phase 1.6)

```bash
.venv/bin/lad eval --no-synthesis --run-label baseline_no_rerank \
  --output-json data/eval/results/reranker_off.json
.venv/bin/lad eval --no-synthesis --rerank --run-label with_rerank \
  --output-json data/eval/results/reranker_on.json
```

Compare `summary.retrieval` in both JSON files, or diff the
`retrieval_raw` CSVs row-by-row for the per-term breakdown.

## 3. Cross-lingual fallback bug fix (Phase 1.7)

This was a code fix, not a toggle — there's no "before" state to rerun
against without reverting `rag/lexical_enrichment.py`'s `expand_query()`.
To reproduce the *diagnostic* that found it (how many real terms get zero
query variants in a target language):

```bash
.venv/bin/python -c "
import random
from lad.rag.lexical_enrichment import _termbase_expansion, _wordnet_expansion
from lad.storage.writer import read_jsonl, PROCESSED_DIR

rows = read_jsonl(PROCESSED_DIR / 'getty_aat' / 'records.jsonl')
en_terms = [r['pref_label']['en'] for r in rows if r.get('pref_label', {}).get('en')]
random.seed(42)
sample = random.sample(en_terms, 500)
zero_ar = sum(1 for t in sample if not (_termbase_expansion(t,'en').get('ar',set()) | _wordnet_expansion(t,'en').get('ar',set())))
print(f'{zero_ar}/{len(sample)} sampled terms have zero Arabic query variants')
"
```

## 4. tRAG generate-then-rank (Phase 2)

Built and unit-tested, but not run live in this environment (no
`ANTHROPIC_API_KEY`, no Jais-2 HF token). Once you have one or both:

```bash
export ANTHROPIC_API_KEY=...          # for --generator claude
# or accept the license at huggingface.co/inceptionai/Jais-2-8B-Chat
# and `huggingface-cli login`          # for --generator jais2

.venv/bin/lad eval --no-synthesis --generator claude \
  --run-label generate_then_rank_claude \
  --output-json data/eval/results/generate_then_rank_claude.json
```

## 5. LAD Publications: real institutional documentation (Phase 4)

```bash
# One-time: ingest and index the real publications
./scripts/11_ingest_lad_publications.sh
./scripts/12_build_lad_publications_gold_set.sh
.venv/bin/python -c "
from lad.rag.embeddings import Embedder
from lad.rag.index import build_index, MUSEUM_SOURCES
embedder = Embedder()
build_index(sources=MUSEUM_SOURCES + ['lad_publications'],
            model_name='labse-plus-lad-publications', embedder=embedder)
"

# The four-way comparison reported in PROJECT_STATUS.md "Phase 4":
.venv/bin/lad eval --no-synthesis --run-label main_gold_set__baseline \
  --output-json data/eval/results/main_gold_set__baseline.json \
  --output-csv-dir data/eval/results/main_gold_set__baseline

.venv/bin/lad eval --no-synthesis --run-label main_gold_set__with_lad_publications \
  --index-model-name labse-plus-lad-publications \
  --output-json data/eval/results/main_gold_set__with_lad_publications.json \
  --output-csv-dir data/eval/results/main_gold_set__with_lad_publications

.venv/bin/lad eval --no-synthesis --run-label pubs_gold_set__baseline \
  --gold-set-path data/eval/lad_publications_gold_set.jsonl \
  --output-json data/eval/results/pubs_gold_set__baseline.json \
  --output-csv-dir data/eval/results/pubs_gold_set__baseline

.venv/bin/lad eval --no-synthesis --run-label pubs_gold_set__with_lad_publications \
  --gold-set-path data/eval/lad_publications_gold_set.jsonl \
  --index-model-name labse-plus-lad-publications \
  --output-json data/eval/results/pubs_gold_set__with_lad_publications.json \
  --output-csv-dir data/eval/results/pubs_gold_set__with_lad_publications
```

The 635-term publications gold set takes ~2 minutes to run (vs. ~40s for
the 120-term main set) — retrieval-only, no synthesis calls involved.

**Reproducing the short-passage diagnostic** (why the A/B above shows zero
effect):

```bash
.venv/bin/python -c "
import json
total = short = 0
for src in ['getty_aat','unesco_thesaurus','europeana','world_digital_library']:
    for l in open(f'data/passages/{src}.jsonl', encoding='utf-8'):
        r = json.loads(l); total += 1
        if len(r['text'].split()) <= 3: short += 1
print(f'{short}/{total} ({100*short/total:.0f}%) museum-subset passages are <=3 tokens')
"
```

## Raw results

Every run above, if given `--output-json`/`--output-csv-dir`, writes to
`data/eval/results/<run_label>.json` (full: summary + every per-row
result) and `data/eval/results/<run_label>/retrieval_raw.csv` (+
`synthesis_raw.csv` if synthesis ran). These are the actual raw evaluation
result artifacts — not hand-transcribed summaries.
