import json

import pytest

from lad.rag.eval import gold_set
from lad.storage import writer


@pytest.fixture(autouse=True)
def _isolate_data(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gold_set, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gold_set, "GOLD_SET_PATH", tmp_path / "eval" / "gold_set.jsonl")


def _write_termbase(tmp_path, entries, filename="interim_termbase.jsonl"):
    path = tmp_path / "termbase" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_gold_set_excludes_single_language_entries_from_interim_fallback(tmp_path):
    _write_termbase(
        tmp_path,
        [
            {"term_id": "1", "pref_label": {"en": "only english"}, "subject_field": "museography"},
            {"term_id": "2", "pref_label": {"en": "gilding", "fr": "dorure"}, "subject_field": "materials_and_techniques"},
        ],
    )

    path = gold_set.build_gold_set(target_size=10)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["term_id"] == "2"
    assert rows[0]["termbase_source"] == "interim"


def test_gold_set_prefers_real_termbase_over_interim_when_both_exist(tmp_path):
    _write_termbase(
        tmp_path,
        [{"term_id": "interim:1", "pref_label": {"en": "only in interim", "fr": "x"}, "subject_field": "x"}],
        filename="interim_termbase.jsonl",
    )
    _write_termbase(
        tmp_path,
        [{"term_id": "kalcium:1", "pref_label": {"en": "gilding", "fr": "dorure", "ar": "تذهيب"}, "subject_field": "Sculpture and Carving"}],
        filename="real_termbase.jsonl",
    )

    path = gold_set.build_gold_set(target_size=10)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["term_id"] == "kalcium:1"
    assert rows[0]["termbase_source"] == "real"


def test_gold_set_source_language_rotates_round_robin(tmp_path):
    entries = [
        {"term_id": str(i), "pref_label": {"en": f"en{i}", "fr": f"fr{i}", "ar": f"ar{i}"}, "subject_field": "x"}
        for i in range(6)
    ]
    _write_termbase(tmp_path, entries, filename="real_termbase.jsonl")

    path = gold_set.build_gold_set(target_size=6)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    source_langs = [row["source_language"] for row in rows]
    # round-robin over 6 rows starting at en -> en, fr, ar, en, fr, ar
    assert source_langs == ["en", "fr", "ar", "en", "fr", "ar"]
    # every row's reference_equivalents covers exactly the two non-source languages
    for row in rows:
        assert set(row["reference_equivalents"]) == {"en", "fr", "ar"} - {row["source_language"]}


def test_gold_set_samples_across_subject_fields(tmp_path):
    entries = []
    for field in ["materials_and_techniques", "museography", "object_typology", "art_historical_period", "provenance"]:
        for i in range(20):
            entries.append(
                {
                    "term_id": f"{field}_{i}",
                    "pref_label": {"en": f"{field} term {i}", "fr": f"terme {field} {i}"},
                    "subject_field": field,
                }
            )
    _write_termbase(tmp_path, entries)

    path = gold_set.build_gold_set(target_size=10)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    fields_represented = {row["subject_field"] for row in rows}
    assert len(fields_represented) >= 4  # roughly even sampling across most/all buckets
    assert len(rows) <= 10


def test_gold_set_backfills_from_untagged_pool_when_tagged_entries_run_short(tmp_path):
    entries = [
        {"term_id": "tagged", "pref_label": {"en": "a", "fr": "b"}, "subject_field": "materials_and_techniques"},
    ] + [
        {"term_id": f"untagged_{i}", "pref_label": {"en": f"u{i}", "fr": f"v{i}"}, "subject_field": None}
        for i in range(10)
    ]
    _write_termbase(tmp_path, entries)

    path = gold_set.build_gold_set(target_size=5)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 5
    # the one tagged entry plus 4 backfilled untagged ones
    assert sum(1 for r in rows if r["subject_field"] is None) == 4
    assert sum(1 for r in rows if r["subject_field"] == "materials_and_techniques") == 1
