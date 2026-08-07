from pathlib import Path

from lad.pipeline.build_termbase_from_kalcium import (
    _iter_data_rows,
    _parse_kalcium_date,
    _split_labels,
    _term_from_row,
    build_real_termbase,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "kalcium_termbase" / "sample.xlsx"


def test_stray_header_row_is_filtered_out():
    rows = _iter_data_rows(FIXTURE_PATH)
    # sample.xlsx has 2 real data rows + 1 stray repeated-header row -- only
    # the 2 real ones should survive.
    assert len(rows) == 2


def test_full_concept_parses_all_three_languages():
    rows = _iter_data_rows(FIXTURE_PATH)
    row_537 = next(r for r in rows if r[0] == 537)
    term = _term_from_row(row_537)

    assert term.term_id == "kalcium:537"
    assert term.source_name == "lad_termbase_real"
    assert term.pref_label == {
        "ar": "الأشياء الجاهزة",
        "en": "Readymade",  # stripped of trailing whitespace
        "fr": "ready-made",  # stripped of leading whitespace
    }
    assert "Marcel Duchamp" in term.scope_note["en"]
    assert term.subject_field == "Sculpture and Carving"
    assert term.reuse_risk.value == "restricted"
    assert term.source_concept_ids == {"kalcium": "537"}


def test_broader_narrower_related_split_and_clean_br_joined_labels():
    rows = _iter_data_rows(FIXTURE_PATH)
    row_537 = next(r for r in rows if r[0] == 537)
    term = _term_from_row(row_537)

    # English related concepts: "Plastic arts<br />Sculptors<br />Visual arts"
    assert term.related_ids == [
        "الفنون التشكيلية", "النحاتون",  # Arabic block (stray \xa0-only entry dropped)
        "Plastic arts", "Sculptors", "Visual arts",  # English block
    ]
    assert term.narrower_ids == [
        "فن النحت الجديد", "فن النحت البريطاني الجديد",
        "New sculpture", "New British sculpture",
    ]


def test_partial_language_coverage_concept_has_only_populated_languages():
    rows = _iter_data_rows(FIXTURE_PATH)
    row = next(r for r in rows if r[0] == 9001)
    term = _term_from_row(row)

    assert term.pref_label == {"en": "Daguerreotype"}
    assert "ar" not in term.pref_label
    assert "fr" not in term.pref_label
    assert term.scope_note == {"en": "A photographic process."}


def test_split_labels_drops_empty_and_nbsp_only_entries():
    assert _split_labels("Plastic arts<br />Sculptors<br />\xa0") == ["Plastic arts", "Sculptors"]
    assert _split_labels(None) == []
    assert _split_labels("") == []
    assert _split_labels("Single label") == ["Single label"]


def test_parse_kalcium_date_handles_real_format_and_invalid_input():
    parsed = _parse_kalcium_date("2025.07.24. 02:10:38")
    assert parsed is not None
    assert parsed.isoformat() == "2025-07-24"
    assert _parse_kalcium_date(None) is None
    assert _parse_kalcium_date("not a date") is None


def test_build_real_termbase_writes_expected_entry_count(tmp_path):
    out_path = tmp_path / "real_termbase.jsonl"
    result_path = build_real_termbase(path=FIXTURE_PATH, out_path=out_path)
    assert result_path == out_path
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # stray header row excluded, both real concepts kept
