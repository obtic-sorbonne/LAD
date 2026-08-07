import csv
import json

from lad.rag.eval.report import write_csv, write_json


def test_write_json_roundtrips(tmp_path):
    results = {"n_terms": 3, "summary": {"retrieval": {"p_at_5": 0.5}}, "retrieval_raw": [{"a": 1}]}
    path = write_json(results, tmp_path / "results.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == results


def test_write_json_creates_parent_dirs(tmp_path):
    path = write_json({"x": 1}, tmp_path / "nested" / "dir" / "results.json")
    assert path.exists()


def test_write_csv_roundtrips(tmp_path):
    rows = [
        {"term": "gilding", "target_language": "fr", "p_at_5": 0.5},
        {"term": "museum", "target_language": "ar", "p_at_5": 0.0},
    ]
    path = write_csv(rows, tmp_path / "results.csv")

    with path.open(encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["term"] == "gilding"
    assert loaded[0]["p_at_5"] == "0.5"
    assert loaded[1]["term"] == "museum"


def test_write_csv_empty_rows_creates_empty_file_not_error(tmp_path):
    path = write_csv([], tmp_path / "empty.csv")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_write_csv_handles_non_ascii_text(tmp_path):
    rows = [{"term": "متحف", "label": "دورة"}]
    path = write_csv(rows, tmp_path / "results.csv")

    with path.open(encoding="utf-8", newline="") as f:
        loaded = list(csv.DictReader(f))
    assert loaded[0]["term"] == "متحف"
