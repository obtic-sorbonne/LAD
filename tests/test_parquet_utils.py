"""Regression test for a real bug: pyarrow can't write a Parquet struct
with zero child fields, which is what you get when a dict-valued field
(e.g. VocabularyTerm.alt_labels) is `{}` on every single row of a source
(Getty AAT never populates alt_labels/scope_note/source_concept_ids at
all) -- Table.from_pylist() infers `struct<>` and pq.write_table() then
raises ArrowNotImplementedError. Fixed by JSON-encoding dict-valued
fields before the Arrow conversion. Shared by pipeline/compact.py and
pipeline/publish_hf.py, since both hit the exact same risk.
"""

from __future__ import annotations

import json

from lad.pipeline.parquet_utils import json_encode_dict_fields


def test_all_empty_dict_column_is_json_encoded_not_left_as_a_dict():
    rows = [
        {"term_id": "a", "alt_labels": {}, "pref_label": {"en": "foo"}},
        {"term_id": "b", "alt_labels": {}, "pref_label": {"en": "bar"}},
    ]

    encoded = json_encode_dict_fields(rows)

    for row in encoded:
        assert isinstance(row["alt_labels"], str)
        assert json.loads(row["alt_labels"]) == {}
        assert isinstance(row["pref_label"], str)

    assert json.loads(encoded[0]["pref_label"]) == {"en": "foo"}


def test_this_shape_actually_writes_to_parquet(tmp_path):
    """The real regression check: what previously raised
    ArrowNotImplementedError must now succeed end to end."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"x": {}}, {"x": {}}]
    table = pa.Table.from_pylist(json_encode_dict_fields(rows))
    out_path = tmp_path / "test.parquet"

    pq.write_table(table, out_path)  # must not raise

    assert out_path.exists()


def test_non_dict_fields_pass_through_unchanged():
    rows = [{"term_id": "a", "broader_ids": ["x", "y"], "confidence": 0.5, "note": None}]

    encoded = json_encode_dict_fields(rows)

    assert encoded[0]["broader_ids"] == ["x", "y"]
    assert encoded[0]["confidence"] == 0.5
    assert encoded[0]["note"] is None
