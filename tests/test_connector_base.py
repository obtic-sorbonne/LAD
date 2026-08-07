"""Tests for Connector.run()'s shared orchestration (base.py) -- pagination
stopping conditions specifically, since those are easy to get subtly wrong
and hard to verify against a live API on demand (see the 404-at-end-of-
pagination fix this covers).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import httpx
import pytest

from lad.connectors.base import Connector
from lad.schema import HeritageRecord
from lad.storage import writer


class _StubConnector(Connector):
    """A minimal connector whose pages are pre-scripted: each entry in
    `pages` is either a list of raw items (normal page) or an Exception
    instance to raise from fetch_page (simulates an HTTP failure)."""

    source_name = "stub_test_source"

    def __init__(self, config: dict[str, Any], pages: list[Any]):
        super().__init__(config)
        self._pages = pages

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        for i in range(len(self._pages)):
            yield {"page": i}

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        outcome = self._pages[cursor["page"]]
        if isinstance(outcome, Exception):
            raise outcome
        # raise_for_status() requires a request to be attached even for a
        # 200 -- httpx.Client normally does this itself; a hand-built
        # Response for a test stub has to do it explicitly.
        request = httpx.Request("GET", "https://example.org")
        return httpx.Response(200, json=outcome, request=request)

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        return response.json()

    def normalize(self, item: dict[str, Any]) -> HeritageRecord:
        return HeritageRecord(
            source_name=self.source_name,
            source_url="https://example.org",
            source_record_id=item["id"],
            retrieval_date=date.today(),
            rights_statement="CC0",
            title=item["id"],
        )


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"{status_code}", request=request, response=response)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Every test in this file writes real files via storage/writer.py --
    redirect those into a throwaway tmp_path instead of the project's data/."""
    monkeypatch.setattr(writer, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(writer, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(writer, "LOGS_DIR", tmp_path / "logs")


def test_404_stops_pagination_cleanly_without_counting_as_an_error():
    """loc.gov (and similar APIs) 404 a page past the true last page instead
    of returning an empty 200 -- this must be treated the same as the
    empty-items stopping condition, not as a failure."""
    pages = [
        [{"id": "a"}, {"id": "b"}],
        _http_status_error(404),
    ]
    connector = _StubConnector({"rate_limit_per_sec": 1000}, pages)

    summary = connector.run()

    assert summary.records_fetched == 2
    assert summary.records_normalized == 2
    assert summary.errors == 0


def test_non_404_http_error_still_counts_as_an_error():
    """A real failure (e.g. 403) must still be treated as an error, not
    silently swallowed the way 404 now is."""
    pages = [
        [{"id": "a"}],
        _http_status_error(403),  # not retryable, not 404 -- fails fast
    ]
    connector = _StubConnector({"rate_limit_per_sec": 1000}, pages)

    summary = connector.run()

    assert summary.records_fetched == 1
    assert summary.errors == 1


def test_empty_page_stops_pagination_without_error():
    pages = [[{"id": "a"}], []]
    connector = _StubConnector({"rate_limit_per_sec": 1000}, pages)

    summary = connector.run()

    assert summary.records_fetched == 1
    assert summary.errors == 0
