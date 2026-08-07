"""Abstract connector interface shared by every source.

A connector implements four source-specific methods (discover, fetch_page,
parse, normalize); run() wires them together with rate limiting, retry,
raw-response caching, checkpointing, rights gating, and JSONL output so that
plumbing is written once, not once per source.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from lad.pipeline.rights import gate_rights
from lad.schema import HeritageRecord, VocabularyTerm
from lad.storage import writer

logger = logging.getLogger(__name__)

RecordType = HeritageRecord | VocabularyTerm


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@dataclass
class RunSummary:
    source_name: str
    records_fetched: int = 0
    records_normalized: int = 0
    flagged_for_review: int = 0
    errors: int = 0
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Connector(abc.ABC):
    source_name: str

    def __init__(self, config: dict[str, Any], client: httpx.Client | None = None):
        self.config = config
        self.rate_limit_per_sec = config.get("rate_limit_per_sec", 1)
        self._client = client or httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": f"LAD-heritage-pipeline/0.1 (contact: {config.get('contact', 'n/a')})"
            },
        )
        self._last_request_at = 0.0

    # ---- implemented per source -----------------------------------------
    @abc.abstractmethod
    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield page/cursor descriptors, resuming from `checkpoint` if present."""

    @abc.abstractmethod
    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        """Perform the HTTP call for one page/cursor."""

    @abc.abstractmethod
    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        """Extract a list of raw item dicts from one page response."""

    @abc.abstractmethod
    def normalize(self, item: dict[str, Any]) -> RecordType:
        """Map one raw item onto HeritageRecord or VocabularyTerm."""

    # ---- shared plumbing --------------------------------------------------
    def _throttled_fetch(self, cursor: dict[str, Any]) -> httpx.Response:
        min_interval = 1.0 / self.rate_limit_per_sec
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        response = self._retrying_fetch(cursor)
        self._last_request_at = time.monotonic()
        return response

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _retrying_fetch(self, cursor: dict[str, Any]) -> httpx.Response:
        response = self.fetch_page(cursor)
        response.raise_for_status()
        return response

    def run(
        self,
        run_date: date | None = None,
        refresh: bool = False,
        page_limit: int | None = None,
    ) -> RunSummary:
        run_date = run_date or datetime.now(timezone.utc).date()
        summary = RunSummary(source_name=self.source_name)
        checkpoint = {} if refresh else writer.load_checkpoint(self.source_name)
        page_index = checkpoint.get("last_page_index", 0) if not refresh else 0

        for cursor in self.discover(checkpoint if not refresh else {}):
            if page_limit is not None and (page_index - checkpoint.get("last_page_index", 0)) >= page_limit:
                break

            try:
                response = self._throttled_fetch(cursor)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    # Some paginated APIs (e.g. loc.gov) 404 a page number
                    # past the true last page instead of returning an empty
                    # 200 -- the same "no more data" signal as the empty-
                    # items case below, just via a different status code.
                    # Not an error: the harvest correctly reached the end.
                    logger.info(
                        "%s cursor=%s returned 404, treating as end of pagination", self.source_name, cursor
                    )
                    break
                logger.exception("fetch failed for %s cursor=%s", self.source_name, cursor)
                summary.errors += 1
                break
            except httpx.HTTPError:
                logger.exception("fetch failed for %s cursor=%s", self.source_name, cursor)
                summary.errors += 1
                break

            writer.write_raw_page(self.source_name, run_date, page_index, response.text)

            try:
                items = self.parse(response)
            except Exception:
                logger.exception("parse failed for %s page=%s", self.source_name, page_index)
                summary.errors += 1
                break

            if not items:
                logger.info("no more items for %s at page=%s, stopping", self.source_name, page_index)
                break

            summary.records_fetched += len(items)

            for item in items:
                try:
                    record = self.normalize(item)
                except Exception:
                    logger.exception("normalize failed for %s item on page=%s", self.source_name, page_index)
                    summary.errors += 1
                    continue

                needs_review = gate_rights(record)
                writer.append_record(self.source_name, record, needs_review=needs_review)
                summary.records_normalized += 1
                if needs_review:
                    summary.flagged_for_review += 1

            page_index += 1
            writer.save_checkpoint(self.source_name, {"last_page_index": page_index, **cursor})

        writer.append_run_summary(
            {
                "source_name": summary.source_name,
                "records_fetched": summary.records_fetched,
                "records_normalized": summary.records_normalized,
                "flagged_for_review": summary.flagged_for_review,
                "errors": summary.errors,
                "started_at": summary.started_at,
            }
        )
        return summary
