"""UNESDOC (UNESCO Digital Library catalogue) connector.

UNESCO DataHub (data.unesco.org) runs Opendatasoft's records API, which
exposes two relevant endpoints for dataset `doc001`:

  - records/1.0/search/  -- paginated, but enforces a hard server-side cap
    verified live: `start + rows` cannot exceed 10000, so pagination alone
    can never reach more than the first 10,000 of doc001's 285,433 records
    (confirmed via a live 400 response: "The sum of `start` + `rows`
    parameters can not be more than 10000... use the Download service.").
  - records/1.0/download/  -- no such cap. A single request with `rows` set
    above the total record count returns the whole dataset as a flat JSON
    array (verified live: 285,433 records in one ~373MB response, ~100s).
    `start` is not honored by this endpoint, so it's a one-shot bulk pull,
    not paginated -- the same pattern as the UNESCO Thesaurus connector.

Each record's `fields` carries `title`, `subject` (comma-separated string),
`url`, `document_type`, `coverage`, `language` (single ISO 639-2 code, or
comma-joined for multilingual documents), `year`, `uuid`. No auth required.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import httpx

from lad.connectors.base import Connector
from lad.schema import HeritageRecord


class UnesdocConnector(Connector):
    source_name = "unesdoc"

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if checkpoint.get("last_page_index", 0) > 0:
            return
        yield {}

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        params = {
            "dataset": self.config.get("dataset", "doc001"),
            "format": "json",
            # Set well above the known total so one request returns
            # everything; bump this in config if the catalogue grows past it.
            "rows": self.config.get("bulk_rows", 500000),
        }
        # This single request takes ~100s for the full dataset -- override
        # the client's default 30s timeout for it specifically.
        return self._client.get(self.config["download_url"], params=params, timeout=180.0)

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        # The download endpoint returns a flat JSON array of records
        # (each already shaped like {"recordid": ..., "fields": {...}}),
        # unlike the search endpoint's {"records": [...]} envelope.
        return response.json()

    def normalize(self, item: dict[str, Any]) -> HeritageRecord:
        fields = item.get("fields", {})
        subject_raw = fields.get("subject", "")
        subject_terms = [s.strip() for s in subject_raw.split(",") if s.strip()]

        return HeritageRecord(
            source_name=self.source_name,
            source_url=fields.get("url", ""),
            source_record_id=item.get("recordid", fields.get("uuid", "")),
            retrieval_date=date.today(),
            # UNESDOC does not publish a per-record machine-readable rights
            # statement in this API; treat as unknown so it's rights-gated
            # into needs_review rather than assumed reusable.
            rights_statement=None,
            license_note=self.config.get("license_note"),
            language_code=fields.get("language"),
            original_language=fields.get("language"),
            alignment_group_id=fields.get("uuid"),
            title=fields.get("title"),
            description=fields.get("description"),
            object_type=fields.get("document_type") or fields.get("type"),
            creator=fields.get("creator"),
            place=fields.get("coverage"),
            date_text=fields.get("year"),
            subject_terms=subject_terms,
            institution="UNESCO",
        )
