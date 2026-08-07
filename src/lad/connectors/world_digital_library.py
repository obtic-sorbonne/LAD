"""World Digital Library connector.

WDL was folded into the Library of Congress; content now lives in the
`world-digital-library` collection on loc.gov. Verified live against:
  GET https://www.loc.gov/collections/world-digital-library/?fo=json&sp=<page>&c=<per_page>
  -> top-level `pagination` has {current, of (total pages), perpage, ...}
  -> `content.results[]` items carry id/title/date/language/type/url plus a
     nested `item` dict (call_number, contributors, created_published,
     format, medium, title).

Public, no API key. Requires a descriptive User-Agent (default urllib UA
gets a 403) -- the shared Connector client already sets one. LOC explicitly
asks callers to self-throttle; default rate_limit_per_sec in config is 1.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import httpx

from lad.connectors.base import Connector
from lad.schema import HeritageRecord


def _first(value: Any) -> str | None:
    """loc.gov fields are inconsistently typed -- some are plain strings,
    others single-element lists -- depending on item type."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    if isinstance(value, str):
        return value
    return None


class WorldDigitalLibraryConnector(Connector):
    source_name = "world_digital_library"

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        start_page = checkpoint.get("last_page_index", 0) + 1  # loc.gov `sp` is 1-indexed
        page_size = self.config.get("page_size", 100)
        page = start_page
        while True:
            yield {"sp": page, "c": page_size}
            page += 1

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        params = {"fo": "json", "sp": cursor["sp"], "c": cursor["c"]}
        return self._client.get(self.config["base_url"], params=params)

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        payload = response.json()
        return payload.get("content", {}).get("results", [])

    def normalize(self, item: dict[str, Any]) -> HeritageRecord:
        nested = item.get("item", {})
        languages = item.get("language") or nested.get("language") or []
        language_code = languages[0] if languages else None
        contributors = nested.get("contributors") or item.get("contributor") or []

        return HeritageRecord(
            source_name=self.source_name,
            source_url=item.get("url", item.get("id", "")),
            source_record_id=item.get("id", ""),
            retrieval_date=date.today(),
            # loc.gov items link out to per-item rights statements rather
            # than exposing one in the search payload; treat as unknown
            # until enriched from the item page.
            rights_statement=None,
            license_note=self.config.get("license_note"),
            language_code=language_code,
            original_language=language_code,
            title=_first(item.get("title")) or _first(nested.get("title")),
            object_type=_first(item.get("type")),
            date_text=_first(item.get("date")) or _first(nested.get("date")),
            creator=", ".join(contributors) if contributors else None,
            material=", ".join(nested.get("medium", [])) or None,
            institution="Library of Congress / World Digital Library",
        )
