"""Europeana Search API connector.

REST, `wskey` auth (free key from the Europeana account section). Field
names below were verified against a live response from
api.europeana.eu/record/v2/search.json (profile=rich) using the public
"api2demo" demo key -- e.g. `title`, `dcDescription`, `dcCreator`,
`dataProvider`, `provider`, `rights`, `language`, `type`, `guid`, `id` are
top-level arrays/strings on each item, not nested under an envelope.

`default_qf` in config can be a single string or a YAML list; httpx repeats
the `qf` param for each list entry, and Europeana AND-combines them.
"""

from __future__ import annotations

import itertools
from datetime import date
from typing import Any, Iterator

import httpx

from lad.connectors.base import Connector
from lad.schema import HeritageRecord


def _first(item: dict[str, Any], field_name: str) -> str | None:
    value = item.get(field_name)
    if isinstance(value, list):
        return str(value[0]) if value else None
    if isinstance(value, str):
        return value
    return None


class EuropeanaConnector(Connector):
    source_name = "europeana"

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        start_page = checkpoint.get("last_page_index", 0)
        page_size = self.config.get("page_size", 100)
        for page in itertools.count(start_page):
            # Europeana's `start` param is 1-indexed and capped at 1000 for
            # the free tier's cursor-less pagination.
            yield {"start": page * page_size + 1, "rows": page_size}

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        params = {
            "wskey": self.config["auth_value"],
            "query": self.config.get("default_query", "heritage"),
            "qf": self.config.get("default_qf"),
            "start": cursor["start"],
            "rows": cursor["rows"],
            "profile": "rich",
        }
        return self._client.get(self.config["base_url"], params=params)

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        payload = response.json()
        if not payload.get("success", True):
            raise RuntimeError(f"Europeana API error: {payload.get('error')}")
        return payload.get("items", [])

    def normalize(self, item: dict[str, Any]) -> HeritageRecord:
        languages = item.get("language") or []
        language_code = languages[0] if languages else None
        rights_values = item.get("rights") or []
        rights = str(rights_values[0]) if rights_values else None
        subjects = item.get("dcSubject") or []
        if not subjects:
            lang_aware = item.get("dcSubjectLangAware") or {}
            subjects = [s for values in lang_aware.values() for s in values]

        return HeritageRecord(
            source_name=self.source_name,
            source_url=item.get("guid", ""),
            source_record_id=item.get("id", ""),
            retrieval_date=date.today(),
            rights_statement=rights,
            license_note=self.config.get("license_note"),
            language_code=language_code,
            original_language=language_code,
            alignment_group_id=_first(item, "europeanaCollectionName"),
            title=_first(item, "title"),
            description=_first(item, "dcDescription"),
            object_type=_first(item, "type"),
            collection=_first(item, "dataProvider"),
            creator=_first(item, "dcCreator"),
            institution=_first(item, "provider"),
            subject_terms=list(subjects),
        )
