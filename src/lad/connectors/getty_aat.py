"""Getty Art & Architecture Thesaurus (AAT) connector.

SPARQL endpoint at vocab.getty.edu, verified live. AAT has no Arabic labels
at all (checked: 39 language values across a sample concept, none Arabic --
consistent with the coverage gap already documented for Europeana and the
stakeholder brief) -- this connector only ever populates `en`/`fr`.

Harvests four top-level facets, each mapped onto one of the LAD Termbase's
subject_field buckets (see schema.py VocabularyTerm.subject_field):
  - Materials Facet (300264091)         -> materials_and_techniques
  - Activities Facet (300264090)        -> materials_and_techniques
  - Objects Facet (300264092)           -> object_typology
  - Styles and Periods Facet (300264088)-> art_historical_period
"AAT has no "museography" or "provenance" facet -- those subject_field
values only ever come from UNESCO Thesaurus / manual curation in the
termbase builder.

Pagination is COUNT-then-OFFSET per facet (verified live: SPARQL COUNT and
paginated SELECT both work), not "empty page stops the run" like the other
connectors -- a multi-facet source can't use that signal, since an empty
page at the end of one facet doesn't mean the harvest is done.

Scope note: only skos:prefLabel (en/fr) is pulled per concept, not
altLabel/broader/narrower -- fetching relations per-concept would be an
expensive N+1 query pattern against a shared public endpoint. Good enough
as termbase enrichment input; not a full AAT mirror.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import httpx

from lad.connectors.base import Connector
from lad.schema import VocabularyTerm

FACETS: list[tuple[str, str]] = [
    ("http://vocab.getty.edu/aat/300264091", "materials_and_techniques"),  # Materials Facet
    ("http://vocab.getty.edu/aat/300264090", "materials_and_techniques"),  # Activities Facet
    ("http://vocab.getty.edu/aat/300264092", "object_typology"),  # Objects Facet
    ("http://vocab.getty.edu/aat/300264088", "art_historical_period"),  # Styles and Periods Facet
]

_COUNT_QUERY = """
SELECT (COUNT(?s) AS ?n) WHERE {{
  ?s skos:inScheme <http://vocab.getty.edu/aat/> ;
     gvp:broaderExtended <{facet_uri}> .
}}
"""

_PAGE_QUERY = """
SELECT ?s ?enLabel ?frLabel WHERE {{
  ?s skos:inScheme <http://vocab.getty.edu/aat/> ;
     gvp:broaderExtended <{facet_uri}> ;
     skos:prefLabel ?enLabel .
  FILTER(lang(?enLabel)="en")
  OPTIONAL {{ ?s skos:prefLabel ?frLabel . FILTER(lang(?frLabel)="fr") }}
}}
ORDER BY ?s
LIMIT {limit} OFFSET {offset}
"""


class GettyAatConnector(Connector):
    source_name = "getty_aat"

    def _facet_count(self, facet_uri: str) -> int:
        response = self._client.get(
            self.config["sparql_url"],
            params={"query": _COUNT_QUERY.format(facet_uri=facet_uri)},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        return int(bindings[0]["n"]["value"]) if bindings else 0

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        page_size = self.config.get("page_size", 200)
        resume_facet = checkpoint.get("facet_index", 0)
        resume_offset = checkpoint.get("facet_offset", 0)

        for facet_index, (facet_uri, subject_field) in enumerate(FACETS):
            if facet_index < resume_facet:
                continue
            offset = resume_offset if facet_index == resume_facet else 0
            total = self._facet_count(facet_uri)
            while offset < total:
                yield {
                    "facet_index": facet_index,
                    "facet_offset": offset,
                    "facet_uri": facet_uri,
                    "subject_field": subject_field,
                    "page_size": page_size,
                }
                offset += page_size

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        # Stashed for normalize(): parse()/normalize() only see the response
        # body, not the cursor, but subject_field is cursor-level info.
        self._current_subject_field = cursor["subject_field"]
        query = _PAGE_QUERY.format(
            facet_uri=cursor["facet_uri"],
            limit=cursor["page_size"],
            offset=cursor["facet_offset"],
        )
        return self._client.get(
            self.config["sparql_url"],
            params={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        payload = response.json()
        return payload.get("results", {}).get("bindings", [])

    def normalize(self, item: dict[str, Any]) -> VocabularyTerm:
        uri = item["s"]["value"]
        en_label = item["enLabel"]["value"]
        pref_label = {"en": en_label}
        if "frLabel" in item:
            pref_label["fr"] = item["frLabel"]["value"]

        return VocabularyTerm(
            source_name=self.source_name,
            source_url=uri,
            source_record_id=uri,
            retrieval_date=date.today(),
            rights_statement=self.config.get("license_note"),
            license_note=self.config.get("license_note"),
            term_id=uri,
            concept_scheme="Getty AAT",
            pref_label=pref_label,
            subject_field=self._current_subject_field,
        )
