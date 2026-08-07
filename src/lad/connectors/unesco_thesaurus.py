"""UNESCO Thesaurus connector.

One-shot SKOS/Turtle download (~4 MB, ~4,500 concepts) from
vocabularies.unesco.org -- verified endpoint (redirect target of the Skosmos
REST API's /rest/v1/unesco/data?format=text/turtle):
https://vocabularies.unesco.org/exports/thesaurus/latest/unesco-thesaurus.ttl

Not paginated: `discover` yields exactly one cursor unless a checkpoint shows
the file was already fetched.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterator

import httpx
from rdflib import RDF, Graph, URIRef
from rdflib.namespace import SKOS

from lad.connectors.base import Connector
from lad.schema import VocabularyTerm

LANGS = {"ar", "en", "fr"}


class UnescoThesaurusConnector(Connector):
    source_name = "unesco_thesaurus"

    def discover(self, checkpoint: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if checkpoint.get("last_page_index", 0) > 0:
            return
        yield {}

    def fetch_page(self, cursor: dict[str, Any]) -> httpx.Response:
        return self._client.get(self.config["download_url"])

    def parse(self, response: httpx.Response) -> list[dict[str, Any]]:
        graph = Graph()
        graph.parse(data=response.text, format="turtle")
        # Stashed for normalize(): the whole vocabulary is one "page", so
        # every item normalized from it needs to query the same graph.
        self._graph = graph
        return [{"uri": str(s)} for s in graph.subjects(RDF.type, SKOS.Concept)]

    def normalize(self, item: dict[str, Any]) -> VocabularyTerm:
        graph = self._graph
        subject = URIRef(item["uri"])

        pref_label: dict[str, str] = {}
        for _, _, label in graph.triples((subject, SKOS.prefLabel, None)):
            if label.language in LANGS:
                pref_label[label.language] = str(label)

        alt_labels: dict[str, list[str]] = {}
        for _, _, label in graph.triples((subject, SKOS.altLabel, None)):
            if label.language in LANGS:
                alt_labels.setdefault(label.language, []).append(str(label))

        scope_note: dict[str, str] = {}
        for _, _, note in graph.triples((subject, SKOS.scopeNote, None)):
            if note.language in LANGS:
                scope_note[note.language] = str(note)

        return VocabularyTerm(
            source_name=self.source_name,
            source_url=str(subject),
            source_record_id=str(subject),
            retrieval_date=date.today(),
            rights_statement=self.config.get("license_note"),
            license_note=self.config.get("license_note"),
            term_id=str(subject),
            concept_scheme="UNESCO Thesaurus",
            pref_label=pref_label,
            alt_labels=alt_labels,
            scope_note=scope_note,
            broader_ids=[str(o) for o in graph.objects(subject, SKOS.broader)],
            narrower_ids=[str(o) for o in graph.objects(subject, SKOS.narrower)],
            related_ids=[str(o) for o in graph.objects(subject, SKOS.related)],
        )
