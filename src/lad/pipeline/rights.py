"""Rights gate: records without a clearly open rights statement are routed
to a needs_review bucket instead of being silently mixed into the main
dataset.

Classification is centralized here (rather than left to each connector) so
that "does this rights string actually mean reusable" is judged one way
across every source -- a connector setting reuse_risk itself is easy to get
wrong, e.g. treating "any rights field present" as clear when the value is
something like rightsstatements.org's InC-EDU (In Copyright, restricted).
"""

from __future__ import annotations

import re

from lad.schema import HeritageRecord, ReuseRisk, VocabularyTerm

RecordType = HeritageRecord | VocabularyTerm

# rightsstatements.org's "InC" family (In Copyright, incl. InC-EDU/InC-NC/...)
# is restrictive; its "NoC" family (No Copyright) is open. CC0, public-domain
# marks, and plain CC BY / CC BY-SA are open -- but NOT CC BY-NC or CC BY-ND,
# which carry reuse restrictions, so those deliberately fall through to
# UNKNOWN rather than being misclassified as CLEAR. Anything unrecognized is
# also UNKNOWN rather than guessed at.
_RESTRICTED_PATTERN = re.compile(
    r"rightsstatements\.org/vocab/inc|all rights reserved|copyrighted", re.I
)
_CLEAR_PATTERN = re.compile(
    r"creativecommons\.org/publicdomain"
    r"|creativecommons\.org/licenses/by(-sa)?/"
    r"|rightsstatements\.org/vocab/noc"
    r"|\bcc0\b"
    r"|\bcc[\s-]?by(-sa)?\b(?!-?(nc|nd))"
    r"|\bodc[\s-]?by\b"
    r"|open data commons attribution"
    r"|public domain",
    re.I,
)


def classify_rights(rights_statement: str | None) -> ReuseRisk:
    if not rights_statement or not rights_statement.strip():
        return ReuseRisk.UNKNOWN
    if _RESTRICTED_PATTERN.search(rights_statement):
        return ReuseRisk.RESTRICTED
    if _CLEAR_PATTERN.search(rights_statement):
        return ReuseRisk.CLEAR
    return ReuseRisk.UNKNOWN


def gate_rights(record: RecordType) -> bool:
    """Classify `record.reuse_risk` from its rights statement and return
    True if it should be routed to needs_review.jsonl (anything not
    confidently CLEAR)."""
    record.reuse_risk = classify_rights(record.rights_statement)
    return record.reuse_risk != ReuseRisk.CLEAR
