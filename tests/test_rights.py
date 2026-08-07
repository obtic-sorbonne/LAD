import pytest

from lad.pipeline.rights import classify_rights
from lad.schema import ReuseRisk


@pytest.mark.parametrize(
    "rights_statement,expected",
    [
        (None, ReuseRisk.UNKNOWN),
        ("", ReuseRisk.UNKNOWN),
        ("http://rightsstatements.org/vocab/InC/1.0/", ReuseRisk.RESTRICTED),
        ("http://rightsstatements.org/vocab/InC-EDU/1.0/", ReuseRisk.RESTRICTED),
        ("All rights reserved", ReuseRisk.RESTRICTED),
        ("http://rightsstatements.org/vocab/NoC-US/1.0/", ReuseRisk.CLEAR),
        ("http://creativecommons.org/publicdomain/mark/1.0/", ReuseRisk.CLEAR),
        ("http://creativecommons.org/licenses/by-sa/4.0/", ReuseRisk.CLEAR),
        ("CC BY-SA 3.0 IGO", ReuseRisk.CLEAR),
        ("CC0", ReuseRisk.CLEAR),
        ("CC BY-NC 4.0", ReuseRisk.UNKNOWN),
        ("http://creativecommons.org/licenses/by-nc/4.0/", ReuseRisk.UNKNOWN),
        ("ODC-BY", ReuseRisk.CLEAR),
        ("Open Data Commons Attribution License", ReuseRisk.CLEAR),
        ("Some unrecognized rights string", ReuseRisk.UNKNOWN),
    ],
)
def test_classify_rights(rights_statement, expected):
    assert classify_rights(rights_statement) == expected
