"""Regenerates sample.xlsx -- a tiny fixture mirroring the real Kalcium
export's exact column layout (verified against the live file: two header
rows, then one row per concept, 74 columns: 11 concept-level + 3x21
per-language blocks). Not run automatically; re-run manually if the
fixture needs to change:

    .venv/bin/python tests/fixtures/kalcium_termbase/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

OUT_PATH = Path(__file__).parent / "sample.xlsx"

HEADER_ROW_1 = (
    [None, "Entry"] + [None] * 9 + ["Arabic"] + [None] * 20 + ["English"] + [None] * 20 + ["French"] + [None] * 20
)

_LANG_FIELDS = [
    "Status", "Note", "Context", "Source [Context]", "Definition", "Source [Definition]",
    "Broader concepts", "Narrower concepts", "Related concepts",
    "<stamp>", "TERM", "Cr. user", "Cr. date", "Ch. user", "Ch. date", "Note", "Context",
    "Part of speech", "Usage status", "Source", "Grammatical Gender",
]

HEADER_ROW_2 = (
    ["Concept", "Cr. user", "Cr. date", "Ch. user", "Ch. date", "Entry class", "<stamp>", "Note", "Image",
     "Source [Image]", "Subject field"]
    + _LANG_FIELDS + _LANG_FIELDS + _LANG_FIELDS
)

# Row for concept 537 ("Readymade"), reproduced from the real export (values
# verified against the live file) -- exercises the full-population case.
ROW_537 = (
    537, "tom", "2022.09.27. 11:19:58", "mariam.alblooshi", "2025.07.24. 02:10:38",
    "Unspecified", "gGWmpalK5L8i9CC7U2aMSQ==", None, None, None, "Sculpture and Carving",
    # Arabic block
    None, None, None, None,
    "اتجاه فني يعتمد على تجميع الأشياء الجاهزة وتوظيفها فنيا.",
    "المعجم الموحد لمصطلحات الفنون التشكيلية، مكتب تنسيق التعريب",
    "الفنون الجميلة",
    "فن النحت الجديد<br />فن النحت البريطاني الجديد",
    "الفنون التشكيلية<br />النحاتون<br />\xa0",
    "EzzBd4cwlk+qlpmk5lGJ3g==", "الأشياء الجاهزة", "tom", "2022.09.27. 11:19:58", "mzaggar",
    "2023.11.28. 11:15:18", "طريقة في الفنّ ابتكرها مارسيل دوشامب عام 1920", None, None,
    "Preferred", "معجم العمارة والفنّ، د. عفيف البهنسي", None,
    # English block
    None, None, None, None,
    "A term devised by Marcel Duchamp to describe pre-existing, mass-produced objects.",
    "The Concise Oxford Dictionary of Art Terms",
    "Fine arts",
    "New sculpture<br />New British sculpture",
    "Plastic arts<br />Sculptors<br />Visual arts",
    "OCSnE6ROl3zWpaDiew/MjA==", "Readymade ", "tom", "2022.09.27. 11:19:58", "mzaggar",
    "2023.11.28. 11:15:45", None, None, None, None, None, None,
    # French block
    None, None, None, None, None, None, None, None, None,
    "pHZbUJqwSjzUMTz+9plHzw==", " ready-made", "tom", "2022.09.27. 11:19:58", "mariam.alblooshi",
    "2025.07.24. 01:37:00", None, None, None, None, None, None,
)

# Concept with only an English term (Arabic/French empty) -- exercises the
# partial-language-coverage path.
ROW_ENGLISH_ONLY = (
    9001, "tom", "2022.01.01. 10:00:00", "tom", "2022.01.01. 10:00:00",
    "Unspecified", "stamp0", None, None, None, "Photography",
    *([None] * 21),
    None, None, None, None, "A photographic process.", "Test source", None, None, None,
    "stamp1", "Daguerreotype", "tom", "2022.01.01. 10:00:00", "tom", "2022.01.01. 10:00:00",
    None, None, None, None, None, None,
    *([None] * 21),
)

# Stray repeated header row that appears mid-sheet in the real export --
# must be filtered out (concept id is the literal string "Concept").
STRAY_HEADER_ROW = HEADER_ROW_2


def main() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Termbase export"
    ws.append(HEADER_ROW_1)
    ws.append(HEADER_ROW_2)
    ws.append(ROW_537)
    ws.append(ROW_ENGLISH_ONLY)
    ws.append(STRAY_HEADER_ROW)

    ws2 = wb.create_sheet("FieldValues")
    ws2.append(["Unspecified", "Aesthetics", "New", "noun", "Preferred", "masculine"])

    wb.save(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
