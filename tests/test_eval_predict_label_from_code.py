"""eval/metrics/caption_to_label.py's predict_label_from_code — the parser for the digit-code
answer format eval.datasets.image_galaxy10.CLASS_CODE_PROMPT actually uses (confirmed live to
work far better than free-text label matching, but a genuinely different parser is needed since
predict_label's keyword matching never matches a bare digit at all).
"""
from __future__ import annotations

import pytest

from eval.datasets.image_galaxy10 import CLASS_CODES
from eval.metrics.caption_to_label import predict_label_from_code


@pytest.mark.parametrize("answer,expected", [
    (" 2", "Round Smooth Galaxies"),
    ("5", "Barred Spiral Galaxies"),
    (" 9\n", "Edge-on Galaxies with Bulge"),
    ("Code: 5", "Barred Spiral Galaxies"),
    ("0", "Disturbed Galaxies"),
])
def test_parses_valid_digit_codes(answer, expected):
    assert predict_label_from_code(answer, CLASS_CODES) == expected


@pytest.mark.parametrize("answer", ["", "no digit here", "ten"])
def test_no_digit_returns_none(answer):
    assert predict_label_from_code(answer, CLASS_CODES) is None


def test_multi_digit_number_does_not_match_an_embedded_digit():
    # "10" must not be parsed as code "1" — the digit isn't standalone.
    assert predict_label_from_code("10", CLASS_CODES) is None
