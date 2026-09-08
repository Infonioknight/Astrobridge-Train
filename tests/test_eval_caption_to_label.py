"""eval/metrics/caption_to_label.py's predict_label — table-driven, one row per caption ->
expected label, including the explicit "no match -> None" path.
"""
from __future__ import annotations

import pytest

from eval.metrics.caption_to_label import (
    GALAXY10_LABEL_SYNONYMS,
    SN_TYPE_SYNONYMS,
    predict_label,
)

SN_LABELS = ["SN Ia", "SN II", "SN Ibc"]


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("This is a SN Ia based on the light curve.", "SN Ia"),
        ("Likely a Type Ia thermonuclear explosion.", "SN Ia"),
        ("Classic SN II hydrogen-rich signature.", "SN II"),
        ("A stripped-envelope supernova, type ic.", "SN Ibc"),
        ("The decline rate suggests a type ic supernova.", "SN Ibc"),
        ("Not enough information to classify this object.", None),
    ],
)
def test_sn_type_matching(caption, expected):
    assert predict_label(caption, SN_LABELS, SN_TYPE_SYNONYMS) == expected


def test_case_insensitive():
    assert predict_label("SN IA CONFIRMED", SN_LABELS, SN_TYPE_SYNONYMS) == "SN Ia"


def test_no_synonyms_falls_back_to_label_name_itself():
    assert predict_label("Classified as Barred Spiral Galaxies.", ["Barred Spiral Galaxies"]) == "Barred Spiral Galaxies"


def test_first_matching_label_in_vocabulary_order_wins_on_hedged_caption():
    # caption mentions both labels' own names — vocabulary order breaks the tie
    caption = "Possibly SN Ia, though the fast decline suggests SN Ibc."
    assert predict_label(caption, ["SN Ia", "SN Ibc"], SN_TYPE_SYNONYMS) == "SN Ia"
    assert predict_label(caption, ["SN Ibc", "SN Ia"], SN_TYPE_SYNONYMS) == "SN Ibc"


def test_galaxy10_synonyms_real_labels():
    assert predict_label(
        "This is classified as edge-on with bulge.",
        list(GALAXY10_LABEL_SYNONYMS), GALAXY10_LABEL_SYNONYMS,
    ) == "Edge-on Galaxies with Bulge"
