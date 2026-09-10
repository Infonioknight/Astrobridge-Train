"""eval/metrics/caption_to_label.py's predict_label_from_free_text — the `verbose_class` answer
format's parser. Pure logic, no model/network.

Cases are written against how the two models actually behave, not hypotheticals: base is expected
to follow the requested `FINAL ANSWER:` format, while equipped tends to revert to the captioning
voice it was fine-tuned on — whose real training captions (confirmed by reading
`transient_caption` values in BuildNg/astrobridge-transients-dataset) all end
"...is consistent with the SN <type> class." Both must parse.
"""
from __future__ import annotations

import pytest

from eval.datasets.lightcurve_yse import SN_LABELS
from eval.metrics.caption_to_label import SN_TYPE_SYNONYMS, make_predictor, predict_label_from_free_text


def _predict(text: str) -> str | None:
    return predict_label_from_free_text(text, SN_LABELS, SN_TYPE_SYNONYMS)


@pytest.mark.parametrize("text,expected", [
    ("FINAL ANSWER: SN Ia", "SN Ia"),
    ("FINAL ANSWER: SN II", "SN II"),
    ("FINAL ANSWER: SN Ibc", "SN Ibc"),
    ("The curve rises over 16 days then declines.\nFINAL ANSWER: SN II", "SN II"),
    ("final answer:  sn ibc  ", "SN Ibc"),  # case/whitespace tolerant
])
def test_explicit_final_answer_format(text, expected):
    assert _predict(text) == expected


@pytest.mark.parametrize("text,expected", [
    # Verbatim phrasing style from the real training captions the equipped model was tuned on.
    ("This asymmetric post-peak morphology, marked by faster blue-band cooling, is consistent with the SN Ia class.", "SN Ia"),
    ("Although not uniquely diagnostic, the photometric behavior is consistent with the SN II class.", "SN II"),
])
def test_equipped_models_trained_caption_voice_parses(text, expected):
    assert _predict(text) == expected


def test_conclusion_wins_over_an_earlier_hedge():
    """A verbose answer may weigh several types before concluding. The LAST mention is the
    verdict — first-match (what `predict_label` does) would return the discarded candidate.
    """
    text = "The early rise resembles a SN IIP, but the post-peak decline is consistent with the SN Ia class."
    assert _predict(text) == "SN Ia"


@pytest.mark.parametrize("text,expected", [
    ("This is best described as SN Ic-BL.", "SN Ibc"),
    ("Photometry indicates SN IIn.", "SN II"),
    ("Consistent with a SN Ib.", "SN Ibc"),
    ("Looks like SN IIP.", "SN II"),
])
def test_real_subtypes_map_to_their_parent_class(text, expected):
    """The dataset only ever contains three labels, but models emit real subtypes that aren't
    among them — those must resolve rather than count as unparsed.
    """
    assert _predict(text) == expected


def test_sn_iib_is_not_silently_read_as_sn_ii():
    """The prefix trap: "sn ii" is a strict substring of "sn iib", so both match at the same
    position. Without the longest-match tie-break this returns SN II depending on iteration order.
    """
    assert _predict("Best match is SN IIb.") == "SN Ibc"


def test_falls_back_to_prose_when_the_final_answer_line_names_nothing():
    text = "It is consistent with the SN Ibc class.\nFINAL ANSWER: unclear"
    assert _predict(text) == "SN Ibc"


def test_no_class_mentioned_returns_none():
    assert _predict("I cannot determine the type from this light curve.") is None


def test_word_boundaries_prevent_matches_inside_unrelated_words():
    # "Ia" appears inside "Iapetus"; a plain substring scan would wrongly return SN Ia.
    assert _predict("The host galaxy resembles Iapetus in colour.") is None


def test_make_predictor_dispatches_verbose_class_to_this_parser():
    predict = make_predictor("verbose_class", SN_LABELS, SN_TYPE_SYNONYMS)
    assert predict("FINAL ANSWER: SN Ibc") == "SN Ibc"


def test_verbose_class_is_a_distinct_format_from_legacy_free_text():
    """Reusing the "free_text" name would silently re-score every pre-existing free-text file
    under genuinely different rules, so the two must dispatch differently.

    `predict_label` returns the first label in VOCABULARY order that appears anywhere in the text
    (so a discarded candidate mentioned in passing can win outright);
    `predict_label_from_free_text` returns the last one mentioned, i.e. the conclusion.
    """
    text = "Initially SN II was considered, but the decline is consistent with the SN Ibc class."
    verbose = make_predictor("verbose_class", SN_LABELS, SN_TYPE_SYNONYMS)
    legacy = make_predictor("free_text", SN_LABELS, SN_TYPE_SYNONYMS)
    assert verbose(text) == "SN Ibc"     # the conclusion actually reached
    assert legacy(text) == "SN II"       # the discarded candidate — the old, different behaviour
