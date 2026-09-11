"""eval/runners/score_image_eval.py's `_predictions_for`/`_margin_block` — pure logic over a
synthetic collect-style objects list, no network/GPU/model. The soft (debiased vote-fraction)
report now lives in score_image_eval_debiased.py — see test_eval_score_image_eval_debiased.py.
"""
from __future__ import annotations

from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, make_predictor
from eval.metrics.classification import classification_report
from eval.metrics.group_scoring import group_report
from eval.runners.score_image_eval import _margin_block, _predictions_for

_predict = make_predictor("free_text", GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS, CLASS_CODES)


def _free_text_objects():
    return [
        {
            "Galaxy10_DECals_index": 1, "ra": 10.0, "dec": 0.0,
            "label_name": "Round Smooth Galaxies",
            "base_answer": "This looks like a round smooth galaxy.",
            "equipped_answer": "A round smooth galaxy with a bright core.",
        },
        {
            "Galaxy10_DECals_index": 2, "ra": 20.0, "dec": 0.0,
            "label_name": "Merging Galaxies",
            "base_answer": "Not sure what this is.",  # unparseable
            "equipped_answer": "Two galaxies merging together.",
        },
    ]


def test_free_text_predictions_and_hard_report_scores_both_sides():
    objects = _free_text_objects()
    base_pred = _predictions_for(objects, "free_text", "base", _predict)
    equipped_pred = _predictions_for(objects, "free_text", "equipped", _predict)

    y_true = [o["label_name"] for o in objects]
    assert classification_report(y_true, equipped_pred, GALAXY10_LABELS)["accuracy"] == 1.0
    assert classification_report(y_true, base_pred, GALAXY10_LABELS)["accuracy"] == 0.5  # one unparseable -> wrong


def test_free_text_group_report_scores_both_sides():
    objects = _free_text_objects()
    equipped_pred = _predictions_for(objects, "free_text", "equipped", _predict)
    y_true = [o["label_name"] for o in objects]
    assert group_report(y_true, equipped_pred)["mean_score"] == 1.0  # both correctly parsed and exact-matched


def _logprob_argmax_objects():
    return [
        {
            "Galaxy10_DECals_index": 1, "label_name": "Round Smooth Galaxies",
            "base": {"predicted_label": "Round Smooth Galaxies", "logprobs": {"Round Smooth Galaxies": -0.1, "Merging Galaxies": -2.0}},
            "equipped": {"predicted_label": "Merging Galaxies", "logprobs": {"Round Smooth Galaxies": -1.5, "Merging Galaxies": -0.2}},
        },
        {
            "Galaxy10_DECals_index": 2, "label_name": "Merging Galaxies",
            "base": {"predicted_label": "Round Smooth Galaxies", "logprobs": {"Round Smooth Galaxies": -0.3, "Merging Galaxies": -1.1}},
            "equipped": {"predicted_label": "Merging Galaxies", "logprobs": {"Round Smooth Galaxies": -3.0, "Merging Galaxies": -0.05}},
        },
    ]


def test_logprob_argmax_predictions_are_never_none():
    """The whole point of this format: predicted_label is always one of the fixed candidates —
    no parsing, so `_predictions_for` never returns `None` for a logprob_argmax file.
    """
    objects = _logprob_argmax_objects()
    predictions = _predictions_for(objects, "logprob_argmax", "equipped", predict=None)
    assert predictions == ["Merging Galaxies", "Merging Galaxies"]
    assert None not in predictions


def test_margin_block_reflects_the_gap_between_top_two_candidates():
    objects = _logprob_argmax_objects()
    margin = _margin_block(objects, "equipped")
    # object 1: -0.2 - (-1.5) = 1.3; object 2: -0.05 - (-3.0) = 2.95
    assert margin["mean"] == (1.3 + 2.95) / 2
    assert margin["n_below_0.5"] == 0
