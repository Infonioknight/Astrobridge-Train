"""eval/runners/score_image_eval.py's _hard_report/_group_report_for — pure logic over a
synthetic collect-style objects list, no network/GPU/model. The soft (debiased vote-fraction)
report now lives in score_image_eval_debiased.py — see test_eval_score_image_eval_debiased.py.
"""
from __future__ import annotations

from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, make_predictor
from eval.runners.score_image_eval import _group_report_for, _hard_report

_predict = make_predictor("free_text", GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS, CLASS_CODES)


def _objects():
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


def test_hard_report_scores_both_sides():
    report_base = _hard_report(_objects(), "base_answer", _predict)
    report_equipped = _hard_report(_objects(), "equipped_answer", _predict)
    assert report_equipped["accuracy"] == 1.0  # both equipped answers correctly parse
    assert report_base["accuracy"] == 0.5  # one unparseable -> wrong


def test_group_report_scores_both_sides():
    report_equipped = _group_report_for(_objects(), "equipped_answer", _predict)
    assert report_equipped["mean_score"] == 1.0  # both correctly parsed and exact-matched
