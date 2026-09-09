"""eval/runners/score_lightcurve_eval.py's _hard_report — pure logic over a synthetic
collect-style objects list, no network/GPU/model. Mirrors test_eval_score_image_eval.py.
"""
from __future__ import annotations

from eval.datasets.lightcurve_yse import SN_CLASS_CODES, SN_LABELS
from eval.metrics.caption_to_label import SN_TYPE_SYNONYMS, make_predictor
from eval.runners.score_lightcurve_eval import _hard_report

_predict = make_predictor("digit_code", SN_LABELS, SN_TYPE_SYNONYMS, SN_CLASS_CODES)


def _objects():
    return [
        {"object_id": "obj1", "label_name": "SN Ia", "base_answer": " 0", "equipped_answer": " 0"},
        {"object_id": "obj2", "label_name": "SN Ibc", "base_answer": " not sure", "equipped_answer": " 2"},
    ]


def test_hard_report_scores_both_sides():
    report_base = _hard_report(_objects(), "base_answer", _predict)
    report_equipped = _hard_report(_objects(), "equipped_answer", _predict)
    assert report_equipped["accuracy"] == 1.0  # both digit codes correctly parse
    assert report_base["accuracy"] == 0.5  # second base answer is unparseable -> wrong


def test_hard_report_reads_class_codes_from_the_predictor_not_a_hardcoded_default():
    shuffled_codes = {"0": "SN Ibc", "1": "SN Ia", "2": "SN II"}
    predict_shuffled = make_predictor("digit_code", SN_LABELS, SN_TYPE_SYNONYMS, shuffled_codes)
    objects = [{"object_id": "obj1", "label_name": "SN Ibc", "base_answer": " 0", "equipped_answer": " 0"}]
    report = _hard_report(objects, "equipped_answer", predict_shuffled)
    assert report["accuracy"] == 1.0
