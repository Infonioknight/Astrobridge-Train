"""eval/metrics/classification.py's classification_report — pure functions, exact-value
assertions against known confusion matrices, no fakes/GPU/network needed.
"""
from __future__ import annotations

import pytest

from eval.metrics.classification import classification_report


def test_perfect_predictions_score_1():
    y_true = ["a", "b", "a", "c"]
    y_pred = ["a", "b", "a", "c"]
    report = classification_report(y_true, y_pred, ["a", "b", "c"])
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    for label in ["a", "b", "c"]:
        assert report["per_class"][label]["precision"] == 1.0
        assert report["per_class"][label]["recall"] == 1.0
        assert report["per_class"][label]["f1"] == 1.0


def test_known_confusion_matrix():
    # true: a a a b b b — pred: a a b b b a
    # a: tp=2 (pos0,pos1), fn=1 (pos2 predicted b), fp=1 (pos5 true b predicted a)
    #   precision = 2/3, recall = 2/3, f1 = 2/3
    # b: tp=2 (pos3,pos4), fn=1 (pos5 predicted a), fp=1 (pos2 true a predicted b)
    #   precision = 2/3, recall = 2/3, f1 = 2/3
    y_true = ["a", "a", "a", "b", "b", "b"]
    y_pred = ["a", "a", "b", "b", "b", "a"]
    report = classification_report(y_true, y_pred, ["a", "b"])
    assert report["n"] == 6
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["per_class"]["a"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["a"]["recall"] == pytest.approx(2 / 3)
    assert report["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
    assert report["per_class"]["a"]["support"] == 3
    assert report["per_class"]["b"]["support"] == 3
    assert report["macro_f1"] == pytest.approx(2 / 3)


def test_none_predictions_count_as_wrong_not_dropped():
    y_true = ["a", "a"]
    y_pred = [None, "a"]
    report = classification_report(y_true, y_pred, ["a"])
    assert report["n"] == 2
    assert report["accuracy"] == 0.5
    assert report["per_class"]["a"]["recall"] == 0.5


def test_imbalanced_class_with_zero_predictions_gets_zero_precision_not_divide_by_zero():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "a", "a"]  # model never predicts "b" at all
    report = classification_report(y_true, y_pred, ["a", "b"])
    assert report["per_class"]["b"]["precision"] == 0.0
    assert report["per_class"]["b"]["recall"] == 0.0
    assert report["per_class"]["b"]["f1"] == 0.0
    assert report["per_class"]["b"]["support"] == 1


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="must match"):
        classification_report(["a", "b"], ["a"], ["a", "b"])


def test_empty_y_true_raises():
    with pytest.raises(ValueError, match="empty"):
        classification_report([], [], ["a"])


def test_true_label_missing_from_vocabulary_raises():
    with pytest.raises(ValueError, match="not in `labels`"):
        classification_report(["a", "c"], ["a", "c"], ["a", "b"])
