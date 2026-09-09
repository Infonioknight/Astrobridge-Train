"""eval/metrics/group_scoring.py — pure functions, no network/GPU."""
from __future__ import annotations

from eval.metrics.group_scoring import GALAXY10_GROUPS, group_report, group_score


def test_exact_match_scores_one():
    assert group_score("Round Smooth Galaxies", "Round Smooth Galaxies") == 1.0


def test_same_group_different_class_scores_half():
    assert group_score("Round Smooth Galaxies", "Cigar Shaped Smooth Galaxies") == 0.5


def test_different_group_scores_zero():
    assert group_score("Round Smooth Galaxies", "Barred Spiral Galaxies") == 0.0


def test_unparseable_prediction_scores_zero_not_excluded():
    assert group_score("Round Smooth Galaxies", None) == 0.0


def test_every_label_is_assigned_to_exactly_one_group():
    from eval.datasets.image_galaxy10 import GALAXY10_LABELS
    assert set(GALAXY10_GROUPS) == set(GALAXY10_LABELS)


def test_group_report_aggregates_correctly():
    y_true = ["Round Smooth Galaxies", "Round Smooth Galaxies", "Barred Spiral Galaxies"]
    y_pred = ["Round Smooth Galaxies", "Cigar Shaped Smooth Galaxies", "Merging Galaxies"]
    report = group_report(y_true, y_pred)
    # (1.0 + 0.5 + 0.0) / 3
    assert report["mean_score"] == 0.5
    assert report["n"] == 3
    assert report["per_group"]["Smooth"]["support"] == 2
    assert report["per_group"]["Smooth"]["mean_score"] == 0.75
    assert report["per_group"]["Spiral"]["support"] == 1
    assert report["per_group"]["Spiral"]["mean_score"] == 0.0


def test_group_report_handles_empty_group():
    y_true = ["Round Smooth Galaxies"]
    y_pred = ["Round Smooth Galaxies"]
    report = group_report(y_true, y_pred)
    assert report["per_group"]["Edge-On"]["support"] == 0
    assert report["per_group"]["Edge-On"]["mean_score"] is None
