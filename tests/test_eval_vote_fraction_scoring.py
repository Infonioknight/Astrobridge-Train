"""eval/metrics/vote_fraction_scoring.py — pure functions over hand-built rows, no network/GPU.
Real rule values transcribed from henrysky/Galaxy10's own construction notebook; these tests
check the *mechanics* (AND/min, OR/max, masking, the ==1.0 special case, ratio clipping), not the
rule values themselves.
"""
from __future__ import annotations

import pandas as pd
import pytest

from eval.metrics.vote_fraction_scoring import class_margin, score_prediction, soft_label_vector


def test_and_term_uses_the_weaker_of_two_conditions():
    # Barred Spiral: has-spiral-arms_yes > 0.8 AND bar_no < 0.2 — second condition barely fails.
    row = pd.Series({"has-spiral-arms_yes_debiased": 0.95, "bar_no_debiased": 0.25})
    margin = class_margin(row, "Barred Spiral Galaxies")
    spiral_margin = 1.0  # (0.95-0.8)/(1-0.8) = 0.75, actually let's just check it's the min
    bar_margin = (0.2 - 0.25) / 0.2  # negative -> clipped to 0
    assert margin == pytest.approx(max(0.0, bar_margin))


def test_or_alternative_uses_the_better_of_two_conditions():
    # Disturbed: (major-disturbance > 0.5 OR minor-disturbance > 0.7) AND merger < 0.2
    row = pd.Series({
        "merging_major-disturbance_debiased": 0.1,  # fails badly
        "merging_minor-disturbance_debiased": 0.9,  # passes well
        "merging_merger_debiased": 0.0,
    })
    margin = class_margin(row, "Disturbed Galaxies")
    # OR term should take the minor-disturbance branch (better), AND-combined with merger<0.2 (perfect)
    minor_margin = (0.9 - 0.7) / (1 - 0.7)
    assert margin == pytest.approx(min(minor_margin, 1.0))


def test_round_smooth_special_case_uses_value_itself_not_a_step_function():
    row = pd.Series({"how-rounded_round_debiased": 0.73})
    assert class_margin(row, "Round Smooth Galaxies") == pytest.approx(0.73)


def test_masked_column_makes_class_unscoreable():
    row = pd.Series({
        "has-spiral-arms_yes_debiased": 0.95,
        "has-spiral-arms_yes_debiased.mask": True,
        "bar_no_debiased": 0.1,
    })
    assert class_margin(row, "Barred Spiral Galaxies") is None


def test_missing_column_treated_same_as_masked():
    row = pd.Series({"bar_no_debiased": 0.1})  # has-spiral-arms columns entirely absent
    assert class_margin(row, "Barred Spiral Galaxies") is None


def test_or_term_survives_if_only_one_alternative_is_masked():
    row = pd.Series({
        "merging_major-disturbance_debiased.mask": True,
        "merging_major-disturbance_debiased": 0.9,  # masked, must be ignored
        "merging_minor-disturbance_debiased": 0.9,  # real, unmasked
        "merging_merger_debiased": 0.0,
    })
    margin = class_margin(row, "Disturbed Galaxies")
    assert margin is not None
    assert margin == pytest.approx(min((0.9 - 0.7) / 0.3, 1.0))


def test_soft_label_vector_anchors_true_class_at_exactly_one():
    row = pd.Series({"how-rounded_round_debiased": 0.6})
    vector = soft_label_vector(row, "Round Smooth Galaxies")
    assert vector["Round Smooth Galaxies"] == 1.0


def test_soft_label_vector_none_when_true_class_unscoreable():
    row = pd.Series({"how-rounded_round_debiased.mask": True, "how-rounded_round_debiased": 0.6})
    assert soft_label_vector(row, "Round Smooth Galaxies") is None


def test_soft_label_vector_ratio_is_clipped_to_one():
    # A competing class can't score HIGHER than the true class after rescaling.
    row = pd.Series({
        "how-rounded_round_debiased": 0.5,
        "smooth-or-featured_smooth_debiased": 0.99,
        "how-rounded_cigar-shaped_debiased": 0.99,  # would exceed true class's own raw margin
    })
    vector = soft_label_vector(row, "Round Smooth Galaxies")
    assert vector["Cigar Shaped Smooth Galaxies"] <= 1.0


def test_score_prediction_unparseable_caption_is_none_not_zero():
    row = pd.Series({"how-rounded_round_debiased": 0.9})
    vector = soft_label_vector(row, "Round Smooth Galaxies")
    assert score_prediction(vector, None) is None


def test_score_prediction_correct_guess_scores_one():
    row = pd.Series({"how-rounded_round_debiased": 0.9})
    vector = soft_label_vector(row, "Round Smooth Galaxies")
    assert score_prediction(vector, "Round Smooth Galaxies") == 1.0
