"""SHUFFLED_CLASS_CODES (eval/datasets/image_galaxy10.py) and the file-self-describing mapping
mechanism both score_image_eval.py and score_image_eval_debiased.py read `class_codes` from —
the critical correctness property: scoring a shuffled-codes collect file MUST resolve digits
against the mapping THAT FILE actually used, never silently fall back to the unshuffled default
(which would map every digit to the wrong class).
"""
from __future__ import annotations

from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS, SHUFFLED_CLASS_CODES
from eval.metrics.caption_to_label import predict_label_from_code


def test_shuffled_codes_is_a_full_derangement():
    # Every one of the 10 codes must map to a DIFFERENT class than the default — the whole
    # diagnostic point breaks if even one code accidentally keeps its original class.
    for code in CLASS_CODES:
        assert SHUFFLED_CLASS_CODES[code] != CLASS_CODES[code]


def test_shuffled_codes_covers_all_ten_labels_exactly_once():
    assert sorted(SHUFFLED_CLASS_CODES.values()) == sorted(GALAXY10_LABELS)
    assert len(set(SHUFFLED_CLASS_CODES.values())) == 10


def test_same_digit_resolves_to_different_labels_under_each_mapping():
    # The real failure mode this whole mechanism guards against: scoring "5" against the wrong
    # mapping would silently give a completely different (and wrong) label.
    default_label = predict_label_from_code("5", CLASS_CODES)
    shuffled_label = predict_label_from_code("5", SHUFFLED_CLASS_CODES)
    assert default_label != shuffled_label
    assert default_label == "Barred Spiral Galaxies"
    assert shuffled_label == "Disturbed Galaxies"


def test_scoring_a_shuffled_file_without_reading_its_class_codes_would_be_wrong():
    """Simulates the exact bug this mechanism prevents: an object whose true label is 'Disturbed
    Galaxies' answered correctly under the SHUFFLED mapping (code 5) — scoring it against the
    DEFAULT mapping instead would silently mark a correct answer as wrong.
    """
    true_label = "Disturbed Galaxies"
    equipped_answer = " 5"  # correct under SHUFFLED_CLASS_CODES

    correct_parse = predict_label_from_code(equipped_answer, SHUFFLED_CLASS_CODES)
    wrong_parse = predict_label_from_code(equipped_answer, CLASS_CODES)  # the bug this guards against

    assert correct_parse == true_label
    assert wrong_parse != true_label
