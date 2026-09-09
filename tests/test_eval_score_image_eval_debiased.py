"""eval/runners/score_image_eval_debiased.py's _soft_report — pure logic over a synthetic
collect-style objects list + a synthetic vote-fraction table, no network/GPU/model.
"""
from __future__ import annotations

import pandas as pd

from eval.datasets.gz_decals_votes import crossmatch_to_sample
from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, make_predictor
from eval.runners.score_image_eval_debiased import _soft_report

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


def test_soft_report_excludes_unmatched_objects():
    votes = pd.DataFrame({
        "ra": [10.0], "dec": [0.0], "iauname": ["x"],
        "how-rounded_round_debiased": [0.9], "how-rounded_round_debiased.mask": [False],
        "merging_merger_debiased": [0.0], "merging_merger_debiased.mask": [False],
    })
    # crossmatch_to_sample needs every _VOTE_COLUMNS/_MASK_COLUMNS present; reindex fills the rest with NaN/None
    from eval.datasets.gz_decals_votes import _MASK_COLUMNS, _VOTE_COLUMNS
    votes = votes.reindex(columns=["ra", "dec", "iauname"] + _VOTE_COLUMNS + _MASK_COLUMNS)

    sample_df = pd.DataFrame(_objects())
    crossmatched = crossmatch_to_sample(sample_df, votes, radius_arcsec=1.0)

    report = _soft_report(crossmatched, "equipped_answer", _predict)
    # object 1 (ra=10.0) crossmatches; object 2 (ra=20.0) doesn't -> excluded
    assert report["n_excluded_no_crossmatch"] == 1
    assert report["n_scored"] <= 1
