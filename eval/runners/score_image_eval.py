#!/usr/bin/env python
"""Step 2/2 of the image eval: score a `collect_image_labels.py` output file. Pure post-
processing — no GPU, no Modal, no model at all — so this can be re-run as many times as needed
(different `caption_to_label` heuristics, different crossmatch radius) without ever re-paying for
inference.

Reports three genuinely different metrics side by side, both for base and equipped:
  - **hard**: hard-label accuracy + per-class precision/recall/F1 (`eval.metrics.classification`)
    — the model's answer either exactly matches the true label or it doesn't.
  - **soft**: crowd-vote-fraction-grounded partial credit (`eval.metrics.vote_fraction_scoring`)
    — a wrong answer scores how plausible it actually was, according to real Galaxy Zoo DECaLS
    volunteer votes for that specific object, not a flat 0. Requires the RA/Dec crossmatch against
    `astronolan/gz-decals-embeddings` (`eval.datasets.gz_decals_votes`); objects with no crossmatch
    within `--crossmatch-radius-arcsec`, or whose OWN true class has no reliable vote data, are
    excluded from the soft score (counted and reported, never silently dropped).
  - **group**: coarse 1.0/0.5/0.0 partial credit by the 4 top-level morphology groups (Smooth,
    Spiral, Edge-On, Disturbed/Merging — `eval.metrics.group_scoring`). Deliberately kept as a
    complement to `soft`, not a replacement — simpler and needs no crossmatch, but coarser and
    not empirically grounded the way the vote-fraction score is; see that module's docstring for
    the real tradeoffs between the two.

Usage:
    uv run python -m eval.runners.score_image_eval --in outputs/eval/raw_generations/galaxy10_seed0_n150.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from captioner.utils.logging import get_logger
from eval.datasets.gz_decals_votes import crossmatch_to_sample, load_vote_fractions
from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, predict_label, predict_label_from_code
from eval.metrics.classification import classification_report
from eval.metrics.group_scoring import group_report
from eval.metrics.vote_fraction_scoring import score_prediction, soft_label_vector

logger = get_logger(__name__)


def _make_predictor(answer_format: str):
    """Picks the parser matching how `collect_image_labels.py` actually prompted the model —
    `predict_label` (free-text keyword matching) and `predict_label_from_code` (digit-code
    matching) are NOT interchangeable; using the wrong one silently returns `None` for every
    object rather than erroring, which is exactly the kind of failure worth guarding against here.
    `answer_format` is read from the collect file itself, not assumed, so scoring an older file
    (pre-digit-code, no `answer_format` key at all) still uses the matcher it was actually
    generated against.
    """
    if answer_format == "digit_code":
        return lambda answer: predict_label_from_code(answer, CLASS_CODES)
    return lambda answer: predict_label(answer, GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS)


def _hard_report(objects: list[dict], answer_key: str, predict) -> dict:
    y_true = [o["label_name"] for o in objects]
    y_pred = [predict(o[answer_key]) for o in objects]
    return classification_report(y_true, y_pred, GALAXY10_LABELS)


def _group_report_for(objects: list[dict], answer_key: str, predict) -> dict:
    y_true = [o["label_name"] for o in objects]
    y_pred = [predict(o[answer_key]) for o in objects]
    return group_report(y_true, y_pred)


def _soft_report(crossmatched: pd.DataFrame, answer_key: str, predict) -> dict:
    scores = []
    n_excluded_no_crossmatch = 0
    n_excluded_true_class_unscoreable = 0
    for _, row in crossmatched.iterrows():
        if not row["_crossmatched"]:
            n_excluded_no_crossmatch += 1
            continue
        vector = soft_label_vector(row, row["label_name"])
        if vector is None:
            n_excluded_true_class_unscoreable += 1
            continue
        predicted = predict(row[answer_key])
        score = score_prediction(vector, predicted)
        if score is not None:
            scores.append(score)

    return {
        "n_scored": len(scores),
        "n_excluded_no_crossmatch": n_excluded_no_crossmatch,
        "n_excluded_true_class_unscoreable": n_excluded_true_class_unscoreable,
        "n_excluded_predicted_class_unscoreable_or_unparseable": len(crossmatched) - len(scores) - n_excluded_no_crossmatch - n_excluded_true_class_unscoreable,
        "mean_soft_score": (sum(scores) / len(scores)) if scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_image_labels.py output JSON")
    parser.add_argument("--crossmatch-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _scored.json")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    answer_format = data.get("answer_format", "free_text")  # older collect files predate this key
    predict = _make_predictor(answer_format)
    logger.info(
        f"Scoring {len(objects)} objects from {args.in_path} (sampling seed={data['sampling']['seed']}, "
        f"answer_format={answer_format!r})."
    )

    report = {
        "source": args.in_path,
        "dataset": data["dataset"],
        "repo_id": data["repo_id"],
        "answer_format": answer_format,
        "sampling": data["sampling"],
        "hard": {
            "base": _hard_report(objects, "base_answer", predict),
            "equipped": _hard_report(objects, "equipped_answer", predict),
        },
        "group": {
            "base": _group_report_for(objects, "base_answer", predict),
            "equipped": _group_report_for(objects, "equipped_answer", predict),
        },
    }

    logger.info("Crossmatching sample against astronolan/gz-decals-embeddings for soft scoring...")
    sample_df = pd.DataFrame(objects)
    votes = load_vote_fractions()
    crossmatched = crossmatch_to_sample(sample_df, votes, radius_arcsec=args.crossmatch_radius_arcsec)

    report["soft"] = {
        "base": _soft_report(crossmatched, "base_answer", predict),
        "equipped": _soft_report(crossmatched, "equipped_answer", predict),
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote scored report to {out_path}")


if __name__ == "__main__":
    main()
