#!/usr/bin/env python
"""Step 2/2 of the lightcurve eval: score a `collect_lightcurve_labels.py` output file. Pure
post-processing — no GPU, no Modal, no model, no network — mirrors `score_image_eval.py` exactly,
minus the group/debiased-vote-fraction metrics (both are Galaxy-Zoo-specific: `group_scoring`'s
morphology groups and `vote_fraction_scoring`'s crowd-vote crossmatch have no SN-typing analog),
so just hard-label accuracy + per-class precision/recall/F1, both sides.

Usage:
    uv run python -m eval.runners.score_lightcurve_eval --in outputs/eval/raw_generations/yse_lightcurve_only_seed0_nall.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.logging import get_logger
from eval.datasets.lightcurve_yse import SN_CLASS_CODES, SN_LABELS
from eval.metrics.caption_to_label import SN_TYPE_SYNONYMS, make_predictor
from eval.metrics.classification import classification_report

logger = get_logger(__name__)


def _hard_report(objects: list[dict], answer_key: str, predict) -> dict:
    y_true = [o["label_name"] for o in objects]
    y_pred = [predict(o[answer_key]) for o in objects]
    report = classification_report(y_true, y_pred, SN_LABELS)

    # Parse rate is a first-class number on the verbose path, not a footnote: tolerating long
    # answers trades guaranteed-parseable output for answers that play to the model's trained
    # voice, so how often that trade actually lands has to be visible. Unparsed answers are still
    # counted as wrong in the metrics above (classification_report treats None as incorrect) —
    # this reports them separately as well, never instead.
    n_unparsed = sum(1 for p in y_pred if p is None)
    report["parsing"] = {
        "n_unparsed": n_unparsed,
        "parse_rate": (len(y_pred) - n_unparsed) / len(y_pred) if y_pred else 0.0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_lightcurve_labels.py output JSON")
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _scored.json")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    answer_format = data.get("answer_format", "free_text")  # older files predate this key
    # Same reasoning as score_image_eval.py's identical line: read the mapping THIS file actually
    # used rather than assuming the current global default.
    file_class_codes = data.get("class_codes", SN_CLASS_CODES)
    predict = make_predictor(answer_format, SN_LABELS, SN_TYPE_SYNONYMS, file_class_codes)
    logger.info(
        f"Scoring {len(objects)} objects from {args.in_path} (track={data.get('track')}, "
        f"sampling={data.get('sampling')}, answer_format={answer_format!r})."
    )

    report = {
        "source": args.in_path,
        "dataset": data["dataset"],
        "track": data.get("track"),
        "repo_id": data["repo_id"],
        "answer_format": answer_format,
        "class_codes": file_class_codes,
        "sampling": data["sampling"],
        "hard": {
            "base": _hard_report(objects, "base_answer", predict),
            "equipped": _hard_report(objects, "equipped_answer", predict),
        },
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote scored report to {out_path}")


if __name__ == "__main__":
    main()
