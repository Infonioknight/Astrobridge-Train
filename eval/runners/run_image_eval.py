#!/usr/bin/env python
"""Galaxy morphology classification eval against `astronolan/galaxy10-aion`'s real test split
(796 objects, 10 Galaxy Zoo classes — see `eval/datasets/image_galaxy10.py`).

Two tracks (`--track`):
  - `base_only` (default): base model only, via the pre-rendered `image_rgb` field — no AION
    involved, no band-identity question to worry about, safe to run today.
  - `equipped_and_base`: also runs our equipped pipeline (AION + fusion stack + LoRA) via the
    `image_bands` field, applying the griz-minus-`i` band-selection hypothesis documented in
    `eval/datasets/image_galaxy10.py`'s module docstring — **prints a loud warning every run**
    since that hypothesis is not yet empirically verified against this specific dataset. Verify
    it first (render a grz composite from a few rows and compare against those same rows'
    `image_rgb`) before trusting `equipped` numbers from this track.

Usage:
    uv run python eval/runners/run_image_eval.py --track base_only
    uv run python eval/runners/run_image_eval.py --track equipped_and_base --backend modal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import get_backend
from eval.datasets.image_galaxy10 import (
    GALAXY10_LABELS,
    build_raw_inputs,
    decode_rgb_image,
    load_galaxy10_aion_bands,
    load_galaxy10_rgb_only,
)
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, predict_label
from eval.metrics.classification import classification_report

logger = get_logger(__name__)

DEFAULT_QUESTION = (
    "Classify this galaxy's morphology. Choose exactly one: "
    + ", ".join(GALAXY10_LABELS) + "."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v3_qwen")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", choices=["base_only", "equipped_and_base"], default="base_only")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N objects (quick smoke test)")
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/classification/galaxy10_<track>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    rgb_table = load_galaxy10_rgb_only()
    if args.limit is not None:
        rgb_table = rgb_table.head(args.limit)

    y_true = rgb_table["label_name"].tolist()

    base_backend = get_backend(args.backend, side="base", cfg=cfg, device=args.device)
    y_pred_base: list[str | None] = []
    for _, row in tqdm(rgb_table.iterrows(), total=len(rgb_table), desc="image eval [base]"):
        image = decode_rgb_image(row["image_rgb"])
        caption = base_backend.generate({"image": image}, args.question, args.max_new_tokens)
        y_pred_base.append(predict_label(caption, GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS))

    report = {
        "dataset": "astronolan/galaxy10-aion",
        "track": args.track,
        "repo_id": args.repo_id,
        "base": classification_report(y_true, y_pred_base, GALAXY10_LABELS),
    }

    if args.track == "equipped_and_base":
        logger.warning(
            "Running the AION-encoder image track with an UNVERIFIED band-selection hypothesis "
            "(griz-minus-i — see eval/datasets/image_galaxy10.py's module docstring). Verify "
            "before trusting these numbers."
        )
        bands_table = load_galaxy10_aion_bands()
        if args.limit is not None:
            bands_table = bands_table.head(args.limit)
        bands_table = bands_table.set_index("Galaxy10_DECals_index").loc[rgb_table["Galaxy10_DECals_index"]].reset_index()

        equipped_backend = get_backend(
            args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
        )
        y_pred_equipped: list[str | None] = []
        for _, row in tqdm(bands_table.iterrows(), total=len(bands_table), desc="image eval [equipped]"):
            raw_inputs = build_raw_inputs(row)
            caption = equipped_backend.generate(raw_inputs, args.question, args.max_new_tokens)
            y_pred_equipped.append(predict_label(caption, GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS))
        report["equipped"] = classification_report(y_true, y_pred_equipped, GALAXY10_LABELS)

    out_path = Path(args.out) if args.out else Path(f"outputs/eval/classification/galaxy10_{args.track}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
