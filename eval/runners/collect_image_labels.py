#!/usr/bin/env python
"""Step 1/2 of the image eval: run both models over a stratified Galaxy10 sample and save their
raw text answers — the compute-intensive step, meant to run once per sample. Step 2
(`score_image_eval.py`) is pure post-processing over this file's output: no GPU, no Modal, no
re-running the model — so scoring logic (the `caption_to_label` heuristic, the vote-fraction
combination rule) can be iterated on for free without ever paying for inference again.

**Every random decision is seeded and saved**: the sample itself (`--seed`, `--n`,
`--min-per-class`) is recorded in the output JSON, and greedy decoding (`do_sample=False` in both
`generate_caption` and `generate_qwen_native_vision_answer`) makes generation itself deterministic
given the same weights/hardware — the only real non-determinism left is floating-point reduction
order on GPU, which this project doesn't control and isn't claiming to.

Usage:
    uv run python -m eval.runners.collect_image_labels --n 150 --seed 0
    uv run python -m eval.runners.collect_image_labels --n 150 --seed 0 --backend modal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import free_local_backend, get_backend
from eval.datasets.image_galaxy10 import (
    CLASS_CODE_PROMPT,
    build_raw_inputs,
    decode_rgb_image,
    load_galaxy10_aion_bands,
    load_galaxy10_rgb_only,
    stratified_sample,
)

logger = get_logger(__name__)

# The free-text "name one of these 10 labels" prompt this used to default to was confirmed live
# (a real n=20 collection run, see git history) to produce ~5-15% parseable answers on either
# side — the model would ramble instead of naming a class. CLASS_CODE_PROMPT (digit-code format,
# confirmed live via eval/prompt_playground.py to get both models to answer compliantly and
# correctly) replaces it entirely, not just as a default — see eval/datasets/image_galaxy10.py's
# docstring for why the digit format itself matters, not just this specific wording.
DEFAULT_QUESTION = CLASS_CODE_PROMPT

# Base model needs real room to reach an answer (even with enable_thinking=False leaving a
# 1-token close tag) — equipped needs much less, and a tight budget here doubles as a signal:
# if it ever needs more than this to state a code, that itself means the caption-tuned prior is
# winning over the instruction, worth noticing rather than just papering over with more tokens.
DEFAULT_BASE_MAX_NEW_TOKENS = 40
DEFAULT_EQUIPPED_MAX_NEW_TOKENS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v3_qwen")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=150, help="total sample size across all 10 classes")
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="the ONE seed that determines the whole sample")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--base-max-new-tokens", type=int, default=DEFAULT_BASE_MAX_NEW_TOKENS)
    parser.add_argument("--equipped-max-new-tokens", type=int, default=DEFAULT_EQUIPPED_MAX_NEW_TOKENS)
    parser.add_argument(
        "--base-enable-thinking", action="store_true", default=False,
        help="Qwen/Qwen3.5-9B's own chat template opens a reasoning block by default (see "
             "captioner.inference.generate_qwen_native_vision_answer's docstring) — confirmed live "
             "that leaving it on makes the base model truncate mid-reasoning before ever reaching "
             "a digit code, regardless of --base-max-new-tokens. Off by default for exactly that "
             "reason; pass this flag to opt back into the model's real default behavior.",
    )
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/raw_generations/galaxy10_seed<seed>_n<n>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    rgb_table = load_galaxy10_rgb_only()
    sampled_rgb = stratified_sample(rgb_table, args.n, args.seed, args.min_per_class)
    sampled_ids = set(sampled_rgb["Galaxy10_DECals_index"])
    logger.info(f"Sampled {len(sampled_rgb)} objects (seed={args.seed}) covering all 10 classes.")

    bands_table = load_galaxy10_aion_bands()
    sampled_bands = bands_table[bands_table["Galaxy10_DECals_index"].isin(sampled_ids)]
    # Align to the exact same row order as sampled_rgb so per-object results line up 1:1 below.
    sampled_bands = sampled_bands.set_index("Galaxy10_DECals_index").loc[sampled_rgb["Galaxy10_DECals_index"]].reset_index()

    # --- Side 1: base model, over the whole sample, via image_rgb --------------------------
    base_backend = get_backend(
        args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=args.base_enable_thinking,
    )
    base_answers: dict[int, str] = {}
    for _, row in tqdm(sampled_rgb.iterrows(), total=len(sampled_rgb), desc="collect [base]"):
        image = decode_rgb_image(row["image_rgb"])
        base_answers[row["Galaxy10_DECals_index"]] = base_backend.generate(
            {"image": image}, args.question, args.base_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(base_backend)

    # --- Side 2: our equipped pipeline, over the whole sample, via image_bands --------------
    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )
    equipped_answers: dict[int, str] = {}
    for _, row in tqdm(sampled_bands.iterrows(), total=len(sampled_bands), desc="collect [equipped]"):
        raw_inputs = build_raw_inputs(row)
        equipped_answers[row["Galaxy10_DECals_index"]] = equipped_backend.generate(
            raw_inputs, args.question, args.equipped_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in sampled_rgb.iterrows():
        idx = row["Galaxy10_DECals_index"]
        objects.append({
            "Galaxy10_DECals_index": int(idx),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "label_name": row["label_name"],
            "base_answer": base_answers[idx],
            "equipped_answer": equipped_answers[idx],
        })

    output = {
        "dataset": "astronolan/galaxy10-aion",
        "repo_id": args.repo_id,
        "question": args.question,
        # "digit_code" tells score_image_eval.py to parse answers via predict_label_from_code
        # (against CLASS_CODES) rather than predict_label's free-text keyword matching — the two
        # parsers are not interchangeable, see caption_to_label.py's docstrings for why.
        "answer_format": "digit_code",
        "base_max_new_tokens": args.base_max_new_tokens,
        "equipped_max_new_tokens": args.equipped_max_new_tokens,
        "base_enable_thinking": args.base_enable_thinking,
        "sampling": {"n": args.n, "min_per_class": args.min_per_class, "seed": args.seed},
        "objects": objects,
    }

    out_path = Path(args.out) if args.out else Path(f"outputs/eval/raw_generations/galaxy10_seed{args.seed}_n{args.n}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(objects)} raw generations to {out_path}")


if __name__ == "__main__":
    main()
