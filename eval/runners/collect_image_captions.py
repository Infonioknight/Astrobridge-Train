#!/usr/bin/env python
"""Plain-captioning collection, one side at a time — no classification framing, no digit/logprob
scoring. `--side equipped` (default): feed each object through the equipped model exactly as it
expects (real AION `image_bands`, the trained chat-template prompt with its DEFAULT system and
instruction — `question=None`/`system=None` fall back to `configs/model.yaml`'s
`system_variants[0]`/`instruction_variants[0]`, the same deterministic default every other
`generate_caption` call in this eval bench uses). `--side base`: the plain, never-LoRA'd Qwen3.5-9B
over the same sample via its native vision pathway, given a plain "describe this image" instruction
— no classification framing there either.

Deliberately writes each side to its OWN file (not merged) — pass `--out` explicitly, or rely on
the side-specific default (`gz10_images.json` / `gz10_images_base.json`). Same `--seed`/`--n`
against the same underlying Galaxy10 test set draws the same objects on both sides (confirmed real:
`load_galaxy10_rgb_only`/`load_galaxy10_aion_bands` read the same parquet files in the same
deterministic order, just different columns), so the two files are directly comparable object-for-
object even though they were never collected in the same run.

This is deliberately NOT collect_image_labels.py: no CLASS_CODE_PROMPT, no REASONING_PROMPT, no
system override — each model captions in its own natural voice, at whatever length it naturally
produces, up to `--max-new-tokens`.

Usage:
    uv run python -m eval.runners.collect_image_captions --n 200 --seed 0 --backend modal
    uv run python -m eval.runners.collect_image_captions --side base --n 200 --seed 0 --backend modal
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
    build_raw_inputs,
    decode_rgb_image,
    load_galaxy10_aion_bands,
    load_galaxy10_rgb_only,
    stratified_sample,
)

logger = get_logger(__name__)

DEFAULT_MAX_NEW_TOKENS = 300

# Base has no equivalent to equipped's trained instruction_variants default — Qwen's native chat
# template needs an explicit question. Deliberately plain, no classification framing, matching
# "the model captions in its own natural voice" for the equipped side.
BASE_CAPTION_QUESTION = "Describe this galaxy image in detail."

_OUT_BY_SIDE = {
    "equipped": "outputs/eval/raw_generations/gz10_images.json",
    "base": "outputs/eval/raw_generations/gz10_images_base.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["equipped", "base"], default="equipped")
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v5")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=150, help="total sample size across all 10 classes")
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="the ONE seed that determines the whole sample")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--base-enable-thinking", action="store_true", default=False,
        help="--side base only. Off by default — Qwen/Qwen3.5-9B's own chat template opens a "
             "reasoning block by default, confirmed live elsewhere in this eval bench to truncate "
             "the actual answer before it's ever reached at a bounded token budget.",
    )
    parser.add_argument("--out", default=None, help="defaults to gz10_images.json (equipped) or gz10_images_base.json (base)")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    if args.side == "equipped":
        table = load_galaxy10_aion_bands()
    else:
        table = load_galaxy10_rgb_only()
    sample = stratified_sample(table, args.n, args.seed, args.min_per_class, label_col="label_name")
    logger.info(f"Sampled {len(sample)} objects (seed={args.seed}, side={args.side!r}) covering all 10 classes.")

    if args.side == "equipped":
        backend = get_backend(
            args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
        )
    else:
        backend = get_backend(
            args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=args.base_enable_thinking,
        )

    objects = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"collect [{args.side} captions]"):
        if args.side == "equipped":
            raw_inputs = build_raw_inputs(row)
            caption = backend.generate(raw_inputs, None, args.max_new_tokens)
        else:
            image = decode_rgb_image(row["image_rgb"])
            caption = backend.generate({"image": image}, BASE_CAPTION_QUESTION, args.max_new_tokens)
        objects.append({
            "Galaxy10_DECals_index": int(row["Galaxy10_DECals_index"]),
            "label_name": row["label_name"],
            "caption": caption,
        })
    if args.backend == "local":
        free_local_backend(backend)

    output = {
        "dataset": "astronolan/galaxy10-aion",
        "side": args.side,
        "repo_id": args.repo_id if args.side == "equipped" else None,
        "question": None if args.side == "equipped" else BASE_CAPTION_QUESTION,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {
            "n": args.n, "min_per_class": args.min_per_class, "seed": args.seed,
            "actual_counts": sample["label_name"].value_counts().to_dict(),
        },
        "objects": objects,
    }

    out_path = Path(args.out) if args.out else Path(_OUT_BY_SIDE[args.side])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(objects)} captions to {out_path}")


if __name__ == "__main__":
    main()
