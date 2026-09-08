#!/usr/bin/env python
"""Quick, manual prompt-engineering tool — NOT part of the formal eval bench (no sampling, no
metrics, no report). Just: load both models once, run them against the 2 saved GZ10 test images
in `test_subjects/`, print the raw answers, so you can edit `OOB_PROMPT`/`EQUIPPED_PROMPT` below
and re-run until the output style is what you want.

Real finding this exists to fix (from a real n=20 collection run): the base model's answers were
getting truncated mid-reasoning before ever stating a class, and the equipped model was mostly
ignoring the classification instruction and reverting to its trained free-form captioning style.
Edit the two prompts below (and `MAX_NEW_TOKENS` if the base model still needs more room to reach
an answer) and re-run — much faster than iterating through the full collect_image_labels.py
pipeline for every wording change.

Usage:
    uv run python -m eval.prompt_playground
    uv run python -m eval.prompt_playground --backend modal   # needs `modal deploy eval/backend.py` first
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from captioner.utils.config import load_config
from eval.backend import free_local_backend, get_backend

# --- Edit these two and re-run --------------------------------------------------------------
OOB_PROMPT = (
    "Classify this galaxy's morphology. Choose exactly one: Disturbed Galaxies, Merging "
    "Galaxies, Round Smooth Galaxies, In-between Round Smooth Galaxies, Cigar Shaped Smooth "
    "Galaxies, Barred Spiral Galaxies, Unbarred Tight Spiral Galaxies, Unbarred Loose Spiral "
    "Galaxies, Edge-on Galaxies without Bulge, Edge-on Galaxies with Bulge."
)
EQUIPPED_PROMPT = OOB_PROMPT  # start identical; diverge once you see how each model actually responds
MAX_NEW_TOKENS = 64
# ---------------------------------------------------------------------------------------------

TEST_IMAGES = [
    ("test_subjects/gz10_image_01.npy", "test_subjects/gz10_image_01.png", "Round Smooth Galaxies"),
    ("test_subjects/gz10_image_02.npy", "test_subjects/gz10_image_02.png", "Barred Spiral Galaxies"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v3_qwen")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    print(f"=== OOB_PROMPT ===\n{OOB_PROMPT}\n")
    base_backend = get_backend(args.backend, side="base", cfg=cfg, device=args.device)
    for npy_path, png_path, true_label in TEST_IMAGES:
        image = Image.open(png_path)
        answer = base_backend.generate({"image": image}, OOB_PROMPT, MAX_NEW_TOKENS)
        print(f"--- {npy_path} (true: {true_label}) ---\n{answer}\n")
    if args.backend == "local":
        free_local_backend(base_backend)

    print(f"=== EQUIPPED_PROMPT ===\n{EQUIPPED_PROMPT}\n")
    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )
    for npy_path, png_path, true_label in TEST_IMAGES:
        pixel_values = np.load(npy_path)
        raw_inputs = {"image": {"pixel_values": torch.from_numpy(pixel_values).unsqueeze(0)}}
        answer = equipped_backend.generate(raw_inputs, EQUIPPED_PROMPT, MAX_NEW_TOKENS)
        print(f"--- {npy_path} (true: {true_label}) ---\n{answer}\n")
    if args.backend == "local":
        free_local_backend(equipped_backend)


if __name__ == "__main__":
    main()
