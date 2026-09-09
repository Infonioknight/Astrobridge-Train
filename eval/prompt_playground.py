#!/usr/bin/env python
"""Quick, manual prompt-engineering tool — NOT part of the formal eval bench (no sampling, no
metrics, no report). Just: load both models once, run them against the 2 saved GZ10 test images
in `test_subjects/`, print the raw answers, so you can edit `OOB_PROMPT`/`EQUIPPED_PROMPT` below
and re-run until the output style is what you want.

Real finding this exists to fix (from a real n=20 collection run): the base model's answers were
getting truncated mid-reasoning before ever stating a class, and the equipped model (fine-tuned
on captions) was mostly ignoring the classification instruction and reverting to its trained
free-form captioning style.

Two independent fixes applied here:
  1. Base model gets more generation room (OOB_MAX_NEW_TOKENS) since 5 tokens was cutting it off
     before it reached an answer at all — that's a budget problem, not a prompt problem.
  2. Both prompts switched to a digit-code format. Forcing a 1-token answer space and ending the
     prompt mid-completion ("Image class code:") gives the model much less room to drift into
     free text than asking it to name a class from a long list. The equipped model, being
     caption-tuned, is the harder case — if it still reverts to captioning under this prompt,
     that's a sign it needs decode-time constraint (logits processor / stop sequence) rather than
     further prompt tweaking, since the fine-tuning objective itself pulls toward verbose output.

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

# --- Class code mapping -----------------------------------------------------------------------
# Codes kept short (single digit / short token) so the model has minimal room to pad before
# emitting its answer, and so the answer is cheap to validate against MAX_NEW_TOKENS.
CLASS_CODES: dict[str, str] = {
    "0": "Disturbed Galaxies",
    "1": "Merging Galaxies",
    "2": "Round Smooth Galaxies",
    "3": "In-between Round Smooth Galaxies",
    "4": "Cigar Shaped Smooth Galaxies",
    "5": "Barred Spiral Galaxies",
    "6": "Unbarred Tight Spiral Galaxies",
    "7": "Unbarred Loose Spiral Galaxies",
    "8": "Edge-on Galaxies without Bulge",
    "9": "Edge-on Galaxies with Bulge",
}

_CODE_LEGEND = "\n".join(f"{code}={name}" for code, name in CLASS_CODES.items())

# --- Edit these two and re-run --------------------------------------------------------------
OOB_PROMPT = (
    "Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n"
    f"{_CODE_LEGEND}\n"
    "Image class code:"
)
EQUIPPED_PROMPT = OOB_PROMPT  # start identical; diverge once you see how each model actually responds

# Base model was truncating mid-reasoning at 5 tokens — give it room to actually land on an
# answer, then we just look at what it produced (no parsing requirement yet at this stage).
OOB_MAX_NEW_TOKENS = 40
# Equipped model: keep tight. If it needs more than this to state a code, that itself is the
# finding — it means the caption prior is winning even under a constrained-format prompt.
EQUIPPED_MAX_NEW_TOKENS = 8
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

    print("<OOB>")
    base_backend = get_backend(args.backend, side="base", cfg=cfg, device=args.device)
    for npy_path, png_path, true_label in TEST_IMAGES:
        image = Image.open(png_path)
        answer = base_backend.generate({"image": image}, OOB_PROMPT, OOB_MAX_NEW_TOKENS)
        print(answer)
    if args.backend == "local":
        free_local_backend(base_backend)

    print("<EQUIPPED>")
    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )
    for npy_path, png_path, true_label in TEST_IMAGES:
        pixel_values = np.load(npy_path)
        raw_inputs = {"image": {"pixel_values": torch.from_numpy(pixel_values).unsqueeze(0)}}
        answer = equipped_backend.generate(raw_inputs, EQUIPPED_PROMPT, EQUIPPED_MAX_NEW_TOKENS)
        print(answer)
    if args.backend == "local":
        free_local_backend(equipped_backend)


if __name__ == "__main__":
    main()