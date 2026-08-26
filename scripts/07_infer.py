#!/usr/bin/env python
"""Ask a free-form question about a single, brand-new object — not part of the manifest/cache
pipeline. Runs the AION encoders live instead of reading from outputs/cache/.

Image input: a .npy file shaped (n_bands, H, W), band order matching configs/modalities.yaml's
    modalities.image.encoder.kwargs.bands.
Spectrum input: a .npz file with arrays `flux`, `wavelength` (both (L,)), optionally `ivar`,
    `mask` (default to fully-trusted/fully-unmasked if omitted — see aion_spectrum.py), plus
    `--spectrum-survey desi|sdss` (required, no safe default — see aion_spectrum.py's docstring
    for why AION needs to know real survey origin).

At least one of --image-npy / --spectrum-npz must be given; both may be given together to
condition the answer on both at once. --question is free text — the frozen base LLM's own
instruction-following is what's being relied on here, not something LoRA/the fusion stack were
specifically trained to do (they only ever saw one fixed captioning instruction during
training) — see inference.py's generate_caption docstring.

Usage:
    python scripts/07_infer.py --checkpoint-dir outputs/checkpoints/stage2/best \\
        --lora-dir outputs/checkpoints/stage2/best/lora --image-npy my_cutout.npy \\
        --question "What kind of object is this and why do you think so?"
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from captioner.inference import generate_caption, load_inference_model
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. outputs/checkpoints/stage2/best")
    parser.add_argument("--lora-dir", default=None, help="e.g. outputs/checkpoints/stage2/best/lora")
    parser.add_argument("--question", required=True, help="free-form question/instruction, e.g. "
                         "'What kind of object is this?'")
    parser.add_argument("--image-npy", default=None, help="path to a (n_bands, H, W) .npy file")
    parser.add_argument("--spectrum-npz", default=None, help="path to a .npz with flux/wavelength/...")
    parser.add_argument("--spectrum-survey", default=None, choices=["desi", "sdss"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(remaining_argv())

    if args.image_npy is None and args.spectrum_npz is None:
        parser.error("At least one of --image-npy or --spectrum-npz is required.")
    if args.spectrum_npz is not None and args.spectrum_survey is None:
        parser.error("--spectrum-survey is required when --spectrum-npz is given.")

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    logger.info("Loading model (this can take a few minutes with no output — that's normal)...")
    model, tokenizer, encoders = load_inference_model(cfg, args.checkpoint_dir, args.lora_dir, args.device)
    logger.info("Model loaded.")

    raw_inputs: dict = {}
    if args.image_npy is not None:
        pixel_values = torch.from_numpy(np.load(args.image_npy)).unsqueeze(0)  # (1, n_bands, H, W)
        raw_inputs["image"] = {"pixel_values": pixel_values}
    if args.spectrum_npz is not None:
        npz = np.load(args.spectrum_npz)
        flux = torch.from_numpy(npz["flux"]).unsqueeze(0).float()
        spectrum_batch: dict = {
            "flux": flux,
            "wavelength": torch.from_numpy(npz["wavelength"]).unsqueeze(0).float(),
            "survey": [args.spectrum_survey],
        }
        if "ivar" in npz:
            spectrum_batch["ivar"] = torch.from_numpy(npz["ivar"]).unsqueeze(0).float()
        if "mask" in npz:
            spectrum_batch["mask"] = torch.from_numpy(npz["mask"]).unsqueeze(0).bool()
        raw_inputs["spectra"] = spectrum_batch

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    answer = generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt.template, args.device,
        raw_inputs, max_new_tokens=args.max_new_tokens, question=args.question,
    )
    logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()
