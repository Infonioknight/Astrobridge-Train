"""collate_batch output feeds Captioner.forward with the exact keys train/loop.py passes — the
one seam not covered by the per-unit tests. Uses the CPU TinyCausalLM fixture.
"""
from __future__ import annotations

import numpy as np
import torch

from captioner.data.collate import collate_batch


def _example(pre, post, cap, image_T=3, spectra_T=2):
    return {
        "object_id": "obj",
        "shown": frozenset({"image", "spectra"}),
        "modality_arrays": {
            "image": np.random.randn(image_T, 8).astype(np.float32),
            "spectra": np.random.randn(spectra_T, 6).astype(np.float32),
        },
        "pre_ids": torch.tensor(pre),
        "post_ids": torch.tensor(post),
        "caption_ids": torch.tensor(cap),
    }


def test_collate_output_runs_through_captioner_forward(captioner, modality_out_dims):
    examples = [
        _example([1, 2, 3], [4, 5], [10, 11, 12]),
        _example([1, 2], [4, 5, 6, 7], [10, 11]),
    ]
    batch = collate_batch(
        examples, ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0
    )

    # exactly how src/captioner/train/loop.py calls it
    out = captioner(
        batch["modality_batch"],
        batch["pre_ids"],
        batch["post_ids"],
        batch["caption_ids"],
        batch["pre_attn_mask"],
        batch["post_attn_mask"],
        batch["caption_attn_mask"],
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in captioner.fusion_stack.parameters())
