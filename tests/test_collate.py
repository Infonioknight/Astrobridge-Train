from __future__ import annotations

import numpy as np
import torch

from captioner.data.collate import collate_batch


def _example(shown_image: bool, shown_spectra: bool, image_T: int = 3, spectra_T: int = 2):
    return {
        "object_id": "obj",
        "shown": frozenset({"image"} if shown_image else set()) | frozenset({"spectra"} if shown_spectra else set()),
        "modality_arrays": {
            "image": np.random.randn(image_T, 8).astype(np.float32) if shown_image else None,
            "spectra": np.random.randn(spectra_T, 6).astype(np.float32) if shown_spectra else None,
        },
        "prompt_ids": torch.tensor([1, 2, 3]),
        "caption_ids": torch.tensor([4, 5]),
    }


def test_absent_modality_is_fully_masked():
    examples = [_example(shown_image=True, shown_spectra=False)]
    batch = collate_batch(
        examples, ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0
    )
    assert torch.all(batch["modality_batch"]["spectra"]["mask"])
    assert not torch.all(batch["modality_batch"]["image"]["mask"])


def test_present_modality_padding_mask_matches_token_count():
    examples = [_example(shown_image=True, shown_spectra=True, image_T=3)]
    batch = collate_batch(
        examples, ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0
    )
    mask = batch["modality_batch"]["image"]["mask"][0]
    assert (~mask).sum().item() == 3
    assert mask.sum().item() == 2


def test_caption_attn_mask_marks_real_tokens_only():
    ex_short = _example(True, True)
    ex_short["caption_ids"] = torch.tensor([9])
    ex_long = _example(True, True)
    ex_long["caption_ids"] = torch.tensor([9, 9, 9])

    batch = collate_batch(
        [ex_short, ex_long], ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0
    )
    assert batch["caption_attn_mask"][0].tolist() == [1, 0, 0]
    assert batch["caption_attn_mask"][1].tolist() == [1, 1, 1]
