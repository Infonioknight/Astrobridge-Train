"""train/loop.py must never hand a fully-masked-caption micro-batch to the model. Confirmed
mechanism: HF's CrossEntropyLoss reduces (mean) over every valid label position in the batch; if
every sample's caption_attn_mask is all-zero, that's 0 valid positions, i.e. 0/0 = NaN. That NaN
loss backpropagates into every trainable parameter's gradient (they're shared across the batch),
which then permanently poisons Adam's exp_avg/exp_avg_sq moving averages for every parameter —
there's no way for a plain AdamW step to self-heal from one NaN gradient. This is a second line
of defense on top of data/dataset.py's caption-existence-aware subset sampling — belt and braces
for any other future way a degenerate batch could slip through.
"""
from __future__ import annotations

import torch

from captioner.train.loop import _is_degenerate_batch


def _batch(caption_attn_mask: torch.Tensor) -> dict:
    return {"caption_attn_mask": caption_attn_mask, "object_id": ["a", "b"], "shown": [frozenset(), frozenset()]}


def test_all_zero_mask_is_degenerate():
    assert _is_degenerate_batch(_batch(torch.zeros(2, 5, dtype=torch.long))) is True


def test_any_real_caption_token_is_not_degenerate():
    mask = torch.zeros(2, 5, dtype=torch.long)
    mask[0, 0] = 1  # just one real token in one sample is enough to make the batch usable
    assert _is_degenerate_batch(_batch(mask)) is False


def test_fully_real_mask_is_not_degenerate():
    assert _is_degenerate_batch(_batch(torch.ones(2, 5, dtype=torch.long))) is False


def test_empty_caption_dimension_is_degenerate():
    # C=0 (every sample tokenized to zero caption tokens) — `.any()` on an empty tensor is False.
    assert _is_degenerate_batch(_batch(torch.zeros(2, 0, dtype=torch.long))) is True
