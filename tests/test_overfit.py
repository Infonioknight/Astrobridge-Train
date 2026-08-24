"""§9 step 5: the cheapest end-to-end wiring check. If the visual prefix isn't actually reaching
the loss (wrong concatenation order, detached tensor, wrong labels), 8 examples won't memorise.
"""
from __future__ import annotations

import torch

from tests.conftest import make_modality_batch


def test_middle_stack_can_memorise_8_examples(captioner, modality_out_dims):
    torch.manual_seed(0)
    B, P, C = 8, 2, 4
    T_max = {"image": 6, "spectra": 4}

    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    prompt_ids = torch.randint(0, 20, (B, P))
    caption_ids = torch.randint(20, 40, (B, C))  # distinct vocab range from prompt, fixed target

    optimizer = torch.optim.AdamW(captioner.parameters(), lr=3e-3)

    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        out = captioner(modality_batch, prompt_ids, caption_ids)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    assert losses[-1] < 0.05, f"Failed to memorise 8 examples; final loss={losses[-1]:.4f} (started at {losses[0]:.4f})"
    assert losses[-1] < losses[0]
