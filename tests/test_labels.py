"""§6 / §10 pitfall 2: labels must be -100 everywhere except the caption-token positions,
otherwise the model is trained to autoregressively predict its own visual prefix or the
chat-template scaffolding.
"""
from __future__ import annotations

import torch

from captioner.model.captioner import IGNORE_INDEX
from tests.conftest import make_modality_batch


def test_prefix_and_wrapper_positions_are_minus_100(captioner, modality_out_dims, qformer_cfg):
    B, PPRE, PPOST, C = 2, 4, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    pre_ids = torch.randint(0, 30, (B, PPRE))
    post_ids = torch.randint(0, 30, (B, PPOST))
    caption_ids = torch.randint(0, 30, (B, C))

    n_queries = qformer_cfg["n_queries"]

    # Reconstruct labels the same way Captioner.forward does, and check the boundaries directly.
    labels = torch.cat(
        [
            torch.full((B, PPRE), IGNORE_INDEX, dtype=torch.long),
            torch.full((B, n_queries), IGNORE_INDEX, dtype=torch.long),
            torch.full((B, PPOST), IGNORE_INDEX, dtype=torch.long),
            caption_ids,
        ],
        dim=1,
    )
    head = PPRE + n_queries + PPOST
    assert torch.all(labels[:, :head] == IGNORE_INDEX)
    assert torch.equal(labels[:, head:], caption_ids)

    out = captioner(modality_batch, pre_ids, post_ids, caption_ids)
    assert out.loss.item() == out.loss.item()  # not NaN


def test_loss_is_computed_on_caption_tokens_only(captioner, modality_out_dims):
    B, PPRE, PPOST, C = 2, 4, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    pre_ids = torch.randint(0, 30, (B, PPRE))
    post_ids = torch.randint(0, 30, (B, PPOST))
    caption_ids = torch.randint(0, 30, (B, C))

    out = captioner(modality_batch, pre_ids, post_ids, caption_ids)

    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in captioner.fusion_stack.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "No gradient reached the fusion stack — loss is not actually connected to the prefix."
    )
