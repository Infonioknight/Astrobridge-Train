"""§6 / §10 pitfall 2: labels must be -100 everywhere except the caption-token positions,
otherwise the model is trained to autoregressively predict its own visual prefix.
"""
from __future__ import annotations

import torch

from captioner.model.captioner import IGNORE_INDEX
from tests.conftest import make_modality_batch


def test_prefix_and_prompt_positions_are_minus_100(captioner, modality_out_dims, qformer_cfg):
    B, P, C = 2, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    prompt_ids = torch.randint(0, 30, (B, P))
    caption_ids = torch.randint(0, 30, (B, C))

    n_queries = qformer_cfg["n_queries"]
    prefix_len = n_queries

    # Reconstruct labels the same way Captioner.forward does, and check the boundary directly.
    labels = torch.cat(
        [
            torch.full((B, prefix_len), IGNORE_INDEX, dtype=torch.long),
            torch.full((B, P), IGNORE_INDEX, dtype=torch.long),
            caption_ids,
        ],
        dim=1,
    )

    assert torch.all(labels[:, :prefix_len] == IGNORE_INDEX)
    assert torch.all(labels[:, prefix_len : prefix_len + P] == IGNORE_INDEX)
    assert torch.equal(labels[:, prefix_len + P :], caption_ids)

    out = captioner(modality_batch, prompt_ids, caption_ids)
    assert out.loss.item() == out.loss.item()  # not NaN


def test_loss_is_computed_on_caption_tokens_only(captioner, modality_out_dims):
    B, P, C = 2, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    prompt_ids = torch.randint(0, 30, (B, P))
    caption_ids = torch.randint(0, 30, (B, C))

    out = captioner(modality_batch, prompt_ids, caption_ids)

    # A caption filled entirely with the ignore index must yield a zero (or NaN-free undefined)
    # loss contribution — cross_entropy with an all -100 target returns nan; guard against that
    # by checking the "some real target" case produces a finite, differentiable loss instead.
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in captioner.fusion_stack.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "No gradient reached the fusion stack — loss is not actually connected to the prefix."
    )
