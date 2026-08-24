"""§9 step 5 / §10 pitfall 3: an absent modality must contribute zero rows and never leak
through as an unmasked zero-vector placeholder. These three tests are the acceptance criteria
for the masking requirement in §6.
"""
from __future__ import annotations

import torch

from captioner.model.qformer import SharedQFormer


def _tiny_qformer():
    torch.manual_seed(0)
    qf = SharedQFormer(n_queries=4, d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0)
    qf.eval()
    return qf


def test_absent_modality_contributes_no_tokens():
    qf = _tiny_qformer()
    B, T_present, T_absent, D = 3, 5, 7, 16

    real_tokens = torch.randn(B, T_present, D)

    absent_tokens = torch.randn(B, T_absent, D)  # arbitrary content — must not matter
    tokens_with_absent = torch.cat([real_tokens, absent_tokens], dim=1)
    mask_with_absent = torch.cat(
        [torch.zeros(B, T_present, dtype=torch.bool), torch.ones(B, T_absent, dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        out_with_absent = qf(tokens_with_absent, key_padding_mask=mask_with_absent)
        out_present_only = qf(real_tokens, key_padding_mask=torch.zeros(B, T_present, dtype=torch.bool))

    assert torch.allclose(out_with_absent, out_present_only, atol=1e-5)


def test_garbage_in_absent_slot_is_bitwise_noop():
    qf = _tiny_qformer()
    B, T_present, T_absent, D = 3, 5, 7, 16

    real_tokens = torch.randn(B, T_present, D)
    mask = torch.cat(
        [torch.zeros(B, T_present, dtype=torch.bool), torch.ones(B, T_absent, dtype=torch.bool)], dim=1
    )

    zeros_absent = torch.zeros(B, T_absent, D)
    garbage_absent = torch.randn(B, T_absent, D) * 1e6  # deliberately extreme, not just nonzero

    tokens_zeros = torch.cat([real_tokens, zeros_absent], dim=1)
    tokens_garbage = torch.cat([real_tokens, garbage_absent], dim=1)

    with torch.no_grad():
        out_zeros = qf(tokens_zeros, key_padding_mask=mask)
        out_garbage = qf(tokens_garbage, key_padding_mask=mask)

    assert torch.equal(out_zeros, out_garbage), (
        "Garbage in a masked/absent slot changed the output — the mask is not truly excluding "
        "those positions from attention (see §6, §10 pitfall 3)."
    )


def test_output_invariant_to_padding_position():
    qf = _tiny_qformer()
    B, T_total, D = 2, 8, 16
    T_real = 5

    tokens = torch.zeros(B, T_total, D)
    tokens[:, :T_real] = torch.randn(B, T_real, D)
    mask = torch.zeros(B, T_total, dtype=torch.bool)
    mask[:, T_real:] = True

    perm = torch.randperm(T_total)
    tokens_perm = tokens[:, perm, :]
    mask_perm = mask[:, perm]

    with torch.no_grad():
        out_original = qf(tokens, key_padding_mask=mask)
        out_permuted = qf(tokens_perm, key_padding_mask=mask_perm)

    assert torch.allclose(out_original, out_permuted, atol=1e-5)
