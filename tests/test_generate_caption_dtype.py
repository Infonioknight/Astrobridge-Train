"""_generate_caption must not crash on the real fp32-fusion-stack vs bf16-LLM dtype mismatch.
Confirmed against production: "RuntimeError: expected mat1 and mat2 to have the same dtype, but
got: float != c10::BFloat16" inside Qwen3.5-9B's linear_attn layer during `make eval`. Root
cause: FusionStack is a plain nn.Module, fp32 by construction; the real LLM is loaded in bf16
(build_llm). During training, accelerator.prepare() wraps every forward in an autocast(bf16)
context (configs/accelerate_ddp.yaml: mixed_precision: bf16), silently reconciling the mismatch
at every layer boundary. 04_eval.py never goes through an Accelerator, so without reproducing
that same autocast context, the mismatch surfaces directly. See groundedness.py's
_generate_caption docstring for the full reasoning.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from captioner.eval.groundedness import _generate_caption
from captioner.model.captioner import Captioner, FusionStack


class _FakeBF16LLM(nn.Module):
    """Mimics the real LLM being persistently stored in bf16 — not just "under autocast"."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(torch.bfloat16)
        self.proj = nn.Linear(d_model, vocab_size).to(torch.bfloat16)
        self.config = SimpleNamespace(hidden_size=d_model)

    def get_input_embeddings(self):
        return self.embed

    def generate(self, inputs_embeds, attention_mask, max_new_tokens, do_sample):
        # This is exactly the operation that raised the real error: a bf16-weight Linear layer
        # fed an fp32 `inputs_embeds` — without the caller reconciling dtypes first (autocast),
        # this raises RuntimeError: "mat1 and mat2 must have the same dtype".
        self.proj(inputs_embeds)
        B = inputs_embeds.shape[0]
        return torch.zeros(B, max_new_tokens, dtype=torch.long)


class _FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        return ["a caption"] * ids.shape[0]


def _make_model():
    fusion_stack = FusionStack(
        modality_out_dims={"image": 4},
        d_shared=8,
        d_llm=16,
        qformer_cfg=dict(n_queries=2, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0),
        projector_hidden_mult=2,
        projector_dropout=0.0,
    )  # fp32 by default — deliberately NOT cast to bf16, matching real 04_eval.py behavior
    return Captioner(fusion_stack, _FakeBF16LLM(), n_queries=2)


def _make_batch():
    return {
        "modality_batch": {
            "image": {"tokens": torch.randn(1, 3, 4), "mask": torch.zeros(1, 3, dtype=torch.bool)},
        },
        "prompt_ids": torch.randint(0, 32, (1, 5)),
    }


def test_generate_caption_survives_fp32_fusion_vs_bf16_llm_mismatch():
    model = _make_model()
    out = _generate_caption(model, _FakeTokenizer(), _make_batch(), device="cpu", max_new_tokens=3)
    assert out == ["a caption"]


def test_reproduces_the_real_crash_without_autocast():
    """Confirms the test fixture actually exercises the real failure mode — i.e. this isn't a
    vacuous test that would pass regardless of whether the autocast fix is present.
    """
    model = _make_model()
    fusion_stack = model.fusion_stack
    batch = _make_batch()
    prefix = fusion_stack(batch["modality_batch"])
    prompt_embeds = model.llm.get_input_embeddings()(batch["prompt_ids"])
    inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)

    import pytest

    with pytest.raises(RuntimeError, match="dtype"):
        model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long),
            max_new_tokens=3,
            do_sample=False,
        )
