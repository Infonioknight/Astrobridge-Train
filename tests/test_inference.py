"""generate_caption's assembly logic (mask construction, prompt formatting, autocast reconciling
fp32 fusion stack vs bf16 LLM) — tested with fake encoders/tokenizer/LLM so it runs on CPU with
no real weights, network, or AION package. Mirrors test_generate_caption_dtype.py's fixtures.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from captioner.inference import generate_caption
from captioner.model.captioner import Captioner, FusionStack


class _FakeBF16LLM(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model).to(torch.bfloat16)
        self.proj = nn.Linear(d_model, vocab_size).to(torch.bfloat16)
        self.config = SimpleNamespace(hidden_size=d_model)

    def get_input_embeddings(self):
        return self.embed

    def generate(self, inputs_embeds, attention_mask, max_new_tokens, do_sample):
        self.proj(inputs_embeds)  # dtype-sensitive, exercises the same real crash if unreconciled
        return torch.zeros(inputs_embeds.shape[0], max_new_tokens, dtype=torch.long)


class _FakeTokenizer:
    def __init__(self):
        self.seen_prompts: list[str] = []

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        self.seen_prompts.append(text)
        return {"input_ids": torch.randint(0, 32, (1, 4))}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"n={ids.shape[0]}"] * ids.shape[0]


class _FakeEncoder:
    """Returns a fixed-size token grid regardless of input — enough to exercise the assembly
    path without needing the real AION package.
    """

    def __init__(self, n_tokens: int, out_dim: int):
        self.n_tokens = n_tokens
        self.out_dim = out_dim

    def encode(self, batch):
        return torch.ones(1, self.n_tokens, self.out_dim)


def _make_model_and_encoders():
    out_dims = {"image": 4, "spectra": 4}
    max_tokens = {"image": 3, "spectra": 3}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=8,
        d_llm=16,
        qformer_cfg=dict(n_queries=2, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0),
        projector_hidden_mult=2,
        projector_dropout=0.0,
    )
    model = Captioner(fusion_stack, _FakeBF16LLM(), n_queries=2)
    encoders = {"image": _FakeEncoder(3, 4), "spectra": _FakeEncoder(3, 4)}
    return model, encoders, out_dims, max_tokens


def test_single_modality_generates_a_caption():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    caption = generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        "Describe using only {modalities}.", "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )
    assert caption == "n=1"


def test_both_modalities_generates_a_caption():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    caption = generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        "Describe using only {modalities}.", "cpu",
        raw_inputs={
            "image": {"pixel_values": torch.zeros(1, 1, 4, 4)},
            "spectra": {"flux": torch.zeros(1, 6), "wavelength": torch.zeros(1, 6), "survey": ["desi"]},
        },
        max_new_tokens=3,
    )
    assert caption == "n=1"


def test_empty_raw_inputs_raises():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    try:
        generate_caption(
            model, _FakeTokenizer(), encoders, out_dims, max_tokens,
            "Describe using only {modalities}.", "cpu", raw_inputs={},
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "empty" in str(e)


def test_question_overrides_the_template_prompt_verbatim():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    tokenizer = _FakeTokenizer()

    generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens,
        "Describe using only {modalities}.", "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3, question="What kind of object is this?",
    )

    assert tokenizer.seen_prompts == ["What kind of object is this?"]


def test_no_question_falls_back_to_template():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    tokenizer = _FakeTokenizer()

    generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens,
        "Describe using only {modalities}.", "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    assert tokenizer.seen_prompts == ["Describe using only an image."]


def test_absent_modality_gets_true_mask_not_zero_content_only():
    """The absent modality's mask must be all-True (excluded), matching §6's rule that absence
    is real exclusion, never an unmasked zero-vector placeholder.
    """
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    # Spy on FusionStack.forward's input to check the mask without needing to inspect internals.
    captured = {}
    original_forward = model.fusion_stack.forward

    def _spy(modality_batch):
        captured["batch"] = modality_batch
        return original_forward(modality_batch)

    model.fusion_stack.forward = _spy

    generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        "Describe using only {modalities}.", "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    assert captured["batch"]["spectra"]["mask"].all()
    assert not captured["batch"]["image"]["mask"].any()
