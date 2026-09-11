"""generate_caption's assembly logic (mask construction, prompt formatting, autocast reconciling
fp32 fusion stack vs bf16 LLM) — tested with fake encoders/tokenizer/LLM so it runs on CPU with
no real weights, network, or AION package. Mirrors test_generate_caption_dtype.py's fixtures.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from captioner.inference import generate_caption, score_completions, score_completions_qwen_native
from captioner.model.captioner import IGNORE_INDEX, Captioner, FusionStack
from tests.conftest import make_prompt_cfg

_PROMPT_CFG = make_prompt_cfg(
    instruction_variants=["Describe this observation.", "Describe the object shown, using only {modalities}."],
)


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
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )
    assert caption == "n=1"


def test_both_modalities_generates_a_caption():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    caption = generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        _PROMPT_CFG, "cpu",
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
            _PROMPT_CFG, "cpu", raw_inputs={},
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "empty" in str(e)


def test_question_fills_the_instruction_slot_of_the_chat_template():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    tokenizer = _FakeTokenizer()

    generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens,
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3, question="What kind of object is this?",
    )

    # seen_prompts is [pre_text, post_text]; the question goes into post_text's {instruction}.
    joined = "".join(tokenizer.seen_prompts)
    assert "What kind of object is this?" in joined
    assert "<|im_start|>assistant" in joined  # the post half carries the assistant header


def test_no_question_uses_the_first_instruction_variant():
    model, encoders, out_dims, max_tokens = _make_model_and_encoders()
    tokenizer = _FakeTokenizer()

    generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens,
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    joined = "".join(tokenizer.seen_prompts)
    assert "Describe this observation." in joined  # instruction_variants[0]
    assert "using only an image" not in joined  # the modality-explicit variant was not chosen


def test_fewer_raw_tokens_than_max_tokens_are_padded_and_masked():
    """Confirmed real: AION's actual token count depends on input size (image pixel dims /
    spectrum length), not just num_encoder_tokens — a 384x384 image or ~7800-sample spectrum can
    both come back with fewer raw tokens than configured max_tokens. Must pad + mask, not raise.
    """
    model, _default_encoders, out_dims, max_tokens = _make_model_and_encoders()
    encoders = {"image": _FakeEncoder(2, 4), "spectra": _FakeEncoder(3, 4)}  # image: 2 < T_m=3

    captured = {}
    original_forward = model.fusion_stack.forward

    def _spy(modality_batch):
        captured["batch"] = modality_batch
        return original_forward(modality_batch)

    model.fusion_stack.forward = _spy

    generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    image_batch = captured["batch"]["image"]
    assert image_batch["tokens"].shape == (1, 3, 4)
    assert not image_batch["mask"][0, :2].any()  # first 2 (real) positions unmasked
    assert image_batch["mask"][0, 2]  # 3rd (padding) position masked


def test_more_raw_tokens_than_max_tokens_are_truncated():
    model, _default_encoders, out_dims, max_tokens = _make_model_and_encoders()
    encoders = {"image": _FakeEncoder(5, 4), "spectra": _FakeEncoder(3, 4)}  # image: 5 > T_m=3

    captured = {}
    original_forward = model.fusion_stack.forward

    def _spy(modality_batch):
        captured["batch"] = modality_batch
        return original_forward(modality_batch)

    model.fusion_stack.forward = _spy

    generate_caption(
        model, _FakeTokenizer(), encoders, out_dims, max_tokens,
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    image_batch = captured["batch"]["image"]
    assert image_batch["tokens"].shape == (1, 3, 4)
    assert not image_batch["mask"].any()  # fully real, none padded


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
        _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        max_new_tokens=3,
    )

    assert captured["batch"]["spectra"]["mask"].all()
    assert not captured["batch"]["image"]["mask"].any()


class _LossByLabelLengthLLM(nn.Module):
    """Returns a caller-controlled `.loss` keyed on the number of real (non-IGNORE_INDEX) label
    positions — lets a test dictate exactly what `Captioner.forward` reports per candidate,
    independent of any real learned distribution, so `score_completions`' own wiring (not HF's
    cross-entropy reduction) is what's under test.
    """

    def __init__(self, loss_by_n_real: dict[int, float], vocab_size: int = 32, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.config = SimpleNamespace(hidden_size=d_model)
        self.loss_by_n_real = loss_by_n_real

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, labels):
        n_real = int((labels != IGNORE_INDEX).sum().item())
        return SimpleNamespace(loss=torch.tensor(self.loss_by_n_real[n_real]))


class _LenKeyedTokenizer:
    """`__call__` returns exactly `len(text)` token ids — deterministic, content-independent
    length — so a test can control candidate token count via string length alone.
    """

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        n = max(1, len(text))
        return {"input_ids": torch.randint(0, 32, (1, n))}


def test_score_completions_does_not_penalise_a_longer_candidate_with_equal_per_token_loss():
    """The actual regression this design exists to prevent: `" SN Ibc"` tokenizes to more tokens
    than `" SN Ia"`, so a raw summed log-prob would structurally disadvantage it. Two candidates
    with equal per-token loss (mean cross-entropy, what `Captioner.forward` already returns) but
    different lengths must score equally.
    """
    out_dims = {"image": 4}
    max_tokens = {"image": 3}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims, d_shared=8, d_llm=16,
        qformer_cfg=dict(n_queries=2, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0),
        projector_hidden_mult=2, projector_dropout=0.0,
    )
    prompt_cfg = make_prompt_cfg(caption_suffix="")  # no suffix, so candidate length == string length
    short_candidate, long_candidate = " SN Ia", " SN Ibc"  # 6 vs 7 chars -> 6 vs 7 fake tokens
    model = Captioner(fusion_stack, _LossByLabelLengthLLM({6: 2.0, 7: 2.0}), n_queries=2)
    encoders = {"image": _FakeEncoder(3, 4)}

    scores = score_completions(
        model, _LenKeyedTokenizer(), encoders, out_dims, max_tokens, prompt_cfg, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        candidates=[short_candidate, long_candidate],
    )

    assert scores[short_candidate] == pytest.approx(scores[long_candidate])
    assert scores[short_candidate] == pytest.approx(-2.0)


def test_score_completions_argmax_can_favour_the_longer_candidate():
    """A longer candidate must be able to win outright when its per-token loss is genuinely
    lower — proving there's no structural length bias in either direction, not just that equal
    losses tie.
    """
    out_dims = {"image": 4}
    max_tokens = {"image": 3}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims, d_shared=8, d_llm=16,
        qformer_cfg=dict(n_queries=2, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0),
        projector_hidden_mult=2, projector_dropout=0.0,
    )
    prompt_cfg = make_prompt_cfg(caption_suffix="")  # no suffix, so candidate length == string length
    short_candidate, long_candidate = " SN Ia", " SN Ibc"  # 6 vs 7 chars -> 6 vs 7 fake tokens
    model = Captioner(fusion_stack, _LossByLabelLengthLLM({6: 3.0, 7: 1.0}), n_queries=2)
    encoders = {"image": _FakeEncoder(3, 4)}

    scores = score_completions(
        model, _LenKeyedTokenizer(), encoders, out_dims, max_tokens, prompt_cfg, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        candidates=[short_candidate, long_candidate],
    )

    assert max(scores, key=scores.get) == long_candidate


def test_score_completions_reasoning_is_appended_before_the_candidate_is_scored():
    out_dims = {"image": 4}
    max_tokens = {"image": 3}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims, d_shared=8, d_llm=16,
        qformer_cfg=dict(n_queries=2, d_model=8, n_layers=1, n_heads=2, ffn_mult=2, dropout=0.0),
        projector_hidden_mult=2, projector_dropout=0.0,
    )
    tokenizer = _FakeTokenizer()  # always returns 4 fake tokens per call, see its __call__ above
    model = Captioner(fusion_stack, _LossByLabelLengthLLM({4: 1.0}), n_queries=2)
    encoders = {"image": _FakeEncoder(3, 4)}

    score_completions(
        model, tokenizer, encoders, out_dims, max_tokens, _PROMPT_CFG, "cpu",
        raw_inputs={"image": {"pixel_values": torch.zeros(1, 1, 4, 4)}},
        candidates=["ABCDEF"], reasoning="some free-form reasoning text",
    )

    joined = "".join(tokenizer.seen_prompts)
    assert "some free-form reasoning text" in joined
    assert "<|im_start|>assistant" in joined


class _FakeBatchFeature(dict):
    def to(self, device):
        return self


class _FakeChatTemplateProcessor:
    """Fakes `apply_chat_template(..., continue_final_message=True)` — maps the assistant turn's
    exact text to a pre-registered token-id sequence, mirroring how a real chat template
    deterministically retokenizes the whole (context + reasoning [+ candidate]) sequence fresh
    each call. Records every text it was asked to tokenize, so a test can assert
    `score_completions_qwen_native` never hand-splices ids onto a previous call's result (the real
    bug this function's current implementation exists to avoid — see its docstring).
    """

    def __init__(self, ids_by_text: dict[str, list[int]]):
        self.ids_by_text = ids_by_text
        self.seen_texts: list[str] = []

    def apply_chat_template(self, messages, continue_final_message, tokenize, return_dict, return_tensors, **kwargs):
        text = messages[1]["content"][0]["text"]
        self.seen_texts.append(text)
        return _FakeBatchFeature({"input_ids": torch.tensor([self.ids_by_text[text]])})


class _FakeVisionModelWithLogits:
    """Returns near-certain logits at whichever (position, token_id) pairs a test registers for
    the exact `input_ids` sequence passed in — enough to make each candidate token's log-prob
    approximately 0 regardless of how many tokens the candidate is, so a test can verify
    length-invariant scoring on this code path too (mirroring `score_completions`'s equivalent
    test), without needing a real vocabulary/model.
    """

    def __init__(self, target_positions_by_ids: dict[tuple, list[tuple[int, int]]], vocab_size: int = 8):
        self.target_positions_by_ids = target_positions_by_ids
        self.vocab_size = vocab_size

    def __call__(self, input_ids, **kwargs):
        key = tuple(input_ids[0].tolist())
        logits = torch.full((1, input_ids.shape[1], self.vocab_size), -10.0)
        for pos, tok in self.target_positions_by_ids[key]:
            logits[0, pos, tok] = 10.0
        return SimpleNamespace(logits=logits)


def test_score_completions_qwen_native_never_hand_splices_ids_and_is_length_invariant():
    """Regression test for a real bug caught on a live A100 run: hand-concatenating candidate
    token ids onto an already-tokenized context crashes deep inside Qwen3.5's 3D RoPE position-id
    computation (`IndexError` from a mask/tensor shape mismatch), because that model recomputes
    position ids from the image-grid tensors fresh every forward call. The fix retokenizes the
    full (reasoning + candidate) text afresh per candidate instead.
    """
    reasoning = ""  # this fake's texts are keyed directly, content is irrelevant here
    one_token_candidate, two_token_candidate = "A", "BB"
    processor = _FakeChatTemplateProcessor({
        reasoning: [1, 2, 3],
        reasoning + one_token_candidate: [1, 2, 3, 4],
        reasoning + two_token_candidate: [1, 2, 3, 5, 6],
    })
    model = _FakeVisionModelWithLogits({
        (1, 2, 3, 4): [(2, 4)],
        (1, 2, 3, 5, 6): [(2, 5), (3, 6)],
    })

    scores = score_completions_qwen_native(
        model, processor, "cpu", "question text", "fake image",
        candidates=[one_token_candidate, two_token_candidate], reasoning=reasoning,
    )

    # Every apply_chat_template call passed a freshly built text, never a manually-spliced id
    # tensor reused across calls — the actual regression this test exists to catch.
    assert processor.seen_texts == [reasoning, reasoning + one_token_candidate, reasoning + two_token_candidate]
    assert scores[one_token_candidate] == pytest.approx(scores[two_token_candidate], abs=1e-3)
