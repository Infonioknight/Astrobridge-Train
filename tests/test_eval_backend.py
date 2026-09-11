"""eval/backend.py's local-backend construction/dispatch — mocked model loading (same pattern
tests/test_load_inference_model_from_hub.py and tests/test_qwen_native_vision.py already use),
no real weights/GPU/network.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from eval.backend import free_local_backend, get_backend


def _cfg():
    return OmegaConf.create({
        "modalities": {"image": {"out_dim": 4, "max_tokens": 3}},
        "prompt": {
            "wrapper_pre": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<observation>",
            "wrapper_post": "</observation>\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
            "caption_suffix": "<|im_end|>",
            "system_variants": ["You are an astronomy assistant."],
            "instruction_variants": ["Describe using only {modalities}."],
        },
    })


def test_equipped_backend_dispatches_to_load_inference_model_from_hub_and_generate_caption():
    fake_model, fake_tokenizer, fake_encoders = object(), object(), {}
    with (
        patch("captioner.inference.load_inference_model_from_hub", return_value=(fake_model, fake_tokenizer, fake_encoders)) as mock_load,
        patch("captioner.inference.generate_caption", return_value="a caption") as mock_generate,
    ):
        backend = get_backend("local", side="equipped", cfg=_cfg(), repo_id="org/repo", device="cpu", modality_names=["image"])
        out = backend.generate({"image": {"pixel_values": "sentinel"}}, "describe it", max_new_tokens=10)

    assert out == "a caption"
    mock_load.assert_called_once_with(_cfg_reload(), "org/repo", device="cpu", modality_names=["image"])
    assert mock_generate.call_args.kwargs["question"] == "describe it"
    assert mock_generate.call_args.kwargs["max_new_tokens"] == 10


def test_equipped_backend_without_repo_id_raises():
    with pytest.raises(ValueError, match="repo_id is required"):
        get_backend("local", side="equipped", cfg=_cfg(), device="cpu")


def test_base_backend_dispatches_to_load_qwen_native_vision_model_and_generate():
    fake_model, fake_processor = object(), object()
    with (
        patch("captioner.inference.load_qwen_native_vision_model", return_value=(fake_model, fake_processor)),
        patch("captioner.inference.generate_qwen_native_vision_answer", return_value="plain answer") as mock_generate,
    ):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
        out = backend.generate({"image": "sentinel_image"}, "what is this?", max_new_tokens=5)

    assert out == "plain answer"
    assert backend.side == "base"
    args, kwargs = mock_generate.call_args
    assert args[0] is fake_model
    assert args[1] is fake_processor


def test_base_backend_rejects_non_image_raw_inputs():
    with patch("captioner.inference.load_qwen_native_vision_model", return_value=(object(), object())):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
        with pytest.raises(KeyError, match="only ever consumes"):
            backend.generate({"lightcurve": {}}, "classify this", max_new_tokens=5)


def test_get_backend_rejects_unknown_kind():
    with pytest.raises(ValueError, match="not recognised"):
        get_backend("carrier-pigeon", side="equipped", cfg=_cfg())


def test_free_local_backend_drops_the_generate_closure():
    with patch("captioner.inference.load_qwen_native_vision_model", return_value=(object(), object())):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
    free_local_backend(backend)
    with pytest.raises(AttributeError):
        backend.generate({"image": "x"}, "q", 5)


def test_free_local_backend_drops_the_classify_closure_too():
    with patch("captioner.inference.load_qwen_native_vision_model", return_value=(object(), object())):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
    free_local_backend(backend)
    with pytest.raises(NotImplementedError):
        backend.classify({"image": "x"}, "reason about it", [" A", " B"])


def test_equipped_backend_classify_dispatches_to_generate_caption_then_score_completions():
    fake_model, fake_tokenizer, fake_encoders = object(), object(), {}
    with (
        patch("captioner.inference.load_inference_model_from_hub", return_value=(fake_model, fake_tokenizer, fake_encoders)),
        patch("captioner.inference.generate_caption", return_value="some reasoning") as mock_generate,
        patch("captioner.inference.score_completions", return_value={" A": -0.1, " B": -0.9}) as mock_score,
    ):
        backend = get_backend("local", side="equipped", cfg=_cfg(), repo_id="org/repo", device="cpu", modality_names=["image"])
        out = backend.classify({"image": {"pixel_values": "sentinel"}}, "reason about it", [" A", " B"], 40)

    assert out == {"reasoning": "some reasoning", "logprobs": {" A": -0.1, " B": -0.9}}
    assert mock_generate.call_args.kwargs["question"] == "reason about it"
    assert mock_generate.call_args.kwargs["max_new_tokens"] == 40
    assert mock_score.call_args.kwargs["question"] == "reason about it"
    assert mock_score.call_args.kwargs["reasoning"] == "some reasoning"
    assert mock_score.call_args[0][8] == [" A", " B"]  # candidates, positional


def test_base_backend_classify_dispatches_to_generate_then_score_completions_qwen_native():
    with (
        patch("captioner.inference.load_qwen_native_vision_model", return_value=(object(), object())),
        patch("captioner.inference.generate_qwen_native_vision_answer", return_value="some reasoning") as mock_generate,
        patch("captioner.inference.score_completions_qwen_native", return_value={" A": -0.2, " B": -0.3}) as mock_score,
    ):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
        out = backend.classify({"image": "sentinel_image"}, "reason about it", [" A", " B"], 40)

    assert out == {"reasoning": "some reasoning", "logprobs": {" A": -0.2, " B": -0.3}}
    assert mock_generate.call_args.kwargs["max_new_tokens"] == 40
    assert mock_generate.call_args.kwargs["enable_thinking"] is False
    assert mock_score.call_args.kwargs["reasoning"] == "some reasoning"


def test_base_backend_classify_rejects_non_image_raw_inputs():
    with patch("captioner.inference.load_qwen_native_vision_model", return_value=(object(), object())):
        backend = get_backend("local", side="base", cfg=_cfg(), device="cpu")
        with pytest.raises(KeyError, match="only ever consumes"):
            backend.classify({"lightcurve": {}}, "reason about it", [" A", " B"])


def _cfg_reload():
    # OmegaConf DictConfig doesn't compare equal by identity to a re-created equivalent config in
    # all mock-assertion contexts; rebuild the same structure for the equality check in
    # assert_called_once_with above.
    return _cfg()
