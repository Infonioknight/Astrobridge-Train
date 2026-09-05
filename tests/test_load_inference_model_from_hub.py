"""load_inference_model_from_hub must load LoRA the *standard* PEFT way (PeftModel.from_pretrained
directly against the repo), unlike load_inference_model's local-checkpoint path, which has to
reconstruct a LoraConfig by hand because that format doesn't carry one. Mocks every real
network/GPU call so this runs on CPU with no HF Hub access.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from captioner.inference import load_inference_model_from_hub


def _cfg(modalities=("image",)):
    return OmegaConf.create(
        {
            "llm": {"name": "fake/llm"},
            "d_shared": 8,
            "qformer": {"n_queries": 2, "d_model": 8, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0},
            "projector": {"hidden_mult": 2, "dropout": 0.0},
            "modalities": {
                name: {"out_dim": 4, "max_tokens": 3, "encoder": {"impl": "fake", "hf_path": "x", "revision": None, "kwargs": {}}}
                for name in modalities
            },
        }
    )


class _FakeLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.config = type("C", (), {"hidden_size": 8})()

    def get_input_embeddings(self):
        return self.embed


def test_loads_lora_via_peft_standard_path_not_manual_reconstruction(tmp_path):
    fake_llm = _FakeLLM()
    middle_pt_path = tmp_path / "middle.pt"
    torch.save({}, middle_pt_path)  # placeholder; state dict content isn't checked by the mock load below

    with patch("captioner.inference.build_llm", return_value=(fake_llm, "fake_tokenizer")) as mock_build_llm, \
         patch("peft.PeftModel.from_pretrained", return_value=fake_llm) as mock_peft_from_pretrained, \
         patch("huggingface_hub.hf_hub_download", return_value=str(middle_pt_path)) as mock_hf_download, \
         patch("captioner.inference.build_encoder", return_value=MagicMock()), \
         patch.object(torch.nn.Module, "load_state_dict", return_value=None):
        model, tokenizer, encoders = load_inference_model_from_hub(_cfg(), "org/some-model", device="cpu")

    mock_build_llm.assert_called_once()
    mock_peft_from_pretrained.assert_called_once_with(fake_llm, "org/some-model")
    mock_hf_download.assert_called_once_with(repo_id="org/some-model", filename="middle.pt")
    assert tokenizer == "fake_tokenizer"
    assert "image" in encoders
    assert model.llm is fake_llm


def test_modality_names_none_builds_every_configured_encoder(tmp_path):
    fake_llm = _FakeLLM()
    middle_pt_path = tmp_path / "middle.pt"
    torch.save({}, middle_pt_path)

    with patch("captioner.inference.build_llm", return_value=(fake_llm, "tok")), \
         patch("peft.PeftModel.from_pretrained", return_value=fake_llm), \
         patch("huggingface_hub.hf_hub_download", return_value=str(middle_pt_path)), \
         patch("captioner.inference.build_encoder", return_value=MagicMock()) as mock_build_encoder, \
         patch.object(torch.nn.Module, "load_state_dict", return_value=None):
        _model, _tok, encoders = load_inference_model_from_hub(
            _cfg(("image", "spectra")), "org/some-model", device="cpu",
        )

    assert set(encoders.keys()) == {"image", "spectra"}
    assert mock_build_encoder.call_count == 2


def test_modality_names_subset_builds_only_the_requested_encoders():
    """Confirmed real cost, not just tidiness: building an unused encoder means downloading and
    loading its own model weights for nothing — e.g. inference/compare.py only ever shows an
    image, so it must never trigger AION's spectrum encoder or ATCAT's light-curve one.
    """
    fake_llm = _FakeLLM()

    with patch("captioner.inference.build_llm", return_value=(fake_llm, "tok")), \
         patch("peft.PeftModel.from_pretrained", return_value=fake_llm), \
         patch("huggingface_hub.hf_hub_download", return_value="/dev/null"), \
         patch("captioner.inference.build_encoder", return_value=MagicMock()) as mock_build_encoder, \
         patch("torch.load", return_value={}), \
         patch.object(torch.nn.Module, "load_state_dict", return_value=None):
        _model, _tok, encoders = load_inference_model_from_hub(
            _cfg(("image", "spectra")), "org/some-model", device="cpu", modality_names=["image"],
        )

    assert set(encoders.keys()) == {"image"}
    mock_build_encoder.assert_called_once()
    assert mock_build_encoder.call_args.args[0] == "image"
