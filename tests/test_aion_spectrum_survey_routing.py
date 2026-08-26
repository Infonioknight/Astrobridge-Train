"""AionSpectrumEncoder must route each object to AION's matching DESISpectrum/SDSSSpectrum
modality class based on its survey origin — confirmed real and required (not hypothetical):
AION's own source (github.com/PolymathicAI/AION, aion/modalities.py) defines these as separate
classes with distinct internal domains (tok_spectrum_desi vs tok_spectrum_sdss, confirmed via
aion-base's config.json `domains_in`/`domains_out`). AstroBridge-Data's spectra come from four
crossmatch files spanning both surveys, so a single batch can (and will) mix survey origins.
Routing everything through DESISpectrum regardless of origin — what this wrapper originally did
before this test was added — wouldn't crash; it would silently mislabel real SDSS spectra as
DESI to AION, a wrong-but-not-erroring failure mode.

Uses a fake `aion.modalities` module matching the confirmed real field signatures
(flux/ivar/mask/wavelength), since the real `polymathic-aion` package isn't installed in this
test environment.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch


@pytest.fixture
def fake_aion_modalities(monkeypatch):
    class FakeModality:
        def __init__(self, flux, ivar, mask, wavelength):
            self.flux, self.ivar, self.mask, self.wavelength = flux, ivar, mask, wavelength
            self.kind = self.__class__.__name__

    class DESISpectrum(FakeModality):
        pass

    class SDSSSpectrum(FakeModality):
        pass

    fake_modalities = types.ModuleType("aion.modalities")
    fake_modalities.DESISpectrum = DESISpectrum
    fake_modalities.SDSSSpectrum = SDSSSpectrum
    fake_aion = types.ModuleType("aion")
    fake_aion.modalities = fake_modalities
    monkeypatch.setitem(sys.modules, "aion", fake_aion)
    monkeypatch.setitem(sys.modules, "aion.modalities", fake_modalities)
    return fake_modalities


def _make_encoder(fake_modalities):
    from captioner.encoders.aion_spectrum import AionSpectrumEncoder

    enc = AionSpectrumEncoder("spectra", "fake/fake", None, {}, num_encoder_tokens=4)
    enc._model = types.SimpleNamespace()
    enc._codec_manager = types.SimpleNamespace()

    seen = []

    def fake_codec_encode(modality):
        seen.append((modality.kind, modality.flux.shape[0]))
        return {"tok": modality.flux}

    def fake_model_encode(tokens, num_encoder_tokens):
        B = tokens["tok"].shape[0]
        return torch.zeros(B, 4, 768)

    enc._codec_manager.encode = fake_codec_encode
    enc._model.encode = fake_model_encode
    return enc, seen


def _base_batch(B: int, L: int = 6) -> dict:
    return {
        "flux": torch.arange(B * L, dtype=torch.float32).reshape(B, L),
        "ivar": torch.ones(B, L),
        "mask": torch.zeros(B, L, dtype=torch.bool),
        "wavelength": torch.ones(B, L),
    }


def test_mixed_batch_splits_by_survey_and_reassembles_in_order(fake_aion_modalities):
    enc, seen = _make_encoder(fake_aion_modalities)
    batch = _base_batch(5)
    batch["survey"] = ["desi", "sdss", "desi", "sdss", "sdss"]

    out = enc.encode(batch)

    assert out.shape == (5, 4, 768)
    assert ("DESISpectrum", 2) in seen
    assert ("SDSSSpectrum", 3) in seen


def test_all_desi_batch_never_calls_sdss_path(fake_aion_modalities):
    enc, seen = _make_encoder(fake_aion_modalities)
    batch = _base_batch(3)
    batch["survey"] = ["desi", "desi", "desi"]

    enc.encode(batch)

    assert seen == [("DESISpectrum", 3)]


def test_missing_survey_raises_clearly(fake_aion_modalities):
    enc, _ = _make_encoder(fake_aion_modalities)
    batch = _base_batch(2)

    with pytest.raises(ValueError, match="survey"):
        enc.encode(batch)


def test_unknown_survey_value_raises_clearly(fake_aion_modalities):
    enc, _ = _make_encoder(fake_aion_modalities)
    batch = _base_batch(2)
    batch["survey"] = ["desi", "boss"]

    with pytest.raises(ValueError, match="boss"):
        enc.encode(batch)
