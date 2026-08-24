"""AION spectrum wrapper (impl=aion_spectrum), against the real `aion` package API:

    spectrum = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)
    tokens = codec_manager.encode(spectrum)
    emb = model.encode(tokens, num_encoder_tokens=N)   # (B, N, 768) — token-level

Confirmed against a working linear-probe (`AionEncoder(SpectrumEncoder)`) that mean-pooled `emb`
over dim=1; this wrapper is that same call with pooling removed, since the Q-Former needs the
full token grid (§5). Where `ivar`/`mask` are missing from a batch, that probe defaulted to
`ivar=ones_like(flux)` / `mask=zeros_like(flux, dtype=bool)` (i.e. "fully trusted, fully
unmasked") — same default kept here. `wavelength` has no safe default: it's not derivable from
flux alone, so a batch missing it is a hard error rather than a silent guess.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from captioner.encoders.aion_common import load_aion
from captioner.encoders.base import EncoderLoadError

AION_OUT_DIM = 768  # AION-base's shared embedding width — same for every modality's tokens


class AionSpectrumEncoder:
    def __init__(
        self,
        name: str,
        hf_path: str,
        revision: str | None,
        kwargs: dict[str, Any],
        num_encoder_tokens: int,
    ) -> None:
        self.name = name
        self.hf_path = hf_path
        self.revision = revision
        self.num_encoder_tokens = num_encoder_tokens
        self.out_dim = AION_OUT_DIM
        self._model = None
        self._codec_manager = None
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        self._device = device
        self._model, self._codec_manager = load_aion(self.hf_path, self.revision, device)

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """batch['flux']: (B, L). Optional batch['ivar'], batch['mask']: (B, L).
        batch['wavelength']: (B, L) or (L,) — required, no safe default.
        """
        if self._model is None:
            raise EncoderLoadError("encode() called before load()")

        from aion.modalities import DESISpectrum

        flux = batch["flux"].to(self._device)
        if "wavelength" not in batch:
            raise ValueError(
                "batch['wavelength'] is required to encode a spectrum — AstroBridge-Data's raw "
                "`spectrum` dict field name for the wavelength grid must be confirmed and wired "
                "into scripts/02_cache_embeddings.py's spectra batch loader; there is no safe "
                "default to fall back to."
            )
        wavelength = batch["wavelength"].to(self._device)
        if wavelength.dim() == 1:
            wavelength = wavelength.unsqueeze(0).expand(flux.shape[0], -1)

        ivar = batch["ivar"].to(self._device) if "ivar" in batch else torch.ones_like(flux)
        mask = batch["mask"].to(self._device) if "mask" in batch else torch.zeros_like(flux, dtype=torch.bool)

        spectrum = DESISpectrum(flux=flux, ivar=ivar, mask=mask, wavelength=wavelength)
        tokens = self._codec_manager.encode(spectrum)
        emb = self._model.encode(tokens, num_encoder_tokens=self.num_encoder_tokens)

        if emb.dim() != 3:
            raise EncoderLoadError(
                f"Expected token-level (B, T, out_dim) from AION.encode, got shape "
                f"{tuple(emb.shape)}. Do not mean-pool in this wrapper — see §5/§10."
            )
        return emb
