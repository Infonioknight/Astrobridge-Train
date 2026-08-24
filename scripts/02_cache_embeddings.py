#!/usr/bin/env python
"""Loads encoders exactly once, encodes every object that has that modality, and writes
float16 shards + an index parquet (§5). Training never imports this module's encoders.

Spectra field names (`flux`/`ivar`/`lambda`/`mask` in AstroBridge-Data's `spectrum` struct) are
confirmed — see _spectra_batch_loader below. Image caching is currently BLOCKED: gapatron/
legacy_survey_south_images_captions only ships rendered RGB PNGs, not the raw per-band calibrated
flux AION's LegacySurveyImage expects — see README.md's "Known blocker" section. `--modality
spectra` works today; `--modality image` (or no --modality flag) will raise a clear error rather
than attempt something that would silently produce wrong embeddings.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from captioner.data.cache import cache_modality, verify_fp16_roundtrip
from captioner.encoders.registry import build_encoder
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def _spectra_batch_loader(raw_by_id: dict):
    """AstroBridge-Data's `spectrum` field (confirmed via the HF datasets-server schema for
    UniverseTBD/AstroBridge-Data) is a nested struct of five float32/bool lists:
    `flux`, `ivar`, `lsf_sigma`, `lambda` (wavelength, in angstroms), `mask`. AION's DESISpectrum
    codec wants `flux`/`ivar`/`mask`/`wavelength`; `lambda` is the wavelength grid — `lsf_sigma`
    is unused here (not part of DESISpectrum's constructor).
    """

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        spectra = [raw_by_id[oid]["spectrum"] for oid in object_ids]
        fluxes = [np.asarray(s["flux"], dtype=np.float32) for s in spectra]
        max_len = max(len(f) for f in fluxes)

        flux_tensor = torch.zeros((len(fluxes), max_len), dtype=torch.float32)
        wave_tensor = torch.zeros((len(fluxes), max_len), dtype=torch.float32)
        ivar_tensor = torch.ones((len(fluxes), max_len), dtype=torch.float32)
        mask_tensor = torch.zeros((len(fluxes), max_len), dtype=torch.bool)

        for i, s in enumerate(spectra):
            n = len(fluxes[i])
            flux_tensor[i, :n] = torch.from_numpy(fluxes[i])
            wave_tensor[i, :n] = torch.from_numpy(np.asarray(s["lambda"], dtype=np.float32))
            ivar_tensor[i, :n] = torch.from_numpy(np.asarray(s["ivar"], dtype=np.float32))
            mask_tensor[i, :n] = torch.from_numpy(np.asarray(s["mask"], dtype=bool))

        return {"flux": flux_tensor, "wavelength": wave_tensor, "ivar": ivar_tensor, "mask": mask_tensor}

    return _load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default=None, help="cache only this modality (default: all)")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities")
    manifest = pd.read_parquet(cfg.manifest.parquet)

    modality_names = [args.modality] if args.modality else list(cfg.modalities.keys())
    if "image" in modality_names:
        # See README.md's "Known blocker" section. Once real flux cutouts exist, add an
        # `_image_batch_loader` returning {"pixel_values": (B, n_bands, H, W) float32} here,
        # matching _spectra_batch_loader's shape below.
        raise NotImplementedError(
            "Image caching is blocked: gapatron/legacy_survey_south_images_captions only "
            "provides rendered RGB PNGs, not the raw per-band calibrated flux AION's "
            "LegacySurveyImage expects. Run `--modality spectra` on its own until real grz/griz "
            "flux cutouts are available."
        )

    from datasets import load_dataset

    spectra_ds = load_dataset(cfg.sources.spectra.hf_path, split=cfg.sources.spectra.split)
    spectra_by_id = {row["object_id"]: row for row in spectra_ds}

    out_dir = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in modality_names:
        modality_cfg = cfg.modalities[name]
        encoder = build_encoder(name, modality_cfg, device=args.device)

        if name == "spectra":
            loader = _spectra_batch_loader(spectra_by_id)
        else:
            raise NotImplementedError(
                f"No batch loader wired for modality {name!r} yet — add one here when the "
                f"modality is added to configs/modalities.yaml (impl={modality_cfg.encoder.impl})."
            )

        sample_ids = manifest.loc[manifest[f"has_{name}"], "object_id"].head(min(50, len(manifest))).tolist()
        if sample_ids:
            sample_batch = loader(sample_ids)
            max_err = verify_fp16_roundtrip(encoder, sample_batch, n=len(sample_ids))
            logger.info(f"[{name}] float16 round-trip max abs error over {len(sample_ids)} objects: {max_err:.6f}")

        cache_modality(name, encoder, modality_cfg, manifest, loader, out_dir, shard=args.shard_size)


if __name__ == "__main__":
    main()
