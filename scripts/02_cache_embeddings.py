#!/usr/bin/env python
"""Loads encoders exactly once, encodes every object that has that modality, and writes
float16 shards + an index parquet (§5). Training never imports this module's encoders.

Both field schemas are confirmed against real data: spectra via AstroBridge-Data's `spectrum`
struct (`flux`/`ivar`/`lambda`/`mask`), image via `legacy_south_all_images.parquet`'s
`image_legacy` (list of per-band `{band, flux, mask, ivar, psf_fwhm, scale}` structs) — see
data/image_dataset.py and the two batch loaders below.
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


def _image_batch_loader(pixels_by_id: dict, bands: list[str]):
    """`pixels_by_id`: {object_id: list of {band, flux, mask, ivar, psf_fwhm, scale} dicts} from
    data/image_dataset.py:load_image_flux_pixels — real calibrated per-band flux from
    legacy_south_all_images.parquet's `image_legacy` field. Bands are matched by name (config's
    "DES-G" -> "g"), not by list position — position isn't guaranteed order in the source data.
    Only `flux` is used; `mask`/`ivar`/`psf_fwhm`/`scale` aren't part of AION's LegacySurveyImage
    constructor (confirmed against a working probe script — see aion_image.py).
    """

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        per_object = []
        for oid in object_ids:
            band_entries = {e["band"].lower(): e for e in pixels_by_id[oid]}
            per_band = []
            for b in bands:
                short = b.split("-")[-1].lower()  # "DES-G" -> "g"
                if short not in band_entries:
                    raise KeyError(
                        f"Band {b!r} (looked up as {short!r}) not available for object {oid!r}; "
                        f"bands present: {sorted(band_entries.keys())}."
                    )
                per_band.append(np.asarray(band_entries[short]["flux"], dtype=np.float32))
            per_object.append(np.stack(per_band, axis=0))  # (n_bands, H, W)

        shapes = {a.shape for a in per_object}
        if len(shapes) > 1:
            raise ValueError(
                f"Inconsistent per-object image shapes in this batch: {shapes}. Expected every "
                "cutout to share the same (n_bands, H, W)."
            )
        pixel_tensor = torch.from_numpy(np.stack(per_object, axis=0))  # (B, n_bands, H, W)
        return {"pixel_values": pixel_tensor}

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

    spectra_by_id = None
    if "spectra" in modality_names:
        from captioner.data.spectra_dataset import load_spectra_table

        spectra_df = load_spectra_table(cfg.sources.spectra.hf_path, revision=cfg.sources.spectra.get("revision"))
        spectra_by_id = spectra_df.set_index("object_id").to_dict(orient="index")

    image_pixels_by_id = None
    if "image" in modality_names:
        from captioner.data.image_dataset import load_image_flux_pixels

        image_pixels_by_id = load_image_flux_pixels(
            cfg.sources.image.hf_path, revision=cfg.sources.image.get("revision")
        )

    out_dir = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in modality_names:
        modality_cfg = cfg.modalities[name]
        encoder = build_encoder(name, modality_cfg, device=args.device)

        if name == "spectra":
            loader = _spectra_batch_loader(spectra_by_id)
        elif name == "image":
            loader = _image_batch_loader(image_pixels_by_id, list(modality_cfg.encoder.kwargs.get("bands", [])))
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
