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


def _flux_to_array(raw_flux, object_id: str, band: str) -> np.ndarray:
    """Converts one band's `flux` field to a clean (H, W) float32 array. Confirmed against a
    real run: `np.asarray(raw_flux, dtype=np.float32)` can fail with
    "setting an array element with a sequence" — meaning `raw_flux` (a list of rows) isn't a
    uniform rectangle. The most likely, well-understood cause for a real image cutout is a row
    near a survey/mosaic edge coming through as `None` (missing) rather than a properly-shaped
    row of NaNs — exactly what `mask`/`ivar` exist elsewhere in this same data to flag. Recovered
    by filling `None` rows with NaN at the modal row width. Anything else (rows that exist but
    genuinely disagree on width) is NOT auto-fixed — that could silently corrupt real flux
    values — it raises with the actual row-length breakdown instead of numpy's opaque error.
    """
    # Validate row-by-row explicitly, rather than trying np.asarray(...) first and reacting to
    # failure — numpy silently converts a top-level `None` to NaN for a float dtype even when
    # the overall structure isn't the 2D grid we need (e.g. `np.asarray([None, None],
    # dtype=float32)` "succeeds" as a 1D array), which would let an all-missing row slip through
    # undetected instead of being caught below.
    rows = list(raw_flux)
    lengths = []
    for r in rows:
        try:
            lengths.append(len(r))
        except TypeError:
            lengths.append(None)  # a None/scalar row — treated as "missing", not corrupt data

    real_lengths = sorted({length for length in lengths if length is not None})
    if not real_lengths:
        raise ValueError(
            f"flux for object={object_id!r} band={band!r} has no usable rows at all — every row "
            f"is None/scalar (row types: {[type(r).__name__ for r in rows]})."
        )
    if len(real_lengths) > 1:
        raise ValueError(
            f"flux for object={object_id!r} band={band!r} has rows of genuinely different "
            f"widths {real_lengths} — not just missing rows. Row-by-row lengths: {lengths}. "
            "This needs a real look before auto-fixing (padding/cropping could silently distort "
            "the image), so it's not attempted automatically."
        )

    width = real_lengths[0]
    n_missing = sum(1 for length in lengths if length is None)
    if n_missing == 0:
        return np.asarray(rows, dtype=np.float32)

    logger.warning(
        f"object={object_id!r} band={band!r}: {n_missing}/{len(rows)} rows were None/missing "
        f"— filled with NaN at width {width}. If this is common, it's worth checking whether "
        "AION's LegacySurveyImage tolerates NaN input or needs these masked differently."
    )
    fixed_rows = [
        np.asarray(r, dtype=np.float32) if length is not None else np.full(width, np.nan, dtype=np.float32)
        for r, length in zip(rows, lengths)
    ]
    return np.stack(fixed_rows, axis=0)


def _canonical_band(label: str) -> str:
    """Normalizes a band label to its bare letter for matching — "DES-G", "des-g", "G", and "g"
    must all resolve to the same key. Confirmed against a real run that the source data's own
    `band` field is a full string like "des-g", not a bare letter — normalizing only the
    configured label ("DES-G" -> "g") and comparing it against the data's *unnormalized* string
    ("des-g") is exactly what broke: "g" != "des-g". Both sides must go through this function.
    """
    return label.strip().lower().split("-")[-1].split("_")[-1]


def _image_batch_loader(pixels_by_id: dict, bands: list[str]):
    """`pixels_by_id`: {object_id: list of {band, flux, mask, ivar, psf_fwhm, scale} dicts} from
    data/image_dataset.py:load_image_flux_pixels — real calibrated per-band flux from
    legacy_south_all_images.parquet's `image_legacy` field. Bands are matched by canonical name
    (see _canonical_band), not by list position — position isn't guaranteed order in the source
    data. Only `flux` is used; `mask`/`ivar`/`psf_fwhm`/`scale` aren't part of AION's
    LegacySurveyImage constructor (confirmed against a working probe script — see aion_image.py).
    """

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        per_object = []
        for oid in object_ids:
            band_entries = {_canonical_band(e["band"]): e for e in pixels_by_id[oid]}
            per_band = []
            for b in bands:
                key = _canonical_band(b)
                if key not in band_entries:
                    raise KeyError(
                        f"Band {b!r} (canonicalized to {key!r}) not available for object {oid!r}; "
                        f"bands present: {sorted(band_entries.keys())}."
                    )
                per_band.append(_flux_to_array(band_entries[key]["flux"], oid, key))
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
