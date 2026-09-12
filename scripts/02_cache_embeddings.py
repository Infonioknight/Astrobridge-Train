#!/usr/bin/env python
"""Loads encoders exactly once, encodes every object that has that modality, and writes
float16 shards + an index parquet (§5). Training never imports this module's encoders.

Both field schemas are confirmed against real data: spectra via AstroBridge-Data's `spectrum`
struct (`flux`/`ivar`/`lambda`/`mask`), image via gapatron/astrobridge-image-captions' flat
per-band `flux_{g,r,i,z}` columns (reassembled into `{band, flux, mask, ivar, psf_fwhm, scale}`
dicts by data/image_dataset.py:load_image_flux_pixels), and light
curves via BuildNg/astrobridge-transients-dataset's `atcat_*` arrays — see data/image_dataset.py,
data/transients_dataset.py and the three batch loaders below.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from captioner.data.cache import cache_modality, verify_fp16_roundtrip
from captioner.data.transients_dataset import prepare_lightcurve_arrays
from captioner.encoders.registry import build_encoder
from captioner.utils.config import load_config, remaining_argv
from captioner.data.spectra_dataset import spectrum_group_key, trimmed_spectrum_arrays
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def _spectra_batch_loader(raw_by_id: dict):
    """AstroBridge-Data's `spectrum` field (confirmed via the HF datasets-server schema for
    UniverseTBD/AstroBridge-Data) is a nested struct of five float32/bool lists:
    `flux`, `ivar`, `lsf_sigma`, `lambda` (wavelength, in angstroms), `mask`. AION's
    DESISpectrum/SDSSSpectrum codecs want `flux`/`ivar`/`mask`/`wavelength`; `lambda` is the
    wavelength grid — `lsf_sigma` is unused (not part of either modality class's constructor).
    `survey` (attached by data/spectra_dataset.py's `_attach_survey_column`) routes each object
    to the matching modality class in aion_spectrum.py — see that file's docstring for why the
    distinction between DESI-origin and SDSS-origin spectra is a real, non-optional requirement.
    """

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        rows = [raw_by_id[oid] for oid in object_ids]
        trimmed = [trimmed_spectrum_arrays(r["spectrum"]) for r in rows]
        max_len = max(len(t[0]) for t in trimmed)

        flux_tensor = torch.zeros((len(trimmed), max_len), dtype=torch.float32)
        wave_tensor = torch.zeros((len(trimmed), max_len), dtype=torch.float32)
        # Padding must read as "no measurement here": ivar=0 (no weight) and mask=True (excluded
        # from AION's normalization). The previous defaults were the exact opposite — ivar=1 and
        # mask=False told AION the padding was fully-trusted real signal.
        ivar_tensor = torch.zeros((len(trimmed), max_len), dtype=torch.float32)
        mask_tensor = torch.ones((len(trimmed), max_len), dtype=torch.bool)

        for i, (flux, wavelength, ivar, mask) in enumerate(trimmed):
            n = len(flux)
            flux_tensor[i, :n] = torch.from_numpy(flux)
            wave_tensor[i, :n] = torch.from_numpy(wavelength)
            ivar_tensor[i, :n] = torch.from_numpy(ivar)
            mask_tensor[i, :n] = torch.from_numpy(mask)
            if n < max_len:
                # Continue the grid upward rather than leaving zeros: `searchsorted` needs the
                # whole row sorted, and a zero-filled tail after ascending values is exactly the
                # non-monotonicity that broke this pipeline in the first place. 0.8 A is AION's
                # own latent-grid resolution (LatentSpectralGrid(resolution=0.8)).
                pad = np.arange(1, max_len - n + 1, dtype=np.float32) * 0.8
                wave_tensor[i, n:] = torch.from_numpy(wavelength[-1] + pad)

        survey = [r["survey"] for r in rows]

        return {
            "flux": flux_tensor,
            "wavelength": wave_tensor,
            "ivar": ivar_tensor,
            "mask": mask_tensor,
            "survey": survey,
        }

    return _load


def _lightcurve_batch_loader(raw_by_id: dict, modality_cfg):
    """Builds ATCAT's five fixed-length inputs from the transients table.

    All the selection logic — accepted-point masking, detection-window trimming, seeded
    downsampling, padding to ATCAT's static 243 — lives in data/transients_dataset.py's
    `prepare_lightcurve_arrays`, kept numpy-only so it is testable without a GPU stack. This
    closure only stacks per-object arrays into a batch and reports what was downsampled.
    """
    kwargs = modality_cfg.encoder.get("kwargs", {})
    seq_len = int(modality_cfg.max_tokens)
    window_days = float(kwargs.get("detection_window_days", 30.0))
    detection_snr = float(kwargs.get("detection_snr", 5.0))
    seed = int(kwargs.get("subsample_seed", 0))

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        stacked: dict[str, list] = {k: [] for k in ("flux", "flux_err", "time", "mask", "channel_index")}
        for oid in object_ids:
            row = raw_by_id[oid]
            arrays, info = prepare_lightcurve_arrays(
                row["lc_mjd"],
                row["atcat_flux"],
                row["atcat_flux_error"],
                row["atcat_band_id"],
                row["atcat_use"],
                object_id=oid,
                seq_len=seq_len,
                detection_window_days=window_days,
                detection_snr=detection_snr,
                seed=seed,
            )
            if info["downsampled"]:
                logger.warning(
                    f"[lightcurve] {oid}: {info['n_in_window']} accepted points inside the "
                    f"detection window exceeds ATCAT's fixed sequence length {seq_len}; kept "
                    f"{info['n_selected']} chosen uniformly at random "
                    f"({info['n_selected'] / info['n_in_window']:.0%} of them, seeded per object "
                    "so re-runs are identical)."
                )
            for key, value in arrays.items():
                stacked[key].append(value)
        return {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in stacked.items()}

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


def _image_batch_loader(pixels_by_id: dict, bands: list[str], id_map: dict[str, str] | None = None):
    """`pixels_by_id`: {object_id_legacy: list of {band, flux, mask, ivar, psf_fwhm, scale} dicts}
    from data/image_dataset.py:load_image_flux_pixels — real calibrated per-band flux, reassembled
    from gapatron/astrobridge-image-captions' flat `flux_{g,r,i,z}` columns. Bands are matched by
    canonical name (see _canonical_band), not by list position — position isn't guaranteed order in
    the source data. Only `flux` is used; `mask`/`ivar`/`psf_fwhm`/`scale` aren't part of AION's
    LegacySurveyImage constructor (confirmed against a working probe script — see aion_image.py).

    `id_map` translates a manifest object_id to the Legacy Survey id `pixels_by_id` is keyed by.
    The two differ for joint-tier objects, which take their object_id from the spectra side of the
    coordinate crossmatch (see data/manifest.py) while their pixels are still filed under the image
    source's own id. Omit it when the two namespaces are the same.
    """

    def _load(object_ids: list[str]) -> dict[str, torch.Tensor]:
        per_object = []
        for oid in object_ids:
            pixel_id = id_map.get(oid, oid) if id_map else oid
            if pixel_id not in pixels_by_id:
                raise KeyError(
                    f"No image pixels for manifest object_id={oid!r} (looked up as {pixel_id!r}). "
                    "The manifest's has_image flag and the image dataset have diverged — rebuild "
                    "with `make manifest` before re-running the cache."
                )
            band_entries = {_canonical_band(e["band"]): e for e in pixels_by_id[pixel_id]}
            per_band = []
            for b in bands:
                key = _canonical_band(b)
                if key not in band_entries:
                    raise KeyError(
                        f"Band {b!r} (canonicalized to {key!r}) not available for object {oid!r} "
                        f"(image id {pixel_id!r}); "
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

        spectra_df = load_spectra_table(
            cfg.sources.spectra.hf_path,
            revision=cfg.sources.spectra.get("revision"),
            files=list(cfg.sources.spectra.get("files") or []) or None,
        )
        spectra_by_id = spectra_df.set_index("object_id").to_dict(orient="index")

    transients_by_id = None
    if "lightcurve" in modality_names:
        from captioner.data.transients_dataset import load_transients_table

        transients_df = load_transients_table(
            cfg.sources.transients.hf_path, revision=cfg.sources.transients.get("revision")
        )
        transients_by_id = transients_df.set_index("object_id").to_dict(orient="index")

    image_pixels_by_id = None
    image_shape_by_id = None
    image_id_map = None
    if "image" in modality_names:
        from captioner.data.image_dataset import image_shape_groups, load_image_flux_pixels

        image_pixels_by_id = load_image_flux_pixels(
            cfg.sources.image.hf_path,
            revision=cfg.sources.image.get("revision"),
            bands=list(cfg.modalities.image.encoder.kwargs.get("bands", [])) or None,
            surveys=list(cfg.sources.image.get("surveys") or []) or None,
        )
        image_shape_by_id = image_shape_groups(image_pixels_by_id)
        # manifest object_id -> the image source's own id. Identical for image-only objects (see
        # manifest.py's backfill) and different for joint ones, which are keyed by the spectra id.
        if "object_id_legacy" in manifest.columns:
            has_legacy = manifest["object_id_legacy"].notna()
            image_id_map = dict(
                zip(manifest.loc[has_legacy, "object_id"], manifest.loc[has_legacy, "object_id_legacy"])
            )

    out_dir = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_dirty = False
    for name in modality_names:
        modality_cfg = cfg.modalities[name]
        encoder = build_encoder(name, modality_cfg, device=args.device)

        if name == "spectra":
            loader = _spectra_batch_loader(spectra_by_id)
        elif name == "image":
            loader = _image_batch_loader(
                image_pixels_by_id, list(modality_cfg.encoder.kwargs.get("bands", [])), image_id_map
            )
        elif name == "lightcurve":
            loader = _lightcurve_batch_loader(transients_by_id, modality_cfg)
        else:
            raise NotImplementedError(
                f"No batch loader wired for modality {name!r} yet — add one here when the "
                f"modality is added to configs/modalities.yaml (impl={modality_cfg.encoder.impl})."
            )

        # Image shards must not mix legacy-south's 160x160 cutouts with legacy-north's 152x152 —
        # see data/cache.py:_shard_object_ids. Keyed by manifest object_id, which is what
        # cache_modality partitions.
        group_key = None
        if name == "image" and image_shape_by_id is not None:
            group_key = {
                oid: image_shape_by_id.get((image_id_map or {}).get(oid, oid))
                for oid in manifest.loc[manifest["has_image"], "object_id"]
            }
        # Spectra need the same treatment for the same reason: DESI rows are 7781 samples and
        # SDSS ~3845-3883, so plain manifest-order sharding mixed them and padded 84.7% of
        # objects out to the longest member of whatever shard they landed in. Keyed on the
        # TRIMMED length so a shard needs no cross-object padding at all.
        if name == "spectra" and spectra_by_id is not None:
            group_key = {
                oid: spectrum_group_key(spectra_by_id[oid]["spectrum"])
                for oid in manifest.loc[manifest["has_spectra"], "object_id"]
                if oid in spectra_by_id
            }

        # The round-trip sample is a real batch through the encoder, so it is subject to the same
        # homogeneity rule as any shard — drawing the first 50 has_<name> objects straight off the
        # manifest would straddle the shape boundary and fail in the batch loader before caching
        # even starts. Take them from a single group instead.
        eligible = manifest.loc[manifest[f"has_{name}"], "object_id"]
        if group_key is not None and len(eligible):
            first_group = group_key.get(eligible.iloc[0])
            eligible = eligible[[group_key.get(o) == first_group for o in eligible]]
        sample_ids = eligible.head(50).tolist()
        if sample_ids:
            sample_batch = loader(sample_ids)
            max_err = verify_fp16_roundtrip(encoder, sample_batch, n=len(sample_ids))
            logger.info(f"[{name}] float16 round-trip max abs error over {len(sample_ids)} objects: {max_err:.6f}")

        _, excluded_ids = cache_modality(
            name, encoder, modality_cfg, manifest, loader, out_dir,
            shard=args.shard_size, group_key=group_key,
        )
        if excluded_ids:
            # Correct the manifest in place: has_<name>=False for objects whose embedding came
            # back non-finite, and re-derive `tier` for them (mirrors data/manifest.py's own
            # `joint` iff has_spectra and has_image rule) so they're never sampled for this
            # modality again — see cache_modality's docstring for why this must happen, not just
            # be logged.
            flag_col = f"has_{name}"
            excluded_mask = manifest["object_id"].isin(excluded_ids)
            manifest.loc[excluded_mask, flag_col] = False
            if "tier" in manifest.columns:
                # Mirrors data/manifest.py's rule — joint iff more than one modality present —
                # derived from the flag columns rather than a hardcoded has_spectra/has_image pair.
                flag_cols = [c for c in manifest.columns if c.startswith("has_")]
                n_present = manifest.loc[excluded_mask, flag_cols].sum(axis=1)
                manifest.loc[excluded_mask, "tier"] = np.where(n_present >= 2, "joint", "single")
            manifest_dirty = True

    if manifest_dirty:
        manifest_path = Path(cfg.manifest.parquet)
        manifest.to_parquet(manifest_path, index=False)
        logger.warning(
            f"Rewrote {manifest_path} to reflect objects excluded above — re-run `make cache` "
            "for any modality processed before this correction, or just be aware the manifest "
            "changed underneath a run already in progress."
        )


if __name__ == "__main__":
    main()
