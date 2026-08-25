"""Loaders for `gapatron/legacy_survey_south_images_captions`, against two *confirmed* real
shapes — neither is the `datasets.load_dataset`-with-an-`image`-column shape this codebase
originally assumed.

**Caption source** (`load_image_captions_table`): raw file pairs, `{object_id}_rgb.png` +
`{object_id}_captions.json`, 2,410 pairs (the repo lists 4,821 files total: 2,410 pairs + one
`.gitattributes`). `caption_blind` is generated without literature/external context, i.e.
grounded only in what the image itself shows, with the dataset's own leak-detection already
checking for stage contamination — used directly for the image tier in 01_generate_captions.py
rather than re-derived via our own keyword decomposition. Pixel data here is a rendered RGB PNG
only, not usable for AION (see aion_image.py) — irrelevant now that real flux exists (below).

**Pixel source** (`load_image_flux_identity_table` / `load_image_flux_pixels`):
`legacy_south_all_images.parquet`, a *separate* file in the same repo — confirmed schema via
`pyarrow.parquet.ParquetFile(...).schema`. 2,399 rows, each one an ALREADY crossmatched pair: a
`target_object_id_target` (= AstroBridge-Data's own `object_id` — a direct join key, no
coordinate crossmatch needed for this subset) plus `object_id_legacy`/`ra_legacy`/`dec_legacy`
(the Legacy Survey side's own identity, real decimal degrees) plus `image_legacy`: a list of
per-band structs (`band`, `flux`, `mask`, `ivar`, `psf_fwhm`, `scale`), each `flux`/`mask`/`ivar`
a 2D (H, W) array. This is real calibrated flux — what AION's LegacySurveyImage actually needs,
unlike the RGB PNGs above. `rgb_legacy` (bytes/path) is also present but unused, same reason.
This table's own `ra`/`dec` are what manifest.py actually joins on — real decimal degrees, no
coordinate parsing needed, unlike the caption source above.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLUX_PARQUET_FILENAME = "legacy_south_all_images.parquet"


def download_caption_jsons(hf_path: str, revision: str | None = None, cache_dir: Path | None = None) -> Path:
    """Pulls only the *_captions.json files, not the PNGs — caption text and coordinates don't
    need pixel data, and this keeps the download to ~90MB instead of ~250MB.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=hf_path,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["*_captions.json"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local_dir)


def load_image_captions_table(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One row per object: object_id, caption_blind. That's the only pair actually consumed
    (scripts/01_generate_captions.py looks up captions by object_id — identity/coordinates for
    the manifest come from the flux parquet instead, see load_image_flux_identity_table above),
    so that's all this returns; a caption with no `caption_blind` is dropped since there's
    nothing to caption the image tier with.
    """
    local_dir = download_caption_jsons(hf_path, revision, cache_dir)
    json_paths = sorted(local_dir.rglob("*_captions.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No *_captions.json files found under {local_dir} — check hf_path={hf_path!r} and "
            "that gated access has actually been granted."
        )

    rows = []
    for p in json_paths:
        with open(p) as fh:
            rec = json.load(fh)
        rows.append({"object_id": rec.get("object_id"), "caption_blind": rec.get("caption_blind")})

    df = pd.DataFrame(rows).dropna(subset=["object_id", "caption_blind"]).reset_index(drop=True)
    return df


def _download_flux_parquet(
    hf_path: str, revision: str | None, filename: str, cache_dir: Path | None
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=hf_path,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def load_image_flux_identity_table(
    hf_path: str,
    revision: str | None = None,
    filename: str = FLUX_PARQUET_FILENAME,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Lightweight identity columns only from the flux parquet — NOT `image_legacy` (the ~1.7GB
    nested flux/ivar/mask arrays) or `rgb_legacy`; see load_image_flux_pixels() for those.
    `object_id` here is `target_object_id_target`, i.e. AstroBridge-Data's own id — this table's
    rows are already-crossmatched pairs, so this becomes a direct join key in manifest.py rather
    than needing coordinate-based matching.
    """
    local_path = _download_flux_parquet(hf_path, revision, filename, cache_dir)
    df = pd.read_parquet(
        local_path,
        columns=["target_object_id_target", "object_id_legacy", "ra_legacy", "dec_legacy", "_dist_arcsec"],
    )
    df = df.rename(columns={"target_object_id_target": "object_id", "ra_legacy": "ra", "dec_legacy": "dec"})
    df["has_image"] = True
    return df


def load_image_flux_pixels(
    hf_path: str,
    revision: str | None = None,
    filename: str = FLUX_PARQUET_FILENAME,
    cache_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """object_id (= target_object_id_target, matching load_image_flux_identity_table) -> the
    per-band list from `image_legacy` (each entry: band/flux/mask/ivar/psf_fwhm/scale), for
    scripts/02_cache_embeddings.py's batch loader. Loaded once into memory (~1.7GB at today's row
    count) — reasonable for 2,399 rows; revisit if the crossmatch grows much larger.
    """
    local_path = _download_flux_parquet(hf_path, revision, filename, cache_dir)
    df = pd.read_parquet(local_path, columns=["target_object_id_target", "image_legacy"])
    return dict(zip(df["target_object_id_target"], df["image_legacy"]))
