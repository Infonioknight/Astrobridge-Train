"""Loaders for `gapatron/astrobridge-image-captions` — 3,487 Legacy Survey cutouts (2,399
legacy-south + 1,088 legacy-north), captioned in four stages by Gemini following the AstroLLaVA
approach. Unlike the repo this replaced (`gapatron/legacy_survey_south_images_captions`, which
split its captions across ~2,410 loose `{object_id}_captions.json` files and its pixels into a
separate `legacy_south_all_images.parquet`), everything here lives in one standard
`datasets`-style parquet dataset under `data/train-*.parquet`.

**Caption source** (`load_image_captions_table`): `caption_blind` by default — the image-only,
open-ended stage, generated with no literature context and no object name, with the dataset's
own leak detection already applied. Used directly for the image tier in 01_generate_captions.py
rather than re-derived via our own keyword decomposition. Three further caption fields exist
(`caption_properties`, `caption_literature`, `caption_fused`); `caption_field` selects between
them so switching is a config change, not a code change.

**Pixel source** (`load_image_flux_identity_table` / `load_image_flux_pixels`): the same parquet,
via flat per-band columns — `flux_{g,r,i,z}`, `ivar_*`, `mask_*`, `psf_fwhm_*`, `scale_*` — rather
than the old repo's single `image_legacy` column of per-band structs. This is real calibrated
flux, what AION's LegacySurveyImage needs. `load_image_flux_pixels` reassembles it into the
per-band-dict shape `scripts/02_cache_embeddings.py:_image_batch_loader` already consumes, so
that side is unchanged.

Two survey-shaped gaps, both real properties of the upstream release rather than load errors:

- **legacy-north has no `ivar`, no `mask` and no i-band at all** (`null`). Harmless here: only
  `flux` is read, and configs/modalities.yaml requests g/r/z.
- **legacy-south cutouts are 160x160, legacy-north 152x152.** AION is fully convolutional over
  the cutout so either size encodes fine, but a single *batch* must be shape-homogeneous —
  `image_shape_groups` exists for that, and data/cache.py shards within those groups.

**Identity**: `object_id` here is the Legacy Survey's own id (brick-style `0001m057-6125` in
south, a bare integer string in north) — NOT AstroBridge-Data's `object_id`, which the old repo's
`target_object_id_target` column carried. There is therefore no shared id namespace to join on,
and this module deliberately returns that column as `object_id_legacy`, never `object_id`, so
data/manifest.py takes its coordinate-crossmatch path (`ra`/`dec` are real decimal degrees) instead
of a direct id merge that would match nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATA_FILE_PATTERN = "data/train-*.parquet"
CAPTION_FIELDS = ("caption_blind", "caption_properties", "caption_literature", "caption_fused")
AVAILABLE_BANDS = ("g", "r", "i", "z")
PER_BAND_FIELDS = ("flux", "ivar", "mask", "psf_fwhm", "scale")


def download_data_shards(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> list[str]:
    """Pulls the parquet shards (~2.4GB total).

    Unlike the old repo — where captions were loose JSON and could be fetched for ~90MB without
    touching the ~250MB of pixels — captions, coordinates and flux share these files, so the first
    caller pays for all of it. In README order that is `make manifest` (via
    load_image_flux_identity_table). `make cache` needs the same bytes anyway and the HF cache is
    shared, so this moves the download earlier rather than adding one.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=hf_path,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[DATA_FILE_PATTERN],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    shards = sorted(Path(local_dir).glob(DATA_FILE_PATTERN))
    if not shards:
        raise FileNotFoundError(
            f"No files matching {DATA_FILE_PATTERN!r} under {local_dir} — check hf_path={hf_path!r}. "
            "This loader expects a `datasets`-style parquet repo (data/train-00000-of-000NN.parquet); "
            "the pre-2026-09 layout of loose *_captions.json files is no longer supported."
        )
    return [str(p) for p in shards]


def _read_columns(paths: Sequence[str], columns: list[str]) -> pd.DataFrame:
    """Reads `columns` across every shard via pyarrow, with `to_pandas(ignore_metadata=True)`.

    `ignore_metadata` is load-bearing, not tidiness — carried over from the old flux parquet and
    kept because the same hazard applies here. Parquet files embed pandas metadata describing every
    column's dtype, and pyarrow applies that restoration to all described columns, not just the
    projected ones. A nested column whose `numpy_type` string `numpy.dtype()` cannot parse (e.g.
    the `list<list<float32>>` flux columns) therefore crashes a plain
    `pd.read_parquet(path, columns=[...])` even when that column was excluded from the read.
    Reading through pyarrow and skipping the restoration lets Arrow's own types drive dtype
    inference instead. See tests/test_parquet_metadata_robustness.py.
    """
    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        table = pq.read_table(path, columns=list(columns))
        frames.append(table.to_pandas(ignore_metadata=True))
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, ignore_index=True)


def _filter_surveys(df: pd.DataFrame, surveys: Sequence[str] | None) -> pd.DataFrame:
    """`surveys=None` keeps everything. Set `sources.image.surveys` in configs/data.yaml to e.g.
    `[legacy-south]` to reproduce the pre-swap object set exactly (2,399 objects, one cutout size,
    ivar/mask present throughout).
    """
    if not surveys:
        return df.reset_index(drop=True)
    wanted = set(surveys)
    unknown = wanted - set(df["survey"].unique())
    if unknown:
        raise ValueError(
            f"configs/data.yaml sources.image.surveys names {sorted(unknown)}, which appear "
            f"nowhere in the dataset's `survey` column (present: {sorted(df['survey'].unique())})."
        )
    return df[df["survey"].isin(wanted)].reset_index(drop=True)


def _assert_unique(df: pd.DataFrame, column: str) -> None:
    """south uses brick-style ids and north bare integers, so the two namespaces should not
    collide — but "should not" is exactly the assumption that quietly turns a later `pd.merge`
    into a cross-product, so it is checked rather than trusted.
    """
    dupes = df[column][df[column].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{dupes.nunique()} duplicated {column} values ({len(dupes)} rows) in the image "
            f"dataset, e.g. {sorted(dupes.unique()[:3].tolist())}. legacy-south (brick-style ids) "
            "and legacy-north (integer ids) are expected to be disjoint namespaces; merging on a "
            "non-unique key would silently inflate the manifest's row count."
        )


def load_image_captions_table(
    hf_path: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    caption_field: str = "caption_blind",
    surveys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per object: object_id, caption_blind. The column is always returned under the name
    `caption_blind` regardless of which `caption_field` sourced it, so 01_generate_captions.py's
    lookup doesn't change when the field does.

    `object_id` here is the Legacy Survey id — the same namespace as `object_id_legacy` from
    load_image_flux_identity_table, which is how 01_generate_captions.py joins the two.
    A row with no caption text is dropped: there is nothing to caption the image tier with.
    """
    if caption_field not in CAPTION_FIELDS:
        raise ValueError(
            f"caption_field={caption_field!r} is not one of this dataset's caption columns "
            f"{list(CAPTION_FIELDS)}."
        )
    shards = download_data_shards(hf_path, revision, cache_dir)
    df = _read_columns(shards, ["object_id", "survey", caption_field])
    df = _filter_surveys(df, surveys)
    df = df.rename(columns={caption_field: "caption_blind"})
    df = df[["object_id", "caption_blind"]]
    df = df.dropna(subset=["object_id", "caption_blind"]).reset_index(drop=True)
    df["object_id"] = df["object_id"].astype(str)
    _assert_unique(df, "object_id")
    return df


def load_image_flux_identity_table(
    hf_path: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    surveys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Lightweight identity columns only — never the flux/ivar/mask arrays; see
    load_image_flux_pixels() for those.

    Returns `object_id_legacy`/`ra`/`dec`/`survey`/`has_image`, and deliberately no `object_id`:
    this dataset carries the Legacy Survey's own id, which shares no namespace with
    AstroBridge-Data's object_id, so data/manifest.py must coordinate-crossmatch on ra/dec rather
    than merge on an id. The index is reset because manifest.py's crossmatch path addresses this
    frame positionally.
    """
    shards = download_data_shards(hf_path, revision, cache_dir)
    df = _read_columns(shards, ["object_id", "survey", "ra", "dec"])
    df = _filter_surveys(df, surveys)
    df = df.rename(columns={"object_id": "object_id_legacy"})
    df["object_id_legacy"] = df["object_id_legacy"].astype(str)
    _assert_unique(df, "object_id_legacy")
    df["has_image"] = True
    return df.reset_index(drop=True)


def load_image_flux_pixels(
    hf_path: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
    bands: Sequence[str] | None = None,
    surveys: Sequence[str] | None = None,
) -> dict[str, list[dict]]:
    """object_id_legacy -> list of per-band dicts (`band`, `flux`, and whichever of
    `ivar`/`mask`/`psf_fwhm`/`scale` the row actually has), for
    scripts/02_cache_embeddings.py's batch loader.

    The per-band-dict shape is the old repo's `image_legacy` layout, reassembled here from this
    dataset's flat `flux_g`/`flux_r`/... columns so the batch loader and its tests are unaffected
    by the source swap. `band` is the bare letter; _canonical_band in 02_cache_embeddings.py
    normalizes both sides, so a configured "DES-G" still matches.

    `bands` restricts which columns are materialized — pass the configured band list. It matters:
    all four bands at full row count is ~1.4GB resident, and i-band is legacy-south-only dead
    weight when configs/modalities.yaml asks for g/r/z. Bands that are `null` for a row (every
    ivar/mask/i-band value in legacy-north) are omitted from that row's list rather than emitted
    as None, so a missing band surfaces as _image_batch_loader's explicit "band not available for
    object" KeyError instead of a NaN that would silently poison the cached embedding.
    """
    wanted = [b.strip().lower().split("-")[-1].split("_")[-1] for b in bands] if bands else list(AVAILABLE_BANDS)
    unknown = sorted(set(wanted) - set(AVAILABLE_BANDS))
    if unknown:
        raise ValueError(
            f"Requested band(s) {unknown} are not in this dataset (available: "
            f"{list(AVAILABLE_BANDS)}). Check configs/modalities.yaml's image encoder kwargs.bands."
        )

    columns = ["object_id", "survey"]
    for band in wanted:
        columns.extend(f"{field}_{band}" for field in PER_BAND_FIELDS)

    pixels: dict[str, list[dict]] = {}
    # One shard at a time rather than _read_columns' concat: this is the heavy path (~1.4GB
    # resident for three bands at full row count) and concatenating every shard before building
    # the dict would hold both the frames and the result at once, roughly doubling the peak.
    for shard in download_data_shards(hf_path, revision, cache_dir):
        df = _filter_surveys(_read_columns([shard], columns), surveys)
        for row in df.itertuples(index=False):
            entries = []
            for band in wanted:
                flux = getattr(row, f"flux_{band}")
                if flux is None or (isinstance(flux, float) and np.isnan(flux)):
                    continue  # legacy-north's absent i-band — omit, don't emit a None flux
                entry = {"band": band, "flux": flux}
                for field in PER_BAND_FIELDS[1:]:
                    entry[field] = getattr(row, f"{field}_{band}", None)
                entries.append(entry)
            pixels[str(row.object_id)] = entries
        del df
    return pixels


def image_shape_groups(pixels_by_id: dict[str, list[dict]]) -> dict[str, tuple]:
    """object_id_legacy -> a hashable (n_bands, H, W) key, for data/cache.py's shard grouping.

    legacy-south cutouts are 160x160 and legacy-north 152x152, so sharding in plain manifest order
    would eventually straddle the boundary and hand _image_batch_loader a batch it refuses
    ("Inconsistent per-object image shapes in this batch"). Keying on the measured shape rather
    than on `survey` keeps that correct even if a future release adds a third size or re-cuts one
    of these two.
    """
    groups: dict[str, tuple] = {}
    for object_id, entries in pixels_by_id.items():
        if not entries:
            groups[object_id] = ()
            continue
        first = np.asarray(entries[0]["flux"], dtype=object)
        n_rows = len(first)
        n_cols = len(first[0]) if n_rows else 0
        groups[object_id] = (len(entries), n_rows, n_cols)
    return groups
