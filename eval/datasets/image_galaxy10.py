"""Loader for `astronolan/galaxy10-aion` — a Galaxy Zoo 10-class morphology benchmark, already
pre-built for AION specifically (not the generic Galaxy10 DECaLS release), by the same person
("Nolan") behind the spectra crossmatch file used elsewhere in this project. Has a real, proper
train/test split already (`data/train-*.parquet` x11, `data/test-*.parquet` x2) — this loader
only ever reads the `test` split.

Confirmed directly against the live HF repo: 796 test rows, columns `image_rgb` (struct
`{bytes, path}`, pre-rendered picture), `ra`, `dec`, `Galaxy10_DECals_index`, `label` (int 0-9),
`label_name` (string), `image_bands` (nested list). Real confirmed 10-class distribution on test:
Round Smooth Galaxies 148, Edge-on Galaxies with Bulge 123, Merging Galaxies 87, Unbarred Tight
Spiral Galaxies 90, Unbarred Loose Spiral Galaxies 90, Barred Spiral Galaxies 86, In-between Round
Smooth Galaxies 69, Disturbed Galaxies 49, Edge-on Galaxies without Bulge 46, Cigar Shaped Smooth
Galaxies 8.

**Real, unresolved-until-verified band question**: `image_bands` has 4 bands (96x96 each), not
the 3 (`[DES-G, DES-R, DES-Z]`) `AionImageEncoder` is configured for — no `image_band_names`-style
column exists to disambiguate directly from the schema. Working hypothesis (not yet empirically
confirmed): griz — g, r, i, z, ordered `[g, r, i, z]` — dropping index 2 (`i`) reproduces a grz
triple in the right order for AION's encoder, since this is the same survey convention our grz
encoder already targets. `load_galaxy10_aion_bands` implements this hypothesis and raises loudly
if `image_bands` doesn't have exactly 4 bands (so a future schema change fails fast rather than
silently mis-selecting), but does NOT itself verify the hypothesis is correct — that's a real,
one-off empirical check (render the resulting grz composite for a handful of rows and compare
against those same rows' `image_rgb` field, itself a Lupton-style grz composite) to run before
trusting predictions from this function; if it disagrees, re-open the question with Nolan
directly rather than continuing to guess.

`load_galaxy10_rgb_only` has no such blocker — it feeds the base-model comparison side (which
never touches AION) directly from the pre-rendered `image_rgb` field, so it's usable immediately.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

GALAXY10_HF_PATH = "astronolan/galaxy10-aion"

# Real, confirmed `label_name` values, in `label` int order (0-9) — see module docstring. Order
# matters for eval/metrics/caption_to_label.py's tie-break behaviour; kept in sync with that
# file's GALAXY10_LABEL_SYNONYMS key order deliberately.
GALAXY10_LABELS = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# Working hypothesis only (see module docstring) — verify before trusting load_galaxy10_aion_bands.
_HYPOTHESIZED_BAND_ORDER = ["g", "r", "i", "z"]
_KEEP_BAND_INDICES = [0, 1, 3]  # g, r, z — dropping index 2 ("i")


def _test_parquet_paths(hf_path: str = GALAXY10_HF_PATH) -> list[str]:
    from huggingface_hub import list_repo_files

    files = sorted(
        f for f in list_repo_files(hf_path, repo_type="dataset")
        if f.startswith("data/test-") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"No 'data/test-*.parquet' files found in {hf_path!r} — check the repo layout hasn't changed."
        )
    return files


def _download_and_read(hf_path: str, columns: list[str], cache_dir: Path | None = None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    import pyarrow.parquet as pq

    frames = []
    for f in _test_parquet_paths(hf_path):
        local_path = hf_hub_download(
            repo_id=hf_path, filename=f, repo_type="dataset", cache_dir=str(cache_dir) if cache_dir else None,
        )
        table = pq.read_table(local_path, columns=columns)
        frames.append(table.to_pandas(ignore_metadata=True))
    return pd.concat(frames, ignore_index=True, sort=False)


def load_galaxy10_rgb_only(hf_path: str = GALAXY10_HF_PATH, cache_dir: Path | None = None) -> pd.DataFrame:
    """One row per test-set object: `label`, `label_name`, and the pre-rendered `image_rgb`
    struct (`{bytes, path}` — decode with `decode_rgb_image` below). Feeds the base-model
    comparison side only; never touches `image_bands`/AION.
    """
    return _download_and_read(hf_path, ["Galaxy10_DECals_index", "image_rgb", "label", "label_name"], cache_dir)


def decode_rgb_image(image_rgb_struct: dict) -> Image.Image:
    """`image_rgb_struct["bytes"]` is a PNG/JPEG-encoded picture (HF's standard `Image` feature
    struct) — decode it into a plain `PIL.Image` for `eval.backend`'s base-side `{"image": ...}`
    raw_inputs contract.
    """
    return Image.open(io.BytesIO(image_rgb_struct["bytes"]))


def load_galaxy10_aion_bands(hf_path: str = GALAXY10_HF_PATH, cache_dir: Path | None = None) -> pd.DataFrame:
    """One row per test-set object: `label`, `label_name`, and `image_bands` — the raw multi-band
    array, untouched (band selection happens in `build_raw_inputs`, not here, so the "not yet
    verified" hypothesis stays isolated to one function).
    """
    return _download_and_read(hf_path, ["Galaxy10_DECals_index", "image_bands", "label", "label_name"], cache_dir)


def build_raw_inputs(row: pd.Series) -> dict:
    """Applies the griz-minus-`i` hypothesis (see module docstring) to turn one
    `load_galaxy10_aion_bands` row's `image_bands` into the `{"image": {"pixel_values": ...}}`
    shape `generate_caption` expects. Raises loudly if `image_bands` doesn't have exactly 4 bands
    — a real schema change should fail fast here, not silently select the wrong 3.
    """
    bands = np.stack([np.asarray(b, dtype=np.float32) for b in row["image_bands"]], axis=0)
    if bands.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands (hypothesized {_HYPOTHESIZED_BAND_ORDER}), got {bands.shape[0]} — "
            "the griz-minus-i selection below assumes exactly 4; re-check the real schema before "
            "trusting this function."
        )
    grz = bands[_KEEP_BAND_INDICES]  # (3, H, W), hypothesized [g, r, z] order
    return {"image": {"pixel_values": torch.from_numpy(grz).unsqueeze(0)}}
