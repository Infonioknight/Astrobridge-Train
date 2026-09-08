"""Loader for `BuildNg/astrobridge-yse-test-dataset-v2` — a real, held-out SN-typing benchmark.
Confirmed directly against the live HF repo, not assumed: 266 rows, every `object_id` unique, and
**zero overlap** with `object_id`s in the training set (`BuildNg/astrobridge-transients-dataset`)
— genuinely disjoint, not accidentally-reused training data. Same schema as the training set
(`object_id, host_image_id, lc_mjd, lc_flux_njy, lc_flux_error_njy, lc_band, atcat_flux,
atcat_flux_error, atcat_band_id, atcat_use, atcat_length, image_flux, image_bands, image_modality,
display_image, class_label`), same 3-class taxonomy — real confirmed counts on this eval set:
`SN Ia` 180, `SN II` 71, `SN Ibc` 15 (same imbalance pattern as training).

Two tracks, from the same underlying dataset:
  - **lightcurve-only** (`load_lightcurve_table`): reuses `captioner.data.transients_dataset`'s
    private loading helpers unchanged, just pointed at this repo instead — same column names,
    same `atcat_*` convention, same invariant checks (`_assert_accepted_counts`), so nothing
    dataset-shape-specific needed reinventing here.
  - **lightcurve + host image** (`load_host_image_table`): genuinely new — the training loader's
    own module docstring calls the host-image columns "debugging artifacts" and never reads them
    for training, so there was no existing loader to extend. `image_modality` is confirmed
    uniformly `LegacySurveyImage` across all 266 rows, matching what `AionImageEncoder` expects,
    but `image_flux`'s actual band count/order has NOT been separately verified here (same-org
    dataset as the training set, so less likely to have an undocumented-band surprise than
    `astronolan/galaxy10-aion` did, but check `build_raw_inputs_with_image`'s shape assertion
    output on a real run before trusting it blindly).

There is deliberately no "base model" comparison for this modality at all (see `eval/backend.py`'s
module docstring) — a raw lightcurve array isn't something an out-of-the-box vision-language model
can consume; only the equipped side is ever evaluated here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from captioner.data.transients_dataset import (
    LIGHTCURVE_COLUMNS,
    _assert_accepted_counts,
    _deduplicate_by_object_id,
    _download_and_read,
    prepare_lightcurve_arrays,
)

YSE_HF_PATH = "BuildNg/astrobridge-yse-test-dataset-v2"

# Real, confirmed values of the `class_label` column — see module docstring. Order matters for
# eval/metrics/caption_to_label.py's tie-break behaviour; kept in sync with that file's
# SN_TYPE_SYNONYMS key order deliberately.
SN_LABELS = ["SN Ia", "SN II", "SN Ibc"]

HOST_IMAGE_COLUMNS = ["object_id", "class_label", "image_flux", "image_bands", "image_modality"]


def load_lightcurve_table(
    hf_path: str = YSE_HF_PATH, revision: str | None = None, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per object: identity, `class_label`, and the raw `atcat_*` light-curve arrays —
    exactly `captioner.data.transients_dataset.load_transients_table`'s shape, reimplemented here
    (not imported) only because that function hardcodes the log message "transients from
    {hf_path}"; the underlying read/dedupe/invariant-check logic is reused directly, unchanged.
    """
    df = _download_and_read(hf_path, revision, cache_dir, LIGHTCURVE_COLUMNS)
    df = _deduplicate_by_object_id(df, "YSE eval lightcurve table")
    _assert_accepted_counts(df)
    return df


def load_host_image_table(
    hf_path: str = YSE_HF_PATH, revision: str | None = None, cache_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per object: `object_id`, `class_label`, and the raw host-image columns
    (`image_flux`, `image_bands`, `image_modality`) — genuinely new, see module docstring for why
    no existing loader covers this.
    """
    df = _download_and_read(hf_path, revision, cache_dir, HOST_IMAGE_COLUMNS)
    return _deduplicate_by_object_id(df, "YSE eval host-image table")


def build_raw_inputs_lightcurve(row: pd.Series, cfg, seed: int = 0) -> dict:
    """Turns one `load_lightcurve_table` row into the `{"lightcurve": {...}}` shape
    `captioner.inference.generate_caption` expects — identical preprocessing
    (detection-window trim / seeded downsample / pad) the training cache used, via
    `prepare_lightcurve_arrays`, so this eval track sees the model under the same input
    distribution it was trained on, not a distribution shift introduced by the eval code itself.
    """
    lc_modality = cfg.modalities.lightcurve
    lc_kwargs = lc_modality.encoder.get("kwargs", {})
    arrays, _info = prepare_lightcurve_arrays(
        row["lc_mjd"], row["atcat_flux"], row["atcat_flux_error"], row["atcat_band_id"], row["atcat_use"],
        object_id=str(row["object_id"]),
        seq_len=int(lc_modality.max_tokens),
        detection_window_days=float(lc_kwargs.get("detection_window_days", 30.0)),
        detection_snr=float(lc_kwargs.get("detection_snr", 5.0)),
        seed=int(lc_kwargs.get("subsample_seed", seed)),
    )
    return {"lightcurve": {k: torch.from_numpy(v[None, :]) for k, v in arrays.items()}}


def build_raw_inputs_with_image(row_lc: pd.Series, row_image: pd.Series, cfg, seed: int = 0) -> dict:
    """Same as `build_raw_inputs_lightcurve`, plus the host image keyed under `"image"` —
    `row_image["image_flux"]`/`row_image["image_bands"]` must already have been resolved to a
    `(n_bands, H, W)` array in the order `cfg.modalities.image.encoder.kwargs.bands` expects
    (currently `[DES-G, DES-R, DES-Z]`) before calling this; this function does NOT reorder or
    validate bands itself — do that once, explicitly, against a real sample of rows (see module
    docstring's "not yet verified" note), not silently inside this per-row helper.
    """
    raw_inputs = build_raw_inputs_lightcurve(row_lc, cfg, seed=seed)
    image_array = np.asarray(row_image["image_flux"], dtype=np.float32)
    expected_bands = len(cfg.modalities.image.encoder.kwargs.bands)
    if image_array.shape[0] != expected_bands:
        raise ValueError(
            f"object={row_image['object_id']!r}: image_flux has {image_array.shape[0]} bands, "
            f"expected {expected_bands} ({list(cfg.modalities.image.encoder.kwargs.bands)}) — "
            "resolve the real band order/identity before building raw_inputs, don't guess here."
        )
    raw_inputs["image"] = {"pixel_values": torch.from_numpy(image_array).unsqueeze(0)}
    return raw_inputs
