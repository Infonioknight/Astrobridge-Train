"""load_image_captions_table / load_image_flux_pixels against gapatron/astrobridge-image-captions'
schema: one `datasets`-style parquet dataset carrying captions, identity and flat per-band flux
columns together, replacing the loose *_captions.json + separate flux parquet layout of
gapatron/legacy_survey_south_images_captions.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from captioner.data.image_dataset import (
    image_shape_groups,
    load_image_captions_table,
    load_image_flux_identity_table,
    load_image_flux_pixels,
)


def _cutout(value: float, shape=(3, 3)) -> list[list[float]]:
    return np.full(shape, value, dtype=np.float32).tolist()


def _write_dataset(path, rows: list[dict]) -> None:
    """Writes a shard with this dataset's real column layout. Bands absent from a row are written
    as null — exactly how legacy-north's missing ivar/mask/i-band arrive.
    """
    columns: dict[str, list] = {
        "object_id": [r["object_id"] for r in rows],
        "survey": [r.get("survey", "legacy-south") for r in rows],
        "ra": [r.get("ra", 1.0) for r in rows],
        "dec": [r.get("dec", -1.0) for r in rows],
        "caption_blind": [r.get("caption_blind") for r in rows],
        "caption_fused": [r.get("caption_fused") for r in rows],
    }
    for band in ("g", "r", "i", "z"):
        columns[f"flux_{band}"] = [r.get(f"flux_{band}") for r in rows]
        columns[f"ivar_{band}"] = [r.get(f"ivar_{band}") for r in rows]
        columns[f"mask_{band}"] = [r.get(f"mask_{band}") for r in rows]
        columns[f"psf_fwhm_{band}"] = [r.get(f"psf_fwhm_{band}") for r in rows]
        columns[f"scale_{band}"] = [r.get(f"scale_{band}") for r in rows]
    pq.write_table(pa.table(columns), path)


def _patch_shards(path):
    return patch("captioner.data.image_dataset.download_data_shards", return_value=[str(path)])


def test_rows_missing_caption_are_dropped(tmp_path):
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "a", "caption_blind": "A galaxy."},
        {"object_id": "b", "caption_blind": None},
    ])
    with _patch_shards(path):
        df = load_image_captions_table("irrelevant/repo")

    assert list(df["object_id"]) == ["a"]
    assert df.iloc[0]["caption_blind"] == "A galaxy."


def test_caption_field_selects_the_stage_but_keeps_the_column_name(tmp_path):
    """01_generate_captions.py looks up `caption_blind` unconditionally, so switching stages must
    not change the returned column name — only where its text came from.
    """
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "a", "caption_blind": "blind text", "caption_fused": "fused text"},
    ])
    with _patch_shards(path):
        df = load_image_captions_table("irrelevant/repo", caption_field="caption_fused")

    assert list(df.columns) == ["object_id", "caption_blind"]
    assert df.iloc[0]["caption_blind"] == "fused text"


def test_unknown_caption_field_raises_before_downloading(tmp_path):
    with pytest.raises(ValueError, match="caption_field"):
        load_image_captions_table("irrelevant/repo", caption_field="caption_imaginary")


def test_identity_table_exposes_object_id_legacy_not_object_id(tmp_path):
    """The manifest join depends on this: an `object_id` column here would send build_manifest
    down its direct-merge path against an id namespace that shares nothing with AstroBridge-Data's.
    """
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "0001m057-6125", "ra": 10.0, "dec": -5.0},
        {"object_id": "100313", "survey": "legacy-north", "ra": 30.0, "dec": 40.0},
    ])
    with _patch_shards(path):
        df = load_image_flux_identity_table("irrelevant/repo")

    assert "object_id" not in df.columns
    assert list(df["object_id_legacy"]) == ["0001m057-6125", "100313"]
    assert list(df["has_image"]) == [True, True]
    assert list(df.index) == [0, 1]  # manifest.py's crossmatch path indexes this positionally


def test_surveys_filter_selects_one_survey(tmp_path):
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "south1", "survey": "legacy-south"},
        {"object_id": "north1", "survey": "legacy-north"},
    ])
    with _patch_shards(path):
        df = load_image_flux_identity_table("irrelevant/repo", surveys=["legacy-south"])

    assert list(df["object_id_legacy"]) == ["south1"]


def test_unknown_survey_name_raises(tmp_path):
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [{"object_id": "south1", "survey": "legacy-south"}])
    with _patch_shards(path), pytest.raises(ValueError, match="legacy-east"):
        load_image_flux_identity_table("irrelevant/repo", surveys=["legacy-east"])


def test_duplicate_ids_across_surveys_raise(tmp_path):
    """south's brick names and north's integers are assumed disjoint; a collision would turn a
    later merge into a cross-product, so it must fail loudly rather than propagate.
    """
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "same", "survey": "legacy-south"},
        {"object_id": "same", "survey": "legacy-north"},
    ])
    with _patch_shards(path), pytest.raises(ValueError, match="duplicated object_id_legacy"):
        load_image_flux_identity_table("irrelevant/repo")


def test_flux_pixels_rebuild_the_per_band_dict_shape(tmp_path):
    """_image_batch_loader consumes {band, flux, ...} dicts — the layout the old repo stored
    natively and this one has to reassemble from flat columns.
    """
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [{
        "object_id": "a",
        "flux_g": _cutout(1.0), "flux_r": _cutout(2.0), "flux_z": _cutout(3.0),
        "psf_fwhm_g": 1.5, "scale_g": 0.262,
    }])
    with _patch_shards(path):
        pixels = load_image_flux_pixels("irrelevant/repo", bands=["DES-G", "DES-R", "DES-Z"])

    assert set(pixels) == {"a"}
    by_band = {e["band"]: e for e in pixels["a"]}
    assert sorted(by_band) == ["g", "r", "z"]
    assert by_band["g"]["psf_fwhm"] == pytest.approx(1.5)

    # pyarrow materializes a list<list<float32>> column as an object array of per-row arrays, not
    # as a 2D array — so a flat np.asarray(..., dtype=float32) over it raises. That is the shape
    # 02_cache_embeddings.py:_flux_to_array is written against (it iterates rows and np.stacks
    # them, which is also how it detects the ragged/None rows this data really contains), so the
    # assertion goes through the same door rather than assuming a rectangle.
    rows = list(by_band["r"]["flux"])
    assert np.stack([np.asarray(r, dtype=np.float32) for r in rows]).tolist() == _cutout(2.0)


def test_null_bands_are_omitted_not_emitted_as_none(tmp_path):
    """legacy-north has no i-band at all. Emitting it as a None flux would reach _flux_to_array
    and become a NaN cutout, which survives the encoder and silently poisons the cached embedding
    — the failure data/cache.py's NaN exclusion exists to catch. Omitting it instead surfaces as
    _image_batch_loader's explicit "band not available" KeyError.
    """
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [{
        "object_id": "north1", "survey": "legacy-north",
        "flux_g": _cutout(1.0), "flux_r": _cutout(2.0), "flux_z": _cutout(3.0), "flux_i": None,
    }])
    with _patch_shards(path):
        pixels = load_image_flux_pixels("irrelevant/repo", bands=["g", "r", "i", "z"])

    assert [e["band"] for e in pixels["north1"]] == ["g", "r", "z"]


def test_requesting_a_band_the_dataset_lacks_raises(tmp_path):
    with pytest.raises(ValueError, match="not in this dataset"):
        load_image_flux_pixels("irrelevant/repo", bands=["DES-Y"])


def test_shape_groups_separate_the_two_cutout_sizes(tmp_path):
    """160x160 south vs 152x152 north — the split data/cache.py must shard around."""
    path = tmp_path / "train-00000.parquet"
    _write_dataset(path, [
        {"object_id": "south1", "survey": "legacy-south",
         "flux_g": _cutout(1.0, (4, 4)), "flux_r": _cutout(1.0, (4, 4)), "flux_z": _cutout(1.0, (4, 4))},
        {"object_id": "north1", "survey": "legacy-north",
         "flux_g": _cutout(1.0, (3, 3)), "flux_r": _cutout(1.0, (3, 3)), "flux_z": _cutout(1.0, (3, 3))},
    ])
    with _patch_shards(path):
        pixels = load_image_flux_pixels("irrelevant/repo", bands=["g", "r", "z"])

    groups = image_shape_groups(pixels)
    assert groups["south1"] == (3, 4, 4)
    assert groups["north1"] == (3, 3, 3)
    assert groups["south1"] != groups["north1"]
