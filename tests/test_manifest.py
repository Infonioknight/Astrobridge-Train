"""build_manifest's direct object_id join — legacy_south_all_images.parquet's
target_object_id_target is AstroBridge-Data's own object_id, so this path should be preferred
over coordinate crossmatching whenever it's available (see manifest.py).
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

from captioner.data.manifest import build_manifest


def _cfg():
    return OmegaConf.create(
        {
            "sources": {"spectra": {"hf_path": "x", "split": "train"}, "image": {"hf_path": "y"}},
            "join": {"key": "_healpix_29", "fallback_radius_arcsec": 1.0},
            "sanity": {"min_joint_objects": 500, "min_per_subset": 50},
        }
    )


def test_object_id_join_used_when_available_and_no_shared_key():
    spectra_df = pd.DataFrame(
        {
            "object_id": ["a", "b", "c"],
            "ra_spectra": [1.0, 2.0, 3.0],
            "dec_spectra": [1.0, 2.0, 3.0],
            "spectrum": [{"flux": [1.0]}] * 3,
            "has_spectra": [True, True, True],
        }
    )
    # "z" has no matching spectra object_id — must not silently vanish or crash.
    image_df = pd.DataFrame(
        {
            "object_id": ["a", "z"],
            "object_id_legacy": ["0001m057-6125", "9999p000-0000"],
            "ra": [10.0, 30.0],
            "dec": [-5.0, -7.0],
            "has_image": [True, True],
        }
    )

    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ):
        manifest, stats = build_manifest(_cfg())

    assert stats["join_method"] == "object_id"
    by_id = manifest.set_index("object_id")
    assert by_id.loc["a", "tier"] == "joint"
    assert by_id.loc["b", "tier"] == "single"
    assert by_id.loc["c", "tier"] == "single"
    assert by_id.loc["z", "tier"] == "single"
    assert by_id.loc["z", "has_spectra"] == False  # noqa: E712
    assert by_id.loc["a", "object_id_legacy"] == "0001m057-6125"
