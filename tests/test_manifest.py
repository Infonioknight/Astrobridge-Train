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


def _cfg_with_transients():
    cfg = _cfg()
    cfg.sources.transients = {"hf_path": "t", "revision": None}
    return cfg


def test_transients_are_appended_not_joined():
    """A ZTF designation shares no namespace with AstroBridge-Data's object_id, so transients must
    arrive as NEW manifest rows carrying has_lightcurve only — never merged onto an existing row,
    and never dropped for failing to match one.
    """
    spectra_df = pd.DataFrame({"object_id": ["a"], "ra_spectra": [1.0], "dec_spectra": [1.0], "has_spectra": [True]})
    image_df = pd.DataFrame({"object_id": ["a"], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame(
        {
            "object_id": ["ZTF0000000001", "ZTF0000000002"],
            "class_label": ["SN Ia", "SN II"],
            "lc_mjd": [[1.0], [2.0]],
            "atcat_length": [1, 1],
            "has_lightcurve": [True, True],
        }
    )

    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        manifest, stats = build_manifest(_cfg_with_transients())

    assert len(manifest) == 3
    by_id = manifest.set_index("object_id")
    assert by_id.loc["ZTF0000000001", "has_lightcurve"] == True  # noqa: E712
    assert by_id.loc["ZTF0000000001", "has_image"] == False  # noqa: E712
    assert by_id.loc["ZTF0000000001", "has_spectra"] == False  # noqa: E712
    # lightcurve-only => single tier; the image+spectra object is still joint
    assert by_id.loc["ZTF0000000001", "tier"] == "single"
    assert by_id.loc["a", "tier"] == "joint"
    assert stats["availability_histogram"]["lightcurve"] == 2
    assert "lightcurve" in stats["modalities"]


def test_transient_lightcurve_arrays_are_not_written_to_the_manifest():
    """02_cache_embeddings.py reloads them from the source; duplicating the arrays into
    manifest.parquet would silently double the on-disk copy for no reader."""
    spectra_df = pd.DataFrame({"object_id": ["a"], "ra_spectra": [1.0], "dec_spectra": [1.0], "has_spectra": [True]})
    image_df = pd.DataFrame({"object_id": ["a"], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame(
        {
            "object_id": ["ZTF0000000001"],
            "class_label": ["SN Ia"],
            "lc_mjd": [[1.0, 2.0]],
            "atcat_flux": [[1.0, 2.0]],
            "atcat_flux_error": [[1.0, 1.0]],
            "atcat_band_id": [[1, 2]],
            "atcat_use": [[True, True]],
            "atcat_length": [2],
            "has_lightcurve": [True],
        }
    )

    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        manifest, _ = build_manifest(_cfg_with_transients())

    for heavy in ("lc_mjd", "atcat_flux", "atcat_flux_error", "atcat_band_id", "atcat_use"):
        assert heavy not in manifest.columns
    # scalars worth keeping for diagnostics / eval slicing
    assert "class_label" in manifest.columns
    assert "atcat_length" in manifest.columns


def test_colliding_transient_ids_raise_rather_than_duplicating_rows():
    spectra_df = pd.DataFrame({"object_id": ["dup"], "ra_spectra": [1.0], "dec_spectra": [1.0], "has_spectra": [True]})
    image_df = pd.DataFrame({"object_id": ["dup"], "ra": [1.0], "dec": [1.0], "has_image": [True]})
    transients_df = pd.DataFrame({"object_id": ["dup"], "atcat_length": [1], "has_lightcurve": [True]})

    with patch("captioner.data.manifest._load_spectra_table", return_value=spectra_df), patch(
        "captioner.data.manifest._load_image_table", return_value=image_df
    ), patch("captioner.data.manifest._load_transients_table", return_value=transients_df):
        try:
            build_manifest(_cfg_with_transients())
            assert False, "expected ValueError on colliding object_id"
        except ValueError as e:
            assert "collide" in str(e)
