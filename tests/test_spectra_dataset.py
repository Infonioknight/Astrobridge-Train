"""load_spectra_table must handle AstroBridge-Data's real multi-file layout, where different
crossmatch-source parquet files carry different columns (confirmed against a real run —
`datasets.load_dataset(hf_path, split="train")` raises `CastError: ... because column names
don't match` on this exact repo, since its automatic multi-file loading requires one strict
schema across every file). pd.concat must fill missing columns with NaN instead of erroring.

Also confirmed against a real run: the four files are not disjoint by object_id — the same
target can appear in more than one of them. `test_deduplicate_*` below covers that directly; it
crashed `01_generate_captions.py`'s `DataFrame.to_dict(orient="index")` in production with
`ValueError: DataFrame index must be unique for orient='index'` before this was fixed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from captioner.data.spectra_dataset import load_spectra_table, _deduplicate_by_object_id


def test_concatenates_files_with_different_columns(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a", "b"], "mention_summary": ["x", "y"], "Z": [0.1, 0.2]})
    df1.to_parquet(tmp_path / "desi_only.parquet")

    df2 = pd.DataFrame({
        "object_id": ["c", "d"], "mention_summary": ["z", "w"], "Z": [0.3, 0.4],
        "survey_metadata_extra": ["stuff", "stuff"],  # extra column absent from df1 — this is what breaks `datasets`
    })
    df2.to_parquet(tmp_path / "sdss_only.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=[
            "observations/spectra/desi_only.parquet",
            "observations/spectra/sdss_only.parquet",
            "README.md",
        ],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert len(df) == 4
    assert set(df["object_id"]) == {"a", "b", "c", "d"}
    by_id = df.set_index("object_id")
    assert pd.isna(by_id.loc["a", "survey_metadata_extra"])
    assert by_id.loc["c", "survey_metadata_extra"] == "stuff"
    # Single-survey filenames determine every row's `survey` directly, no column needed.
    assert by_id.loc["a", "survey"] == "desi"
    assert by_id.loc["c", "survey"] == "sdss"


def test_non_parquet_files_are_ignored(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a"], "mention_summary": ["x"]})
    df1.to_parquet(tmp_path / "desi_only.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/desi_only.parquet", "observations/spectra/README.md", "LICENSE"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert len(df) == 1


def test_mixed_survey_filename_requires_survey_column(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a"], "mention_summary": ["x"]})  # no survey column
    df1.to_parquet(tmp_path / "desi_sdss_crossmatch.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/desi_sdss_crossmatch.parquet"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        try:
            load_spectra_table("UniverseTBD/AstroBridge-Data")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "survey" in str(e)


def test_mixed_survey_filename_uses_own_survey_column(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a", "b"], "mention_summary": ["x", "y"], "survey": ["DESI", "SDSS"]})
    df1.to_parquet(tmp_path / "desi_sdss_crossmatch.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/desi_sdss_crossmatch.parquet"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    by_id = df.set_index("object_id")
    assert by_id.loc["a", "survey"] == "desi"  # lowercased
    assert by_id.loc["b", "survey"] == "sdss"


def test_deduplicate_prefers_smallest_dist_arcsec():
    df = pd.DataFrame({
        "object_id": ["x", "x", "a"],
        "mention_summary": ["worse match", "better match", "only one"],
        "_dist_arcsec": [5.0, 0.05, 0.1],
    })
    deduped = _deduplicate_by_object_id(df)

    assert deduped["object_id"].is_unique
    assert len(deduped) == 2
    assert deduped.set_index("object_id").loc["x", "mention_summary"] == "better match"


def test_deduplicate_falls_back_to_first_without_dist_arcsec():
    df = pd.DataFrame({"object_id": ["x", "x", "a"], "mention_summary": ["first", "second", "only one"]})
    deduped = _deduplicate_by_object_id(df)

    assert deduped["object_id"].is_unique
    assert deduped.set_index("object_id").loc["x", "mention_summary"] == "first"


def test_deduplicate_is_a_noop_when_already_unique():
    df = pd.DataFrame({"object_id": ["a", "b"], "mention_summary": ["x", "y"]})
    deduped = _deduplicate_by_object_id(df)

    assert len(deduped) == 2
    assert list(deduped["object_id"]) == ["a", "b"]


def test_load_spectra_table_result_is_never_duplicated(tmp_path):
    df1 = pd.DataFrame({"object_id": ["x", "a"], "mention_summary": ["desi text", "text a"], "_dist_arcsec": [0.5, 0.1]})
    df1.to_parquet(tmp_path / "desi_only.parquet")

    df2 = pd.DataFrame({"object_id": ["x", "b"], "mention_summary": ["sdss text", "text b"], "_dist_arcsec": [0.05, 0.2]})
    df2.to_parquet(tmp_path / "sdss_only.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/desi_only.parquet", "observations/spectra/sdss_only.parquet"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert df["object_id"].is_unique
    # This is the exact operation that crashed in production — must succeed now.
    df.set_index("object_id")[["mention_summary"]].to_dict(orient="index")
