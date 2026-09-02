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

from captioner.data.spectra_dataset import (
    load_spectra_table,
    _canonical_object_id,
    _canonicalize_object_ids,
    _deduplicate_by_object_id,
)


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


def test_canonical_object_id_collapses_every_real_spelling():
    """The three spellings the real files actually use for one object (see
    `_canonical_object_id`'s docstring) must all reduce to the same string.
    """
    assert _canonical_object_id("b'    462849895556999168'") == "462849895556999168"
    assert _canonical_object_id(462849895556999168) == "462849895556999168"
    assert _canonical_object_id("462849895556999168") == "462849895556999168"
    assert _canonical_object_id(b"    462849895556999168") == "462849895556999168"
    # A plain id that merely starts with "b" is not a bytes repr and must survive untouched.
    assert _canonical_object_id("b1234") == "b1234"
    assert _canonical_object_id("0441p020-5245") == "0441p020-5245"


def test_missing_object_id_is_rejected_not_canonicalized_to_nan():
    df = pd.DataFrame({"object_id": ["a", None], "mention_summary": ["x", "y"]})
    try:
        _canonicalize_object_ids(df)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "object_id" in str(e)


def test_one_object_spelled_three_ways_yields_one_row(tmp_path):
    """The exact production failure: the same object arrives as a plain digit string, as int64,
    and as a stringified bytes repr, so it reached the manifest under ids that look unrelated and
    `make cache` raised `KeyError: '299516890357721088'` looking one of them up.
    """
    pd.DataFrame({
        "object_id": ["39632951360620188"],
        "mention_summary": ["desi text"],
        "_dist_arcsec": [0.5],
    }).to_parquet(tmp_path / "desi_crossmatch.parquet")

    pd.DataFrame({
        "object_id": ["b'    462849895556999168'"],
        "mention_summary": ["sdss text"],
        "_dist_arcsec": [0.4],
    }).to_parquet(tmp_path / "sdss_crossmatch.parquet")

    pd.DataFrame({
        "object_id": [39632951360620188, 462849895556999168],  # int64: same two objects
        "mention_summary": ["desi text again", "sdss text again"],
        "survey": ["DESI", "SDSS"],
        "_dist_arcsec": [0.1, 0.9],
    }).to_parquet(tmp_path / "desi_sdss_subset_crossmatch.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=[
            "observations/spectra/desi_crossmatch.parquet",
            "observations/spectra/sdss_crossmatch.parquet",
            "observations/spectra/desi_sdss_subset_crossmatch.parquet",
        ],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert set(df["object_id"]) == {"39632951360620188", "462849895556999168"}
    assert df["object_id"].is_unique
    assert all(isinstance(oid, str) for oid in df["object_id"])
    # Deduplication still tie-breaks on the crossmatch distance, now across all three spellings.
    by_id = df.set_index("object_id")
    assert by_id.loc["39632951360620188", "mention_summary"] == "desi text again"
    assert by_id.loc["462849895556999168", "mention_summary"] == "sdss text"
    # Survey routing (aion_spectrum.py depends on it) survives the collapse.
    assert by_id.loc["39632951360620188", "survey"] == "desi"
    assert by_id.loc["462849895556999168", "survey"] == "sdss"


def test_manifest_str_ids_match_the_cache_lookup_keys(tmp_path):
    """02_cache_embeddings.py keys its batch loader by the loader's raw `object_id` while asking
    for the manifest's `astype(str)` version — this is what raised KeyError. Same keys now.
    """
    pd.DataFrame({
        "object_id": [462849895556999168],
        "mention_summary": ["from the int64 file"],
    }).to_parquet(tmp_path / "sdss_crossmatch.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/sdss_crossmatch.parquet"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    raw_by_id = df.set_index("object_id").to_dict(orient="index")
    manifest_object_id = df["object_id"].astype(str).iloc[0]  # what manifest.py writes out
    assert raw_by_id[manifest_object_id]["mention_summary"] == "from the int64 file"


def _fake_repo(tmp_path, filenames):
    return (
        patch(
            "huggingface_hub.list_repo_files",
            return_value=[f"observations/spectra/{n}" for n in filenames],
        ),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
        ),
    )


def _write_spectra_files(tmp_path):
    pd.DataFrame({
        "object_id": ["shared", "desi_only"],
        "mention_summary": ["from desi", "desi exclusive"],
        "_dist_arcsec": [0.1, 0.2],
    }).to_parquet(tmp_path / "desi_crossmatch.parquet")

    pd.DataFrame({
        "object_id": ["shared", "subset_only"],
        "mention_summary": ["from subset", "subset exclusive"],
        "survey": ["DESI", "SDSS"],
        "split": ["test", "train"],
        "_dist_arcsec": [0.9, 0.3],
    }).to_parquet(tmp_path / "desi_sdss_subset_crossmatch.parquet")


def test_files_pin_reads_only_the_named_file(tmp_path):
    _write_spectra_files(tmp_path)
    listed, download = _fake_repo(
        tmp_path, ["desi_crossmatch.parquet", "desi_sdss_subset_crossmatch.parquet"]
    )
    with listed, download:
        df = load_spectra_table(
            "UniverseTBD/AstroBridge-Data",
            files=["observations/spectra/desi_sdss_subset_crossmatch.parquet"],
        )

    assert set(df["object_id"]) == {"shared", "subset_only"}
    assert df.set_index("object_id").loc["shared", "mention_summary"] == "from subset"
    assert set(df["split"]) == {"test", "train"}


def test_upstream_split_survives_the_dedup_tie_break(tmp_path):
    _write_spectra_files(tmp_path)
    listed, download = _fake_repo(
        tmp_path, ["desi_crossmatch.parquet", "desi_sdss_subset_crossmatch.parquet"]
    )
    with listed, download:
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    by_id = df.set_index("object_id")
    assert by_id.loc["shared", "mention_summary"] == "from desi"
    assert by_id.loc["shared", "split"] == "test"
    assert by_id.loc["subset_only", "split"] == "train"
    assert pd.isna(by_id.loc["desi_only", "split"])
