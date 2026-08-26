"""load_spectra_table must handle AstroBridge-Data's real multi-file layout, where different
crossmatch-source parquet files carry different columns (confirmed against a real run —
`datasets.load_dataset(hf_path, split="train")` raises `CastError: ... because column names
don't match` on this exact repo, since its automatic multi-file loading requires one strict
schema across every file). pd.concat must fill missing columns with NaN instead of erroring.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from captioner.data.spectra_dataset import load_spectra_table


def test_concatenates_files_with_different_columns(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a", "b"], "mention_summary": ["x", "y"], "Z": [0.1, 0.2]})
    df1.to_parquet(tmp_path / "desi_only.parquet")

    df2 = pd.DataFrame({
        "object_id": ["c", "d"], "mention_summary": ["z", "w"], "Z": [0.3, 0.4],
        "survey": ["SDSS", "SDSS"],  # extra column absent from df1 — this is what breaks `datasets`
    })
    df2.to_parquet(tmp_path / "desi_sdss.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=[
            "observations/spectra/desi_only.parquet",
            "observations/spectra/desi_sdss.parquet",
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
    assert pd.isna(by_id.loc["a", "survey"])
    assert by_id.loc["c", "survey"] == "SDSS"


def test_non_parquet_files_are_ignored(tmp_path):
    df1 = pd.DataFrame({"object_id": ["a"], "mention_summary": ["x"]})
    df1.to_parquet(tmp_path / "only.parquet")

    with patch(
        "huggingface_hub.list_repo_files",
        return_value=["observations/spectra/only.parquet", "observations/spectra/README.md", "LICENSE"],
    ), patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo_id, filename, **k: str(tmp_path / Path(filename).name),
    ):
        df = load_spectra_table("UniverseTBD/AstroBridge-Data")

    assert len(df) == 1
