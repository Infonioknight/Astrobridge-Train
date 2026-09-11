"""Reproduces a real crash hit in production: legacy_south_all_images.parquet's embedded pandas
metadata describes a nested struct column (image_legacy) with a numpy_type string
numpy.dtype() can't parse. Plain `pd.read_parquet(path, columns=[...])` chokes on it even when
that column is excluded from the read, because pyarrow's pandas-metadata dtype restoration looks
at every column described in the metadata, not just the ones being materialized. The fix
(_read_parquet_columns / _read_parquet in image_dataset.py / spectra_dataset.py) reads via
pyarrow directly with `to_pandas(ignore_metadata=True)`, which skips that restoration entirely.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

from captioner.data.image_dataset import load_image_flux_identity_table


def _write_poisoned_parquet(path):
    table = pa.table({
        "target_object_id_target": pa.array(["a", "b"]),
        "object_id_legacy": pa.array(["legacy_a", "legacy_b"]),
        "ra_legacy": pa.array([1.0, 2.0]),
        "dec_legacy": pa.array([-1.0, -2.0]),
        "_dist_arcsec": pa.array([0.1, 0.2]),
    })
    # A nested-struct dtype string that numpy.dtype() cannot parse, exactly matching the real
    # crash — for a column (image_legacy) that isn't even present in this table, simulating it
    # being excluded via columns=[...] while its bogus metadata entry still lives in the file.
    bad_meta = {
        "index_columns": [],
        "column_indexes": [],
        "columns": [
            {
                "name": "image_legacy",
                "field_name": "image_legacy",
                "pandas_type": "nested",
                "numpy_type": "nested<band: [string], flux: [list<element: list<element: float>>]>",
                "metadata": None,
            },
        ],
        "creator": {"library": "test", "version": "0"},
        "pandas_version": "2.0.0",
    }
    table = table.cast(table.schema.with_metadata({b"pandas": json.dumps(bad_meta).encode()}))
    pq.write_table(table, path)


def test_plain_pandas_read_would_crash_on_poisoned_metadata(tmp_path):
    """Documents the failure this test suite guards against — not testing our code, testing that
    the naive approach really does break, so this test file's premise stays honest over time.
    """
    path = tmp_path / "poisoned.parquet"
    _write_poisoned_parquet(str(path))
    with pytest.raises((ValueError, TypeError)):
        pd.read_parquet(path, columns=["target_object_id_target", "ra_legacy"])


def test_load_image_flux_identity_table_survives_poisoned_metadata(tmp_path):
    path = tmp_path / "poisoned.parquet"
    _write_poisoned_parquet(str(path))

    with patch("captioner.data.image_dataset._download_flux_parquet", return_value=str(path)):
        df = load_image_flux_identity_table("fake/repo")

    assert list(df["object_id"]) == ["a", "b"]
    assert list(df["object_id_legacy"]) == ["legacy_a", "legacy_b"]
