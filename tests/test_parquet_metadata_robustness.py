"""Reproduces a real crash hit in production: a parquet file whose embedded pandas metadata
describes a nested column with a numpy_type string numpy.dtype() can't parse. Plain
`pd.read_parquet(path, columns=[...])` chokes on it even when that column is excluded from the
read, because pyarrow's pandas-metadata dtype restoration looks at every column described in the
metadata, not just the ones being materialized. The fix (_read_columns / _read_parquet in
image_dataset.py / spectra_dataset.py) reads via pyarrow directly with
`to_pandas(ignore_metadata=True)`, which skips that restoration entirely.

First hit on the old flux parquet's `image_legacy` struct column; the hazard is unchanged for
gapatron/astrobridge-image-captions, whose `flux_*`/`ivar_*`/`mask_*` columns are nested
list<list<...>> in exactly the same way.
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
        "object_id": pa.array(["a", "b"]),
        "survey": pa.array(["legacy-south", "legacy-north"]),
        "ra": pa.array([1.0, 2.0]),
        "dec": pa.array([-1.0, -2.0]),
    })
    # A nested dtype string that numpy.dtype() cannot parse, exactly matching the real crash — for
    # a column (flux_g) that isn't even present in this table, simulating it being excluded via
    # columns=[...] while its bogus metadata entry still lives in the file.
    bad_meta = {
        "index_columns": [],
        "column_indexes": [],
        "columns": [
            {
                "name": "flux_g",
                "field_name": "flux_g",
                "pandas_type": "nested",
                "numpy_type": "nested<element: [list<element: list<element: float>>]>",
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
        pd.read_parquet(path, columns=["object_id", "ra"])


def test_load_image_flux_identity_table_survives_poisoned_metadata(tmp_path):
    path = tmp_path / "poisoned.parquet"
    _write_poisoned_parquet(str(path))

    with patch("captioner.data.image_dataset.download_data_shards", return_value=[str(path)]):
        df = load_image_flux_identity_table("fake/repo")

    assert list(df["object_id_legacy"]) == ["a", "b"]
    assert list(df["ra"]) == [1.0, 2.0]
