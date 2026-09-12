"""load_image_captions_table reads `gapatron/astrobridge-image-captions`'s parquet shards and
returns (object_id, caption) from the `caption_fused` column — see its docstring. Rows with an
empty/missing fused caption or object_id are dropped.
"""
from __future__ import annotations

import pandas as pd
import pytest

from captioner.data.image_dataset import IMAGE_CAPTION_COLUMN, load_image_captions_table


def _write_shard(tmp_path, rows, name="data/train-00000-of-00001.parquet"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def _patch_hub(tmp_path, monkeypatch):
    """load_image_captions_table imports list_repo_files / hf_hub_download from huggingface_hub
    inside the function body, so patching them on the module works."""
    import huggingface_hub

    shards: dict[str, str] = {}

    def _set(rows_by_file: dict[str, list[dict]]):
        for name, rows in rows_by_file.items():
            shards[name] = str(_write_shard(tmp_path, rows, name))

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **kw: list(shards.keys()))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: shards[filename])
    return _set


def test_caption_fused_column_is_returned_as_caption(_patch_hub):
    _patch_hub({
        "data/train-00000-of-00001.parquet": [
            {"object_id": "a", IMAGE_CAPTION_COLUMN: "A compact source.", "flux_g": [1, 2]},
            {"object_id": "b", IMAGE_CAPTION_COLUMN: "An extended disk.", "flux_g": [3, 4]},
        ],
    })
    df = load_image_captions_table("gapatron/astrobridge-image-captions")
    assert list(df.columns) == ["object_id", "caption"]
    assert dict(zip(df["object_id"], df["caption"])) == {"a": "A compact source.", "b": "An extended disk."}


def test_rows_missing_caption_are_dropped(_patch_hub):
    _patch_hub({
        "data/train-00000-of-00001.parquet": [
            {"object_id": "a", IMAGE_CAPTION_COLUMN: "A galaxy."},
            {"object_id": "b", IMAGE_CAPTION_COLUMN: None},
        ],
    })
    df = load_image_captions_table("irrelevant/repo")

    assert list(df["object_id"]) == ["a"]


def test_multiple_shards_are_concatenated(_patch_hub):
    _patch_hub({
        "data/train-00000-of-00002.parquet": [{"object_id": "a", IMAGE_CAPTION_COLUMN: "One."}],
        "data/train-00001-of-00002.parquet": [{"object_id": "b", IMAGE_CAPTION_COLUMN: "Two."}],
    })
    df = load_image_captions_table("gapatron/astrobridge-image-captions")
    assert set(df["object_id"]) == {"a", "b"}


def test_no_parquet_shards_raises_clearly(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo_id, **kw: ["README.md"])
    with pytest.raises(FileNotFoundError, match="parquet"):
        load_image_captions_table("gapatron/astrobridge-image-captions")
