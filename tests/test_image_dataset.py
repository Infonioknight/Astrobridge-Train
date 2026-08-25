"""load_image_captions_table only ever needs to produce (object_id, caption_blind) pairs — see
its docstring for why the J-name/coordinate parsing that used to live here was removed (identity
for the manifest join comes from load_image_flux_identity_table instead, real decimal degrees,
no parsing needed).
"""
from __future__ import annotations

import json
from unittest.mock import patch

from captioner.data.image_dataset import load_image_captions_table


def _write_json(path, **fields):
    path.write_text(json.dumps(fields))


def test_rows_missing_caption_blind_are_dropped(tmp_path):
    _write_json(tmp_path / "a_captions.json", object_id="a", caption_blind="A galaxy.")
    _write_json(tmp_path / "b_captions.json", object_id="b", caption_blind=None)

    with patch("captioner.data.image_dataset.download_caption_jsons", return_value=tmp_path):
        df = load_image_captions_table("irrelevant/repo")

    assert list(df["object_id"]) == ["a"]
    assert df.iloc[0]["caption_blind"] == "A galaxy."


def test_rows_missing_object_id_are_dropped(tmp_path):
    _write_json(tmp_path / "a_captions.json", object_id=None, caption_blind="A galaxy.")
    _write_json(tmp_path / "b_captions.json", object_id="b", caption_blind="A star.")

    with patch("captioner.data.image_dataset.download_caption_jsons", return_value=tmp_path):
        df = load_image_captions_table("irrelevant/repo")

    assert list(df["object_id"]) == ["b"]


def test_no_json_files_raises_clearly(tmp_path):
    with patch("captioner.data.image_dataset.download_caption_jsons", return_value=tmp_path):
        try:
            load_image_captions_table("irrelevant/repo")
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
