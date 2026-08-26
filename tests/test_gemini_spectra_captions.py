"""load_gemini_spectra_captions against the confirmed real JSONL shape (one JSON object per
line: object_key, output.caption, output.is_insufficient, output.thought_summaries + other
metadata we don't need) — see spectra_dataset.py's module docstring.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from captioner.data.spectra_dataset import load_gemini_spectra_captions


def _record(object_key: str, caption: str | None, is_insufficient: bool = False) -> dict:
    return {
        "object_key": object_key,
        "dataset_source": "sdss",
        "ra": 139.5,
        "dec": 4.4,
        "strategy": "combined_v1",
        "model": "gemini-3.7-flash",
        "output": {
            "caption": caption,
            "thought_summaries": ["long chain-of-thought text that must never end up as a caption"],
            "is_insufficient": is_insufficient,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "provenance": {"crossmatch_radius_arcsec": 1.0, "prompt_template": "combined_v1"},
        },
    }


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_drops_insufficient_and_missing_rows(tmp_path):
    records = [
        _record("gmw_00000046", "A real caption about the spectrum."),
        _record("gmw_00000057", "Another real caption.", is_insufficient=True),  # must be dropped
        _record("gmw_00000099", None),  # missing caption — must be dropped
        {"output": {"caption": "orphan"}},  # missing object_key — must be dropped
    ]
    path = tmp_path / "captions.jsonl"
    _write_jsonl(path, records)

    with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
        df = load_gemini_spectra_captions("fake/repo", "captions.jsonl")

    assert len(df) == 1
    assert df.iloc[0]["wiki_entity_id"] == "gmw_00000046"
    assert df.iloc[0]["caption"] == "A real caption about the spectrum."
    assert "thought_summaries" not in df.columns


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "captions.jsonl"
    path.write_text(
        json.dumps(_record("gmw_1", "caption one")) + "\n\n" + json.dumps(_record("gmw_2", "caption two")) + "\n"
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
        df = load_gemini_spectra_captions("fake/repo", "captions.jsonl")

    assert len(df) == 2
    assert set(df["wiki_entity_id"]) == {"gmw_1", "gmw_2"}
