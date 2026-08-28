"""Loader + light-curve preparation for BuildNg/astrobridge-transients-dataset.

Deliberately torch-free (as is the module under test), so the selection logic with real failure
modes — accepted-point masking, detection-window trimming, seeded downsampling, padding — is
testable without a GPU stack.
"""
from __future__ import annotations

import huggingface_hub
import numpy as np
import pandas as pd
import pytest

from captioner.data import transients_dataset as td
from captioner.data.transients_dataset import load_transients_table, prepare_lightcurve_arrays


def _write_shard(tmp_path, rows, filename="part-00000.parquet"):
    path = tmp_path / filename
    pd.DataFrame(rows).to_parquet(path)
    return path


def _patch_download(monkeypatch, paths):
    monkeypatch.setattr(td, "_list_parquet_files", lambda hf_path, revision: [f"data/{i}" for i in range(len(paths))])
    it = {f"data/{i}": str(p) for i, p in enumerate(paths)}
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: it[filename]
    )


def _row(object_id="ZTF0000000001", n=3, length=None, **over):
    row = {
        "object_id": object_id,
        "class_label": "SN Ia",
        "lc_mjd": [float(i) for i in range(n)],
        "atcat_flux": [100.0] * n,
        "atcat_flux_error": [1.0] * n,
        "atcat_band_id": [1, 2] * n,
        "atcat_use": [True] * n,
        "atcat_length": n if length is None else length,
        "transient_caption": "The light curve rises to a peak.",
        # Debugging artifacts that must never be read into training — see module docstring.
        "image_flux": [0.5] * 4,
        "host_image_caption": "a spiral galaxy",
        "host_image_id": "legacy_9011_1_1",
    }
    row["atcat_band_id"] = row["atcat_band_id"][:n]
    row.update(over)
    return row


class TestLoadTransientsTable:
    def test_host_image_columns_are_never_loaded(self, tmp_path, monkeypatch):
        """The host image is a debugging artifact for this dataset. Reading it at all is the bug —
        it must not be possible for it to reach training by accident."""
        _patch_download(monkeypatch, [_write_shard(tmp_path, [_row()])])
        df = load_transients_table("x")
        for forbidden in ("image_flux", "display_image", "host_image_caption", "host_image_id"):
            assert forbidden not in df.columns
        assert df["has_lightcurve"].all()

    def test_accepted_count_mismatch_raises(self, tmp_path, monkeypatch):
        """`atcat_use` is the only thing keeping the band_id=0 excluded-i sentinel from being read
        as u-band by ATCAT, so a use/length disagreement must fail loudly, not silently."""
        bad = _row(n=3, length=99)
        _patch_download(monkeypatch, [_write_shard(tmp_path, [bad])])
        with pytest.raises(ValueError, match="atcat_length"):
            load_transients_table("x")

    def test_duplicate_object_ids_are_collapsed(self, tmp_path, monkeypatch):
        a = _write_shard(tmp_path, [_row("ZTF0000000001")], "a.parquet")
        b = _write_shard(tmp_path, [_row("ZTF0000000001")], "b.parquet")
        _patch_download(monkeypatch, [a, b])
        df = load_transients_table("x")
        assert len(df) == 1

    def test_multiple_shards_are_concatenated(self, tmp_path, monkeypatch):
        a = _write_shard(tmp_path, [_row("ZTF0000000001")], "a.parquet")
        b = _write_shard(tmp_path, [_row("ZTF0000000002")], "b.parquet")
        _patch_download(monkeypatch, [a, b])
        assert len(load_transients_table("x")) == 2


class TestPrepareLightcurveArrays:
    def test_pads_to_seq_len_and_marks_mask(self):
        arrays, info = prepare_lightcurve_arrays(
            [0.0, 1.0, 2.0], [100.0] * 3, [1.0] * 3, [1, 2, 1], [True] * 3, seq_len=10
        )
        for k in ("flux", "flux_err", "time", "mask", "channel_index"):
            assert arrays[k].shape == (10,)
        assert arrays["mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        assert arrays["time"][0] == 0.0
        assert info["n_selected"] == 3
        assert info["downsampled"] == 0

    def test_unaccepted_points_are_dropped(self):
        """band_id 0 is the excluded-i sentinel and is u-band to ATCAT; only `use` keeps it out."""
        arrays, info = prepare_lightcurve_arrays(
            [0.0, 1.0, 2.0], [100.0] * 3, [1.0] * 3, [1, 0, 2], [True, False, True], seq_len=5
        )
        assert info["n_accepted"] == 2
        assert arrays["mask"].tolist() == [1, 1, 0, 0, 0]
        assert 0 not in arrays["channel_index"][: info["n_selected"]].tolist()

    def test_detection_window_trims_baseline(self):
        """The real shape of the long light curves: a short outburst inside a multi-year baseline.
        Only the window around detections may reach ATCAT — outside it is out of distribution."""
        mjd = [0.0, 1000.0, 1001.0, 1002.0, 2000.0]
        flux = [1.0, 100.0, 100.0, 100.0, 1.0]
        err = [1.0] * 5
        arrays, info = prepare_lightcurve_arrays(
            mjd, flux, err, [1] * 5, [True] * 5, seq_len=10, detection_window_days=30.0, detection_snr=5.0
        )
        assert info["n_accepted"] == 5
        assert info["n_in_window"] == 3
        assert arrays["time"][:3].tolist() == [0.0, 1.0, 2.0]

    def test_downsamples_over_length_and_is_deterministic(self):
        n = 300
        args = ([float(i) for i in range(n)], [100.0] * n, [1.0] * n, [1, 2] * (n // 2), [True] * n)
        a, ia = prepare_lightcurve_arrays(*args, object_id="ZTF9", seq_len=50)
        b, ib = prepare_lightcurve_arrays(*args, object_id="ZTF9", seq_len=50)
        assert ia["downsampled"] == 1 and ia["n_selected"] == 50
        assert np.array_equal(a["time"], b["time"]), "same object must select the same points"
        assert int(a["mask"].sum()) == 50

    def test_downsampling_differs_between_objects(self):
        n = 300
        args = ([float(i) for i in range(n)], [100.0] * n, [1.0] * n, [1] * n, [True] * n)
        a, _ = prepare_lightcurve_arrays(*args, object_id="ZTF_A", seq_len=50)
        b, _ = prepare_lightcurve_arrays(*args, object_id="ZTF_B", seq_len=50)
        assert not np.array_equal(a["time"], b["time"])

    def test_selected_points_stay_in_time_order(self):
        """Random selection returns points in arbitrary order; ATCAT is a causal model over time,
        so the sequence handed to it must still be chronological."""
        n = 200
        arrays, _ = prepare_lightcurve_arrays(
            [float(i) for i in range(n)], [100.0] * n, [1.0] * n, [1] * n, [True] * n,
            object_id="ZTF9", seq_len=40,
        )
        t = arrays["time"][:40]
        assert np.all(np.diff(t) > 0)

    def test_downsampling_preserves_band_composition(self):
        """Measured on the five real over-length objects: random selection holds g/r composition to
        within a percentage point. A fixed stride could alias on a regular pattern; this cannot."""
        n = 600
        band = ([1] * 2 + [2] * 4) * (n // 6)  # deliberately blocky, 1/3 g and 2/3 r
        arrays, _ = prepare_lightcurve_arrays(
            [float(i) for i in range(n)], [100.0] * n, [1.0] * n, band, [True] * n,
            object_id="ZTF9", seq_len=243,
        )
        kept = arrays["channel_index"][:243]
        frac_g = float((kept == 1).sum()) / 243
        assert abs(frac_g - 1 / 3) < 0.05

    def test_no_accepted_photometry_raises(self):
        with pytest.raises(ValueError, match="no accepted photometry"):
            prepare_lightcurve_arrays([0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1, 1], [False, False])

    def test_ragged_inputs_raise(self):
        with pytest.raises(ValueError, match="disagree on length"):
            prepare_lightcurve_arrays([0.0, 1.0], [1.0], [1.0, 1.0], [1, 1], [True, True])

    def test_no_detection_keeps_all_accepted_points(self):
        """Nothing clears the S/N threshold — better to keep every accepted point than to invent a
        window."""
        arrays, info = prepare_lightcurve_arrays(
            [0.0, 500.0], [1.0, 1.0], [1.0, 1.0], [1, 2], [True, True], seq_len=5
        )
        assert info["n_in_window"] == 2
