"""AtcatLightcurveEncoder's contract enforcement.

The ONNX session is faked, so this exercises the wrapper's own guarantees — token-level output, no
pooling, static 243 sequence length, dtype casting against the export's declared types, and the
all-zero-mask refusal — without needing onnxruntime or the 12MB model. Requires torch only because
the encoder returns a torch Tensor, matching the AION wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from captioner.encoders.atcat_lightcurve import (
    ATCAT_INPUTS,
    ATCAT_OUT_DIM,
    ATCAT_SEQ_LEN,
    AtcatLightcurveEncoder,
)
from captioner.encoders.base import EncoderLoadError


class _FakeInput:
    def __init__(self, name, type_):
        self.name, self.type = name, type_


class _FakeSession:
    """Mimics onnxruntime's declared-dtype behaviour: it records what it was fed so the test can
    assert the wrapper cast to the export's types rather than assuming float32."""

    def __init__(self, dtypes=None, out_dim=ATCAT_OUT_DIM):
        self.dtypes = dtypes or {
            "flux": "tensor(float)",
            "flux_err": "tensor(float)",
            "time": "tensor(float)",
            "mask": "tensor(int64)",
            "channel_index": "tensor(int64)",
        }
        self.out_dim = out_dim
        self.seen = None
        self.requested = None

    def get_inputs(self):
        return [_FakeInput(n, t) for n, t in self.dtypes.items()]

    def get_outputs(self):
        return [_FakeInput(n, "tensor(float)") for n in ("last", "mean", "sequence")]

    def run(self, names, feed):
        self.requested, self.seen = names, feed
        b = feed["flux"].shape[0]
        return [np.zeros((b, ATCAT_SEQ_LEN, self.out_dim), dtype=np.float32)]


def _encoder(session=None, **kwargs):
    enc = AtcatLightcurveEncoder(
        name="lightcurve", hf_path="light-curve/atcat", revision=None,
        kwargs={"onnx_file": "atcat_f32.onnx", "output": "sequence", **kwargs},
        num_encoder_tokens=ATCAT_SEQ_LEN,
    )
    session = session or _FakeSession()
    enc._session = session
    enc._input_dtypes = {i.name: i.type for i in session.get_inputs()}
    return enc, session


def _batch(b=2, valid=5):
    arrays = {k: np.zeros((b, ATCAT_SEQ_LEN), dtype=np.float64) for k in ATCAT_INPUTS}
    arrays["mask"][:, :valid] = 1
    arrays["channel_index"][:, :valid] = 1
    return arrays


class TestConstruction:
    def test_pooled_outputs_are_rejected(self):
        """`last` and `mean` are (B, 384) pooled vectors. The Q-Former exists to cross-attend over
        structure pooling destroys, so accepting them would silently break the encoder contract."""
        for pooled in ("last", "mean"):
            with pytest.raises(ValueError, match="pooled"):
                AtcatLightcurveEncoder("lightcurve", "light-curve/atcat", None,
                                       {"output": pooled}, ATCAT_SEQ_LEN)

    def test_max_tokens_must_match_the_static_export_length(self):
        with pytest.raises(ValueError, match="static sequence length"):
            AtcatLightcurveEncoder("lightcurve", "light-curve/atcat", None, {}, 512)


class TestEncode:
    def test_returns_token_level_output(self):
        enc, _ = _encoder()
        emb = enc.encode(_batch(b=3))
        assert emb.dim() == 3
        assert emb.shape == (3, ATCAT_SEQ_LEN, ATCAT_OUT_DIM)

    def test_requests_only_the_sequence_output(self):
        enc, session = _encoder()
        enc.encode(_batch())
        assert session.requested == ["sequence"], "onnxruntime prunes unrequested heads"

    def test_casts_inputs_to_the_declared_onnx_dtypes(self):
        enc, session = _encoder()
        enc.encode(_batch())
        assert session.seen["flux"].dtype == np.float32
        assert session.seen["mask"].dtype == np.int64

    def test_honours_a_different_declared_dtype(self):
        """Guards the assumption that mask is int64: the export could equally declare it float."""
        session = _FakeSession(dtypes={
            "flux": "tensor(double)", "flux_err": "tensor(double)", "time": "tensor(double)",
            "mask": "tensor(bool)", "channel_index": "tensor(int32)",
        })
        enc, session = _encoder(session=session)
        enc.encode(_batch())
        assert session.seen["flux"].dtype == np.float64
        assert session.seen["mask"].dtype == np.bool_
        assert session.seen["channel_index"].dtype == np.int32

    def test_accepts_torch_tensors(self):
        enc, _ = _encoder()
        batch = {k: torch.from_numpy(v) for k, v in _batch().items()}
        assert enc.encode(batch).shape[0] == 2

    def test_all_zero_mask_row_is_refused(self):
        """An empty sequence leaves the band_id=0 sentinel as the only content, which ATCAT reads
        as u-band. Such objects must be excluded upstream, never encoded."""
        enc, _ = _encoder()
        batch = _batch(b=2, valid=5)
        batch["mask"][1, :] = 0
        with pytest.raises(ValueError, match="all-zero"):
            enc.encode(batch)

    def test_missing_input_raises_with_the_field_list(self):
        enc, _ = _encoder()
        batch = _batch()
        del batch["channel_index"]
        with pytest.raises(ValueError, match="channel_index"):
            enc.encode(batch)

    def test_wrong_sequence_length_raises(self):
        enc, _ = _encoder()
        batch = {k: np.zeros((2, 100)) for k in ATCAT_INPUTS}
        batch["mask"][:, :5] = 1
        with pytest.raises(ValueError, match=f"expected \\(B, {ATCAT_SEQ_LEN}\\)"):
            enc.encode(batch)

    def test_encode_before_load_raises(self):
        enc = AtcatLightcurveEncoder("lightcurve", "light-curve/atcat", None, {}, ATCAT_SEQ_LEN)
        with pytest.raises(EncoderLoadError, match="before load"):
            enc.encode(_batch())

    def test_unexpected_out_dim_raises(self):
        enc, _ = _encoder(session=_FakeSession(out_dim=256))
        with pytest.raises(EncoderLoadError, match="out_dim"):
            enc.encode(_batch())
