"""J-name coordinate parsing for the image dataset (data/image_dataset.py) — the one piece of
real astrometry math in the data pipeline, worth testing directly since a sign error here would
silently put every southern-hemisphere object in the wrong place for the ra/dec crossmatch.
"""
from __future__ import annotations

import pytest

from captioner.data.image_dataset import _best_jname, parse_jname


def test_parses_truncated_jname_negative_dec():
    ra, dec = parse_jname("J0000-0541")
    assert ra == pytest.approx(0.0)
    assert dec == pytest.approx(-5.683333, abs=1e-5)


def test_parses_truncated_jname_positive_dec():
    ra, dec = parse_jname("J1200+3000")
    assert ra == pytest.approx(180.0, abs=1e-5)
    assert dec == pytest.approx(30.0, abs=1e-5)


def test_parses_full_precision_jname_with_seconds():
    ra, dec = parse_jname("J000012.3-054130.2")
    assert ra == pytest.approx(0.0512500, abs=1e-5)
    assert dec == pytest.approx(-5.6917222, abs=1e-5)


def test_rejects_unparseable_string():
    with pytest.raises(ValueError):
        parse_jname("not-a-jname")


def test_best_jname_prefers_signed_variant():
    # "J0000 0541" is the sign-less display variant seen alongside the real one in real data —
    # picking it by accident would silently flip southern objects to the northern hemisphere.
    assert _best_jname(["J0000 0541", "J0000-0541"]) == "J0000-0541"
    assert _best_jname(["J0000-0541", "J0000 0541"]) == "J0000-0541"


def test_best_jname_falls_back_when_nothing_signed():
    assert _best_jname(["J0000 0541"]) == "J0000 0541"


def test_best_jname_empty_list():
    assert _best_jname([]) is None
