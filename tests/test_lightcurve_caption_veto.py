"""The lightcurve tier's caption veto (§4).

Unlike gapatron's `caption_blind` and the Gemini spectra captions, the transients dataset's
`transient_caption` is NOT modality-blind: measured on 30 sampled rows, 30/30 name a catalog
designation, quote a spectroscopic redshift, or say "classified in literature as" — none of which a
light curve can support. Those sentences must be dropped, not reassigned, or the model learns to
emit designations and redshifts from photometry alone.

Torch-free on purpose, like the module under test.
"""
from __future__ import annotations

from captioner.data.captions import (
    LIGHTCURVE_UNSUPPORTABLE,
    compose_captions,
    decompose_object,
    validate_no_leakage,
)

LC = frozenset({"lightcurve"})


def _claims(text, available=LC):
    return decompose_object("ZTF0000000001", text, None, None, available)


class TestVeto:
    def test_designation_and_redshift_sentence_is_dropped(self):
        text = "SN 2018bti is classified in literature as a Type Ia supernova at a redshift of $z = 0.0248$."
        assert _claims(text) == []

    def test_photometry_sentence_survives(self):
        text = "The transient rises to a peak brightness of $g \\approx 18.07$ mag around MJD 58262.3."
        claims = _claims(text)
        assert len(claims) == 1
        assert claims[0].supporting == LC

    def test_mixed_caption_keeps_only_the_photometric_half(self):
        text = (
            "SN 2018bti is classified in literature as a Type Ia supernova at a redshift of $z = 0.0248$. "
            "In the sampled ZTF photometry, the transient rises to a peak brightness around MJD 58262.3."
        )
        claims = _claims(text)
        assert len(claims) == 1
        assert "rises to a peak" in claims[0].text
        assert "literature" not in claims[0].text

    def test_mjd_is_not_mistaken_for_a_designation(self):
        """A naive `at\\s?\\d{4}` designation pattern also matches "at 58346" — an MJD — and would
        silently veto legitimate photometry. Years are anchored to 19xx/20xx to prevent that."""
        text = "The light curve rose over 13 days from initial detection at 58346 to peak brightness."
        assert len(_claims(text)) == 1

    def test_survey_names_and_ztf_ids_are_vetoed(self):
        assert _claims("The transient was reported as ASASSN-20ed (AT 2020hvn) at MJD 58962.") == []
        assert _claims("SN 2018crl (ZTF18aaykjei) peaked near MJD 58293.") == []

    def test_spectroscopic_class_is_vetoed(self):
        assert _claims("The object is a Type Ia supernova whose brightness peaked on MJD 58279.") == []

    def test_non_photometric_sentence_is_dropped_for_lack_of_keywords(self):
        assert _claims("The host is a barred spiral seen at moderate inclination.") == []


class TestComposition:
    def test_produces_a_lightcurve_tier_caption_with_no_leakage(self):
        text = "The light curve rises to a peak around MJD 58262.3 and then fades over 40 days."
        claims = _claims(text)
        captions = compose_captions("ZTF0000000001", claims, LC)
        assert len(captions) == 1
        assert captions[0].subset == LC
        assert validate_no_leakage(captions[0]) == []

    def test_fully_vetoed_object_yields_no_caption(self):
        text = "SN 2018cdt is classified in literature as a Type Ia supernova at redshift $z = 0.05$."
        assert compose_captions("ZTF0000000001", _claims(text), LC) == []


class TestExistingBehaviourUnchanged:
    """The veto is scoped to the lightcurve tag. A sentence mentioning redshift is legitimately
    spectra-supported, so vetoing globally would silently change existing spectra decomposition."""

    def test_redshift_sentence_still_supports_spectra(self):
        text = "The spectrum shows a redshift of z = 0.05 from strong emission lines."
        claims = decompose_object("a", text, None, None, frozenset({"image", "spectra"}))
        assert len(claims) == 1
        assert claims[0].supporting == frozenset({"spectra"})

    def test_image_tagging_is_untouched_by_the_veto(self):
        text = "The morphology is an extended spiral disk classified in literature as barred."
        claims = decompose_object("a", text, None, None, frozenset({"image"}))
        assert len(claims) == 1
        assert claims[0].supporting == frozenset({"image"})

    def test_veto_regex_is_not_consulted_when_lightcurve_absent(self):
        assert LIGHTCURVE_UNSUPPORTABLE.search("classified in literature") is not None
        claims = decompose_object("a", "The spectrum shows emission line structure.", None, None,
                                  frozenset({"spectra"}))
        assert len(claims) == 1
