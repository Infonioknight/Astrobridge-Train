"""Loader for `gapatron/legacy_survey_south_images_captions` against its *real* shape, confirmed
by inspecting the repo and one caption JSON directly (see conversation history / commit that
introduced this file) — not the `datasets.load_dataset`-with-an-`image`-column shape the rest of
this codebase originally assumed.

What's actually there: raw file pairs, `{object_id}_rgb.png` + `{object_id}_captions.json`,
~4,821 of them. No `ra`/`dec` (or any coordinate) columns anywhere — only a J2000 sexagesimal
name buried in `identity_strings_shown_to_model` (e.g. "J0000-0541"), which this module parses.

Two separable concerns:
  - Caption text + coordinates: resolved here, need nothing but the JSON files (~90MB total for
    ~4.8k of them) — `caption_blind` is generated without literature/external context, i.e.
    grounded only in what the image itself shows, with the dataset's own leak-detection
    (`leak_flags_per_stage`/`any_leaks_detected`) already checking for stage contamination. This
    makes it a direct, pre-vetted source for the image tier — see 01_generate_captions.py, which
    uses it in place of running our own keyword-based decomposition for that tier.
  - Pixel data: only available here as a rendered RGB PNG, not raw per-band calibrated flux —
    AION's LegacySurveyImage wants the latter (see aion_image.py). That gap is NOT resolved in
    this module; pixel loading for 02_cache_embeddings.py stays blocked until real grz/griz flux
    cutouts are available (see README.md's "Known blocker" note). Nothing here assumes otherwise.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

_JNAME_RE = re.compile(r"^J(\d{4}|\d{6}(?:\.\d+)?)([+-])(\d{4}|\d{6}(?:\.\d+)?)$")


def parse_jname(jname: str) -> tuple[float, float]:
    """Parses a truncated-or-full IAU J2000 sexagesimal name — "J0000-0541" or
    "J000012.3-054130.2" — into (ra_deg, dec_deg). Raises ValueError rather than guessing on
    anything that doesn't match the expected pattern.
    """
    m = _JNAME_RE.match(jname.strip())
    if not m:
        raise ValueError(f"Could not parse J-name coordinate string: {jname!r}")
    ra_part, sign, dec_part = m.groups()

    ra_int_len = len(ra_part.split(".")[0])
    if ra_int_len == 4:
        hh, mm, ss = int(ra_part[0:2]), int(ra_part[2:4]), 0.0
    else:
        hh, mm, ss = int(ra_part[0:2]), int(ra_part[2:4]), float(ra_part[4:])
    ra_deg = 15.0 * (hh + mm / 60 + ss / 3600)

    dec_int_len = len(dec_part.split(".")[0])
    if dec_int_len == 4:
        dd, dm, ds = int(dec_part[0:2]), int(dec_part[2:4]), 0.0
    else:
        dd, dm, ds = int(dec_part[0:2]), int(dec_part[2:4]), float(dec_part[4:])
    dec_deg = (1.0 if sign == "+" else -1.0) * (dd + dm / 60 + ds / 3600)

    return ra_deg, dec_deg


def _best_jname(identity_strings: list[str]) -> str | None:
    """`identity_strings_shown_to_model` sometimes carries a sign-less display variant alongside
    the properly-signed one (e.g. "J0000 0541" next to "J0000-0541" when dec is *negative* —
    the space-separated form silently drops the sign). Always prefer an entry with an explicit
    +/- sign; parsing the sign-less one would silently produce the wrong hemisphere for southern
    objects, which is most of them in a "south" survey.
    """
    signed = [s for s in identity_strings if "+" in s or "-" in s]
    return signed[0] if signed else (identity_strings[0] if identity_strings else None)


def download_caption_jsons(hf_path: str, revision: str | None = None, cache_dir: Path | None = None) -> Path:
    """Pulls only the *_captions.json files, not the PNGs — caption text and coordinates don't
    need pixel data, and this keeps the download to ~90MB instead of ~250MB.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=hf_path,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["*_captions.json"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(local_dir)


def load_image_captions_table(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One row per object: object_id, ra, dec, caption_blind/_properties/_literature, and the
    structured photometry fields gapatron's authors held out of the blind stage (mag_G/R/I/Z,
    effective_radius_arcsec, ellipticity, position_angle_deg) — usable for claim-tagging the same
    way AstroBridge's structured fields are (§4).
    """
    local_dir = download_caption_jsons(hf_path, revision, cache_dir)
    json_paths = sorted(local_dir.rglob("*_captions.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"No *_captions.json files found under {local_dir} — check hf_path={hf_path!r} and "
            "that gated access has actually been granted."
        )

    rows = []
    n_unparseable_coords = 0
    for p in json_paths:
        with open(p) as fh:
            rec = json.load(fh)

        jname = _best_jname(rec.get("identity_strings_shown_to_model", []) or [])
        ra = dec = None
        if jname:
            try:
                ra, dec = parse_jname(jname)
            except ValueError:
                n_unparseable_coords += 1
        else:
            n_unparseable_coords += 1

        photom = rec.get("photometry_reference_not_shown_to_model", {}) or {}
        mag = photom.get("mag", {}) or {}

        rows.append(
            {
                "object_id": rec.get("object_id"),
                "ra": ra,
                "dec": dec,
                "caption_blind": rec.get("caption_blind"),
                "caption_properties": rec.get("caption_properties"),
                "caption_literature": rec.get("caption_literature"),
                "mag_G": mag.get("G"),
                "mag_R": mag.get("R"),
                "mag_I": mag.get("I"),
                "mag_Z": mag.get("Z"),
                "effective_radius_arcsec": photom.get("effective_radius_arcsec"),
                "ellipticity": photom.get("ellipticity"),
                "position_angle_deg": photom.get("position_angle_deg"),
            }
        )

    df = pd.DataFrame(rows)
    if n_unparseable_coords:
        logger.warning(
            f"{n_unparseable_coords}/{len(df)} image-dataset objects had no parseable J-name "
            "coordinate — they'll be dropped before the ra/dec crossmatch in manifest.py."
        )
    df = df.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    df["has_image"] = True
    return df
