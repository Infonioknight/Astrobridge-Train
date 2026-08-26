#!/usr/bin/env python
"""Decomposes literature-derived text into per-tier captions and runs the leakage-validation
CI gate (§4, §9 step 3). Gate: zero leakage violations, plus claim-survival rate reported, plus
a sample written out for the required manual read of 50 captions spanning both tiers.

Two caption sources, kept deliberately separate:
  - spectra (+ relational/joint claims): decomposed from AstroBridge-Data's `mention_summary`,
    same as always.
  - image: gapatron's own `caption_blind` field (data/image_dataset.py) — generated without
    literature context, i.e. grounded only in the image, with the dataset's own leak-detection
    already checking for stage contamination. Used directly rather than re-derived via keyword
    tagging, and this is also what makes image-only objects (no AstroBridge row at all) get a
    caption now — they used to be silently dropped here, since the old version of this script
    only ever looked at AstroBridge-Data's table.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from captioner.data.captions import (
    Caption,
    Claim,
    compose_captions,
    decompose_object,
    claim_survival_rate,
    _split_sentences,
)
from captioner.data.image_dataset import load_image_captions_table
from captioner.eval.claims import claim_kind_histogram, validate_all
from captioner.utils.config import load_config
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config("base", "data")
    manifest = pd.read_parquet(cfg.manifest.parquet)

    # AstroBridge-Data carries the literature-derived fields directly.
    from captioner.data.spectra_dataset import load_spectra_table

    spectra_ds = load_spectra_table(cfg.sources.spectra.hf_path, revision=cfg.sources.spectra.get("revision"))
    text_by_object = spectra_ds.set_index("object_id")[
        ["mention_summary", "evidence_quotes", "arxiv_id"]
    ].to_dict(orient="index")

    image_captions_df = load_image_captions_table(
        cfg.sources.image.hf_path, revision=cfg.sources.image.get("revision")
    )
    image_caption_by_object = image_captions_df.set_index("object_id")["caption_blind"].to_dict()

    all_captions: list[Caption] = []
    n_source_sentences = 0
    n_surviving = 0
    n_image_from_gapatron = 0
    n_image_available = 0
    n_no_usable_text = 0

    for _, row in manifest.iterrows():
        object_id = row["object_id"]
        available = frozenset(
            m for m in ("image", "spectra") if bool(row.get(f"has_{m}", False))
        )
        if not available:
            continue

        text_row = text_by_object.get(object_id)
        claims: list[Claim] = []

        if text_row is not None:
            claims = decompose_object(
                object_id=object_id,
                mention_summary=text_row.get("mention_summary") or "",
                evidence_quotes=text_row.get("evidence_quotes"),
                arxiv_id=text_row.get("arxiv_id"),
                available_modalities=available,
                generator=cfg.captions.generator,
            )
            n_source_sentences += len(_split_sentences(text_row.get("mention_summary") or ""))
            n_surviving += len(claims)

            # Drop any mention_summary-derived pure-image claims — gapatron's caption_blind
            # (below) is the preferred, pre-vetted source for the image tier, so prefer it and
            # avoid tagging the same content twice from two different heuristics. Spectra and
            # relational/joint claims from decomposition are kept as-is.
            claims = [c for c in claims if c.supporting != frozenset({"image"})]

        if "image" in available:
            # manifest's canonical object_id is AstroBridge-Data's id (from
            # target_object_id_target); the caption JSON files are keyed by the Legacy Survey's
            # own naming (object_id_legacy) — different namespace, so look up by that instead.
            n_image_available += 1
            image_lookup_id = row.get("object_id_legacy") or object_id
            blind = image_caption_by_object.get(image_lookup_id)
            if blind:
                claims.append(
                    Claim(
                        text=blind,
                        supporting=frozenset({"image"}),
                        kind="observation",
                        provenance=f"gapatron:{object_id}",
                    )
                )
                n_image_from_gapatron += 1

        if not claims:
            n_no_usable_text += 1
            continue

        captions = compose_captions(object_id, claims, available)
        all_captions.extend(captions)

    violations = validate_all(all_captions)
    survival_rate = claim_survival_rate(n_source_sentences, n_surviving)
    kind_hist = claim_kind_histogram(all_captions)

    image_caption_match_rate = n_image_from_gapatron / n_image_available if n_image_available else None
    if image_caption_match_rate is not None and image_caption_match_rate < 0.5:
        logger.warning(
            f"Only {image_caption_match_rate:.1%} of image-available objects "
            f"({n_image_from_gapatron}/{n_image_available}) found a caption_blind match. This is "
            "the untested assumption that legacy_south_all_images.parquet's object_id_legacy "
            "shares an id namespace with the caption JSON files' own object_id field — it may "
            "not hold. Check a few manifest rows' object_id_legacy against real "
            "*_captions.json filenames before trusting image-tier caption coverage."
        )

    report = {
        "n_objects_processed": len({c.object_id for c in all_captions}),
        "n_captions": len(all_captions),
        "n_leakage_violations": len(violations),
        "claim_survival_rate": survival_rate,
        "claim_kind_histogram": kind_hist,
        "n_image_captions_from_gapatron_blind": n_image_from_gapatron,
        "n_image_available": n_image_available,
        "image_caption_match_rate": image_caption_match_rate,
        "n_objects_with_no_usable_text": n_no_usable_text,
        "generator": cfg.captions.generator,
    }

    out_dir = Path(cfg.captions.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions_df = pd.DataFrame(
        [
            {
                "object_id": c.object_id,
                "subset": sorted(c.subset),
                "text": c.text,
                "n_claims": len(c.claims),
                "source": c.source,
                "generator": c.generator,
            }
            for c in all_captions
        ]
    )
    captions_df.to_parquet(cfg.captions.parquet, index=False)
    Path(cfg.captions.report).write_text(json.dumps(report, indent=2))

    # Sample 50 captions spanning both tiers for the required manual read (§9 step 3 gate).
    rng = random.Random(0)
    single = [c for c in all_captions if len(c.subset) == 1]
    joint = [c for c in all_captions if len(c.subset) > 1]
    sample = rng.sample(single, min(25, len(single))) + rng.sample(joint, min(25, len(joint)))
    sample_path = out_dir / "manual_review_sample.jsonl"
    with open(sample_path, "w") as f:
        for c in sample:
            f.write(json.dumps({"object_id": c.object_id, "subset": sorted(c.subset), "text": c.text}) + "\n")

    logger.info(f"Captions: {report}")
    logger.info(f"Manual review sample written to {sample_path}")

    if violations:
        logger.error(f"{len(violations)} leakage violations found — gate failed. Examples:")
        for v in violations[:10]:
            logger.error(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
