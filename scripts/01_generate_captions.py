#!/usr/bin/env python
"""Image-only branch (train-v2-images): one caption per image object from
`gapatron/astrobridge-image-captions`' `caption_fused` field, plus the leakage-validation gate.

With a single modality there is no joint tier and no cross-modal decomposition — the
`mention_summary`/Gemini/transient caption sources and `decompose_object` are all gone with
their datasets. `caption_fused` is a pre-vetted, image-grounded caption (Gemini's merge of a
blind pass, a properties pass, and a literature pass, with the dataset's own leak detection),
used directly.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from captioner.data.captions import Caption, Claim, compose_captions
from captioner.data.image_dataset import load_image_captions_table
from captioner.eval.claims import claim_kind_histogram, validate_all
from captioner.utils.config import load_config
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config("base", "data")
    manifest = pd.read_parquet(cfg.manifest.parquet)

    image_captions_df = load_image_captions_table(
        cfg.sources.image_captions.hf_path, revision=cfg.sources.image_captions.get("revision")
    )
    image_caption_by_object = image_captions_df.set_index("object_id")["caption"].to_dict()

    all_captions: list[Caption] = []
    n_image_available = 0
    n_image_matched = 0

    for _, row in manifest.iterrows():
        object_id = row["object_id"]
        if not bool(row.get("has_image", False)):
            continue
        n_image_available += 1

        # manifest's canonical object_id is AstroBridge-Data's id (target_object_id_target); the
        # caption parquet is keyed by the Legacy Survey's own naming (object_id_legacy).
        image_lookup_id = row.get("object_id_legacy") or object_id
        fused = image_caption_by_object.get(image_lookup_id)
        if not fused:
            continue
        n_image_matched += 1

        claim = Claim(
            text=fused,
            supporting=frozenset({"image"}),
            kind="observation",
            provenance=f"gapatron:{object_id}",
        )
        all_captions.extend(compose_captions(object_id, [claim], frozenset({"image"})))

    violations = validate_all(all_captions)
    kind_hist = claim_kind_histogram(all_captions)

    match_rate = n_image_matched / n_image_available if n_image_available else None
    if match_rate is not None and match_rate < 0.5:
        logger.warning(
            f"Only {match_rate:.1%} of image objects ({n_image_matched}/{n_image_available}) "
            "matched a caption_fused row. This is the untested assumption that "
            "legacy_south_all_images.parquet's object_id_legacy shares an id namespace with "
            "gapatron/astrobridge-image-captions' object_id column — verify before trusting "
            "caption coverage."
        )

    report = {
        "n_objects_processed": len({c.object_id for c in all_captions}),
        "n_captions": len(all_captions),
        "n_leakage_violations": len(violations),
        "claim_kind_histogram": kind_hist,
        "n_image_available": n_image_available,
        "n_image_captions_matched": n_image_matched,
        "image_caption_match_rate": match_rate,
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

    # Sample 50 captions for the required manual read (§9 step 3 gate).
    rng = random.Random(0)
    sample = rng.sample(all_captions, min(50, len(all_captions)))
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
