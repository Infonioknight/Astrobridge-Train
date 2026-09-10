"""Builds manifest.parquet: one row per object, with modality-availability flags, tier, and
split — plus manifest_stats.json.

IMAGE-ONLY BRANCH (train-v2-images). One modality, so there is no cross-modal join, no joint
tier, and no upstream split column (neither image source carries one) — every object is
`has_image=True`, `tier="single"`, and val/test are a seeded stratified draw over all objects.
The multi-modality join / crossmatch / transient-append machinery was removed with the other
modalities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_UPSTREAM_SPLIT_COLUMN = "split_upstream"


def _load_image_table(cfg: DictConfig) -> pd.DataFrame:
    """Identity + scalar metadata for every image object, from `legacy_south_all_images.parquet`
    (see data/image_dataset.py). `object_id` is AstroBridge-Data's own id; `object_id_legacy` is
    the Legacy Survey side's naming, used to look captions up in scripts/01_generate_captions.py.
    """
    from captioner.data.image_dataset import load_image_flux_identity_table

    df = load_image_flux_identity_table(
        cfg.sources.image.hf_path,
        revision=cfg.sources.image.get("revision"),
    )
    df["has_image"] = True
    return df


def _assert_unique_object_id(df: pd.DataFrame, label: str) -> None:
    if "object_id" not in df.columns:
        return
    dupes = df["object_id"][df["object_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{label} has {dupes.nunique()} duplicated object_id values ({len(dupes)} rows) — "
            "the loader should have deduplicated before returning."
        )


def build_manifest(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    image_df = _load_image_table(cfg)
    _assert_unique_object_id(image_df, "image table")

    manifest = image_df.copy()
    if "object_id" not in manifest.columns:
        manifest["object_id"] = manifest.index.astype(str)
    manifest["object_id"] = manifest["object_id"].astype(str)
    manifest["has_image"] = manifest["has_image"].fillna(False).astype(bool)
    manifest["tier"] = "single"  # one modality, never joint

    # Drop the heavy raw-pixel column if the identity loader carried it — 02_cache_embeddings.py
    # reloads pixels from the source, so it would just duplicate the dataset on disk.
    heavy = [c for c in ("image", "image_legacy", "rgb_legacy") if c in manifest.columns]
    if heavy:
        logger.info(f"Dropping heavy array columns from manifest.parquet: {heavy}")

    leading = ["object_id", "has_image", "tier"]
    manifest = manifest[
        leading + [c for c in manifest.columns if c not in leading and c not in heavy]
    ].copy()

    stats = {
        "n_total_objects": len(manifest),
        "modalities": ["image"],
        "n_image_only": len(manifest),
        "n_joint": 0,
        "tier_histogram": {"single": len(manifest)},
    }
    return manifest, stats


def _draw_val_test(manifest: pd.DataFrame, frac_val: float, frac_test: float, rng) -> tuple[set, set]:
    ids = manifest["object_id"].to_numpy().copy()
    rng.shuffle(ids)
    n_val = int(round(len(ids) * float(frac_val)))
    n_test = int(round(len(ids) * float(frac_test)))
    return set(ids[:n_val]), set(ids[n_val : n_val + n_test])


def assign_splits(manifest: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Seeded 80/10/10 over all objects. No upstream split column exists for the image sources,
    and there is only one tier, so `cfg.splits.policy` is moot — kept in config for schema
    stability but not consulted here.
    """
    rng = np.random.default_rng(int(cfg.splits.seed))
    manifest = manifest.copy()
    manifest["split"] = "train"

    val_ids, test_ids = _draw_val_test(manifest, cfg.splits.val, cfg.splits.test, rng)
    manifest.loc[manifest["object_id"].isin(val_ids), "split"] = "val"
    manifest.loc[manifest["object_id"].isin(test_ids), "split"] = "test"

    assert manifest.groupby("object_id")["split"].nunique().max() <= 1, (
        "An object_id was assigned to more than one split — correctness bug."
    )
    return manifest


def write_manifest(cfg: DictConfig) -> None:
    manifest, stats = build_manifest(cfg)
    manifest = assign_splits(manifest, cfg)

    split_hist = manifest["split"].value_counts().to_dict()
    stats["split_histogram"] = {k: int(v) for k, v in split_hist.items()}
    stats["split_source"] = "seeded_draw"

    for required in ("train", "test"):
        if int(split_hist.get(required, 0)) == 0:
            raise ValueError(
                f"Split {required!r} is empty ({split_hist}). The image table produced too few "
                f"objects for a {cfg.splits.val}/{cfg.splits.test} draw, or something upstream "
                "returned nothing."
            )

    for subset_name, count in stats["tier_histogram"].items():
        if count < cfg.sanity.min_per_subset:
            logger.warning(
                f"Tier {subset_name!r} has only {count} objects (< min_per_subset={cfg.sanity.min_per_subset})."
            )

    out_dir = Path(cfg.manifest.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(cfg.manifest.parquet, index=False)
    Path(cfg.manifest.stats).write_text(json.dumps(stats, indent=2))

    logger.info(f"Wrote manifest with {len(manifest)} rows to {cfg.manifest.parquet}")
    logger.info(f"Tier histogram: {stats['tier_histogram']}")
    logger.info(f"Split histogram: {stats['split_histogram']} (source: {stats['split_source']})")
