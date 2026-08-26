"""Builds manifest.parquet: one row per object, with modality-availability flags, tier, and
split — plus manifest_stats.json (tier histogram, join diagnostics). No join/count is ever
hardcoded; everything here is a runtime fact (§1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def _load_spectra_table(cfg: DictConfig) -> pd.DataFrame:
    """Uses data/spectra_dataset.py, not `datasets.load_dataset` — the latter fails against the
    real repo's multiple, non-identically-schemaed parquet files (confirmed against a real run,
    see that module's docstring).
    """
    from captioner.data.spectra_dataset import load_spectra_table

    df = load_spectra_table(cfg.sources.spectra.hf_path, revision=cfg.sources.spectra.get("revision"))
    df = df.rename(columns={"ra_spectra": "ra", "dec_spectra": "dec"})
    df["has_spectra"] = True
    return df


def _load_image_table(cfg: DictConfig) -> pd.DataFrame:
    """Sources identity from `legacy_south_all_images.parquet`, not the caption-only RGB dataset
    (see data/image_dataset.py) — this table's rows are objects that already have real
    per-band calibrated flux (usable for AION), and its `object_id` is AstroBridge-Data's own id
    (a direct join key below — see build_manifest's "object_id" join path), not a coordinate
    match. The caption-only dataset is still used for `caption_blind` text in
    scripts/01_generate_captions.py, just not for identity here.
    """
    from captioner.data.image_dataset import load_image_flux_identity_table

    return load_image_flux_identity_table(
        cfg.sources.image.hf_path,
        revision=cfg.sources.image.get("revision"),
    )


def _crossmatch_coords(left: pd.DataFrame, right: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    left_coord = SkyCoord(ra=left["ra"].to_numpy() * u.deg, dec=left["dec"].to_numpy() * u.deg)
    right_coord = SkyCoord(ra=right["ra"].to_numpy() * u.deg, dec=right["dec"].to_numpy() * u.deg)
    idx, sep2d, _ = left_coord.match_to_catalog_sky(right_coord)
    within = sep2d.arcsec <= radius_arcsec

    matched = left.copy()
    matched["_match_idx"] = np.where(within, idx, -1)
    matched["_sep_arcsec"] = sep2d.arcsec
    return matched


def build_manifest(cfg: DictConfig) -> tuple[pd.DataFrame, dict]:
    spectra_df = _load_spectra_table(cfg)
    image_df = _load_image_table(cfg)

    join_key = cfg.join.key
    have_key_both = join_key in spectra_df.columns and join_key in image_df.columns

    if have_key_both:
        logger.info(f"Joining on {join_key}")
        merged_key = pd.merge(
            spectra_df, image_df, on=join_key, how="outer", suffixes=("_spec", "_img")
        )
        join_method = join_key
    elif "object_id" in image_df.columns:
        # legacy_south_all_images.parquet's object_id is target_object_id_target — already
        # AstroBridge-Data's own id (these rows are pre-crossmatched by the dataset authors) —
        # a direct, authoritative join, no coordinate fuzz-matching needed for this subset.
        overlap = image_df["object_id"].isin(spectra_df["object_id"]).mean()
        logger.info(f"Direct object_id join: {overlap:.1%} of image-table object_ids matched a spectra object_id")
        if overlap < 0.5:
            logger.warning(
                "Less than half of the image table's object_id values matched AstroBridge-Data's "
                "object_id — the assumption that legacy_south_all_images.parquet's "
                "target_object_id_target shares AstroBridge-Data's id namespace may not hold for "
                "this data revision. Verify before trusting the resulting joint tier."
            )
        merged_key = pd.merge(
            spectra_df, image_df, on="object_id", how="outer", suffixes=("_spec", "_img")
        )
        join_method = "object_id"
    else:
        logger.warning(
            f"{join_key} not present in both tables; falling back to coordinate crossmatch "
            f"at {cfg.join.fallback_radius_arcsec} arcsec"
        )
        matched = _crossmatch_coords(spectra_df, image_df, cfg.join.fallback_radius_arcsec)
        joint_mask = matched["_match_idx"] >= 0
        joint = matched[joint_mask].copy()
        joint_image_cols = image_df.iloc[joint["_match_idx"].to_numpy()].reset_index(drop=True)
        joint = joint.reset_index(drop=True)
        for c in joint_image_cols.columns:
            if c not in joint.columns:
                joint[c] = joint_image_cols[c]

        spec_only = matched[~joint_mask].copy()
        image_only_mask = ~image_df.index.isin(matched.loc[joint_mask, "_match_idx"])
        image_only = image_df[image_only_mask].copy()

        merged_key = pd.concat([joint, spec_only, image_only], ignore_index=True, sort=False)
        join_method = f"coord@{cfg.join.fallback_radius_arcsec}arcsec"

    merged_key["has_spectra"] = merged_key.get("has_spectra", False).fillna(False).astype(bool)
    merged_key["has_image"] = merged_key.get("has_image", False).fillna(False).astype(bool)

    def tier(row) -> str:
        if row["has_spectra"] and row["has_image"]:
            return "joint"
        return "single"

    merged_key["tier"] = merged_key.apply(tier, axis=1)

    if "object_id" not in merged_key.columns:
        merged_key["object_id"] = merged_key.index.astype(str)

    # Drop heavy nested/array columns before writing — the raw `spectrum` struct (flux/ivar/
    # mask/lambda, thousands of floats per object) and whatever the image dataset's pixel-cutout
    # field is called. Nothing downstream reads these back out of manifest.parquet:
    # 02_cache_embeddings.py reloads the raw HF datasets itself for encoding, and everything else
    # only needs scalars/metadata. Keeping them here would silently duplicate the full raw
    # dataset size into a second on-disk copy for no reason.
    heavy_columns = [c for c in ("spectrum", "image") if c in merged_key.columns]
    if heavy_columns:
        logger.info(f"Dropping heavy array columns from manifest.parquet: {heavy_columns}")

    manifest = merged_key[["object_id", "has_spectra", "has_image", "tier"] + [
        c for c in merged_key.columns
        if c not in ("object_id", "has_spectra", "has_image", "tier") and c not in heavy_columns
    ]].copy()

    stats = _compute_stats(manifest, join_method, cfg)
    return manifest, stats


def _compute_stats(manifest: pd.DataFrame, join_method: str, cfg: DictConfig) -> dict:
    tier_hist = manifest["tier"].value_counts().to_dict()
    n_joint = int(tier_hist.get("joint", 0))
    n_total = len(manifest)

    stats = {
        "join_method": join_method,
        "n_total_objects": n_total,
        "n_spectra_only": int(((manifest["has_spectra"]) & (~manifest["has_image"])).sum()),
        "n_image_only": int(((manifest["has_image"]) & (~manifest["has_spectra"])).sum()),
        "n_joint": n_joint,
        "tier_histogram": {k: int(v) for k, v in tier_hist.items()},
    }

    if n_joint < cfg.sanity.min_joint_objects:
        logger.warning(
            f"Only {n_joint} joint objects (< configured min_joint_objects="
            f"{cfg.sanity.min_joint_objects}). Val/test are drawn from joint objects only — "
            "eval sets may be very thin. Not failing the build; proceeding."
        )
    return stats


def assign_splits(manifest: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Object-level 80/10/10. Val/test drawn from joint objects only; unpaired objects go to
    train only. Same object_id never appears in two splits.
    """
    rng = np.random.default_rng(int(cfg.splits.seed))
    manifest = manifest.copy()
    manifest["split"] = "train"

    joint_ids = manifest.loc[manifest["tier"] == "joint", "object_id"].to_numpy()
    rng.shuffle(joint_ids)

    n_val = int(round(len(joint_ids) * cfg.splits.val))
    n_test = int(round(len(joint_ids) * cfg.splits.test))
    val_ids = set(joint_ids[:n_val])
    test_ids = set(joint_ids[n_val : n_val + n_test])

    manifest.loc[manifest["object_id"].isin(val_ids), "split"] = "val"
    manifest.loc[manifest["object_id"].isin(test_ids), "split"] = "test"

    assert manifest.groupby("object_id")["split"].nunique().max() <= 1, (
        "An object_id was assigned to more than one split — this is a correctness bug."
    )
    return manifest


def write_manifest(cfg: DictConfig) -> None:
    manifest, stats = build_manifest(cfg)
    manifest = assign_splits(manifest, cfg)

    split_hist = manifest["split"].value_counts().to_dict()
    stats["split_histogram"] = {k: int(v) for k, v in split_hist.items()}

    for subset_name, count in stats["tier_histogram"].items():
        if count < cfg.sanity.min_per_subset:
            logger.warning(
                f"Tier {subset_name!r} has only {count} objects "
                f"(< min_per_subset={cfg.sanity.min_per_subset})."
            )

    out_dir = Path(cfg.manifest.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(cfg.manifest.parquet, index=False)
    Path(cfg.manifest.stats).write_text(json.dumps(stats, indent=2))

    logger.info(f"Wrote manifest with {len(manifest)} rows to {cfg.manifest.parquet}")
    logger.info(f"Tier histogram: {stats['tier_histogram']}")
