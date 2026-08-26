"""Loader for `UniverseTBD/AstroBridge-Data`, working around a real schema-union failure in
`datasets.load_dataset(hf_path, split="train")` against this repo — confirmed against a real
run, not a guess:

    datasets.table.CastError: Couldn't cast ... because column names don't match

The repo's `observations/spectra/` folder holds four separate parquet files, one per crossmatch
source (DESI-only, DESI+SDSS, DESI+SDSS-subset, SDSS-only). `datasets`'s automatic multi-file
loading tries to cast every file into one strict Arrow schema inferred up front, and fails,
because the SDSS-crossmatched files carry extra `survey`/`survey_metadata` columns the DESI-only
file doesn't have. The core columns this codebase actually reads (`object_id`,
`ra_spectra`/`dec_spectra`, `spectrum`, `Z`/`ZERR`/`ZWARN`, `mention_summary`/`evidence_quotes`/
`arxiv_id`, etc.) are present in every file — only the SDSS-specific extras differ.

Loading each file separately with pandas and `pd.concat`ing sidesteps the cast entirely:
`pd.concat` fills a column missing from one file with NaN for those rows instead of requiring an
exact schema match across all files, which is exactly the flexibility `datasets` doesn't give us
here.

Separately: these files' `spectrum` column is a nested struct (flux/ivar/lsf_sigma/lambda/mask —
the same shape of thing as the image side's `image_legacy`, which hit a real crash reading via
plain `pd.read_parquet`: pyarrow's pandas-metadata-driven dtype restoration chokes on a nested
struct dtype string `numpy.dtype()` can't parse). Reading via `pyarrow.parquet` directly with
`ignore_metadata=True` avoids that class of failure — see data/image_dataset.py's
`_read_parquet_columns` for the full explanation; used here defensively for the same reason.

A third real issue, also confirmed against a real run: the four files are NOT disjoint by
`object_id` — a target can appear in more than one crossmatch-source file (e.g. `..._subset...`
looks like a literal subset of the non-subset file, and a target with both DESI and SDSS spectra
plausibly appears in more than one of the survey-specific files too). `pd.concat` alone leaves
duplicate `object_id` rows in the result, which breaks every downstream `set_index("object_id")`
lookup — one of them (`DataFrame.to_dict(orient="index")` in 01_generate_captions.py) raises
`ValueError: DataFrame index must be unique for orient='index'` outright; others (e.g. a plain
`Series.to_dict()`) would have silently kept whichever duplicate happened to sort last instead of
erroring, which is worse. `load_spectra_table` now deduplicates by `object_id` before returning —
preferring the row with the smallest `_dist_arcsec` (the crossmatch quality metric already in
the data) when duplicates disagree on it, since that's an objective tie-break rather than an
arbitrary one based on file processing order.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from captioner.utils.logging import get_logger

logger = get_logger(__name__)

SPECTRA_DIR_PREFIX = "observations/spectra/"


def _read_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas(ignore_metadata=True)


def load_spectra_table(
    hf_path: str, revision: str | None = None, cache_dir: Path | None = None
) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f
        for f in list_repo_files(hf_path, repo_type="dataset", revision=revision)
        if f.startswith(SPECTRA_DIR_PREFIX) and f.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {SPECTRA_DIR_PREFIX!r} in {hf_path!r} — check the "
            "repo layout hasn't changed."
        )

    frames = []
    for f in files:
        local_path = hf_hub_download(
            repo_id=hf_path,
            filename=f,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        frames.append(_read_parquet(local_path))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return _deduplicate_by_object_id(combined)


def _deduplicate_by_object_id(df: pd.DataFrame) -> pd.DataFrame:
    n_dupe_rows = int(df["object_id"].duplicated(keep=False).sum())
    if n_dupe_rows == 0:
        return df

    n_dupe_objects = df.loc[df["object_id"].duplicated(keep=False), "object_id"].nunique()
    if "_dist_arcsec" in df.columns:
        # Prefer the closest crossmatch when the same object appears in more than one source
        # file — an objective tie-break already present in the data, not an arbitrary one based
        # on which file happened to be listed/processed first.
        df = df.sort_values("_dist_arcsec", na_position="last")
        tie_break = "smallest _dist_arcsec"
    else:
        tie_break = "first occurrence (no _dist_arcsec column to break ties on)"

    deduped = df.drop_duplicates(subset="object_id", keep="first").reset_index(drop=True)
    logger.warning(
        f"{n_dupe_rows} rows across {n_dupe_objects} object_ids appeared in more than one "
        f"source parquet file (e.g. a target with both DESI and SDSS spectra, or a '_subset' "
        f"file overlapping its non-subset counterpart) — kept one row per object_id, tie-broken "
        f"by {tie_break}. This affects which mention_summary/spectrum version each duplicated "
        f"object ends up with; worth spot-checking a few of them if caption quality looks off "
        f"for objects that had duplicates."
    )
    return deduped
