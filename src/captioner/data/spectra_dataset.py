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
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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

    return pd.concat(frames, ignore_index=True, sort=False)
