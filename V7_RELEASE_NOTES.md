<!-- Local only. Deliberately NOT shipped in the HF model card. -->

**v7 fixes a bug that made every previous version's spectra captions meaningless.**

AstroBridge-Data pads `spectrum.lambda` with a `-1` sentinel on ~94% of SDSS rows, so the
wavelength grid is not monotonic. AION's spectrum codec interpolates onto its latent grid with
`torch.searchsorted`, which requires a sorted array and returns undefined results otherwise —
without raising. The result was a collapsed tokenizer: a quasar and an early-type galaxy shared
**99.6% of their 273 AION token codes**, and a linear class probe on the cached embeddings scored
*below its own majority baseline*. Every spectrum reaching the LLM was effectively the same
vector.

Downstream, v5 and v6 emitted one caption per modality. v6 published 82 unique captions out of
397, with a single caption covering 40.1% of the test set.

### What changed

- **Spectra are trimmed to a strictly increasing wavelength grid** before AION sees them, and the
  encoder now hard-errors on a non-monotonic grid instead of silently mis-encoding it.
  Class probe on the cached spectra embeddings: **0.347 → 0.632** (majority 0.334).
- **Splits are stratified by survey.** The source's own split labels put 331 of 334 DESI spectra
  in test, leaving **2 DESI spectra in training** while the spectra test set was 72% DESI. Train
  DESI objects: **2 → 233**. Ratio also moved from 80/10/10 to 70/20/10.
- **Prefix vectors are scaled to the LLM's embedding space.** They were norm ~30.6 against
  Qwen3.5-9B's own ~0.89 — 34x out of distribution, with no gradient pressure to correct it
  because RMSNorm divides the magnitude out. Now ~1.03.
- **Effective batch size now means what the config says.** `grad_accum` ignored `num_processes`,
  so an 8-rank run trained at batch 256 rather than 32 — 17 optimizer steps per epoch, 510 for
  all of stage 1. Now ~119 per epoch.
- Image captions re-pinned to `gapatron/astrobridge-image-captions@b8967f45` (all 3,487 rewritten
  upstream; pixel data unchanged).

### Result

Caption diversity on the test split, versus v6:

| tier | distinct fraction | top-1 share |
|---|---|---|
| image | 1.000 | 0.017 |
| lightcurve | 1.000 | 0.017 |
| spectra (SDSS) | 1.000 | 0.017 |
| spectra (DESI) | 1.000 | 0.053 |
| **v6, all tiers** | **0.207** | **0.401** |

Every generated caption is distinct.

### Known limitations — read before comparing against v4/v5/v6

- **This version's test split is not comparable object-for-object with v4/v5/v6.** The split
  policy, the train/val/test ratio, and the image caption source all changed.
- **The external caption-eval harness has pre-existing test leakage, in every version including
  this one.** Training reads `desi_sdss_subset_crossmatch_nolan_1.0arcsec.parquet` while that
  harness reads `desi_sdss_crossmatch_nolan_1.0arcsec.parquet`, and the two files disagree on the
  `split` label for 25.3% of shared objects. ~68% of the harness's test set is in v7's training
  split; it was ~66.5% for v6. Cross-version comparisons remain meaningful; absolute numbers from
  that harness are optimistic for all versions.
- Stage 2 early-stopped at epoch 5 on ~3.8k training examples, with train loss 0.66 against val
  1.18 — this model overfits quickly and is small-data limited.
- Joint (image+spectra) training is still disabled; `configs/modalities.yaml` weights it 0.
