# diag/

Throwaway-looking scripts that are worth keeping: each one is the evidence behind a claim in
`../DIAGNOSIS.md`, and several are the checks that would have caught these bugs years earlier.

## Keep running these

| script | what it answers |
|---|---|
| `prefix_health.py <ckpt-dir>` | Is the fusion stack's prefix a near-constant? Reports `‖dev‖/‖mean‖`, prefix norm vs the LLM's embedding norm, and mean pairwise cosine between objects, with spectra split by survey. **Relative instrument, not a pass/fail gate** — see DIAGNOSIS.md; it still reports FAIL on a model whose captions are 100% distinct. |
| `cache_health.py [cache-root]` | Did the encoders emit distinguishable embeddings? Flags total collapse, and flags `centred-cos mean > 0.15` as SUSPECT — the signal that separated the broken spectra cache (+0.244) from the healthy ones (-0.015, +0.022). |
| `caption_diversity.py <ckpt-dir> [lora-dir]` | **The gate that matters.** Generates captions for real test objects and reports distinct fraction + top-1 share, spectra split by DESI/SDSS. |
| `spectra_probe.py [cache-root]` | Linear/MLP class probe on the spectra cache. 0.347 before the wavelength fix, 0.632 after (majority 0.334). |

## One-off investigations, kept for reproducibility

| script | finding |
|---|---|
| `aion_tokens.py` | AION's spectrum tokenizer collapsed: a quasar and an early-type galaxy shared 99.6% of their 273 codes. |
| `fix_test.py` | The before/after that proved it: trimming `wavelength <= 0` takes identical-token fraction 0.965 → 0.098 and the class probe 0.356 → 0.660. |
| `spec_padding_test.py` | Showed the shard-padding bug was real but minor (3.4% on the embedding) — not the root cause. |
| `capacity_gpu.py`, `cap_lc.py` | The Q-Former is not capacity-limited: same architecture reaches 0.770 on lightcurve SN typing (majority 0.590) with `‖dev‖/‖mean‖` = 1.28. |
| `image_audit.py` | Image tier is healthy: AION predicts pixel-derived properties at R² 0.60–0.99. |
| `aion_control.py` | Readout sweep over AION spectra embeddings. The redshift regression in it is ill-conditioned (768 features, 700 samples) — don't read the R² numbers. |

`spec_labels.json` is keyword-derived class pseudo-labels (quasar / starforming / earlytype /
agn) built from the Gemini spectra captions. Crude, but they carry real signal: nine hand-computed
continuum medians score 0.61 against them where the broken AION embedding scored below its own
majority baseline.

`*.sbatch` are the Slurm wrappers used to run the GPU-bound ones on ghx4-interactive.
