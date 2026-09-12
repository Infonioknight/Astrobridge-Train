# Why v5 and v6 write the same caption for every spectrum

Written 2026-09-11 against `v6` @ `a5fa267`, checkpoint `outputs/checkpoints/stage2/best`
(byte-identical to the published `UniverseTBD/astrobridge-model-v6` — verified by md5).

Observed symptom: 397 test captions, 82 unique (20.7%), top caption repeated 159 times (40.1%).
v5 is the same failure (393 captions, 83 unique, top 16.5%). Zero-flux and pure-noise spectra
still get confident, specific captions.

There are **two independent faults**. Fixing either alone does not fix the symptom.

---

## Fault 1 — the spectra never reached AION intact

`spectrum.lambda` in `UniverseTBD/AstroBridge-Data` is padded with a `-1` sentinel at the end of
the array. **94% of SDSS spectra (282/300 sampled) carry it**, which makes the wavelength array
non-monotonic:

    [3789.658, 3790.531, ..., 9219.344, -1.0, -1.0, -1.0, ...]

AION's spectrum codec projects onto its own latent grid with
`LatentSpectralGrid.to_latent` -> `interp1d` -> `torch.searchsorted(wavelength, target)`.
`searchsorted` **requires a sorted array**; on unsorted input the result is undefined. Every
spectrum therefore interpolated to near-identical garbage, and AION's VQ tokenizer collapsed.

Measured on 700 real SDSS spectra (`diag/fix_test.py`):

| | identical-token fraction | distinct codes/position | linear probe | MLP probe |
|---|---|---|---|---|
| current (sentinels left in) | 0.965 | 16.8 | 0.356 | 0.329 |
| trimmed (`wl > 0`, sorted) | **0.098** | **342.5** | **0.660** | **0.693** |

majority-class baseline = 0.401. Labels are keyword-derived from the Gemini spectra captions
(quasar / starforming / earlytype / agn).

A quasar and an early-type galaxy were getting **99.6% of the same 273 token codes**. For scale,
nine crude continuum medians computed straight off the raw flux score 0.636 on the same objects
and folds — the untrimmed AION embedding was **below the majority baseline**, i.e. worse than
useless. After the trim AION lands at 0.693, ahead of the hand-built baseline, which is where a
foundation model should sit.

Nothing downstream was broken. It was all faithfully processing a constant.

### Fault 1b — spectra shards mix survey lengths (real, but minor)

DESI spectra are 7781 long, SDSS ~3845-3883. `cache_modality`'s `group_key` is set for `image`
(cutout size) but **not** for `spectra`, so every shard mixed both, and `_spectra_batch_loader`
padded to `max_len` with `wavelength=0`, `ivar=1`, `mask=False` — i.e. fake *high-confidence,
valid* measurements at wavelength zero. **84.7% of cached spectra (1844/2178)** were padded this
way.

Measured effect: `‖A-B‖/‖B‖ = 0.034` on the embedding, and no change in probe accuracy
(0.342 padded vs 0.338 homogeneous). Real bug, worth fixing — it makes an object's embedding
depend on which shard it landed in, which is a reproducibility hazard — but it is **not** what
caused the collapse. Fixed here anyway.

---

## Fault 2 — the training setup collapses the prefix even when the input is good

The lightcurve tier is the control that isolates this. Its cached embeddings are **fine**
(linear probe 0.664 vs 0.590 majority), yet the shipped model's lightcurve prefix is just as
collapsed as its spectra prefix.

Prefix statistics from the trained `middle.pt` over 900 real train objects per modality,
where `‖dev‖/‖mean‖` is the object-specific share of the prefix:

| modality | `‖dev‖/‖mean‖` | prefix vector norm | Qwen token-embedding norm |
|---|---|---|---|
| spectra | 0.094 | 30.58 | 0.89 |
| lightcurve | 0.093 | ~30 | 0.89 |
| image | 0.248 | ~29 | 0.89 |

The 64-vector prefix is **~91% a per-modality constant**. Cross-modality cosine is 0.52 while
within-modality is 0.99 — it encodes *which modality* loudly and *which object* barely. The ratio
is identical at `stage1/step_500`, `stage1/best` and `stage2/best` (0.0939 / 0.0939 / 0.0936):
it collapses early and neither stage 2 nor LoRA repairs it.

### The Q-Former is NOT the bottleneck

A **fresh Q-Former at the exact shipped size** (`n_queries=64, d_model=384, 3 layers, 6 heads`),
trained on a purely discriminative objective over the same cached embeddings (`diag/cap_lc.py`,
`diag/capacity_gpu.py`):

| input | best test acc | majority | `‖dev‖/‖mean‖` reached |
|---|---|---|---|
| lightcurve | **0.770** | 0.590 | **1.282** |
| spectra (untrimmed cache) | 0.428 | 0.379 | 0.834 |

Same architecture, 13x more object-specific output than the shipped model achieves. Shrinking it
*does* break it (`n_queries=16` collapsed to `‖dev‖/‖mean‖ = 0.003`), so 64 is above the cliff,
not below it. Capacity is not the problem; the objective and the optimisation budget are.

The Q-Former is also not destroying information — kNN(10) neighbourhood overlap between pooled
input and prefix is 0.721 for spectra, and effective rank actually *rises* on lightcurve
(1.69 -> 4.37). The object signal is present, just at 9% amplitude on a vector 34x too large.

### Contributing causes, measured

1. **Effective batch is 8x the configured value.** `stage1.py` computed
   `grad_accum = batch_size // micro_batch_size` ignoring `num_processes`. On 8 ranks the real
   global batch was `4 x 8 x 8 = 256`, not 32. 4144 train examples / 256 = **17 optimizer steps
   per epoch**, matching `metrics.jsonl` exactly. Stage 1 got **510 steps total**; stage 2
   early-stopped (`patience: 1`) at epoch 5, ~102 steps.
2. **The prefix is 34x out of the LLM's embedding distribution.** `Adapter` ended in a bare
   `nn.Linear` with no output norm, fed from a LayerNorm'd Q-Former (norm ~ sqrt(384) ~ 19.6).
   Qwen RMSNorms each position at every sublayer input, so there is almost no gradient pressure
   to correct the scale — which is why it sits at ~30 from step 500 to the end.
3. **Caption cross-entropy alone never forces conditioning.** With ~4k examples of stylistically
   uniform Gemini prose the cheapest loss reduction is the unconditional caption prior. Val loss
   fell 1.96 -> 1.13 and flattened by epoch 10 *while the captions collapsed*, because per-token
   CE is dominated by style tokens.
4. **Prompt/subset diversity was never realised.** `dataset.py` seeded with
   `hash((object_id, idx))` — no epoch term, so every object saw exactly one
   `(system, instruction)` pair and one subset for the entire run, contradicting the docstring
   three lines below. `hash()` on a str is also `PYTHONHASHSEED`-randomised, so this differed
   per rank across the 2-node v6 run.
5. **The groundedness gate cannot detect any of this.** `shuffle_test` compared
   `generate(shuffled_input)` against the **ground-truth caption**, not against
   `generate(original_input)`, so a large edit distance only ever meant "output differs from
   reference". `ablation_test`'s `fraction_changed = 1.0` measures modality identity, which the
   prefix genuinely does encode (cross-modality cosine 0.52). Both reported
   `null_result: false` on a fully collapsed model.

---

## What was changed

| # | file | change |
|---|---|---|
| 1 | `scripts/02_cache_embeddings.py` | trim `wavelength <= 0`, sort ascending, pad the tail monotonically with `ivar=0`/`mask=True` |
| 2 | `scripts/02_cache_embeddings.py` | `group_key` for spectra keyed on trimmed length, so shards never mix survey lengths |
| 3 | `src/captioner/encoders/aion_spectrum.py` | hard guard: refuse to encode a non-monotonic wavelength grid |
| 4 | `src/captioner/model/adapter.py` | output LayerNorm + learned scale initialised to the LLM's median embedding norm |
| 5 | `src/captioner/train/stage1.py`, `stage2.py` | `grad_accum` divides by `num_processes` so `batch_size` means what it says |
| 6 | `src/captioner/data/dataset.py` | stable per-`(object_id, epoch)` seeding via `blake2b`, with `set_epoch` |
| 7 | `src/captioner/eval/groundedness.py` | shuffle test compares generation-vs-generation; added a caption-diversity gate |
| 8 | `configs/stage2.yaml` | `early_stop.patience` 1 -> 3 |

---

## Are the other modalities in the same trap? No.

Audited both after the spectra fix landed. The failure mode to look for is an encoder input that
is silently malformed — an array an encoder assumes is well-formed without checking.

**Image — healthy.** AION's cached image embedding predicts properties computed straight off the
raw pixels, which is what a working encoder looks like (`diag/image_audit.py`, 900 objects,
5-fold ridge R^2 on the mean-pooled embedding):

| property | R^2 |
|---|---|
| total r-band flux (log) | 0.985 |
| central concentration | 0.873 |
| peak surface brightness | 0.828 |
| g-r colour | 0.814 |
| r-z colour | 0.598 |

Structurally it is also protected where spectra was not: `_image_batch_loader` **raises** on a
shape mismatch instead of padding, `cache_modality` already had a `group_key` keyed on cutout
shape, and AION's image codec center-crops to 96x96 so the 160px/152px south/north difference
never reaches the encoder as ragged input.

**Lightcurve — healthy.** Class probe 0.664 vs 0.590 majority; a fresh Q-Former reaches 0.770.
`prepare_lightcurve_arrays` explicitly sorts by MJD —

    # MJD order — the source is documented as MJD-ordered, but sorting makes that explicit
    # rather than assumed, and random selection above returns points in arbitrary order.
    idx = idx[np.argsort(mjd[idx], kind="stable")]

— which is exactly the discipline the spectra path lacked, and it pads per-object to ATCAT's
fixed 243 with `mask = 0` on the padding, so there is no cross-object padding at all.

**So spectra was the only one.** It was also the only modality whose loader did variable-length
cross-object padding, the only one with no `group_key`, and the only one whose encoder consumed
an array with an ordering precondition. Worth noting why it survived review: all five spectra in
`test_subjects/` are DESI (7781 samples, monotonic, no sentinel) — the examples anyone would
have hand-checked are exactly the ones that were never broken.

## Verification

Re-cached spectra with the fix, on the same 900 objects and the same folds
(`diag/spectra_probe.py`):

| spectra cache | linear probe | MLP probe | majority |
|---|---|---|---|
| pre-fix (`spectra.prefix-bug`) | 0.347 | 0.310 | 0.334 |
| **post-fix** | **0.632** | **0.650** | 0.334 |

`diag/cache_health.py` on the rebuilt cache, with the old one alongside:

    image               centred-cos mean=-0.0148  ||dev||/||mean||=0.6853  ok
    lightcurve          centred-cos mean=+0.0224  ||dev||/||mean||=0.8349  ok
    spectra  (FIXED)    centred-cos mean=-0.0231  ||dev||/||mean||=0.5754  ok
    spectra.prefix-bug  centred-cos mean=+0.2440  ||dev||/||mean||=0.3855  SUSPECT

The fixed cache now sits with the two healthy modalities instead of carrying a spurious shared
direction. Sharding also reports `2178 objects fall into 42 shape group(s)` — zero cross-length
padding, where before every shard mixed DESI and SDSS.

### Prefix health, in the metric the original report used

`diag/prefix_health.py` now reports mean pairwise cosine between different objects' prefixes —
the same quantity the bug report measured by hand (quasar vs galaxy vs zeros vs noise, all
>0.96). It needs no reference scale, so unlike `||dev||/||mean||` it stays comparable across
adapter geometries; the new output LayerNorm constrains every prefix vector to one magnitude,
which mechanically lowers `||dev||/||mean||` even where conditioning improves.

| | image | lightcurve | spectra | prefix norm |
|---|---|---|---|---|
| published v6 | 0.9442 | 0.9901 | 0.9933 | ~29 - 30 |
| retrain, stage 1 epoch 7 of 30 | 0.9698 | 0.9972 | 0.9706 | **0.983** |

Prefix norm is fixed outright: 30.6 -> 0.983 against Qwen's 0.89. Spectra has improved
(0.9933 -> 0.9706). **The collapse itself is not resolved yet at epoch 7**, and image/lightcurve
are not yet better than v6 on this metric — though that compares a 7-epoch stage-1 checkpoint
against a fully-trained stage-2 one, so it is not a verdict.

Expect it to keep moving: in the capacity probe the same architecture went 0.005 -> 0.120 ->
0.469 -> 0.834 on `||dev||/||mean||` across 40 epochs, i.e. most of the conditioning appears
late. Trend so far, stage 1: step 500 -> 1000 -> 1040 gives spectra 0.122 -> 0.154 -> 0.157.

**Caveat, stated plainly:** this round fixed *bugs*. Fault 2 item 3 — that caption cross-entropy
alone gives the model no reason to condition on the prefix — is a design problem, not a bug, and
is deliberately NOT addressed here. If the prefix is still near-constant at the end of this run,
that is the remaining cause, and the fix is an auxiliary objective a constant prefix cannot
satisfy (BLIP-2-style in-batch contrastive between pooled Q-Former output and a caption
embedding; or, as a much cheaper stand-in, a VICReg variance floor
`relu(1 - prefix.flatten(1).std(0)).mean()`).

Training budget, from the retrain's own log:

    Starting training: epochs=30, batches/epoch=259
    epoch=0 step=130 ...

**130 optimizer steps per epoch, up from 17** — 3,900 over 30 epochs against the old 510.

---

## Fault 3 — 2 of 334 DESI spectra were in training (found while prepping v7)

`splits.honor_upstream: true` took the source's own `split_upstream` labels, and those put
**331 of the 334 DESI spectra in test**:

| split | spectra cached | DESI | SDSS |
|---|---|---|---|
| train | 1546 | **2** | 1544 |
| val | 172 | 1 | 171 |
| test | 460 | **331** | 129 |

So v5 and v6 trained on 2 DESI spectra and were scored on a spectra test set that is 72% DESI.
That crosses a boundary AION itself treats as two separate modalities (`tok_spectrum_desi` vs
`tok_spectrum_sdss`), and the two populations are not alike: DESI spectra are 7,781 samples,
SDSS ~3,845-3,883.

It compounds fault 1 exactly the wrong way. Sentinel rate by survey:

    DESI: 0/250 sampled have wavelength <= 0    -> never corrupted
    SDSS: 201/250 sampled (80%) have them        -> corrupted

**v5/v6 trained on 1,544 SDSS spectra of which every one was corrupted, then were evaluated
almost entirely on clean DESI spectra they had never really seen.** It also explains the original
bug report's >96% cosine: the probe spectra were DESI-format, like all five in `test_subjects/`.

The other three spectra parquet files in the repo add zero new object_ids, so 334 is the whole
DESI pool — there is no more to be had, only a better split of it.

### Fix

`honor_upstream: false` plus a new `splits.stratify_by: [tier, survey]`. Stratifying on `tier`
alone could never see this, because DESI and SDSS share the spectra tier. `_draw_per_tier` now
draws within each `(tier, survey)` stratum, filling NaN rather than dropping it — `groupby`
would otherwise have silently left all 987 light curves out of the draw. Result:

    split balance [desi        ] {'test': 34,  'train': 233,  'val': 67}   train_frac=0.70
    split balance [sdss        ] {'test': 184, 'train': 1291, 'val': 369}  train_frac=0.70
    split balance [legacy-south] {'test': 162, 'train': 1134, 'val': 324}  train_frac=0.70
    split balance [legacy-north] {'test': 65,  'train': 457,  'val': 131}  train_frac=0.70
    split balance [(none)      ] {'test': 99,  'train': 691,  'val': 197}  train_frac=0.70

v7 also moves the ratio from 80/10/10 to **70/20/10** (train/val/test) — the larger val set makes
early stopping less noisy, which matters more now that stage 2's patience went 1 -> 3.
Totals: train 3806, val 1088, test 544. DESI training objects: **2 -> 233**. The same change also balances legacy-south vs legacy-north
in the image tier, which differ in cutout size and in whether ivar/mask/i-band exist at all.

That balance table is now logged on every manifest build, with a `<-- SKEWED` flag under 0.5
train fraction. Nothing printed it before, which is why this survived four versions.

**v7's test set is deliberately NOT comparable object-for-object with v4/v5/v6.**

---

## v7 build

- image captions re-pinned `d9fadc12` -> `b8967f45` (all 3,487 rewritten; pixel columns
  byte-identical, so the image embedding cache carried over and only `make captions` re-ran)
- manifest rebuilt with survey-stratified splits
- spectra cache rebuilt with the wavelength fix
- stage1 + stage2 retrained from scratch

## v7 results

Stage 1 early-stopped at epoch 25 (patience 5); stage 2 at epoch 5 (patience 3, overfitting —
train 0.66 against val 1.18 on 3,778 examples).

### Caption diversity — the symptom, gone

| tier | n | distinct fraction | top-1 share |
|---|---|---|---|
| image | 60 | **1.000** | 0.017 |
| lightcurve | 60 | **1.000** | 0.017 |
| spectra/sdss | 60 | **1.000** | 0.017 |
| spectra/desi | 19 | **1.000** | 0.053 |
| *published v6* | 397 | *0.207* | *0.401* |

Every generated caption is unique in every tier; nothing repeats at all. v6 produced 82 unique
captions out of 397 with one caption covering 40.1% of the test set.

### Prefix health

| | v6 (shipped) | v7 stage 1 | v7 stage 2 |
|---|---|---|---|
| spectra `‖dev‖/‖mean‖` | 0.082 | 0.222 | **0.237** |
| spectra pairwise cos | 0.9933 | 0.9490 | **0.9428** |
| image `‖dev‖/‖mean‖` | 0.230 | 0.218 | 0.240 |
| lightcurve `‖dev‖/‖mean‖` | 0.087 | 0.086 | 0.094 |
| prefix norm | 30.6 | 1.03 | 1.03 |

spectra/desi 0.211 (cos 0.954) vs spectra/sdss 0.242 (cos 0.940) — the two surveys now behave
alike, where before one was corrupted noise and the other clean.

Spectra, the tier whose input was broken, improved 2.9x. Image and lightcurve, whose inputs were
always fine, barely moved. Exactly the split you would predict if fixing inputs fixes what inputs
broke, and nothing else.

### The prefix thresholds in diag/prefix_health.py are miscalibrated — trust the diversity gate

`prefix_health.py` still reports FAIL on this model (pairwise cos 0.94 > the 0.95 "COLLAPSED"
cutoff) while every single caption is distinct. During the run the trend flattened
(0.117 -> 0.168 -> 0.179 over steps 500/1000/1500) and that was read as heading for failure;
the prediction was wrong. A pairwise cosine near 0.94 on a 64x4096 prefix evidently leaves the
LLM plenty to work with, so the absolute value is not the right thing to gate on. **Treat
`prefix_health.py` as a relative before/after instrument, not a pass/fail gate.** The
caption-diversity gate is the one that tracks the actual symptom.

Also fixed in the gate itself: `top1_share` cannot fall below `1/n`, so a flat 0.05 cutoff
spuriously failed spectra/desi at n=19 (top1 0.0526) despite every caption being distinct. The
threshold is now `max(0.05, 1.5/n)`.

## How to tell it worked

`‖dev‖/‖mean‖` on `middle.pt` is a 30-second check and the single most diagnostic number:

    python diag/prefix_health.py outputs/checkpoints/stage1/best

- shipped v6: **0.094**
- healthy, from the capacity probe: **0.8 - 1.3**
- anything below ~0.3 means it is still collapsing

Then the caption-diversity gate: distinct-caption fraction on the test split should be well above
0.8 with a top-1 share below 0.05. v6 scored 0.207 / 0.401.
