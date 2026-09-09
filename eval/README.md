# Classification eval bench — trained model vs. base LLM

Answers one question per modality: does the trained model (AION encoders + fusion stack +
LoRA-adapted `Qwen/Qwen3.5-9B`, published at `UniverseTBD/astrobridge-model-v3_qwen`) actually
*understand* what it's looking at, measured against real external ground-truth labels — not
whether its captions merely sound plausible. This is a first step, deliberately narrower than
caption-quality eval: prompt the model to classify a labeled benchmark object into a fixed set of
choices, and compare against ground truth.

Read `eval/backend.py`'s module docstring first if unfamiliar — every runner here is written
against `EvalBackend`'s plain `.generate(raw_inputs, question, max_new_tokens) -> str` interface
and never touches `modal`/`transformers`/`captioner.inference` directly; that file is the one
place "how do I call the model, and where" lives.

**Important — module invocation only.** `eval/` is a package but is not `pip install`ed (only
`captioner`, under `src/`, is). Run every script here as a module, from the repo root:

```bash
uv run python -m eval.runners.collect_lightcurve_labels --n 90 --seed 0
uv run python -m eval.runners.run_image_eval --track base_only
```

**Not** `uv run python eval/runners/collect_lightcurve_labels.py ...` — that fails with
`ModuleNotFoundError: No module named 'eval'`, since running a script directly puts its own
directory on `sys.path`, not the repo root (`-m` puts the current working directory there
instead, which is what makes `from eval.backend import ...` resolve).

## Tracks

### Lightcurve — SN typing, collect + score (`collect_lightcurve_labels.py`, then `score_lightcurve_eval.py`)

Real, held-out benchmark: `BuildNg/astrobridge-yse-test-dataset-v2`, confirmed 266 objects, zero
`object_id` overlap with the training set. Ground truth: `class_label` ∈ `{SN Ia, SN II, SN Ibc}`
(confirmed imbalanced: 180/71/15). Same two-step, both-models, digit-code shape as the image
track below (see that section for why the shape is split this way) — the base-model side gets
`render_lightcurve_plot`'s rendered flux-vs-time PNG instead of raw `atcat_*` arrays (an
out-of-the-box vision-language model can't consume those directly, same reasoning as the image
track's `image_rgb` vs `image_bands`), the equipped side gets the raw arrays via
`build_raw_inputs_lightcurve`.

```bash
uv run python -m eval.runners.collect_lightcurve_labels --n 90 --seed 0
uv run python -m eval.runners.score_lightcurve_eval --in outputs/eval/raw_generations/yse_lightcurve_only_seed0_n90.json
```

Omit `--n` to use the entire 266-object eval set (no sampling needed at that size); pass `--n` for
a quick smoke test first. `--track lightcurve_plus_image` tests whether adding the host image
improves classification over `lightcurve_only` alone on the equipped side — genuinely new
data-loading code (see `eval/datasets/lightcurve_yse.py`'s module docstring for the one real
caveat: the host image's band order hasn't been separately verified against this specific
dataset); the base side always sees the same rendered lightcurve plot regardless of `--track`,
since it has no way to consume a second image alongside it.

### Image — Galaxy Zoo morphology, collect + score (`collect_image_labels.py`, then `score_image_eval.py` and/or `score_image_eval_debiased.py`)

Real benchmark with a proper train/test split already: `astronolan/galaxy10-aion` (a Galaxy10
DECaLS release pre-built for AION specifically), 796 test objects, 10 morphology classes.

This track is deliberately split into two scripts, not one — collection (running the models) is
the expensive, potentially-billed-on-Modal step; scoring is pure post-processing over whatever
collection already saved. Re-run scoring as many times as you want (different `caption_to_label`
heuristics, a different crossmatch radius) without ever paying for inference again.

**Step 1 — collect** (`eval/runners/collect_image_labels.py`): draws a seeded, stratified sample
covering all 10 classes (a floor guarantees even `Cigar Shaped Smooth Galaxies`, which only has 8
objects total, still appears), runs the base model over `image_rgb` for the whole sample, frees
it, then runs the equipped model over `image_bands` for the whole sample — same VRAM-bounding
order as everywhere else in this folder. Saves **raw caption text** from both sides (not just a
parsed label) plus `ra`/`dec`/`label_name` per object.

```bash
uv run python -m eval.runners.collect_image_labels --n 150 --seed 0
```

Every random decision here is seeded and recorded in the output: `--seed` fully determines the
sample (`numpy.random.default_rng(seed)`, consumed in a fixed class order — same seed always
draws the same 150 objects), and generation itself is deterministic (`do_sample=False` on both
sides) modulo GPU floating-point reduction order, which isn't something this project controls.
The equipped side applies an **unverified hypothesis** about `image_bands` (4 bands where AION
expects 3 — see `eval/datasets/image_galaxy10.py`'s module docstring for the full reasoning and
the verification step to run first); don't trust `equipped_answer` values until that's checked.

Output: `outputs/eval/raw_generations/galaxy10_seed<seed>_n<n>.json`.

**Step 2 — score, default path** (`eval/runners/score_image_eval.py`): loads that file, no
model/GPU/Modal/network involved at all — fast, always safe to re-run.

```bash
uv run python -m eval.runners.score_image_eval --in outputs/eval/raw_generations/galaxy10_seed0_n150.json
```

Reports a hard accuracy/F1 breakdown **and** the coarse group score (see "Group scoring" below)
for both sides, side by side.

**Step 2b — score, debiased vote-fraction path** (`eval/runners/score_image_eval_debiased.py`):
a separate, parked script — needs a real network crossmatch against
`astronolan/gz-decals-embeddings`, meaningfully slower than the default path above, so it's kept
independent rather than bundled into every scoring run.

```bash
uv run python -m eval.runners.score_image_eval_debiased --in outputs/eval/raw_generations/galaxy10_seed0_n150.json
```

Reports the crowd-vote-fraction soft score (see "Crowd-vote soft scoring" below) for both sides.

### Spectra — not built yet

Left as an obvious extension point: a new `eval/datasets/spectra_<source>.py` +
`eval/runners/run_spectra_eval.py`, following the same shape as the two tracks above. Nothing
else in this folder needs to change to add it.

## Choosing a compute backend

Every runner takes `--backend local|modal` (default `local`). Switching is a one-line CLI flag —
the actual difference in *how* either backend works only ever needs editing in `eval/backend.py`,
nowhere else.

- `--backend local`: runs in-process on whatever GPU/CPU this machine has. Simplest, no setup
  beyond the project's normal `uv pip install -e ".[dev]"`.
- `--backend modal`: talks to an already-deployed Modal app. One-time setup:
  ```bash
  uv tool install modal      # see the main README's uv-vs-pip note
  modal setup                 # logs in
  modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
  modal deploy eval/backend.py
  ```
  **Live-verified**: deployed and round-trip tested against both `_equipped_infer` and
  `_base_infer` — both returned real generations end-to-end. One real fix needed along the way:
  Modal's CLI looks for a variable literally named `app` by default, so the `modal.App` instance
  in `eval/backend.py` is named `app`, not `modal_app` (an earlier version used `modal_app` and
  failed deploy with `module 'backend' has no attribute 'app'`).

## Metrics & output

Both tracks report overall accuracy **and** per-class precision/recall/F1 (`eval/metrics/
classification.py`) — not just a single accuracy number, since both taxonomies here are
imbalanced enough that a model always guessing the majority class would otherwise look
deceptively good. A model answer that doesn't match any known label counts as wrong, not silently
dropped (`eval/metrics/caption_to_label.py`).

Both tracks' collect step lands in `outputs/eval/raw_generations/` (`yse_<track>_seed<seed>_n<n>.json`
for lightcurve, `galaxy10_seed<seed>_n<n>.json` for image); scoring writes `..._scored.json`
alongside it. Image scoring additionally has a separate `score_image_eval_debiased.py` writing
`..._debiased_scored.json` (soft, see below) — the lightcurve track has no soft-score analog (no
Galaxy-Zoo-style crowd vote fractions exist for SN typing), so `score_lightcurve_eval.py` is the
only scoring script for that track.

### Crowd-vote soft scoring (image track only)

Hard accuracy treats every wrong answer the same — confusing two visually similar spiral classes
counts exactly as badly as confusing a spiral with a smooth round galaxy. The soft score
(`eval/metrics/vote_fraction_scoring.py`) fixes that using **real Galaxy Zoo DECaLS crowd vote
fractions** (`astronolan/gz-decals-embeddings`, crossmatched by RA/Dec via
`eval/datasets/gz_decals_votes.py`), not an invented distance metric:

1. Each of Galaxy10's 10 classes has a real, published defining rule — transcribed directly from
   `henrysky/Galaxy10`'s own dataset-construction notebook, e.g. "Barred Spiral" =
   `has-spiral-arms_yes_debiased > 0.8` AND `bar_no_debiased < 0.2`.
2. For one object, each class gets a continuous "how well does this object satisfy that rule"
   margin in `[0, 1]` — an AND-conjunction takes the *weakest* condition (`min`), an OR takes the
   *best* alternative (`max`).
3. The whole 10-class vector is rescaled by the *true* class's own margin, so the true class
   always lands at exactly `1.0` and every other class is "X% as plausible as the truth, according
   to real crowd votes for this specific object."
4. A wrong prediction scores `vector[predicted_class]` — a near-miss (a plausible confusion the
   crowd itself was genuinely divided on) scores well above 0; a wild miss scores near 0.

Objects that don't crossmatch within `--crossmatch-radius-arcsec` (default 1.0), or whose own true
class has no reliable vote data (`_debiased.mask` set, or the column entirely missing), are
excluded from the soft score and counted separately in the report (`n_excluded_no_crossmatch`,
`n_excluded_true_class_unscoreable`) — never silently folded into the average as a 0.

### Group scoring (image track only) — a simpler complement to the soft score, not a replacement

`eval/metrics/group_scoring.py` reports a third, deliberately coarser number: `1.0` exact match,
`0.5` same top-level morphology group, `0.0` different group or unparseable. The 4 groups —
**Smooth** (Round/In-between/Cigar), **Spiral** (Barred/Unbarred Tight/Unbarred Loose), **Edge-On**
(with/without Bulge), **Disturbed/Merging** — map onto the real Galaxy Zoo decision tree's own
top-level branches, not an arbitrary split.

Needs no crossmatch and no external vote data at all, so it's always computable — but it's coarser
than the soft score by design: a near-miss within a group (Round Smooth predicted as In-between
Round Smooth) scores the same `0.5` as a real miss within that same group (Round Smooth predicted
as Cigar Shaped), and a plausible cross-group confusion (an edge-on spiral vs. a face-on spiral)
scores a flat `0.0` the soft score might not. Use both — group scoring for a quick, easily
sanity-checked read; the soft score for the actual fine-grained signal.

## Quick smoke tests before a full/billed run

Both tracks' collect scripts take `--n` directly (lightcurve: `collect_lightcurve_labels.py`,
omit for the full 266-object set, min-feasible bounded by SN Ibc's 15 real objects; image:
`collect_image_labels.py`) — pass a small `--n` (e.g. `--n 20`) for a quick end-to-end check
before committing to a full run, especially on `--backend modal` where every collect run is
billed.
