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
uv run python -m eval.runners.run_lightcurve_eval --track lightcurve_only
uv run python -m eval.runners.run_image_eval --track base_only
```

**Not** `uv run python eval/runners/run_lightcurve_eval.py ...` — that fails with
`ModuleNotFoundError: No module named 'eval'`, since running a script directly puts its own
directory on `sys.path`, not the repo root (`-m` puts the current working directory there
instead, which is what makes `from eval.backend import ...` resolve).

## Tracks

### Lightcurve — SN typing (`eval/runners/run_lightcurve_eval.py`)

Real, held-out benchmark: `BuildNg/astrobridge-yse-test-dataset-v2`, confirmed 266 objects, zero
`object_id` overlap with the training set. Ground truth: `class_label` ∈ `{SN Ia, SN II, SN Ibc}`
(confirmed imbalanced: 180/71/15).

```bash
uv run python -m eval.runners.run_lightcurve_eval --track lightcurve_only
uv run python -m eval.runners.run_lightcurve_eval --track lightcurve_plus_image
```

No base-model comparison for this track — a raw lightcurve array isn't something an
out-of-the-box vision-language model can consume at all; only the equipped model is scored.
`lightcurve_plus_image` tests whether adding the host image improves classification over
`lightcurve_only` alone — genuinely new data-loading code (see `eval/datasets/lightcurve_yse.py`'s
module docstring for the one real caveat: the host image's band order hasn't been separately
verified against this specific dataset).

### Image — Galaxy Zoo morphology (`eval/runners/run_image_eval.py`)

Real benchmark with a proper train/test split already: `astronolan/galaxy10-aion` (a Galaxy10
DECaLS release pre-built for AION specifically), 796 test objects, 10 morphology classes.

```bash
uv run python -m eval.runners.run_image_eval --track base_only            # safe, no blockers
uv run python -m eval.runners.run_image_eval --track equipped_and_base    # see caveat below
```

`base_only` (the default) compares the base model against ground truth using the dataset's
pre-rendered `image_rgb` field — no AION involved, safe to run today. `equipped_and_base` also
runs our equipped pipeline through AION, but applies an **unverified hypothesis** about the raw
`image_bands` field (4 bands where AION expects 3 — see `eval/datasets/image_galaxy10.py`'s
module docstring for the full reasoning and the verification step to run first). The runner
prints a loud warning every time this track runs, on purpose — don't trust its `equipped` numbers
until that's checked.

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
  **Not yet live-verified** whether `modal deploy` cleanly targets `eval/backend.py` as a library
  file rather than a script meant to be the sole entrypoint — spike this once (deploy, then a
  throwaway `modal.Function.from_name("astrobridge-eval-backend", "_equipped_infer").remote(...)`
  call) before trusting `--backend modal` end-to-end. See `eval/backend.py`'s module docstring for
  the fallback design if it doesn't round-trip cleanly.

## Metrics & output

Every track reports overall accuracy **and** per-class precision/recall/F1 (`eval/metrics/
classification.py`) — not just a single accuracy number, since both taxonomies here are
imbalanced enough that a model always guessing the majority class would otherwise look
deceptively good. A model answer that doesn't match any known label counts as wrong, not silently
dropped (`eval/metrics/caption_to_label.py`).

Reports land in `outputs/eval/classification/{dataset_slug}_{track}.json`, e.g.
`outputs/eval/classification/yse_lightcurve_only.json` — same `outputs/eval/` tree
`scripts/04_eval.py`'s groundedness report already uses.

## `--limit` for quick smoke tests

Every runner takes `--limit N` to evaluate only the first N objects — useful for confirming a
backend/track actually runs end-to-end before committing to a full (potentially billed, on Modal)
run over the whole benchmark.
