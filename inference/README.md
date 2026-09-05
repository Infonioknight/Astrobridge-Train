# Running the captioner on a Modal GPU

Lets you ask the published AstroBridge captioner a free-form question about a real image and/or
spectrum, using a rented GPU from [Modal](https://modal.com) — no local GPU, no local checkpoint
files, just this folder and a Modal account. The model itself is pulled fresh from HF Hub
(`UniverseTBD/astrobridge-captioner-v3`, see `modal_app.py`'s `MODEL_REPO_ID`) on every remote call.

If you haven't read `modal_app.py` yet, read it first — every piece (`image`, `app`, `infer`,
`main`) has a docstring explaining what it's for and what's safe to change. This README is about
*running* it; that file is about *understanding* it.

## 1. One-time setup

```bash
uv tool install modal      # installs the `modal` CLI globally (see the pip-vs-uv note below)
modal setup                 # opens a browser, logs you into your Modal account
```

`modal setup` writes a token to `~/.modal.toml`. You only do this once per machine.

## 2. Auth: giving the remote container access to the model

`UniverseTBD/astrobridge-captioner-v3` may be private (check on huggingface.co — the padlock icon
on the repo page tells you). If it's private, the remote container needs a Hugging Face token to
download it:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Get a token (read access is enough) from https://huggingface.co/settings/tokens. This only needs
doing once — Modal stores the secret, and `modal_app.py` already references it by name
(`secrets=[modal.Secret.from_name("huggingface-secret")]`).

If the model repo is public, you can skip this step entirely and remove the `secrets=[...]` line
from `modal_app.py` — but leaving it in when the repo is public is harmless too.

## 3. Running from a terminal

```bash
cd captioner   # modal_app.py does `add_local_python_source("captioner")`, so run this from
               # wherever `src/captioner` is importable — the repo root, typically.

modal run inference/modal_app.py \
    --question "What kind of object is this and why do you think so?" \
    --image-npy test_subjects/image_03.npy
```

Other flags (all optional except `--question`; give any combination of the three):
- `--spectrum-npz path/to/spectrum.npz --spectrum-survey desi` (or `sdss`) — a spectrum,
  instead of or alongside an image.
- `--lightcurve-npz path/to/lightcurve.npz` — a light curve. Needs `mjd`/`flux`/`flux_err`/
  `band_id` arrays (optionally `use`) in SNANA FLUXCAL / ATCAT band-id conventions — see
  `inference/local.py`'s module docstring for exact units; the preprocessing (detection-window
  trim, downsample, pad) is applied identically to how the training cache was built.
- `--max-new-tokens 300` — raise this if answers are getting cut off mid-sentence.

The very first call ever will be slow (~1-2 min) — it's downloading the ~18GB base LLM plus the
adapter/fusion-stack from HF Hub. That's normal, not stuck; the `infer` function's `timeout=600`
in `modal_app.py` exists specifically to give this room. **Every call after that is fast**, even
from a brand-new container: `modal_app.py` mounts a Modal Volume (`astrobridge-hf-cache`) at HF's
cache path, so the download only ever happens once, ever — the weights persist in Modal's own
storage between calls, not in the container (which is wiped every cold start) and not on your
machine (you don't need any local disk for this at all).

## 4. Running from a notebook

Two steps: deploy the app once (turns it into a standing, callable app rather than a one-shot
script run), then call it from anywhere — a notebook, another script, doesn't matter.

```bash
modal deploy inference/modal_app.py
```

Then, in a notebook cell:

```python
import modal

infer_fn = modal.Function.from_name("astrobridge-captioner-infer", "infer")

answer = infer_fn.remote(
    question="What kind of object is this and why do you think so?",
    image_npy_bytes=open("test_subjects/image_03.npy", "rb").read(),
)
print(answer)
```

Note the notebook path calls `infer` directly (the GPU function), not `main` (the CLI-only local
entrypoint) — `main` only exists for the `modal run` terminal flow. That's also why the notebook
call uses `image_npy_bytes=` (raw file bytes) instead of `image_npy=` (a file path): `main()` is
the piece that turns a path into bytes for you on the terminal path; from a notebook you do that
read yourself before calling `.remote()`.

If you change `modal_app.py` after deploying, run `modal deploy inference/modal_app.py` again to
push the update — `Function.from_name` always points at whatever was most recently deployed under
that name.

## 5. Comparing our model against plain, out-of-the-box Qwen

`compare.py` runs ONE uniform prompt against every image in a batch, two ways each, using the
same loaded base weights for both sides (model is loaded once for the whole batch, not once per
image):

- **Plain Qwen** — the grz array turned into an ordinary RGB picture (the standard Lupton
  composite), fed through Qwen's own native vision pathway, LoRA disabled. Nothing this project
  trained touches this side at all.
- **Our equipped pipeline** — AION image encoder → fusion stack → LoRA-adapted Qwen, exactly what
  `modal_app.py` does.

```bash
modal run inference/compare.py
```

Defaults: runs over every `test_subjects/image_*.npy`, using the prompt `"Describe the object
shown in this image."`, writing results to `inference/comparison_results.json`. Override any of
that:

```bash
modal run inference/compare.py \
    --images-glob "test_subjects/image_*.npy" \
    --prompt "Describe the object shown in this image." \
    --output inference/comparison_results.json
```

Output JSON:
```json
{
  "prompt": "Describe the object shown in this image.",
  "outputs": {
    "image_01.npy": ["<plain Qwen answer>", "<our equipped model answer>"],
    "image_02.npy": ["<plain Qwen answer>", "<our equipped model answer>"]
  }
}
```
Each image's value is `[plain_qwen_answer, equipped_answer]`, in that order.

This always runs both sides for every image — there's no opt-out flag, unlike `modal_app.py`'s
single answer — so expect roughly double the billed GPU time of a normal `modal_app.py` call,
times however many images are in the batch (5, by default). See Cost below.

## Cost

Real numbers from Modal's own pricing page (per-second billing, no charge while idle —
confirmed, not a guess):

| Resource | Rate |
|---|---|
| GPU: L4 (currently set) | $0.000222/sec |
| CPU: 2 cores (currently set) | $0.0000262/sec |
| RAM: 16 GiB (currently set) | $0.0000355/sec |
| **Total, this config** | **≈ $0.000284/sec ≈ $1.02/hour** |

You're billed only for actual execution time (model load + generation), not for time between
calls. A single call is realistically well under a minute of billed time once the HF cache volume
is warm (see above) — so each call costs on the order of a couple of cents, not dollars.

L4 was picked deliberately over A10G/A100: 24GB VRAM already comfortably fits the ~18GB bf16
model, at roughly 1/3 of A100's per-second rate. A100's extra VRAM/bandwidth isn't something this
single-request, batch-of-one workload benefits from — bumping the GPU up doesn't make individual
calls meaningfully cheaper here, only more expensive per second.

## Common things to change

All covered in more detail in `modal_app.py`'s own docstring, but the short version:

| Want to... | Change... |
|---|---|
| Use a different/future model version | `MODEL_REPO_ID` at the top of `modal_app.py` |
| Use a bigger/smaller/cheaper GPU | `gpu="L4"` on the `@app.function(...)` line — see Cost above before bumping up |
| Give it more CPU/RAM | `cpu=2.0` / `memory=16384` (MiB) on the `@app.function(...)` line |
| Allow longer answers | `--max-new-tokens` (terminal) or `max_new_tokens=` (notebook) |
| Fix a cold-start timeout | `timeout=600` on the `@app.function(...)` line |

## Why `uv tool install modal` and not `pip install modal`

This project moved its own dependency management from `pip` to `uv` (see the main `README.md`).
`modal` is a CLI tool you run (`modal run`, `modal deploy`), not something `captioner`'s own code
imports — so it belongs installed globally via `uv tool install`, not tracked as a project
dependency in `pyproject.toml`. If you'd rather pin it as an actual dev dependency of this repo
instead, `uv add --dev modal` works too and updates `uv.lock`.
