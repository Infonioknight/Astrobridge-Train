"""Runs the published AstroBridge captioner on a Modal GPU — from a terminal (`modal run`) or
from a notebook (`modal.Function.from_name(app, "infer").remote(...)` after `modal deploy`).

Deliberately separate from inference/local.py: that script assumes you already have a local
checkpoint directory (outputs/checkpoints/stage2/best) and a local GPU. This file instead pulls
the model straight from the published HF Hub repo (see captioner.inference.
load_inference_model_from_hub) and runs on a *rented* GPU via Modal, so trying the model needs
neither a local GPU nor a local checkpoint — just this file and a Modal account. Accepts the same
three modalities as local.py (image, spectrum, light curve, any combination) — see that script's
module docstring for the exact per-modality array shapes/units expected.

For a side-by-side comparison against plain out-of-the-box Qwen (its own native vision pathway,
no LoRA, nothing this project trained), see inference/compare.py instead — this file only ever
runs our fully-equipped pipeline.

--- The four pieces, and what each is for ---

1. `image` (a modal.Image): describes the *container* the remote GPU runs your code in — think
   of it as "what would I `pip install` on a fresh machine before running this." It does NOT
   include your model weights; those get downloaded at call time from HF Hub instead (see
   MODEL_REPO_ID below), so nothing large is baked into the container image.

2. `app` (a modal.App): the top-level handle Modal groups everything under. The name
   ("astrobridge-captioner-infer") is how a notebook finds this app later via
   `modal.Function.from_name(name, function_name)` — see the README for that flow.

3. `infer` (an `@app.function(...)`-decorated function): this is the part that actually runs on
   the remote GPU. Calling `infer(...)` directly in this file (or importing it) runs it locally
   like any other Python function; calling `infer.remote(...)` is what actually ships the call to
   Modal's GPU. Everything inside its body executes in the remote container, not on your machine.

4. `main` (an `@app.local_entrypoint()`-decorated function): this is what runs on *your* machine
   when you type `modal run inference/modal_app.py ...` — it's the CLI argument parsing layer,
   which then calls `infer.remote(...)` to do the actual work remotely.

--- Things you'll likely want to change ---

- MODEL_REPO_ID: point this at a different published model (e.g. a future v4) without touching
  anything else.
- gpu="L4" on the @app.function(...) line: see README.md's "Cost" section before bumping this up
  — L4 already comfortably fits this model; a bigger GPU costs more per second without making
  this single-request workload meaningfully faster.
- timeout=600: raise this if you increase --max-new-tokens a lot, or if cold-start (downloading
  the base LLM + published adapter from HF Hub on a fresh container) is taking longer than 10 min
  on a slow connection.
- Dependencies: nothing to maintain here — `image` installs straight from `pyproject.toml`/
  `uv.lock` via `uv_sync()` (see that call's comment), so it always matches the rest of the
  project automatically.
"""
from __future__ import annotations

import modal

MODEL_REPO_ID = "UniverseTBD/astrobridge-captioner-v3"

app = modal.App("astrobridge-captioner-infer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Installs from THIS repo's real pyproject.toml + uv.lock (frozen — exact pinned versions,
    # no drift) rather than a hand-picked package list. Confirmed real gap in an earlier version
    # of this file: a manually-curated "minimal inference-only" list missed `pandas` (needed
    # just to import captioner.train.stage1, transitively pulled in by captioner.inference, even
    # though inference itself never calls the pandas-using functions in that file) — Python
    # still runs every top-level import in a module it loads, regardless of which specific name
    # you're importing from it. uv_sync() installs the same dependency set used everywhere else
    # in this project, so this whole class of bug can't recur just because inference's own real
    # needs are a subset of the full project's.
    .uv_sync()
    # Ships your local `src/captioner` package into the container, so `import captioner` works
    # remotely exactly like it does locally. If you rename/move the package, update this path.
    .add_local_python_source("captioner")
    # `configs/` isn't part of the `captioner` Python package (it's a sibling directory at the
    # repo root, alongside `src/`), so add_local_python_source above never ships it — confirmed
    # real: without this, load_config() fails with "No such file or directory: '/configs/...'".
    # That exact path isn't a coincidence: add_local_python_source mounts the package at
    # /root/captioner/... (skipping the local src/ layer), and utils/config.py's CONFIG_DIR
    # climbs 3 parents up from itself to find `configs/` — from /root/captioner/utils/config.py
    # that lands on exactly /configs, which is why mounting configs/ there (not somewhere else)
    # makes utils/config.py work completely unmodified.
    .add_local_dir("configs", remote_path="/configs")
)

# Persistent, Modal-hosted storage (lives in Modal's cloud, not your machine — needs zero local
# disk) for HF Hub's download cache. Without this, every cold-start container (any time Modal
# spins up a fresh instance — after the previous one's been idle a few minutes, or to handle
# concurrent calls) re-downloads the ~18GB base LLM plus the adapter from scratch, since a fresh
# container's own filesystem is otherwise empty every time. Mounted at HF's own default cache
# path below, so no extra env vars are needed — `huggingface_hub`/`transformers` already look
# there by default. First call ever populates it; every call after that, on any container,
# finds the weights already present and skips the download entirely.
hf_cache_volume = modal.Volume.from_name("astrobridge-hf-cache", create_if_missing=True)


@app.function(
    # L4 over A10G/A100: 24GB VRAM comfortably fits the ~18GB bf16 model with headroom to spare,
    # at roughly 1/3 A100's per-second rate and ~27% less than A10G — see README.md's "Cost"
    # section before bumping this up.
    gpu="L4",
    # This workload does light numpy/tokenization only, no heavy CPU compute — 2 cores / 16 GiB
    # is already generous, not a bare minimum. `memory` is in MiB (Modal's own unit), not GB.
    cpu=2.0,
    memory=16384,
    image=image,
    # `huggingface-secret` is a Modal Secret you create once in the Modal dashboard (or via
    # `modal secret create huggingface-secret HF_TOKEN=hf_...`), holding an HF token with read
    # access to MODEL_REPO_ID. Needed if the model repo is private — see the README's "Auth"
    # section for exactly how to set this up.
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=600,
)
def infer(
    question: str,
    image_npy_bytes: bytes | None = None,
    spectrum_npz_bytes: bytes | None = None,
    spectrum_survey: str | None = None,
    lightcurve_npz_bytes: bytes | None = None,
    max_new_tokens: int = 128,
) -> str:
    """Runs entirely inside the remote container. `*_bytes` args (rather than file paths) because
    a remote GPU container can't see your local filesystem — the actual file contents have to be
    sent over the wire; see `main()` below for how they get read and passed in.

    Mirrors inference/local.py's raw_inputs construction exactly, modality for modality — see
    that file's module docstring for each modality's exact expected array shapes/keys/units
    (image band order and shape, spectrum survey requirement, light-curve SNANA/ATCAT conventions
    and the detection-window/downsample/pad preprocessing prepare_lightcurve_arrays applies).
    """
    import io

    import numpy as np
    import torch

    from captioner.data.transients_dataset import prepare_lightcurve_arrays
    from captioner.inference import generate_caption, load_inference_model_from_hub
    from captioner.utils.config import load_config

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only build encoders for the modalities this specific call actually provides — no reason to
    # download/load AION's spectrum encoder or ATCAT's light-curve one on an image-only call.
    requested_modalities = [
        name for name, given in (
            ("image", image_npy_bytes is not None),
            ("spectra", spectrum_npz_bytes is not None),
            ("lightcurve", lightcurve_npz_bytes is not None),
        ) if given
    ]
    model, tokenizer, encoders = load_inference_model_from_hub(
        cfg, MODEL_REPO_ID, device=device, modality_names=requested_modalities,
    )
    # Makes this call's downloads (if any were needed) durably visible to every other container —
    # without this, a fresh cold-start container isn't guaranteed to see files another container
    # wrote to the volume moments earlier. Cheap/instant when nothing new was actually downloaded.
    hf_cache_volume.commit()

    raw_inputs: dict = {}
    if image_npy_bytes is not None:
        pixel_values = torch.from_numpy(np.load(io.BytesIO(image_npy_bytes))).unsqueeze(0)
        raw_inputs["image"] = {"pixel_values": pixel_values}
    if spectrum_npz_bytes is not None:
        if spectrum_survey is None:
            raise ValueError("spectrum_survey ('desi' or 'sdss') is required when a spectrum is given.")
        npz = np.load(io.BytesIO(spectrum_npz_bytes))
        spectrum_batch: dict = {
            "flux": torch.from_numpy(npz["flux"]).unsqueeze(0).float(),
            "wavelength": torch.from_numpy(npz["wavelength"]).unsqueeze(0).float(),
            "survey": [spectrum_survey],
        }
        if "ivar" in npz:
            spectrum_batch["ivar"] = torch.from_numpy(npz["ivar"]).unsqueeze(0).float()
        if "mask" in npz:
            spectrum_batch["mask"] = torch.from_numpy(npz["mask"]).unsqueeze(0).bool()
        raw_inputs["spectra"] = spectrum_batch
    if lightcurve_npz_bytes is not None:
        npz = np.load(io.BytesIO(lightcurve_npz_bytes))
        required = ("mjd", "flux", "flux_err", "band_id")
        absent = [k for k in required if k not in npz]
        if absent:
            raise ValueError(f"lightcurve .npz is missing array(s) {absent}; required: {list(required)} (plus optional `use`).")
        use = npz["use"] if "use" in npz else np.ones(len(npz["mjd"]), dtype=bool)
        lc_modality = cfg.modalities.lightcurve
        lc_kwargs = lc_modality.encoder.get("kwargs", {})
        arrays, info = prepare_lightcurve_arrays(
            npz["mjd"], npz["flux"], npz["flux_err"], npz["band_id"], use,
            object_id="modal-inference",  # diagnostic-only label, not a real object identity
            seq_len=int(lc_modality.max_tokens),
            detection_window_days=float(lc_kwargs.get("detection_window_days", 30.0)),
            detection_snr=float(lc_kwargs.get("detection_snr", 5.0)),
            seed=int(lc_kwargs.get("subsample_seed", 0)),
        )
        print(f"Light curve: {info['n_accepted']} accepted, {info['n_in_window']} in window, {info['n_selected']} encoded.")
        raw_inputs["lightcurve"] = {k: torch.from_numpy(v[None, :]) for k, v in arrays.items()}

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    return generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt.template, device,
        raw_inputs, max_new_tokens=max_new_tokens, question=question,
    )


@app.local_entrypoint()
def main(
    question: str,
    image_npy: str = None,
    spectrum_npz: str = None,
    spectrum_survey: str = None,
    lightcurve_npz: str = None,
    max_new_tokens: int = 128,
):
    """Runs on YOUR machine — reads local files (if given) into bytes, then hands off to the
    remote GPU function. This is the function `modal run inference/modal_app.py --question "..."`
    actually calls; its keyword arguments become the CLI flags automatically (Modal derives
    `--image-npy` etc. from `image_npy` here).
    """
    image_bytes = open(image_npy, "rb").read() if image_npy else None
    spectrum_bytes = open(spectrum_npz, "rb").read() if spectrum_npz else None
    lightcurve_bytes = open(lightcurve_npz, "rb").read() if lightcurve_npz else None

    answer = infer.remote(
        question=question,
        image_npy_bytes=image_bytes,
        spectrum_npz_bytes=spectrum_bytes,
        spectrum_survey=spectrum_survey,
        lightcurve_npz_bytes=lightcurve_bytes,
        max_new_tokens=max_new_tokens,
    )
    print(f"Answer: {answer}")
