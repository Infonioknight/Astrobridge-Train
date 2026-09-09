"""THE single editable module for "how do I call the model" — swap local GPU vs. a rented Modal
GPU by changing `kind="local"` to `kind="modal"` in a runner's `get_backend(...)` call, nothing
else. Every `eval/runners/*.py` script and `eval/metrics/*.py` module is written against
`EvalBackend`'s plain `.generate(...)` interface and never imports `modal`, `transformers`, or
`captioner.inference` directly — this file is the only place those live.

**Two sides, loaded one at a time, never both simultaneously**: "equipped" (our AION encoders +
fusion stack + LoRA-adapted Qwen — `captioner.inference.load_inference_model_from_hub` +
`generate_caption`) and "base" (plain, genuinely out-of-the-box Qwen, no LoRA, native vision
pathway — `captioner.inference.load_qwen_native_vision_model` + `generate_qwen_native_vision_
answer`). A runner that needs both calls `get_backend(..., side="equipped")`, finishes the whole
dataset, lets that backend/model get garbage-collected, then calls `get_backend(...,
side="base")` — the same VRAM-bounding sequencing `inference/compare.py` used on the (now
deleted) `modal-inference` branch, just moved up into the runner's control flow instead of being
buried inside one Modal function body. Not every runner needs both sides: the lightcurve track
only ever uses "equipped" (a raw lightcurve array isn't something a plain out-of-the-box
vision-language model can consume at all — there is no meaningful "base" comparison for it), the
image track uses both.

**Why `.generate()` takes a `raw_inputs` dict, not per-modality kwargs**: matches `generate_
caption`'s existing contract exactly on the equipped side (`{"image": {"pixel_values": ...}}`
etc. — see `captioner.inference.generate_caption`'s docstring for the full per-modality shape).
The base side's contract is narrower and different in kind — a plain vision-language model only
ever consumes one rendered picture, never raw AION-shaped tensors — so a base-backend's
`raw_inputs` is always exactly `{"image": <PIL.Image>}`. This asymmetry is real, not
overlooked: a base backend raises `KeyError` loudly if asked for anything else, rather than
pretending to support modalities it structurally cannot.

**Why `.generate()` returns a string (caption text), never a label**: keeps this file about model
invocation only. Turning that caption into a predicted class belongs to
`eval/metrics/caption_to_label.py` — this is also what lets a future caption-quality eval reuse
this file completely unchanged.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Literal

from omegaconf import DictConfig

Side = Literal["equipped", "base"]


@dataclass
class EvalBackend:
    """Thin wrapper around one already-loaded model (or one already-deployed Modal function) for
    exactly one side. Callers never construct this directly — use `get_backend(...)`.
    """

    side: Side
    _generate: Callable[[dict, str, int], str]

    def generate(self, raw_inputs: dict, question: str, max_new_tokens: int = 128) -> str:
        return self._generate(raw_inputs, question, max_new_tokens)


def get_backend(
    kind: Literal["local", "modal"],
    side: Side,
    cfg: DictConfig,
    *,
    repo_id: str | None = None,
    device: str = "cuda",
    modality_names: list[str] | None = None,
    modal_app_name: str = "astrobridge-eval-backend",
    enable_thinking: bool | None = None,
) -> EvalBackend:
    """The one function every runner calls. `repo_id`/`modality_names` only matter for
    `side="equipped"`; ignored (but harmless to pass) for `side="base"`. `enable_thinking` is the
    reverse — only matters for `side="base"` — see `generate_qwen_native_vision_answer`'s
    docstring for what it actually does and why it defaults to `None` (the model's own default,
    thinking ON) rather than being force-disabled here.
    """
    if kind == "local":
        return _local_backend(
            side, cfg, repo_id=repo_id, device=device, modality_names=modality_names, enable_thinking=enable_thinking,
        )
    if kind == "modal":
        return _modal_backend(side, modal_app_name=modal_app_name, enable_thinking=enable_thinking)
    raise ValueError(f"kind={kind!r} not recognised — expected 'local' or 'modal'.")


# --- local backend ------------------------------------------------------------------------


def _local_backend(
    side: Side,
    cfg: DictConfig,
    *,
    repo_id: str | None,
    device: str,
    modality_names: list[str] | None,
    enable_thinking: bool | None = None,
) -> EvalBackend:
    from captioner.inference import (
        generate_caption,
        generate_qwen_native_vision_answer,
        load_inference_model_from_hub,
        load_qwen_native_vision_model,
    )

    if side == "equipped":
        if repo_id is None:
            raise ValueError("repo_id is required for side='equipped' (the published model to load).")
        model, tokenizer, encoders = load_inference_model_from_hub(
            cfg, repo_id, device=device, modality_names=modality_names,
        )
        out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
        max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

        def _generate(raw_inputs: dict, question: str, max_new_tokens: int) -> str:
            return generate_caption(
                model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt.template, device,
                raw_inputs, max_new_tokens=max_new_tokens, question=question,
            )

        return EvalBackend(side="equipped", _generate=_generate)

    # side == "base"
    vision_model, processor = load_qwen_native_vision_model(cfg, device=device)

    def _generate(raw_inputs: dict, question: str, max_new_tokens: int) -> str:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys "
                f"{list(raw_inputs)}. There is no native-vision equivalent for other modalities."
            )
        return generate_qwen_native_vision_answer(
            vision_model, processor, device, question, raw_inputs["image"], max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    return EvalBackend(side="base", _generate=_generate)


def free_local_backend(backend: EvalBackend) -> None:
    """Call between `get_backend(..., side="equipped")` and `get_backend(..., side="base")` (or
    vice versa) in the SAME process, so the first model's VRAM is actually released before the
    second one loads — otherwise both stay resident simultaneously, doubling peak VRAM for no
    reason (each side is only ever used for the whole dataset in turn, never interleaved).
    `backend` itself is not reusable after this call.
    """
    import torch

    del backend._generate
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- modal backend -------------------------------------------------------------------------
#
# Reuses the exact pattern proven on the (now deleted) `modal-inference` branch's
# `inference/modal_app.py`/`compare.py`: `uv_sync()` (installs from this repo's real
# pyproject.toml/uv.lock — a hand-picked pip_install list once silently missed `pandas`, a
# transitive import) + `add_local_python_source("captioner")` + `add_local_dir("configs",
# remote_path="/configs")` (required because `add_local_python_source` mounts the package at
# `/root/captioner/...`, skipping the local `src/` layer, and `utils/config.py`'s `CONFIG_DIR`
# path-climbing lands on exactly `/configs` from that remote layout).
#
# `modal deploy`/`modal run` look for a variable literally named `app` by default (confirmed
# live: naming it `modal_app` fails deploy with "module 'backend' has no attribute 'app'") — so
# the actual `modal.App` instance below is named `app`, not `modal_app`, despite this section's
# comments still saying "modal_app" in prose for clarity against `inference/modal_app.py`'s name.

import modal  # noqa: E402  (kept below the docstring/local-backend code on purpose — this is the
                # one place Modal-specific imports belong; nothing above this line touches modal)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_sync()
    .add_local_python_source("captioner", "eval")
    .add_local_dir("configs", remote_path="/configs")
)

app = modal.App("astrobridge-eval-backend")

hf_cache_volume = modal.Volume.from_name("astrobridge-hf-cache", create_if_missing=True)

# TODO: still needs a real published Gemma-4/next-generation repo id if this ever targets
# anything other than the current Qwen-based model. For now this is deliberately the real,
# confirmed-live published repo: UniverseTBD/astrobridge-model-v3_qwen (adapter_config.json +
# adapter_model.safetensors + middle.pt confirmed present on HF Hub).
DEFAULT_MODEL_REPO_ID = "UniverseTBD/astrobridge-model-v3_qwen"


@app.function(
    gpu="L4",
    cpu=2.0,
    memory=16384,
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=1800,
)
def _equipped_infer(
    raw_inputs_serialized: dict,
    question: str,
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    modality_names: list[str] | None = None,
    max_new_tokens: int = 128,
) -> str:
    """Runs entirely inside the remote container. `raw_inputs_serialized` must already be in the
    shape `generate_caption` expects (torch tensors / lists — Modal serializes these directly via
    cloudpickle, no manual bytes-encoding needed for this eval use case, unlike modal_app.py's
    CLI which had to accept raw file bytes from a terminal).
    """
    import torch

    from captioner.inference import generate_caption, load_inference_model_from_hub
    from captioner.utils.config import load_config

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, encoders = load_inference_model_from_hub(
        cfg, repo_id, device=device, modality_names=modality_names,
    )
    hf_cache_volume.commit()

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}
    return generate_caption(
        model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt.template, device,
        raw_inputs_serialized, max_new_tokens=max_new_tokens, question=question,
    )


@app.function(
    gpu="L4",
    cpu=2.0,
    memory=16384,
    image=modal_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=1800,
)
def _base_infer(image_bytes: bytes, question: str, max_new_tokens: int = 128, enable_thinking: bool | None = None) -> str:
    """`image_bytes`: a PNG/JPEG-encoded picture, not raw pixel_values — the base model's native
    vision pathway consumes an ordinary image, decoded here inside the remote container.
    `enable_thinking`: see `generate_qwen_native_vision_answer`'s docstring.
    """
    import io

    import torch
    from PIL import Image

    from captioner.inference import generate_qwen_native_vision_answer, load_qwen_native_vision_model
    from captioner.utils.config import load_config

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_model, processor = load_qwen_native_vision_model(cfg, device=device)
    hf_cache_volume.commit()

    image = Image.open(io.BytesIO(image_bytes))
    return generate_qwen_native_vision_answer(
        vision_model, processor, device, question, image, max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )


def _modal_backend(side: Side, *, modal_app_name: str, enable_thinking: bool | None = None) -> EvalBackend:
    """Talks to an already-`modal deploy`ed app — never defines `@app.function` itself (that
    lives above, once, shared by every runner). `modal deploy eval/backend.py` must have been
    run at least once before this works; see the module-level TODO/verification note above.
    Redeploy again after any change to `_equipped_infer`/`_base_infer` (including a new parameter
    like `enable_thinking`) — a running deployment keeps serving whatever code it was deployed
    with until `modal deploy` is re-run.
    """
    if side == "equipped":
        fn = modal.Function.from_name(modal_app_name, "_equipped_infer")

        def _generate(raw_inputs: dict, question: str, max_new_tokens: int) -> str:
            return fn.remote(raw_inputs_serialized=raw_inputs, question=question, max_new_tokens=max_new_tokens)

        return EvalBackend(side="equipped", _generate=_generate)

    fn = modal.Function.from_name(modal_app_name, "_base_infer")

    def _generate(raw_inputs: dict, question: str, max_new_tokens: int) -> str:
        if "image" not in raw_inputs or len(raw_inputs) != 1:
            raise KeyError(
                f"base backend only ever consumes {{'image': <PIL.Image>}} — got keys "
                f"{list(raw_inputs)}."
            )
        import io

        buf = io.BytesIO()
        raw_inputs["image"].save(buf, format="PNG")
        return fn.remote(
            image_bytes=buf.getvalue(), question=question, max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

    return EvalBackend(side="base", _generate=_generate)
