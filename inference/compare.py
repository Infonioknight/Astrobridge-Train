"""Batch, two-sided comparison: our fully-equipped pipeline vs. plain, genuinely-out-of-the-box
Qwen — same uniform prompt across every image, written out to one JSON file.

Side 1 ("equipped"): AION image encoder -> fusion stack -> LoRA-adapted Qwen, our own custom
    visual-prefix injection (captioner.inference.generate_caption).
Side 2 ("plain Qwen"): the same grz array converted to an ordinary RGB picture (utils/image.py's
    grz_to_rgb — same Lupton composite the standalone review notebook uses), fed through Qwen's
    own native multimodal chat-template pathway. Nothing this project trained touches this side
    at all — it answers purely on the base model's own pretraining plus what it can see in the
    picture.

**The two sides use genuinely different model objects, loaded sequentially, not one shared
model.** This wasn't the original plan — earlier versions of this file tried to reuse one loaded
model for both sides via PEFT's disable_adapter(), which seemed right but wasn't: confirmed real,
by inspecting transformers' actual model-mapping tables, that `AutoModelForCausalLM` (what side
1's model is loaded through) resolves to `Qwen3_5ForCausalLM` — a genuinely different, text-only
Python class from `Qwen3_5ForConditionalGeneration` (what side 2 needs), not just a LoRA-wrapping
difference. See captioner.inference.load_qwen_native_vision_model's docstring for the full story,
including why this doesn't make the comparison unfair (same underlying language-model weights
either way, same checkpoint — the two classes just expose different amounts of it).
Consequence: side 1 runs for the *entire* batch first, its weights are freed, then side 2 loads
and runs for the entire batch — this keeps peak VRAM to one ~9B model at a time (fits an L4's
24GB) rather than needing both loaded simultaneously, at the cost of two full model loads per
`modal run` instead of one (still once per *batch*, not once per image).

Output JSON shape:
    {
      "prompt": "<the one uniform prompt used for every image>",
      "outputs": {
        "image_01.npy": ["<plain Qwen answer>", "<our equipped model answer>"],
        "image_02.npy": ["<plain Qwen answer>", "<our equipped model answer>"],
        ...
      }
    }

Same Modal mechanics as modal_app.py (read that file's docstring first if unfamiliar — the
`image`/`app`/`infer`/`main` pieces mean the same thing here). This file always runs both sides
for every image — that's the entire point — so expect meaningfully more billed GPU time than a
single-sided inference/modal_app.py call (two full model loads, not one, plus generation for
every image on both sides). See inference/README.md's "Cost" section for real per-second numbers.

Usage:
    modal run inference/compare.py
    modal run inference/compare.py --images-glob "test_subjects/image_*.npy" \\
        --prompt "Describe the object shown in this image." --output inference/comparison_results.json
"""
from __future__ import annotations

import modal

# Deliberately duplicated from modal_app.py rather than imported from it: `inference/` has no
# __init__.py, and `modal run` loads its target file directly (as `__main__`), so a cross-file
# import here (`from inference.modal_app import ...`) isn't something to rely on without a real
# Modal account to test it against — this file stays fully self-contained instead. Keep these
# three definitions in sync with modal_app.py's if either changes.
MODEL_REPO_ID = "UniverseTBD/astrobridge-captioner-v3"

DEFAULT_PROMPT = "Describe the object shown in this image."

app = modal.App("astrobridge-captioner-compare")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Installs from THIS repo's real pyproject.toml + uv.lock (frozen — exact pinned versions),
    # not a hand-picked list — see modal_app.py's `image` definition for why (a real bug: a
    # manually-curated list missed `pandas`, needed only to *import* captioner.train.stage1's
    # module, transitively pulled in by captioner.inference, regardless of which specific name
    # is used from it). `pillow` (needed for grz_to_rgb's PIL.Image output) is already a real
    # pyproject.toml dependency now, so it comes along automatically too.
    .uv_sync()
    .add_local_python_source("captioner")
    # `configs/` isn't part of the `captioner` package itself — see modal_app.py's `image` for
    # the full explanation (confirmed real: load_config() otherwise fails looking for
    # '/configs/base.yaml', because add_local_python_source mounts the package at
    # /root/captioner/... and utils/config.py's path math lands on exactly /configs from there).
    .add_local_dir("configs", remote_path="/configs")
)

hf_cache_volume = modal.Volume.from_name("astrobridge-hf-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    cpu=2.0,
    memory=16384,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=1800,  # a full batch of images can run well past modal_app.py's single-call 600s
)
def infer(
    images: list[tuple[str, bytes]],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 128,
) -> dict:
    """`images`: (name, .npy file bytes) pairs — name becomes the JSON key, e.g. "image_03.npy".
    Runs entirely inside the remote container.

    Loads and runs the two sides SEQUENTIALLY, not interleaved per image — confirmed real reason,
    not just tidiness: side 1's model (our equipped pipeline) and side 2's model (plain Qwen,
    native vision) are genuinely different Python classes with their own separate weights in
    memory (see generate_qwen_native_vision_answer's docstring for why one loaded model can't
    serve both). Running all of side 1 across the whole batch, freeing its VRAM, then loading
    side 2 for the whole batch keeps peak memory to one model at a time instead of needing both
    loaded simultaneously — the difference between fitting on a single L4 (24GB) and needing a
    much larger, more expensive GPU just to hold two ~9B models at once.

    Returns {"prompt": prompt, "outputs": {name: [plain_qwen_answer, equipped_answer]}}.
    """
    import gc
    import io

    import numpy as np
    import torch

    from captioner.inference import (
        generate_caption,
        generate_qwen_native_vision_answer,
        load_inference_model_from_hub,
        load_qwen_native_vision_model,
    )
    from captioner.utils.config import load_config
    from captioner.utils.image import grz_to_rgb

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_arrays: dict[str, np.ndarray] = {
        name: np.load(io.BytesIO(npy_bytes)) for name, npy_bytes in images  # (3, H, W), g/r/z order
    }

    # --- Side 1: our fully-equipped pipeline, across the whole batch ---
    # Only ever shows an image in this script — no need to also download/load the spectra or
    # light-curve encoders, which would otherwise happen unconditionally.
    model, tokenizer, encoders = load_inference_model_from_hub(
        cfg, MODEL_REPO_ID, device=device, modality_names=["image"],
    )
    hf_cache_volume.commit()  # see modal_app.py's infer() for why this matters

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    equipped_answers: dict[str, str] = {}
    for name, image_array in image_arrays.items():
        equipped_answers[name] = generate_caption(
            model, tokenizer, encoders, out_dims, max_tokens, cfg.prompt.template, device,
            raw_inputs={"image": {"pixel_values": torch.from_numpy(image_array).unsqueeze(0)}},
            max_new_tokens=max_new_tokens, question=prompt,
        )
        print(f"[{name}] equipped-model side done.")

    # Free side 1's weights before loading side 2's — see docstring above for why this matters.
    del model, tokenizer, encoders
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Side 2: plain Qwen, native vision, across the whole batch ---
    vision_model, processor = load_qwen_native_vision_model(cfg, device=device)
    hf_cache_volume.commit()

    outputs: dict[str, list[str]] = {}
    for name, image_array in image_arrays.items():
        rgb_image = grz_to_rgb(image_array)
        plain_qwen_answer = generate_qwen_native_vision_answer(
            vision_model, processor, device, prompt, rgb_image, max_new_tokens=max_new_tokens,
        )
        outputs[name] = [plain_qwen_answer, equipped_answers[name]]
        print(f"[{name}] plain-Qwen side done.")

    return {"prompt": prompt, "outputs": outputs}


@app.local_entrypoint()
def main(
    images_glob: str = "test_subjects/image_*.npy",
    prompt: str = DEFAULT_PROMPT,
    output: str = "inference/comparison_results.json",
    max_new_tokens: int = 128,
):
    """Runs on YOUR machine — collects matching local files into bytes, sends the whole batch to
    the remote GPU in one call, writes the JSON result locally once it comes back.
    """
    import glob
    import json
    from pathlib import Path

    paths = sorted(glob.glob(images_glob))
    if not paths:
        raise SystemExit(f"No files matched --images-glob {images_glob!r}")

    images = [(Path(p).name, open(p, "rb").read()) for p in paths]
    print(f"Sending {len(images)} image(s) to Modal: {[name for name, _ in images]}")

    result = infer.remote(images=images, prompt=prompt, max_new_tokens=max_new_tokens)

    Path(output).write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result['outputs'])} comparison(s) to {output}")
