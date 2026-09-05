"""Single-object, no-cache inference: raw image/spectrum arrays in, caption out — for a
brand-new object that was never part of the manifest/embedding-cache pipeline. Distinct from
data/dataset.py's CaptionerDataset, which only reads pre-cached embeddings keyed by object_id.

Runs the same three stages training does (encoders -> FusionStack -> LLM), just live instead of
cached, and with the same autocast(bf16) wrapping eval/groundedness.py needs to reconcile the
fp32 FusionStack against the bf16 LLM (see that module's docstring for why that's required).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from captioner.encoders.registry import build_encoder
from captioner.model.captioner import Captioner, FusionStack
from captioner.publish import filter_missing_lora_keys
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.prompt import human_readable_subset


def load_inference_model(
    cfg: DictConfig,
    checkpoint_dir: Path,
    lora_dir: Path | None,
    device: str = "cuda",
    modality_names: list[str] | None = None,
) -> tuple[Captioner, Any, dict]:
    """`checkpoint_dir` needs middle.pt (e.g. outputs/checkpoints/stage2/best, or stage1's).
    `lora_dir` is the matching lora/ subfolder — pass None for a stage-1-only checkpoint.

    `modality_names`: which encoders to actually build — defaults to every modality in
    `cfg.modalities` (`None`), but pass an explicit subset (e.g. `["image"]`) when a caller only
    ever uses some of them. Confirmed real cost, not just tidiness: building an encoder means
    downloading and loading its own model weights (AION for image/spectra, ATCAT for light
    curve) — building all three when only one is ever used wastes real time and, on a rented GPU,
    real money.

    `lora_dir` holds this repo's own checkpoint format (a raw filtered state_dict at
    lora/adapter.pt — see train/checkpoint.py's save_checkpoint), NOT PEFT's own
    save_pretrained format (adapter_config.json + safetensors). `PeftModel.from_pretrained`
    expects the latter, so the LoRA config is reconstructed here from configs/stage2.yaml and
    the state dict loaded manually — same approach scripts/06_publish_model.py uses to convert
    into the standard format for publishing.
    """
    llm, tokenizer = build_llm(cfg)
    if lora_dir is not None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=float(cfg.lora.dropout),
            target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)
        lora_state = torch.load(Path(lora_dir) / "adapter.pt", map_location="cpu", weights_only=False)
        missing, _unexpected = llm.load_state_dict(lora_state, strict=False)
        real_missing = filter_missing_lora_keys(missing)
        if real_missing:
            raise RuntimeError(
                f"LoRA state from {Path(lora_dir) / 'adapter.pt'} didn't fully load onto a "
                f"freshly LoRA-wrapped {cfg.llm.name} — missing keys: {real_missing[:10]}. "
                "This means either the checkpoint doesn't match configs/stage2.yaml's current "
                "lora.* settings, or the checkpoint is corrupt."
            )
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
    )
    fusion_stack.load_state_dict(
        torch.load(Path(checkpoint_dir) / "middle.pt", map_location="cpu", weights_only=False)
    )

    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(device)
    model.eval()

    names = list(cfg.modalities) if modality_names is None else modality_names
    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in names}
    return model, tokenizer, encoders


def load_inference_model_from_hub(
    cfg: DictConfig, repo_id: str, device: str = "cuda", modality_names: list[str] | None = None
) -> tuple[Captioner, Any, dict]:
    """Loads a model *published* via scripts/06_publish_model.py (e.g. from `make publish`),
    as opposed to `load_inference_model` above which reads this repo's own internal training-
    checkpoint layout directly off disk.

    `modality_names`: see load_inference_model's docstring — same meaning, same real cost
    reason for passing an explicit subset rather than always building every encoder.

    The published repo is genuinely a different format, not just a different location: publish
    converts the raw lora/adapter.pt state_dict into PEFT's own save_pretrained layout
    (adapter_config.json + adapter_model.safetensors), specifically so it loads the standard
    way. That's why this function can call `PeftModel.from_pretrained(llm, repo_id)` directly,
    unlike `load_inference_model`, which has to reconstruct the LoraConfig by hand because the
    local checkpoint format doesn't carry one.

    `middle.pt` (the fusion stack) isn't a PEFT/HF concept, so it's just downloaded as a plain
    file via `hf_hub_download` and loaded the same way as everywhere else in this codebase.
    """
    from huggingface_hub import hf_hub_download
    from peft import PeftModel

    llm, tokenizer = build_llm(cfg)
    llm = PeftModel.from_pretrained(llm, repo_id)
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
    )
    middle_path = hf_hub_download(repo_id=repo_id, filename="middle.pt")
    fusion_stack.load_state_dict(torch.load(middle_path, map_location="cpu", weights_only=False))

    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(device)
    model.eval()

    names = list(cfg.modalities) if modality_names is None else modality_names
    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in names}
    return model, tokenizer, encoders


@torch.no_grad()
def generate_caption(
    model: Captioner,
    tokenizer,
    encoders: dict,
    modality_out_dims: dict[str, int],
    modality_max_tokens: dict[str, int],
    prompt_template: str,
    device: str,
    raw_inputs: dict[str, dict[str, Any]],
    max_new_tokens: int = 128,
    question: str | None = None,
) -> str:
    """`raw_inputs`: {modality_name: encoder-specific batch dict}, only for modalities actually
    present — e.g. {"image": {"pixel_values": ...}} or
    {"spectra": {"flux": ..., "ivar": ..., "mask": ..., "wavelength": ..., "survey": [...]}}, or
    {"lightcurve": {"flux": ..., "flux_err": ..., "time": ..., "mask": ..., "channel_index": ...}}
    (build that one with data/transients_dataset.py's `prepare_lightcurve_arrays`, so live inference
    applies exactly the same detection-window trim and padding the cache was built with).
    See each encoder's `encode()` docstring (encoders/aion_image.py, aion_spectrum.py) for the
    exact field contract. A modality absent from `raw_inputs` is treated as not shown at all —
    true exclusion (all-True mask), never a zero-content placeholder (§6).

    `question`: free-form text used verbatim as the prompt instead of `prompt_template`'s fixed
    "Describe the object shown, using only {modalities}." — LoRA/the fusion stack were only ever
    trained against that one fixed instruction, so this leans entirely on the frozen base LLM's
    own general instruction-following ability generalizing to a different instruction while
    still conditioning on the visual/spectral prefix. Untested territory, not a guarantee —
    that's the actual point of exposing it (see scripts/07_infer.py's --question flag).
    """
    if not raw_inputs:
        raise ValueError("raw_inputs is empty — at least one modality must be provided.")

    shown = frozenset(raw_inputs.keys())
    modality_batch: dict[str, dict[str, torch.Tensor]] = {}
    for name, out_dim in modality_out_dims.items():
        T_m = modality_max_tokens[name]
        tokens = torch.zeros((1, T_m, out_dim), dtype=torch.float32, device=device)
        mask = torch.ones((1, T_m), dtype=torch.bool, device=device)  # True = pad/absent by default
        if name in raw_inputs:
            # AION's real token count depends on the actual input (image pixel dims / spectrum
            # length), not just the `num_encoder_tokens` the encoder was built with — confirmed
            # real, not a bug in the encoder: a 384x384 image or a ~7800-sample spectrum can both
            # produce fewer raw tokens than max_tokens. data/collate.py already pads/truncates to
            # max_tokens per-object during training for exactly this reason; mirror that here
            # instead of requiring an exact match (which training never assumed either).
            raw_tokens = encoders[name].encode(raw_inputs[name]).to(torch.float32)  # (1, T_raw, out_dim)
            n = min(raw_tokens.shape[1], T_m)
            tokens[:, :n] = raw_tokens[:, :n].to(device)
            mask[:, :n] = False
        modality_batch[name] = {"tokens": tokens, "mask": mask}

    prompt_text = question if question is not None else prompt_template.format(modalities=human_readable_subset(shown))
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        prefix = model.fusion_stack(modality_batch)
        prompt_embeds = model.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        gen = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]


def load_qwen_native_vision_model(cfg: DictConfig, device: str = "cuda"):
    """Loads a genuinely separate, plain Qwen3.5 instance for generate_qwen_native_vision_answer
    — NOT reusable with the model load_inference_model_from_hub produces, and this isn't a style
    choice, it's confirmed real: `AutoModelForCausalLM` (what build_llm uses) resolves to
    `Qwen3_5ForCausalLM` for this model_type — a genuinely different, text-only Python class from
    `Qwen3_5ForConditionalGeneration` (what `AutoModelForImageTextToText`/`AutoModelForMultimodal
    LM` resolve to), confirmed by direct inspection of transformers' real model-mapping tables,
    not assumed. `Qwen3_5ForCausalLM` structurally has no vision tower at all — no amount of
    unwrapping LoRA/PEFT changes that, since the limitation is in the base class itself.

    This does mean a second, real set of weights gets loaded (same underlying language-model
    weights as the text-only class — `Qwen3_5ForConditionalGeneration`'s language_model submodule
    is built from the same checkpoint's `config.text_config`, just with a vision tower attached
    alongside it — but a second copy in memory all the same). inference/compare.py loads this
    sequentially, after freeing the text-only model, specifically to keep peak VRAM within one
    GPU's budget rather than needing both loaded simultaneously.

    Returns (model, processor). No LoRA involved at any point — this model is never PEFT-wrapped,
    so there's nothing to disable; it answers purely on Qwen's own pretraining.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    trust_remote_code = bool(cfg.llm.get("trust_remote_code", False))
    processor = AutoProcessor.from_pretrained(cfg.llm.name, trust_remote_code=trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.llm.name,
        torch_dtype=torch.bfloat16 if cfg.llm.dtype == "bfloat16" else torch.float32,
        attn_implementation=cfg.llm.attn_impl,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_qwen_native_vision_answer(model, processor, device: str, question: str, image, max_new_tokens: int = 128) -> str:
    """The "plain Qwen, genuinely out of the box" side of a comparison against generate_caption's
    fully-equipped answer — routed through Qwen's own native multimodal pathway instead of this
    project's AION encoders + fusion stack. Nothing this project trained touches this call.

    `model`: from load_qwen_native_vision_model — a plain `Qwen3_5ForConditionalGeneration`
    instance, never PEFT-wrapped, so there's no LoRA to disable here at all (unlike an earlier
    version of this function, which mistakenly tried to reuse the text-only LoRA-wrapped model —
    see load_qwen_native_vision_model's docstring for why that doesn't work).

    `image`: a PIL.Image — build one from a raw grz array with utils/image.py's grz_to_rgb first.
    Confirmed real, not guessed: Qwen's chat_template.jinja checks for an "image" key in each
    content block (`'image' in item or 'image_url' in item or item.type == 'image'`), matching
    the {"type": "image", "image": ...} shape used below.
    """
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(device)

    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
