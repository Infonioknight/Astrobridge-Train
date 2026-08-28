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
    cfg: DictConfig, checkpoint_dir: Path, lora_dir: Path | None, device: str = "cuda"
) -> tuple[Captioner, Any, dict]:
    """`checkpoint_dir` needs middle.pt (e.g. outputs/checkpoints/stage2/best, or stage1's).
    `lora_dir` is the matching lora/ subfolder — pass None for a stage-1-only checkpoint.
    Only builds encoders for modalities actually requested at call time — see generate_caption.

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

    encoders = {name: build_encoder(name, cfg.modalities[name], device=device) for name in cfg.modalities}
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
