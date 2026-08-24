"""Stage 1: LLM frozen, encoders not loaded (training reads cached embeddings only). Trainable:
projectors, modality_identity, qformer, adapter — exactly `configs/stage1.yaml: trainable`.

Single-node multi-GPU via `accelerate` (DDP): each process loads a full copy of the model
normally (no `device_map="auto"` — that's model-parallel, incompatible with DDP) and
`accelerator.prepare(...)` places it on that process's GPU and wraps it for gradient
synchronization. Launch with `accelerate launch scripts/03_train_stage1.py ...` — see
`captioner/README.md`'s cluster section for the `accelerate config` this expects.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from captioner.data.collate import make_collate_fn
from captioner.data.dataset import CaptionerDataset
from captioner.model.captioner import Captioner, FusionStack
from captioner.train.loop import run_training
from captioner.utils.logging import get_logger
from captioner.utils.seeding import seed_everything

logger = get_logger(__name__)


def _cosine_with_warmup(optimizer, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        import math

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_llm(model_cfg: DictConfig):
    trust_remote_code = bool(model_cfg.llm.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.llm.name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if model_cfg.llm.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    llm = AutoModelForCausalLM.from_pretrained(
        model_cfg.llm.name,
        torch_dtype=torch.bfloat16 if model_cfg.llm.dtype == "bfloat16" else torch.float32,
        attn_implementation=model_cfg.llm.attn_impl,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
    )
    for p in llm.parameters():
        p.requires_grad = False
    return llm, tokenizer


def get_llm_hidden_size(llm) -> int:
    """`Qwen3_5ForConditionalGeneration`-style architectures often nest text config under
    `config.text_config` rather than exposing `hidden_size` flatly — check both rather than
    assume, so a config layout change fails loudly instead of picking up the wrong `d_llm`.
    """
    if hasattr(llm.config, "hidden_size"):
        return int(llm.config.hidden_size)
    if hasattr(llm.config, "text_config") and hasattr(llm.config.text_config, "hidden_size"):
        return int(llm.config.text_config.hidden_size)
    raise AttributeError(
        f"Could not find hidden_size on {type(llm.config).__name__} (checked both "
        "config.hidden_size and config.text_config.hidden_size) — inspect the real config "
        "object for this model and adjust get_llm_hidden_size()."
    )


def run_stage1(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    grad_accum = max(1, int(cfg.batch_size) // int(cfg.micro_batch_size))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)

    manifest = pd.read_parquet(cfg.manifest.parquet)
    captions = pd.read_parquet(cfg.captions.parquet)
    tier_histogram = json.loads(Path(cfg.manifest.stats).read_text())["tier_histogram"]

    llm, tokenizer = build_llm(cfg)
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
    )
    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))

    cache_root = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    train_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "train", tokenizer, cfg.prompt.template)
    val_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "val", tokenizer, cfg.prompt.template)

    collate_fn = make_collate_fn(train_ds.modality_names, out_dims, max_tokens, tokenizer.pad_token_id)
    micro_bs = int(cfg.micro_batch_size)
    train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=micro_bs, shuffle=False, collate_fn=collate_fn)

    trainable_params = [p for _, p in fusion_stack.trainable_named_parameters(list(cfg.trainable))]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    total_steps = (len(train_loader) // grad_accum) * int(cfg.epochs)
    scheduler = _cosine_with_warmup(optimizer, total_steps, float(cfg.schedule.warmup_frac))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    run_dir = Path(cfg.checkpoint.dir)
    result = run_training(
        accelerator, model, train_loader, val_loader, optimizer, scheduler, cfg, run_dir,
        epochs=int(cfg.epochs), early_stop_patience=int(cfg.early_stop.patience),
        tier_histogram=tier_histogram,
    )
    if accelerator.is_main_process:
        logger.info(f"Stage 1 complete: {result}")
