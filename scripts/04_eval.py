#!/usr/bin/env python
"""Groundedness gate (§9 step 7 / §9 step 8): shuffle + ablation tests must be non-null for
every modality before stage 2 is allowed to start, and again after stage 2 for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel

from captioner.data.dataset import CaptionerDataset
from captioner.eval.report import build_groundedness_report
from captioner.model.captioner import Captioner, FusionStack
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. outputs/checkpoints/stage1/best")
    parser.add_argument("--lora-dir", default=None, help="outputs/checkpoints/stage2/best/lora, if evaluating stage 2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/eval/groundedness_report.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model")
    manifest = pd.read_parquet(cfg.manifest.parquet)
    captions = pd.read_parquet(cfg.captions.parquet)

    llm, tokenizer = build_llm(cfg)
    if args.lora_dir:
        llm = PeftModel.from_pretrained(llm, args.lora_dir)
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
        torch.load(Path(args.checkpoint_dir) / "middle.pt", map_location="cpu", weights_only=False)
    )
    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(cfg.get("device", "cuda"))

    cache_root = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    dataset = CaptionerDataset(manifest, captions, cfg, cache_root, args.split, tokenizer, cfg.prompt.template)

    report = build_groundedness_report(
        model, dataset, dataset.modality_names, tokenizer, cfg.get("device", "cuda"), Path(args.out)
    )
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
