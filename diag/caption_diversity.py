#!/usr/bin/env python
"""Generate captions for real test objects and report how many are distinct.

The number the v5/v6 gate never had. v6 published 82 unique captions out of 397 (0.207), top
caption 40.1% of the set. Healthy: distinct_fraction > 0.8, top1_share < 0.05.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from captioner.data.collate import collate_batch  # noqa: E402
from captioner.data.dataset import CaptionerDataset  # noqa: E402
from captioner.eval.groundedness import _generate_caption  # noqa: E402
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm  # noqa: E402
from captioner.train.stage1 import build_llm, get_llm_hidden_size  # noqa: E402
from captioner.utils.config import load_config  # noqa: E402


def main() -> int:
    ckpt = Path(sys.argv[1])
    lora = Path(sys.argv[2]) if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else None
    cfg = load_config("base", "data", "modalities", "model", "stage2", argv=[])
    llm, tokenizer = build_llm(cfg)
    if lora is not None:
        from peft import LoraConfig, get_peft_model

        llm = get_peft_model(
            llm,
            LoraConfig(
                r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
                lora_dropout=float(cfg.lora.dropout),
                target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
            ),
        )
        llm.load_state_dict(
            torch.load(lora / "adapter.pt", map_location="cpu", weights_only=False), strict=False
        )
    fs = FusionStack(
        {n: int(c.out_dim) for n, c in cfg.modalities.items()},
        int(cfg.d_shared), get_llm_hidden_size(llm), dict(cfg.qformer),
        int(cfg.projector.hidden_mult), float(cfg.projector.dropout),
        adapter_target_norm=llm_embedding_norm(llm),
    )
    fs.load_state_dict(torch.load(ckpt / "middle.pt", map_location="cpu", weights_only=False))
    model = Captioner(fs, llm, n_queries=int(cfg.qformer.n_queries)).to("cuda").eval()

    ds = CaptionerDataset(
        pd.read_parquet(cfg.manifest.parquet), pd.read_parquet(cfg.captions.parquet),
        cfg, Path(cfg.cache.out_dir), "test", tokenizer, cfg.prompt,
    )
    # Break spectra out by survey. DESI and SDSS are different AION modalities
    # (tok_spectrum_desi vs tok_spectrum_sdss) and, before v7, DESI was 331/334 in test and 2 in
    # train — averaging them into one "spectra" number hid exactly that. Reported separately so
    # it cannot hide again.
    manifest = pd.read_parquet(cfg.manifest.parquet)
    survey_of = dict(zip(manifest["object_id"], manifest.get("survey", pd.Series(dtype=object))))

    by_mod: dict[str, list[str]] = {}
    for i in range(min(len(ds), 600)):
        ex = ds[i]
        if not ex["shown"]:
            continue
        mod = sorted(ex["shown"])[0]
        survey = survey_of.get(ex["object_id"])
        key = f"{mod}/{survey}" if mod == "spectra" and isinstance(survey, str) else mod
        if len(by_mod.setdefault(key, [])) >= 60:
            continue
        batch = collate_batch([ex], ds.modality_names, ds.out_dims, ds.max_tokens, tokenizer.pad_token_id)
        by_mod[key].append(_generate_caption(model, tokenizer, batch, "cuda")[0].strip())

    failed = False
    print(f"\n  {'modality':<18}{'n':>5}{'distinct':>10}{'top1':>9}   verdict")
    for mod, caps in sorted(by_mod.items()):
        if not caps:
            continue
        c = Counter(caps)
        distinct = len(c) / len(caps)
        top1 = c.most_common(1)[0][1] / len(caps)
        # top1 has a hard floor of 1/n, so a flat 0.05 cutoff fails small groups spuriously.
        ok = distinct >= 0.8 and top1 <= max(0.05, 1.5 / len(caps))
        failed |= not ok
        print(f"  {mod:<18}{len(caps):>5}{distinct:>10.3f}{top1:>9.3f}   {'ok' if ok else 'COLLAPSED'}")
        print(f"      most common: {c.most_common(1)[0][0][:140]}")
    print("\n  reference — published v6: distinct 0.207, top1 0.401")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
