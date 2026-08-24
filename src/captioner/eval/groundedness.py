"""Groundedness gate (§8, §9 step 7): the model must be conditioning on modality *content*, not
just modality *presence*. Standard text metrics (BLEU/ROUGE/BERTScore) are secondary and must
never be used for early stopping (§10 pitfall 8) — that stays val_loss, enforced in train/loop.py.
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from captioner.model.captioner import Captioner


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


@torch.no_grad()
def _generate_caption(model: Captioner, tokenizer, batch: dict, device: str, max_new_tokens: int = 64) -> list[str]:
    model.eval()
    fusion_stack = model.fusion_stack
    modality_batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch["modality_batch"].items()}
    prompt_ids = batch["prompt_ids"].to(device)

    prefix = fusion_stack(modality_batch)
    embed_fn = model.llm.get_input_embeddings()
    prompt_embeds = embed_fn(prompt_ids)
    inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    gen = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def shuffle_test(
    model: Captioner, dataset: Dataset, modality: str, tokenizer, device: str, n: int = 200, seed: int = 0
) -> dict:
    """Replace `modality` content with another object's; presence flags UNCHANGED.
    Null result (near-zero edit distance / delta NLL) => the model is not conditioning on that
    modality's content, only its presence.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    edit_distances = []
    for idx in indices:
        ex = dataset[idx]
        if modality not in ex["shown"]:
            continue
        original_caption = tokenizer.decode(ex["caption_ids"], skip_special_tokens=True)

        donor_idx = rng.choice([j for j in range(len(dataset)) if j != idx])
        donor = dataset[donor_idx]
        if donor["modality_arrays"].get(modality) is None:
            continue

        shuffled = dict(ex)
        shuffled["modality_arrays"] = dict(ex["modality_arrays"])
        shuffled["modality_arrays"][modality] = donor["modality_arrays"][modality]

        from captioner.data.collate import collate_batch

        batch = collate_batch(
            [shuffled], dataset.modality_names, dataset.out_dims, dataset.max_tokens, tokenizer.pad_token_id
        )
        generated = _generate_caption(model, tokenizer, batch, device)[0]
        edit_distances.append(_edit_distance(generated, original_caption))

    if not edit_distances:
        return {"modality": modality, "n": 0, "mean_edit_distance": None, "note": "no eligible examples"}

    mean_ed = sum(edit_distances) / len(edit_distances)
    return {
        "modality": modality,
        "n": len(edit_distances),
        "mean_edit_distance": mean_ed,
        "null_result": mean_ed < 3.0,  # near-zero change => not conditioning on content
    }


def ablation_test(model: Captioner, dataset: Dataset, modality: str, tokenizer, device: str, n: int = 200, seed: int = 0) -> dict:
    """Remove the modality entirely (mask it out, even if available). Claims sourced from it
    must disappear from the generated caption, not persist.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    changed = 0
    total = 0
    for idx in indices:
        ex = dataset[idx]
        if modality not in ex["shown"]:
            continue
        total += 1

        with_modality = dict(ex)
        without_modality = dict(ex)
        without_modality["modality_arrays"] = dict(ex["modality_arrays"])
        without_modality["modality_arrays"][modality] = None
        without_modality["shown"] = ex["shown"] - {modality}

        from captioner.data.collate import collate_batch

        batch_with = collate_batch(
            [with_modality], dataset.modality_names, dataset.out_dims, dataset.max_tokens, tokenizer.pad_token_id
        )
        batch_without = collate_batch(
            [without_modality], dataset.modality_names, dataset.out_dims, dataset.max_tokens, tokenizer.pad_token_id
        )
        cap_with = _generate_caption(model, tokenizer, batch_with, device)[0]
        cap_without = _generate_caption(model, tokenizer, batch_without, device)[0]
        if cap_with != cap_without:
            changed += 1

    fraction_changed = changed / total if total else None
    return {
        "modality": modality,
        "n": total,
        "fraction_caption_changed": fraction_changed,
        "null_result": (fraction_changed is not None and fraction_changed < 0.1),
    }


def joint_claim_fraction(captions) -> float:
    """Fraction of claims with len(supporting) > 1 — the 'not a union of facts' number."""
    total = 0
    joint = 0
    for c in captions:
        for claim in c.claims:
            total += 1
            if len(claim.supporting) > 1:
                joint += 1
    return joint / total if total else 0.0
