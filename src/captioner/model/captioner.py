"""Fusion stack + sequence assembly against the frozen-or-LoRA LLM decoder.

Iterates over the modality registry (`modality_names`) everywhere — no `if image ... elif
spectra ...` branching (§0). Training never loads raw encoders; it consumes cached embeddings
via the batch dict produced by data/collate.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from captioner.model.adapter import Adapter
from captioner.model.modality_embed import ModalityIdentity
from captioner.model.projectors import ModalityProjector
from captioner.model.qformer import SharedQFormer

IGNORE_INDEX = -100


class FusionStack(nn.Module):
    """Everything upstream of the LLM: projectors -> modality identity -> Q-Former -> adapter.
    This is exactly what gets saved as middle.pt (§11).
    """

    def __init__(
        self,
        modality_out_dims: dict[str, int],
        d_shared: int,
        d_llm: int,
        qformer_cfg: dict,
        projector_hidden_mult: int,
        projector_dropout: float,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_out_dims.keys())
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(out_dim, d_shared, projector_hidden_mult, projector_dropout)
                for name, out_dim in modality_out_dims.items()
            }
        )
        self.modality_identity = ModalityIdentity(self.modality_names, d_shared)
        self.qformer = SharedQFormer(**qformer_cfg)
        self.adapter = Adapter(d_shared, d_llm, dropout=qformer_cfg.get("dropout", 0.1))

    def forward(self, modality_batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """modality_batch: {name: {"tokens": (B, T_m, out_dim), "mask": (B, T_m) bool, True=pad/absent}}
        for every modality in the registry, present or not — absent examples arrive as an
        all-True mask row, never as unmasked zero content (§6).

        Returns adapter(qformer_out): (B, n_queries, d_llm).
        """
        token_chunks: list[Tensor] = []
        mask_chunks: list[Tensor] = []
        for name in self.modality_names:
            entry = modality_batch[name]
            projected = self.projectors[name](entry["tokens"])          # (B, T_m, d_shared)
            projected = self.modality_identity(projected, name)
            token_chunks.append(projected)
            mask_chunks.append(entry["mask"])

        tokens = torch.cat(token_chunks, dim=1)          # (B, T_total, d_shared)
        key_padding_mask = torch.cat(mask_chunks, dim=1)  # (B, T_total) bool, True = ignore

        queries = self.qformer(tokens, key_padding_mask=key_padding_mask)  # (B, n_queries, d_shared)
        return self.adapter(queries)                                       # (B, n_queries, d_llm)

    def trainable_named_parameters(self, groups: list[str]):
        """`groups` are names from configs/stage*.yaml `trainable` / `param_groups` lists —
        projectors, modality_identity, qformer, adapter."""
        modules = {
            "projectors": self.projectors,
            "modality_identity": self.modality_identity,
            "qformer": self.qformer,
            "adapter": self.adapter,
        }
        for group in groups:
            for n, p in modules[group].named_parameters():
                yield f"{group}.{n}", p


class Captioner(nn.Module):
    """Wraps FusionStack + the LLM and performs the inputs_embeds/labels assembly from §6."""

    def __init__(self, fusion_stack: FusionStack, llm: nn.Module, n_queries: int) -> None:
        super().__init__()
        self.fusion_stack = fusion_stack
        self.llm = llm
        self.n_queries = n_queries

    def forward(
        self,
        modality_batch: dict[str, dict[str, Tensor]],
        prompt_ids: Tensor,       # (B, P)
        caption_ids: Tensor,      # (B, C)
        prompt_attn_mask: Tensor | None = None,   # (B, P) 1 for real prompt tokens, 0 for pad
        caption_attn_mask: Tensor | None = None,  # (B, C) 1 for real caption tokens, 0 for pad
    ):
        device = prompt_ids.device
        B, P = prompt_ids.shape
        C = caption_ids.shape[1]

        prefix = self.fusion_stack(modality_batch)                        # (B, n_queries, d_llm)
        embed_fn = self.llm.get_input_embeddings()
        prompt_embeds = embed_fn(prompt_ids)                               # (B, P, d_llm)
        target_embeds = embed_fn(caption_ids)                              # (B, C, d_llm)

        inputs_embeds = torch.cat([prefix, prompt_embeds, target_embeds], dim=1)

        labels = torch.cat(
            [
                torch.full((B, self.n_queries), IGNORE_INDEX, dtype=torch.long, device=device),
                torch.full((B, P), IGNORE_INDEX, dtype=torch.long, device=device),
                caption_ids.masked_fill(caption_attn_mask == 0, IGNORE_INDEX)
                if caption_attn_mask is not None
                else caption_ids,
            ],
            dim=1,
        )

        prefix_mask = torch.ones((B, self.n_queries), dtype=torch.long, device=device)
        prompt_mask = prompt_attn_mask if prompt_attn_mask is not None else torch.ones((B, P), dtype=torch.long, device=device)
        cap_mask = caption_attn_mask if caption_attn_mask is not None else torch.ones((B, C), dtype=torch.long, device=device)
        attention_mask = torch.cat([prefix_mask, prompt_mask, cap_mask], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
