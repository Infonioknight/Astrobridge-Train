"""Dataset: samples the TARGET SUBSET first, then intersects with availability (§6) — sampling
per-modality independently would under-sample exactly the subsets that matter (joint).

Iterates over `self.modality_names` (from configs/modalities.yaml) everywhere; adding a third
modality requires no change here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import Dataset

from captioner.data.cache import assert_cache_matches_config, cache_dir_for, load_cache_index
from captioner.encoders.registry import encoder_hash, encoder_spec


def _human_readable_subset(subset: frozenset[str]) -> str:
    names = sorted(subset)
    if len(names) == 1:
        return f"a {names[0]}" if names[0] != "image" else "an image"
    return " and ".join(names)


class ModalityCacheReader:
    """Lazily mmaps npy shards for one modality and returns (T, out_dim) float32 arrays by object_id."""

    def __init__(self, cache_root: Path, modality: str, spec_hash: str) -> None:
        self.dir = cache_dir_for(cache_root, modality, spec_hash)
        self.index = load_cache_index(cache_root, modality, spec_hash).set_index("object_id")
        self._shards: dict[str, np.ndarray] = {}

    def get(self, object_id: str) -> np.ndarray:
        row = self.index.loc[object_id]
        shard_file = row["shard_file"]
        if shard_file not in self._shards:
            self._shards[shard_file] = np.load(self.dir / shard_file, mmap_mode="r")
        arr = self._shards[shard_file][int(row["row_in_shard"])]
        return np.asarray(arr, dtype=np.float32)


class CaptionerDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        captions: pd.DataFrame,
        modalities_cfg: DictConfig,
        cache_root: Path,
        split: str,
        tokenizer,
        prompt_template: str,
        max_caption_tokens: int = 128,
    ) -> None:
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        self.captions = captions
        self.modality_names = list(modalities_cfg.modalities.keys())
        self.modality_cfg = modalities_cfg.modalities
        self.out_dims = {n: int(modalities_cfg.modalities[n].out_dim) for n in self.modality_names}
        self.max_tokens = {n: int(modalities_cfg.modalities[n].max_tokens) for n in self.modality_names}

        self.readers: dict[str, ModalityCacheReader] = {}
        for name in self.modality_names:
            spec = encoder_spec(self.modality_cfg[name])
            spec_hash = encoder_hash(spec)
            index_df = load_cache_index(cache_root, name, spec_hash)
            assert_cache_matches_config(index_df, self.modality_cfg[name])
            self.readers[name] = ModalityCacheReader(cache_root, name, spec_hash)

        self.subset_weights: list[tuple[frozenset[str], float]] = [
            (frozenset(entry["subset"]), float(entry["weight"]))
            for entry in modalities_cfg.dropout.subset_weights
        ]

        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_caption_tokens = max_caption_tokens

        self._captions_by_key = {
            (row["object_id"], frozenset(row["subset"])): row["text"]
            for _, row in captions.iterrows()
        }

    def __len__(self) -> int:
        return len(self.manifest)

    def _availability(self, row: pd.Series) -> frozenset[str]:
        present = set()
        for name in self.modality_names:
            flag_col = f"has_{name}"
            if flag_col in row and bool(row[flag_col]):
                present.add(name)
        return frozenset(present)

    def _sample_target_subset(self, available: frozenset[str], rng: np.random.Generator) -> frozenset[str]:
        """Sample the target subset first from configured weights, restricted to subsets that
        are actually available, then renormalize — never sample per-modality independently.
        """
        candidates = [(s, w) for s, w in self.subset_weights if s.issubset(available)]
        if not candidates:
            return frozenset()
        weights = np.array([w for _, w in candidates], dtype=np.float64)
        weights = weights / weights.sum()
        idx = rng.choice(len(candidates), p=weights)
        return candidates[idx][0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        object_id = row["object_id"]
        available = self._availability(row)

        rng = np.random.default_rng(hash((object_id, idx)) & 0xFFFFFFFF)
        shown = self._sample_target_subset(available, rng)

        modality_arrays: dict[str, np.ndarray | None] = {}
        for name in self.modality_names:
            if name in shown:
                modality_arrays[name] = self.readers[name].get(object_id)
            else:
                modality_arrays[name] = None

        caption_text = self._captions_by_key.get((object_id, shown), "")
        prompt_text = self.prompt_template.format(modalities=_human_readable_subset(shown))

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        caption_ids = self.tokenizer(
            caption_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_caption_tokens,
            return_tensors="pt",
        )["input_ids"][0]

        return {
            "object_id": object_id,
            "shown": shown,
            "modality_arrays": modality_arrays,
            "prompt_ids": prompt_ids,
            "caption_ids": caption_ids,
        }
