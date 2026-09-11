"""pick_prompt_variants / build_wrapper_text — the per-example (system, instruction) sampling
that makes the model robust to prompt wording (configs/model.yaml prompt block, RETRAIN_SKETCH.md
Part A). Pure functions, no model.
"""
from __future__ import annotations

import numpy as np

from captioner.utils.prompt import build_wrapper_text, pick_prompt_variants
from tests.conftest import make_prompt_cfg


def test_same_seed_picks_the_same_pair():
    cfg = make_prompt_cfg(
        system_variants=["S1", "S2", "S3", "S4"],
        instruction_variants=["I1", "I2", "I3", "I4", "I5"],
    )
    a = pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(7))
    b = pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(7))
    assert a == b


def test_different_seeds_explore_the_space():
    cfg = make_prompt_cfg(
        system_variants=["S1", "S2", "S3", "S4"],
        instruction_variants=["I1", "I2", "I3", "I4", "I5"],
    )
    seen = {pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(s)) for s in range(200)}
    # over 200 seeds the sampler should touch most of the 4x5 grid, not collapse to one pair
    assert len(seen) >= 12


def test_modalities_placeholder_is_filled_only_in_the_instruction():
    cfg = make_prompt_cfg(
        system_variants=["You are an assistant. {modalities} should NOT appear filled here."],
        instruction_variants=["Describe this {modalities}."],
    )
    system, instruction = pick_prompt_variants(cfg, frozenset({"image", "spectra"}), np.random.default_rng(0))
    assert instruction == "Describe this an image and a spectrum."
    assert "{modalities}" in system  # left untouched — system variants are never formatted


def test_build_wrapper_text_splits_around_the_observation_tag():
    cfg = make_prompt_cfg()
    pre, post = build_wrapper_text(cfg, "SYS", "INSTR")
    assert pre.endswith("<observation>")
    assert "SYS" in pre and "<|im_start|>system" in pre
    assert post.startswith("</observation>")
    assert "INSTR" in post and post.rstrip().endswith("<|im_start|>assistant")
