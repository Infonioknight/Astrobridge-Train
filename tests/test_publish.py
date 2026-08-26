"""Pure-logic pieces of scripts/06_publish_model.py — the parts that don't need a GPU, real
weights, or network access.
"""
from __future__ import annotations

from captioner.publish import build_model_card, filter_missing_lora_keys


def test_filter_missing_lora_keys_drops_non_lora_keys():
    missing = ["base_model.model.layers.0.q_proj.lora_A.weight", "base_model.model.embed_tokens.weight"]
    assert filter_missing_lora_keys(missing) == ["base_model.model.layers.0.q_proj.lora_A.weight"]


def test_filter_missing_lora_keys_empty_when_nothing_missing():
    assert filter_missing_lora_keys([]) == []


def test_filter_missing_lora_keys_all_lora():
    missing = ["a.lora_A.weight", "b.lora_B.weight"]
    assert filter_missing_lora_keys(missing) == missing


def test_model_card_includes_base_model_and_repo_id():
    state = {"config_hash": "abc123", "quantization": None, "git_sha": "deadbeef", "tier_histogram": {"joint": 5}}
    card = build_model_card("Qwen/Qwen3.5-9B", "org/my-model", state, eval_report=None)

    assert "Qwen/Qwen3.5-9B" in card
    assert "org/my-model" in card
    assert "abc123" in card
    assert "deadbeef" in card
    assert "Not yet run" in card


def test_model_card_includes_eval_report_when_present():
    state = {"config_hash": "x", "quantization": "nf4", "git_sha": "y", "tier_histogram": {}}
    eval_report = {"per_modality": {"image": {"shuffle_test": {"null_result": False}}}}

    card = build_model_card("Qwen/Qwen3.5-9B", "org/my-model", state, eval_report)

    assert "null_result" in card
    assert "false" in card.lower()
    assert "Not yet run" not in card
