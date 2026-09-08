"""generate_qwen_native_vision_answer is the "plain Qwen, genuinely out of the box" side of the
comparison in inference/compare.py — routed through Qwen's own chat-template/image pathway.
Takes a plain, standalone model directly (from load_qwen_native_vision_model), never a
Captioner/PeftModel — confirmed real, not a style choice: `AutoModelForCausalLM` (what this
project's own trained model is loaded through) resolves to `Qwen3_5ForCausalLM`, a genuinely
different, text-only class from `Qwen3_5ForConditionalGeneration` (what this function needs),
so there's never any LoRA/PEFT wrapper to unwrap for this path in the first place — see
load_qwen_native_vision_model's docstring for the full story.
"""
from __future__ import annotations

import torch

from captioner.inference import generate_qwen_native_vision_answer


class _FakeVisionModel:
    def generate(self, input_ids, pixel_values=None, max_new_tokens=None, do_sample=None, **kw):
        # Real signature also takes pixel_values/attention_mask/image_grid_thw/etc — kw absorbs
        # whatever the processor's chat template produced beyond what this fake bothers to inspect.
        B, _prompt_len = input_ids.shape
        new_tokens = torch.zeros(B, max_new_tokens, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


class _FakeChatTemplateResult(dict):
    """apply_chat_template's real return value supports .to(device); a plain dict doesn't."""

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.seen_messages = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, return_tensors):
        self.seen_messages = messages
        return _FakeChatTemplateResult(input_ids=torch.randint(0, 32, (1, 4)))

    def decode(self, ids, skip_special_tokens=True):
        return f"n_new_tokens={ids.shape[0]}"


def test_generates_an_answer():
    out = generate_qwen_native_vision_answer(
        _FakeVisionModel(), _FakeProcessor(), "cpu", "What is this?", image=object(), max_new_tokens=3,
    )
    assert out == "n_new_tokens=3"


def test_message_shape_matches_what_the_chat_template_checks_for():
    """chat_template.jinja checks `'image' in item or 'image_url' in item or item.type == 'image'`
    — confirmed by reading the real template file. The content block built here must satisfy it.
    """
    processor = _FakeProcessor()
    sentinel_image = object()

    generate_qwen_native_vision_answer(
        _FakeVisionModel(), processor, "cpu", "What is this?", image=sentinel_image, max_new_tokens=3,
    )

    [message] = processor.seen_messages
    assert message["role"] == "user"
    image_blocks = [c for c in message["content"] if c.get("type") == "image"]
    text_blocks = [c for c in message["content"] if c.get("type") == "text"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image"] is sentinel_image
    assert text_blocks == [{"type": "text", "text": "What is this?"}]
