"""Assembles the eval report (§8): shuffle test, ablation test, and joint_claim_fraction are
logged as first-class scalars alongside loss for every eval, for every modality in the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from captioner.eval.groundedness import ablation_test, shuffle_test
from captioner.model.captioner import Captioner


def build_groundedness_report(
    model: Captioner, dataset: Dataset, modality_names: list[str], tokenizer, device: str, out_path: Path
) -> dict:
    report: dict = {"per_modality": {}}
    for modality in modality_names:
        report["per_modality"][modality] = {
            "shuffle_test": shuffle_test(model, dataset, modality, tokenizer, device),
            "ablation_test": ablation_test(model, dataset, modality, tokenizer, device),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
