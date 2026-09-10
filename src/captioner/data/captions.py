"""Caption containers + the leakage validator.

IMAGE-ONLY BRANCH (train-v2-images): `decompose_object` and its keyword tables are gone — with
a single modality there is no cross-modal decomposition. `scripts/01_generate_captions.py` wraps
each `caption_fused` string in one `Claim`/`Caption` directly. `compose_captions` and
`validate_no_leakage` are kept: still exercised (trivially, one modality) and cheap to keep so
re-adding a modality doesn't need a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Claim:
    text: str
    supporting: frozenset[str]     # len > 1 => joint claim; empty => dropped
    kind: Literal["observation", "inference", "relation"]
    provenance: str                # arxiv_id + quote_id where available


@dataclass
class Caption:
    object_id: str
    subset: frozenset[str]
    text: str
    claims: list[Claim] = field(default_factory=list)
    source: Literal["paper_decomposed", "llm_generated", "human_checked"] = "paper_decomposed"
    generator: str = "rule_based_v0"


def compose_captions(object_id: str, claims: list[Claim], available_modalities: frozenset[str]) -> list[Caption]:
    """One Caption per tier the object supports: `single` per present modality, plus `joint` if
    more than one is present. Image-only: always exactly one, `subset=frozenset({"image"})`.
    """
    captions: list[Caption] = []
    for modality in sorted(available_modalities):
        subset = frozenset({modality})
        subset_claims = [c for c in claims if c.supporting == subset]
        if subset_claims:
            text = " ".join(c.text for c in subset_claims)
            captions.append(Caption(object_id=object_id, subset=subset, text=text, claims=subset_claims))

    if len(available_modalities) > 1:
        subset = frozenset(available_modalities)
        subset_claims = [c for c in claims if c.supporting.issubset(subset)]
        if subset_claims:
            text = " ".join(c.text for c in subset_claims)
            captions.append(Caption(object_id=object_id, subset=subset, text=text, claims=subset_claims))

    return captions


def validate_no_leakage(c: Caption) -> list[str]:
    """Violation if any claim.supporting is not a subset of c.subset."""
    violations = []
    for claim in c.claims:
        if not claim.supporting.issubset(c.subset):
            violations.append(
                f"object={c.object_id} tier={sorted(c.subset)}: claim {claim.provenance!r} "
                f"supported by {sorted(claim.supporting)}, which is not a subset of the tier."
            )
    return violations
