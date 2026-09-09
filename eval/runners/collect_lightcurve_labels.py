#!/usr/bin/env python
"""Step 1/2 of the lightcurve eval, same two-step shape as the image track
(`collect_image_labels.py` / `score_image_eval.py`): run both models over a stratified YSE sample
and save their raw text answers — the compute-intensive step, meant to run once per sample.
Scoring (`score_lightcurve_eval.py`) is pure post-processing over this file's output: no GPU, no
Modal, no re-running the model.

**Both models, not equipped-only** (a change from this track's original, older design — see
`eval.datasets.lightcurve_yse`'s module docstring): the base model gets `render_lightcurve_plot`'s
rendered flux-vs-time PNG (`{"image": <PIL.Image>}`, exactly the image track's base-side contract),
the equipped model gets the raw `atcat_*` arrays via `build_raw_inputs_lightcurve` (or
`build_raw_inputs_with_image` under `--track lightcurve_plus_image`) — genuinely different inputs
per side, same as the image track's `image_rgb` vs `image_bands`.

**Digit-code answer format**, same reasoning and same confirmed-live finding as the image track
(`eval.datasets.image_galaxy10.CLASS_CODE_PROMPT`'s docstring): free-text label matching produced
unreliable compliance, so both sides answer with `SN_CLASS_CODES`' digit code instead.

**Every random decision is seeded and saved**: the sample itself (`--seed`, `--n`,
`--min-per-class`) is recorded in the output JSON, and greedy decoding (`do_sample=False`) makes
generation itself deterministic given the same weights/hardware.

Usage:
    uv run python -m eval.runners.collect_lightcurve_labels --n 90 --seed 0
    uv run python -m eval.runners.collect_lightcurve_labels --n 90 --seed 0 --backend modal
    uv run python -m eval.runners.collect_lightcurve_labels --track lightcurve_plus_image --backend modal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import free_local_backend, get_backend
from eval.datasets.lightcurve_yse import (
    SN_CLASS_CODE_PROMPT,
    SN_CLASS_CODES,
    build_raw_inputs_lightcurve,
    build_raw_inputs_with_image,
    load_host_image_table,
    load_lightcurve_table,
    render_lightcurve_plot,
    stratified_sample,
)

logger = get_logger(__name__)

# Same reasoning as collect_image_labels.py's identical constants: the base model needs real room
# to reach an answer (even with enable_thinking=False leaving a 1-token close tag), equipped needs
# much less — a tight equipped budget doubles as a signal, not just a cost-saver.
DEFAULT_BASE_MAX_NEW_TOKENS = 40
DEFAULT_EQUIPPED_MAX_NEW_TOKENS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v3_qwen")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", choices=["lightcurve_only", "lightcurve_plus_image"], default="lightcurve_only")
    parser.add_argument("--n", type=int, default=None, help="total sample size across all 3 classes; omit to use the whole 266-object eval set")
    parser.add_argument("--min-per-class", type=int, default=3, help="SN Ibc only has 15 objects total in this eval set, so keep this low")
    parser.add_argument("--seed", type=int, default=0, help="the ONE seed that determines the whole sample (irrelevant if --n is omitted)")
    parser.add_argument("--question", default=SN_CLASS_CODE_PROMPT)
    parser.add_argument("--base-max-new-tokens", type=int, default=DEFAULT_BASE_MAX_NEW_TOKENS)
    parser.add_argument("--equipped-max-new-tokens", type=int, default=DEFAULT_EQUIPPED_MAX_NEW_TOKENS)
    parser.add_argument(
        "--base-enable-thinking", action="store_true", default=False,
        help="Off by default — see collect_image_labels.py's identical flag: leaving Qwen3.5's "
             "own reasoning block open was confirmed live to make it truncate before ever "
             "reaching a digit code, on the image track's identical prompt shape.",
    )
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/raw_generations/yse_<track>_seed<seed>_n<n>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    lc_table = load_lightcurve_table()
    if args.track == "lightcurve_plus_image":
        image_table = load_host_image_table()
        lc_table = lc_table.merge(image_table, on=["object_id", "class_label"], how="inner", suffixes=("", "_img"))
        logger.info(f"{len(lc_table)} objects have both lightcurve and host-image data.")

    if args.n is not None:
        lc_table = stratified_sample(lc_table, args.n, args.seed, args.min_per_class)
    logger.info(f"Evaluating {len(lc_table)} objects (track={args.track!r}).")

    # --- Side 1: base model, over the whole sample, via a rendered lightcurve plot -----------
    base_backend = get_backend(
        args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=args.base_enable_thinking,
    )
    base_answers: dict[str, str] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [base]"):
        image = render_lightcurve_plot(row)
        base_answers[row["object_id"]] = base_backend.generate(
            {"image": image}, args.question, args.base_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(base_backend)

    # --- Side 2: our equipped pipeline, over the whole sample, via raw atcat_* arrays --------
    modality_names = ["image", "lightcurve"] if args.track == "lightcurve_plus_image" else ["lightcurve"]
    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=modality_names,
    )
    equipped_answers: dict[str, str] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [equipped]"):
        raw_inputs = (
            build_raw_inputs_with_image(row, row, cfg) if args.track == "lightcurve_plus_image"
            else build_raw_inputs_lightcurve(row, cfg)
        )
        equipped_answers[row["object_id"]] = equipped_backend.generate(
            raw_inputs, args.question, args.equipped_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in lc_table.iterrows():
        oid = row["object_id"]
        objects.append({
            "object_id": oid,
            "label_name": row["class_label"],
            "base_answer": base_answers[oid],
            "equipped_answer": equipped_answers[oid],
        })

    output = {
        "dataset": "BuildNg/astrobridge-yse-test-dataset-v2",
        "track": args.track,
        "repo_id": args.repo_id,
        "question": args.question,
        # "digit_code" tells score_lightcurve_eval.py to parse via predict_label_from_code
        # (against class_codes below), not predict_label's free-text keyword matching.
        "answer_format": "digit_code",
        "class_codes": SN_CLASS_CODES,
        "base_max_new_tokens": args.base_max_new_tokens,
        "equipped_max_new_tokens": args.equipped_max_new_tokens,
        "base_enable_thinking": args.base_enable_thinking,
        "sampling": {"n": args.n, "min_per_class": args.min_per_class, "seed": args.seed},
        "objects": objects,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        n_desc = str(args.n) if args.n is not None else "all"
        out_path = Path(f"outputs/eval/raw_generations/yse_{args.track}_seed{args.seed}_n{n_desc}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(objects)} raw generations to {out_path}")


if __name__ == "__main__":
    main()
