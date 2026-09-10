#!/usr/bin/env python
"""Quick, manual prompt-engineering tool for the lightcurve/SN-typing track — the lightcurve
analog of `eval/prompt_playground.py` (see that file's docstring for the general idea: load both
models once, run them against a couple of saved test objects, print raw answers, edit
`LC_OOB_PROMPT`/`LC_EQUIPPED_PROMPT` below and re-run). NOT part of the formal eval bench — no
sampling, no metrics, no report.

**Why this exists, specifically**: unlike the image track's `CLASS_CODE_PROMPT` (confirmed live,
via `eval/prompt_playground.py`, to get both models to comply), `eval.datasets.lightcurve_yse.
SN_CLASS_CODE_PROMPT` has never actually been run against either real model — it was written by
pattern-matching the image prompt's shape, not tuned against real generations. Two real unknowns
specific to this track, not present on the image side: (1) the base model sees a *rendered plot*
(`render_lightcurve_plot`), a much less familiar picture type than an actual galaxy photo, so
compliance could be worse for that reason alone; (2) the equipped model's LoRA/fusion stack has
zero training exposure to any instruction beyond the fixed captioning one (`configs/model.yaml`),
same caveat as images but untested for this modality. Run this before trusting a full
`collect_lightcurve_labels.py` run, the same way the image playground was used before trusting
`CLASS_CODE_PROMPT`.

**Test objects reused from `test_subjects/`, not re-downloaded**: `lightcurve_01.npz`..`05.npz`
are real ZTF light curves from the *training* set (`BuildNg/astrobridge-transients-dataset`), not
the YSE eval set — fine for prompt engineering (only format compliance matters here, not measuring
real eval accuracy), but don't mistake a good result here for an actual eval number; run the real
collect/score scripts against `BuildNg/astrobridge-yse-test-dataset-v2` for that.

Usage:
    uv run python -m eval.lightcurve_prompt_playground
    uv run python -m eval.lightcurve_prompt_playground --backend modal   # needs `modal deploy eval/backend.py` first
    uv run python -m eval.lightcurve_prompt_playground --mode descriptive --backend modal
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from captioner.data.transients_dataset import load_transient_captions
from captioner.utils.config import load_config
from eval.backend import free_local_backend, get_backend
from eval.datasets.lightcurve_yse import (
    SN_CLASS_CODE_PROMPT,
    SN_CLASS_CODES,
    SN_LABELS,
    build_raw_inputs_lightcurve,
    render_lightcurve_plot,
)
from eval.metrics.caption_to_label import predict_label_from_code

# The training set, not the YSE eval set — see TEST_LIGHTCURVES' comment: the 5 saved test
# objects are real training-set ZTF light curves specifically so a real ground-truth caption
# exists for them at all (BuildNg/astrobridge-yse-test-dataset-v2 has no caption column, only
# class_label — see eval.datasets.lightcurve_yse's module docstring for that dataset's real
# schema). Confirmed live: all 5 object_ids below have a real transient_caption in this repo.
TRAIN_HF_PATH = "BuildNg/astrobridge-transients-dataset"

# --- Class code mapping -----------------------------------------------------------------------
# The mapping and the prompt itself live in eval/datasets/lightcurve_yse.py (SN_CLASS_CODES,
# SN_CLASS_CODE_PROMPT) — once a wording is confirmed working here, move any changes back there so
# the formal eval bench (collect_lightcurve_labels.py/score_lightcurve_eval.py) uses the exact
# same prompt, kept in one place rather than duplicated between playground and real pipeline.

# --- Edit these two and re-run --------------------------------------------------------------
LC_OOB_PROMPT = SN_CLASS_CODE_PROMPT
LC_EQUIPPED_PROMPT = LC_OOB_PROMPT  # start identical; diverge once you see how each model actually responds

# Same reasoning as the image playground's identical constants: base needs real room to reach an
# answer at all (even with LC_OOB_ENABLE_THINKING=False leaving a 1-token close tag), equipped
# should need much less — if it doesn't, that itself is the finding.
LC_OOB_MAX_NEW_TOKENS = 40
LC_EQUIPPED_MAX_NEW_TOKENS = 8

# Same confirmed-real finding as the image playground (Qwen/Qwen3.5-9B's chat template opens an
# empty <think> block by default) — only affects the base side, which is the only side that goes
# through Qwen's chat template at all (see captioner.inference.generate_caption).
LC_OOB_ENABLE_THINKING = False
# ---------------------------------------------------------------------------------------------

# --- Descriptive mode (--mode descriptive) ----------------------------------------------------
# Free-form caption instead of a digit code, compared directly against each object's real
# ground-truth transient_caption (TRAIN_HF_PATH above) — a caption-QUALITY check, not a
# classification-accuracy check: does the model's description of the light curve's actual shape
# (rise/peak/decline, timescale, band behavior) resemble the real one, regardless of whether it
# also states the right SN type.
#
# Equipped is DELIBERATELY given a differently-worded instruction from the literal trained
# template (configs/model.yaml's "Describe the object shown, using only {modalities}."), not that
# exact phrasing — the risk with using the exact trained wording is that it could just trigger
# memorized/templated captioning jargon regardless of the actual input (the fixed-instruction
# fine-tuning risk this whole eval bench exists to probe), which would look like "it works" while
# actually testing nothing. A semantically-equivalent but differently-phrased instruction is the
# real test of generalization: if equipped only produces sensible, object-specific output under
# the exact trained words and reverts to boilerplate the moment the phrasing changes, that itself
# is a finding (instruction-brittleness), not a null result.
LC_DESCRIPTIVE_EQUIPPED_PROMPT = (
    "What does this time-series data tell you about the object's behavior?"
)
LC_DESCRIPTIVE_OOB_PROMPT = (
    "Briefly analyze and describe the light curve shown in this image: its brightness evolution "
    "over time (rise, peak, decline), approximate timescale, and what type of transient it "
    "might be."
)
# Captions need real room, unlike a 1-token digit code — both sides get the same generous budget
# here (contrast with the digit-code mode's deliberately asymmetric OOB=40/EQUIPPED=8).
LC_DESCRIPTIVE_MAX_NEW_TOKENS = 150
# ---------------------------------------------------------------------------------------------

TEST_LIGHTCURVES = [
    ("test_subjects/lightcurve_01.npz", "ZTF18AAHVNDQ", "SN Ia"),
    ("test_subjects/lightcurve_02.npz", "ZTF18AAILMNV", "SN Ia"),
    ("test_subjects/lightcurve_03.npz", "ZTF18AAIWZIE", "SN Ia"),
    ("test_subjects/lightcurve_04.npz", "ZTF18AASPRUI", "SN Ia"),
    ("test_subjects/lightcurve_05.npz", "ZTF18AATLFUS", "SN II"),
]

# --- Legend-position-bias diagnostic (--reorder-legend) --------------------------------------
# A real, live n=30 run of collect_lightcurve_labels.py found base emitting the SAME digit ("2" =
# SN Ibc, the LAST-listed class) on 28/30 objects regardless of the input plot — a near-total
# collapse to one answer. This reverses the legend order (SN Ibc now listed/coded FIRST, SN Ia
# now LAST) to tell apart two hypotheses without touching the real prompt: if base is truly
# position-biased (always emits whichever digit is listed last, regardless of what it means), its
# raw digit should flip from "2" to "0" here, and PREDICT_LABEL_REORDERED below will decode that
# new "0" back to "SN Ibc" — i.e. still SN Ibc, proving it's tracking the LAST-LISTED-CLASS
# regardless of digit. If instead it keeps emitting "2" (now decoding to SN Ia), the bias is tied
# to the digit token itself, not legend position. Either result rules something out; a genuinely
# varied response here (finally tracking the actual plots) would be the most encouraging outcome.
REORDERED_SN_LABELS = list(reversed(SN_LABELS))  # ["SN Ibc", "SN II", "SN Ia"]
REORDERED_SN_CLASS_CODES = {str(i): label for i, label in enumerate(REORDERED_SN_LABELS)}
REORDERED_SN_CLASS_CODE_LEGEND = "\n".join(f"{code}={name}" for code, name in REORDERED_SN_CLASS_CODES.items())
REORDERED_LC_PROMPT = (
    "Supernova light curve classifier. Output ONLY the digit code, nothing else.\n"
    f"{REORDERED_SN_CLASS_CODE_LEGEND}\n"
    "Light curve class code:"
)


def _load_lc_row(npz_path: str, object_id: str) -> pd.Series:
    """`test_subjects/*.npz` keys (`mjd`, `flux`, `flux_err`, `band_id`, `use`) -> the
    `lc_mjd`/`atcat_flux`/`atcat_flux_error`/`atcat_band_id`/`atcat_use` column names
    `render_lightcurve_plot`/`build_raw_inputs_lightcurve` expect (see `eval.datasets.
    lightcurve_yse`'s module docstring for the real column names this maps onto).
    """
    d = np.load(npz_path)
    return pd.Series({
        "object_id": object_id,
        "lc_mjd": d["mjd"],
        "atcat_flux": d["flux"],
        "atcat_flux_error": d["flux_err"],
        "atcat_band_id": d["band_id"],
        "atcat_use": d["use"],
    })


def _ground_truth_captions(object_ids: list[str]) -> dict[str, str]:
    captions = load_transient_captions(TRAIN_HF_PATH)
    by_id = captions.set_index("object_id")["transient_caption"]
    missing = [oid for oid in object_ids if oid not in by_id.index]
    if missing:
        raise KeyError(
            f"No real transient_caption found for {missing} in {TRAIN_HF_PATH!r} — TEST_LIGHTCURVES "
            "must only use objects confirmed to have one (see TRAIN_HF_PATH's comment)."
        )
    return {oid: by_id[oid] for oid in object_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v3_qwen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["digit_code", "descriptive"], default="digit_code")
    parser.add_argument(
        "--reorder-legend", action="store_true", default=False,
        help="digit_code mode only. Use REORDERED_LC_PROMPT (SN Ibc listed/coded first, SN Ia "
             "last — see that constant's comment) instead of LC_OOB_PROMPT/LC_EQUIPPED_PROMPT, to "
             "test whether a collapsed answer is tracking legend POSITION rather than the light "
             "curve itself.",
    )
    args = parser.parse_args()
    if args.reorder_legend and args.mode == "descriptive":
        raise SystemExit("--reorder-legend only applies to --mode digit_code (there's no legend to reorder in descriptive mode).")

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    rows = [(_load_lc_row(npz_path, object_id), true_label) for npz_path, object_id, true_label in TEST_LIGHTCURVES]

    if args.mode == "descriptive":
        oob_prompt, equipped_prompt = LC_DESCRIPTIVE_OOB_PROMPT, LC_DESCRIPTIVE_EQUIPPED_PROMPT
        oob_max_tokens = equipped_max_tokens = LC_DESCRIPTIVE_MAX_NEW_TOKENS
        ground_truth = _ground_truth_captions([oid for _, oid, _ in TEST_LIGHTCURVES])

        def _report(side: str, object_id: str, true_label: str, raw_answer: str) -> None:
            print(f"[{side} | {object_id} | true={true_label}]")
            print(f"  ground truth: {ground_truth[object_id]}")
            print(f"  {side}:       {raw_answer!r}")
    else:
        codes = REORDERED_SN_CLASS_CODES if args.reorder_legend else SN_CLASS_CODES
        oob_prompt = REORDERED_LC_PROMPT if args.reorder_legend else LC_OOB_PROMPT
        equipped_prompt = REORDERED_LC_PROMPT if args.reorder_legend else LC_EQUIPPED_PROMPT
        oob_max_tokens, equipped_max_tokens = LC_OOB_MAX_NEW_TOKENS, LC_EQUIPPED_MAX_NEW_TOKENS

        def _report(side: str, object_id: str, true_label: str, raw_answer: str) -> None:
            decoded = predict_label_from_code(raw_answer, codes)
            print(f"[true={true_label}] raw={raw_answer!r} -> decoded={decoded!r}")

    mode_tag = " (reordered legend)" if args.reorder_legend else f" ({args.mode})"

    print(f"<OOB>{mode_tag}")
    base_backend = get_backend(
        args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=LC_OOB_ENABLE_THINKING,
    )
    for (row, true_label), (_, object_id, _) in zip(rows, TEST_LIGHTCURVES):
        image = render_lightcurve_plot(row)
        answer = base_backend.generate({"image": image}, oob_prompt, oob_max_tokens)
        _report("base", object_id, true_label, answer)
    if args.backend == "local":
        free_local_backend(base_backend)

    print(f"<EQUIPPED>{mode_tag}")
    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["lightcurve"],
    )
    for (row, true_label), (_, object_id, _) in zip(rows, TEST_LIGHTCURVES):
        raw_inputs = build_raw_inputs_lightcurve(row, cfg)
        answer = equipped_backend.generate(raw_inputs, equipped_prompt, equipped_max_tokens)
        _report("equipped", object_id, true_label, answer)
    if args.backend == "local":
        free_local_backend(equipped_backend)


if __name__ == "__main__":
    main()
