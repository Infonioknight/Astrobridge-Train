"""Turns a model's free-text answer into one of a fixed set of labels — isolated in its own file,
deliberately, since this matching heuristic is the single most likely piece of this eval bench to
need iteration once real generations are actually seen. Nothing in `backend.py` or `eval/datasets/`
needs to change if this file's matching logic changes.

Deliberately simple (case-insensitive keyword/synonym match, first hit wins), not an LLM-judge or
embedding-similarity classifier — the eval prompts themselves are expected to constrain the model
to answer with (close to) one of the known labels, so this is a parser for an already-mostly-
constrained answer, not a general-purpose classifier of its own. Real limitation, not hidden: a
caption that hedges or lists multiple candidates ("possibly SN Ia, though the fast decline
suggests Ibc") will match whichever keyword the caption happens to mention first, not necessarily
the model's actual best guess — acceptable for a first pass, worth revisiting once real
generations are inspected.
"""
from __future__ import annotations

# Synonyms are matched in the order listed; a label's own name is always checked too, implicitly.
SN_TYPE_SYNONYMS: dict[str, list[str]] = {
    "SN Ia": ["type ia", "ia supernova", "thermonuclear", "white dwarf"],
    "SN II": ["type ii", "ii supernova", "core-collapse", "hydrogen-rich", "hydrogen rich"],
    "SN Ibc": ["type ib", "type ic", "type ibc", "ibc supernova", "stripped-envelope", "stripped envelope"],
}

GALAXY10_LABEL_SYNONYMS: dict[str, list[str]] = {
    "Disturbed Galaxies": ["disturbed"],
    "Merging Galaxies": ["merging", "merger"],
    "Round Smooth Galaxies": ["round smooth", "completely round"],
    "In-between Round Smooth Galaxies": ["in-between round", "in between round"],
    "Cigar Shaped Smooth Galaxies": ["cigar shaped", "cigar-shaped"],
    "Barred Spiral Galaxies": ["barred spiral"],
    "Unbarred Tight Spiral Galaxies": ["unbarred tight spiral", "tight spiral"],
    "Unbarred Loose Spiral Galaxies": ["unbarred loose spiral", "loose spiral"],
    "Edge-on Galaxies without Bulge": ["edge-on without bulge", "edge on without bulge"],
    "Edge-on Galaxies with Bulge": ["edge-on with bulge", "edge on with bulge"],
}


def predict_label(
    caption: str, label_vocabulary: list[str], synonyms: dict[str, list[str]] | None = None,
) -> str | None:
    """Returns the first label in `label_vocabulary` order whose own name or a documented synonym
    appears (case-insensitive) in `caption`; `None` if nothing matches — counted as a wrong/
    abstained prediction downstream (`classification.classification_report`), never silently
    dropped.

    `label_vocabulary` order matters: it's the tie-break when a caption happens to mention more
    than one label's keywords (see module docstring's hedging-caption limitation). Pass the
    dataset's own label order (e.g. `SN_TYPE_SYNONYMS`'s key order, or Galaxy10's `label` int
    order) so ties resolve predictably, not by accident of dict iteration.
    """
    synonyms = synonyms or {}
    text = caption.lower()
    for label in label_vocabulary:
        candidates = [label.lower(), *[s.lower() for s in synonyms.get(label, [])]]
        if any(c in text for c in candidates):
            return label
    return None


def predict_label_from_code(answer: str, class_codes: dict[str, str]) -> str | None:
    """Parses a digit-code answer (e.g. `" 2"`, `"5."`, `"Code: 5"`) into the label it maps to,
    via `class_codes` (digit string -> label name — see `eval.datasets.image_galaxy10.CLASS_CODES`).

    Real, separate parser from `predict_label` above, not a variant of it: once
    `eval.datasets.image_galaxy10.CLASS_CODE_PROMPT` is the actual prompt in use (confirmed live,
    via `eval/prompt_playground.py`, to get much more reliable format compliance than asking a
    model to name a class from a long list in free text), the answers being parsed are bare digit
    codes, not label-name text — `predict_label`'s keyword/synonym matching would never match a
    digit at all and would return `None` for every single object, silently.

    Matches the first STANDALONE digit in `answer` — not a digit embedded in a longer number, so
    `"10"` or `"2.5"` don't accidentally match code `"2"` — via a regex lookaround, not a plain
    substring search (which `"2" in "12"` would wrongly pass). `None` if no valid standalone code
    digit appears at all.
    """
    import re

    match = re.search(r"(?<!\d)([0-9])(?!\d)", answer)
    if match is None:
        return None
    return class_codes.get(match.group(1))


def make_predictor(
    answer_format: str,
    label_vocabulary: list[str],
    synonyms: dict[str, list[str]] | None = None,
    class_codes: dict[str, str] | None = None,
):
    """Picks `predict_label` or `predict_label_from_code` based on how a collect script actually
    prompted the model — shared here (not duplicated per runner script) since both
    `eval/runners/score_image_eval.py` and `eval/runners/score_image_eval_debiased.py` need the
    exact same logic: using the wrong parser for a given answer format silently returns `None` for
    every object rather than erroring, so getting this dispatch right matters in more than one
    place. Kept dataset-agnostic (parameters, not hardcoded Galaxy10 constants) so this also works
    for the lightcurve/SN-typing track's `SN_TYPE_SYNONYMS`.

    `answer_format` is expected to be read from the collect file itself (`data.get("answer_format",
    "free_text")` — older files predate the key and default to free-text matching), not assumed.
    """
    if answer_format == "digit_code":
        if class_codes is None:
            raise ValueError("class_codes is required when answer_format='digit_code'.")
        return lambda answer: predict_label_from_code(answer, class_codes)
    return lambda answer: predict_label(answer, label_vocabulary, synonyms)
