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

import re

# Synonyms are matched in the order listed; a label's own name is always checked too, implicitly.
#
# The `sn <subtype>` / `type <subtype>` entries exist because models emit REAL astronomical
# subtypes that aren't among the dataset's three labels — the dataset only ever contains
# `SN Ia`/`SN II`/`SN Ibc`, but a model describing one will happily write "SN IIP" or "SN Ic".
# Each maps to its parent class here. `SN IIb` is the one genuinely debatable call: despite the
# "II" in its name it's a stripped-envelope transitional type, grouped with Ibc in BTS-style
# taxonomies, so that's where it's mapped — deliberately, not by prefix accident.
SN_TYPE_SYNONYMS: dict[str, list[str]] = {
    "SN Ia": ["sn ia", "type ia", "ia supernova", "thermonuclear", "white dwarf"],
    "SN II": [
        "sn ii", "type ii", "sn iip", "type iip", "sn iil", "type iil", "sn iin", "type iin",
        "ii supernova", "core-collapse", "hydrogen-rich", "hydrogen rich",
    ],
    "SN Ibc": [
        "sn ibc", "type ibc", "sn ib/c", "type ib/c", "sn ib", "type ib", "sn ic", "type ic",
        "sn ic-bl", "type ic-bl", "sn iib", "type iib",
        "ibc supernova", "stripped-envelope", "stripped envelope",
    ],
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
    match = re.search(r"(?<!\d)([0-9])(?!\d)", answer)
    if match is None:
        return None
    return class_codes.get(match.group(1))


_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.IGNORECASE)


def _last_longest_match(text: str, label_vocabulary: list[str], synonyms: dict[str, list[str]]) -> str | None:
    """The label whose name/synonym occurs LATEST in `text`; ties at the same end position go to
    the LONGEST match. Both rules are load-bearing, not defensive polish:

    - **Latest, not first** (unlike `predict_label`): these are conclusion-style answers, so the
      final mention is the model's actual verdict. "The early rise resembles a IIP, but the
      decline is consistent with the SN Ia class" concludes SN Ia — first-match would call it II.
    - **Longest at a tie**: `"sn ii"` is a strict prefix of `"sn iib"`, so both match at the same
      position; without the length tie-break, `SN IIb` would silently be read as `SN II` depending
      on which label happened to be checked first.

    Matches are word-boundary anchored, so `"ia"` inside an unrelated word can't trigger.
    """
    lowered = text.lower()
    best_key: tuple[int, int] | None = None
    best_label: str | None = None
    for label in label_vocabulary:
        for candidate in [label.lower(), *[s.lower() for s in synonyms.get(label, [])]]:
            for match in re.finditer(rf"\b{re.escape(candidate)}\b", lowered):
                key = (match.end(), match.end() - match.start())
                if best_key is None or key > best_key:
                    best_key, best_label = key, label
    return best_label


def predict_label_from_free_text(
    answer: str, label_vocabulary: list[str], synonyms: dict[str, list[str]] | None = None,
) -> str | None:
    """Parses a deliberately-verbose answer that merely CONTAINS a class, rather than requiring
    the model to emit nothing but a label. `None` only if no known label or synonym appears at all
    — counted as wrong/unparsed downstream, never silently dropped.

    Prefers an explicit `FINAL ANSWER: <class>` line when the model produced one, and otherwise
    reads the class out of the surrounding prose (see `_last_longest_match` for the two matching
    rules). That dual path is the point: the base model is expected to follow the requested
    format, while the equipped model tends to revert to the captioning voice it was fine-tuned on
    — whose real training captions end "...is consistent with the SN Ia class." — so both
    behaviours parse instead of only the compliant one.

    Falls back to scanning the whole answer if a `FINAL ANSWER` line exists but names nothing
    recognisable (e.g. "FINAL ANSWER: unclear"), rather than returning `None` while a perfectly
    good class sits in the prose above it.
    """
    match = _FINAL_ANSWER_RE.search(answer)
    if match is not None:
        from_final = _last_longest_match(match.group(1), label_vocabulary, synonyms or {})
        if from_final is not None:
            return from_final
    return _last_longest_match(answer, label_vocabulary, synonyms or {})


def make_predictor(
    answer_format: str,
    label_vocabulary: list[str],
    synonyms: dict[str, list[str]] | None = None,
    class_codes: dict[str, str] | None = None,
):
    """Picks the parser matching how a collect script actually prompted the model — shared here
    (not duplicated per runner script) since several scoring scripts need the exact same logic:
    using the wrong parser for a given answer format silently returns `None` for every object
    rather than erroring, so getting this dispatch right matters in more than one place. Kept
    dataset-agnostic (parameters, not hardcoded Galaxy10 constants) so this works for the
    lightcurve/SN-typing track's `SN_TYPE_SYNONYMS` too.

    Formats:
      - `"digit_code"` -> `predict_label_from_code`
      - `"verbose_class"` -> `predict_label_from_free_text` (deliberately-verbose answers that
        merely contain a class; prefers a `FINAL ANSWER:` line, else reads the last class named)
      - anything else, including the `"free_text"` default for files predating the key ->
        `predict_label` (first-match keyword scan)

    `"verbose_class"` is a separate name from `"free_text"` on purpose rather than replacing it:
    the two use genuinely different matching rules (last-vs-first match, longest-wins tie-break),
    so reusing the name would silently re-score every pre-existing free-text file under rules it
    was never evaluated with.

    `answer_format` is expected to be read from the collect file itself (`data.get("answer_format",
    "free_text")`), not assumed.
    """
    if answer_format == "digit_code":
        if class_codes is None:
            raise ValueError("class_codes is required when answer_format='digit_code'.")
        return lambda answer: predict_label_from_code(answer, class_codes)
    if answer_format == "verbose_class":
        return lambda answer: predict_label_from_free_text(answer, label_vocabulary, synonyms)
    return lambda answer: predict_label(answer, label_vocabulary, synonyms)
