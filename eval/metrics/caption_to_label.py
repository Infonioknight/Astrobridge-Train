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
