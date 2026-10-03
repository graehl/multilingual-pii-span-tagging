"""Annotator/language/tag boundary conventions shared by training and scoring.

Rules refer to the annotation's vocabulary, before a many:many label map.
Only internal segmentation is optional: outer edges, types, punctuation and
unlabelled words remain constraints. Adjacent spans can join across whitespace.
An explicit ``annotator`` wins over the existing source provenance fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "pii-annotation-conventions-v1"
PREFIX_BITS = {"B": 1, "I": 2, "E": 4, "S": 8}


def annotator_identity(row, default=None):
    return row.get("annotator") or row.get("mix_source") or row.get("src") or row.get("source") or default


class AnnotationConventions:
    def __init__(self, document):
        if not isinstance(document, dict) or set(document) != {"schema", "rules"}:
            raise ValueError("annotation conventions require exactly schema and rules")
        if document["schema"] != SCHEMA or not isinstance(document["rules"], list):
            raise ValueError("unsupported annotation conventions schema/rules")
        self.rules = {}
        for rule in document["rules"]:
            if not isinstance(rule, dict) or set(rule) != {"annotator", "language", "tag", "segmentation"}:
                raise ValueError("each convention requires annotator, language, tag, segmentation")
            if any(not isinstance(value, str) or not value for value in rule.values()):
                raise ValueError("convention fields must be nonempty strings")
            if rule["annotator"] == "*" or rule["tag"] in {"*", "O"}:
                raise ValueError("conventions require a specific annotator and entity tag")
            if rule["segmentation"] not in {"exact", "internal"}:
                raise ValueError("segmentation must be exact or internal")
            key = (rule["annotator"], rule["language"], rule["tag"])
            if key in self.rules:
                raise ValueError(f"duplicate annotation convention {key}")
            self.rules[key] = rule["segmentation"]
        self.document = {
            "schema": SCHEMA,
            "rules": sorted(document["rules"], key=lambda r: (r["annotator"], r["language"], r["tag"])),
        }
        self.sha256 = hashlib.sha256(json.dumps(self.document, sort_keys=True).encode()).hexdigest()

    def internal(self, row, tag):
        annotator = annotator_identity(row)
        language = row.get("lang")
        return self.rules.get((annotator, language, tag), self.rules.get((annotator, "*", tag))) == "internal"

    def tags(self, row):
        return {tag for _annotator, _language, tag in self.rules if self.internal(row, tag)}


def load_annotation_conventions(path):
    """Read a standalone policy or the policy embedded in a correctness map."""
    document = json.loads(Path(path).read_text())
    return AnnotationConventions(document.get("annotation_conventions", document))


def resolve_annotation_conventions(path=None, *, resume_config=None, correctness_map=None):
    if path is None and resume_config is None and correctness_map is not None:
        mapping = json.loads(Path(correctness_map).read_text())
        if "annotation_conventions" in mapping:
            path = correctness_map
    requested = load_annotation_conventions(path) if path is not None else None
    if resume_config is None:
        return requested
    saved = getattr(resume_config, "pii_annotation_conventions", None)
    if requested is not None and requested.document != saved:
        raise ValueError(
            "annotation conventions cannot change during exact resume; start a new initialization stage"
        )
    return AnnotationConventions(saved) if saved is not None else None


def join_internal_spans(text, spans, eligible):
    """Canonicalize eligible same-tag spans; return spans and original indices.

    Inputs must be nonoverlapping. Unlike the historical merged-overlap
    diagnostic this never crosses punctuation, words, or a different tag.
    """
    result = []
    for index, (start, end, tag) in sorted(enumerate(spans), key=lambda item: item[1][:2]):
        if not 0 <= start < end <= len(text):
            raise ValueError("internal segmentation span lies outside the text")
        if result and start < result[-1][1]:
            raise ValueError("internal segmentation requires nonoverlapping primary spans")
        if result and tag in eligible and tag == result[-1][2]:
            gap = text[result[-1][1] : start]
            if not gap or gap.isspace():
                previous = result[-1]
                result[-1] = (previous[0], end, tag, previous[3] + [index])
                continue
        result.append((start, end, tag, [index]))
    return result


def convention_training_row(row, conventions):
    """Join only declared spans, preserving row provenance and objective weights."""
    eligible = conventions.tags(row)
    if not eligible:
        return row, eligible
    groups = join_internal_spans(row["text"], row["spans"], eligible)
    weights = row.get("primary_span_objective_weights")
    if weights is not None:
        if len(weights) != len(row["spans"]):
            raise ValueError("primary span weights must align with spans")
        if any(len({weights[i] for i in indices}) != 1 for *_, indices in groups):
            raise ValueError("cannot join spans with different objective weights")
    result = dict(row, spans=[[start, end, tag] for start, end, tag, _ in groups])
    if weights is not None:
        result["primary_span_objective_weights"] = [weights[indices[0]] for *_, indices in groups]
    return result, eligible


def internal_prefix_masks(length):
    """Permitted BIOES letters at fixed outer edges (zero means ordinary CE)."""
    return [8] if length == 1 else [9] + [15] * (length - 2) + [12]


def join_acceptable_internal(text, spans, eligible):
    """Join same-compatible-type components for convention-aware exact scoring.

    Each span carries a set of acceptable types. Only a common declared type
    can license a join; the joined span retains that intersection. Overlapping
    predictions stay separate, so duplicates cannot disappear into a match.
    """
    result = []
    for start, end, accepted in sorted(spans):
        accepted = frozenset(accepted)
        if result and 0 <= start < end <= len(text):
            previous = result[-1]
            common = previous[2] & accepted & eligible
            gap = text[previous[1] : start]
            if 0 <= previous[0] < previous[1] <= start and common and (not gap or gap.isspace()):
                result[-1] = (previous[0], end, common | (previous[2] & accepted & {"O"}))
                continue
        result.append((start, end, accepted))
    return result
