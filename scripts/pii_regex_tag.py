#!/usr/bin/env python3
# acli: 1 complete
"""Tag structured identifiers with configured regexes after the classifier.

A classifier learns the identifiers it saw; a checksum- or grammar-shaped one
it never saw is better recognised by a pattern. This step applies the rules in
a `pii-regex-tag/v1` config to already-predicted rows and adds the spans they
find.

Group 1 of each pattern is the tagged span, so a cue may sit outside it. That
is what makes a rule such as the bank account number usable at all: the digits
carry no evidence of what they are, but "Account No:" before them does, and
only the digits are tagged. The same convention is what the production toolkit's Privacy module
reads, so one set of patterns serves both.

The label depends on the ontology: `--ontology ont3` (the default) gives the
successor type, `the production toolkit` keeps the pack's own class name. A rule missing a tag
for the requested ontology is refused rather than guessed at.

How a match argues with the model is a separate config, because the rules
follow the ontology and change rarely while the right arbitration follows how
good the model is. `skip` keeps the model's span, `add` emits both, `detail`
emits nothing and instead writes the rule's fine class onto a model span it
agrees with, and `score` lets the two compete. Which is best is an open
question to be settled by evaluation; without a policy the model keeps its
spans.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "agents"))
import acli  # noqa: E402

SCHEMA = "pii-regex-tag/v1"
POLICY_SCHEMA = "pii-regex-policy/v1"
OVERLAP_POLICIES = ("skip", "add", "detail", "score")
AGREEMENT_LEVELS = ("type", "family", "any")
DETAIL_SPAN_OWNERS = ("model", "rule")
MAX_PATTERN_CHARS = 200


@dataclass(frozen=True)
class Policy:
    """How a rule's match argues with the model, and how loudly.

    Which of these wins is an open question, to be answered by what improves
    the evaluation rather than by taste, so all four are available and the
    config states the default.

    `skip`   the model keeps any span it already found, and the match is dropped
    `add`    both are emitted, which is what the deployed pack does
    `detail` nothing new is emitted; instead a model span the rule agrees with
             is given the rule's fine class, so the rule names rather than
             asserts. A rule need not be precise enough to stand alone to be
             useful this way.
    `score`  the match competes, and whichever of the rule and the model scores
             higher keeps the span

    Under `detail` the two spans rarely coincide to the character, so
    `min_iou` and `max_edge_shift` say how close is close enough and
    `span_owner` says whose offsets survive when they differ.
    """

    mode: str = "skip"
    agreement: str = "family"
    min_iou: float = 0.8
    max_edge_shift: int = 1
    span_owner: str = "model"
    min_model_score: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in OVERLAP_POLICIES:
            raise ValueError(f"overlap must be one of {OVERLAP_POLICIES}, not {self.mode!r}")
        if self.agreement not in AGREEMENT_LEVELS:
            raise ValueError(f"agreement must be one of {AGREEMENT_LEVELS}, not {self.agreement!r}")
        if self.span_owner not in DETAIL_SPAN_OWNERS:
            raise ValueError(f"span owner must be one of {DETAIL_SPAN_OWNERS}")
        if not 0.0 <= self.min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1")
        if self.max_edge_shift < 0:
            raise ValueError("max_edge_shift must not be negative")


def _prefix_matches(prefixes: tuple[str, ...], document: str) -> bool:
    have = document.lower().replace("_", "-").split("-")
    for wanted in prefixes:
        want = wanted.lower().replace("_", "-").split("-")
        if have[: len(want)] == want:
            return True
    return False


def language_allows(
    rule_languages: tuple[str, ...],
    document: str | None,
    except_languages: tuple[str, ...] = (),
) -> bool:
    """BCP 47 prefix match, the same question the production toolkit's per-rule `language` asks.

    A rule naming no language runs everywhere. `language` is a whitelist: the
    rule runs only on documents whose tag one of its prefixes covers, matching
    by subtag so `en` covers `en-US`. It then does not run where the document
    language is unknown, since a rule that says where it belongs should not
    fire where that cannot be checked.

    `except_language` is the blacklist, and is the instrument for the use this
    is actually expected to see: a rule found misfiring in one language in
    production. Excluding that one language through the whitelist would mean
    enumerating every language it should keep running on, which is both
    laborious and wrong the moment a new language is served. A rule with only
    an exclusion still runs on documents of unknown language, because the
    evidence against it is specific and absent here.
    """
    if except_languages and document and _prefix_matches(except_languages, document):
        return False
    if not rule_languages:
        return True
    if not document:
        return False
    return _prefix_matches(rule_languages, document)


class Rule:
    """One compiled pattern, its ontology label and the fine class it names."""

    __slots__ = ("except_languages", "fine", "languages", "name", "pattern", "source", "tag")

    def __init__(
        self,
        name: str,
        source: str,
        tag: str,
        fine: str | None = None,
        language: Any = None,
        except_language: Any = None,
    ):
        self.name = name
        self.source = source
        self.tag = tag
        self.fine = fine or name
        self.languages = tuple(sorted(_tag_list(language)))
        self.except_languages = tuple(sorted(_tag_list(except_language)))
        try:
            self.pattern = re.compile(source)
        except re.error as error:
            raise ValueError(f"{name}: pattern does not compile: {error}") from error
        if self.pattern.groups != 1:
            raise ValueError(
                f"{name}: pattern must have exactly one capture group, the tagged span; "
                f"found {self.pattern.groups}"
            )
        if len(source) > MAX_PATTERN_CHARS:
            raise ValueError(f"{name}: pattern is {len(source)} characters, over {MAX_PATTERN_CHARS}")


def load_rules(path: Path, ontology: str | None = None) -> tuple[list[Rule], dict[str, str], dict[str, Any]]:
    """Compile every rule, resolving each label for one ontology.

    Also returns the coarse family of each ontology type, which is what a
    policy consults when it asks whether a rule and the model agree.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, found {payload.get('schema')!r}")
    resolved = ontology or payload.get("default_ontology")
    if not resolved:
        raise ValueError(f"{path}: no ontology requested and the config names no default")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"{path}: rules must be a nonempty list")
    rules = []
    for entry in raw_rules:
        name = entry.get("class")
        tags = entry.get("tags") or {}
        if not name or not entry.get("regex"):
            raise ValueError(f"{path}: every rule needs a class and a regex")
        if resolved not in tags:
            raise ValueError(f"{name}: no tag for ontology {resolved!r}; has {sorted(tags)}")
        rules.append(Rule(name, entry["regex"], tags[resolved], entry.get("fine"), entry.get("language")))
    families = dict(payload.get("families") or {})
    return rules, families, {"config": str(path), "ontology": resolved, "rules": len(rules)}


def load_policy(path: Path | None) -> tuple[Policy, dict[str, Any], dict[str, Any]]:
    """Read a policy, which is keyed to a trained model rather than to the rules.

    The rules are a property of the ontology and change rarely; which of them
    should outrank a model, and at what score, is a property of how good that
    model is, so it lives in its own file and is versioned with the model.
    """
    if path is None:
        return Policy(), {"scores": {}, "gates": {}}, {"policy": None}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: expected schema {POLICY_SCHEMA}, found {payload.get('schema')!r}")
    fields = {
        key: payload[key]
        for key in ("mode", "agreement", "min_iou", "max_edge_shift", "span_owner", "min_model_score")
        if key in payload
    }
    scores = {str(k): float(v) for k, v in (payload.get("scores") or {}).items()}
    gates = {str(k): dict(v) for k, v in (payload.get("gates") or {}).items()}
    for rule, gate in gates.items():
        unknown = set(gate) - {"require_tags", "except_tags"}
        if unknown:
            raise ValueError(f"{rule}: unknown gate keys {sorted(unknown)}")
    return (
        Policy(**fields),
        {"scores": scores, "gates": gates},
        {"policy": str(path), "policy_for_model": payload.get("model"), "gated_rules": sorted(gates)},
    )


def find_spans(text: str, rules: list[Rule], language: str | None = None) -> list[dict[str, Any]]:
    """Every rule's matches over one text, as spans of its configured label.

    A rule that names a language is skipped on a document of another.
    """
    found = []
    for rule in rules:
        if not language_allows(rule.languages, language, rule.except_languages):
            continue
        for match in rule.pattern.finditer(text):
            start, end = match.span(1)
            if start < 0 or start >= end:
                continue
            found.append(
                {
                    "start": start,
                    "end": end,
                    "label": rule.tag,
                    "fine": rule.fine,
                    "fine_rule": rule.name,
                    "source": "regex",
                }
            )
    found.sort(key=lambda span: (span["start"], span["end"], span["fine_rule"]))
    return found


def _intersection_over_union(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if overlap <= 0:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return overlap / union if union else 0.0


def _close_enough(found: dict[str, Any], model: dict[str, Any], policy: Policy) -> bool:
    """Whether two spans are the same span for the purpose of naming it."""
    if (found["start"], found["end"]) == (model["start"], model["end"]):
        return True
    edges = max(abs(found["start"] - model["start"]), abs(found["end"] - model["end"]))
    if edges <= policy.max_edge_shift:
        return True
    return _intersection_over_union(found, model) >= policy.min_iou


def _tag_list(value: Any) -> set[str]:
    """One tag is the common case, so a bare string means a list of one."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(tag) for tag in value}


def _tag_names(label: str | None, families: dict[str, str]) -> set[str]:
    """A model label answers to its own name and to its coarse family."""
    if label is None:
        return set()
    family = families.get(label)
    return {label} if family is None else {label, family}


def gate_allows(
    span: dict[str, Any],
    overlapping: list[dict[str, Any]],
    policy: Policy,
    families: dict[str, str],
    gates: dict[str, dict[str, list[str]]],
) -> tuple[bool, str | None]:
    """Whether a rule's belief about its documents lets this match act.

    An overbroad pattern is harmless when it may only fire where the model
    already chose a compatible tag, so a rule can name the tags that permit it
    and the tags that forbid it. Both take a coarse family as readily as a
    type. With a closed tag set the two lists are duals, so the pair is
    complete and no graded support term is needed; keep them both anyway,
    because they fail in opposite directions when the ontology grows. A new
    type simply never satisfies a require list, while it silently escapes an
    except list.
    """
    gate = gates.get(span["fine_rule"])
    if not gate:
        return True, None
    require = _tag_list(gate.get("require_tags"))
    forbid = _tag_list(gate.get("except_tags"))
    for model in overlapping:
        if forbid & _tag_names(model.get("label"), families):
            return False, "gate_except_tag"
    if not require:
        return True, None
    for model in overlapping:
        if _intersection_over_union(span, model) < policy.min_iou:
            continue
        if require & _tag_names(model.get("label"), families):
            return True, None
    return False, "gate_require_tag"


def _agrees(found: dict[str, Any], model: dict[str, Any], policy: Policy, families) -> bool:
    if policy.agreement == "any":
        return True
    if found["label"] == model.get("label"):
        return True
    if policy.agreement == "type":
        return False
    coarse = families.get(model.get("label"))
    return coarse is not None and coarse == families.get(found["label"])


def merge_spans(
    existing: list[dict[str, Any]],
    found: list[dict[str, Any]],
    policy: Policy | str = "skip",
    *,
    families: dict[str, str] | None = None,
    scores: dict[str, float] | None = None,
    gates: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Let each match argue with the model under the policy.

    Returns the resulting spans and a count of what each outcome was, so an
    evaluation can see why a policy did better rather than only that it did.
    """
    if isinstance(policy, str):
        policy = Policy(mode=policy)
    families = families or {}
    scores = scores or {}
    gates = gates or {}
    kept = [dict(span) for span in existing]
    counts: dict[str, int] = {}

    def tally(outcome: str) -> None:
        counts[outcome] = counts.get(outcome, 0) + 1

    for span in found:
        overlapping = [
            other for other in kept if other["start"] < span["end"] and span["start"] < other["end"]
        ]
        allowed, refusal = gate_allows(span, overlapping, policy, families, gates)
        if not allowed:
            tally(refusal or "gate_refused")
            continue
        if policy.mode == "add":
            kept.append(dict(span))
            tally("added")
        elif policy.mode == "skip":
            if overlapping:
                tally("skipped_overlap")
            else:
                kept.append(dict(span))
                tally("added")
        elif policy.mode == "detail":
            # The rule names a span rather than asserting one, so it emits
            # nothing of its own and only writes its fine class onto a model
            # span it agrees with.
            named = False
            for model in overlapping:
                if not _close_enough(span, model, policy):
                    tally("detail_span_too_far")
                    continue
                if not _agrees(span, model, policy, families):
                    tally("detail_label_disagrees")
                    continue
                score = model.get("score")
                if policy.min_model_score is not None and (score is None or score < policy.min_model_score):
                    tally("detail_model_unsure")
                    continue
                model["fine"] = span["fine"]
                model["fine_rule"] = span["fine_rule"]
                if policy.span_owner == "rule":
                    model["start"], model["end"] = span["start"], span["end"]
                tally("detail_attached")
                named = True
                break
            if not named and not overlapping:
                tally("detail_no_model_span")
        elif policy.mode == "score":
            # An unspecified score is inactive: a rule that names no score does
            # not compete. `score` is the old single number and behaves as an
            # anti-confidence against the model's detection, kept for
            # compatibility; new rules should use the declared confidences.
            if span["fine_rule"] not in scores:
                tally("score_not_configured")
                continue
            rule_score = scores[span["fine_rule"]]
            beaten = [
                model
                for model in overlapping
                if model.get("score") is not None and model["score"] < rule_score
            ]
            unscored = [model for model in overlapping if model.get("score") is None]
            if unscored:
                tally("score_model_unscored")
            elif len(beaten) == len(overlapping):
                for model in beaten:
                    kept.remove(model)
                kept.append(dict(span))
                tally("replaced_model" if overlapping else "added")
            else:
                tally("score_model_wins")
    kept.sort(key=lambda span: (span["start"], span["end"]))
    return kept, counts


def tag_rows(
    rows: list[dict[str, Any]],
    rules: list[Rule],
    *,
    span_keys: list[str],
    policy: Policy | str = "skip",
    families: dict[str, str] | None = None,
    scores: dict[str, float] | None = None,
    gates: dict[str, dict[str, list[str]]] | None = None,
    text_key: str = "text",
    language_keys: tuple[str, ...] = ("bcp47", "lang", "language"),
    texts: list[str] | None = None,
    languages: list[str | None] | None = None,
) -> dict[str, Any]:
    """Apply the rules to each row in place, over each named span list."""
    if texts is not None and len(texts) != len(rows):
        raise ValueError("texts must line up with rows")
    if isinstance(policy, str):
        policy = Policy(mode=policy)
    counts: dict[str, Any] = {"rows": 0, "matched_rows": 0}
    outcomes: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for index, row in enumerate(rows):
        text = texts[index] if texts is not None else row.get(text_key)
        if not isinstance(text, str):
            raise ValueError(f"row {row.get('id', index)!r} has no {text_key!r} to scan")
        counts["rows"] += 1
        if languages is not None:
            language = languages[index]
        else:
            language = next((row[key] for key in language_keys if isinstance(row.get(key), str)), None)
        found = find_spans(text, rules, language)
        if not found:
            continue
        touched = False
        for key in span_keys:
            if key not in row:
                continue
            spans, row_outcomes = merge_spans(
                row[key], found, policy, families=families, scores=scores, gates=gates
            )
            row[key] = spans
            for outcome, value in row_outcomes.items():
                outcomes[outcome] = outcomes.get(outcome, 0) + value
                if outcome in ("added", "detail_attached", "replaced_model"):
                    touched = True
        if touched:
            counts["matched_rows"] += 1
            for span in found:
                by_rule[span["fine_rule"]] = by_rule.get(span["fine_rule"], 0) + 1
    counts["mode"] = policy.mode
    counts["outcomes"] = dict(sorted(outcomes.items()))
    counts["by_rule"] = dict(sorted(by_rule.items()))
    return counts


def run(args) -> dict[str, Any]:
    rules, families, receipt = load_rules(args.config, args.ontology)
    policy, arbitration, policy_receipt = load_policy(args.policy)
    overrides = {
        key: getattr(args, key)
        for key in ("mode", "agreement", "min_iou", "max_edge_shift", "span_owner", "min_model_score")
        if getattr(args, key, None) is not None
    }
    if overrides:
        policy = replace(policy, **overrides)
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    counts = tag_rows(
        rows,
        rules,
        span_keys=args.span_key,
        policy=policy,
        families=families,
        scores=arbitration["scores"],
        gates=arbitration["gates"],
        text_key=args.text_key,
    )
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"{args.output} exists; refusing to overwrite")
        with args.output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "schema": "pii-regex-tag-run/v1",
        **receipt,
        **policy_receipt,
        "policy_applied": {
            "mode": policy.mode,
            "agreement": policy.agreement,
            "min_iou": policy.min_iou,
            "max_edge_shift": policy.max_edge_shift,
            "span_owner": policy.span_owner,
            "min_model_score": policy.min_model_score,
            "overridden": sorted(overrides),
        },
        "span_keys": list(args.span_key),
        "input": str(args.input),
        "output": None if args.output is None else str(args.output),
        **counts,
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, exit_codes={0: "tagged", 2: "invalid input"})
    parser.add_argument("--input", type=Path, required=True, help="rows as JSONL")
    parser.add_argument("--output", type=Path, help="rows with the added spans")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "pii_regex_tags_v1.json",
        help="pii-regex-tag/v1 rules",
    )
    parser.add_argument("--ontology", help="label set to tag with; default from the config (ont3)")
    parser.add_argument(
        "--span-key",
        action="append",
        default=None,
        help="row key holding a span list (repeatable); default both ont3 projections",
    )
    parser.add_argument("--text-key", default="text", help="row key holding the text")
    parser.add_argument(
        "--policy",
        type=Path,
        help="pii-regex-policy/v1 arbitration, which belongs to a trained model "
        "rather than to the rules; without one the model keeps its spans",
    )
    parser.add_argument(
        "--mode",
        choices=OVERLAP_POLICIES,
        help="override the policy's mode: skip, add, detail or score",
    )
    parser.add_argument("--agreement", choices=AGREEMENT_LEVELS, help="detail: label agreement level")
    parser.add_argument("--min-iou", type=float, help="detail: how much two spans must share")
    parser.add_argument("--max-edge-shift", type=int, help="detail: characters an edge may differ by")
    parser.add_argument(
        "--span-owner", choices=DETAIL_SPAN_OWNERS, help="detail: whose offsets survive a near match"
    )
    parser.add_argument("--min-model-score", type=float, help="detail: floor on the model's score")
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    if not args.span_key:
        args.span_key = ["reference_aware_preds", "named_only_preds", "preds", "spans"]
    acli.emit(run(args), acli.resolve_format(args))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        acli.die(str(error), 2)
