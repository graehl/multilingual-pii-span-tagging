#!/usr/bin/env python3
"""Conservatively complete high-precision PII spans before translation.

Rules activate independently for each source corpus and canonical type. A rule
is active when the type is absent from the source's declared inventory, or
when fewer than half of the rule detections already match a gold span of that
type. Existing spans always win: an uncovered detection is added only when it
does not overlap any source annotation. The report retains source file/line
identity for every affected row so placeholder transport can retranslate only
the changed carriers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from pii_annotation_cache import atomic_text_output, normalize_span, validate_spans
    from pii_projector import TAGSET_PATH, Tagset
except ModuleNotFoundError:  # Imported as scripts.pii_rule_completion in tests.
    from scripts.pii_annotation_cache import atomic_text_output, normalize_span, validate_spans
    from scripts.pii_projector import TAGSET_PATH, Tagset

RULESET_VERSION = "pii-high-precision-completion-v3"
DEFAULT_COVERAGE_THRESHOLD = 0.5

EMAIL_RE = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{0,62}[A-Za-z0-9])?"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w@-])"
)
ACCOUNT_CANDIDATE_RE = re.compile(r"(?<!\d)(?<!\d[ -])(?:\d[ -]?){12,18}\d(?!\d)(?![ -]\d)")
ACCOUNT_CONTEXT_RE = re.compile(
    r"\b(?:account|amex|american\s+express|bank|card|credit|debit|mastercard|payment|visa)\b",
    re.IGNORECASE,
)

# The number shape is broad enough for common local and international
# formatting, while admission requires an immediately adjacent, explicit role
# cue. Keep extensions in the protected span.
PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\w@])"
    r"(?:\+[ \t]*)?"
    r"(?:"
    r"\([0-9]{1,4}\)(?:[ \t.-]*[0-9]){4,12}"
    r"|[0-9]{1,3}[ \t.-]*\([0-9]{2,4}\)(?:[ \t.-]*[0-9]){4,10}"
    r"|[0-9](?:[0-9 .-]*[0-9])"
    r")"
    r"(?:[ \t]*(?:x|ext(?:ension)?\.?)[ \t]*[0-9]{1,6})?"
    r"(?![\w@])",
    re.IGNORECASE,
)
PHONE_EXTENSION_RE = re.compile(r"[ \t]*(?:x|ext(?:ension)?\.?)[ \t]*[0-9]{1,6}$", re.IGNORECASE)
DATE_SHAPE_RE = re.compile(r"(?:[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}|[0-9]{1,2}-[0-9]{1,2}-[0-9]{4})$")
IPV4_SHAPE_RE = re.compile(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")

PHONE_CUE = (
    r"(?:phone(?:[ \t]+(?:number|no\.?))?|telephone(?:[ \t]+(?:number|no\.?))?|"
    r"tel\.?|mobile[ \t]+(?:phone|number|no\.?)|cell[ \t]+(?:phone|number|no\.?)|"
    r"tel[eé]fono|t[eé]l[eé]phone|telefone|telem[oó]vel|telefonnummer|telefon|"
    r"cellulare|telefoon|telepon|ponsel|numer[ \t]+telefonu|"
    r"телефон|мобільний[ \t]+телефон|هاتف|جوال|फोन|मोबाइल|"
    r"電話番号|電話|휴대폰|전화번호|전화|电话号码|电话|手機|手机|"
    r"điện[ \t]+thoại|số[ \t]+điện[ \t]+thoại)"
)
FAX_CUE = r"(?:fax|facsimile|telefax|faks|传真|傳真|ファックス|팩스|فاكس|फैक्स)"


def contextual_cue_patterns(cue: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    prefix = re.compile(
        rf"(?<!\w){cue}(?!\w)[ \t]*(?:[:=：#-][ \t]*)?$",
        re.IGNORECASE,
    )
    suffix = re.compile(
        rf"^[ \t]*(?:\({cue}(?!\w)\)|\[{cue}(?!\w)\])",
        re.IGNORECASE,
    )
    return prefix, suffix


PHONE_CUE_BEFORE_RE, PHONE_CUE_AFTER_RE = contextual_cue_patterns(PHONE_CUE)
FAX_CUE_BEFORE_RE, FAX_CUE_AFTER_RE = contextual_cue_patterns(FAX_CUE)


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    label: str
    rule: str


@dataclass(frozen=True)
class CompletionRule:
    name: str
    label: str
    detect: Callable[[str], Iterable[Detection]]


def detect_emails(text: str) -> Iterable[Detection]:
    for match in EMAIL_RE.finditer(text):
        yield Detection(match.start(), match.end(), "email", "email-rfc-shape-v1")


def luhn_valid(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def recognized_card_shape(digits: str) -> bool:
    """Recognize Visa, Mastercard, or American Express length/prefix shapes."""
    if digits.startswith("4") and len(digits) in {13, 16, 19}:
        return True
    if len(digits) == 16:
        prefix2 = int(digits[:2])
        prefix4 = int(digits[:4])
        if 51 <= prefix2 <= 55 or 2221 <= prefix4 <= 2720:
            return True
    return len(digits) == 15 and digits[:2] in {"34", "37"}


def detect_luhn_accounts(text: str) -> Iterable[Detection]:
    for match in ACCOUNT_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ -]", "", match.group())
        context = text[max(0, match.start() - 80) : min(len(text), match.end() + 40)]
        if recognized_card_shape(digits) and luhn_valid(digits) and ACCOUNT_CONTEXT_RE.search(context):
            # The surface is not sufficient to distinguish a payment card
            # from another financial account identifier. Keep the broader
            # redaction-safe account_number label.
            yield Detection(
                match.start(),
                match.end(),
                "account_number",
                "luhn-card-shape-with-account-context-v2",
            )


def plausible_phone_surface(surface: str) -> bool:
    if len(surface) > 40:
        return False
    base = PHONE_EXTENSION_RE.sub("", surface).strip()
    digits = re.sub(r"\D", "", base)
    if not 7 <= len(digits) <= 15:
        return False
    if base.count("(") != base.count(")") or base.count("(") > 1:
        return False
    if DATE_SHAPE_RE.fullmatch(base) or IPV4_SHAPE_RE.fullmatch(base):
        return False
    return True


def detect_contextual_numbers(
    text: str,
    *,
    label: str,
    rule: str,
    cue_before: re.Pattern[str],
    cue_after: re.Pattern[str],
) -> Iterable[Detection]:
    for match in PHONE_CANDIDATE_RE.finditer(text):
        if not plausible_phone_surface(match.group()):
            continue
        before = text[max(0, match.start() - 64) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 48)]
        if cue_before.search(before) or cue_after.search(after):
            yield Detection(match.start(), match.end(), label, rule)


def detect_phone_numbers(text: str) -> Iterable[Detection]:
    yield from detect_contextual_numbers(
        text,
        label="phone_number",
        rule="contextual-phone-shape-v1",
        cue_before=PHONE_CUE_BEFORE_RE,
        cue_after=PHONE_CUE_AFTER_RE,
    )


def detect_fax_numbers(text: str) -> Iterable[Detection]:
    yield from detect_contextual_numbers(
        text,
        label="fax_number",
        rule="contextual-fax-shape-v1",
        cue_before=FAX_CUE_BEFORE_RE,
        cue_after=FAX_CUE_AFTER_RE,
    )


PHONE_FAX_RULES = (
    CompletionRule("phone", "phone_number", detect_phone_numbers),
    CompletionRule("fax", "fax_number", detect_fax_numbers),
)
RULES = (
    CompletionRule("email", "email", detect_emails),
    CompletionRule("luhn_account", "account_number", detect_luhn_accounts),
    *PHONE_FAX_RULES,
)


def overlap_fraction_match(left: Detection, right: list[Any]) -> bool:
    intersection = max(0, min(left.end, int(right[1])) - max(left.start, int(right[0])))
    return intersection >= 0.8 * (left.end - left.start) and intersection >= 0.8 * (
        int(right[1]) - int(right[0])
    )


def intersects(detection: Detection, span: list[Any]) -> bool:
    return detection.start < int(span[1]) and detection.end > int(span[0])


def inventory_for_schema(tagset: Tagset, schema: str) -> set[str]:
    if schema in {"canonical", "_identity"}:
        return set(tagset.nodes)
    try:
        return set(tagset.sources[schema].values())
    except KeyError as error:
        raise ValueError(f"unknown inventory schema {schema!r}") from error


def complete_records(
    records: list[dict[str, Any]],
    declared_inventory: set[str],
    *,
    source_id: str,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    apply: bool = True,
    rules: tuple[CompletionRule, ...] = RULES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return completed canonical records and a source-level audit report."""
    if not 0 <= coverage_threshold <= 1:
        raise ValueError("coverage threshold must be in [0, 1]")

    detections_by_record: list[dict[str, list[Detection]]] = []
    if len({rule.name for rule in rules}) != len(rules):
        raise ValueError("completion rule names must be unique")
    rule_counts = {rule.name: Counter() for rule in rules}
    for record in records:
        spans = record["spans"]
        row_detections: dict[str, list[Detection]] = {}
        for rule in rules:
            detections = list(rule.detect(record["text"]))
            row_detections[rule.name] = detections
            counts = rule_counts[rule.name]
            counts["detections"] += len(detections)
            for detection in detections:
                same_label = [span for span in spans if span[2] == rule.label]
                if any(overlap_fraction_match(detection, span) for span in same_label):
                    counts["same_type_matches"] += 1
                elif any(intersects(detection, span) for span in spans):
                    counts["conflicting_overlaps"] += 1
                else:
                    counts["uncovered"] += 1
        detections_by_record.append(row_detections)

    activation: dict[str, dict[str, Any]] = {}
    for rule in rules:
        counts = rule_counts[rule.name]
        detections = counts["detections"]
        coverage = counts["same_type_matches"] / detections if detections else None
        inventory_absent = rule.label not in declared_inventory
        below_threshold = coverage is not None and coverage < coverage_threshold
        activation[rule.name] = {
            "canonical_label": rule.label,
            "inventory_absent": inventory_absent,
            "detections": detections,
            "same_type_matches": counts["same_type_matches"],
            "same_type_coverage": coverage,
            "uncovered": counts["uncovered"],
            "conflicting_overlaps": counts["conflicting_overlaps"],
            "active": inventory_absent or below_threshold,
            "activation_reasons": [
                reason
                for reason, condition in (
                    ("declared_inventory_gap", inventory_absent),
                    ("same_type_coverage_below_threshold", below_threshold),
                )
                if condition
            ],
            "added": 0,
        }

    completed: list[dict[str, Any]] = []
    affected: list[dict[str, Any]] = []
    proposed_affected: list[dict[str, Any]] = []
    added_by_language: Counter[str] = Counter()
    for record, row_detections in zip(records, detections_by_record, strict=True):
        output = dict(record)
        spans = [list(span) for span in record["spans"]]
        additions = []
        for rule in rules:
            if not activation[rule.name]["active"]:
                continue
            for detection in row_detections[rule.name]:
                if any(span[2] == rule.label and overlap_fraction_match(detection, span) for span in spans):
                    continue
                if any(intersects(detection, span) for span in spans):
                    continue
                additions.append(
                    {
                        "start": detection.start,
                        "end": detection.end,
                        "label": detection.label,
                        "rule": detection.rule,
                        "surface": record["text"][detection.start : detection.end],
                        "context": record["text"][
                            max(0, detection.start - 80) : min(len(record["text"]), detection.end + 80)
                        ],
                    }
                )
                if apply:
                    span = [detection.start, detection.end, detection.label]
                    spans.append(span)
                    activation[rule.name]["added"] += 1
                    added_by_language[str(record.get("lang") or "<unknown>")] += 1
        spans.sort(key=lambda span: (span[0], span[1], span[2]))
        output["spans"] = spans
        if additions:
            locator = record.get("_rule_source_locator") or {"id": record.get("id")}
            proposal = {
                "id": record.get("id"),
                "lang": record.get("lang"),
                "source": locator,
                "additions": additions,
            }
            proposed_affected.append(proposal)
            if apply:
                output["rule_completion"] = {
                    "ruleset": RULESET_VERSION,
                    "source": locator,
                    "additions": additions,
                }
                affected.append(proposal)
        output.pop("_rule_source_locator", None)
        completed.append(output)

    return completed, {
        "schema_version": 1,
        "ruleset": RULESET_VERSION,
        "rules_evaluated": [rule.name for rule in rules],
        "source_id": source_id,
        "documents": len(records),
        "coverage_threshold": coverage_threshold,
        "threshold_contract": "activate when same-type matches / detections is strictly below threshold",
        "declared_inventory": sorted(declared_inventory),
        "applied": apply,
        "rules": activation,
        "added_spans": sum(rule["added"] for rule in activation.values()),
        "added_by_language": dict(sorted(added_by_language.items())),
        "affected_documents": len(affected),
        "affected": affected,
        "proposed_spans": sum(len(row["additions"]) for row in proposed_affected),
        "proposed_affected_documents": len(proposed_affected),
        "proposed_affected": proposed_affected,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_canonical_rows(path: Path, source_schema: str) -> list[dict[str, Any]]:
    tagset = Tagset()
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            spans = sorted(
                (normalize_span(tagset, source_schema, span) for span in row["spans"]),
                key=lambda span: (span[0], span[1], span[2]),
            )
            validate_spans(str(path), line_number, row["text"], spans)
            records.append(
                {
                    **row,
                    "spans": spans,
                    "_rule_source_locator": {"path": str(path), "line_1based": line_number},
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-schema", required=True, help="schema of input span labels")
    parser.add_argument(
        "--inventory-schema",
        help="declared source inventory (default: --source-schema)",
    )
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument(
        "--mode",
        choices=("audit", "apply"),
        default="audit",
        help="audit proposals without changing spans, or apply after the >=98%% precision review passes",
    )
    args = parser.parse_args()

    tagset = Tagset()
    records = read_canonical_rows(args.input, args.source_schema)
    inventory_schema = args.inventory_schema or args.source_schema
    completed, report = complete_records(
        records,
        inventory_for_schema(tagset, inventory_schema),
        source_id=args.source_id,
        coverage_threshold=args.coverage_threshold,
        apply=args.mode == "apply",
    )
    report["input"] = {"path": str(args.input), "sha256": file_sha256(args.input)}
    report["tagset"] = {"path": TAGSET_PATH, "sha256": file_sha256(Path(TAGSET_PATH))}
    with atomic_text_output(str(args.output)) as output:
        for row in completed:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with atomic_text_output(str(args.report)) as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"PII RULE COMPLETION: proposed {report['proposed_spans']} spans in "
        f"{report['proposed_affected_documents']}/{report['documents']} documents; "
        f"applied {report['added_spans']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
