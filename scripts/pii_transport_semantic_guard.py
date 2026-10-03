#!/usr/bin/env python
"""High-precision deterministic guards for transported PII records.

These checks only reject contradictions that can be established from the
materialized row itself.  They deliberately do not judge general prose,
translation quality, dates against an unstated present, or card checksums.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

try:
    from pii_annotation_cache import atomic_text_output
except ModuleNotFoundError:  # Imported as scripts.pii_transport_semantic_guard in tests.
    from scripts.pii_annotation_cache import atomic_text_output

RULESET = "final20-transport-semantic-guard-v2"
TAGSET_PATH = Path(__file__).with_name("pii_tagset.yaml")


def _target_snake_case_labels() -> tuple[str, ...]:
    tagset = yaml.safe_load(TAGSET_PATH.read_text(encoding="utf-8"))
    return tuple(sorted((label for label in tagset["nodes"] if "_" in label), key=len, reverse=True))


TARGET_SNAKE_CASE_LABELS = _target_snake_case_labels()
_TAG_LITERAL_RE = re.compile(
    rf"(?<![\w])(?P<label>{'|'.join(map(re.escape, TARGET_SNAKE_CASE_LABELS))})(?![\w])",
    re.IGNORECASE,
)

_GROUPED_OR_DECIMAL_AMOUNT = r"(?:\d{1,3}(?:[,.\u00a0 '\u2019]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
_CURRENCY_DESIGNATOR = (
    r"(?:USD|EUR|GBP|JPY|CNY|RMB|KRW|INR|AUD|CAD|CHF|SEK|NOK|DKK|PLN|CZK|"
    r"TRY|IDR|VND|UAH|RUB|BRL|MXN|AED|SAR|EGP|dollars?|euros?|pounds?|yuan|"
    r"yen|won|rupees?|[$€£¥₹₩₽₺₴₱₪])"
)
_PREFIXED_CURRENCY_RE = re.compile(
    rf"(?<![\w]){_CURRENCY_DESIGNATOR}\s*(?P<amount>{_GROUPED_OR_DECIMAL_AMOUNT})",
    re.IGNORECASE,
)
_SUFFIXED_CURRENCY_RE = re.compile(
    rf"(?<![\w])(?P<amount>{_GROUPED_OR_DECIMAL_AMOUNT})\s*{_CURRENCY_DESIGNATOR}(?![\w])",
    re.IGNORECASE,
)


def _is_covered(start: int, end: int, spans: list[list[Any]]) -> bool:
    def bounds(span: Any) -> tuple[int, int]:
        if isinstance(span, dict):
            return int(span["start"]), int(span["end"])
        return int(span[0]), int(span[1])

    return any(span_start <= start and end <= span_end for span_start, span_end in map(bounds, spans))


def colon_field_key_tag_literals(text: str) -> list[str]:
    """Return canonical tag tokens used as colon-delimited object keys."""
    findings = []
    for match in _TAG_LITERAL_RE.finditer(text):
        if re.match(r"""["']?\s*:""", text[match.end() :]):
            findings.append(match.group("label").casefold())
    return findings


def literal_tag_values_outside_spans(text: str, spans: list[list[Any]]) -> list[str]:
    """Return canonical snake-case tag names that leaked outside annotations."""
    field_keys = set(colon_field_key_tag_literals(text))
    findings = []
    for match in _TAG_LITERAL_RE.finditer(text):
        label = match.group("label").casefold()
        # A machine-shaped label can legitimately name a colon-delimited
        # object field; reject it only when the token itself is an unwrapped
        # value. Arbitrary table syntax is deliberately outside this contract.
        if label in field_keys and re.match(r"""["']?\s*:""", text[match.end() :]):
            continue
        if not _is_covered(match.start(), match.end(), spans):
            findings.append(label)
    return findings


def currency_amounts_outside_spans(text: str, spans: list[list[Any]]) -> list[str]:
    """Return currency-marked numeric amounts whose number is unannotated."""
    findings = []
    observed_offsets = set()
    for pattern in (_PREFIXED_CURRENCY_RE, _SUFFIXED_CURRENCY_RE):
        for match in pattern.finditer(text):
            offsets = match.span("amount")
            if offsets in observed_offsets:
                continue
            observed_offsets.add(offsets)
            if not _is_covered(*offsets, spans):
                findings.append(match.group(0))
    return findings


def currency_amounts_inside_spans(text: str, spans: list[list[Any]]) -> list[str]:
    """Return currency-marked numeric amounts whose number is annotated."""
    findings = []
    observed_offsets = set()
    for pattern in (_PREFIXED_CURRENCY_RE, _SUFFIXED_CURRENCY_RE):
        for match in pattern.finditer(text):
            offsets = match.span("amount")
            if offsets in observed_offsets:
                continue
            observed_offsets.add(offsets)
            if _is_covered(*offsets, spans):
                findings.append(match.group(0))
    return findings


def semantic_guard_reasons(text: str, spans: list[list[Any]]) -> list[str]:
    """Return stable whole-row rejection reasons under policy v2."""
    reasons = [f"literal_tag_outside_span:{value}" for value in literal_tag_values_outside_spans(text, spans)]
    reasons.extend(
        f"currency_amount_outside_span:{value}" for value in currency_amounts_outside_spans(text, spans)
    )
    return reasons


def _audit_id(row: dict[str, Any]) -> int | None:
    if row.get("audit_id") is not None:
        return int(row["audit_id"])
    semantic_review = row.get("semantic_review") or {}
    if semantic_review.get("audit_id") is not None:
        return int(semantic_review["audit_id"])
    return None


def analyze_guard(input_path: Path, adjudication_path: Path | None = None) -> dict[str, Any]:
    """Measure guard retention on a frozen JSONL artifact."""
    input_bytes = input_path.read_bytes()
    rows = [json.loads(line) for line in input_bytes.splitlines()]
    rejected = []
    reason_counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, 1):
        reasons = semantic_guard_reasons(row["text"], row["spans"])
        if not reasons:
            continue
        reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
        rejected.append(
            {
                "row_number": row_number,
                "audit_id": _audit_id(row),
                "id": row.get("id"),
                "lang": row.get("lang"),
                "reasons": reasons,
            }
        )

    report: dict[str, Any] = {
        "schema": "pii-transport-semantic-guard-analysis-v1",
        "ruleset": RULESET,
        "input": {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
            "documents": len(rows),
        },
        "retained_documents": len(rows) - len(rejected),
        "rejected_documents": len(rejected),
        "rejected_fraction": len(rejected) / len(rows) if rows else 0.0,
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejected": rejected,
    }
    if adjudication_path is not None:
        adjudication_bytes = adjudication_path.read_bytes()
        adjudications = {
            int(row["audit_id"]): row
            for row in (json.loads(line) for line in adjudication_bytes.splitlines())
        }
        rejected_ids = {row["audit_id"] for row in rejected if row["audit_id"] is not None}
        supported_ids = {
            audit_id for audit_id, row in adjudications.items() if row["adjudication"] == "supported"
        }
        unsupported_ids = {
            audit_id for audit_id, row in adjudications.items() if row["adjudication"] == "unsupported"
        }
        report["adjudication"] = {
            "path": str(adjudication_path),
            "sha256": hashlib.sha256(adjudication_bytes).hexdigest(),
            "supported_documents": len(supported_ids),
            "supported_rejected": sorted(rejected_ids & supported_ids),
            "supported_not_rejected": sorted(supported_ids - rejected_ids),
            "known_unsupported_rejected": sorted(rejected_ids & unsupported_ids),
            "not_in_adjudication_rejected": sorted(rejected_ids - set(adjudications)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = analyze_guard(args.input, args.adjudication)
    with atomic_text_output(str(args.report)) as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(
        f"SEMANTIC_GUARD: retained {report['retained_documents']}/{report['input']['documents']} "
        f"and rejected {report['rejected_documents']} -> {args.report}"
    )


if __name__ == "__main__":
    main()
