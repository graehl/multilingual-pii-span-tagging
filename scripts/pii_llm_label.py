#!/usr/bin/env python
"""Prompt-based LLM PII labeling harness (topics/pii-adaptation.md;
user 2026-07-21: upper-bounding and teaching).

Prompt = fixed task file (prompts/pii-label/task.txt, slots {tags}
{example} {text}) + a one-shot example selected by BCP-47 language tag
from prompts/pii-label/examples.json (resolution walks subtags:
zh-CN -> zh -> en). Output is parsed from a JSON array of
{"t": surface, "type": tag, "n": occurrence} and aligned to the source
by exact nth-occurrence string match — offsets from LLMs are unreliable,
surfaces are not.

An opt-in tag catalog adds one definition and prototypical surface for
every configured tag. The historical name-only prompt remains the default
so existing teacher outputs retain an exact reproducible control.
Candidate task templates and language-example bundles are opt-in paths;
their resolved paths and hashes are frozen in the prompt contract.

Emits pred-format jsonl identical to pii_eval predict, so the LLM slots
into the same scorecards as the task-specific anchors (upper-bound
measurement) and into pii_untagged_onboard silver (teaching).
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import regex

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from chat_session import FormatOptions, ModelFormat  # noqa: E402
from scripts.pii_prompt_template import load_prompt_template  # noqa: E402
from scripts.pii_subclass import (  # noqa: E402
    SubclassSpec,
    load_subclass_spec,
    parse_candidate_subclass_annotation,
    parse_subclass_annotation,
    render_subclass_catalog,
)

PROMPT_DIR = os.path.join(REPO, "prompts", "pii-label")

CORE_TAGS = [
    "person_name",
    "given_name",
    "family_name",
    "organization",
    "occupation",
    "age",
    "gender",
    "address",
    "street_address",
    "city",
    "region",
    "country",
    "postal_code",
    "email",
    "phone_number",
    "url",
    "username",
    "ip_address",
    "national_id",
    "passport_number",
    "drivers_license",
    "tax_id",
    "account_number",
    "iban",
    "card_number",
    "password",
    "api_key",
    "date",
    "date_of_birth",
    "time",
    "medical_record_number",
    "insurance_id",
    "employee_id",
    "customer_id",
]


FORMATS = {
    # original: per-surface occurrence counting (model must count)
    "json": {
        "rules": (
            "- Output ONLY a JSON array, no prose. Each element:\n"
            '  {"t": "<surface copied verbatim>", "type": "<tag>", '
            '"n": <occurrence number of that exact string, 1-based>}\n'
            "- Copy surfaces exactly: no normalization, no translation.\n"
            "- If the text contains no PII, output []."
        ),
    },
    # user 2026-07-22: global increment ids in reading order - no
    # occurrence counting; alignment is a left-to-right cursor scan
    "json-seq": {
        "rules": (
            "- Output ONLY a JSON array, no prose, spans in reading "
            "order. Each element:\n"
            '  {"i": <span number: 1, 2, 3, ... in order of appearance>, '
            '"t": "<surface copied verbatim>", "type": "<tag>"}\n'
            "- Copy surfaces exactly: no normalization, no translation.\n"
            "- If the text contains no PII, output []."
        ),
    },
    "json-offsets": {
        "rules": (
            "- Output ONLY a JSON array, no prose, spans in reading order. Each element:\n"
            '  {"start": <zero-based Unicode character start>, "end": <exclusive Unicode '
            'character end>, "t": "<exact source substring>", "type": "<tag>"}\n'
            "- Coordinates are half-open [start, end) offsets in the decoded original input. "
            "The controller will reject the whole response unless t equals input[start:end].\n"
            "- Emit one element per positive tag. Distinct tags may share an interval; never "
            "repeat the same start, end, and type.\n"
            "- Copy surfaces exactly: no normalization, no translation.\n"
            "- If the text contains no in-scope positive, output []."
        ),
    },
    "json-offsets-confidence": {
        "rules": (
            "- Output ONLY a JSON array, no prose, spans in reading order. Each element:\n"
            '  {"start": <zero-based Unicode character start>, "end": <exclusive Unicode '
            'character end>, "t": "<exact source substring>", "type": "<tag>", '
            '"repair_confidence": <number from 0 through 1>}\n'
            "- Coordinates are half-open [start, end) offsets in the decoded original input. "
            "The controller will reject the whole response unless t equals input[start:end].\n"
            "- repair_confidence is the teacher's proposed probability that its returned "
            "non-uncertain fine label should replace or retain the incumbent decision; it is "
            "audited rather than trusted as an objective weight.\n"
            "- Emit one element per positive tag. Distinct tags may share an interval; never "
            "repeat the same start, end, and type.\n"
            "- Copy surfaces exactly: no normalization, no translation.\n"
            "- If the text contains no in-scope positive, output []."
        ),
    },
    "json-subclasses": {
        "rules": (
            "- Output ONLY one JSON object with exactly primary_spans, bernoulli_spans, "
            "and subclass_spans arrays; no prose.\n"
            "- primary_spans items are {start,end,t,type} for added successor primary spans.\n"
            "- bernoulli_spans items are {carrier_start,carrier_end,carrier_type,start,end,t,type}.\n"
            "- subclass_spans items are "
            "{carrier_start,carrier_end,carrier_type,start,end,t,family,value}.\n"
            "- All offsets are zero-based Unicode half-open coordinates in the decoded input; "
            "t must equal input[start:end].\n"
            "- A full-span Bernoulli or subclass value must use start/end exactly equal to its "
            "carrier. A component-span subclass uses its real internal subspan.\n"
            "- Omit unknown families. Emit Q only when none of the named outcomes is reliably "
            "known to apply; Q never means unknown."
        ),
    },
    "json-candidate-subclasses": {
        "rules": (
            "- Output ONLY one JSON object with exactly candidate_decisions, added_candidates, "
            "and subclass_spans arrays; no prose.\n"
            "- candidate_decisions contains exactly one item for every controller candidate, "
            "in candidate_id order: {candidate_id,primary_type,bernoulli_types}.\n"
            "- primary_type is one of that candidate's allowed_primary_types. bernoulli_types "
            "is the ordered list of every true channel from its bernoulli_channels; [] explicitly "
            "means all applicable listed channels are false.\n"
            "- added_candidates contains only positive successor carriers absent from the "
            "controller ledger: {start,end,t,primary_type,bernoulli_types}.\n"
            "- subclass_spans items are "
            "{carrier_start,carrier_end,carrier_type,start,end,t,family,value}.\n"
            "- All offsets are zero-based Unicode half-open coordinates in the decoded input; "
            "t must equal input[start:end].\n"
            "- A full-span categorical value must use start/end exactly equal to its carrier. "
            "A component-span subclass uses its real internal subspan.\n"
            "- O is valid only as a controller-candidate primary decision. O never appears in "
            "added_candidates or subclass_spans. Omitted categorical families remain unknown; "
            "Q means reliably none of the listed outcomes."
        ),
    },
    "json-groups": {
        "rules": (
            "- Output ONLY a JSON array, no prose. Group each distinct exact surface once. "
            "Each element:\n"
            '  {"t": "<surface copied from the decoded input>", "types": '
            '[["<tag>"], ["O"], ["<tag>", "<alternative tag>"]]}\n'
            "- The outer types list has exactly one entry for EVERY occurrence of t in the "
            "decoded input, from left to right. Its length MUST equal the number of source "
            'occurrences. Use ["O"] when that occurrence is not in scope.\n'
            "- A non-O entry is a nonempty set of plausible allowed tags for that one "
            "occurrence. Multiple tags are permissive alternatives, not multiple entities. "
            "O is exclusive and may not appear with another tag.\n"
            "- Include a surface group only when at least one occurrence has a non-O tag. "
            "The decoded input is NFKC-normalized; NFKC-normalize t as well, then copy it "
            "verbatim. Never repeat t, translate it, or omit its O occurrences.\n"
            "- If the text contains no PII, output []."
        ),
    },
    "json-groups-lexical": {
        "rules": (
            "- Output ONLY a JSON array, no prose. Group each distinct exact lexical surface "
            "once. Each element:\n"
            '  {"t": "<surface copied from the decoded input>", "types": '
            '[["<tag>"], ["O"], ["<tag>", "<alternative tag>"]]}\n'
            "- The outer types list has exactly one entry for EVERY complete lexical occurrence "
            "of t in source order. Do not count a short word inside a longer word: `he` is not "
            "an occurrence inside `the`, and `it` is not an occurrence inside `with`.\n"
            "- Put every positive tag for one occurrence in that occurrence's inner list. Use "
            '["O"] only as a positional placeholder for an occurrence with no emitted positive.\n'
            "- Copy surfaces exactly from the decoded input: no normalization, no translation.\n"
            "- If the text contains no PII, output []."
        ),
    },
    # transport-format redaction rewrite (user 2026-07-22): replace each
    # span with the tuning-corpus placeholder token [TYPE_k], numbered
    # left to right - the format our cascade corpora already use, so
    # models have heavy prior exposure; may beat json-like outputs
    "redact": {
        "rules": (
            "- Output the ENTIRE input text unchanged, except replace "
            "every PII span with a numbered tag token in square "
            "brackets: [PERSON_NAME_1], [DATE_2], ... (tag name "
            "uppercase, numbered 1, 2, 3, ... left to right).\n"
            "- Do not translate, reorder, add, or drop any other "
            "characters.\n"
            "- If the text contains no PII, output the input unchanged."
        ),
    },
    # bracketed inline rewrite (the quick contrastive; also one arm of
    # the OpenAI privacy-filter annotation protocol)
    "inline": {
        "rules": (
            "- Output the ENTIRE input text verbatim, unchanged, except "
            "wrap every PII span as [surface](tag).\n"
            "- Do not translate, reorder, add, or drop any other "
            "characters.\n"
            "- If the text contains no PII, output the input unchanged."
        ),
    },
}


SPAN_POLICIES = {
    "legacy": "",
    "maximal-entity-v1": (
        "Span selection policy:\n"
        "- Prefer the longest exact surface that forms one entity of the selected tag. Do not "
        "split a full patient name, organization, facility, address, or identifier into generic "
        "or component labels when a configured full-entity tag applies.\n"
        "- Do not absorb a title, role, punctuation, or neighboring location into a name or "
        "facility span unless the selected tag explicitly defines it as part of that entity.\n"
        "- Emit a component separately only when the text independently presents it as a "
        "sensitive entity or no configured full-entity tag applies.\n"
        "- Do not emit both coarse and fine labels for the same mention. Preserve overlapping "
        "spans only when the text genuinely presents two independently meaningful entities.\n"
        "- Before returning, make a final coverage pass for missed people, organizations and "
        "facilities, locations and addresses, dates, identifiers, and contact details."
    ),
}


CATALOG_ROW_RE = re.compile(r"^\| `([^`]+)` \| (.*?) \| (.*?) \|$")


def load_tag_catalog(path: str | Path) -> dict[str, dict[str, str]]:
    """Load the label-catalog Markdown table as a strict prompt catalog."""
    catalog = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = CATALOG_ROW_RE.match(line)
        if match is None:
            continue
        tag, definition, examples = match.groups()
        if tag in catalog:
            raise ValueError(f"{path}:{line_number}: duplicate tag {tag!r}")
        if not definition.strip():
            raise ValueError(f"{path}:{line_number}: empty definition for {tag!r}")
        if examples.strip() in {"", "-", "—"}:
            raise ValueError(f"{path}:{line_number}: no prototypical surface for {tag!r}")
        catalog[tag] = {"definition": definition.strip(), "examples": examples.strip()}
    if not catalog:
        raise ValueError(f"{path}: no tag catalog rows found")
    return catalog


def load_id_list(path: str | Path) -> list[str]:
    """Load a strict newline-delimited row-id selection."""
    ids = Path(path).read_text(encoding="utf-8").splitlines()
    if not ids or any(not row_id for row_id in ids):
        raise ValueError(f"{path}: id list must contain nonempty lines")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return ids


def select_docs_by_ids(docs: list[dict], ids: list[str], *, source: str | Path) -> list[dict]:
    """Select requested rows while preserving source order."""
    requested = set(ids)
    available = {doc.get("id") for doc in docs}
    missing = requested - available
    if missing:
        raise ValueError(f"{source}: missing selected ids {sorted(missing)[:3]}")
    return [doc for doc in docs if doc.get("id") in requested]


def render_tag_catalog(tags: list[str], catalog: dict[str, dict[str, str]]) -> str:
    """Render configured tags in inventory order and reject partial catalogs."""
    missing = [tag for tag in tags if tag not in catalog]
    if missing:
        raise ValueError(f"tag catalog is missing configured tags: {', '.join(missing)}")
    return "\n".join(
        f"- {tag}: {catalog[tag]['definition']} Example: {catalog[tag]['examples']}" for tag in tags
    )


def render_input_text(text, input_render="plain"):
    if input_render == "plain":
        return text
    if input_render == "json-string":
        return json.dumps(text, ensure_ascii=False)
    raise ValueError(f"unsupported input rendering {input_render!r}")


def render_example(ex, fmt, input_render="plain"):
    # A language may supply several short worked segments instead of one. More
    # shots help up to a point, and short dense segments from the domains we
    # care about teach more per prefix token than one long easy sentence.
    if isinstance(ex, list):
        return "\n\n".join(render_example(item, fmt, input_render) for item in ex)
    text, labels = ex["text"], ex["labels"]
    if fmt == "json":
        out = json.dumps(labels, ensure_ascii=False)
    elif fmt == "json-seq":
        out = json.dumps(
            [{"i": k + 1, "t": l["t"], "type": l["type"]} for k, l in enumerate(labels)], ensure_ascii=False
        )
    elif fmt in {"json-offsets", "json-offsets-confidence"}:
        offset_labels = []
        for label in labels:
            surface = label["t"]
            occurrence = int(label.get("n", 1) or 1)
            occurrences = source_occurrences(text, surface)
            if not 1 <= occurrence <= len(occurrences):
                raise ValueError(
                    f"example surface {surface!r} occurrence {occurrence} is absent from the text"
                )
            start = occurrences[occurrence - 1]
            item = {
                "start": start,
                "end": start + len(surface),
                "t": surface,
                "type": label["type"],
            }
            if fmt == "json-offsets-confidence":
                item["repair_confidence"] = float(label.get("repair_confidence", 1.0))
            offset_labels.append(item)
        offset_labels.sort(key=lambda item: (item["start"], item["end"], item["type"]))
        out = json.dumps(offset_labels, ensure_ascii=False)
    elif fmt in {"json-subclasses", "json-candidate-subclasses"}:
        guidance = ex.get("guidance")
        output_field = (
            "candidate_subclass_output" if fmt == "json-candidate-subclasses" else "subclass_output"
        )
        output = ex.get(output_field)
        if not isinstance(guidance, dict) or not isinstance(output, dict):
            raise ValueError(f"{fmt} examples need guidance and {output_field} objects")
        return (
            "Example controller guidance JSON string:\n"
            + json.dumps(json.dumps(guidance, ensure_ascii=False), ensure_ascii=False)
            + "\nExample input:\n"
            + render_input_text(text, input_render)
            + "\nExample output:\n"
            + json.dumps(output, ensure_ascii=False)
            + "\n"
        )
    elif fmt in {"json-groups", "json-groups-lexical"}:
        groups = {}
        for label in labels:
            surface = label["t"]
            occurrence_finder = (
                lexical_source_occurrences if fmt == "json-groups-lexical" else source_occurrences
            )
            occurrences = occurrence_finder(text, surface)
            occurrence = int(label.get("n", 1) or 1)
            if not 1 <= occurrence <= len(occurrences):
                raise ValueError(
                    f"example surface {surface!r} occurrence {occurrence} is absent from the text"
                )
            group = groups.setdefault(
                surface,
                {
                    "first_start": occurrences[0],
                    "types": [["O"] for _ in occurrences],
                },
            )
            slot = group["types"][occurrence - 1]
            if slot == ["O"]:
                slot.clear()
            if label["type"] not in slot:
                slot.append(label["type"])
        out = json.dumps(
            [
                {"t": surface, "types": group["types"]}
                for surface, group in sorted(
                    groups.items(), key=lambda item: (item[1]["first_start"], item[0])
                )
            ],
            ensure_ascii=False,
        )
    elif fmt == "redact":
        spans = []
        for l in labels:
            pos, found = -1, 0
            while found < l.get("n", 1):
                pos = text.find(l["t"], pos + 1)
                found += 1
            spans.append((pos, pos + len(l["t"]), l["type"]))
        spans.sort()
        parts, last = [], 0
        for k, (a, b, ty) in enumerate(spans, 1):
            parts += [text[last:a], f"[{ty.upper()}_{k}]"]
            last = b
        parts.append(text[last:])
        out = "".join(parts)
    else:  # inline: wrap each labeled surface at its nth occurrence
        spans = []
        for l in labels:
            pos, found = -1, 0
            while found < l.get("n", 1):
                pos = text.find(l["t"], pos + 1)
                found += 1
            spans.append((pos, pos + len(l["t"]), l["t"], l["type"]))
        spans.sort()
        parts, last = [], 0
        for a, b, t, ty in spans:
            parts += [text[last:a], f"[{t}]({ty})"]
            last = b
        parts.append(text[last:])
        out = "".join(parts)
    return "Example input:\n" + render_input_text(text, input_render) + "\nExample output:\n" + out + "\n"


SOURCE_PARAGRAPH_FIELD = "annotation_guidance"


def add_source_paragraph_guidance(docs, default_lang):
    """Give each row the paper's ctx-v6 guidance: its source paragraph as context.

    A row that already carries ``annotation_guidance`` (the paper's own
    evaluation rows do) keeps it byte for byte. Otherwise the paragraph is
    the row's ``document_context`` before/after text around the sentence.
    Returns the field name to pass as the guidance field.
    """
    for index, doc in enumerate(docs):
        if doc.get(SOURCE_PARAGRAPH_FIELD) is not None:
            continue
        context = doc.get("document_context") or {}
        if not isinstance(context, dict) or not set(context) <= {"before", "after"}:
            raise ValueError(
                f"input row {index} id={doc.get('id')!r}: document_context must hold before/after"
            )
        paragraph = " ".join(
            part for part in (context.get("before"), doc["text"], context.get("after")) if part
        )
        doc[SOURCE_PARAGRAPH_FIELD] = json.dumps(
            {
                "bcp47": doc.get("lang", default_lang),
                "context": paragraph,
                "context_is_source_paragraph": True,
                "instruction": "context resolves references only; annotate the input sentence alone",
            },
            ensure_ascii=False,
        )
    return SOURCE_PARAGRAPH_FIELD


def resolve_example(examples, lang):
    """BCP-47 fallback: full tag, then progressively stripped subtags,
    then en."""
    tag = (lang or "en").replace("_", "-")
    parts = tag.split("-")
    while parts:
        key = "-".join(parts)
        if key in examples:
            return key, examples[key]
        parts.pop()
    return "en", examples["en"]


def load_language_rules(path):
    """Load per-language annotation instructions as {language tag: [rule, ...]}."""
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or not rules:
        raise ValueError(f"{path}: language rules must be a nonempty object")
    for language, lines in rules.items():
        if (
            not isinstance(language, str)
            or not language
            or not isinstance(lines, list)
            or not lines
            or any(not isinstance(line, str) or not line.strip() for line in lines)
        ):
            raise ValueError(f"{path}: language {language!r} needs a nonempty list of rule strings")
    return rules


def resolve_language_rules(rules, lang):
    """Same-language rules only: full tag, then stripped subtags; never another language."""
    parts = (lang or "en").replace("_", "-").split("-")
    while parts:
        key = "-".join(parts)
        if key in rules:
            return key, rules[key]
        parts.pop()
    return None, []


def render_language_rules(rules, lang):
    """Render one language's extra rules as a prompt block, or nothing."""
    key, lines = resolve_language_rules(rules, lang)
    if not lines:
        return ""
    return (
        f"Additional rules for this input's language ({key}):\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )


def require_language_specific_examples(examples, docs, default_lang="en"):
    """Reject cross-language one-shot fallback for every requested input language."""
    missing = []
    for language in sorted({doc.get("lang", default_lang) for doc in docs}):
        example_language, _example = resolve_example(examples, language)
        requested_primary = (language or default_lang).replace("_", "-").split("-", 1)[0].lower()
        example_primary = example_language.replace("_", "-").split("-", 1)[0].lower()
        if example_primary != requested_primary:
            missing.append(f"{language}->{example_language}")
    if missing:
        raise ValueError(
            "no same-language worked example for: "
            + ", ".join(missing)
            + "; add native examples or explicitly pass --allow-example-fallback "
            "for historical replay"
        )


def normalize_proposals(text, proposals, tagset):
    normalized = []
    for index, proposal in enumerate(proposals):
        if isinstance(proposal, dict):
            start = proposal.get("start")
            end = proposal.get("end")
            label = proposal.get("label") or proposal.get("type")
        elif isinstance(proposal, (list, tuple)) and len(proposal) == 3:
            start, end, label = proposal
        else:
            raise ValueError(f"proposal {index} has unsupported shape")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"proposal {index} start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"proposal {index} end must be an integer")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"proposal {index} has invalid bounds [{start}, {end})")
        if label not in tagset:
            raise ValueError(f"proposal {index} has unknown tag {label!r}")
        normalized.append((start, end, label))
    normalized.sort()
    return normalized


def render_proposal_text(text, proposals, tagset, element_name="XLMR_PROPOSAL"):
    """Wrap disjoint student spans next to their surfaces without exposing offsets."""
    if element_name not in {"XLMR_PROPOSAL", "PII_PROPOSAL"}:
        raise ValueError(f"unsupported proposal element {element_name!r}")
    normalized = normalize_proposals(text, proposals, tagset)
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"proposal spans overlap: {previous[:2]} and {current[:2]}")
    parts = []
    cursor = 0
    for start, end, label in normalized:
        parts.extend(
            (
                text[cursor:start],
                f'<{element_name} type="{label}">',
                text[start:end],
                f"</{element_name}>",
            )
        )
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def render_proposal_list(text, proposals, tagset):
    """Render fallible proposal candidates without requiring disjoint markup."""
    return json.dumps(
        [
            {"start": start, "end": end, "type": label, "surface": text[start:end]}
            for start, end, label in normalize_proposals(text, proposals, tagset)
        ],
        ensure_ascii=False,
    )


def build_prompt(
    task_tpl,
    examples,
    lang,
    tags,
    text,
    fmt="json",
    proposals=None,
    proposal_views=None,
    tag_catalog=None,
    span_policy="legacy",
    input_render="plain",
    proposal_render="markup",
    guidance=None,
    language_rules=None,
):
    if "{{include" in task_tpl:
        raise ValueError("Expand prompt includes with load_prompt_template before build_prompt")
    task_tpl = task_tpl.replace("{format_example}", "{example}")
    key, ex = resolve_example(examples, lang)
    if fmt not in {"json-subclasses", "json-candidate-subclasses"}:
        segments = ex if isinstance(ex, list) else [ex]
        invalid_example_tags = sorted(
            {label["type"] for segment in segments for label in segment["labels"]} - set(tags)
        )
        if invalid_example_tags:
            raise ValueError(
                f"example language {key}: labels outside the configured tag inventory: "
                + ", ".join(invalid_example_tags)
            )
    ex_block = render_example(ex, fmt, input_render)
    if proposals is not None and proposal_views:
        raise ValueError("single proposals and proposal views are mutually exclusive")
    if proposal_views:
        if proposal_render == "list":
            prompt_text = "Original input:\n" + render_input_text(text, input_render)
            prompt_text += "\n\n" + "\n\n".join(
                f'Fallible proposal candidates from "{source}":\n'
                + render_proposal_list(text, view_proposals, set(tags))
                for source, view_proposals in proposal_views
            )
        else:
            prompt_text = "\n\n".join(
                (
                    f'Proposal view from "{source}" (marked copy of the same original text):\n'
                    + render_proposal_text(
                        text,
                        view_proposals,
                        set(tags),
                        element_name="PII_PROPOSAL",
                    )
                )
                for source, view_proposals in proposal_views
            )
    elif proposals is not None:
        if proposal_render == "list":
            prompt_text = (
                "Original input:\n"
                + render_input_text(text, input_render)
                + "\n\nFallible current annotation candidates:\n"
                + render_proposal_list(text, proposals, set(tags))
            )
        else:
            prompt_text = render_proposal_text(text, proposals, set(tags))
    else:
        prompt_text = render_input_text(text, input_render)
    # replace-based templating: task.txt legitimately contains literal
    # JSON braces, which str.format would parse as fields
    rendered_tags = ", ".join(tags)
    if tag_catalog is not None:
        rendered_tags += (
            "\n\nTag definitions and prototypical surfaces "
            "(the examples illustrate the type; copy only spans from the actual input):\n"
            + render_tag_catalog(tags, tag_catalog)
        )
    policy = SPAN_POLICIES[span_policy]
    if policy:
        rendered_tags += "\n\n" + policy
    if guidance is not None:
        if not isinstance(guidance, str) or not guidance:
            raise ValueError("review guidance must be a nonempty string")
        if "{guidance}" not in task_tpl:
            raise ValueError("review guidance requires a {guidance} task-template placeholder")
    if language_rules is not None and "{language_rules}" not in task_tpl:
        raise ValueError("language rules require a {language_rules} task-template placeholder")
    out = (
        task_tpl.replace("{tags}", rendered_tags)
        .replace("{format_rules}", FORMATS[fmt]["rules"])
        .replace("{language_rules}", render_language_rules(language_rules, lang) if language_rules else "")
        .replace("{example}", ex_block)
        .replace("{guidance}", json.dumps(guidance, ensure_ascii=False) if guidance else "")
        .replace("{text}", prompt_text)
    )
    return key, out


def build_prompt_contract(
    task_tpl,
    examples,
    docs,
    tags,
    fmt,
    default_lang,
    model,
    proposal_field=None,
    proposal_view_fields=(),
    tag_catalog=None,
    tag_catalog_sha256=None,
    span_policy="legacy",
    model_revision=None,
    input_render="plain",
    proposal_render="markup",
    unicode_normalization="none",
    guidance_field=None,
    language_rules=None,
    language_rules_sha256=None,
):
    """Record the exact prompt contract and one rendered sample per language."""
    samples = []
    seen_languages = set()
    for doc in docs:
        language = doc.get("lang", default_lang)
        if language in seen_languages:
            continue
        seen_languages.add(language)
        proposals = doc[proposal_field] if proposal_field is not None else None
        proposal_views = [(source, doc[field]) for source, field in proposal_view_fields]
        guidance = doc[guidance_field] if guidance_field is not None else None
        example_language, prompt = build_prompt(
            task_tpl,
            examples,
            language,
            tags,
            doc["text"],
            fmt,
            proposals=proposals,
            proposal_views=proposal_views,
            tag_catalog=tag_catalog,
            span_policy=span_policy,
            input_render=input_render,
            proposal_render=proposal_render,
            guidance=guidance,
            language_rules=language_rules,
        )
        samples.append(
            {
                "language": language,
                "example_language": example_language,
                **(
                    {"language_rules_language": resolve_language_rules(language_rules, language)[0]}
                    if language_rules is not None
                    else {}
                ),
                "document_id": doc["id"],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "proposal_count": len(proposals) if proposals is not None else 0,
                "proposal_view_counts": {
                    source: len(view_proposals) for source, view_proposals in proposal_views
                },
                "prompt": prompt,
            }
        )
    contract = {
        "version": (
            8
            if language_rules is not None
            else 7
            if guidance_field is not None
            else 6
            if input_render != "plain" or proposal_render != "markup"
            else 5
            if span_policy != "legacy"
            else (
                4
                if tag_catalog is not None
                else (3 if proposal_view_fields else (2 if proposal_field is not None else 1))
            )
        ),
        "model": model,
        "format": fmt,
        "input_render": input_render,
        "proposal_render": proposal_render,
        "proposal_field": proposal_field,
        "guidance_field": guidance_field,
        "proposal_views": [{"source": source, "field": field} for source, field in proposal_view_fields],
        "tags": list(tags),
        "task_template_sha256": hashlib.sha256(task_tpl.encode()).hexdigest(),
        "examples_sha256": hashlib.sha256(
            json.dumps(examples, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "samples": samples,
    }
    if unicode_normalization != "none":
        contract["unicode_normalization"] = unicode_normalization
    if language_rules is not None:
        contract["language_rules"] = {"sha256": language_rules_sha256, "languages": sorted(language_rules)}
    if tag_catalog is not None:
        contract["tag_catalog"] = {
            "sha256": tag_catalog_sha256,
            "configured_entries": len(tags),
            "available_entries": len(tag_catalog),
            "rendered_sha256": hashlib.sha256(render_tag_catalog(tags, tag_catalog).encode()).hexdigest(),
        }
    if span_policy != "legacy":
        policy = SPAN_POLICIES[span_policy]
        contract["span_policy"] = {
            "name": span_policy,
            "sha256": hashlib.sha256(policy.encode()).hexdigest(),
            "text": policy,
        }
    if model_revision is not None:
        contract["model_revision"] = model_revision
    return contract


def inventory_tag(label, tagset):
    """Resolve model-emitted tag case to the exact configured inventory label."""
    normalized = str(label).strip().lower()
    return next((tag for tag in tagset if tag.lower() == normalized), None)


def source_occurrences(text: str, surface: str) -> list[int]:
    """Return every overlapping exact occurrence of a nonempty source surface."""
    if not surface:
        return []
    occurrences = []
    search_start = 0
    while (index := text.find(surface, search_start)) >= 0:
        occurrences.append(index)
        search_start = index + 1
    return occurrences


def lexical_source_occurrences(text: str, surface: str) -> list[int]:
    """Return exact occurrences that do not cut through Unicode word characters."""
    occurrences = source_occurrences(text, surface)
    if not surface:
        return occurrences
    require_left_boundary = surface[0].isalnum() or surface[0] == "_"
    require_right_boundary = surface[-1].isalnum() or surface[-1] == "_"
    return [
        start
        for start in occurrences
        if (
            not require_left_boundary
            or start == 0
            or not (text[start - 1].isalnum() or text[start - 1] == "_")
        )
        and (
            not require_right_boundary
            or start + len(surface) == len(text)
            or not (text[start + len(surface)].isalnum() or text[start + len(surface)] == "_")
        )
    ]


def reader_graphemes(text: str) -> list[str]:
    """Return NFKC/casefolded extended grapheme clusters for repair ranking."""
    return regex.findall(r"\X", unicodedata.normalize("NFKC", text).casefold())


def bounded_reader_distance(left: str, right: str, maximum: int) -> int | None:
    """Return grapheme Damerau-Levenshtein distance up to ``maximum``."""
    left_graphemes = reader_graphemes(left)
    right_graphemes = reader_graphemes(right)
    if abs(len(left_graphemes) - len(right_graphemes)) > maximum:
        return None

    sentinel = len(left_graphemes) + len(right_graphemes)
    matrix = [[0 for _right in range(len(right_graphemes) + 2)] for _left in range(len(left_graphemes) + 2)]
    matrix[0][0] = sentinel
    for left_index in range(len(left_graphemes) + 1):
        matrix[left_index + 1][0] = sentinel
        matrix[left_index + 1][1] = left_index
    for right_index in range(len(right_graphemes) + 1):
        matrix[0][right_index + 1] = sentinel
        matrix[1][right_index + 1] = right_index

    last_row: dict[str, int] = {}
    for left_index, left_grapheme in enumerate(left_graphemes, 1):
        last_match_column = 0
        for right_index, right_grapheme in enumerate(right_graphemes, 1):
            transposition_row = last_row.get(right_grapheme, 0)
            transposition_column = last_match_column
            substitution_cost = 1
            if left_grapheme == right_grapheme:
                substitution_cost = 0
                last_match_column = right_index
            matrix[left_index + 1][right_index + 1] = min(
                matrix[left_index][right_index] + substitution_cost,
                matrix[left_index + 1][right_index] + 1,
                matrix[left_index][right_index + 1] + 1,
                matrix[transposition_row][transposition_column]
                + (left_index - transposition_row - 1)
                + 1
                + (right_index - transposition_column - 1),
            )
        last_row[left_grapheme] = left_index
    distance = matrix[-1][-1]
    return distance if distance <= maximum else None


def normalized_surface_candidates(text: str, surface: str, occurrence_count: int) -> set[str]:
    normalized = unicodedata.normalize("NFKC", surface)
    maximum_length = min(
        len(text),
        max(len(surface), len(normalized)) + max(8, len(surface) // 2),
    )
    candidates = set()
    for start in range(len(text)):
        for end in range(start + 1, min(len(text), start + maximum_length) + 1):
            candidate = text[start:end]
            if (
                unicodedata.normalize("NFKC", candidate) == normalized
                and len(source_occurrences(text, candidate)) == occurrence_count
            ):
                candidates.add(candidate)
    return candidates


def edited_surface_candidates(
    text: str,
    surface: str,
    occurrence_count: int,
    maximum: int,
) -> dict[str, int]:
    normalized = unicodedata.normalize("NFKC", surface)
    surface_length = len(reader_graphemes(normalized))
    minimum_length = max(1, surface_length - maximum)
    maximum_length = surface_length + maximum
    grapheme_matches = list(regex.finditer(r"\X", text))
    candidates = {}
    seen = set()
    for start_index, start_match in enumerate(grapheme_matches):
        for candidate_length in range(minimum_length, maximum_length + 1):
            end_index = start_index + candidate_length
            if end_index > len(grapheme_matches):
                break
            start = start_match.start()
            end = grapheme_matches[end_index - 1].end()
            candidate = text[start:end]
            if candidate in seen:
                continue
            seen.add(candidate)
            distance = bounded_reader_distance(normalized, candidate, maximum)
            if distance is None or len(source_occurrences(text, candidate)) != occurrence_count:
                continue
            candidates[candidate] = distance
    return candidates


def align_grouped_surface(
    text: str,
    emitted_surface: str,
    occurrence_count: int,
) -> tuple[str | None, list[int], str, int | None]:
    canonical_surface = unicodedata.normalize("NFKC", emitted_surface)
    exact = source_occurrences(text, canonical_surface)
    if exact:
        if len(exact) == occurrence_count:
            method = "exact" if canonical_surface == emitted_surface else "nfkc"
            return canonical_surface, exact, method, 0
        return None, [], "occurrence_count_mismatch", None

    normalized = normalized_surface_candidates(text, emitted_surface, occurrence_count)
    if len(normalized) == 1:
        source_surface = next(iter(normalized))
        return source_surface, source_occurrences(text, source_surface), "nfkc", 0
    if len(normalized) > 1:
        return None, [], "ambiguous_repair", None

    maximum = 2 if len(reader_graphemes(emitted_surface)) >= 12 else 1
    edited = edited_surface_candidates(text, emitted_surface, occurrence_count, maximum)
    if not edited:
        return None, [], "unmatched", None
    emitted_length = len(reader_graphemes(emitted_surface))

    def candidate_rank(candidate: str, distance: int) -> tuple[int, int]:
        candidate_length = len(reader_graphemes(candidate))
        return distance, abs(candidate_length - emitted_length)

    best_rank = min(candidate_rank(candidate, distance) for candidate, distance in edited.items())
    best = [
        candidate
        for candidate, distance in edited.items()
        if candidate_rank(candidate, distance) == best_rank
    ]
    if len(best) != 1:
        return None, [], "ambiguous_repair", None
    source_surface = best[0]
    return source_surface, source_occurrences(text, source_surface), "edit", best_rank[0]


def align_grouped_lexical_surface(
    text: str,
    emitted_surface: str,
    occurrence_count: int,
    emitted_surfaces: list[str],
) -> tuple[str | None, list[int], str, int | None]:
    """Align complete lexical occurrences, excluding covered nested matches when unambiguous."""
    canonical_surface = unicodedata.normalize("NFKC", emitted_surface)
    exact = lexical_source_occurrences(text, canonical_surface)
    if len(exact) == occurrence_count:
        method = "exact" if canonical_surface == emitted_surface else "nfkc"
        return canonical_surface, exact, method, 0
    if not exact:
        return None, [], "unmatched", None

    covering_ranges = []
    for other_surface in emitted_surfaces:
        canonical_other = unicodedata.normalize("NFKC", other_surface)
        if canonical_other == canonical_surface or len(canonical_other) <= len(canonical_surface):
            continue
        covering_ranges.extend(
            (start, start + len(canonical_other))
            for start in lexical_source_occurrences(text, canonical_other)
        )
    uncovered = [
        start
        for start in exact
        if not any(
            cover_start <= start and start + len(canonical_surface) <= cover_end
            for cover_start, cover_end in covering_ranges
        )
    ]
    if len(uncovered) == occurrence_count:
        method = "exact_nested_exclusion"
        return canonical_surface, uncovered, method, 0
    return None, [], "occurrence_count_mismatch", None


def parse_labels_grouped(
    raw,
    text,
    tagset,
    *,
    alignment_repairs: list[dict] | None = None,
    label_sets: list[dict] | None = None,
    lexical_occurrences: bool = False,
    allow_empty_positive_groups: bool = False,
):
    """Align one permissive label set to every occurrence of each grouped surface."""
    stats = {
        "bad_json": 0,
        "unknown_tag": 0,
        "unmatched": 0,
        "occurrence_count_mismatch": 0,
        "ambiguous_repair": 0,
        "invalid_group": 0,
        "duplicate_surface": 0,
        "repaired_nfkc": 0,
        "repaired_edit": 0,
        "quarantined": 0,
    }
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        stats["bad_json"] = 1
        stats["quarantined"] = 1
        return [], stats
    try:
        groups = json.loads(match.group(0))
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        stats["quarantined"] = 1
        return [], stats
    if not isinstance(groups, list):
        stats["invalid_group"] = 1
        stats["quarantined"] = 1
        return [], stats

    emitted_surfaces = [
        group["t"] for group in groups if isinstance(group, dict) and isinstance(group.get("t"), str)
    ]
    parsed_label_sets = []
    seen_emitted_surfaces = set()
    seen_source_surfaces = set()
    fatal = False
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("t"), str):
            stats["invalid_group"] += 1
            fatal = True
            continue
        emitted_surface = group["t"]
        type_slots = group.get("types")
        if not emitted_surface or not isinstance(type_slots, list) or not type_slots:
            stats["invalid_group"] += 1
            fatal = True
            continue
        canonical_emitted_surface = unicodedata.normalize("NFKC", emitted_surface)
        if canonical_emitted_surface in seen_emitted_surfaces:
            stats["duplicate_surface"] += 1
            fatal = True
            continue
        seen_emitted_surfaces.add(canonical_emitted_surface)

        if lexical_occurrences:
            source_surface, occurrences, method, distance = align_grouped_lexical_surface(
                text,
                emitted_surface,
                len(type_slots),
                emitted_surfaces,
            )
        else:
            source_surface, occurrences, method, distance = align_grouped_surface(
                text,
                emitted_surface,
                len(type_slots),
            )
        if source_surface is None:
            stats[method] += 1
            fatal = True
            continue
        if source_surface in seen_source_surfaces:
            stats["duplicate_surface"] += 1
            fatal = True
            continue
        seen_source_surfaces.add(source_surface)
        if method not in {"exact", "exact_nested_exclusion"}:
            stats[f"repaired_{method}"] += 1
            if alignment_repairs is not None:
                alignment_repairs.append(
                    {
                        "emitted_surface": emitted_surface,
                        "source_surface": source_surface,
                        "method": method,
                        "edit_distance": distance,
                        "occurrence_count": len(occurrences),
                    }
                )

        group_has_positive = False
        for start, raw_types in zip(occurrences, type_slots, strict=True):
            if not isinstance(raw_types, list) or not raw_types:
                stats["invalid_group"] += 1
                fatal = True
                continue
            if any(not isinstance(raw_type, str) for raw_type in raw_types):
                stats["invalid_group"] += 1
                fatal = True
                continue
            normalized_names = [raw_type.strip() for raw_type in raw_types]
            o_positions = [name.lower() == "o" for name in normalized_names]
            if any(o_positions):
                if len(normalized_names) != 1:
                    stats["invalid_group"] += 1
                    fatal = True
                continue

            tags = []
            for name in normalized_names:
                tag = inventory_tag(name, tagset)
                if tag is None:
                    stats["unknown_tag"] += 1
                    fatal = True
                elif tag in tags:
                    stats["invalid_group"] += 1
                    fatal = True
                else:
                    tags.append(tag)
            if not tags:
                continue
            group_has_positive = True
            parsed_label_sets.append(
                {
                    "start": start,
                    "end": start + len(source_surface),
                    "labels": tags,
                }
            )
        if not group_has_positive and not allow_empty_positive_groups:
            stats["invalid_group"] += 1
            fatal = True

    if fatal:
        stats["quarantined"] = 1
        return [], stats
    parsed_label_sets.sort(key=lambda item: (item["start"], item["end"], item["labels"]))
    if label_sets is not None:
        label_sets.extend(parsed_label_sets)
    predictions = [
        {"start": item["start"], "end": item["end"], "label": label}
        for item in parsed_label_sets
        for label in item["labels"]
    ]
    return predictions, stats


def parse_labels_grouped_lexical(
    raw,
    text,
    tagset,
    *,
    alignment_repairs: list[dict] | None = None,
    label_sets: list[dict] | None = None,
):
    """Parse grouped lexical occurrences and discard transport-only all-O groups."""
    return parse_labels_grouped(
        raw,
        text,
        tagset,
        alignment_repairs=alignment_repairs,
        label_sets=label_sets,
        lexical_occurrences=True,
        allow_empty_positive_groups=True,
    )


def parse_labels_seq(raw, text, tagset):
    """Align ordered surfaces at nondecreasing source starts.

    Equal starts preserve nested proposal alternatives. Repeated identical
    surface/type items consume distinct occurrences when available. When a
    surface occurs both inside the preceding surface and later on its own,
    prefer the non-overlapping occurrence; otherwise preserve an intentional
    nested proposal. Duplicates remain explicit for the layer validator.
    """
    m = re.search(r"\[.*\]", raw, re.S)
    stats = {"bad_json": 0, "unknown_tag": 0, "unmatched": 0, "out_of_order": 0}
    if not m:
        stats["bad_json"] = 1
        return [], stats
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        return [], stats
    preds, previous_start, previous_end = [], 0, 0
    used_typed_spans = set()
    for it in items:
        if not isinstance(it, dict) or "t" not in it:
            continue
        if str(it.get("type", "")).strip() == "O":
            # A positional O is the program's "deliberately untagged" marker
            # (json-groups convention); it is an omission, not an unknown tag.
            stats["explicit_o"] = stats.get("explicit_o", 0) + 1
            continue
        tag = inventory_tag(it.get("type", ""), tagset)
        if tag is None:
            stats["unknown_tag"] += 1
            continue
        surface = str(it["t"])
        occurrences = []
        search_start = 0
        while (idx := text.find(surface, search_start)) >= 0:
            occurrences.append(idx)
            search_start = idx + 1
        if not occurrences:
            stats["unmatched"] += 1
            continue
        eligible = [idx for idx in occurrences if idx >= previous_start]
        candidates = eligible or occurrences
        unused_candidates = [
            candidate
            for candidate in candidates
            if (candidate, candidate + len(surface), tag) not in used_typed_spans
        ]
        non_overlapping = [candidate for candidate in unused_candidates if candidate >= previous_end]
        idx = next(
            iter(non_overlapping or unused_candidates),
            candidates[0],
        )
        if idx < previous_start:
            stats["out_of_order"] += 1
        preds.append({"start": idx, "end": idx + len(surface), "label": tag})
        used_typed_spans.add((idx, idx + len(surface), tag))
        previous_start = idx
        previous_end = idx + len(surface)
    return preds, stats


MECHANICAL_ALIGNMENT_RULE = "mechanical-affix-alignment-v1"
HEBREW_ARTICLE_ABSORBING_PREFIXES = frozenset("בלכ")
ARABIC_CONJUNCTION_PREFIXES = frozenset("وف")
MECHANICAL_STEM_MIN_GRAPHEMES = 3
MECHANICAL_EXTENSION_MAX_GRAPHEMES = 5


def _word_character(character: str) -> bool:
    return character.isalnum() or unicodedata.category(character).startswith("M")


def _written_word_end(text: str, index: int) -> int:
    while index < len(text) and _word_character(text[index]):
        index += 1
    return index


def propose_affix_alignment(items: list[dict], text: str) -> list[dict]:
    """Propose deinflection repairs for emitted surfaces that are absent from the text.

    Two mechanical cases, each accepted only with exactly one candidate:
    a dictionary form whose final grapheme the text rewrote under an attached
    ending (Tamil ஆய்வகம் for ஆய்வகத்தில்) becomes the whole written word; a
    Hebrew surface that restored an article absorbed by ב, ל or כ loses the ה.
    The repairs use the reviewed-alignment record so provenance stays explicit.
    """
    repairs = []
    for position, item in enumerate(items, 1):
        surface = item["t"]
        if not surface or surface in text:
            continue
        candidates = set()
        graphemes = regex.findall(r"\X", surface)
        if len(graphemes) > MECHANICAL_STEM_MIN_GRAPHEMES:
            stem = "".join(graphemes[:-1])
            for start in source_occurrences(text, stem):
                if start and _word_character(text[start - 1]):
                    continue
                stem_end = start + len(stem)
                end = _written_word_end(text, stem_end)
                extension = regex.findall(r"\X", text[stem_end:end])
                if 0 < len(extension) <= MECHANICAL_EXTENSION_MAX_GRAPHEMES:
                    candidates.add((start, end, "stem-rewrite"))
        if surface.startswith("ה") and len(surface) > 2:
            rest = surface[1:]
            for start in source_occurrences(text, rest):
                if (
                    start
                    and text[start - 1] in HEBREW_ARTICLE_ABSORBING_PREFIXES
                    and (start == 1 or not _word_character(text[start - 2]))
                ):
                    candidates.add((start, start + len(rest), "absorbed-article"))
        if surface.startswith("ال") and len(surface) > 3:
            # Arabic ل + الـ is written لل: the article's alif is dropped, its lam kept.
            written = "ل" + surface[2:]
            for start in source_occurrences(text, "ل" + written):
                before = text[start - 1] if start else ""
                if (
                    not before
                    or not _word_character(before)
                    or (
                        before in ARABIC_CONJUNCTION_PREFIXES
                        and (start == 1 or not _word_character(text[start - 2]))
                    )
                ):
                    candidates.add((start + 1, start + 1 + len(written), "absorbed-alif"))
        if len(candidates) != 1:
            continue
        start, end, kind = candidates.pop()
        if source_occurrences(text, text[start:end]) != [start]:
            continue
        repairs.append(
            {
                "i": position,
                "start": start,
                "end": end,
                "emitted": surface,
                "source_surface": text[start:end],
                "kind": "deinflection",
                "reason": f"{MECHANICAL_ALIGNMENT_RULE}:{kind}",
            }
        )
    return repairs


def mechanical_alignment(raw: str, text: str) -> dict | None:
    """Wrap proposed repairs as a source- and response-bound alignment record."""
    items = json.loads(raw)
    if not isinstance(items, list) or any(not isinstance(item, dict) or "t" not in item for item in items):
        return None
    repairs = propose_affix_alignment(items, text)
    repaired = {repair["i"]: repair["source_surface"] for repair in repairs}
    surfaces = [repaired.get(position, item["t"]) for position, item in enumerate(items, 1)]
    positions = [source_occurrences(text, surface) for surface in surfaces]
    # Out-of-order output is repairable only when every surface has one source position.
    order = None
    if all(len(found) == 1 for found in positions):
        keys = [(found[0], -len(surface)) for found, surface in zip(positions, surfaces)]
        if keys != sorted(keys):
            order = "unique-source-position"
    if not repairs and order is None:
        return None
    alignment = {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "response_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "reviewer": MECHANICAL_ALIGNMENT_RULE,
        "repairs": repairs,
    }
    if order is not None:
        alignment["order"] = order
    return alignment


def validated_annotation_sequence(
    raw: str, text: str, tagset: set[str], *, reviewed_alignment: dict | None = None
) -> tuple[list[dict], dict]:
    """Preserve reference/name nesting and apply only source-bound reviewed alignments."""
    items = json.loads(raw)
    if not isinstance(items, list) or any(
        not isinstance(item, dict)
        or set(item) != {"i", "t", "type"}
        or not isinstance(item["t"], str)
        or not item["t"]
        or type(item["i"]) is not int
        for item in items
    ):
        raise ValueError("Expected the exact annotation array schema")
    if [item["i"] for item in items] != list(range(1, len(items) + 1)):
        raise ValueError("Invalid sequence numbering")
    details = {}
    repairs = []
    occurrence_repairs = {}
    if reviewed_alignment is not None:
        if (
            reviewed_alignment["source_sha256"] != hashlib.sha256(text.encode()).hexdigest()
            or reviewed_alignment["response_text_sha256"] != hashlib.sha256(raw.encode()).hexdigest()
        ):
            raise ValueError("Reviewed alignment source or response hash differs")
        seen = set()
        for repair in reviewed_alignment["repairs"]:
            if any(type(repair[key]) is not int for key in ("i", "start", "end")):
                raise ValueError("Reviewed alignment locators must be integers")
            index = repair["i"] - 1
            start, end = repair["start"], repair["end"]
            if (
                index in seen
                or not 0 <= index < len(items)
                or not 0 <= start < end <= len(text)
                or not repair["reason"]
            ):
                raise ValueError("Invalid reviewed alignment locator")
            item = items[index]
            if item["t"] != repair["emitted"] or text[start:end] != repair["source_surface"]:
                raise ValueError("Reviewed alignment literal differs")
            kind = repair.get("kind", "deinflection")
            if kind == "occurrence":
                if item["t"] != repair["source_surface"] or len(source_occurrences(text, item["t"])) < 2:
                    raise ValueError("Reviewed occurrence requires an unchanged repeated literal surface")
                occurrence_repairs[index] = (start, end)
            elif kind == "deinflection":
                if item["t"] in text:
                    raise ValueError("Reviewed alignment must not replace an already literal surface")
                if source_occurrences(text, repair["source_surface"]) != [start]:
                    raise ValueError("Reviewed alignment needs one unambiguous source occurrence")
            else:
                raise ValueError(f"Unknown reviewed alignment kind: {kind}")
            seen.add(index)
            item["t"] = repair["source_surface"]
            repairs.append(dict(repair))
        order = reviewed_alignment.get("order")
        if order not in (None, "unique-source-position"):
            raise ValueError(f"Unknown reviewed alignment order: {order}")
        if not repairs and order is None:
            raise ValueError("Empty reviewed alignment")
        if order is not None:
            # Every surface occurs once, so its source position orders it losslessly;
            # at equal starts the longer (containing) span comes first.
            keyed = []
            for item in items:
                occurrences = source_occurrences(text, item["t"])
                if len(occurrences) != 1:
                    raise ValueError("Source-order repair requires every surface to occur exactly once")
                keyed.append(((occurrences[0], -len(item["t"])), item))
            items = [item for _, item in sorted(keyed, key=lambda pair: pair[0])]
            details["alignment_order"] = order
        details["alignment_repairs"] = repairs
        details["alignment_review"] = reviewed_alignment
    preds, stats = parse_labels_seq(json.dumps(items, ensure_ascii=False), text, tagset)
    # A reviewed occurrence can repair provisional order; final order is checked below.
    fatal_parse = any(
        value for key, value in stats.items() if key != "out_of_order" or not occurrence_repairs
    )
    if fatal_parse or len(preds) != len(items):
        raise ValueError(f"Literal span validation failed: {stats}")
    # Corrections must not silently move unreviewed proposals via the parser cursor.
    for index, (start, end) in occurrence_repairs.items():
        preds[index] = {**preds[index], "start": start, "end": end}
    if any(left["start"] > right["start"] for left, right in zip(preds, preds[1:])):
        raise ValueError("Reviewed alignment violates source order")
    references = {"person_reference", "organization_reference"}
    nested = []
    for index, left in enumerate(preds):
        for right in preds[index + 1 :]:
            if left["start"] >= right["end"] or right["start"] >= left["end"]:
                continue
            if (left["label"] in references) == (right["label"] in references):
                raise ValueError("Overlapping spans within one annotation layer")
            ref, named = (left, right) if left["label"] in references else (right, left)
            if not (
                ref["start"] <= named["start"] < named["end"] <= ref["end"]
                and (ref["start"], ref["end"]) != (named["start"], named["end"])
            ):
                raise ValueError("Overlapping spans are not a reference containing a named entity")
            nested.append((ref, named))
    if nested:
        omitted = {tuple(ref[key] for key in ("start", "end", "label")) for ref, _ in nested}
        details["flat_projection"] = {
            "policy": "named-before-containing-reference-v1",
            "preds": [p for p in preds if tuple(p[k] for k in ("start", "end", "label")) not in omitted],
            "omitted_references": [
                p for p in preds if tuple(p[k] for k in ("start", "end", "label")) in omitted
            ],
        }
    return preds, details


def _parse_labels_offsets(raw, text, tagset, *, confidence_field=None):
    """Validate model-supplied half-open character offsets without repair."""
    stats = {
        "bad_json": 0,
        "invalid_item": 0,
        "unknown_tag": 0,
        "invalid_bounds": 0,
        "surface_mismatch": 0,
        "duplicate_typed_span": 0,
        "out_of_order": 0,
        "quarantined": 0,
    }
    if confidence_field is not None:
        stats["invalid_repair_confidence"] = 0
    try:
        items = json.loads(raw.strip())
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        stats["quarantined"] = 1
        return [], stats
    if not isinstance(items, list):
        stats["invalid_item"] = 1
        stats["quarantined"] = 1
        return [], stats

    predictions = []
    seen = set()
    previous_start = -1
    fatal = False
    required = {"start", "end", "t", "type"}
    if confidence_field is not None:
        required.add(confidence_field)
    for item in items:
        if not isinstance(item, dict) or set(item) != required:
            stats["invalid_item"] += 1
            fatal = True
            continue
        start = item["start"]
        end = item["end"]
        surface = item["t"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            stats["invalid_bounds"] += 1
            fatal = True
            continue
        if not isinstance(surface, str) or text[start:end] != surface:
            stats["surface_mismatch"] += 1
            fatal = True
            continue
        tag = inventory_tag(item["type"], tagset)
        if tag is None:
            stats["unknown_tag"] += 1
            fatal = True
            continue
        confidence = None
        if confidence_field is not None:
            confidence = item[confidence_field]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                stats["invalid_repair_confidence"] += 1
                fatal = True
                continue
        if start < previous_start:
            stats["out_of_order"] += 1
            fatal = True
        previous_start = start
        typed_span = (start, end, tag)
        if typed_span in seen:
            stats["duplicate_typed_span"] += 1
            fatal = True
            continue
        seen.add(typed_span)
        prediction = {"start": start, "end": end, "label": tag}
        if confidence_field is not None:
            prediction[confidence_field] = float(confidence)
        predictions.append(prediction)

    if fatal:
        stats["quarantined"] = 1
        return [], stats
    return predictions, stats


def parse_labels_offsets(raw, text, tagset):
    return _parse_labels_offsets(raw, text, tagset)


def parse_labels_offsets_confidence(raw, text, tagset):
    return _parse_labels_offsets(raw, text, tagset, confidence_field="repair_confidence")


INLINE_RE = re.compile(r"\[([^\[\]]{1,120})\]\(([A-Za-z_]{2,40})\)")


def parse_labels_inline(raw, text, tagset):
    """inline: reconstruct offsets by stripping marks and verifying the
    residue tracks the original text; mutation breaks alignment for the
    remainder (counted)."""
    stats = {"bad_json": 0, "unknown_tag": 0, "unmatched": 0, "mutated": 0}
    preds, cursor = [], 0
    for m in INLINE_RE.finditer(raw):
        surface = m.group(1)
        tag = inventory_tag(m.group(2), tagset)
        if tag is None:
            stats["unknown_tag"] += 1
            continue
        idx = text.find(surface, cursor)
        if idx < 0:
            stats["unmatched"] += 1
            continue
        preds.append({"start": idx, "end": idx + len(surface), "label": tag})
        cursor = idx + 1
    # crude mutation check: unmarked residue should mostly exist in text
    residue = INLINE_RE.sub(lambda mm: mm.group(1), raw).strip()
    if residue and text and abs(len(residue) - len(text)) > max(20, len(text) // 10):
        stats["mutated"] = 1
    return preds, stats


REDACT_RE = re.compile(r"\[([A-Z_]{2,40})_(\d+)\]")


def parse_labels_redact(raw, text, tagset):
    """redact: two-cursor sync. Each placeholder consumes original text
    until the output's following literal context re-matches."""
    stats = {"bad_json": 0, "unknown_tag": 0, "unmatched": 0, "desync": 0}
    preds = []
    o = 0  # cursor in original text
    pos = 0  # cursor in model output
    for m in REDACT_RE.finditer(raw):
        tag = inventory_tag(m.group(1), tagset)
        # literal chunk between previous token and this one should match
        lit = raw[pos : m.start()]
        li = text.find(lit[-30:], o) if lit else o
        if lit and li < 0:
            stats["desync"] += 1
            pos = m.end()
            continue
        start = (li + len(lit[-30:])) if lit else o
        # span end: find the next literal context after this token
        after = raw[m.end() : m.end() + 30]
        after = REDACT_RE.split(after)[0][:20]
        if after.strip():
            end = text.find(after, start)
            if end < 0:
                stats["unmatched"] += 1
                pos = m.end()
                continue
        else:
            end = len(text)
        if tag is not None and end > start:
            preds.append({"start": start, "end": end, "label": tag})
        elif tag is None:
            stats["unknown_tag"] += 1
        o = end
        pos = m.end()
    return preds, stats


def parse_labels(raw, text, tagset):
    """First JSON array in raw -> aligned spans; unmatched surfaces and
    unknown tags are dropped (counted)."""
    m = re.search(r"\[.*\]", raw, re.S)
    stats = {"bad_json": 0, "unknown_tag": 0, "unmatched": 0}
    if not m:
        stats["bad_json"] = 1
        return [], stats
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        return [], stats
    preds = []
    for it in items:
        if not isinstance(it, dict) or "t" not in it:
            continue
        tag = inventory_tag(it.get("type", ""), tagset)
        if tag is None:
            stats["unknown_tag"] += 1
            continue
        surface = str(it["t"])
        n = int(it.get("n", 1) or 1)
        start, found = -1, 0
        pos = 0
        while True:
            idx = text.find(surface, pos)
            if idx < 0:
                break
            found += 1
            if found == n:
                start = idx
                break
            pos = idx + 1
        if start < 0:
            stats["unmatched"] += 1
            continue
        preds.append({"start": start, "end": start + len(surface), "label": tag})
    return preds, stats


def validated_resume_count(path, docs):
    """Require an existing output to be an exact prefix of the input."""
    completed = 0
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed resume row") from error
            if completed >= len(docs):
                raise ValueError(f"{path}:{line_number}: resume output is longer than the input")
            expected_id = docs[completed]["id"]
            if row.get("id") != expected_id:
                raise ValueError(
                    f"{path}:{line_number}: resume id {row.get('id')!r} "
                    f"does not match input prefix id {expected_id!r}"
                )
            completed += 1
    return completed


def session_turn_messages(history: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    return [*history, {"role": "user", "content": prompt}]


def advanced_session_history(
    history: list[dict[str, str]],
    prompt: str,
    raw: str,
) -> list[dict[str, str]]:
    return [
        *history,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": raw},
    ]


def session_turn_token_ids(
    renderer,
    history: list[dict[str, str]],
    prompt: str,
    committed_token_ids: list[int] | tuple[int, ...] | None,
) -> list[int]:
    """Render one turn without re-tokenizing the committed assistant replies."""
    messages = session_turn_messages(history, prompt)
    if not history:
        return renderer.token_ids(messages, add_generation_prompt=True)
    if not committed_token_ids:
        raise ValueError("nonempty session history requires a committed context handle")
    base = renderer.text(history, add_generation_prompt=False).rstrip("\n")
    extended = renderer.text(messages, add_generation_prompt=True)
    if not extended.startswith(base):
        raise RuntimeError("chat template rewrote committed history instead of appending a turn")
    delta = renderer.tokenizer.encode(extended[len(base) :], add_special_tokens=False)
    if not delta:
        raise RuntimeError("chat template produced no uncached tokens for the next turn")
    return [*committed_token_ids, *delta]


def persist_loaded_model(model, tokenizer, destination: str | os.PathLike[str]) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite model checkpoint: {target}")
    staging = target.with_name(f".{target.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale model-checkpoint staging path exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    staging.replace(target)
    return target


def annotation_base_spans(doc: dict, guidance_field: str | None = None) -> list:
    """Return authoritative carriers directly or from frozen guidance JSON."""
    base_spans = doc.get("base_spans")
    if isinstance(base_spans, list):
        return base_spans
    if guidance_field is not None:
        guidance = doc.get(guidance_field)
        if isinstance(guidance, str):
            try:
                guidance_payload = json.loads(guidance)
            except json.JSONDecodeError as error:
                raise ValueError(f"{guidance_field} is not JSON") from error
            base_spans = guidance_payload.get("base_spans") if isinstance(guidance_payload, dict) else None
            if isinstance(base_spans, list):
                return base_spans
    raise ValueError("subclass annotation requires authoritative base_spans on every input row")


def annotation_candidate_ledger(doc: dict, guidance_field: str | None) -> list:
    """Return the controller-enumerated candidate ledger from frozen guidance JSON."""
    if guidance_field is None:
        raise ValueError("candidate-subclass annotation requires a guidance field")
    guidance = doc.get(guidance_field)
    if not isinstance(guidance, str):
        raise ValueError(f"{guidance_field} must be a JSON string")
    try:
        payload = json.loads(guidance)
    except json.JSONDecodeError as error:
        raise ValueError(f"{guidance_field} is not JSON") from error
    candidate_ledger = payload.get("candidate_ledger") if isinstance(payload, dict) else None
    if not isinstance(candidate_ledger, list):
        raise ValueError(f"{guidance_field} has no candidate_ledger list")
    return candidate_ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--gold", required=True, help="docs jsonl ({id,text,...}); spans ignored")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--task-template",
        default=None,
        help=(
            "task-template path; defaults to the installed task selected by proposal mode, "
            "so a candidate prompt can be tested without replacing the historical control"
        ),
    )
    ap.add_argument(
        "--examples",
        default=None,
        help=(
            "language-example JSON path; defaults to prompts/pii-label/examples.json, "
            "so candidate examples can be tested without replacing the historical control"
        ),
    )
    ap.add_argument(
        "--allow-example-fallback",
        action="store_true",
        help="allow cross-language (normally English) shots only for historical replay",
    )
    ap.add_argument("--lang", default="en", help="BCP-47 tag for examples")
    ap.add_argument("--tags", default=None, help="comma list of tags (default: core inventory)")
    ap.add_argument(
        "--tags-contract",
        default=None,
        help="load the exact ordered tag inventory from an existing prompt-contract JSON",
    )
    ap.add_argument(
        "--tag-catalog",
        default=None,
        help=(
            "optional Markdown label catalog; every configured tag must have one definition "
            "and prototypical surface"
        ),
    )
    ap.add_argument(
        "--span-policy",
        choices=SPAN_POLICIES,
        default="legacy",
        help="optional named span-selection contract; legacy preserves the historical prompt",
    )
    ap.add_argument("--fmt", choices=FORMATS, default="json")
    ap.add_argument(
        "--input-render",
        choices=("plain", "json-string"),
        default="plain",
        help="render source text directly or as one JSON string with whitespace escapes visible",
    )
    ap.add_argument(
        "--unicode-normalization",
        choices=("none", "NFKC"),
        default="none",
        help="require intake text to already use this canonical Unicode coordinate system",
    )
    ap.add_argument(
        "--proposal-render",
        choices=("markup", "list"),
        default="markup",
        help="render disjoint inline markup or an overlap-preserving candidate list",
    )
    ap.add_argument(
        "--proposal-field",
        default=None,
        help=(
            "JSONL field containing disjoint XLM-R span proposals; when set, render exact surfaces "
            "with the soft-primed prompt and retain offsets only controller-side"
        ),
    )
    ap.add_argument(
        "--guidance-field",
        default=None,
        help="JSONL field containing the non-source controller guidance rendered by {guidance}",
    )
    ap.add_argument(
        "--source-paragraph-guidance",
        action="store_true",
        help="render the paper's ctx-v6 guidance: each row's annotation_guidance, else its "
        "document_context before/after text joined around the sentence as source paragraph",
    )
    ap.add_argument(
        "--subclass-spec",
        default=None,
        help="closed ontology-v3 categorical and Bernoulli contract for subclass formats",
    )
    ap.add_argument(
        "--proposal-view",
        action="append",
        default=[],
        metavar="SOURCE=FIELD",
        help=(
            "independent proposal source and JSONL field; repeat to render separate marked copies "
            "of the original text without merging overlapping detectors"
        ),
    )
    ap.add_argument("--lora", default=None, help="PEFT adapter dir (gemma-4 glue applied)")
    ap.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="load linear weights through bitsandbytes int8 quantization",
    )
    ap.add_argument(
        "--save-quantized-model",
        default=None,
        help="atomically persist the model produced by --load-in-8bit for direct reload",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--id-list",
        help="process only these newline-delimited row ids, preserving source order",
    )
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument(
        "--thinking",
        choices=("off", "on", "auto"),
        default="off",
        help="chat-template thinking mode; parsed output always strips the thinking block",
    )
    ap.add_argument(
        "--prompt-contract-out",
        default=None,
        help="write the frozen prompt inputs plus one rendered sample per requested language before model load",
    )
    ap.add_argument(
        "--prompt-contract-only",
        action="store_true",
        help="write --prompt-contract-out and exit without importing or loading an inference model",
    )
    ap.add_argument(
        "--batch", type=int, default=8, help="segments per generate call (left-padded); >1 needs a pad token"
    )
    ap.add_argument(
        "--session-field",
        default=None,
        help=(
            "input metadata field grouping segments into local chat histories; "
            "use document_id for one session per intake document"
        ),
    )
    ap.add_argument(
        "--session-recovery-retries",
        type=int,
        default=1,
        help="retries forked from the context handle immediately before a rejected segment",
    )
    ap.add_argument(
        "--shared-prefix-cache",
        action="store_true",
        help="prefill the exact token prefix shared by independent requests once",
    )
    ap.add_argument(
        "--shared-prefix-min-tokens",
        type=int,
        default=64,
        help="minimum exact shared prefix required for independent or session-root caching",
    )
    ap.add_argument(
        "--timing-receipt",
        default=None,
        help=("write synchronized prefix-prefill and model.generate timings; independent-request mode only"),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append after an existing output only when its ids exactly match the input prefix",
    )
    args = ap.parse_args()

    proposal_view_fields = []
    for spec in args.proposal_view:
        source, separator, field = spec.partition("=")
        if not separator or not source or not field:
            ap.error(f"--proposal-view must be SOURCE=FIELD, got {spec!r}")
        proposal_view_fields.append((source, field))
    if args.proposal_field and proposal_view_fields:
        ap.error("--proposal-field and --proposal-view are mutually exclusive")
    if args.tags and args.tags_contract:
        ap.error("--tags and --tags-contract are mutually exclusive")
    if args.limit and args.id_list:
        ap.error("--limit and --id-list are mutually exclusive")
    if args.save_quantized_model and not args.load_in_8bit:
        ap.error("--save-quantized-model requires --load-in-8bit")
    if not args.prompt_contract_only and not args.out:
        ap.error("--out is required unless --prompt-contract-only is set")
    if len({source for source, _field in proposal_view_fields}) != len(proposal_view_fields):
        ap.error("--proposal-view source names must be unique")
    if args.session_recovery_retries < 0:
        ap.error("--session-recovery-retries must be nonnegative")
    if args.session_field:
        if args.fmt not in {"json-groups", "json-groups-lexical"}:
            ap.error("--session-field requires a grouped JSON format")
        if args.batch != 1:
            ap.error("--session-field currently requires --batch 1")
        if args.resume:
            ap.error("--session-field is not yet resumable")
    if args.timing_receipt and args.session_field:
        ap.error("--timing-receipt currently requires independent-request mode")

    task_name = (
        "task-multiview.txt"
        if proposal_view_fields
        else ("task-primed.txt" if args.proposal_field else "task.txt")
    )
    task_path = Path(args.task_template or os.path.join(PROMPT_DIR, task_name))
    examples_path = Path(args.examples or os.path.join(PROMPT_DIR, "examples.json"))
    task_tpl, template_assembly = load_prompt_template(task_path, args.model)
    if template_assembly is not None and args.resume:
        if not args.prompt_contract_out:
            raise ValueError("Resuming an included template requires its saved --prompt-contract-out")
        previous = json.loads(Path(args.prompt_contract_out).read_text())
        if previous.get("template_assembly") != template_assembly:
            raise ValueError("Prompt includes differ from the saved resume contract")
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    subclass_spec: SubclassSpec | None = (
        load_subclass_spec(Path(args.subclass_spec)) if args.subclass_spec else None
    )
    subclass_formats = {"json-subclasses", "json-candidate-subclasses"}
    if args.fmt in subclass_formats:
        if subclass_spec is None:
            ap.error(f"--fmt {args.fmt} requires --subclass-spec")
        if task_tpl.count("{subclass_catalog}") != 1:
            raise ValueError(
                f"{args.fmt} task template requires exactly one {{subclass_catalog}} placeholder"
            )
        task_tpl = task_tpl.replace("{subclass_catalog}", render_subclass_catalog(subclass_spec))
    elif subclass_spec is not None:
        ap.error("--subclass-spec is only valid with a subclass JSON format")
    if args.tags_contract:
        tags = json.loads(Path(args.tags_contract).read_text(encoding="utf-8"))["tags"]
    else:
        tags = args.tags.split(",") if args.tags else CORE_TAGS
    tagset = set(tags)
    tag_catalog = load_tag_catalog(args.tag_catalog) if args.tag_catalog else None
    tag_catalog_sha256 = None
    if args.tag_catalog:
        tag_catalog_sha256 = hashlib.sha256(Path(args.tag_catalog).read_bytes()).hexdigest()

    docs = [json.loads(l) for l in open(args.gold)]
    if args.id_list:
        docs = select_docs_by_ids(docs, load_id_list(args.id_list), source=args.gold)
    elif args.limit:
        docs = docs[: args.limit]
    if not args.allow_example_fallback:
        require_language_specific_examples(examples, docs, args.lang)
    if args.fmt in {"json-groups", "json-groups-lexical"} and args.unicode_normalization != "NFKC":
        ap.error("grouped JSON formats require --unicode-normalization NFKC")
    if args.unicode_normalization != "none":
        for index, doc in enumerate(docs):
            if doc["text"] != unicodedata.normalize(args.unicode_normalization, doc["text"]):
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} is not "
                    f"{args.unicode_normalization}-normalized at intake"
                )
    if subclass_spec is not None:
        for doc in docs:
            doc["_subclass_base_spans"] = annotation_base_spans(doc, args.guidance_field)
            if args.fmt == "json-candidate-subclasses":
                doc["_subclass_candidate_ledger"] = annotation_candidate_ledger(
                    doc,
                    args.guidance_field,
                )
    if args.source_paragraph_guidance:
        if args.guidance_field:
            ap.error("--source-paragraph-guidance supplies the guidance field; omit --guidance-field")
        args.guidance_field = add_source_paragraph_guidance(docs, args.lang)
    if args.guidance_field:
        if "{guidance}" not in task_tpl:
            raise ValueError("--guidance-field requires a {guidance} task-template placeholder")
        for index, doc in enumerate(docs):
            guidance = doc.get(args.guidance_field)
            if not isinstance(guidance, str) or not guidance:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks nonempty string guidance "
                    f"field {args.guidance_field!r}"
                )
    if args.session_field:
        for index, doc in enumerate(docs):
            value = doc.get(args.session_field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks nonempty string "
                    f"session field {args.session_field!r}"
                )
    if args.proposal_field:
        for index, doc in enumerate(docs):
            if args.proposal_field not in doc:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks proposal field {args.proposal_field!r}"
                )
    for source, field in proposal_view_fields:
        for index, doc in enumerate(docs):
            if field not in doc:
                raise ValueError(
                    f"input row {index} id={doc.get('id')!r} lacks proposal-view field "
                    f"{field!r} for source {source!r}"
                )
    resume_count = 0
    if args.resume and os.path.exists(args.out):
        resume_count = validated_resume_count(args.out, docs)
        print(f"RESUME: retaining {resume_count}/{len(docs)} completed rows from {args.out}", flush=True)

    if args.prompt_contract_out:
        contract = build_prompt_contract(
            task_tpl,
            examples,
            docs,
            tags,
            args.fmt,
            args.lang,
            args.model,
            proposal_field=args.proposal_field,
            proposal_view_fields=proposal_view_fields,
            tag_catalog=tag_catalog,
            tag_catalog_sha256=tag_catalog_sha256,
            span_policy=args.span_policy,
            model_revision=args.model_revision,
            input_render=args.input_render,
            proposal_render=args.proposal_render,
            unicode_normalization=args.unicode_normalization,
            guidance_field=args.guidance_field,
        )
        contract["task_template_path"] = str(task_path.resolve())
        if template_assembly is not None:
            contract["template_assembly"] = template_assembly
        contract["examples_path"] = str(examples_path.resolve())
        contract["quantization"] = "bitsandbytes-int8" if args.load_in_8bit else "none"
        contract["thinking"] = args.thinking
        if args.id_list:
            contract["input_selection"] = {
                "id_list_path": str(Path(args.id_list).resolve()),
                "id_list_sha256": hashlib.sha256(Path(args.id_list).read_bytes()).hexdigest(),
                "rows": len(docs),
                "order": "source order",
            }
        if args.session_field:
            contract["session"] = {
                "group_field": args.session_field,
                "order": "input order within each group",
                "output_unit": "one input segment",
                "recovery_retries": args.session_recovery_retries,
                "recovery_branch": "fork of the pre-segment context handle",
                "context_reuse": "common exact root plus growing per-document KV handle",
                "materialization_rejection_continues_history": True,
            }
        os.makedirs(os.path.dirname(os.path.abspath(args.prompt_contract_out)), exist_ok=True)
        with open(args.prompt_contract_out, "w") as output:
            json.dump(contract, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(
            f"PROMPT: froze {len(contract['samples'])} rendered language samples "
            f"-> {args.prompt_contract_out}",
            flush=True,
        )
    if args.prompt_contract_only:
        if not args.prompt_contract_out:
            ap.error("--prompt-contract-only requires --prompt-contract-out")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import decodelib

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model_load_options: dict[str, object] = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "revision": args.model_revision,
    }
    if args.load_in_8bit:
        model_load_options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_load_options).eval()
    if args.save_quantized_model:
        started = time.perf_counter()
        saved_model = persist_loaded_model(model, tok, args.save_quantized_model)
        print(
            f"MODEL CHECKPOINT: saved {saved_model} in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    if args.lora:
        from peft import PeftConfig, PeftModel

        from gemma4_lora import register_gemma4_clippable_lora

        lcfg = PeftConfig.from_pretrained(args.lora)
        register_gemma4_clippable_lora(lcfg)  # no-op unless gemma-4 target
        model = PeftModel.from_pretrained(model, args.lora, config=lcfg).eval()
        print(f"loaded LoRA {args.lora}", flush=True)

    pfn = {
        "json": parse_labels,
        "json-seq": parse_labels_seq,
        "json-offsets": parse_labels_offsets,
        "json-offsets-confidence": parse_labels_offsets_confidence,
        "json-groups": parse_labels_grouped,
        "json-groups-lexical": parse_labels_grouped_lexical,
        "inline": parse_labels_inline,
        "redact": parse_labels_redact,
    }.get(args.fmt)
    model_format = ModelFormat(
        tok,
        options=FormatOptions(thinking=args.thinking, strip_thinking_output=True),
        generation_config=model.generation_config,
    )
    renderer = model_format.renderer
    stop_token_ids = set(model_format.stop_token_ids)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    def render_doc_prompt(doc: dict) -> tuple[str, str]:
        key, prompt = build_prompt(
            task_tpl,
            examples,
            doc.get("lang", args.lang),
            tags,
            doc["text"],
            args.fmt,
            proposals=doc[args.proposal_field] if args.proposal_field else None,
            proposal_views=[(source, doc[field]) for source, field in proposal_view_fields],
            tag_catalog=tag_catalog,
            span_policy=args.span_policy,
            input_render=args.input_render,
            proposal_render=args.proposal_render,
            guidance=(doc[args.guidance_field] if args.guidance_field else None),
        )
        return key, prompt

    def tokenize_messages(messages: list[dict[str, str]]) -> list[int]:
        return renderer.token_ids(messages, add_generation_prompt=True)

    def tokenize_doc(doc: dict) -> tuple[str, list[int]]:
        key, prompt = render_doc_prompt(doc)
        return key, tokenize_messages([{"role": "user", "content": prompt}])

    def generate_cached_tokens(
        token_ids: list[int],
        prefix: decodelib.PrefilledPrefix,
        *,
        consume_prefix: bool = False,
    ) -> tuple[str, float, decodelib.PrefilledPrefix, tuple[int, ...]]:
        decode_batch = decodelib.prepare_cached_batch(
            model,
            [token_ids],
            pad_token_id=pad_id,
            padding_side="left",
            prefix=prefix,
            fork_prefix=not consume_prefix,
        )
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **decode_batch.model_inputs(),
                max_new_tokens=args.max_new,
                do_sample=False,
                eos_token_id=sorted(stop_token_ids),
                pad_token_id=pad_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
        new_tokens = generated.sequences[0][decode_batch.output_prompt_width :]
        if not len(new_tokens) or int(new_tokens[-1]) not in stop_token_ids:
            raise RuntimeError("session generation ended without an end-of-turn token")
        raw = tok.decode(new_tokens, skip_special_tokens=True)
        committed = (*prefix.token_ids, *generated.sequences[0].tolist())
        handle = decodelib.prefix_from_generate(model, committed, generated.past_key_values)
        return raw, time.perf_counter() - started, handle, committed

    shared_prefix_ids: list[int] | None = None
    shared_prefix = None
    session_root_backup = None
    prefix_prefill_seconds = None
    timing_batches = []

    def synchronize_timing_device() -> None:
        if args.timing_receipt and torch.device(model.device).type == "cuda":
            torch.cuda.synchronize(model.device)

    if args.session_field and resume_count < len(docs):
        first_docs: dict[str, dict] = {}
        for doc in docs[resume_count:]:
            first_docs.setdefault(doc[args.session_field], doc)
        first_turns = [tokenize_doc(doc)[1] for doc in first_docs.values()]
        session_root_ids, _suffixes = decodelib.split_common_prefix(
            first_turns,
            minimum_tokens=args.shared_prefix_min_tokens,
        )
        session_root_prefix = decodelib.prefill_prefix(model, session_root_ids)
        session_root_backup = session_root_prefix.copy_to(model, "cpu")
        print(
            f"SESSION PREFIX: cached {session_root_prefix.token_count} exact tokens across "
            f"{len(first_docs)} document contexts",
            flush=True,
        )
        del session_root_prefix
        torch.cuda.empty_cache()
    elif args.shared_prefix_cache and resume_count < len(docs):
        for doc in docs[resume_count:]:
            _key, token_ids = tokenize_doc(doc)
            tokens = token_ids
            if shared_prefix_ids is None:
                shared_prefix_ids = tokens[:-1]
            else:
                shared = decodelib.common_prefix_length([shared_prefix_ids, tokens])
                shared_prefix_ids = shared_prefix_ids[: min(shared, len(tokens) - 1)]
            if len(shared_prefix_ids) < args.shared_prefix_min_tokens:
                raise ValueError(
                    f"exact shared prompt prefix has {len(shared_prefix_ids)} usable tokens; "
                    f"minimum is {args.shared_prefix_min_tokens}"
                )
        assert shared_prefix_ids is not None
        synchronize_timing_device()
        prefill_started = time.perf_counter()
        shared_prefix = decodelib.prefill_prefix(model, shared_prefix_ids)
        synchronize_timing_device()
        prefix_prefill_seconds = time.perf_counter() - prefill_started
        print(
            f"PREFIX: cached {shared_prefix.token_count} exact tokens across "
            f"{len(docs) - resume_count} remaining requests",
            flush=True,
        )

    agg = {}
    lat = []
    n_done = resume_count
    session_last_indices = (
        {doc[args.session_field]: index for index, doc in enumerate(docs)} if args.session_field else {}
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a" if resume_count else "w") as out:
        session_histories: dict[str, list[dict[str, str]]] = {}
        session_handles: dict[str, decodelib.PrefilledPrefix] = {}
        session_committed_tokens: dict[str, tuple[int, ...]] = {}
        for b0 in range(resume_count, len(docs), args.batch):
            chunk = docs[b0 : b0 + args.batch]
            if args.session_field:
                doc = chunk[0]
                session_value = doc[args.session_field]
                history = session_histories.get(session_value, [])
                key, prompt = render_doc_prompt(doc)
                if history:
                    if session_value not in session_handles:
                        raise RuntimeError(f"session {session_value!r} lost its context handle")
                    pre_segment_handle = session_handles.pop(session_value)
                    committed_token_ids = session_committed_tokens.pop(session_value)
                else:
                    if session_root_backup is None:
                        raise RuntimeError("session mode has no reusable root context handle")
                    pre_segment_handle = session_root_backup.copy_to(model, model.device)
                    committed_token_ids = None
                token_ids = session_turn_token_ids(
                    renderer,
                    history,
                    prompt,
                    committed_token_ids,
                )
                pre_segment_token_count = pre_segment_handle.token_count
                fed_tokens = len(token_ids) - pre_segment_token_count
                if fed_tokens <= 0:
                    raise RuntimeError("session turn did not extend its context handle")
                pre_segment_backup = None
                if args.session_recovery_retries:
                    pre_segment_backup = (
                        pre_segment_handle.copy_to(model, "cpu") if history else session_root_backup
                    )
                attempt_raws = []
                attempt_latencies = []
                selected_handle = None
                for attempt in range(args.session_recovery_retries + 1):
                    if attempt:
                        if pre_segment_backup is None:
                            raise RuntimeError("session recovery has no pre-segment context fork")
                        selected_handle = None
                        torch.cuda.empty_cache()
                        pre_segment_handle = pre_segment_backup.copy_to(model, model.device)
                    raw, attempt_latency, selected_handle, selected_committed_tokens = generate_cached_tokens(
                        token_ids,
                        pre_segment_handle,
                        consume_prefix=True,
                    )
                    del pre_segment_handle
                    attempt_raws.append(raw)
                    attempt_latencies.append(attempt_latency)
                    alignment_repairs = []
                    label_sets = []
                    preds, stats = pfn(
                        raw,
                        doc["text"],
                        tagset,
                        alignment_repairs=alignment_repairs,
                        label_sets=label_sets,
                    )
                    if not stats["quarantined"] or attempt == args.session_recovery_retries:
                        break
                if selected_handle is None:
                    raise RuntimeError("session generation produced no context handle")
                structured_materializable = not bool(stats["quarantined"])
                session_histories[session_value] = advanced_session_history(
                    history,
                    prompt,
                    raw,
                )
                session_handles[session_value] = selected_handle
                session_committed_tokens[session_value] = selected_committed_tokens
                del pre_segment_backup
                for stat, value in stats.items():
                    agg[stat] = agg.get(stat, 0) + value
                latency = sum(attempt_latencies)
                lat.append(latency)
                n_done += 1
                row = {
                    "id": doc["id"],
                    "preds": preds,
                    "parse_stats": stats,
                    "raw": raw,
                    "example_lang": key,
                    "latency_s": round(latency, 3),
                    "label_sets": label_sets,
                    "alignment_repairs": alignment_repairs,
                    "session": {
                        "field": args.session_field,
                        "value": session_value,
                        "recovery_attempts": len(attempt_raws) - 1,
                        "recovery_branch": "pre_segment_context_handle",
                        "continuable": True,
                        "structured_materializable": structured_materializable,
                        "cached_tokens_per_attempt": pre_segment_token_count,
                        "fed_tokens_per_attempt": fed_tokens,
                        "committed_tokens": len(selected_committed_tokens),
                    },
                }
                if len(attempt_raws) > 1:
                    row["raw_attempts"] = attempt_raws
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if b0 == session_last_indices[session_value]:
                    del session_histories[session_value]
                    del session_handles[session_value]
                    del session_committed_tokens[session_value]
                del selected_handle
                if n_done % 40 < args.batch:
                    msg = f"llm-label {n_done}/{len(docs)} parse-stats {agg}"
                    print(msg, flush=True)
                    hf = os.environ.get("AGENTCTL_HEADLINE_FILE")
                    if hf:
                        open(hf, "w").write(msg + "\n")
                continue
            t_batch = time.perf_counter()
            keys, id_lists = [], []
            for doc in chunk:
                key, token_ids = tokenize_doc(doc)
                keys.append(key)
                id_lists.append(
                    token_ids[len(shared_prefix_ids) :] if shared_prefix_ids is not None else token_ids
                )
            decode_batch = decodelib.prepare_batch(
                model,
                id_lists,
                pad_token_id=pad_id,
                padding_side="left",
                prefix=shared_prefix,
            )
            synchronize_timing_device()
            generate_started = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(
                    **decode_batch.model_inputs(),
                    max_new_tokens=args.max_new,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            synchronize_timing_device()
            generate_seconds = time.perf_counter() - generate_started
            if args.timing_receipt:
                visible_output_tokens = 0
                for generated_row in gen[:, decode_batch.output_prompt_width :]:
                    for token_id in generated_row.tolist():
                        if token_id in stop_token_ids:
                            break
                        if token_id != pad_id:
                            visible_output_tokens += 1
                timing_batches.append(
                    {
                        "rows": len(chunk),
                        "seconds": generate_seconds,
                        "visible_output_tokens": visible_output_tokens,
                    }
                )
            dt_seg = (time.perf_counter() - t_batch) / len(chunk)
            for r, doc in enumerate(chunk):
                raw = tok.decode(
                    gen[r][decode_batch.output_prompt_width :],
                    skip_special_tokens=True,
                )
                raw = model_format.visible_reply(raw)
                alignment_repairs = []
                label_sets = []
                subclass_spans = []
                candidate_decisions = []
                if args.fmt in {"json-groups", "json-groups-lexical"}:
                    assert pfn is not None
                    preds, stats = pfn(
                        raw,
                        doc["text"],
                        tagset,
                        alignment_repairs=alignment_repairs,
                        label_sets=label_sets,
                    )
                elif args.fmt == "json-subclasses":
                    assert subclass_spec is not None
                    preds, subclass_spans, stats = parse_subclass_annotation(
                        raw,
                        doc["text"],
                        doc["_subclass_base_spans"],
                        subclass_spec,
                    )
                elif args.fmt == "json-candidate-subclasses":
                    assert subclass_spec is not None
                    preds, subclass_spans, candidate_decisions, stats = parse_candidate_subclass_annotation(
                        raw,
                        doc["text"],
                        doc["_subclass_base_spans"],
                        doc["_subclass_candidate_ledger"],
                        subclass_spec,
                    )
                else:
                    assert pfn is not None
                    preds, stats = pfn(raw, doc["text"], tagset)
                for k, v in stats.items():
                    agg[k] = agg.get(k, 0) + v
                lat.append(dt_seg)
                n_done += 1
                row = {
                    "id": doc["id"],
                    "preds": preds,
                    "parse_stats": stats,
                    "raw": raw,
                    "example_lang": keys[r],
                    "latency_s": round(dt_seg, 3),
                }
                if args.fmt in {"json-groups", "json-groups-lexical"}:
                    row["label_sets"] = label_sets
                    row["alignment_repairs"] = alignment_repairs
                if args.fmt in subclass_formats:
                    row["subclass_spans"] = subclass_spans
                if args.fmt == "json-candidate-subclasses":
                    row["candidate_decisions"] = candidate_decisions
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n_done % 40 < args.batch:
                msg = f"llm-label {n_done}/{len(docs)} parse-stats {agg}"
                print(msg, flush=True)
                hf = os.environ.get("AGENTCTL_HEADLINE_FILE")
                if hf:
                    open(hf, "w").write(msg + "\n")
    ls = sorted(lat)
    print(f"DONE {len(docs)} docs -> {args.out}; parse-stats {agg}", flush=True)
    if ls:
        print(
            f"LATENCY per-doc s: mean {sum(ls) / len(ls):.2f} "
            f"p50 {ls[len(ls) // 2]:.2f} p90 {ls[int(0.9 * len(ls))]:.2f} "
            f"max {ls[-1]:.2f} total {sum(ls):.0f}s",
            flush=True,
        )
    if args.timing_receipt:
        timed_rows = sum(batch["rows"] for batch in timing_batches)
        timed_seconds = sum(batch["seconds"] for batch in timing_batches)
        timed_tokens = sum(batch["visible_output_tokens"] for batch in timing_batches)
        warm_batches = timing_batches[1:] or timing_batches
        warm_rows = sum(batch["rows"] for batch in warm_batches)
        warm_seconds = sum(batch["seconds"] for batch in warm_batches)
        warm_tokens = sum(batch["visible_output_tokens"] for batch in warm_batches)
        if not timed_rows or not timed_tokens or not warm_rows or not warm_tokens:
            raise RuntimeError("timing receipt requires generated visible output tokens")
        device = torch.device(model.device)
        device_properties = torch.cuda.get_device_properties(device)
        receipt = {
            "schema": "pii-llm-direct-generate-timing",
            "version": 1,
            "model": args.model,
            "model_revision": args.model_revision,
            "device": str(model.device),
            "batch_size": args.batch,
            "rows": timed_rows,
            "shared_prefix_tokens": len(shared_prefix_ids or []),
            "prefix_prefill_seconds": prefix_prefill_seconds,
            "generate_seconds": timed_seconds,
            "visible_output_tokens": timed_tokens,
            "seconds_per_row": timed_seconds / timed_rows,
            "seconds_per_visible_output_token": timed_seconds / timed_tokens,
            "warm_excludes_first_batch": len(timing_batches) > 1,
            "warm_generate_seconds": warm_seconds,
            "warm_rows": warm_rows,
            "warm_visible_output_tokens": warm_tokens,
            "warm_seconds_per_row": warm_seconds / warm_rows,
            "warm_seconds_per_visible_output_token": warm_seconds / warm_tokens,
            "gpu": {
                "name": device_properties.name,
                "total_memory_bytes": device_properties.total_memory,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "batches": timing_batches,
        }
        timing_path = Path(args.timing_receipt)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
