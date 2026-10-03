"""Closed categorical subclass contracts shared by annotation and training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUBCLASS_SPEC_SCHEMA = "pii-subclass-families"
SUBCLASS_SPEC_VERSIONS = frozenset({1, 2, 3, 4})
SUBCLASS_SCOPES = frozenset({"full_primary_span", "component_span"})
SUBCLASS_DECODE_CONSTRAINTS = frozenset({"whole_carrier_consistent", "component_sequence"})


def permissive_name_components(text: str, spans: list[tuple[int, int, str]]) -> set[tuple[int, int, str]]:
    """Compare one fixed carrier modulo adjacent given/family partitioning.

    Touching pieces or whitespace-only gaps of the same kind coalesce.
    Other labels, punctuation and unlabelled name material remain barriers.
    A carrier with overlapping predictions retains exact spans: it earns no
    partition tolerance, and its conflicting labels remain scoring errors.
    This scoring projection never repairs stored training labels.
    """
    ordered = sorted(spans)
    if any(not 0 <= start < end <= len(text) for start, end, _value in ordered):
        raise ValueError("name component is outside text")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        return set(ordered)
    result: list[tuple[int, int, str]] = []
    for start, end, value in ordered:
        if (
            result
            and value in {"given_name", "family_name"}
            and result[-1][2] == value
            and not text[result[-1][1] : start].strip()
        ):
            result[-1] = (result[-1][0], end, value)
        else:
            result.append((start, end, value))
    return set(result)


@dataclass(frozen=True)
class BernoulliChannel:
    name: str
    scope: str
    applicable_types: frozenset[str]


@dataclass(frozen=True)
class SequenceGrammar:
    version: int
    maximum_components: tuple[tuple[str, int], ...]
    edge_only_values: frozenset[str]
    ignored_values: frozenset[str]
    symbols: tuple[tuple[str, str], ...]
    pattern: str | None
    description: str


@dataclass(frozen=True)
class SubclassFamily:
    name: str
    scope: str
    decode_constraint: str
    applicable_types: frozenset[str]
    outcomes: tuple[str, ...]
    definition: str
    q_definition: str
    sequence_grammar: SequenceGrammar | None


@dataclass(frozen=True)
class SubclassBlock:
    family: str
    primary_type: str
    start: int
    width: int


@dataclass(frozen=True)
class SubclassSpec:
    primary_output_types: tuple[str, ...]
    bernoulli_channels: tuple[BernoulliChannel, ...]
    families: tuple[SubclassFamily, ...]
    sidecar_families: tuple[SubclassFamily, ...]
    blocks: tuple[SubclassBlock, ...]
    sha256: str
    path: Path

    @property
    def head_rows(self) -> int:
        return sum(block.width for block in self.blocks)

    @property
    def family_by_name(self) -> dict[str, SubclassFamily]:
        return {family.name: family for family in self.annotation_families}

    @property
    def annotation_families(self) -> tuple[SubclassFamily, ...]:
        return self.families + self.sidecar_families

    @property
    def sidecar_family_names(self) -> frozenset[str]:
        return frozenset(family.name for family in self.sidecar_families)

    @property
    def bernoulli_by_name(self) -> dict[str, BernoulliChannel]:
        return {channel.name: channel for channel in self.bernoulli_channels}

    @property
    def block_by_key(self) -> dict[tuple[str, str], tuple[int, SubclassBlock]]:
        return {(block.family, block.primary_type): (index, block) for index, block in enumerate(self.blocks)}

    def config_blocks(self) -> list[dict[str, Any]]:
        return [
            {
                "family": block.family,
                "primary_type": block.primary_type,
                "start": block.start,
                "width": block.width,
            }
            for block in self.blocks
        ]


def _unique_nonempty_strings(value: Any, where: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{where} must be a nonempty list of unique strings")
    return tuple(value)


def _legacy_sequence_grammar(
    raw: Any,
    outcomes: tuple[str, ...],
    where: str,
) -> SequenceGrammar:
    if not isinstance(raw, dict) or set(raw) != {
        "maximum_components",
        "edge_only_values",
        "ignored_values",
        "description",
    }:
        raise ValueError(f"{where} has an invalid shape")
    maximum_components = raw["maximum_components"]
    if (
        not isinstance(maximum_components, dict)
        or not maximum_components
        or any(
            key not in outcomes or isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for key, limit in maximum_components.items()
        )
    ):
        raise ValueError(f"{where}.maximum_components is invalid")
    edge_only_values = _unique_nonempty_strings(
        raw["edge_only_values"],
        f"{where}.edge_only_values",
    )
    if any(value not in outcomes for value in edge_only_values):
        raise ValueError(f"{where}.edge_only_values is invalid")
    ignored_values = _unique_nonempty_strings(
        raw["ignored_values"],
        f"{where}.ignored_values",
    )
    if any(value not in outcomes for value in ignored_values):
        raise ValueError(f"{where}.ignored_values is invalid")
    if set(edge_only_values) & set(ignored_values):
        raise ValueError(f"{where} edge-only and ignored values overlap")
    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{where}.description must be nonempty")
    return SequenceGrammar(
        version=1,
        maximum_components=tuple(maximum_components.items()),
        edge_only_values=frozenset(edge_only_values),
        ignored_values=frozenset(ignored_values),
        symbols=(),
        pattern=None,
        description=description,
    )


def _regex_sequence_grammar(
    raw: Any,
    outcomes: tuple[str, ...],
    where: str,
) -> SequenceGrammar:
    if not isinstance(raw, dict) or set(raw) != {
        "symbols",
        "ignored_values",
        "pattern",
        "maximum_components",
        "description",
    }:
        raise ValueError(f"{where} has an invalid shape")
    ignored_values = _unique_nonempty_strings(
        raw["ignored_values"],
        f"{where}.ignored_values",
    )
    if any(value not in outcomes for value in ignored_values):
        raise ValueError(f"{where}.ignored_values is invalid")
    modeled_values = tuple(value for value in outcomes if value not in ignored_values)
    symbols = raw["symbols"]
    if (
        not isinstance(symbols, dict)
        or set(symbols) != set(modeled_values)
        or any(
            not isinstance(symbol, str) or len(symbol) != 1 or not symbol.isascii() or not symbol.isalnum()
            for symbol in symbols.values()
        )
        or len(set(symbols.values())) != len(symbols)
    ):
        raise ValueError(
            f"{where}.symbols must map every non-ignored outcome to one unique ASCII alphanumeric"
        )
    pattern = raw["pattern"]
    if not isinstance(pattern, str) or not pattern or len(pattern) > 512:
        raise ValueError(f"{where}.pattern must be a nonempty regular expression")
    try:
        re.compile(pattern)
    except re.error as error:
        raise ValueError(f"{where}.pattern is not a valid regular expression: {error}") from error
    maximum_components = raw["maximum_components"]
    if (
        not isinstance(maximum_components, dict)
        or not maximum_components
        or any(
            key not in modeled_values or isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
            for key, limit in maximum_components.items()
        )
    ):
        raise ValueError(f"{where}.maximum_components is invalid")
    symbol_by_value = tuple((value, symbols[value]) for value in modeled_values)
    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{where}.description must be nonempty")
    return SequenceGrammar(
        version=2,
        maximum_components=tuple(maximum_components.items()),
        edge_only_values=frozenset(),
        ignored_values=frozenset(ignored_values),
        symbols=symbol_by_value,
        pattern=pattern,
        description=description,
    )


def load_subclass_spec(path: Path) -> SubclassSpec:
    """Load and close the categorical family, outcome, and head-block order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    common_keys = {
        "schema",
        "version",
        "primary_output_types",
        "bernoulli_channels",
        "families",
        "training_contract",
    }
    if not isinstance(payload, dict) or not {"schema", "version"} <= set(payload):
        raise ValueError("subclass spec has an invalid top-level shape")
    spec_version = payload.get("version")
    if payload["schema"] != SUBCLASS_SPEC_SCHEMA or spec_version not in SUBCLASS_SPEC_VERSIONS:
        raise ValueError(
            f"unsupported subclass spec {payload.get('schema')!r} version {payload.get('version')!r}"
        )
    expected_keys = common_keys | ({"sidecar_families"} if spec_version == 4 else set())
    if set(payload) != expected_keys:
        raise ValueError("subclass spec has an invalid top-level shape")
    primary_output_types = _unique_nonempty_strings(payload["primary_output_types"], "primary_output_types")

    bernoulli_channels = []
    bernoulli_names = set()
    for index, entry in enumerate(payload["bernoulli_channels"]):
        where = f"bernoulli_channels[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"name", "scope", "applicable_types"}:
            raise ValueError(f"{where} has an invalid shape")
        name = entry["name"]
        scope = entry["scope"]
        if not isinstance(name, str) or not name or name in bernoulli_names:
            raise ValueError(f"{where}.name must be unique and nonempty")
        if scope not in SUBCLASS_SCOPES:
            raise ValueError(f"{where}.scope is unsupported: {scope!r}")
        bernoulli_names.add(name)
        bernoulli_channels.append(
            BernoulliChannel(
                name=name,
                scope=scope,
                applicable_types=frozenset(
                    _unique_nonempty_strings(entry["applicable_types"], f"{where}.applicable_types")
                ),
            )
        )

    families = []
    sidecar_families = []
    family_names = set()
    blocks = []
    row_start = 0
    raw_trainable_families = payload["families"]
    raw_sidecar_families = payload.get("sidecar_families", [])
    if not isinstance(raw_trainable_families, list) or not raw_trainable_families:
        raise ValueError("families must be a nonempty list")
    if not isinstance(raw_sidecar_families, list):
        raise ValueError("sidecar_families must be a list")
    raw_families = [(False, index, entry) for index, entry in enumerate(raw_trainable_families)] + [
        (True, index, entry) for index, entry in enumerate(raw_sidecar_families)
    ]
    for is_sidecar, index, entry in raw_families:
        collection = "sidecar_families" if is_sidecar else "families"
        where = f"{collection}[{index}]"
        required_keys = {
            "name",
            "scope",
            "decode_constraint",
            "applicable_types",
            "outcomes",
            "definition",
            "Q_definition",
        }
        optional_keys = {"sequence_grammar"}
        if (
            not isinstance(entry, dict)
            or not required_keys <= set(entry)
            or set(entry) - required_keys - optional_keys
        ):
            raise ValueError(f"{where} has an invalid shape")
        name = entry["name"]
        scope = entry["scope"]
        decode_constraint = entry["decode_constraint"]
        if not isinstance(name, str) or not name or name in family_names:
            raise ValueError(f"{where}.name must be unique and nonempty")
        if scope not in SUBCLASS_SCOPES:
            raise ValueError(f"{where}.scope is unsupported: {scope!r}")
        if decode_constraint not in SUBCLASS_DECODE_CONSTRAINTS:
            raise ValueError(f"{where}.decode_constraint is unsupported: {decode_constraint!r}")
        if scope == "full_primary_span" and decode_constraint != "whole_carrier_consistent":
            raise ValueError(f"{where} full-primary scope requires whole-carrier decoding")
        if scope == "component_span" and decode_constraint != "component_sequence":
            raise ValueError(f"{where} component scope requires component-sequence decoding")
        applicable_types = _unique_nonempty_strings(entry["applicable_types"], f"{where}.applicable_types")
        outcomes = _unique_nonempty_strings(entry["outcomes"], f"{where}.outcomes")
        if not is_sidecar and outcomes[0] != "Q":
            raise ValueError(f"{where}.outcomes must put Q first")
        definition = entry["definition"]
        q_definition = entry["Q_definition"]
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"{where}.definition must be nonempty")
        if not isinstance(q_definition, str) or not q_definition.strip():
            raise ValueError(f"{where}.Q_definition must be nonempty")
        raw_grammar = entry.get("sequence_grammar")
        sequence_grammar = None
        if decode_constraint == "component_sequence":
            grammar_where = f"{where}.sequence_grammar"
            sequence_grammar = (
                _legacy_sequence_grammar(raw_grammar, outcomes, grammar_where)
                if spec_version == 1
                else _regex_sequence_grammar(raw_grammar, outcomes, grammar_where)
            )
        elif raw_grammar is not None:
            raise ValueError(f"{where} whole-carrier family cannot define a sequence grammar")
        family_names.add(name)
        family = SubclassFamily(
            name=name,
            scope=scope,
            decode_constraint=decode_constraint,
            applicable_types=frozenset(applicable_types),
            outcomes=outcomes,
            definition=definition,
            q_definition=q_definition,
            sequence_grammar=sequence_grammar,
        )
        if is_sidecar:
            sidecar_families.append(family)
        else:
            families.append(family)
            for primary_type in applicable_types:
                blocks.append(
                    SubclassBlock(
                        family=name,
                        primary_type=primary_type,
                        start=row_start,
                        width=len(outcomes),
                    )
                )
                row_start += len(outcomes)

    contract = payload["training_contract"]
    expected_contract = {
        "conditioning": "Each family block is conditioned on the activating ordinary primary type.",
        "coverage": "A full_primary_span value applies to every character and overlapping model token of the exact carrier. A component_span value applies to every character and overlapping token of the exact declared internal component.",
        "unknown": "A missing family/component is masked and never converted to Q.",
        "loss_unit": "A full_primary_span family uses one categorical cross-entropy after length-normalized logit pooling. A component_span family uses token-local categorical cross-entropy over the component, with its total component weight divided across overlapping model tokens.",
        "weight_initialization": "Each subclass objective_weight initially inherits its carrier primary-span objective weight. Its independent learning_weight defaults to 1.0; effective optimization mass is objective_weight times learning_weight.",
        "ordinary_primary_loss": "Primary BIOES boundary/type loss remains independently weighted and unchanged; component-span subclass decoding does not enter the single-winner primary Viterbi lattice.",
    }
    if contract != expected_contract:
        raise ValueError("subclass training_contract does not match the implemented contract")
    return SubclassSpec(
        primary_output_types=primary_output_types,
        bernoulli_channels=tuple(bernoulli_channels),
        families=tuple(families),
        sidecar_families=tuple(sidecar_families),
        blocks=tuple(blocks),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        path=path,
    )


def render_subclass_catalog(spec: SubclassSpec) -> str:
    """Render the exact closed categorical choices into an annotation prompt."""
    lines = []
    for family in spec.annotation_families:
        grammar = ""
        if family.sequence_grammar is not None:
            sequence_grammar = family.sequence_grammar
            grammar = f" Sequence grammar: {sequence_grammar.description}"
            if sequence_grammar.pattern is not None:
                symbols = ", ".join(f"{symbol}={value}" for value, symbol in sequence_grammar.symbols)
                grammar += (
                    f" Exact regular expression after ignoring "
                    f"{', '.join(sorted(sequence_grammar.ignored_values))}: "
                    f"/{sequence_grammar.pattern}/ ({symbols})."
                )
        sidecar = (
            "; annotation/evaluation sidecar; zero training weight"
            if family.name in spec.sidecar_family_names
            else ""
        )
        q_definition = f" Q: {family.q_definition}" if "Q" in family.outcomes else ""
        lines.append(
            f"- {family.name} ({family.scope}; {family.decode_constraint}; applies to "
            f"{', '.join(sorted(family.applicable_types))}{sidecar}): {family.definition} "
            f"Allowed values: {', '.join(family.outcomes)}.{q_definition}"
            f"{grammar}"
        )
    return "\n".join(lines)


def validate_sequence_grammar(
    family: SubclassFamily,
    components: list[tuple[int, int, str]],
) -> None:
    """Validate one carrier's ordered component sequence against inventory grammar."""
    grammar = family.sequence_grammar
    if grammar is None:
        return
    ordered = [component for component in sorted(components) if component[2] not in grammar.ignored_values]
    sequence = tuple(component[2] for component in ordered)
    if grammar.pattern is not None:
        encoded = "".join(dict(grammar.symbols)[value] for value in sequence)
        if re.fullmatch(grammar.pattern, encoded) is None:
            raise ValueError(
                f"{family.name} component sequence {encoded!r} does not match /{grammar.pattern}/"
            )
        return
    counts: dict[str, int] = {}
    for position, (_, _, value) in enumerate(ordered):
        counts[value] = counts.get(value, 0) + 1
        if value in grammar.edge_only_values and position not in {0, len(ordered) - 1}:
            raise ValueError(f"{family.name} value {value!r} must occur at a sequence edge")
    for value, limit in grammar.maximum_components:
        if counts.get(value, 0) > limit:
            raise ValueError(
                f"{family.name} value {value!r} has {counts[value]} components; maximum is {limit}"
            )


def _component_runs(
    values: tuple[str, ...],
    token_indices: list[int],
    offsets: list[tuple[int, int]],
    carrier_start: int,
    carrier_end: int,
) -> list[tuple[int, int, str]]:
    """Group equal-valued tokens into component spans inside their carrier.

    Tokens are selected by overlapping the carrier rather than by sitting
    inside it, because a carrier boundary need not fall on a token boundary:
    a gold carrier is annotated over characters and can cut a token in half.
    The straddling token still votes for its component, but the component it
    produces is clipped to the carrier, so a component can never claim
    characters the carrier does not cover.
    """
    runs = []
    run_start = 0
    for position in range(1, len(values) + 1):
        if position < len(values) and values[position] == values[run_start]:
            continue
        first_token = token_indices[run_start]
        last_token = token_indices[position - 1]
        start = max(offsets[first_token][0], carrier_start)
        end = min(offsets[last_token][1], carrier_end)
        if start < end:
            runs.append((start, end, values[run_start]))
        run_start = position
    return runs


def constrained_component_values(
    logits: list[list[float]],
    family: SubclassFamily,
) -> tuple[str, ...]:
    """Maximize token-local scores subject to one inventory sequence grammar."""
    if not logits:
        return ()
    width = len(family.outcomes)
    if any(len(row) != width for row in logits):
        raise ValueError(f"{family.name} logits do not match its {width} outcomes")
    grammar = family.sequence_grammar
    if grammar is None:
        return tuple(family.outcomes[max(range(width), key=row.__getitem__)] for row in logits)

    limits = dict(grammar.maximum_components)
    states: dict[tuple[str | None, tuple[str, ...]], tuple[float, tuple[str, ...]]] = {(None, ()): (0.0, ())}
    for row in logits:
        next_states = {}
        for (prior, sequence), (score, values) in states.items():
            for value_index, value in enumerate(family.outcomes):
                next_sequence = sequence
                if value not in grammar.ignored_values and value != prior:
                    next_sequence = (*sequence, value)
                    if value in limits and next_sequence.count(value) > limits[value]:
                        continue
                key = (value, next_sequence)
                candidate = (score + float(row[value_index]), (*values, value))
                incumbent = next_states.get(key)
                if incumbent is None or (
                    candidate[0],
                    tuple(-family.outcomes.index(item) for item in candidate[1]),
                ) > (
                    incumbent[0],
                    tuple(-family.outcomes.index(item) for item in incumbent[1]),
                ):
                    next_states[key] = candidate
        states = next_states
    best = None
    for score, values in states.values():
        abstract_runs = _component_runs(
            values,
            list(range(len(values))),
            [(i, i + 1) for i in range(len(values))],
            0,
            len(values),
        )
        try:
            validate_sequence_grammar(family, abstract_runs)
        except ValueError:
            continue
        candidate = (
            score,
            tuple(-family.outcomes.index(value) for value in values),
            values,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError(f"{family.name} grammar admits no sequence for {len(logits)} tokens")
    return best[2]


def decode_subclass_logits(
    token_logits: Any,
    offsets: list[tuple[int, int]],
    primary_spans: Any,
    spec: SubclassSpec,
) -> list[dict[str, Any]]:
    """Decode conditional categorical logits under each family's span contract."""
    rows = [[float(value) for value in row] for row in token_logits]
    if len(rows) != len(offsets) or any(len(row) != spec.head_rows for row in rows):
        raise ValueError("subclass logits must align with tokenizer offsets and spec head rows")
    if not isinstance(primary_spans, list):
        raise ValueError("primary spans must be a list")
    carriers = set()
    for index, span in enumerate(primary_spans):
        if isinstance(span, dict):
            start, end, primary_type = span.get("start"), span.get("end"), span.get("type")
        elif isinstance(span, (list, tuple)) and len(span) == 3:
            start, end, primary_type = span
        else:
            raise ValueError(f"primary span {index} has an invalid shape")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end
            or not isinstance(primary_type, str)
            or not primary_type
        ):
            raise ValueError(f"primary span {index} is invalid")
        carriers.add((start, end, primary_type))
    if len(carriers) != len(primary_spans):
        raise ValueError("primary spans must be unique")
    decoded = []
    family_by_name = spec.family_by_name
    for carrier_start, carrier_end, primary_type in sorted(carriers):
        token_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start != end and start < carrier_end and carrier_start < end
        ]
        if not token_indices:
            continue
        for block_id, block in enumerate(spec.blocks):
            if block.primary_type != primary_type:
                continue
            family = family_by_name[block.family]
            block_logits = [
                rows[token_index][block.start : block.start + block.width] for token_index in token_indices
            ]
            if family.scope == "full_primary_span":
                pooled = [
                    sum(row[outcome] for row in block_logits) / len(block_logits)
                    for outcome in range(block.width)
                ]
                value = family.outcomes[max(range(block.width), key=pooled.__getitem__)]
                runs = [(carrier_start, carrier_end, value)]
            else:
                values = constrained_component_values(block_logits, family)
                runs = _component_runs(values, token_indices, offsets, carrier_start, carrier_end)
                validate_sequence_grammar(family, runs)
            decoded.extend(
                {
                    "carrier_start": carrier_start,
                    "carrier_end": carrier_end,
                    "type": primary_type,
                    "start": start,
                    "end": end,
                    "family": family.name,
                    "value": value,
                    "block_id": block_id,
                }
                for start, end, value in runs
            )
    return sorted(
        decoded,
        key=lambda item: (
            item["carrier_start"],
            item["carrier_end"],
            item["family"],
            item["start"],
            item["end"],
        ),
    )


def _valid_offset_item(item: Any, text: str, required: set[str]) -> tuple[int, int] | None:
    if not isinstance(item, dict) or set(item) != required:
        return None
    start = item.get("start")
    end = item.get("end")
    surface = item.get("t")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(text)
        or not isinstance(surface, str)
        or surface != text[start:end]
    ):
        return None
    return start, end


def _load_strict_json_response(raw: str) -> Any:
    """Load bare JSON or one observed Markdown JSON fence and no other prose."""
    stripped = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n```", stripped, re.DOTALL)
    if fenced is not None:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)


def _validated_base_spans(base_spans: Any, text: str) -> set[tuple[int, int, str]]:
    if not isinstance(base_spans, list):
        raise ValueError("base_spans must be a list")
    result = set()
    for index, span in enumerate(base_spans):
        if not isinstance(span, (list, tuple, dict)):
            raise ValueError(f"base_spans[{index}] has an invalid shape")
        if isinstance(span, dict):
            required = {"start", "end", "type"}
            if not required <= set(span):
                raise ValueError(f"base_spans[{index}] lacks start, end, or type")
            start, end, primary_type = span["start"], span["end"], span["type"]
        else:
            if len(span) != 3:
                raise ValueError(f"base_spans[{index}] must have three values")
            start, end, primary_type = span
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
            or not isinstance(primary_type, str)
            or not primary_type
        ):
            raise ValueError(f"base_spans[{index}] is invalid")
        identity = (start, end, primary_type)
        if identity in result:
            raise ValueError(f"duplicate base span {identity!r}")
        result.add(identity)
    return result


def parse_subclass_annotation(
    raw: str,
    text: str,
    base_spans: Any,
    spec: SubclassSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Validate one nested primary/Bernoulli/categorical teacher response."""
    stats = {
        "bad_json": 0,
        "invalid_top_level": 0,
        "invalid_item": 0,
        "invalid_bounds_or_surface": 0,
        "unknown_primary_type": 0,
        "unknown_bernoulli_channel": 0,
        "unknown_subclass_family_or_value": 0,
        "missing_or_incompatible_carrier": 0,
        "partial_full_span_value": 0,
        "duplicate_or_conflicting_component": 0,
        "missing_required_sidecar": 0,
        "invalid_sequence_grammar": 0,
        "out_of_order": 0,
        "quarantined": 0,
    }
    try:
        payload = _load_strict_json_response(raw)
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        stats["quarantined"] = 1
        return [], [], stats
    if not isinstance(payload, dict) or set(payload) != {
        "primary_spans",
        "bernoulli_spans",
        "subclass_spans",
    }:
        stats["invalid_top_level"] = 1
        stats["quarantined"] = 1
        return [], [], stats
    if any(not isinstance(payload[field], list) for field in payload):
        stats["invalid_top_level"] = 1
        stats["quarantined"] = 1
        return [], [], stats

    try:
        carriers = _validated_base_spans(base_spans, text)
    except ValueError:
        stats["missing_or_incompatible_carrier"] = 1
        stats["quarantined"] = 1
        return [], [], stats

    predictions = []
    categorical = []
    fatal = False
    prior_start = -1
    primary_seen = set()
    primary_required = {"start", "end", "t", "type"}
    for item in payload["primary_spans"]:
        bounds = _valid_offset_item(item, text, primary_required)
        if bounds is None:
            stats["invalid_bounds_or_surface"] += 1
            fatal = True
            continue
        start, end = bounds
        primary_type = item["type"]
        if primary_type not in spec.primary_output_types:
            stats["unknown_primary_type"] += 1
            fatal = True
            continue
        identity = (start, end, primary_type)
        if identity in primary_seen or identity in carriers:
            stats["duplicate_or_conflicting_component"] += 1
            fatal = True
            continue
        if start < prior_start:
            stats["out_of_order"] += 1
            fatal = True
        prior_start = start
        primary_seen.add(identity)
        carriers.add(identity)
        predictions.append({"start": start, "end": end, "label": primary_type})

    bernoulli_by_name = spec.bernoulli_by_name
    bernoulli_required = {
        "carrier_start",
        "carrier_end",
        "carrier_type",
        "start",
        "end",
        "t",
        "type",
    }
    bernoulli_seen = set()
    prior_start = -1
    for item in payload["bernoulli_spans"]:
        bounds = _valid_offset_item(item, text, bernoulli_required)
        if bounds is None:
            stats["invalid_bounds_or_surface"] += 1
            fatal = True
            continue
        start, end = bounds
        channel = bernoulli_by_name.get(item["type"])
        if channel is None:
            stats["unknown_bernoulli_channel"] += 1
            fatal = True
            continue
        carrier = (item["carrier_start"], item["carrier_end"], item["carrier_type"])
        if carrier not in carriers or carrier[2] not in channel.applicable_types:
            stats["missing_or_incompatible_carrier"] += 1
            fatal = True
            continue
        if not carrier[0] <= start < end <= carrier[1]:
            stats["missing_or_incompatible_carrier"] += 1
            fatal = True
            continue
        if channel.scope == "full_primary_span" and (start, end) != carrier[:2]:
            stats["partial_full_span_value"] += 1
            fatal = True
            continue
        identity = (*carrier, start, end, channel.name)
        if identity in bernoulli_seen:
            stats["duplicate_or_conflicting_component"] += 1
            fatal = True
            continue
        if start < prior_start:
            stats["out_of_order"] += 1
            fatal = True
        prior_start = start
        bernoulli_seen.add(identity)
        predictions.append({"start": start, "end": end, "label": channel.name})

    family_by_name = spec.family_by_name
    subclass_required = {
        "carrier_start",
        "carrier_end",
        "carrier_type",
        "start",
        "end",
        "t",
        "family",
        "value",
    }
    categorical_seen = set()
    component_intervals: dict[tuple[int, int, str, str], list[tuple[int, int]]] = {}
    component_sequences: dict[tuple[int, int, str, str], list[tuple[int, int, str]]] = {}
    prior_start = -1
    for item in payload["subclass_spans"]:
        bounds = _valid_offset_item(item, text, subclass_required)
        if bounds is None:
            stats["invalid_bounds_or_surface"] += 1
            fatal = True
            continue
        start, end = bounds
        family = family_by_name.get(item["family"])
        value = item["value"]
        if family is None or value not in family.outcomes:
            stats["unknown_subclass_family_or_value"] += 1
            fatal = True
            continue
        carrier = (item["carrier_start"], item["carrier_end"], item["carrier_type"])
        if carrier not in carriers or carrier[2] not in family.applicable_types:
            stats["missing_or_incompatible_carrier"] += 1
            fatal = True
            continue
        if not carrier[0] <= start < end <= carrier[1]:
            stats["missing_or_incompatible_carrier"] += 1
            fatal = True
            continue
        if family.scope == "full_primary_span" and (start, end) != carrier[:2]:
            stats["partial_full_span_value"] += 1
            fatal = True
            continue
        family_carrier = (*carrier, family.name)
        identity = (*family_carrier, start, end, value)
        if identity in categorical_seen:
            stats["duplicate_or_conflicting_component"] += 1
            fatal = True
            continue
        intervals = component_intervals.setdefault(family_carrier, [])
        if family.scope == "full_primary_span" and intervals:
            stats["duplicate_or_conflicting_component"] += 1
            fatal = True
            continue
        if family.scope == "component_span" and any(
            start < old_end and old_start < end for old_start, old_end in intervals
        ):
            stats["duplicate_or_conflicting_component"] += 1
            fatal = True
            continue
        if start < prior_start:
            stats["out_of_order"] += 1
            fatal = True
        prior_start = start
        intervals.append((start, end))
        component_sequences.setdefault(family_carrier, []).append((start, end, value))
        categorical_seen.add(identity)
        categorical.append(
            {
                "carrier_start": carrier[0],
                "carrier_end": carrier[1],
                "type": carrier[2],
                "start": start,
                "end": end,
                "family": family.name,
                "value": value,
            }
        )

    for family_carrier, components in component_sequences.items():
        family = family_by_name[family_carrier[-1]]
        try:
            validate_sequence_grammar(family, components)
        except ValueError:
            stats["invalid_sequence_grammar"] += 1
            fatal = True

    if "reference_form" in spec.sidecar_family_names:
        expected_reference_carriers = {
            carrier for carrier in carriers if carrier[2] in {"person_reference", "organization_reference"}
        }
        observed_reference_carriers = {
            family_carrier[:3]
            for family_carrier in component_sequences
            if family_carrier[-1] == "reference_form"
        }
        if observed_reference_carriers != expected_reference_carriers:
            stats["missing_required_sidecar"] += len(
                expected_reference_carriers - observed_reference_carriers
            )
            fatal = True

    if fatal:
        stats["quarantined"] = 1
        return [], [], stats
    predictions.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    categorical.sort(key=lambda item: (item["start"], item["end"], item["family"], item["value"]))
    return predictions, categorical, stats


def _validated_candidate_ledger(
    candidate_ledger: Any,
    text: str,
    carriers: set[tuple[int, int, str]],
    spec: SubclassSpec,
) -> list[dict[str, Any]]:
    if not isinstance(candidate_ledger, list):
        raise ValueError("candidate_ledger must be a list")
    required = {
        "candidate_id",
        "start",
        "end",
        "surface",
        "base_type",
        "allowed_primary_types",
        "bernoulli_channels",
        "sources",
    }
    result = []
    prior_bounds = (-1, -1)
    seen_ids = set()
    seen_spans = set()
    for index, candidate in enumerate(candidate_ledger):
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ValueError(f"candidate_ledger[{index}] has an invalid shape")
        candidate_id = candidate["candidate_id"]
        start = candidate["start"]
        end = candidate["end"]
        surface = candidate["surface"]
        base_type = candidate["base_type"]
        allowed_primary_types = candidate["allowed_primary_types"]
        bernoulli_channels = candidate["bernoulli_channels"]
        sources = candidate["sources"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen_ids
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
            or not isinstance(surface, str)
            or surface != text[start:end]
            or base_type is not None
            and (not isinstance(base_type, str) or (start, end, base_type) not in carriers)
        ):
            raise ValueError(f"candidate_ledger[{index}] has invalid identity or bounds")
        if (start, end) < prior_bounds or (start, end) in seen_spans:
            raise ValueError("candidate_ledger must be uniquely sorted by source interval")
        prior_bounds = (start, end)
        seen_ids.add(candidate_id)
        seen_spans.add((start, end))
        allowed = _unique_nonempty_strings(
            allowed_primary_types,
            f"candidate_ledger[{index}].allowed_primary_types",
        )
        if any(
            value != "O" and value not in spec.primary_output_types and value != base_type
            for value in allowed
        ):
            raise ValueError(f"candidate_ledger[{index}] has an unsupported primary outcome")
        if base_type is not None and base_type not in allowed:
            raise ValueError(f"candidate_ledger[{index}] cannot preserve its base carrier")
        if (
            not isinstance(bernoulli_channels, list)
            or any(not isinstance(name, str) or not name for name in bernoulli_channels)
            or len(bernoulli_channels) != len(set(bernoulli_channels))
        ):
            raise ValueError(f"candidate_ledger[{index}].bernoulli_channels must be unique strings")
        channels = tuple(bernoulli_channels)
        if any(name not in spec.bernoulli_by_name for name in channels):
            raise ValueError(f"candidate_ledger[{index}] has an unknown Bernoulli channel")
        if any(
            not any(outcome in spec.bernoulli_by_name[name].applicable_types for outcome in allowed)
            for name in channels
        ):
            raise ValueError(f"candidate_ledger[{index}] has an inapplicable Bernoulli channel")
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source for source in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError(f"candidate_ledger[{index}].sources must be unique nonempty strings")
        result.append(candidate)
    return result


def parse_candidate_subclass_annotation(
    raw: str,
    text: str,
    base_spans: Any,
    candidate_ledger: Any,
    spec: SubclassSpec,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Validate exhaustive controller-candidate decisions and categorical output."""
    stats = {
        "bad_json": 0,
        "invalid_top_level": 0,
        "invalid_candidate_ledger": 0,
        "missing_or_duplicate_candidate_decision": 0,
        "invalid_candidate_decision": 0,
        "invalid_added_candidate": 0,
        "quarantined": 0,
    }
    try:
        payload = _load_strict_json_response(raw)
    except json.JSONDecodeError:
        stats["bad_json"] = 1
        stats["quarantined"] = 1
        return [], [], [], stats
    if not isinstance(payload, dict) or set(payload) != {
        "candidate_decisions",
        "added_candidates",
        "subclass_spans",
    }:
        stats["invalid_top_level"] = 1
        stats["quarantined"] = 1
        return [], [], [], stats
    if any(not isinstance(payload[field], list) for field in payload):
        stats["invalid_top_level"] = 1
        stats["quarantined"] = 1
        return [], [], [], stats

    try:
        carriers = _validated_base_spans(base_spans, text)
        candidates = _validated_candidate_ledger(candidate_ledger, text, carriers, spec)
    except ValueError:
        stats["invalid_candidate_ledger"] = 1
        stats["quarantined"] = 1
        return [], [], [], stats

    decisions = payload["candidate_decisions"]
    decision_required = {"candidate_id", "primary_type", "bernoulli_types"}
    if (
        len(decisions) != len(candidates)
        or any(not isinstance(item, dict) or set(item) != decision_required for item in decisions)
        or [item.get("candidate_id") for item in decisions]
        != [candidate["candidate_id"] for candidate in candidates]
    ):
        stats["missing_or_duplicate_candidate_decision"] = 1
        stats["quarantined"] = 1
        return [], [], [], stats

    primary_spans = []
    bernoulli_spans = []
    normalized_decisions = []
    for candidate, decision in zip(candidates, decisions, strict=True):
        primary_type = decision["primary_type"]
        true_channels = decision["bernoulli_types"]
        allowed_primary_types = candidate["allowed_primary_types"]
        allowed_channels = candidate["bernoulli_channels"]
        roles_are_inapplicable = primary_type not in {"person_name", "person_reference"}
        if (
            primary_type not in allowed_primary_types
            or not isinstance(true_channels, list)
            or len(true_channels) != len(set(true_channels))
            or true_channels != [name for name in allowed_channels if name in true_channels]
            or any(name not in allowed_channels for name in true_channels)
            or roles_are_inapplicable
            and true_channels
        ):
            stats["invalid_candidate_decision"] += 1
            continue
        start = candidate["start"]
        end = candidate["end"]
        surface = candidate["surface"]
        if primary_type in spec.primary_output_types and primary_type != candidate["base_type"]:
            primary_spans.append({"start": start, "end": end, "t": surface, "type": primary_type})
        for channel in true_channels:
            bernoulli_spans.append(
                {
                    "carrier_start": start,
                    "carrier_end": end,
                    "carrier_type": primary_type,
                    "start": start,
                    "end": end,
                    "t": surface,
                    "type": channel,
                }
            )
        normalized_decisions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "start": start,
                "end": end,
                "primary_type": primary_type,
                "bernoulli_types": list(true_channels),
            }
        )

    added_required = {"start", "end", "t", "primary_type", "bernoulli_types"}
    prior_added_bounds = (-1, -1)
    added_seen = set()
    enumerated_intervals = {(candidate["start"], candidate["end"]) for candidate in candidates}
    for item in payload["added_candidates"]:
        bounds = _valid_offset_item(item, text, added_required)
        primary_type = item.get("primary_type") if isinstance(item, dict) else None
        true_channels = item.get("bernoulli_types") if isinstance(item, dict) else None
        if bounds is None or primary_type not in spec.primary_output_types:
            stats["invalid_added_candidate"] += 1
            continue
        start, end = bounds
        applicable_channels = [
            channel.name for channel in spec.bernoulli_channels if primary_type in channel.applicable_types
        ]
        if (
            not isinstance(true_channels, list)
            or len(true_channels) != len(set(true_channels))
            or true_channels != [name for name in applicable_channels if name in true_channels]
            or any(name not in applicable_channels for name in true_channels)
            or (start, end) < prior_added_bounds
            or (start, end) in enumerated_intervals
            or (start, end, primary_type) in added_seen
        ):
            stats["invalid_added_candidate"] += 1
            continue
        prior_added_bounds = (start, end)
        added_seen.add((start, end, primary_type))
        primary_spans.append({"start": start, "end": end, "t": item["t"], "type": primary_type})
        for channel in true_channels:
            bernoulli_spans.append(
                {
                    "carrier_start": start,
                    "carrier_end": end,
                    "carrier_type": primary_type,
                    "start": start,
                    "end": end,
                    "t": item["t"],
                    "type": channel,
                }
            )

    if stats["invalid_candidate_decision"] or stats["invalid_added_candidate"]:
        stats["quarantined"] = 1
        return [], [], [], stats
    normalized = {
        "primary_spans": sorted(
            primary_spans,
            key=lambda item: (item["start"], item["end"], item["type"]),
        ),
        "bernoulli_spans": sorted(
            bernoulli_spans,
            key=lambda item: (item["start"], item["end"], item["type"]),
        ),
        "subclass_spans": payload["subclass_spans"],
    }
    predictions, categorical, legacy_stats = parse_subclass_annotation(
        json.dumps(normalized, ensure_ascii=False),
        text,
        base_spans,
        spec,
    )
    stats.update(legacy_stats)
    if stats["quarantined"]:
        return [], [], [], stats
    return predictions, categorical, normalized_decisions, stats


def validate_component_weight(value: Any, where: str) -> float:
    """Return one explicit component weight, rejecting bools and non-finite values."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{where} must be finite and nonnegative")
    return float(value)
