"""Train and serve a context-conditioned character surface generator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from trainlib import LengthBucketedBatchSampler

try:
    from pii_projector import TAGSET_PATH, Tagset
    from pii_slot_morphology import SlotMorphology
    from pii_surface_audit import iter_jsonl, parse_span, row_language
    from pii_surface_pool import record_hash, tags_for_span

    from pii_surface.agreement_mining import SCHEMA as AGREEMENT_SCHEMA
    from pii_surface.exact_agreement_pool import (
        file_sha256,
        split_for_source,
        validate_exact_candidate,
        validated_agreement_report,
    )
    from pii_surface.mix_policy import LEARNED_OPEN_CLASS_TAGS
except ModuleNotFoundError:  # Imported as scripts.pii_surface.context_generator in tests.
    from scripts.pii_projector import TAGSET_PATH, Tagset
    from scripts.pii_slot_morphology import SlotMorphology
    from scripts.pii_surface.agreement_mining import SCHEMA as AGREEMENT_SCHEMA
    from scripts.pii_surface.exact_agreement_pool import (
        file_sha256,
        split_for_source,
        validate_exact_candidate,
        validated_agreement_report,
    )
    from scripts.pii_surface.mix_policy import LEARNED_OPEN_CLASS_TAGS
    from scripts.pii_surface_audit import iter_jsonl, parse_span, row_language
    from scripts.pii_surface_pool import record_hash, tags_for_span

SCHEMA = "pii-surface-context-char-gru-v1"
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
LEFT = "<left>"
TARGET = "<target>"
RIGHT = "<right>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK, LEFT, TARGET, RIGHT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAG_CONDITIONING_EXACT = "exact"
TAG_CONDITIONING_P20_P9 = "exact+p20+p9-additive"
TAG_CONDITIONING_MODES = (TAG_CONDITIONING_EXACT, TAG_CONDITIONING_P20_P9)
P20_CUT = "redaction_20_v1"
P9_CUT = "redaction_9_v1"
_TAGSET = Tagset()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ContextExample:
    candidate_id: str
    source_text_sha256: str
    split: str
    language: str
    tag: str
    left: str
    right: str
    target: str
    # Case the slot demands, derived from the carrier context by
    # scripts/pii_slot_morphology.py. "none" means undetected or disabled, so
    # a model trained without slot conditioning sees a single constant value.
    slot: str = "none"

    @property
    def context_length(self) -> int:
        return len(self.left) + len(self.right) + 3


@dataclass(frozen=True)
class LoadedContextInputs:
    """One provenance-checked corpus reusable across scoped expert fits."""

    examples: tuple[ContextExample, ...]
    input_receipt: dict[str, Any]
    examples_by_source: dict[str, int]
    rejected_by_source: dict[str, dict[str, int]]


class CharacterVocabulary:
    def __init__(self, tokens: Sequence[str]):
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("character vocabulary must start with the declared special tokens")
        if len(set(tokens)) != len(tokens):
            raise ValueError("character vocabulary tokens must be unique")
        self.tokens = tuple(tokens)
        self.ids = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, examples: Sequence[ContextExample]) -> CharacterVocabulary:
        characters = sorted(
            {
                character
                for example in examples
                for text in (example.left, example.right, example.target)
                for character in text
            }
        )
        return cls((*SPECIAL_TOKENS, *characters))

    def encode_characters(self, text: str) -> list[int]:
        unknown = self.ids[UNK]
        return [self.ids.get(character, unknown) for character in text]

    def encode_context(self, left: str, right: str) -> list[int]:
        return [
            self.ids[LEFT],
            *self.encode_characters(left),
            self.ids[TARGET],
            *self.encode_characters(right),
            self.ids[RIGHT],
        ]

    def encode_target(self, target: str) -> list[int]:
        return [self.ids[BOS], *self.encode_characters(target), self.ids[EOS]]

    def decode(self, token_ids: Iterable[int]) -> str | None:
        characters = []
        for token_id in token_ids:
            token = self.tokens[int(token_id)]
            if token == EOS:
                break
            if token in SPECIAL_TOKENS:
                return None
            characters.append(token)
        return "".join(characters)


def load_examples(
    agreement_path: Path,
    *,
    context_chars: int,
    max_target_chars: int,
    holdout_modulus: int = 10,
    holdout_fold: int = 0,
    slot_morphology: Any = None,
) -> tuple[list[ContextExample], dict[str, int]]:
    if context_chars < 0:
        raise ValueError("context_chars must be nonnegative")
    if max_target_chars <= 0:
        raise ValueError("max_target_chars must be positive")
    examples = []
    rejected = Counter()
    candidate_ids = set()
    with agreement_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_exact_candidate(row, f"{agreement_path}:{line_number}")
            if row["schema"] != AGREEMENT_SCHEMA:
                raise ValueError(f"{agreement_path}:{line_number}: unexpected agreement schema")
            if row["candidate_id"] in candidate_ids:
                raise ValueError(f"{agreement_path}:{line_number}: duplicate candidate_id")
            candidate_ids.add(row["candidate_id"])
            if row["tag"] not in LEARNED_OPEN_CLASS_TAGS:
                rejected["closed_or_structured_tag"] += 1
                continue
            if len(row["surface"]) > max_target_chars:
                rejected["target_too_long"] += 1
                continue
            relative_start = row["start"] - row["context_start"]
            relative_end = row["end"] - row["context_start"]
            left = row["context"][max(0, relative_start - context_chars) : relative_start]
            right = row["context"][relative_end : relative_end + context_chars]
            examples.append(
                ContextExample(
                    candidate_id=row["candidate_id"],
                    source_text_sha256=row["source_text_sha256"],
                    split=split_for_source(row["source_text_sha256"], holdout_modulus, holdout_fold),
                    language=row["language"],
                    tag=row["tag"],
                    left=left,
                    right=right,
                    target=row["surface"],
                    # the case the carrier context demands, from the same
                    # detector the realizer uses, so training and realization
                    # cannot disagree about what slot a surface sits in
                    slot=(
                        slot_morphology.detect_slot(row["language"], left, right) or "none"
                        if slot_morphology is not None
                        else "none"
                    ),
                )
            )
    return examples, dict(sorted(rejected.items()))


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_carrier_examples(
    carrier_pool_report_path: Path,
    *,
    context_chars: int,
    max_target_chars: int,
) -> tuple[list[ContextExample], dict[str, int], list[dict[str, Any]]]:
    """Recover exact carrier/surface pairs from a surface-pool build receipt."""
    report = json.loads(carrier_pool_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"{carrier_pool_report_path}: expected surface-pool schema version 1")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"{carrier_pool_report_path}: missing configuration")
    holdout_modulus = int(configuration.get("holdout_modulus", 0))
    holdout_fold = int(configuration.get("holdout_fold", -1))
    if holdout_modulus < 2 or not 0 <= holdout_fold < holdout_modulus:
        raise ValueError(f"{carrier_pool_report_path}: invalid source partition configuration")
    raw_languages = configuration.get("languages")
    languages = set(raw_languages) if isinstance(raw_languages, list) else None
    raw_tags = configuration.get("tags")
    if raw_tags is not None and (
        not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags)
    ):
        raise ValueError(f"{carrier_pool_report_path}: invalid tag scope")
    tags = set(raw_tags) if isinstance(raw_tags, list) else None
    input_files = report.get("input_files")
    source_manifests = report.get("source_manifests")
    if not isinstance(input_files, dict) or not isinstance(source_manifests, dict):
        raise ValueError(f"{carrier_pool_report_path}: missing input or source-manifest map")

    examples = []
    rejected = Counter()
    source_files = []
    candidate_ids = set()
    for dataset, raw_paths in sorted(input_files.items()):
        if not isinstance(dataset, str) or not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError(f"{carrier_pool_report_path}: malformed input-files entry")
        manifest = source_manifests.get(dataset)
        if not isinstance(manifest, dict):
            raise ValueError(f"{carrier_pool_report_path}: missing source manifest for {dataset!r}")
        manifest_path = resolve_project_path(str(manifest.get("path", "")))
        expected_manifest_hash = manifest.get("sha256")
        if not manifest_path.is_file() or file_sha256(manifest_path) != expected_manifest_hash:
            raise ValueError(f"{carrier_pool_report_path}: source manifest drift for {dataset!r}")
        for raw_path in raw_paths:
            path = resolve_project_path(str(raw_path))
            if not path.is_file():
                raise ValueError(f"{carrier_pool_report_path}: missing carrier input {path}")
            source_files.append({"dataset": dataset, "path": str(path), "sha256": file_sha256(path)})
            for line_number, row in enumerate(iter_jsonl(path), 1):
                language = row_language(row)
                text = row.get("text")
                if language is None or not isinstance(text, str):
                    rejected["rows_missing_language_or_text"] += 1
                    continue
                if languages is not None and language not in languages:
                    continue
                row_id = str(row.get("id", line_number))
                source_record_hash = record_hash(dataset, row_id, row)
                source_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                split = split_for_source(source_text_sha256, holdout_modulus, holdout_fold)
                raw_spans = row.get("spans")
                if not isinstance(raw_spans, list):
                    rejected["rows_with_non_list_spans"] += 1
                    continue
                for raw_span in raw_spans:
                    span = parse_span(raw_span)
                    if span is None or span.start < 0 or span.end <= span.start or span.end > len(text):
                        rejected["invalid_spans"] += 1
                        continue
                    value = text[span.start : span.end]
                    if not value.strip():
                        rejected["empty_values"] += 1
                        continue
                    emitted = tags_for_span(
                        language,
                        span.label,
                        value,
                        derive_name_components=False,
                    )
                    if not emitted:
                        rejected["unsupported_labels"] += 1
                        continue
                    for tag, surface in emitted:
                        if tags is not None and tag not in tags:
                            rejected["tag_outside_scope"] += 1
                            continue
                        if tag not in LEARNED_OPEN_CLASS_TAGS:
                            rejected["closed_or_structured_tag"] += 1
                            continue
                        if surface != value:
                            raise ValueError("direct carrier tags must preserve the annotated span surface")
                        if len(surface) > max_target_chars:
                            rejected["target_too_long"] += 1
                            continue
                        target_start = span.start
                        target_end = span.end
                        candidate_id = hashlib.sha256(
                            f"carrier\0{dataset}\0{source_record_hash}\0{target_start}\0{target_end}\0{tag}".encode()
                        ).hexdigest()[:24]
                        if candidate_id in candidate_ids:
                            raise ValueError(f"{path}: duplicate carrier candidate identity")
                        candidate_ids.add(candidate_id)
                        examples.append(
                            ContextExample(
                                candidate_id=candidate_id,
                                source_text_sha256=source_text_sha256,
                                split=split,
                                language=language,
                                tag=tag,
                                left=text[max(0, target_start - context_chars) : target_start],
                                right=text[target_end : target_end + context_chars],
                                target=surface,
                            )
                        )
    return examples, dict(sorted(rejected.items())), source_files


def load_natural_examples(
    natural_pool_report_path: Path,
    *,
    context_chars: int,
    max_target_chars: int,
) -> tuple[list[ContextExample], dict[str, int], list[dict[str, Any]]]:
    """Backward-compatible name for loading exact surface carriers."""
    return load_carrier_examples(
        natural_pool_report_path,
        context_chars=context_chars,
        max_target_chars=max_target_chars,
    )


def indexed_vocabulary(values: Iterable[str]) -> tuple[tuple[str, ...], dict[str, int]]:
    items = (UNK, *sorted(set(values)))
    return items, {value: index for index, value in enumerate(items)}


@dataclass(frozen=True)
class ModelDimensions:
    character_embedding: int = 64
    condition_embedding: int = 24
    context_hidden: int = 128
    decoder_hidden: int = 192
    dropout: float = 0.1


@dataclass(frozen=True)
class TagConditioning:
    """Frozen tag-to-cut lookups for one model checkpoint."""

    mode: str = TAG_CONDITIONING_EXACT
    p20_labels: tuple[str, ...] = ()
    p9_labels: tuple[str, ...] = ()
    tag_to_p20_ids: tuple[int, ...] = ()
    tag_to_p9_ids: tuple[int, ...] = ()

    def validate(self, tag_count: int) -> None:
        if self.mode not in TAG_CONDITIONING_MODES:
            raise ValueError(f"unknown tag conditioning mode {self.mode!r}")
        if self.mode == TAG_CONDITIONING_EXACT:
            if any((self.p20_labels, self.p9_labels, self.tag_to_p20_ids, self.tag_to_p9_ids)):
                raise ValueError("exact tag conditioning cannot carry projected-cut lookups")
            return
        if not self.p20_labels or not self.p9_labels:
            raise ValueError("hierarchical tag conditioning requires nonempty P20 and P9 vocabularies")
        if self.p20_labels[0] != UNK or self.p9_labels[0] != UNK:
            raise ValueError("projected-cut vocabularies must start with <unk>")
        if len(self.tag_to_p20_ids) != tag_count or len(self.tag_to_p9_ids) != tag_count:
            raise ValueError("projected-cut lookup length must equal the exact-tag vocabulary length")
        if any(not 0 <= value < len(self.p20_labels) for value in self.tag_to_p20_ids):
            raise ValueError("P20 lookup contains an out-of-range id")
        if any(not 0 <= value < len(self.p9_labels) for value in self.tag_to_p9_ids):
            raise ValueError("P9 lookup contains an out-of-range id")

    def to_config(self) -> dict[str, Any]:
        if self.mode == TAG_CONDITIONING_EXACT:
            return {"mode": self.mode}
        return {
            "mode": self.mode,
            "cuts": {"p20": P20_CUT, "p9": P9_CUT},
            "p20_labels": list(self.p20_labels),
            "p9_labels": list(self.p9_labels),
            "tag_to_p20_ids": list(self.tag_to_p20_ids),
            "tag_to_p9_ids": list(self.tag_to_p9_ids),
            "tagset": {"path": str(TAGSET_PATH), "sha256": file_sha256(Path(TAGSET_PATH))},
        }

    @classmethod
    def from_config(cls, raw: Any, *, tag_count: int) -> TagConditioning:
        if raw is None:
            result = cls()
        elif not isinstance(raw, dict):
            raise ValueError("tag_conditioning must be an object")
        else:
            allowed = {
                "mode",
                "cuts",
                "p20_labels",
                "p9_labels",
                "tag_to_p20_ids",
                "tag_to_p9_ids",
                "tagset",
            }
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"tag_conditioning has unknown keys: {', '.join(sorted(unknown))}")
            result = cls(
                mode=str(raw.get("mode", TAG_CONDITIONING_EXACT)),
                p20_labels=tuple(raw.get("p20_labels") or ()),
                p9_labels=tuple(raw.get("p9_labels") or ()),
                tag_to_p20_ids=tuple(int(value) for value in raw.get("tag_to_p20_ids") or ()),
                tag_to_p9_ids=tuple(int(value) for value in raw.get("tag_to_p9_ids") or ()),
            )
            if result.mode == TAG_CONDITIONING_P20_P9 and raw.get("cuts") != {
                "p20": P20_CUT,
                "p9": P9_CUT,
            }:
                raise ValueError("hierarchical tag conditioning must declare the P20 and P9 cuts")
        result.validate(tag_count)
        return result


def build_tag_conditioning(tags: Sequence[str], mode: str) -> TagConditioning:
    if mode == TAG_CONDITIONING_EXACT:
        return TagConditioning()
    if mode != TAG_CONDITIONING_P20_P9:
        raise ValueError(f"unknown tag conditioning mode {mode!r}")
    projected_p20 = []
    projected_p9 = []
    for tag in tags:
        if tag == UNK:
            projected_p20.append(UNK)
            projected_p9.append(UNK)
            continue
        node = tag.casefold()
        if node not in _TAGSET.nodes:
            raise ValueError(f"exact tag {tag!r} has no canonical ontology node")
        projected_p20.append(_TAGSET.project_canonical_cut(node, P20_CUT))
        projected_p9.append(_TAGSET.project_canonical_cut(node, P9_CUT))
    p20_labels, p20_ids = indexed_vocabulary(projected_p20[1:])
    p9_labels, p9_ids = indexed_vocabulary(projected_p9[1:])
    result = TagConditioning(
        mode=mode,
        p20_labels=p20_labels,
        p9_labels=p9_labels,
        tag_to_p20_ids=tuple(p20_ids[value] for value in projected_p20),
        tag_to_p9_ids=tuple(p9_ids[value] for value in projected_p9),
    )
    result.validate(len(tags))
    return result


class ContextCharacterGenerator(nn.Module):
    """Encode both carrier sides and autoregressively generate one surface."""

    p20_embedding: nn.Embedding | None
    p9_embedding: nn.Embedding | None
    tag_to_p20_ids: torch.Tensor
    tag_to_p9_ids: torch.Tensor

    def __init__(
        self,
        *,
        vocabulary_size: int,
        language_count: int,
        tag_count: int,
        dimensions: ModelDimensions,
        tag_conditioning: TagConditioning | None = None,
        slot_count: int = 0,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.tag_conditioning = tag_conditioning or TagConditioning()
        self.tag_conditioning.validate(tag_count)
        self.character_embedding = nn.Embedding(
            vocabulary_size,
            dimensions.character_embedding,
            padding_idx=0,
        )
        self.language_embedding = nn.Embedding(language_count, dimensions.condition_embedding)
        self.tag_embedding = nn.Embedding(tag_count, dimensions.condition_embedding)
        # Slot conditioning is opt-in. With slot_count 0 the module has no slot
        # embedding and the condition width is unchanged, so a checkpoint
        # trained before this existed still loads.
        self.slot_count = int(slot_count or 0)
        self.slot_embedding = (
            nn.Embedding(self.slot_count, dimensions.condition_embedding) if self.slot_count else None
        )
        self.context_encoder = nn.GRU(
            dimensions.character_embedding,
            dimensions.context_hidden,
            batch_first=True,
            bidirectional=True,
        )
        condition_width = 2 * dimensions.context_hidden + 2 * dimensions.condition_embedding
        if self.slot_embedding is not None:
            condition_width += dimensions.condition_embedding
        self.decoder_initial = nn.Linear(condition_width, dimensions.decoder_hidden)
        self.decoder = nn.GRU(
            dimensions.character_embedding + condition_width,
            dimensions.decoder_hidden,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dimensions.dropout)
        self.output = nn.Linear(dimensions.decoder_hidden, vocabulary_size)
        self.p20_embedding = None
        self.p9_embedding = None
        if self.tag_conditioning.mode == TAG_CONDITIONING_P20_P9:
            self.p20_embedding = nn.Embedding(
                len(self.tag_conditioning.p20_labels),
                dimensions.condition_embedding,
            )
            self.p9_embedding = nn.Embedding(
                len(self.tag_conditioning.p9_labels),
                dimensions.condition_embedding,
            )
            self.register_buffer(
                "tag_to_p20_ids",
                torch.tensor(self.tag_conditioning.tag_to_p20_ids, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "tag_to_p9_ids",
                torch.tensor(self.tag_conditioning.tag_to_p9_ids, dtype=torch.long),
                persistent=False,
            )

    def encode_tag(self, tag_ids: torch.Tensor) -> torch.Tensor:
        encoded = self.tag_embedding(tag_ids)
        if self.tag_conditioning.mode == TAG_CONDITIONING_P20_P9:
            assert self.p20_embedding is not None and self.p9_embedding is not None
            encoded = (
                encoded
                + self.p20_embedding(self.tag_to_p20_ids[tag_ids])
                + self.p9_embedding(self.tag_to_p9_ids[tag_ids])
            )
        return encoded

    def encode_condition(
        self,
        context_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        language_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        *,
        slot_ids: torch.Tensor | None = None,
        tag_condition_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.dropout(self.character_embedding(context_ids))
        packed = pack_padded_sequence(
            embedded,
            context_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _packed_output, hidden = self.context_encoder(packed)
        context = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        tag = self.encode_tag(tag_ids)
        if tag_condition_weights is not None:
            if tag_condition_weights.shape not in {(len(tag_ids),), (len(tag_ids), 1)}:
                raise ValueError("tag condition weights must have shape [batch] or [batch, 1]")
            tag = tag * tag_condition_weights.reshape(-1, 1).to(tag)
        parts = [context, self.language_embedding(language_ids), tag]
        if self.slot_embedding is not None:
            if slot_ids is None:
                raise ValueError("model was built with slot conditioning but no slot_ids were given")
            parts.append(self.slot_embedding(slot_ids))
        return torch.cat(parts, dim=-1)

    def decode_logits(
        self,
        decoder_input_ids: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.dropout(self.character_embedding(decoder_input_ids))
        repeated = condition.unsqueeze(1).expand(-1, embedded.shape[1], -1)
        initial = torch.tanh(self.decoder_initial(condition)).unsqueeze(0)
        decoded, _hidden = self.decoder(torch.cat((embedded, repeated), dim=-1), initial)
        return self.output(self.dropout(decoded))

    def forward(
        self,
        context_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        language_ids: torch.Tensor,
        tag_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *,
        slot_ids: torch.Tensor | None = None,
        tag_condition_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(
            context_ids,
            context_lengths,
            language_ids,
            tag_ids,
            slot_ids=slot_ids,
            tag_condition_weights=tag_condition_weights,
        )
        return self.decode_logits(decoder_input_ids, condition)


class EncodedExamples(Dataset):
    def __init__(
        self,
        examples: Sequence[ContextExample],
        *,
        vocabulary: CharacterVocabulary,
        language_ids: Mapping[str, int],
        tag_ids: Mapping[str, int],
        slot_ids: Mapping[str, int] | None = None,
        blank_context: bool = False,
    ):
        self.examples = list(examples)
        self.vocabulary = vocabulary
        self.language_ids = language_ids
        self.tag_ids = tag_ids
        self.slot_ids = slot_ids
        self.blank_context = blank_context

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        left, right = ("", "") if self.blank_context else (example.left, example.right)
        target = self.vocabulary.encode_target(example.target)
        return {
            "context_ids": self.vocabulary.encode_context(left, right),
            "decoder_input_ids": target[:-1],
            "target_ids": target[1:],
            "language_id": self.language_ids.get(example.language, 0),
            "tag_id": self.tag_ids.get(example.tag, 0),
            "slot_id": self.slot_ids.get(example.slot, 0) if self.slot_ids else 0,
            "example": example,
        }


def pad_rows(rows: Sequence[Sequence[int]], padding: int = 0) -> torch.Tensor:
    width = max(len(row) for row in rows)
    result = torch.full((len(rows), width), padding, dtype=torch.long)
    for row_index, row in enumerate(rows):
        result[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def collate_examples(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context_ids": pad_rows([row["context_ids"] for row in rows]),
        "context_lengths": torch.tensor([len(row["context_ids"]) for row in rows]),
        "decoder_input_ids": pad_rows([row["decoder_input_ids"] for row in rows]),
        "target_ids": pad_rows([row["target_ids"] for row in rows], padding=-100),
        "language_ids": torch.tensor([row["language_id"] for row in rows]),
        "tag_ids": torch.tensor([row["tag_id"] for row in rows]),
        # absent means the batch carries no slot annotation, which is the
        # unconditioned path; the model ignores these unless it has a slot
        # embedding, so 0 is a placeholder rather than a silent fallback
        "slot_ids": torch.tensor([row.get("slot_id", 0) for row in rows]),
        "examples": [row["example"] for row in rows],
    }


def make_loader(
    dataset: EncodedExamples,
    *,
    batch_size: int,
    bucket_width: int,
    seed: int,
) -> DataLoader:
    lengths = [example.context_length + len(example.target) for example in dataset.examples]
    sampler = LengthBucketedBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        bucket_width=bucket_width,
        fill="nearest",
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_examples, num_workers=0)


def batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()
    }


def loss_sum_and_characters(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, int((targets != -100).sum().item())


@torch.inference_mode()
def evaluate(
    model: ContextCharacterGenerator,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_characters = 0
    groups: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        logits = model(
            batch["context_ids"],
            batch["context_lengths"],
            batch["language_ids"],
            batch["tag_ids"],
            batch["decoder_input_ids"],
            slot_ids=batch["slot_ids"],
        )
        losses = F.cross_entropy(
            logits.transpose(1, 2),
            batch["target_ids"],
            ignore_index=-100,
            reduction="none",
        )
        mask = batch["target_ids"] != -100
        total_loss += float(losses[mask].sum().item())
        total_characters += int(mask.sum().item())
        for row_index, example in enumerate(raw_batch["examples"]):
            row_mask = mask[row_index]
            key = example.language, example.tag
            groups[key][0] += float(losses[row_index][row_mask].sum().item())
            groups[key][1] += int(row_mask.sum().item())
    mean_nll = total_loss / max(1, total_characters)
    return {
        "characters": total_characters,
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(20.0, mean_nll)),
        "groups": [
            {
                "language": language,
                "tag": tag,
                "characters": int(values[1]),
                "mean_nll": values[0] / max(1.0, values[1]),
                "perplexity": math.exp(min(20.0, values[0] / max(1.0, values[1]))),
            }
            for (language, tag), values in sorted(groups.items())
        ],
    }


def linear_warmup_cosine(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def generation_is_usable(value: str | None, max_chars: int) -> bool:
    return bool(
        value
        and value.strip()
        and len(value) <= max_chars
        and not any(character in "\r\n\t" or ord(character) < 32 for character in value)
        and not (value.startswith("[") and value.endswith("]"))
    )


class LoadedContextSurfaceGenerator:
    """One checkpoint-backed generator shared by training and decode paths."""

    def __init__(self, path: Path, *, device: str | torch.device = "cpu"):
        self.path = Path(path)
        self.config_path = self.path / "config.json"
        self.weights_path = self.path / "model.pt"
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if config.get("schema") != SCHEMA:
            raise ValueError(f"{self.config_path}: expected schema {SCHEMA!r}")
        expected_hash = config.get("weights_sha256")
        if expected_hash != file_sha256(self.weights_path):
            raise ValueError(f"{self.weights_path}: hash differs from config receipt")
        self.config = config
        self.vocabulary = CharacterVocabulary(config["vocabulary"])
        self.languages = tuple(config["languages"])
        self.tags = tuple(config["tags"])
        self.language_ids = {value: index for index, value in enumerate(self.languages)}
        self.tag_ids = {value: index for index, value in enumerate(self.tags)}
        self.slots = tuple(config.get("slots") or ())
        self.slot_ids = {value: index for index, value in enumerate(self.slots)}
        self.tag_conditioning = TagConditioning.from_config(
            config.get("tag_conditioning"),
            tag_count=len(self.tags),
        )
        self.context_chars = int(config["context_chars"])
        self.max_target_chars = int(config["max_target_chars"])
        self.device = torch.device(device)
        dimensions = ModelDimensions(**config["dimensions"])
        self.model = ContextCharacterGenerator(
            vocabulary_size=len(self.vocabulary.tokens),
            language_count=len(self.languages),
            tag_count=len(self.tags),
            dimensions=dimensions,
            tag_conditioning=self.tag_conditioning,
            slot_count=len(self.slots),
        ).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        language: str,
        tag: str,
        left: str,
        right: str,
        *,
        seed: int,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> tuple[str, str] | None:
        if language not in self.language_ids or tag not in self.tag_ids:
            return None
        temperature = float(self.config["decoding"]["temperature"] if temperature is None else temperature)
        top_k = int(self.config["decoding"]["top_k"] if top_k is None else top_k)
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        left = left[-self.context_chars :] if self.context_chars else ""
        right = right[: self.context_chars] if self.context_chars else ""
        context_ids = torch.tensor(
            [self.vocabulary.encode_context(left, right)],
            dtype=torch.long,
            device=self.device,
        )
        condition = self.model.encode_condition(
            context_ids,
            torch.tensor([context_ids.shape[1]], device=self.device),
            torch.tensor([self.language_ids[language]], device=self.device),
            torch.tensor([self.tag_ids[tag]], device=self.device),
        )
        hidden = torch.tanh(self.model.decoder_initial(condition)).unsqueeze(0)
        token_id = self.vocabulary.ids[BOS]
        generated = []
        torch_generator = torch.Generator(device=self.device).manual_seed(int(seed) % (2**63 - 1))
        forbidden = [self.vocabulary.ids[token] for token in (PAD, BOS, UNK, LEFT, TARGET, RIGHT)]
        for _position in range(self.max_target_chars + 1):
            decoder_input = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            embedded = self.model.character_embedding(decoder_input)
            decoded, hidden = self.model.decoder(
                torch.cat((embedded, condition.unsqueeze(1)), dim=-1),
                hidden,
            )
            logits = self.model.output(decoded[:, -1]).squeeze(0)
            logits[forbidden] = -torch.inf
            if not generated:
                logits[self.vocabulary.ids[EOS]] = -torch.inf
            if temperature == 0:
                token_id = int(logits.argmax().item())
            else:
                scaled = logits / temperature
                if top_k and top_k < scaled.numel():
                    threshold = torch.topk(scaled, top_k).values[-1]
                    scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
                token_id = int(torch.multinomial(scaled.softmax(dim=-1), 1, generator=torch_generator).item())
            if token_id == self.vocabulary.ids[EOS]:
                break
            generated.append(token_id)
        value = self.vocabulary.decode(generated)
        if not generation_is_usable(value, self.max_target_chars):
            return None
        assert value is not None
        mode = "greedy" if temperature == 0 else f"sample-t{temperature:g}-k{top_k}"
        return value, f"context-char-gru:{self.config['model_version']}:{mode}"

    def receipt(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "model_version": self.config["model_version"],
            "config_sha256": file_sha256(self.config_path),
            "weights_sha256": file_sha256(self.weights_path),
            "context_chars": self.context_chars,
            "supported_languages": list(self.languages[1:]),
            "supported_tags": list(self.tags[1:]),
            "tag_conditioning": self.tag_conditioning.to_config(),
            "decoding": self.config["decoding"],
        }


def select_diagnostic_examples(examples: Sequence[ContextExample]) -> list[ContextExample]:
    selected = {}
    for example in sorted(examples, key=lambda item: (item.language, item.tag, item.candidate_id)):
        selected.setdefault((example.language, example.tag), example)
    return list(selected.values())


def write_headline(text: str) -> None:
    path = os.environ.get("AGENTCTL_HEADLINE_FILE")
    if path:
        Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def load_context_inputs(
    *,
    agreement_path: Path | None,
    agreement_report_path: Path | None,
    natural_pool_report_path: Path | None = None,
    carrier_pool_report_paths: Sequence[Path] = (),
    context_chars: int,
    max_target_chars: int,
    slot_morphology: Any = None,
) -> LoadedContextInputs:
    """Load and verify all carrier sources once for one or more model fits."""
    if (agreement_path is None) != (agreement_report_path is None):
        raise ValueError("agreement and agreement-report must be supplied together")
    agreement_examples: list[ContextExample] = []
    agreement_rejected: dict[str, int] = {}
    if agreement_path is not None and agreement_report_path is not None:
        validated_agreement_report(agreement_report_path, agreement_path)
        agreement_examples, agreement_rejected = load_examples(
            agreement_path,
            context_chars=context_chars,
            max_target_chars=max_target_chars,
            slot_morphology=slot_morphology,
        )
    pool_report_paths = list(carrier_pool_report_paths)
    if natural_pool_report_path is not None:
        pool_report_paths.insert(0, natural_pool_report_path)
    if len({path.resolve() for path in pool_report_paths}) != len(pool_report_paths):
        raise ValueError("carrier-pool reports must be unique")
    carrier_examples: list[ContextExample] = []
    carrier_inputs = []
    pool_names = set()
    for pool_report_path in pool_report_paths:
        pool_report = json.loads(pool_report_path.read_text(encoding="utf-8"))
        pool_name = str(pool_report.get("pool_version") or pool_report_path.stem)
        if pool_name in pool_names:
            raise ValueError(f"duplicate carrier-pool version: {pool_name}")
        pool_names.add(pool_name)
        loaded_examples, rejected, source_files = load_carrier_examples(
            pool_report_path,
            context_chars=context_chars,
            max_target_chars=max_target_chars,
        )
        configuration = pool_report.get("configuration")
        carrier_role = (
            configuration.get("carrier_role", "train-and-audit")
            if isinstance(configuration, dict)
            else "train-and-audit"
        )
        if carrier_role not in {"train-and-audit", "train-only"}:
            raise ValueError(f"{pool_report_path}: unsupported carrier role {carrier_role!r}")
        withheld_audit_examples = 0
        if carrier_role == "train-only":
            withheld_audit_examples = sum(example.split == "audit" for example in loaded_examples)
            loaded_examples = [example for example in loaded_examples if example.split == "train"]
        carrier_examples.extend(loaded_examples)
        carrier_inputs.append(
            {
                "pool_version": pool_name,
                "carrier_role": carrier_role,
                "report_path": str(pool_report_path),
                "report_sha256": file_sha256(pool_report_path),
                "examples": len(loaded_examples),
                "withheld_audit_examples": withheld_audit_examples,
                "rejected": rejected,
                "source_files": source_files,
            }
        )
    examples = (*agreement_examples, *carrier_examples)
    candidate_ids = [example.candidate_id for example in examples]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("context-generator inputs contain duplicate candidate identities")
    return LoadedContextInputs(
        examples=examples,
        input_receipt={
            "agreement_path": str(agreement_path) if agreement_path is not None else None,
            "agreement_sha256": file_sha256(agreement_path) if agreement_path is not None else None,
            "agreement_report_path": (
                str(agreement_report_path) if agreement_report_path is not None else None
            ),
            "agreement_report_sha256": (
                file_sha256(agreement_report_path) if agreement_report_path is not None else None
            ),
            "carrier_pools": carrier_inputs,
        },
        examples_by_source={
            "exact_agreement": len(agreement_examples),
            **{item["pool_version"]: item["examples"] for item in carrier_inputs},
        },
        rejected_by_source={
            "exact_agreement": agreement_rejected,
            **{item["pool_version"]: item["rejected"] for item in carrier_inputs},
        },
    )


def train_context_generator(
    *,
    agreement_path: Path | None,
    agreement_report_path: Path | None,
    natural_pool_report_path: Path | None = None,
    carrier_pool_report_paths: Sequence[Path] = (),
    output: Path,
    model_version: str,
    context_chars: int = 20,
    max_target_chars: int = 64,
    slot_morphology_path: Path | None = None,
    dimensions: ModelDimensions = ModelDimensions(),
    batch_size: int = 64,
    bucket_width: int = 16,
    epochs: int = 20,
    learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    gradient_clip: float = 1.0,
    seed: int = 20260815,
    device: str = "auto",
    decoding_temperature: float = 0.8,
    decoding_top_k: int = 20,
    tag_conditioning_mode: str = TAG_CONDITIONING_EXACT,
    loaded_inputs: LoadedContextInputs | None = None,
    include_languages: Sequence[str] = (),
    include_tags: Sequence[str] = (),
    fixed_vocabulary: Sequence[str] | None = None,
    scope: Mapping[str, Any] | None = None,
    vocabulary_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not model_version:
        raise ValueError("model_version must be nonempty")
    if batch_size <= 0 or epochs <= 0 or bucket_width <= 0:
        raise ValueError("batch_size, epochs, and bucket_width must be positive")
    if not 0 <= warmup_ratio < 1 or not 0 <= dimensions.dropout < 1:
        raise ValueError("warmup_ratio and dropout must be in [0, 1)")
    if loaded_inputs is None:
        loaded_inputs = load_context_inputs(
            agreement_path=agreement_path,
            agreement_report_path=agreement_report_path,
            natural_pool_report_path=natural_pool_report_path,
            carrier_pool_report_paths=carrier_pool_report_paths,
            context_chars=context_chars,
            max_target_chars=max_target_chars,
            slot_morphology=(SlotMorphology.load(slot_morphology_path) if slot_morphology_path else None),
        )
    elif any(
        value
        for value in (
            agreement_path,
            agreement_report_path,
            natural_pool_report_path,
            *carrier_pool_report_paths,
        )
    ):
        raise ValueError("loaded_inputs cannot be combined with input paths")
    language_scope = frozenset(include_languages)
    tag_scope = frozenset(include_tags)
    examples = [
        replace(
            example,
            left=example.left[-context_chars:] if context_chars else "",
            right=example.right[:context_chars] if context_chars else "",
        )
        for example in loaded_inputs.examples
        if (not language_scope or example.language in language_scope)
        and (not tag_scope or example.tag in tag_scope)
    ]
    train_examples = [example for example in examples if example.split == "train"]
    audit_examples = [example for example in examples if example.split == "audit"]
    if not train_examples or not audit_examples:
        raise ValueError("context generator requires nonempty train and audit partitions")
    vocabulary = (
        CharacterVocabulary(fixed_vocabulary)
        if fixed_vocabulary is not None
        else CharacterVocabulary.build(train_examples)
    )
    languages, language_ids = indexed_vocabulary(example.language for example in train_examples)
    tags, tag_ids = indexed_vocabulary(example.tag for example in train_examples)
    tag_conditioning = build_tag_conditioning(tags, tag_conditioning_mode)
    # Slot conditioning switches itself on only when the data actually carries
    # more than the constant "none", so a corpus prepared without a
    # slot-morphology config trains the identical architecture as before.
    observed_slots = {example.slot for example in train_examples}
    if observed_slots - {"none"}:
        slots, slot_ids = indexed_vocabulary(example.slot for example in train_examples)
    else:
        slots, slot_ids = (), None
    train_dataset = EncodedExamples(
        train_examples,
        vocabulary=vocabulary,
        language_ids=language_ids,
        tag_ids=tag_ids,
        slot_ids=slot_ids,
    )
    audit_dataset = EncodedExamples(
        audit_examples,
        vocabulary=vocabulary,
        language_ids=language_ids,
        tag_ids=tag_ids,
        slot_ids=slot_ids,
    )
    blank_audit_dataset = EncodedExamples(
        audit_examples,
        vocabulary=vocabulary,
        language_ids=language_ids,
        tag_ids=tag_ids,
        slot_ids=slot_ids,
        blank_context=True,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        bucket_width=bucket_width,
        seed=seed,
    )
    audit_loader = make_loader(
        audit_dataset,
        batch_size=batch_size,
        bucket_width=bucket_width,
        seed=seed + 1,
    )
    blank_audit_loader = make_loader(
        blank_audit_dataset,
        batch_size=batch_size,
        bucket_width=bucket_width,
        seed=seed + 1,
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = ContextCharacterGenerator(
        vocabulary_size=len(vocabulary.tokens),
        language_count=len(languages),
        tag_count=len(tags),
        slot_count=len(slots),
        dimensions=dimensions,
        tag_conditioning=tag_conditioning,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = round(total_steps * warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: linear_warmup_cosine(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )
    best_state = None
    best_audit_nll = math.inf
    history = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_characters = 0
        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["context_ids"],
                batch["context_lengths"],
                batch["language_ids"],
                batch["tag_ids"],
                batch["decoder_input_ids"],
                slot_ids=batch["slot_ids"],
            )
            loss_sum, characters = loss_sum_and_characters(logits, batch["target_ids"])
            (loss_sum / max(1, characters)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()
            train_loss += float(loss_sum.detach().item())
            train_characters += characters
            global_step += 1
        audit = evaluate(model, audit_loader, device=resolved_device)
        train_nll = train_loss / max(1, train_characters)
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_mean_nll": train_nll,
                "train_perplexity": math.exp(min(20.0, train_nll)),
                "audit_mean_nll": audit["mean_nll"],
                "audit_perplexity": audit["perplexity"],
            }
        )
        print(
            f"SURFACE-GENERATOR epoch={epoch} step={global_step} "
            f"train_ppl={math.exp(min(20.0, train_nll)):.4f} "
            f"audit_ppl={audit['perplexity']:.4f}",
            flush=True,
        )
        write_headline(
            f"surface generator epoch {epoch}/{epochs}: audit char perplexity {audit['perplexity']:.3f}"
        )
        if audit["mean_nll"] < best_audit_nll:
            best_audit_nll = audit["mean_nll"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    audit = evaluate(model, audit_loader, device=resolved_device)
    blank_audit = evaluate(model, blank_audit_loader, device=resolved_device)

    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "model.pt"
    config_path = output / "config.json"
    report_path = output / "report.json"
    samples_path = output / "samples.jsonl"
    if any(path.exists() for path in (weights_path, config_path, report_path, samples_path)):
        raise FileExistsError(f"refusing to overwrite context-generator output in {output}")
    torch.save(best_state, weights_path)
    config = {
        "schema": SCHEMA,
        "model_version": model_version,
        "scope": dict(scope or {}),
        "context_chars": context_chars,
        "max_target_chars": max_target_chars,
        "vocabulary": list(vocabulary.tokens),
        "languages": list(languages),
        "tags": list(tags),
        # empty when the corpus carried no slot annotation; the loader then
        # builds a model with no slot embedding, matching older checkpoints
        "slots": list(slots),
        "tag_conditioning": tag_conditioning.to_config(),
        "dimensions": asdict(dimensions),
        "decoding": {"temperature": decoding_temperature, "top_k": decoding_top_k},
        "vocabulary_receipt": dict(vocabulary_receipt or {}),
        "weights_sha256": file_sha256(weights_path),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = LoadedContextSurfaceGenerator(output, device=resolved_device)
    diagnostic_samples = []
    with samples_path.open("w", encoding="utf-8") as samples_output:
        for example in select_diagnostic_examples(audit_examples):
            generated = loaded.generate(
                example.language,
                example.tag,
                example.left,
                example.right,
                seed=stable_seed(seed, example.candidate_id),
            )
            row = {
                **asdict(example),
                "generated": generated[0] if generated else None,
                "generator": generated[1] if generated else None,
                "target_seen_in_train": any(
                    train.language == example.language
                    and train.tag == example.tag
                    and train.target == example.target
                    for train in train_examples
                ),
            }
            samples_output.write(json.dumps(row, ensure_ascii=False) + "\n")
            diagnostic_samples.append(row)
    train_targets = {(example.language, example.tag, example.target) for example in train_examples}
    audit_seen = sum(
        (example.language, example.tag, example.target) in train_targets for example in audit_examples
    )
    supported_audit_examples = sum(
        example.language in language_ids and example.tag in tag_ids for example in audit_examples
    )
    audit_target_characters = sum(len(example.target) for example in audit_examples)
    audit_unknown_target_characters = sum(
        character not in vocabulary.ids for example in audit_examples for character in example.target
    )
    audit_context_contains_target = sum(
        example.target in example.left or example.target in example.right for example in audit_examples
    )
    train_cell_counts = Counter((example.language, example.tag) for example in train_examples)
    audit_cell_counts = Counter((example.language, example.tag) for example in audit_examples)
    report = {
        "schema": SCHEMA,
        "status": "intrinsic_candidate_not_training_admitted",
        "model_version": model_version,
        "input": loaded_inputs.input_receipt,
        "configuration": {
            "scope": dict(scope or {}),
            "context_chars_per_side": context_chars,
            "max_target_chars": max_target_chars,
            "dimensions": asdict(dimensions),
            "batch_size": batch_size,
            "bucket_width": bucket_width,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "gradient_clip": gradient_clip,
            "seed": seed,
            "device": str(resolved_device),
            "decoding": config["decoding"],
            "tag_conditioning": tag_conditioning.to_config(),
        },
        "data": {
            "train_examples": len(train_examples),
            "audit_examples": len(audit_examples),
            "examples_by_source": loaded_inputs.examples_by_source,
            "examples_by_source_before_scope": loaded_inputs.examples_by_source,
            "rejected_by_source": loaded_inputs.rejected_by_source,
            "train_examples_by_language": dict(
                sorted(Counter(example.language for example in train_examples).items())
            ),
            "audit_examples_by_language": dict(
                sorted(Counter(example.language for example in audit_examples).items())
            ),
            "train_examples_by_language_tag": [
                {"language": language, "tag": tag, "examples": count}
                for (language, tag), count in sorted(train_cell_counts.items())
            ],
            "audit_examples_by_language_tag": [
                {"language": language, "tag": tag, "examples": count}
                for (language, tag), count in sorted(audit_cell_counts.items())
            ],
            "train_languages": sorted({example.language for example in train_examples}),
            "audit_languages": sorted({example.language for example in audit_examples}),
            "train_language_tag_cells": len({(example.language, example.tag) for example in train_examples}),
            "audit_language_tag_cells": len({(example.language, example.tag) for example in audit_examples}),
            "audit_examples_supported_for_generation": supported_audit_examples,
            "audit_examples_supported_for_generation_fraction": supported_audit_examples
            / len(audit_examples),
            "audit_targets_seen_in_train": audit_seen,
            "audit_targets_seen_in_train_fraction": audit_seen / len(audit_examples),
            "audit_target_characters": audit_target_characters,
            "audit_unknown_target_characters": audit_unknown_target_characters,
            "audit_unknown_target_character_fraction": audit_unknown_target_characters
            / max(1, audit_target_characters),
            "audit_bounded_context_contains_target": audit_context_contains_target,
        },
        "metrics": {
            "audit_with_context": audit,
            "audit_blank_context": blank_audit,
            "context_mean_nll_delta": blank_audit["mean_nll"] - audit["mean_nll"],
        },
        "history": history,
        "artifacts": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "weights": {"path": str(weights_path), "sha256": file_sha256(weights_path)},
            "samples": {
                "path": str(samples_path),
                "sha256": file_sha256(samples_path),
                "rows": len(diagnostic_samples),
                "selection": "lexicographically smallest candidate_id in every audit language/tag cell",
            },
        },
        "contract": {
            "partition": (
                "exact agreements and every carrier pool use one source_text_sha256 split, preserving "
                "the source-document boundary across repeated rows, datasets, and pool receipts"
            ),
            "natural_components": (
                "carrier pools retain a fine name component only when the source directly labels that fine "
                "span; generic person spans are not split by a positional name heuristic"
            ),
            "use": "intrinsic candidate only until generated samples pass contextual review and a matched tagger treatment improves",
            "fallback": "unsupported or unusable generation returns None; caller retains exact retrieval or source-preservation policy",
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_headline(
        f"surface generator complete: audit char perplexity {audit['perplexity']:.3f}; "
        f"blank-context {blank_audit['perplexity']:.3f}"
    )
    return report
