#!/usr/bin/env python
"""Multilingual encoder token-classifier on the unified PII tagset
(topics/pii-robust-buildout.md: the deliverable-candidate arm).

Full fine-tune of a multilingual encoder (XLM-R-large / mDeBERTa-v3 /
mmBERT) on a corpus assembled by
scripts/pii_assemble_corpus.py: records {text, spans: [[start, end,
node]]} in the unified tagset, BIOES-aligned here via offset mapping.
The default stock head matches prior runs. ``--head-kind`` selects
concatenated encoder layers feeding a full affine, factorized-linear, or
low-rank nonlinear MLP head. An affine head may additionally concatenate
relative token positions with ``--token-offsets=-1,0,1``. ``--decoder crf``
uses the factorized typed-BIOES CRF in pii_crf_model.py.
``--annotated-boundary-loss`` adds a training-only class-agnostic BIOES loss
on annotated-spans-only rows without changing the inference graph.
``--partial-boundary-retention-loss`` distills a frozen warm-start parent's
BIOES distribution only onto otherwise-unannotated tokens of those rows.
``--partial-o-loss`` softly treats otherwise-unannotated tokens on partial-label
rows as O without changing the weight of trusted O supervision.
``--partial-entity-pu-loss`` instead treats tagged partial tokens as positive
and untagged tokens as unlabeled under a frozen trustworthy-complete class prior.
Its bounded positive-margin and parent-presence KL options are ablations.
``--partial-expected-entity-ratio-loss`` leaves untagged partial tokens latent
but constrains their batch-mean entity probability to a frozen interval derived
from trustworthy complete rows. It keeps ordinary BIOES loss on known spans.
``--complete-presence-loss`` adds a projected entity-vs-O objective only on
complete rows when training through an ontology correctness map.
``--complete-boundary-loss`` adds a projected class-agnostic BIOES objective
only on annotated entity tokens of those trustworthy complete rows.
``--reference-primary-positive-loss-weight`` adds a separately normalized
ordinary BIOES loss over explicitly supervised positive reference tokens. It
does not create negatives on untagged partial-row tokens or add model outputs.
``--partial-primary-objective-weight`` scales ordinary BIOES supervision on
explicitly tagged tokens from positive-only rows. One preserves the current
objective; zero retains those rows' other supervised channels while removing
their primary-label contribution.
``--logical-step-objective-normalization`` weights every physical batch by its
supervised mass so each data objective is normalized once across the complete
gradient-accumulated optimizer step.
``--predicate-loss-weight`` adds independent masked binary token predicates
from explicit ``predicate_spans`` while leaving unlabelled channels and every
token outside a matching activating primary span unknown. Optional compact
objective-weight profiles expand to one weight per token and predicate logit.
``--reference-type-residual-loss-weight`` adds one separately supervised
semantic score per new reference type and shares it across that type's four
BIOES primary rows without replacing ordinary boundary/type supervision.
``--subclass-loss-weight`` adds carrier-conditioned categorical decisions from
explicit ``subclass_spans``. Whole-carrier families pool one decision over the
primary span, while component families retain token-local internal subspans;
each component carries independent objective and learning weights.
``--partial-negative-loss`` uses a source's exhaustively annotated type subset
as negative evidence outside its spans without treating unrelated privacy
types as absent.
``--o-token-loss-weight`` changes the supervised cross-entropy contribution of
trusted O tokens relative to entity tokens without changing inference.
``--fine-label-loss-weight`` can demote fine-label cross-entropy in favor of a
cut-marginal objective, allowing initialized fine rows to act as latent
components of a coarse reporting bucket.

``--union-members`` trains one fine head on two label vocabularies at once. The
head keeps its fine BIOES rows, and a coarser class ``c`` is read off them
functionally: ``y[p-c] = logsumexp_{k in members(c)} z[p-k]``, with ``y[O] =
z[O]``. That is a union -- "this token starts some member of c" -- which no
single affine row can represent, so merging the rows loses accuracy the
projection keeps. Rows say which vocabulary they are annotated in through
``label_space`` (scripts/pii_v2_training_adapter.py --union-labels), and the
loss matrix over the two is:

  * a coarse-labelled (``label_space: "v2"``) row's entity token is supervised
    only through the union: cross-entropy on ``y``, i.e. the fine class is
    latent, exact at the ``y`` level and never a guess at which member was meant;
  * a fine-labelled (``label_space: "v1"``) row's entity token is supervised by
    the convex blend ``w * CE_fine(z) + (1 - w) * CE_marginal(y)``, where the
    group is its own label's owning class and ``w`` follows
    ``--union-v1-fine-weight-schedule``. This fades label *precision*, never data
    presence: sampling weights are untouched, and ``w = 1`` is bitwise the plain
    fine cross-entropy;
  * every ``O`` token, in either space, keeps the plain fine cross-entropy, and
    masked positions stay masked.

The saved checkpoint therefore remains a native fine tagger -- no row is renamed,
reordered or dropped -- so existing fine decode and evaluation tooling applies to
it unchanged, and validation reports both the coarse-space span metrics (fine
decode mapped through the member inverse) and, for any fine-labelled validation
row, its native fine-space metrics.

Long docs are pre-windowed at sentence-friendly char boundaries
(pii_segment_eval.segments, verbatim slices) so tab/meddocan-length
records train on their whole text instead of a 512-token prefix.

Checkpoint selection uses full-draw eval loss by default; ``--selection-metric
span-f1`` instead selects exact typed span F1 when validation supplies hard
product-head labels. The honest product selector remains fresh multilingual
span F1 — run scripts/pii_eval.py predict/score with
PII_EVAL_LOCAL_MODEL=<checkpoint> over the *-fresh sets on the saved
checkpoints afterward; do not report the eval_loss winner unexamined.

Usage:
  pii_encoder_train.py --data untracked/pii-eval/ft/unified-v1 \
      --out untracked/pii-eval/ft/enc-xlmr-v1 \
      [--model FacebookAI/xlm-roberta-large] [--epochs 3] [--batch 32]
      [--lr 2e-5] [--max-chars 900] [--seed 7] [--decoder linear]
      [--head-kind mlp --encoder-layers=-1,-3 --head-rank 256]
      [--head-kind affine --encoder-layers=-1 --token-offsets=-1,0,1]
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import get_scheduler
from transformers.trainer import seed_worker
from transformers.trainer_utils import get_last_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pii_annotation_conventions import (  # noqa: E402
    convention_training_row,
    internal_prefix_masks,
    resolve_annotation_conventions,
)
from pii_annotation_sampling import share_annotation_sampling_mass  # noqa: E402
from pii_bioes import count_bioes_violations, decode_bioes_spans  # noqa: E402
from pii_character_projection import load_character_projection  # noqa: E402
from pii_continuous_character_cnn import project_continuous_characters  # noqa: E402
from pii_continuous_character_model import (  # noqa: E402
    CHARACTER_PROJECTION_FILENAME,
    SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES,
    ContinuousCharacterForTokenClassification,
)
from pii_crf_model import MMBertCrfForTokenClassification  # noqa: E402
from pii_document_context import (  # noqa: E402
    DOCUMENT_START,
    check_context_mixture,
    document_context_parts,
    encode_document_context,
    resolve_context_configurations,
    resolve_context_side,
    resolve_context_weights,
    select_document_context,
)
from pii_domain_model import load_posteriors as load_domain_posteriors  # noqa: E402
from pii_domain_model import text_key as domain_key  # noqa: E402
from pii_dual_head import (  # noqa: E402
    DEFAULT_P20_CUT,
    NEW_LABEL_SPACE,
    TAGSET_PATH,
    bioes_structure_cost_matrix,
    bioes_transition_legality,
    bucketed_incompatibility_matrix,
    bucketed_structure_cost_matrix,
    build_correctness_map,
    coarse_incompatibility_matrix,
    continuation_label_mask,
    dual_head_loss,
    fallback_source_rows,
    head_entity_labels,
    head_loss_terms,
    initialize_affine_classifier,
    load_affine_projection,
    load_bucket_map,
    load_correctness_map,
    mapped_structure_cost_rows,
    new_space_labels,
    parse_transition_schedule,
    semantic_sha256,
    span_risk_gate,
    transition_completion_step,
    transition_weight,
    write_affine_projection_receipt,
)
from pii_head_input_norm import calibration_moments as head_input_calibration_moments  # noqa: E402
from pii_language_policy import (  # noqa: E402
    DEFAULT_LANGUAGE_ROUND_PATH,
    validate_core_language_support,
)
from pii_layered_head_model import (  # noqa: E402
    DEFERRED_REFERENCE_TYPES,
    LayerConcatForTokenClassification,
    resolve_encoder_layers,
    resolve_token_offsets,
)
from pii_native_heads import (  # noqa: E402
    NativeHeadBatchSampler,
    NativeHeadCollator,
    NativeHeadDataset,
    NativeHeadLossMixin,
    load_native_head_config,
    native_head_coverage,
    validate_native_tokenization,
)
from pii_predicate_exposure import assign_predicate_exposure_strata  # noqa: E402
from pii_projector import Tagset  # noqa: E402
from pii_prompt_slots import LANGUAGE_NAMES as PROMPT_LANGUAGE_NAMES  # noqa: E402
from pii_prompt_slots import PromptSlots, layout_slots  # noqa: E402
from pii_prompt_slots import install as install_prompt_slots  # noqa: E402
from pii_prompt_slots import target_bounds as prompt_target_bounds  # noqa: E402
from pii_reference_projection import PROJECTION_FIELD, project_reference_row  # noqa: E402
from pii_seed_tree import DEFAULT_CONFIG_PATH as REPRODUCIBILITY_CONFIG_PATH  # noqa: E402
from pii_seed_tree import SeedTree, load_seed_tree  # noqa: E402
from pii_segment_eval import segments  # noqa: E402
from pii_soft_registers import UNKNOWN_CONDITION, placeholder_id, reserve_positions  # noqa: E402
from pii_soft_registers import install as install_soft_registers  # noqa: E402
from pii_source_classes import DEFAULT_SPEC as DEFAULT_SOURCE_CLASS_SPEC  # noqa: E402
from pii_source_classes import SourceClasses  # noqa: E402
from pii_subclass import (  # noqa: E402
    SubclassSpec,
    load_subclass_spec,
    validate_component_weight,
    validate_sequence_grammar,
)
from pii_surface.rerealize import parse_string_equalities  # noqa: E402
from pii_surface.sample_time import (  # noqa: E402
    SampleTimeSurfaceRealizer,
    SurfaceDrawBatchSampler,
)
from pii_training_quarantine import validate_training_inputs  # noqa: E402

import log_format  # noqa: E402
from trainlib import (  # noqa: E402
    BATCH_FORMATIONS,
    DEFAULT_RECURSIVE_LENGTH_BUCKET_WIDTH,
    DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS,
    DRAW_POLICIES,
    CheckpointMirrorCallback,
    OrderlyCheckpointCallback,
    RecursiveWeightedBatchSampler,
    RunMetricsCallback,
    WeightedLengthBatchSampler,
    add_checkpoint_mirror_args,
    checkpoint_step,
    detach_best_directory,
    example_weights_from_pools,
    is_valid_trainer_checkpoint,
    link_best_to_checkpoint,
    prune_checkpoints_before_selected,
    select_validation_rows,
    thin_optimizer_state,
)

DEFAULT_REGISTER_LANGUAGE_SPEC = Path(__file__).with_name("pii_register_languages_v1.json")


COMPLETE_SUPERVISION = "complete"
ANNOTATED_SPANS_ONLY = "annotated_spans_only"
SUPERVISION_MODES = frozenset({COMPLETE_SUPERVISION, ANNOTATED_SPANS_ONLY})
BOUNDARY_LABELS = ("O", "B", "I", "E", "S")
BOUNDARY2ID = {label: index for index, label in enumerate(BOUNDARY_LABELS)}
MLM_ACTIVE_BATCH_COMPUTE_OVERHEAD_PRIOR = 0.40
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_VICTORY_LAP_LR_SCALE = 1.0 / 15.0
ENTITY_PU_PRIOR_SCHEMA = "pii-entity-pu-class-prior"
ENTITY_PU_PRIOR_SCHEMA_VERSION = 1
ENTITY_TOKEN_PRIOR_SCHEMA = "pii-entity-token-class-prior"
ENTITY_TOKEN_PRIOR_SCHEMA_VERSION = 1
PREDICATE_SPEC_SCHEMA = "pii-token-predicate-channels"
PREDICATE_SPEC_VERSION = 1
ADDED_HEAD_BALANCE_SCHEMA = "pii-ont3-added-head-balance"
ADDED_HEAD_BALANCE_VERSION = 1
PREDICATE_TOKEN_RULE = (
    "A token is positive iff it overlaps a positive character extent. Other tokens "
    "inside the same explicitly labeled activating span are known negative for that "
    "declared channel. Every undeclared channel and every token outside an activating "
    "span is unknown."
)
LEGACY_PREDICATE_TOKEN_RULE = (
    "A token is positive iff it overlaps a positive character extent. Other tokens "
    "inside the same explicitly labeled activating span are known negative. Every token "
    "outside such a span is unknown."
)
DEFAULT_PREDICATE_SPEC_PATH = Path(HERE) / "pii_predicate_channels_v1.json"
DEFAULT_SUBCLASS_SPEC_PATH = Path(HERE) / "pii_subclass_families_v2.json"
SUBCLASS_SCOPE_IDS = {"full_primary_span": 0, "component_span": 1}

# Which vocabulary a row's span labels are in. A corpus that says nothing is in
# the head's own fine vocabulary, which is every corpus built before union-head
# training. The producer is scripts/pii_v2_training_adapter.py --union-labels;
# its member-document names are restated here rather than imported, because that
# module is stdlib-only corpus work and this one is the torch side.
FINE_LABEL_SPACE = "v1"
UNION_LABEL_SPACE = "v2"
LABEL_SPACES = frozenset({FINE_LABEL_SPACE, UNION_LABEL_SPACE})
UNION_MEMBERS_SCHEMA = "pii-ontology-v2-union-head-members"
UNION_MEMBERS_SCHEMA_VERSION = 1
UNION_OUTSIDE_LABEL = "O"
DEFAULT_UNION_FINE_WEIGHT_SCHEDULE = "constant:1.0"
CONSTANT_UNION_FINE_WEIGHT = ("constant", 1.0, 1.0)


@dataclass(frozen=True)
class PredicateSpec:
    channels: tuple[str, ...]
    applicable_types: dict[str, frozenset[str]]
    sha256: str
    path: Path


@dataclass(frozen=True)
class AddedHeadBalanceSpec:
    reference_positive_weights: dict[str, float]
    predicate_positive_weights: dict[str, dict[str, float]]
    sha256: str
    path: Path
    document: dict[str, object]


@dataclass(frozen=True)
class LogicalStepObjectiveMasses:
    """Applied supervision mass shared by one optimizer step's microbatches."""

    totals: dict[str, torch.Tensor]
    physical_batches: int
    primary_group_weight_sum: torch.Tensor
    data_weight_sum: torch.Tensor


def mapped_head_objective_mass(
    own_labels: torch.Tensor,
    cross_labels: torch.Tensor,
    *,
    own_o_label_id: int,
    cross_o_label_id: int,
    o_token_weight: float = 1.0,
    token_objective_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact denominator used by one mapped token-classification head."""
    if own_labels.shape != cross_labels.shape:
        raise ValueError("mapped objective labels must share one shape")
    if token_objective_weights is None:
        weights = torch.ones_like(own_labels, dtype=torch.float)
    else:
        if token_objective_weights.shape != own_labels.shape:
            raise ValueError("primary objective weights must match mapped labels")
        weights = token_objective_weights.float()
    direct = own_labels != -100
    mapped = (cross_labels != -100) & ~direct
    active = (direct | mapped) & (weights > 0)
    if not torch.any(active):
        return weights.new_zeros(())
    is_o = (direct & (own_labels == own_o_label_id)) | (mapped & (cross_labels == cross_o_label_id))
    applied = torch.where(is_o, weights * o_token_weight, weights)
    return applied[active].sum()


def logical_step_mean_contribution(
    mean_loss: torch.Tensor,
    local_mass: float | torch.Tensor,
    step_masses: LogicalStepObjectiveMasses | None,
    component: str,
) -> torch.Tensor:
    """Scale a physical-batch mean into its logical-step weighted-mean contribution."""
    if step_masses is None:
        return mean_loss
    local = torch.as_tensor(local_mass, device=mean_loss.device, dtype=mean_loss.dtype)
    total = step_masses.totals.get(component)
    if not torch.any(local > 0):
        return mean_loss * 0.0
    if total is None or not torch.any(total > 0):
        raise ValueError(f"logical-step component {component!r} has local but no total mass")
    return mean_loss * (local / total.to(device=mean_loss.device, dtype=mean_loss.dtype))


def load_predicate_spec(path: Path) -> PredicateSpec:
    """Load the closed token-predicate contract used by data, head, and loss."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "version",
        "channels",
        "target_encoding",
    }:
        raise ValueError("predicate spec must contain schema, version, channels, and target_encoding")
    if payload["schema"] != PREDICATE_SPEC_SCHEMA or payload["version"] != PREDICATE_SPEC_VERSION:
        raise ValueError(f"unsupported predicate spec {payload['schema']!r} version {payload['version']!r}")
    target = payload["target_encoding"]
    expected_target = {
        "positive": 1,
        "known_negative": 0,
        "unknown": -100,
    }
    if (
        not isinstance(target, dict)
        or {key: value for key, value in target.items() if key != "token_rule"} != expected_target
        or target.get("token_rule") not in {PREDICATE_TOKEN_RULE, LEGACY_PREDICATE_TOKEN_RULE}
    ):
        raise ValueError("predicate target_encoding does not match the masked token contract")
    entries = payload["channels"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("predicate spec channels must be a nonempty list")
    channels = []
    applicable_types = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "applicable_types",
            "definition",
        }:
            raise ValueError(f"predicate channel {index} has an invalid shape")
        name = entry["name"]
        types = entry["applicable_types"]
        definition = entry["definition"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"predicate channel {index} needs a nonempty name")
        if name in applicable_types:
            raise ValueError(f"duplicate predicate channel {name!r}")
        if (
            not isinstance(types, list)
            or not types
            or any(not isinstance(value, str) or not value for value in types)
            or len(set(types)) != len(types)
        ):
            raise ValueError(f"predicate channel {name!r} needs unique applicable types")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"predicate channel {name!r} needs a definition")
        channels.append(name)
        applicable_types[name] = frozenset(types)
    return PredicateSpec(
        channels=tuple(channels),
        applicable_types=applicable_types,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        path=path,
    )


def load_added_head_balance_spec(
    path: Path,
    predicate_spec: PredicateSpec,
    condition_types: tuple[str, ...],
) -> AddedHeadBalanceSpec:
    """Load frozen added-head positive multipliers derived without proof rows."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "version",
        "status",
        "source_diagnostic",
        "derivation",
        "reference_positive_weights",
        "predicate_positive_weights",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"{path}: added-head balance spec has an invalid shape")
    if payload["schema"] != ADDED_HEAD_BALANCE_SCHEMA or payload["version"] != ADDED_HEAD_BALANCE_VERSION:
        raise ValueError(f"{path}: unsupported added-head balance spec")

    def checked_weights(values: object, expected: set[str], where: str) -> dict[str, float]:
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"{path}: {where} must name exactly {sorted(expected)}")
        result = {}
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 1.0
            ):
                raise ValueError(f"{path}: {where}.{name} must be finite and at least 1")
            result[name] = float(value)
        return result

    references = checked_weights(
        payload["reference_positive_weights"],
        {"organization_reference", "person_reference"},
        "reference_positive_weights",
    )
    predicate_payload = payload["predicate_positive_weights"]
    if not isinstance(predicate_payload, dict) or set(predicate_payload) != set(condition_types):
        raise ValueError(f"{path}: predicate_positive_weights must name exactly {sorted(condition_types)}")
    predicates = {
        condition: checked_weights(
            predicate_payload[condition],
            set(predicate_spec.channels),
            f"predicate_positive_weights.{condition}",
        )
        for condition in condition_types
    }
    return AddedHeadBalanceSpec(
        reference_positive_weights=references,
        predicate_positive_weights=predicates,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        path=path,
        document=payload,
    )


def resolve_training_seeds(
    legacy_seed, reproducibility_config, reproducibility_seed=None, validation_seed=None
):
    """Return model/data/replay seeds while preserving explicit legacy runs.

    ``validation_seed`` detaches which rows are validated from everything else a seed
    controls. Welded to the run seed, two seeds of the same recipe validate on different
    subsets whenever the validation view is capped, so their curves measure different
    tests and a warm-started leg can appear to regress purely by changing what it is
    scored on. Held fixed, a seed changes the model and the data order and leaves the
    yardstick alone.
    """
    if legacy_seed is not None:
        if reproducibility_seed is not None:
            raise ValueError("legacy --seed cannot be combined with --reproducibility-seed")
        return {
            "mode": "legacy-single-seed",
            "model": legacy_seed,
            "sampling": legacy_seed,
            "replay": legacy_seed,
            "validation": legacy_seed if validation_seed is None else validation_seed,
            "validation_pinned": validation_seed is not None,
        }
    tree = load_seed_tree(reproducibility_config, root_seed=reproducibility_seed)
    return {
        "mode": tree.scheme,
        "root": tree.root_seed,
        "model": tree.fork("training", "model"),
        "sampling": tree.fork("training", "sampling"),
        "replay": tree.fork("training", "replay"),
        "validation": (tree.fork("training", "validation") if validation_seed is None else validation_seed),
        "validation_pinned": validation_seed is not None,
    }


def bioes(category, n):
    if n == 1:
        return [f"S-{category}"]
    return [f"B-{category}"] + [f"I-{category}"] * (n - 2) + [f"E-{category}"]


def boundary_label_id(label: str) -> int:
    """Project one fine BIOES label into the class-agnostic boundary vocabulary."""
    return BOUNDARY2ID[label if label == "O" else label.split("-", 1)[0]]


def collapse_bioes_logits(logits: torch.Tensor, label_names: list[str]) -> torch.Tensor:
    """Log-sum fine-type logits into exact O/B/I/E/S probability masses."""
    groups = []
    for boundary in BOUNDARY_LABELS:
        indices = [
            index
            for index, label in enumerate(label_names)
            if (label == "O" and boundary == "O") or label.startswith(f"{boundary}-")
        ]
        if not indices:
            raise ValueError(f"fine label vocabulary has no {boundary!r} boundary labels")
        groups.append(torch.logsumexp(logits[..., indices].float(), dim=-1))
    return torch.stack(groups, dim=-1)


def build_coarse_cut_groups(label_names, cut, boundary_free=False):
    """Column groups + target remap projecting fine BIOES labels onto a
    reporting cut's (boundary x bucket) vocabulary (bucket-only when
    boundary_free).

    The decision metric is redaction-20 (fine tags are diagnostic), so
    the objective can reward coarse-bucket agreement directly: the
    bucket probability is the *sum* of member fine probabilities, i.e.
    a logsumexp over member logits — the "sum of softmax" OR, which a
    single affine bucket logit cannot express."""
    tagset = Tagset()
    keys = {}
    group_of = []
    for label in label_names:
        if label == "O":
            key = "O"
        else:
            boundary, category = label.split("-", 1)
            bucket = tagset.project_canonical_cut(category, cut)
            # boundary_free gives span-membership partial credit: a token
            # predicted I-date_of_birth against gold B-date still lands in
            # the right bucket group.
            key = bucket if boundary_free else f"{boundary}-{bucket}"
        group_of.append(keys.setdefault(key, len(keys)))
    index_groups = [[] for _ in keys]
    for column, gid in enumerate(group_of):
        index_groups[gid].append(column)
    return index_groups, torch.tensor(group_of)


def coarse_agreement_loss(logits, labels, index_groups, remap, *, normalize_groups=False):
    """CE on cut-marginal probabilities, optionally with uniform component priors."""
    valid = labels != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    grouped_logits = []
    for indices in index_groups:
        score = torch.logsumexp(logits[..., indices].float(), dim=-1)
        if normalize_groups:
            score = score - math.log(len(indices))
        grouped_logits.append(score)
    grouped = torch.stack(grouped_logits, dim=-1)
    remap = remap.to(labels.device)
    coarse_labels = torch.where(valid, remap[labels.clamp(min=0)], labels)
    return F.cross_entropy(grouped.view(-1, grouped.shape[-1]), coarse_labels.view(-1), ignore_index=-100)


class CoarseAgreementMixin:
    """Add cut-marginal agreement terms to any trainer's loss.

    `coarse_agreement` is a list of (weight, index_groups, remap)
    triples set on the instance after construction; empty = no-op."""

    coarse_agreement = ()
    fine_label_loss_weight = 1.0
    normalize_coarse_groups = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.coarse_agreement or (not model.training and self.fine_label_loss_weight == 1.0):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        labels = inputs["labels"]
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        loss = self.fine_label_loss_weight * loss
        for weight, index_groups, remap in self.coarse_agreement:
            loss = loss + weight * coarse_agreement_loss(
                outputs.logits,
                labels,
                index_groups,
                remap,
                normalize_groups=self.normalize_coarse_groups,
            )
        # Normalize by total term weight so adding agreement terms does
        # not scale the objective (an implicit ~sum-of-weights LR
        # increase, since coarse gradients correlate with fine CE's).
        loss = loss / (self.fine_label_loss_weight + sum(weight for weight, _, _ in self.coarse_agreement))
        return (loss, outputs) if return_outputs else loss


def annotated_boundary_loss(
    logits: torch.Tensor,
    boundary_labels: torch.Tensor,
    label_names: list[str],
) -> torch.Tensor:
    """Cross-entropy over collapsed boundaries, ignoring non-partial tokens."""
    valid = boundary_labels != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    collapsed = collapse_bioes_logits(logits, label_names)
    return F.cross_entropy(collapsed[valid], boundary_labels[valid])


def partial_boundary_retention_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    consistency_mask: torch.Tensor,
    label_names: list[str],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill parent BIOES probabilities only where partial rows have no gold."""
    valid = consistency_mask.bool()
    if not torch.any(valid):
        return student_logits.sum() * 0.0
    student = collapse_bioes_logits(student_logits, label_names)[valid] / temperature
    teacher = collapse_bioes_logits(teacher_logits, label_names)[valid] / temperature
    return (
        F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher.detach(), dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def partial_o_loss(
    logits: torch.Tensor,
    consistency_mask: torch.Tensor,
    o_label_id: int,
) -> torch.Tensor:
    """Apply O cross-entropy only outside spans on partial-label rows."""
    valid = consistency_mask.bool()
    if not torch.any(valid):
        return logits.sum() * 0.0
    selected = logits[valid]
    targets = torch.full(
        (selected.shape[0],),
        o_label_id,
        dtype=torch.long,
        device=selected.device,
    )
    return F.cross_entropy(selected, targets)


def entity_presence_loss(
    logits: torch.Tensor,
    presence_labels: torch.Tensor,
    o_label_id: int,
) -> torch.Tensor:
    """Project fine logits to entity-vs-O CE on trustworthy complete rows."""
    valid = presence_labels != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    if not 0 <= o_label_id < logits.shape[-1]:
        raise ValueError(f"O label id {o_label_id} is outside {logits.shape[-1]} logits")
    entity_indices = [index for index in range(logits.shape[-1]) if index != o_label_id]
    if not entity_indices:
        raise ValueError("entity-presence loss requires at least one non-O output")
    selected = logits[valid].float()
    projected = torch.stack(
        (
            selected[:, o_label_id],
            torch.logsumexp(selected[:, entity_indices], dim=-1),
        ),
        dim=-1,
    )
    return F.cross_entropy(projected, presence_labels[valid])


def entity_presence_log_odds(logits: torch.Tensor, o_label_id: int) -> torch.Tensor:
    """Return log p(entity) - log p(O) after pooling every non-O row."""
    if not 0 <= o_label_id < logits.shape[-1]:
        raise ValueError(f"O label id {o_label_id} is outside {logits.shape[-1]} logits")
    entity_indices = [index for index in range(logits.shape[-1]) if index != o_label_id]
    if not entity_indices:
        raise ValueError("entity presence requires at least one non-O output")
    values = logits.float()
    return torch.logsumexp(values[..., entity_indices], dim=-1) - values[..., o_label_id]


@dataclass(frozen=True)
class NonnegativePURisk:
    """Binary positive-unlabeled risk and its auditable components."""

    loss: torch.Tensor
    positive_risk: torch.Tensor
    negative_risk: torch.Tensor
    unclipped_negative_risk: torch.Tensor
    positive_tokens: int
    unlabeled_tokens: int
    groups: int
    clipped_groups: int


@dataclass(frozen=True)
class ExpectedEntityRatioLoss:
    """Batch entity-rate hinge and the quantities needed to audit it."""

    loss: torch.Tensor
    predicted_ratio: torch.Tensor
    lower_ratio: torch.Tensor
    upper_ratio: torch.Tensor
    tokens: int
    rows: int


def expected_entity_ratio_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    group_ids: torch.Tensor,
    class_priors: torch.Tensor,
    o_label_id: int,
    *,
    lower_width: float,
) -> ExpectedEntityRatioLoss:
    """Constrain partial rows' aggregate entity probability without token negatives.

    Every real token on an ``annotated_spans_only`` row participates: known
    entity tokens come from ``positive_mask`` and the remaining latent tokens
    come from ``unlabeled_mask``. Complete rows carry group ``-1`` and no mask.
    The per-row trustworthy-complete prior is token-mass averaged so the
    interval and prediction use exactly the same denominator.
    """
    if positive_mask.shape != logits.shape[:-1] or unlabeled_mask.shape != logits.shape[:-1]:
        raise ValueError("entity-ratio positive and unlabeled masks must match the logit token shape")
    if group_ids.ndim != 1 or class_priors.ndim != 1 or group_ids.shape != class_priors.shape:
        raise ValueError("entity-ratio group ids and class priors must be aligned per-row vectors")
    if group_ids.shape[0] != logits.shape[0]:
        raise ValueError("entity-ratio row metadata must align with the logit batch")
    if not math.isfinite(lower_width) or not 0 <= lower_width < 1:
        raise ValueError("entity-ratio lower width must be finite and in [0, 1)")
    positive = positive_mask.bool()
    unlabeled = unlabeled_mask.bool()
    if torch.any(positive & unlabeled):
        raise ValueError("entity-ratio positive and unlabeled masks overlap")
    real_tokens = positive | unlabeled
    active_rows = group_ids >= 0
    if torch.any(real_tokens & ~active_rows.unsqueeze(1)):
        raise ValueError("entity-ratio token masks are active on a row without a prior group")
    if torch.any(active_rows & ~real_tokens.any(dim=1)):
        raise ValueError("entity-ratio prior group is active on a row without real tokens")
    active_priors = class_priors[active_rows].float()
    if active_priors.numel() and (
        not torch.all(torch.isfinite(active_priors))
        or not torch.all((active_priors > 0) & (active_priors < 1))
    ):
        raise ValueError("entity-ratio class priors must be finite and strictly in (0, 1)")
    for group_id in sorted(set(group_ids[active_rows].detach().cpu().tolist())):
        group_priors = class_priors[group_ids == group_id].float()
        if not torch.allclose(group_priors, group_priors[0].expand_as(group_priors)):
            raise ValueError(f"entity-ratio group {group_id} carries inconsistent class priors")

    token_count = int(torch.count_nonzero(real_tokens).item())
    zero = logits.sum() * 0.0
    if not token_count:
        return ExpectedEntityRatioLoss(zero, zero, zero, zero, 0, 0)
    if not 0 <= o_label_id < logits.shape[-1]:
        raise ValueError(f"O label id {o_label_id} is outside {logits.shape[-1]} logits")

    entity_probability = 1.0 - logits.float().softmax(dim=-1)[..., o_label_id]
    token_priors = class_priors.float().unsqueeze(1).expand_as(entity_probability)
    predicted_ratio = entity_probability[real_tokens].mean()
    upper_ratio = token_priors[real_tokens].mean()
    lower_ratio = torch.clamp_min(token_priors - lower_width, 0.0)[real_tokens].mean()
    loss = F.relu(lower_ratio - predicted_ratio) + F.relu(predicted_ratio - upper_ratio)
    return ExpectedEntityRatioLoss(
        loss=loss,
        predicted_ratio=predicted_ratio,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
        tokens=token_count,
        rows=int(torch.count_nonzero(active_rows).item()),
    )


def nonnegative_pu_entity_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    group_ids: torch.Tensor,
    class_priors: torch.Tensor,
    o_label_id: int,
    *,
    positive_margin: float | None = None,
) -> NonnegativePURisk:
    """Estimate stratified non-negative PU risk on binary entity presence.

    Each nonnegative ``group_ids`` value denotes one source/language stratum.
    Its class prior must be constant across rows in the batch. Logistic loss is
    the default. ``positive_margin`` replaces only the positive-class logistic
    term with a hinge that has zero gradient after the requested entity-vs-O
    log-odds margin; the unlabeled negative-risk correction remains logistic.
    """
    if positive_mask.shape != logits.shape[:-1] or unlabeled_mask.shape != logits.shape[:-1]:
        raise ValueError("PU positive and unlabeled masks must match the logit token shape")
    if group_ids.ndim != 1 or class_priors.ndim != 1 or group_ids.shape != class_priors.shape:
        raise ValueError("PU group ids and class priors must be aligned per-row vectors")
    if group_ids.shape[0] != logits.shape[0]:
        raise ValueError("PU row metadata must align with the logit batch")
    if positive_margin is not None and (not math.isfinite(positive_margin) or positive_margin < 0):
        raise ValueError("PU positive margin must be finite and nonnegative")
    positive = positive_mask.bool()
    unlabeled = unlabeled_mask.bool()
    if torch.any(positive & unlabeled):
        raise ValueError("PU positive and unlabeled masks overlap")
    active_rows = group_ids >= 0
    if torch.any((positive | unlabeled) & ~active_rows.unsqueeze(1)):
        raise ValueError("PU token masks are active on a row without a source/language group")
    active_groups = sorted(set(group_ids[active_rows].detach().cpu().tolist()))
    zero = logits.sum() * 0.0
    if not active_groups:
        return NonnegativePURisk(zero, zero, zero, zero, 0, 0, 0, 0)

    log_odds = entity_presence_log_odds(logits, o_label_id)
    group_losses = []
    group_positive_risks = []
    group_negative_risks = []
    group_unclipped_negative_risks = []
    group_token_counts = []
    positive_tokens = 0
    unlabeled_tokens = 0
    clipped_groups = 0
    for group_id in active_groups:
        rows = group_ids == group_id
        priors = class_priors[rows].float()
        if not torch.all(torch.isfinite(priors)) or not torch.all((priors > 0) & (priors < 1)):
            raise ValueError(f"PU group {group_id} class prior must be finite and strictly in (0, 1)")
        if not torch.allclose(priors, priors[0].expand_as(priors)):
            raise ValueError(f"PU group {group_id} carries inconsistent class priors")
        prior = priors[0]
        group_positive = positive & rows.unsqueeze(1)
        group_unlabeled = unlabeled & rows.unsqueeze(1)
        positive_count = int(torch.count_nonzero(group_positive).item())
        unlabeled_count = int(torch.count_nonzero(group_unlabeled).item())
        if not positive_count or not unlabeled_count:
            raise ValueError(
                f"PU group {group_id} needs both positive and unlabeled tokens in each physical batch"
            )
        positive_scores = log_odds[group_positive]
        unlabeled_scores = log_odds[group_unlabeled]
        positive_loss = (
            F.softplus(-positive_scores)
            if positive_margin is None
            else F.relu(positive_margin - positive_scores)
        )
        negative_on_positive = F.softplus(positive_scores)
        negative_on_unlabeled = F.softplus(unlabeled_scores)
        positive_risk = prior * positive_loss.mean()
        negative_risk_unclipped = negative_on_unlabeled.mean() - prior * negative_on_positive.mean()
        negative_risk = torch.clamp_min(negative_risk_unclipped, 0.0)
        group_losses.append(positive_risk + negative_risk)
        group_positive_risks.append(positive_risk)
        group_negative_risks.append(negative_risk)
        group_unclipped_negative_risks.append(negative_risk_unclipped)
        group_token_counts.append(positive_count + unlabeled_count)
        positive_tokens += positive_count
        unlabeled_tokens += unlabeled_count
        clipped_groups += int(float(negative_risk_unclipped.detach()) < 0.0)

    weights = logits.new_tensor(group_token_counts, dtype=torch.float32)
    weights = weights / weights.sum()

    def weighted(values):
        return torch.sum(weights * torch.stack(values))

    return NonnegativePURisk(
        loss=weighted(group_losses),
        positive_risk=weighted(group_positive_risks),
        negative_risk=weighted(group_negative_risks),
        unclipped_negative_risk=weighted(group_unclipped_negative_risks),
        positive_tokens=positive_tokens,
        unlabeled_tokens=unlabeled_tokens,
        groups=len(active_groups),
        clipped_groups=clipped_groups,
    )


def parent_presence_kl_loss(
    student_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    unlabeled_mask: torch.Tensor,
    o_label_id: int,
) -> torch.Tensor:
    """Teacher-to-student Bernoulli entity-presence KL on unlabeled tokens."""
    valid = unlabeled_mask.bool()
    if not torch.any(valid):
        return student_logits.sum() * 0.0
    student_odds = entity_presence_log_odds(student_logits, o_label_id)
    parent_odds = entity_presence_log_odds(parent_logits, o_label_id)
    student_binary = torch.stack((torch.zeros_like(student_odds), student_odds), dim=-1)
    parent_binary = torch.stack((torch.zeros_like(parent_odds), parent_odds), dim=-1)
    divergence = F.kl_div(
        F.log_softmax(student_binary[valid], dim=-1),
        F.softmax(parent_binary[valid].detach(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return divergence.mean()


def family_presence_loss(
    logits: torch.Tensor,
    family_labels: torch.Tensor,
    family_label_groups: list[list[int]],
) -> torch.Tensor:
    """Project fine logits to O-vs-coarse-family CE on trustworthy complete rows."""
    valid = family_labels != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    if not family_label_groups or any(not group for group in family_label_groups):
        raise ValueError("family-presence loss requires nonempty label groups")
    selected = logits[valid].float()
    projected = torch.stack(
        [torch.logsumexp(selected[:, group], dim=-1) for group in family_label_groups],
        dim=-1,
    )
    return F.cross_entropy(projected, family_labels[valid])


def build_family_presence_projection(correctness_map):
    """Coarse-family projection tables for the family-presence objective.

    Returns ``(family_label_groups, own_targets, secondary_targets)``. Groups
    index the new-head logits: group 0 is the ``O`` row and each later group
    holds one ontology family's BIOES rows. ``secondary_targets`` maps each
    new-space label id to its family target. ``own_targets`` maps each
    old-space label id through its accepted new set: a single covered family
    supervises that family, while a multi-family accepted set is masked
    (``-100``) because the old label does not identify one coarse target.
    """
    from pii_ontology_v2 import load_ontology  # noqa: PLC0415 — avoid mandatory spec load

    ontology = load_ontology()
    family_names = sorted(ontology.families)
    family_index = {name: 1 + position for position, name in enumerate(family_names)}
    new_outside = correctness_map.new_outside_id
    groups: list[list[int]] = [[new_outside]] + [[] for _ in family_names]
    secondary_targets: dict[int, int] = {}
    new_family_by_id: dict[int, int] = {}
    for new_id, name in enumerate(correctness_map.new_labels):
        if name == "O":
            continue
        _prefix, _, primary_type = name.partition("-")
        family = family_index[ontology.family_of(primary_type)]
        groups[family].append(new_id)
        secondary_targets[new_id] = family
        new_family_by_id[new_id] = family
    empty = [family_names[position - 1] for position in range(1, len(groups)) if not groups[position]]
    if empty:
        raise ValueError(f"families with no new-head rows: {', '.join(empty)}")
    own_targets: dict[int, int] = {}
    for old_id in range(len(correctness_map.old_labels)):
        allowed = correctness_map.old_to_new[old_id]
        families = {
            new_family_by_id[new_id]
            for new_id in torch.nonzero(allowed, as_tuple=True)[0].tolist()
            if new_id != new_outside
        }
        if len(families) == 1:
            own_targets[old_id] = families.pop()
    return groups, own_targets, secondary_targets


def partial_negative_loss(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    prohibited_label_groups: list[list[int]],
) -> torch.Tensor:
    """Penalize source-covered types outside spans while allowing every other label."""
    total_loss = logits.sum() * 0.0
    total_tokens = 0
    all_indices = set(range(logits.shape[-1]))
    for group_id, prohibited_indices in enumerate(prohibited_label_groups):
        valid = group_ids == group_id
        if not torch.any(valid):
            continue
        prohibited = sorted(set(prohibited_indices))
        allowed = sorted(all_indices - set(prohibited))
        if not prohibited or not allowed:
            raise ValueError("partial-negative groups require both prohibited and allowed labels")
        selected = logits[valid].float()
        binary_logits = torch.stack(
            (
                torch.logsumexp(selected[:, prohibited], dim=-1),
                torch.logsumexp(selected[:, allowed], dim=-1),
            ),
            dim=-1,
        )
        count = binary_logits.shape[0]
        total_loss = total_loss + F.cross_entropy(
            binary_logits,
            torch.ones(count, dtype=torch.long, device=binary_logits.device),
            reduction="sum",
        )
        total_tokens += count
    return total_loss / total_tokens if total_tokens else logits.sum() * 0.0


def token_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reconstruct the stock token CE directly from the returned logits."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def o_weighted_token_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    o_label_id: int,
    o_token_weight: float,
) -> torch.Tensor:
    """Token CE with one relative weight for trusted O targets."""
    class_weights = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    class_weights[o_label_id] = o_token_weight
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=class_weights,
    )


def per_row_o_weighted_token_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    o_label_id: int,
    row_o_weights: torch.Tensor,
) -> torch.Tensor:
    """Token CE where each row supplies its own relative weight for O targets.

    A physical batch mixes intake methods, and each method's O labels are
    trustworthy to a different degree, so the weight has to vary per slot within
    the batch. ``F.cross_entropy(weight=...)`` cannot express that: its class
    weight is shared by every row. This computes unreduced token losses, scales
    the O positions by their own row's weight, and renormalizes by the weight
    actually applied so the result stays a weighted mean rather than shrinking
    as weights fall.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    row = row_o_weights.reshape(-1, 1).expand(labels.shape).reshape(-1).to(per_token.dtype)
    is_o = flat_labels == o_label_id
    weights = torch.where(is_o, row, torch.ones_like(per_token))
    # ignored positions contribute neither loss nor normalizer
    weights = weights * (flat_labels != -100).to(per_token.dtype)
    return (per_token * weights).sum() / weights.sum().clamp_min(1e-8)


def masked_predicate_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted binary cross-entropy over explicitly known token/channel cells."""
    if logits.shape != labels.shape:
        raise ValueError(
            f"predicate logits and labels must have the same shape, got {logits.shape} and {labels.shape}"
        )
    if weights is None:
        weights = (labels != -100).to(dtype=logits.dtype)
    if weights.shape != labels.shape:
        raise ValueError(
            f"predicate weights and labels must have the same shape, got {weights.shape} and {labels.shape}"
        )
    if not torch.all(torch.isfinite(weights)) or torch.any(weights < 0):
        raise ValueError("predicate weights must be finite and nonnegative")
    valid = labels != -100
    if torch.any((~valid) & (weights != 0)):
        raise ValueError("unknown predicate targets must have zero objective weight")
    known_targets = labels[valid]
    if torch.any((known_targets != 0) & (known_targets != 1)):
        raise ValueError("known predicate targets must be binary")
    active = valid & (weights > 0)
    if not torch.any(active):
        return logits.sum() * 0.0
    targets = labels[active]
    cell_weights = weights.float()[active]
    if positive_weights is not None:
        if positive_weights.shape != labels.shape:
            raise ValueError("predicate positive weights must align with labels")
        if not torch.all(torch.isfinite(positive_weights)) or torch.any(positive_weights < 1):
            raise ValueError("predicate positive weights must be finite and at least 1")
        selected_positive_weights = positive_weights.float()[active]
        cell_weights = cell_weights * torch.where(
            targets == 1,
            selected_positive_weights,
            torch.ones_like(selected_positive_weights),
        )
    per_cell = F.binary_cross_entropy_with_logits(
        logits.float()[active],
        targets.float(),
        reduction="none",
    )
    return (per_cell * cell_weights).sum() / cell_weights.sum()


def token_channel_positive_weights(
    labels: torch.Tensor,
    channel_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    """Broadcast one frozen positive learning multiplier per binary channel."""
    if channel_weights is None:
        return None
    if channel_weights.ndim != 1 or channel_weights.shape[0] != labels.shape[-1]:
        raise ValueError("binary positive weight vector does not align with channels")
    return (
        channel_weights.to(device=labels.device, dtype=torch.float)
        .view(*((1,) * (labels.ndim - 1)), -1)
        .expand_as(labels)
    )


def conditioned_predicate_positive_weights(
    predicate_labels: torch.Tensor,
    predicate_condition_ids: torch.Tensor | None,
    condition_channel_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    """Select one frozen positive multiplier row per gold carrier type."""
    if condition_channel_weights is None:
        return None
    if predicate_condition_ids is None:
        raise ValueError("conditioned predicate positive weights require condition ids")
    if predicate_condition_ids.shape != predicate_labels.shape[:2]:
        raise ValueError("predicate condition ids do not align with labels")
    if (
        condition_channel_weights.ndim != 2
        or condition_channel_weights.shape[1] != predicate_labels.shape[-1]
    ):
        raise ValueError("predicate positive weight matrix does not align with channels")
    supervised = torch.any(predicate_labels != -100, dim=-1)
    invalid = (predicate_condition_ids < 0) | (predicate_condition_ids >= condition_channel_weights.shape[0])
    if torch.any(supervised & invalid):
        raise ValueError("supervised predicate cell has no positive-weight condition row")
    safe_ids = predicate_condition_ids.clamp(min=0, max=condition_channel_weights.shape[0] - 1)
    selected = condition_channel_weights.to(
        device=predicate_labels.device,
        dtype=torch.float,
    )[safe_ids]
    return torch.where(invalid.unsqueeze(-1), torch.ones_like(selected), selected)


def select_conditioned_predicate_logits(
    logits: torch.Tensor,
    condition_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one semantic-primary-type predicate block for every token."""
    if logits.ndim != 4:
        raise ValueError(
            f"conditioned predicate logits must have shape [batch, token, type, channel], got {logits.shape}"
        )
    if condition_ids.shape != logits.shape[:2]:
        raise ValueError(
            "predicate condition ids must align with batch and token axes, "
            f"got {condition_ids.shape} and {logits.shape}"
        )
    active = torch.ones_like(condition_ids, dtype=torch.bool)
    if weights is not None:
        if weights.shape != (*logits.shape[:2], logits.shape[-1]):
            raise ValueError("predicate weights do not align with conditioned logits")
        active = torch.any(weights > 0, dim=-1)
    invalid = (condition_ids < 0) | (condition_ids >= logits.shape[2])
    if torch.any(active & invalid):
        raise ValueError("supervised predicate tokens require an in-range gold condition id")
    gather_ids = condition_ids.clamp(min=0, max=logits.shape[2] - 1)
    gather_ids = gather_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, logits.shape[-1])
    return logits.gather(2, gather_ids).squeeze(2)


def masked_subclass_categorical_loss(
    logits: torch.Tensor,
    block_ids: torch.Tensor,
    target_ids: torch.Tensor,
    scope_ids: torch.Tensor,
    token_masks: torch.Tensor,
    objective_weights: torch.Tensor,
    learning_weights: torch.Tensor,
    blocks: list[dict],
) -> torch.Tensor:
    """One weighted categorical decision per declared span component."""
    if logits.ndim != 3:
        raise ValueError(f"subclass logits must have shape [batch, token, row], got {logits.shape}")
    component_shape = block_ids.shape
    if any(
        tensor.shape != component_shape
        for tensor in (target_ids, scope_ids, objective_weights, learning_weights)
    ):
        raise ValueError("subclass component fields must have one shared [batch, component] shape")
    if token_masks.shape != (*component_shape, logits.shape[1]):
        raise ValueError("subclass token masks must align with batch, component, and token axes")
    if component_shape[0] != logits.shape[0]:
        raise ValueError("subclass components and logits must have the same batch size")
    if not torch.all(torch.isfinite(objective_weights)) or not torch.all(torch.isfinite(learning_weights)):
        raise ValueError("subclass weights must be finite")
    if torch.any(objective_weights < 0) or torch.any(learning_weights < 0):
        raise ValueError("subclass weights must be nonnegative")
    record_active = block_ids >= 0
    if torch.any(~record_active & (target_ids != -1)) or torch.any(~record_active & (scope_ids != -1)):
        raise ValueError("padded subclass components must use -1 ids")
    if torch.any(~record_active & (objective_weights != 0)) or torch.any(
        ~record_active & (learning_weights != 0)
    ):
        raise ValueError("padded subclass components must have zero weights")
    if torch.any(~record_active.unsqueeze(-1) & token_masks.bool()):
        raise ValueError("padded subclass components cannot select tokens")

    effective_weights = objective_weights * learning_weights
    weighted_loss = logits.sum() * 0.0
    normalizer = logits.new_zeros(())
    for batch_index, component_index in torch.nonzero(record_active, as_tuple=False).tolist():
        block_id = int(block_ids[batch_index, component_index])
        if not 0 <= block_id < len(blocks):
            raise ValueError(f"subclass block id {block_id} is out of range")
        block = blocks[block_id]
        start = int(block["start"])
        width = int(block["width"])
        target = int(target_ids[batch_index, component_index])
        scope = int(scope_ids[batch_index, component_index])
        if not 0 <= target < width:
            raise ValueError(f"subclass target {target} is outside block width {width}")
        if scope not in SUBCLASS_SCOPE_IDS.values():
            raise ValueError(f"subclass scope id {scope} is unsupported")
        selected = token_masks[batch_index, component_index].bool()
        if not torch.any(selected):
            raise ValueError("an active subclass component must select at least one token")
        component_logits = logits[batch_index, selected, start : start + width].float()
        if component_logits.shape[-1] != width:
            raise ValueError("subclass block escapes the model logit inventory")
        if scope == SUBCLASS_SCOPE_IDS["full_primary_span"]:
            component_loss = F.cross_entropy(
                component_logits.mean(dim=0, keepdim=True),
                torch.tensor([target], device=logits.device),
            )
        else:
            component_loss = F.cross_entropy(
                component_logits,
                torch.full(
                    (component_logits.shape[0],),
                    target,
                    dtype=torch.long,
                    device=logits.device,
                ),
            )
        weight = effective_weights[batch_index, component_index].to(component_loss.dtype)
        weighted_loss = weighted_loss + weight * component_loss
        normalizer = normalizer + weight
    if not torch.any(normalizer > 0):
        return logits.sum() * 0.0
    return weighted_loss / normalizer


class PredicateLossMixin:
    """Blend token-weighted masked predicate BCE with the primary objective."""

    predicate_loss_weight = 0.0

    def _record_predicate_training_exposure(
        self,
        predicate_labels: torch.Tensor,
        predicate_weights: torch.Tensor,
        predicate_condition_ids: torch.Tensor | None,
        secondary_labels: torch.Tensor | None,
        primary_objective_weights: torch.Tensor | None,
    ) -> None:
        channels = getattr(self, "predicate_exposure_channels", None)
        if channels is None:
            return
        if predicate_labels.shape[-1] != len(channels):
            raise ValueError(
                "predicate exposure channels do not align with batch labels: "
                f"channels={len(channels)} shape={tuple(predicate_labels.shape)}"
            )
        annotated = predicate_labels != -100
        positive = annotated & (predicate_labels == 1)
        negative = annotated & (predicate_labels == 0)
        positive_cells = positive.sum(dim=(0, 1)).detach().cpu().tolist()
        negative_cells = negative.sum(dim=(0, 1)).detach().cpu().tolist()
        positive_weight = (
            torch.where(positive, predicate_weights, torch.zeros_like(predicate_weights))
            .sum(dim=(0, 1))
            .detach()
            .cpu()
            .tolist()
        )
        negative_weight = (
            torch.where(negative, predicate_weights, torch.zeros_like(predicate_weights))
            .sum(dim=(0, 1))
            .detach()
            .cpu()
            .tolist()
        )
        if not hasattr(self, "_predicate_exposure_counts"):
            self._predicate_exposure_counts = {
                "physical_batches": 0,
                "windows": 0,
                "predicate_positive_cells": Counter(),
                "predicate_known_negative_cells": Counter(),
                "predicate_positive_objective_weight": Counter(),
                "predicate_known_negative_objective_weight": Counter(),
                "condition_positive_cells": Counter(),
                "condition_known_negative_cells": Counter(),
                "condition_positive_objective_weight": Counter(),
                "condition_known_negative_objective_weight": Counter(),
                "primary_positive_tokens": Counter(),
                "primary_positive_objective_weight": Counter(),
                "reference_positive_tokens": Counter(),
                "reference_known_negative_tokens": Counter(),
                "reference_positive_objective_weight": Counter(),
                "reference_known_negative_objective_weight": Counter(),
            }
        counts = self._predicate_exposure_counts
        counts["physical_batches"] += 1
        counts["windows"] += int(predicate_labels.shape[0])
        for index, channel in enumerate(channels):
            counts["predicate_positive_cells"][channel] += int(positive_cells[index])
            counts["predicate_known_negative_cells"][channel] += int(negative_cells[index])
            counts["predicate_positive_objective_weight"][channel] += float(positive_weight[index])
            counts["predicate_known_negative_objective_weight"][channel] += float(negative_weight[index])

        condition_types = tuple(getattr(self, "predicate_exposure_condition_types", ()))
        if condition_types:
            if predicate_condition_ids is None:
                raise ValueError("conditioned predicate exposure requires predicate_condition_ids")
            if predicate_condition_ids.shape != predicate_labels.shape[:2]:
                raise ValueError("predicate condition ids do not align with batch labels")
            supervised_tokens = torch.any(annotated, dim=-1)
            invalid = (predicate_condition_ids < 0) | (predicate_condition_ids >= len(condition_types))
            if torch.any(supervised_tokens & invalid):
                raise ValueError("supervised predicate exposure has an invalid condition id")
            for condition_id, primary_type in enumerate(condition_types):
                condition_mask = predicate_condition_ids == condition_id
                for channel_id, channel in enumerate(channels):
                    key = (primary_type, channel)
                    positive_mask = positive[:, :, channel_id] & condition_mask
                    negative_mask = negative[:, :, channel_id] & condition_mask
                    counts["condition_positive_cells"][key] += int(positive_mask.sum().detach().cpu())
                    counts["condition_known_negative_cells"][key] += int(negative_mask.sum().detach().cpu())
                    counts["condition_positive_objective_weight"][key] += float(
                        torch.where(
                            positive_mask,
                            predicate_weights[:, :, channel_id],
                            torch.zeros_like(predicate_weights[:, :, channel_id]),
                        )
                        .sum()
                        .detach()
                        .cpu()
                    )
                    counts["condition_known_negative_objective_weight"][key] += float(
                        torch.where(
                            negative_mask,
                            predicate_weights[:, :, channel_id],
                            torch.zeros_like(predicate_weights[:, :, channel_id]),
                        )
                        .sum()
                        .detach()
                        .cpu()
                    )

        primary_labels = tuple(getattr(self, "primary_exposure_labels", ()))
        reference_ids = getattr(self, "predicate_exposure_reference_label_ids", None)
        if not primary_labels and reference_ids is None:
            return
        if secondary_labels is None:
            raise ValueError("primary/reference exposure tracking requires secondary_labels")
        if primary_objective_weights is None:
            primary_objective_weights = (secondary_labels != -100).to(torch.float)
        elif primary_objective_weights.shape != secondary_labels.shape:
            raise ValueError("primary/reference exposure weights do not align with secondary labels")
        valid_reference_target = secondary_labels != -100
        if primary_labels:
            invalid_primary = valid_reference_target & (
                (secondary_labels < 0) | (secondary_labels >= len(primary_labels))
            )
            if torch.any(invalid_primary):
                raise ValueError("primary exposure target is outside the configured label inventory")
            active_primary = valid_reference_target & (primary_objective_weights > 0)
            raw_ids = secondary_labels[valid_reference_target]
            active_ids = secondary_labels[active_primary]
            token_counts = torch.bincount(raw_ids, minlength=len(primary_labels))
            objective_counts = torch.zeros(
                len(primary_labels),
                dtype=primary_objective_weights.dtype,
                device=primary_objective_weights.device,
            )
            objective_counts.scatter_add_(0, active_ids, primary_objective_weights[active_primary])
            for label, tokens, weight in zip(
                primary_labels,
                token_counts.detach().cpu().tolist(),
                objective_counts.detach().cpu().tolist(),
                strict=True,
            ):
                counts["primary_positive_tokens"][label] += int(tokens)
                counts["primary_positive_objective_weight"][label] += float(weight)
        if reference_ids is None:
            return
        for primary_type, label_ids in reference_ids.items():
            positive_reference = torch.zeros_like(valid_reference_target)
            for label_id in label_ids:
                positive_reference |= secondary_labels == label_id
            counts["reference_positive_tokens"][primary_type] += int(positive_reference.sum().detach().cpu())
            counts["reference_known_negative_tokens"][primary_type] += int(
                (valid_reference_target & ~positive_reference).sum().detach().cpu()
            )
            counts["reference_positive_objective_weight"][primary_type] += float(
                torch.where(
                    positive_reference,
                    primary_objective_weights,
                    torch.zeros_like(primary_objective_weights),
                )
                .sum()
                .detach()
                .cpu()
            )
            counts["reference_known_negative_objective_weight"][primary_type] += float(
                torch.where(
                    valid_reference_target & ~positive_reference,
                    primary_objective_weights,
                    torch.zeros_like(primary_objective_weights),
                )
                .sum()
                .detach()
                .cpu()
            )

    def predicate_training_exposure_receipt(self) -> dict[str, object]:
        channels = tuple(getattr(self, "predicate_exposure_channels", ()))
        condition_types = tuple(getattr(self, "predicate_exposure_condition_types", ()))
        primary_labels = tuple(getattr(self, "primary_exposure_labels", ()))
        reference_ids = getattr(self, "predicate_exposure_reference_label_ids", {})
        counts = getattr(self, "_predicate_exposure_counts", {})

        def channel_values(name: str, *, floating: bool = False) -> dict[str, float | int]:
            source = counts.get(name, {})
            cast = float if floating else int
            return {channel: cast(source.get(channel, 0)) for channel in channels}

        return {
            "schema": "pii-ont3-training-exposure",
            "schema_version": 1,
            "scope": "actual training-mode physical batches observed by this trainer process",
            "physical_batches": int(counts.get("physical_batches", 0)),
            "windows": int(counts.get("windows", 0)),
            "predicate": {
                "positive_token_cells": channel_values("predicate_positive_cells"),
                "known_negative_token_cells": channel_values("predicate_known_negative_cells"),
                "positive_objective_weight": channel_values(
                    "predicate_positive_objective_weight", floating=True
                ),
                "known_negative_objective_weight": channel_values(
                    "predicate_known_negative_objective_weight", floating=True
                ),
            },
            "predicate_by_primary_type": {
                primary_type: {
                    channel: {
                        "positive_token_cells": int(
                            counts.get("condition_positive_cells", {}).get((primary_type, channel), 0)
                        ),
                        "known_negative_token_cells": int(
                            counts.get("condition_known_negative_cells", {}).get((primary_type, channel), 0)
                        ),
                        "positive_objective_weight": float(
                            counts.get("condition_positive_objective_weight", {}).get(
                                (primary_type, channel), 0
                            )
                        ),
                        "known_negative_objective_weight": float(
                            counts.get("condition_known_negative_objective_weight", {}).get(
                                (primary_type, channel), 0
                            )
                        ),
                    }
                    for channel in channels
                }
                for primary_type in condition_types
            },
            "primary_output": {
                label: {
                    "positive_tokens": int(counts.get("primary_positive_tokens", {}).get(label, 0)),
                    "positive_objective_weight": float(
                        counts.get("primary_positive_objective_weight", {}).get(label, 0)
                    ),
                }
                for label in primary_labels
            },
            "reference": {
                primary_type: {
                    "positive_tokens": int(counts.get("reference_positive_tokens", {}).get(primary_type, 0)),
                    "known_negative_tokens": int(
                        counts.get("reference_known_negative_tokens", {}).get(primary_type, 0)
                    ),
                    "positive_objective_weight": float(
                        counts.get("reference_positive_objective_weight", {}).get(primary_type, 0)
                    ),
                    "known_negative_objective_weight": float(
                        counts.get("reference_known_negative_objective_weight", {}).get(
                            primary_type,
                            0,
                        )
                    ),
                }
                for primary_type in sorted(reference_ids)
            },
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        predicate_labels = inputs.pop("predicate_labels", None)
        predicate_weights = inputs.pop("predicate_weights", None)
        predicate_condition_ids = inputs.pop("predicate_condition_ids", None)
        if predicate_labels is None:
            raise ValueError("predicate loss requires predicate_labels on every tagging batch")
        if predicate_weights is None:
            raise ValueError("predicate loss requires predicate_weights on every tagging batch")
        if getattr(model, "training", True):
            # Exposure is counted against the head's own label inventory: the mapped
            # target when a correctness map supplies one, else the native target.
            head_targets = inputs.get("secondary_labels")
            if head_targets is None:
                head_targets = inputs.get("labels")
            self._record_predicate_training_exposure(
                predicate_labels,
                predicate_weights,
                predicate_condition_ids,
                head_targets,
                inputs.get("primary_objective_weights"),
            )
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        predicate_logits = getattr(outputs, "predicate_logits", None)
        if predicate_logits is None:
            raise RuntimeError("predicate loss requires model predicate_logits")
        if predicate_logits.ndim == 4:
            if predicate_condition_ids is None:
                raise ValueError("conditioned predicate loss requires predicate_condition_ids")
            predicate_logits = select_conditioned_predicate_logits(
                predicate_logits,
                predicate_condition_ids,
                predicate_weights,
            )
        elif predicate_condition_ids is not None:
            raise ValueError("unconditioned predicate logits cannot consume predicate_condition_ids")
        predicate_loss = masked_predicate_bce_loss(
            predicate_logits,
            predicate_labels,
            predicate_weights,
            conditioned_predicate_positive_weights(
                predicate_labels,
                predicate_condition_ids,
                getattr(self, "predicate_positive_weights", None),
            ),
        )
        if torch.any(predicate_weights > 0):
            loss = (loss + self.predicate_loss_weight * predicate_loss) / (1.0 + self.predicate_loss_weight)
        return (loss, outputs) if return_outputs else loss


class SubclassLossMixin:
    """Blend carrier-conditioned categorical components with primary loss."""

    subclass_loss_weight = 0.0

    def _record_subclass_training_exposure(
        self,
        block_ids: torch.Tensor,
        target_ids: torch.Tensor,
        objective_weights: torch.Tensor,
        learning_weights: torch.Tensor,
    ) -> None:
        blocks = tuple(getattr(self, "subclass_exposure_blocks", ()))
        if not blocks:
            return
        expected_shape = block_ids.shape
        if (
            block_ids.ndim != 2
            or target_ids.shape != expected_shape
            or objective_weights.shape != expected_shape
            or learning_weights.shape != expected_shape
        ):
            raise ValueError("subclass exposure component fields must share one [batch, component] shape")
        effective_weights = objective_weights * learning_weights
        active = effective_weights > 0
        if not hasattr(self, "_subclass_exposure_counts"):
            self._subclass_exposure_counts = {
                "physical_batches": 0,
                "windows": 0,
                "components": Counter(),
                "objective_weight": Counter(),
                "effective_weight": Counter(),
            }
        counts = self._subclass_exposure_counts
        counts["physical_batches"] += 1
        counts["windows"] += int(block_ids.shape[0])
        coordinates = torch.nonzero(active, as_tuple=False).detach().cpu().tolist()
        block_values = block_ids.detach().cpu().tolist()
        target_values = target_ids.detach().cpu().tolist()
        objective_values = objective_weights.detach().cpu().tolist()
        effective_values = effective_weights.detach().cpu().tolist()
        for batch_index, component_index in coordinates:
            block_id = int(block_values[batch_index][component_index])
            target_id = int(target_values[batch_index][component_index])
            if block_id < 0 or block_id >= len(blocks):
                raise ValueError(f"active subclass exposure has invalid block id {block_id}")
            block = blocks[block_id]
            outcomes = block["outcomes"]
            if target_id < 0 or target_id >= len(outcomes):
                raise ValueError(
                    f"active subclass exposure has invalid target id {target_id} for block {block_id}"
                )
            key = (block_id, target_id)
            counts["components"][key] += 1
            counts["objective_weight"][key] += float(objective_values[batch_index][component_index])
            counts["effective_weight"][key] += float(effective_values[batch_index][component_index])

    def subclass_training_exposure_receipt(self) -> dict[str, object]:
        """Return actual training-batch exposure for categorical subclasses."""
        blocks = tuple(getattr(self, "subclass_exposure_blocks", ()))
        counts = getattr(self, "_subclass_exposure_counts", {})
        by_family_target: dict[str, dict[str, float | int]] = {}
        block_receipts = {}
        for block_id, block in enumerate(blocks):
            targets = {}
            for target_id, outcome in enumerate(block["outcomes"]):
                key = (block_id, target_id)
                values = {
                    "components": int(counts.get("components", {}).get(key, 0)),
                    "objective_weight": float(counts.get("objective_weight", {}).get(key, 0)),
                    "effective_weight": float(counts.get("effective_weight", {}).get(key, 0)),
                }
                targets[outcome] = values
                aggregate_key = f"{block['family']}={outcome}"
                aggregate = by_family_target.setdefault(
                    aggregate_key,
                    {"components": 0, "objective_weight": 0.0, "effective_weight": 0.0},
                )
                for name, value in values.items():
                    aggregate[name] += value
            block_receipts[f"{block['family']}@{block['primary_type']}"] = {
                "start": block["start"],
                "width": block["width"],
                "targets": targets,
            }
        return {
            "schema": "pii-ont3-subclass-training-exposure",
            "schema_version": 1,
            "scope": "actual training-mode physical batches observed by this trainer process",
            "physical_batches": int(counts.get("physical_batches", 0)),
            "windows": int(counts.get("windows", 0)),
            "blocks": block_receipts,
            "by_family_target": by_family_target,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        field_names = (
            "subclass_block_ids",
            "subclass_target_ids",
            "subclass_scope_ids",
            "subclass_token_masks",
            "subclass_objective_weights",
            "subclass_learning_weights",
        )
        fields = {name: inputs.pop(name, None) for name in field_names}
        missing = [name for name, value in fields.items() if value is None]
        if missing:
            raise ValueError("subclass loss requires every component field: " + ", ".join(missing))
        if getattr(model, "training", True):
            self._record_subclass_training_exposure(
                fields["subclass_block_ids"],
                fields["subclass_target_ids"],
                fields["subclass_objective_weights"],
                fields["subclass_learning_weights"],
            )
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        subclass_logits = getattr(outputs, "subclass_logits", None)
        if subclass_logits is None:
            raise RuntimeError("subclass loss requires model subclass_logits")
        subclass_loss = masked_subclass_categorical_loss(
            subclass_logits,
            fields["subclass_block_ids"],
            fields["subclass_target_ids"],
            fields["subclass_scope_ids"],
            fields["subclass_token_masks"],
            fields["subclass_objective_weights"],
            fields["subclass_learning_weights"],
            self.subclass_blocks,
        )
        effective_weights = fields["subclass_objective_weights"] * fields["subclass_learning_weights"]
        if torch.any(effective_weights > 0):
            loss = (loss + self.subclass_loss_weight * subclass_loss) / (1.0 + self.subclass_loss_weight)
        return (loss, outputs) if return_outputs else loss


class PredicateSubclassLossMixin:
    """Blend primary, Bernoulli, categorical, and semantic losses symmetrically."""

    predicate_loss_weight = 0.0
    subclass_loss_weight = 0.0
    reference_type_residual_loss_weight = 0.0
    reference_primary_positive_loss_weight = 0.0
    logical_step_objective_normalization = False
    _record_predicate_training_exposure = PredicateLossMixin._record_predicate_training_exposure
    predicate_training_exposure_receipt = PredicateLossMixin.predicate_training_exposure_receipt
    _record_subclass_training_exposure = SubclassLossMixin._record_subclass_training_exposure
    subclass_training_exposure_receipt = SubclassLossMixin.subclass_training_exposure_receipt

    def _primary_group_objective_weights(self) -> dict[str, float]:
        return {
            "primary": 1.0,
            "complete_presence": getattr(self, "complete_presence_loss_weight", 0.0),
            "complete_family": getattr(self, "complete_family_loss_weight", 0.0),
            "complete_boundary": getattr(self, "complete_boundary_loss_weight", 0.0),
            "partial_o": getattr(self, "partial_o_loss_weight", 0.0),
            "partial_expected_entity_ratio": getattr(
                self,
                "partial_expected_entity_ratio_loss_weight",
                0.0,
            ),
            "partial_parent_presence": getattr(
                self,
                "partial_parent_presence_kl_weight",
                0.0,
            ),
        }

    def _data_objective_weights(self) -> dict[str, float]:
        return {
            "primary_group": 1.0,
            "reference_primary_positive": self.reference_primary_positive_loss_weight,
            "predicate": self.predicate_loss_weight,
            "reference_type_residual": self.reference_type_residual_loss_weight,
            "subclass": self.subclass_loss_weight,
        }

    def _batch_objective_masses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        correctness_map = getattr(self, "correctness_map", None)
        if correctness_map is None:
            raise RuntimeError("logical-step objective normalization requires a correctness map")
        old_labels = batch["labels"]
        new_labels = batch["secondary_labels"]
        primary_weights = batch.get("primary_objective_weights")
        masses = {
            "primary": mapped_head_objective_mass(
                new_labels,
                old_labels,
                own_o_label_id=correctness_map.new_outside_id,
                cross_o_label_id=correctness_map.old_outside_id,
                o_token_weight=float(getattr(self, "o_token_loss_weight", 1.0)),
                token_objective_weights=primary_weights,
            )
        }

        reference_types = set(correctness_map.legacy_outside_unknown_primary_types)
        reference_ids = {
            index
            for index, label in enumerate(correctness_map.new_labels)
            if label != "O" and label.split("-", 1)[-1] in reference_types
        }
        reference_mask = torch.zeros_like(new_labels, dtype=torch.bool)
        for label_id in reference_ids:
            reference_mask |= new_labels == label_id
        reference_labels = new_labels.masked_fill(~reference_mask, -100)
        reference_weights = (
            None if primary_weights is None else primary_weights.masked_fill(reference_labels == -100, 0.0)
        )
        masses["reference_primary_positive"] = mapped_head_objective_mass(
            reference_labels,
            torch.full_like(old_labels, -100),
            own_o_label_id=correctness_map.new_outside_id,
            cross_o_label_id=correctness_map.old_outside_id,
            token_objective_weights=reference_weights,
        )

        predicate_labels = batch["predicate_labels"]
        predicate_weights = batch["predicate_weights"]
        predicate_positive_weights = conditioned_predicate_positive_weights(
            predicate_labels,
            batch.get("predicate_condition_ids"),
            getattr(self, "predicate_positive_weights", None),
        )
        masses["predicate"] = self._binary_objective_normalizer(
            predicate_labels,
            predicate_weights,
            predicate_positive_weights,
        )

        reference_type_labels = batch.get("reference_type_labels")
        reference_type_weights = batch.get("reference_type_weights")
        if reference_type_labels is not None and reference_type_weights is not None:
            reference_positive_weights = token_channel_positive_weights(
                reference_type_labels,
                getattr(self, "reference_type_residual_positive_weights", None),
            )
            masses["reference_type_residual"] = self._binary_objective_normalizer(
                reference_type_labels,
                reference_type_weights,
                reference_positive_weights,
            )
        else:
            masses["reference_type_residual"] = old_labels.new_zeros((), dtype=torch.float)

        masses["subclass"] = batch["subclass_objective_weights"] * batch["subclass_learning_weights"]
        masses["subclass"] = masses["subclass"].sum()
        for component, field in (
            ("complete_presence", "complete_presence_labels"),
            ("complete_family", "complete_family_labels"),
            ("complete_boundary", "complete_boundary_labels"),
        ):
            labels = batch.get(field)
            masses[component] = (
                old_labels.new_zeros((), dtype=torch.float)
                if labels is None
                else torch.count_nonzero(labels != -100).float()
            )
        consistency_mask = batch.get("consistency_mask")
        consistency_mass = (
            old_labels.new_zeros((), dtype=torch.float)
            if consistency_mask is None
            else torch.count_nonzero(consistency_mask).float()
        )
        masses["partial_o"] = consistency_mass
        masses["partial_parent_presence"] = consistency_mass
        positive_mask = batch.get("partial_entity_positive_mask")
        masses["partial_expected_entity_ratio"] = (
            old_labels.new_zeros((), dtype=torch.float)
            if consistency_mask is None or positive_mask is None
            else torch.count_nonzero(consistency_mask.bool() | positive_mask.bool()).float()
        )
        return masses

    def _get_num_items_in_batch(self, batch_samples, device):
        if not self.logical_step_objective_normalization or not getattr(self, "is_in_train", False):
            return super()._get_num_items_in_batch(batch_samples, device)
        if not self.dual_head_retired:
            raise ValueError("logical-step objective normalization currently requires a mapped single head")
        if getattr(self, "partial_entity_pu_loss_weight", 0.0):
            raise ValueError("logical-step objective normalization does not yet support PU risk")
        if getattr(self, "entity_dice_weight", 0.0):
            raise ValueError("logical-step objective normalization does not support entity Dice")
        totals: dict[str, torch.Tensor] = {}
        for batch in batch_samples:
            for component, mass in self._batch_objective_masses(batch).items():
                totals[component] = totals.get(component, mass.new_zeros(())) + mass
        if not torch.any(totals["primary"] > 0):
            raise ValueError("logical optimizer step has no primary supervision mass")
        primary_group_weight_sum = totals["primary"].new_zeros(())
        for component, weight in self._primary_group_objective_weights().items():
            mass = totals[component]
            primary_group_weight_sum = primary_group_weight_sum + torch.where(
                mass > 0,
                mass.new_tensor(float(weight)),
                mass.new_zeros(()),
            )
        data_weight_sum = primary_group_weight_sum.new_tensor(1.0)
        for component, weight in self._data_objective_weights().items():
            if component == "primary_group":
                continue
            mass = totals[component]
            data_weight_sum = data_weight_sum + torch.where(
                mass > 0,
                mass.new_tensor(float(weight)),
                mass.new_zeros(()),
            )
        return LogicalStepObjectiveMasses(
            totals=dict(totals),
            physical_batches=len(batch_samples),
            primary_group_weight_sum=primary_group_weight_sum,
            data_weight_sum=data_weight_sum,
        )

    def _record_eval_objective_loss(
        self,
        name: str,
        loss: torch.Tensor,
        normalizer: float | torch.Tensor,
    ) -> None:
        """Accumulate one validation component under its own applied mass."""
        totals = getattr(self, "_ont3_eval_objective_totals", None)
        if totals is None:
            return
        normalizer_value = float(
            normalizer.detach().float().cpu() if isinstance(normalizer, torch.Tensor) else normalizer
        )
        if normalizer_value <= 0:
            return
        loss_value = float(loss.detach().float().cpu())
        if not math.isfinite(loss_value) or not math.isfinite(normalizer_value):
            raise ValueError(f"non-finite validation objective component {name}")
        total = totals.setdefault(name, [0.0, 0.0, 0.0])
        total[0] += loss_value * normalizer_value
        total[1] += normalizer_value
        total[2] += 1.0

    @staticmethod
    def _binary_objective_normalizer(
        labels: torch.Tensor,
        weights: torch.Tensor,
        positive_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the exact weight mass used by masked binary BCE."""
        active = (labels != -100) & (weights > 0)
        if not torch.any(active):
            return weights.new_zeros(())
        effective = weights.float()[active]
        if positive_weights is not None:
            effective = effective * torch.where(
                labels[active] == 1,
                positive_weights.float()[active],
                torch.ones_like(effective),
            )
        return effective.sum()

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Publish support-weighted validation losses for each ont3 objective."""
        if getattr(self, "_ont3_eval_objective_totals", None) is not None:
            raise RuntimeError("nested ont3 validation objective collection is unsupported")
        self._ont3_eval_objective_totals = {}
        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            totals = self._ont3_eval_objective_totals
        finally:
            self._ont3_eval_objective_totals = None
        for name, values in sorted(totals.items()):
            reduced = torch.tensor(
                values,
                dtype=torch.float64,
                device=self.args.device,
            )
            reduced = self.accelerator.reduce(reduced, reduction="sum")
            numerator, normalizer, active_batches = reduced.detach().cpu().tolist()
            if normalizer <= 0:
                continue
            output.metrics[f"{metric_key_prefix}_ont3_{name}_loss"] = numerator / normalizer
            output.metrics[f"{metric_key_prefix}_ont3_{name}_normalizer"] = normalizer
            output.metrics[f"{metric_key_prefix}_ont3_{name}_active_batches"] = active_batches
        return output

    def _record_reference_type_residual_exposure(
        self,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        primary_types = tuple(getattr(self, "reference_type_residual_types", ()))
        if labels.ndim != 3 or labels.shape != weights.shape or labels.shape[-1] != len(primary_types):
            raise ValueError("reference residual exposure tensors do not align with configured types")
        active = (labels != -100) & (weights > 0)
        positive = active & (labels == 1)
        negative = active & (labels == 0)
        if not hasattr(self, "_reference_type_residual_exposure_counts"):
            self._reference_type_residual_exposure_counts = {
                "physical_batches": 0,
                "windows": 0,
                "positive_tokens": Counter(),
                "known_negative_tokens": Counter(),
                "positive_objective_weight": Counter(),
                "known_negative_objective_weight": Counter(),
            }
        counts = self._reference_type_residual_exposure_counts
        counts["physical_batches"] += 1
        counts["windows"] += int(labels.shape[0])
        for index, primary_type in enumerate(primary_types):
            positive_mask = positive[:, :, index]
            negative_mask = negative[:, :, index]
            counts["positive_tokens"][primary_type] += int(positive_mask.sum().detach().cpu())
            counts["known_negative_tokens"][primary_type] += int(negative_mask.sum().detach().cpu())
            counts["positive_objective_weight"][primary_type] += float(
                torch.where(positive_mask, weights[:, :, index], 0).sum().detach().cpu()
            )
            counts["known_negative_objective_weight"][primary_type] += float(
                torch.where(negative_mask, weights[:, :, index], 0).sum().detach().cpu()
            )

    def reference_type_residual_training_exposure_receipt(self) -> dict[str, object]:
        """Return actual training-batch exposure for the semantic residual."""
        primary_types = tuple(getattr(self, "reference_type_residual_types", ()))
        counts = getattr(self, "_reference_type_residual_exposure_counts", {})
        positive_weights = getattr(self, "reference_type_residual_positive_weights", None)
        multipliers = (
            [1.0] * len(primary_types)
            if positive_weights is None
            else [float(value) for value in positive_weights.tolist()]
        )
        return {
            "schema": "pii-ont3-reference-type-residual-training-exposure",
            "schema_version": 1,
            "scope": "actual training-mode physical batches observed by this trainer process",
            "physical_batches": int(counts.get("physical_batches", 0)),
            "windows": int(counts.get("windows", 0)),
            "types": {
                primary_type: {
                    "positive_tokens": int(counts.get("positive_tokens", {}).get(primary_type, 0)),
                    "known_negative_tokens": int(
                        counts.get("known_negative_tokens", {}).get(primary_type, 0)
                    ),
                    "positive_objective_weight": float(
                        counts.get("positive_objective_weight", {}).get(primary_type, 0)
                    ),
                    "known_negative_objective_weight": float(
                        counts.get("known_negative_objective_weight", {}).get(primary_type, 0)
                    ),
                    "positive_learning_multiplier": multipliers[index],
                }
                for index, primary_type in enumerate(primary_types)
            },
        }

    def _reference_primary_positive_terms(
        self,
        logits: torch.Tensor,
        old_space_labels: torch.Tensor | None,
        new_space_labels: torch.Tensor | None,
        primary_objective_weights: torch.Tensor | None,
    ):
        """Return exact primary BIOES loss restricted to known reference positives."""
        correctness_map = getattr(self, "correctness_map", None)
        if correctness_map is None or old_space_labels is None or new_space_labels is None:
            return None
        reference_types = set(correctness_map.legacy_outside_unknown_primary_types)
        reference_label_ids = {
            index
            for index, label in enumerate(correctness_map.new_labels)
            if label != "O" and label.split("-", 1)[-1] in reference_types
        }
        if not reference_label_ids:
            return None
        reference_mask = torch.zeros_like(new_space_labels, dtype=torch.bool)
        for label_id in reference_label_ids:
            reference_mask |= new_space_labels == label_id
        reference_new_labels = new_space_labels.masked_fill(~reference_mask, -100)
        reference_old_labels = torch.full_like(old_space_labels, -100)
        reference_objective_weights = (
            None
            if primary_objective_weights is None
            else primary_objective_weights.masked_fill(reference_new_labels == -100, 0.0)
        )
        terms = head_loss_terms(
            logits,
            reference_new_labels,
            reference_old_labels,
            correctness_map.old_to_new,
            own_o_label_id=correctness_map.new_outside_id,
            cross_o_label_id=correctness_map.old_outside_id,
            token_objective_weights=reference_objective_weights,
        )
        return terms if terms.covered else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logical_step_masses = (
            num_items_in_batch
            if getattr(model, "training", True) and isinstance(num_items_in_batch, LogicalStepObjectiveMasses)
            else None
        )
        if (
            getattr(model, "training", True)
            and self.logical_step_objective_normalization
            and logical_step_masses is None
        ):
            raise ValueError("training batch is missing logical-step objective masses")
        old_space_labels = inputs.get("labels")
        new_space_labels = inputs.get("secondary_labels")
        primary_objective_weights = inputs.get("primary_objective_weights")
        complete_presence_labels = inputs.get("complete_presence_labels")
        predicate_labels = inputs.pop("predicate_labels", None)
        predicate_weights = inputs.pop("predicate_weights", None)
        predicate_condition_ids = inputs.pop("predicate_condition_ids", None)
        if predicate_labels is None or predicate_weights is None:
            raise ValueError("joint predicate/subclass loss requires predicate labels and weights")
        subclass_names = (
            "subclass_block_ids",
            "subclass_target_ids",
            "subclass_scope_ids",
            "subclass_token_masks",
            "subclass_objective_weights",
            "subclass_learning_weights",
        )
        subclass = {name: inputs.pop(name, None) for name in subclass_names}
        missing = [name for name, value in subclass.items() if value is None]
        if missing:
            raise ValueError(
                "joint predicate/subclass loss requires every subclass field: " + ", ".join(missing)
            )
        reference_type_labels = inputs.pop("reference_type_labels", None)
        reference_type_weights = inputs.pop("reference_type_weights", None)
        if self.reference_type_residual_loss_weight and (
            reference_type_labels is None or reference_type_weights is None
        ):
            raise ValueError("reference-type residual loss requires reference labels and weights")
        if getattr(model, "training", True):
            # Exposure is counted against the head's own label inventory: the mapped
            # target when a correctness map supplies one, else the native target.
            head_targets = inputs.get("secondary_labels")
            if head_targets is None:
                head_targets = inputs.get("labels")
            self._record_predicate_training_exposure(
                predicate_labels,
                predicate_weights,
                predicate_condition_ids,
                head_targets,
                inputs.get("primary_objective_weights"),
            )
            if self.reference_type_residual_loss_weight:
                self._record_reference_type_residual_exposure(
                    reference_type_labels,
                    reference_type_weights,
                )
            self._record_subclass_training_exposure(
                subclass["subclass_block_ids"],
                subclass["subclass_target_ids"],
                subclass["subclass_objective_weights"],
                subclass["subclass_learning_weights"],
            )
        primary_loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        predicate_logits = getattr(outputs, "predicate_logits", None)
        if predicate_logits is None:
            raise RuntimeError("joint predicate/subclass loss requires model predicate_logits")
        if predicate_logits.ndim == 4:
            if predicate_condition_ids is None:
                raise ValueError("conditioned predicate loss requires predicate_condition_ids")
            predicate_logits = select_conditioned_predicate_logits(
                predicate_logits,
                predicate_condition_ids,
                predicate_weights,
            )
        elif predicate_condition_ids is not None:
            raise ValueError("unconditioned predicate logits cannot consume predicate_condition_ids")
        predicate_positive_weights = conditioned_predicate_positive_weights(
            predicate_labels,
            predicate_condition_ids,
            getattr(self, "predicate_positive_weights", None),
        )
        predicate_loss = masked_predicate_bce_loss(
            predicate_logits,
            predicate_labels,
            predicate_weights,
            predicate_positive_weights,
        )
        predicate_mass = self._binary_objective_normalizer(
            predicate_labels,
            predicate_weights,
            predicate_positive_weights,
        )

        reference_type_loss = None
        if self.reference_type_residual_loss_weight:
            reference_type_logits = getattr(outputs, "reference_type_logits", None)
            if reference_type_logits is None:
                raise RuntimeError("reference-type residual loss requires model reference_type_logits")
            reference_positive_weights = token_channel_positive_weights(
                reference_type_labels,
                getattr(self, "reference_type_residual_positive_weights", None),
            )
            reference_type_loss = masked_predicate_bce_loss(
                reference_type_logits,
                reference_type_labels,
                reference_type_weights,
                reference_positive_weights,
            )
            reference_type_mass = self._binary_objective_normalizer(
                reference_type_labels,
                reference_type_weights,
                reference_positive_weights,
            )
        else:
            reference_type_mass = predicate_weights.new_zeros(())

        subclass_logits = getattr(outputs, "subclass_logits", None)
        if subclass_logits is None:
            raise RuntimeError("joint predicate/subclass loss requires model subclass_logits")
        subclass_loss = masked_subclass_categorical_loss(
            subclass_logits,
            subclass["subclass_block_ids"],
            subclass["subclass_target_ids"],
            subclass["subclass_scope_ids"],
            subclass["subclass_token_masks"],
            subclass["subclass_objective_weights"],
            subclass["subclass_learning_weights"],
            self.subclass_blocks,
        )

        reference_primary_terms = (
            None
            if old_space_labels is None
            or new_space_labels is None
            or getattr(self, "correctness_map", None) is None
            else self._reference_primary_positive_terms(
                outputs.logits,
                old_space_labels,
                new_space_labels,
                primary_objective_weights,
            )
        )
        reference_primary_loss = (
            None
            if reference_primary_terms is None
            else reference_primary_terms.total / reference_primary_terms.normalizer
        )

        if not getattr(model, "training", True):
            correctness_map = getattr(self, "correctness_map", None)
            if correctness_map is not None and old_space_labels is not None and new_space_labels is not None:
                primary_terms = head_loss_terms(
                    outputs.logits,
                    new_space_labels,
                    old_space_labels,
                    correctness_map.old_to_new,
                    own_o_label_id=correctness_map.new_outside_id,
                    cross_o_label_id=correctness_map.old_outside_id,
                    token_objective_weights=primary_objective_weights,
                )
                if primary_terms.covered:
                    self._record_eval_objective_loss(
                        "primary",
                        primary_terms.total / primary_terms.normalizer,
                        primary_terms.normalizer,
                    )
                reference_types = set(correctness_map.legacy_outside_unknown_primary_types)
                reference_label_ids = {
                    index
                    for index, label in enumerate(correctness_map.new_labels)
                    if label != "O" and label.split("-", 1)[-1] in reference_types
                }
                reference_mask = torch.zeros_like(new_space_labels, dtype=torch.bool)
                for label_id in reference_label_ids:
                    reference_mask |= new_space_labels == label_id
                inherited_new_labels = new_space_labels.masked_fill(reference_mask, -100)
                inherited_objective_weights = (
                    None
                    if primary_objective_weights is None
                    else primary_objective_weights.masked_fill(
                        (inherited_new_labels == -100) & (old_space_labels == -100),
                        0.0,
                    )
                )
                inherited_terms = head_loss_terms(
                    outputs.logits,
                    inherited_new_labels,
                    old_space_labels,
                    correctness_map.old_to_new,
                    own_o_label_id=correctness_map.new_outside_id,
                    cross_o_label_id=correctness_map.old_outside_id,
                    token_objective_weights=inherited_objective_weights,
                )
                if inherited_terms.covered:
                    self._record_eval_objective_loss(
                        "ont2_compatible_primary",
                        inherited_terms.total / inherited_terms.normalizer,
                        inherited_terms.normalizer,
                    )
                if reference_primary_terms is not None:
                    self._record_eval_objective_loss(
                        "reference_primary_positive",
                        reference_primary_loss,
                        reference_primary_terms.normalizer,
                    )
            self._record_eval_objective_loss(
                "predicate",
                predicate_loss,
                predicate_mass,
            )
            if self.reference_type_residual_loss_weight:
                self._record_eval_objective_loss(
                    "reference_type_residual",
                    reference_type_loss,
                    reference_type_mass,
                )
            subclass_effective_weights = (
                subclass["subclass_objective_weights"] * subclass["subclass_learning_weights"]
            )
            self._record_eval_objective_loss(
                "subclass",
                subclass_loss,
                subclass_effective_weights.sum(),
            )
            if correctness_map is not None and complete_presence_labels is not None:
                complete_presence_tokens = torch.count_nonzero(complete_presence_labels != -100)
                if complete_presence_tokens.item():
                    self._record_eval_objective_loss(
                        "complete_presence",
                        entity_presence_loss(
                            outputs.logits,
                            complete_presence_labels,
                            correctness_map.new_outside_id,
                        ),
                        complete_presence_tokens,
                    )

        numerator = primary_loss
        denominator = logical_step_masses.data_weight_sum if logical_step_masses is not None else 1.0
        if self.reference_primary_positive_loss_weight and reference_primary_loss is not None:
            reference_primary_contribution = logical_step_mean_contribution(
                reference_primary_loss,
                reference_primary_terms.normalizer,
                logical_step_masses,
                "reference_primary_positive",
            )
            numerator = (
                numerator + self.reference_primary_positive_loss_weight * reference_primary_contribution
            )
            if logical_step_masses is None:
                denominator += self.reference_primary_positive_loss_weight
            if getattr(model, "training", True):
                self.dual_head_telemetry.update(
                    {
                        "dual_reference_primary_positive_loss": float(reference_primary_loss.detach()),
                        "dual_reference_primary_positive_weight": (
                            self.reference_primary_positive_loss_weight
                        ),
                        "dual_reference_primary_positive_tokens": float(reference_primary_terms.covered),
                        "dual_reference_primary_positive_normalizer": float(
                            reference_primary_terms.normalizer.detach()
                        ),
                    }
                )
        if torch.any(predicate_weights > 0):
            predicate_contribution = logical_step_mean_contribution(
                predicate_loss,
                predicate_mass,
                logical_step_masses,
                "predicate",
            )
            numerator = numerator + self.predicate_loss_weight * predicate_contribution
            if logical_step_masses is None:
                denominator += self.predicate_loss_weight
        if self.reference_type_residual_loss_weight and torch.any(reference_type_weights > 0):
            reference_type_contribution = logical_step_mean_contribution(
                reference_type_loss,
                reference_type_mass,
                logical_step_masses,
                "reference_type_residual",
            )
            numerator = numerator + self.reference_type_residual_loss_weight * reference_type_contribution
            if logical_step_masses is None:
                denominator += self.reference_type_residual_loss_weight
        subclass_effective_weights = (
            subclass["subclass_objective_weights"] * subclass["subclass_learning_weights"]
        )
        if torch.any(subclass_effective_weights > 0):
            subclass_mass = subclass_effective_weights.sum()
            subclass_contribution = logical_step_mean_contribution(
                subclass_loss,
                subclass_mass,
                logical_step_masses,
                "subclass",
            )
            numerator = numerator + self.subclass_loss_weight * subclass_contribution
            if logical_step_masses is None:
                denominator += self.subclass_loss_weight
        loss = numerator / denominator
        return (loss, outputs) if return_outputs else loss


def type_only_supervision_loss(
    logits: torch.Tensor,
    type_only_labels: torch.Tensor,
    type_label_ids: list[list[int]],
) -> torch.Tensor:
    """Cross entropy on a type's summed boundary tags rather than an exact tag.

    A token supervised this way is told which primary type it belongs to and
    nothing about whether it begins, continues or ends the span. The target
    probability is the sum over that type's B, I, E and S tags, so every way of
    placing the boundary is equally correct and the gradient only pushes
    probability mass from other types onto this one.

    Returns a zero that carries gradient when nothing in the batch is supervised
    this way, so the caller can add it unconditionally.
    """
    supervised = type_only_labels != -100
    if not bool(supervised.any()):
        return logits.sum() * 0.0
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    selected = type_only_labels[supervised]
    token_log_probabilities = log_probabilities[supervised]
    # One row per supervised token holding the log of its type's total mass.
    gathered = torch.stack(
        [
            torch.logsumexp(
                token_log_probabilities[:, torch.tensor(ids, device=logits.device)],
                dim=-1,
            )
            for ids in type_label_ids
        ],
        dim=-1,
    )
    return -gathered.gather(1, selected.unsqueeze(1).long()).mean()


class TypeOnlySupervisionMixin:
    """Add the boundary-free term for partial rows, leaving the stock loss intact."""

    type_only_label_ids: list[list[int]] = []
    type_only_loss_weight = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        type_only_labels = inputs.pop("type_only_labels", None)
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if type_only_labels is not None and self.type_only_loss_weight and model.training:
            loss = loss + self.type_only_loss_weight * type_only_supervision_loss(
                outputs.logits,
                type_only_labels,
                self.type_only_label_ids,
            )
        return (loss, outputs) if return_outputs else loss


class OTokenLossWeightMixin:
    """Replace stock token CE while preserving any base-trainer auxiliaries."""

    o_label_id = 0
    o_token_loss_weight = 1.0
    native_objective = False
    bioes_structure_cost_matrix = None
    bioes_transition_legality = None
    bioes_illegal_flip_scale = 1.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        row_o_weights = inputs.pop("o_weight", None)
        segmentation_prefix_masks = inputs.pop("segmentation_prefix_masks", None)
        if segmentation_prefix_masks is not None and not self.native_objective:
            raise ValueError("internal segmentation requires the native primary objective")
        outside_allowed = inputs.pop("native_outside_allowed", None)
        if outside_allowed is not None and not self.native_objective:
            raise ValueError("native unknown-outside targets require the native primary objective")
        objective_weights = inputs.pop("primary_objective_weights", None) if self.native_objective else None
        if (
            self.native_objective
            and model.training
            and (
                self.bioes_structure_cost_matrix is not None
                or objective_weights is not None
                or outside_allowed is not None
                or segmentation_prefix_masks is not None
            )
        ):
            labels = inputs["labels"]
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            if row_o_weights is not None:
                weights = (labels != -100).float() if objective_weights is None else objective_weights.float()
                objective_weights = weights * torch.where(
                    labels == self.o_label_id,
                    row_o_weights.to(weights.device).reshape(-1, 1),
                    1.0,
                )
            own_labels = labels
            cross_labels = torch.full_like(labels, -100)
            membership = torch.empty(0, outputs.logits.shape[-1], dtype=torch.bool, device=labels.device)
            mapped_costs = None
            o_token_weight = self.o_token_loss_weight if row_o_weights is None else 1.0
            if outside_allowed is not None:
                membership = outside_allowed.to(device=labels.device, dtype=torch.bool)
                if membership.shape != (labels.shape[0], outputs.logits.shape[-1]):
                    raise ValueError("native outside membership must match batch rows and head labels")
                if not membership[:, self.o_label_id].all():
                    raise ValueError("native outside membership must include O")
                uncertain = (labels == self.o_label_id) & (membership.sum(dim=1) > 1)[:, None]
                own_labels = labels.masked_fill(uncertain, -100)
                row_ids = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)
                cross_labels = torch.where(uncertain, row_ids, cross_labels)
                # Apply the original O weight to both direct and marginal O
                # tokens; row indices in the marginal map are not O label IDs.
                if o_token_weight != 1.0:
                    weights = (labels != -100).float() if objective_weights is None else objective_weights
                    objective_weights = weights * torch.where(labels == self.o_label_id, o_token_weight, 1.0)
                    o_token_weight = 1.0
                if self.bioes_structure_cost_matrix is not None:
                    costs = self.bioes_structure_cost_matrix.to(outputs.logits.device)
                    mapped_costs = costs[None].masked_fill(~membership[:, :, None], float("inf")).amin(dim=1)
            terms = head_loss_terms(
                outputs.logits,
                own_labels,
                cross_labels,
                membership,
                own_o_label_id=self.o_label_id,
                cross_o_label_id=self.o_label_id,
                o_token_weight=o_token_weight,
                token_objective_weights=objective_weights,
                structure_cost_matrix=self.bioes_structure_cost_matrix,
                mapped_structure_cost_rows=mapped_costs,
                transition_legality=self.bioes_transition_legality,
                illegal_flip_scale=self.bioes_illegal_flip_scale,
                segmentation_prefix_masks=segmentation_prefix_masks,
                label_names=[model.config.id2label[i] for i in range(outputs.logits.shape[-1])],
            )
            weighted_loss = terms.total / terms.normalizer.clamp_min(1e-8)
            loss = loss + weighted_loss - token_classification_loss(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss
        if row_o_weights is not None and model.training:
            labels = inputs["labels"]
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            stock_loss = token_classification_loss(outputs.logits, labels)
            weighted_loss = per_row_o_weighted_token_classification_loss(
                outputs.logits,
                labels,
                self.o_label_id,
                row_o_weights,
            )
            loss = loss + weighted_loss - stock_loss
            return (loss, outputs) if return_outputs else loss
        if self.o_token_loss_weight == 1.0 or not model.training:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs["labels"]
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        stock_loss = token_classification_loss(outputs.logits, labels)
        weighted_loss = o_weighted_token_classification_loss(
            outputs.logits,
            labels,
            self.o_label_id,
            self.o_token_loss_weight,
        )
        loss = loss + weighted_loss - stock_loss
        return (loss, outputs) if return_outputs else loss


def resolve_native_primary_objective(requested: dict, *, resume_config: object | None = None) -> dict:
    """Keep historical resumes exact and reject changes to a recorded native loss."""
    if resume_config is None:
        return {"version": "weighted-softmax-margin-v1", **requested}
    saved = getattr(resume_config, "pii_native_primary_objective", {"version": "legacy"})
    if saved == {"version": "legacy"}:
        return saved
    if saved.get("version") != "weighted-softmax-margin-v1":
        raise ValueError(f"unsupported checkpoint native primary objective: {saved!r}")
    if saved != {"version": "weighted-softmax-margin-v1", **requested}:
        raise ValueError(
            "native primary objective cannot change during exact resume; "
            "use the recorded margin settings or --init-from-checkpoint"
        )
    return saved


def o_token_weighted_trainer_class(
    trainer_class,
    *,
    o_weight_config=None,
    o_token_loss_weight=1.0,
    native_objective=False,
):
    """Install the native CE adapter or historical O-only weighting."""
    if not native_objective and o_weight_config is None and o_token_loss_weight == 1.0:
        return trainer_class
    return type(
        "OTokenWeighted" + trainer_class.__name__,
        (OTokenLossWeightMixin, trainer_class),
        {"native_objective": native_objective},
    )


@dataclass(frozen=True)
class UnionLabelGroups:
    """One fine head's BIOES rows, grouped into unions by coarse class.

    A group is one (BIES letter, coarse class) pair, never a mixture of letters:
    the union of ``B-person_name`` is exactly the ``B-`` rows of its member fine
    labels, so a boundary decision is never marginalized away. ``O`` is nobody's
    member and stays the head's own row.
    """

    names: tuple[str, ...]
    membership: torch.Tensor
    group_of_union_label: dict[str, int]
    representative_of_union_label: dict[str, str]
    group_of_fine_label: dict[str, int]
    projected_id2label: dict[int, str]
    fine_labels: tuple[str, ...]
    union_classes: tuple[str, ...]
    outside_members: tuple[str, ...]
    members_sha256: str


def load_union_members(path, nodes, label2id) -> UnionLabelGroups:
    """Bind a published member document to this corpus's fine BIOES inventory.

    The document maps each coarse class to the ordered fine labels that project to
    it, and names the fine labels that project to no class at all. It has to be
    total over the head's inventory and disjoint across classes: a fine label in
    two classes would make the unions overlap, and one in none of them plus none
    of the outside list would be a head row nothing accounts for. The document's
    own inventory has to be this corpus's inventory, in order, because the group
    membership is expressed in head rows.
    """
    members_path = Path(path)
    data = members_path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{members_path}: expected a JSON object")
    schema = document.get("schema")
    version = document.get("schema_version")
    if schema != UNION_MEMBERS_SCHEMA or version != UNION_MEMBERS_SCHEMA_VERSION:
        raise ValueError(
            f"{members_path}: {schema!r}/{version!r} is not the "
            f"{UNION_MEMBERS_SCHEMA!r}/{UNION_MEMBERS_SCHEMA_VERSION} member document"
        )
    if document.get("outside_label") != UNION_OUTSIDE_LABEL:
        raise ValueError(
            f"{members_path}: outside label {document.get('outside_label')!r} is not {UNION_OUTSIDE_LABEL!r}"
        )
    fine_labels = document.get("fine_labels")
    if fine_labels != list(nodes):
        raise ValueError(
            f"{members_path}: the member document inverts a {len(fine_labels or ())}-label inventory,"
            f" but this corpus publishes {len(nodes)}; a union group is expressed in head rows, so the"
            " two inventories have to be the same list in the same order"
        )
    members = document.get("members")
    outside_members = document.get("outside_members")
    if not isinstance(members, dict) or not members or not isinstance(outside_members, list):
        raise ValueError(f"{members_path}: expected a nonempty members object and an outside list")
    inventory = frozenset(str(node) for node in nodes)
    owner_of_fine_label = {}
    for union_class, class_members in sorted(members.items()):
        if not isinstance(class_members, list) or not class_members:
            raise ValueError(
                f"{members_path}: coarse class {union_class!r} has no member, so the union head could"
                " never predict it"
            )
        for member in class_members:
            if member not in inventory:
                raise ValueError(
                    f"{members_path}: {union_class!r} claims {member!r}, which is not a head row"
                )
            if member in owner_of_fine_label:
                raise ValueError(
                    f"{members_path}: {member!r} belongs to both {owner_of_fine_label[member]!r} and"
                    f" {union_class!r}; overlapping unions would double-count a head row"
                )
            owner_of_fine_label[member] = union_class
    accounted = sorted([*owner_of_fine_label, *(str(node) for node in outside_members)])
    if accounted != sorted(nodes):
        raise ValueError(
            f"{members_path}: the member document accounts for {len(accounted)} of this corpus's"
            f" {len(nodes)} head rows; every fine label belongs to exactly one coarse class or to"
            " the declared outside list"
        )

    union_classes = tuple(sorted(members))
    names = tuple(f"{prefix}-{union_class}" for union_class in union_classes for prefix in "BIES")
    group_of_union_label = {name: index for index, name in enumerate(names)}
    representative_of_union_label = {
        f"{prefix}-{union_class}": f"{prefix}-{members[union_class][0]}"
        for union_class in union_classes
        for prefix in "BIES"
    }
    group_of_fine_label = {
        f"{prefix}-{member}": group_of_union_label[f"{prefix}-{owner}"]
        for member, owner in owner_of_fine_label.items()
        for prefix in "BIES"
    }
    membership = torch.zeros((len(names), len(label2id)), dtype=torch.bool)
    for fine_label, group in group_of_fine_label.items():
        membership[group, label2id[fine_label]] = True
    projected_id2label = {
        index: (label if label == UNION_OUTSIDE_LABEL else _projected_fine_label(label, owner_of_fine_label))
        for label, index in label2id.items()
    }
    return UnionLabelGroups(
        names=names,
        membership=membership,
        group_of_union_label=group_of_union_label,
        representative_of_union_label=representative_of_union_label,
        group_of_fine_label=group_of_fine_label,
        projected_id2label=projected_id2label,
        fine_labels=tuple(str(node) for node in nodes),
        union_classes=union_classes,
        outside_members=tuple(str(node) for node in outside_members),
        members_sha256=hashlib.sha256(data).hexdigest(),
    )


def _projected_fine_label(label: str, owner_of_fine_label) -> str:
    """One fine BIOES label in the coarse vocabulary, or O when it has no owner."""
    prefix, node = label.split("-", 1)
    owner = owner_of_fine_label.get(node)
    return UNION_OUTSIDE_LABEL if owner is None else f"{prefix}-{owner}"


def union_marginal_correction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    union_groups: torch.Tensor,
    union_fine_trusted: torch.Tensor,
    membership: torch.Tensor,
    fine_weight: float,
) -> torch.Tensor:
    """The union objective minus the plain fine token cross-entropy.

    Added to the stock token CE this yields the union objective exactly, and it is
    written as the difference so that a token whose target does not change
    contributes nothing at all: a batch with no union-supervised token and full
    fine weight returns an exact zero, leaving the stock loss bitwise untouched.

    A union-supervised token replaces its fine term by
    ``-(logsumexp_{k in members} z[k] - logsumexp_all z)``; a fine-labelled token
    inside a group blends the two by ``fine_weight``. Both logsumexp terms are
    taken in float32 over the (typically bf16) logits, and the whole correction is
    normalized by this batch's supervised-token count, which is the denominator the
    token heads in use here divide by (an unweighted ``CrossEntropyLoss`` mean over
    unmasked positions). A head that normalized its own loss differently would need
    this term rescaled to match it.
    """
    if not 0.0 <= fine_weight <= 1.0:
        raise ValueError(f"union fine weight {fine_weight} is not a convex blend weight")
    supervised = labels != -100
    grouped = (union_groups >= 0) & supervised
    marginal_only = grouped & (union_fine_trusted == 0)
    changing = marginal_only if fine_weight == 1.0 else grouped
    if not torch.any(changing):
        return logits.sum() * 0.0
    supervised_tokens = int(torch.count_nonzero(supervised))
    selected = logits[changing].float()
    denominator = torch.logsumexp(selected, dim=-1)
    members = membership.to(selected.device)[union_groups[changing]]
    numerator = torch.logsumexp(selected.masked_fill(~members, float("-inf")), dim=-1)
    marginal_loss = denominator - numerator
    fine_loss = denominator - selected.gather(1, labels[changing].unsqueeze(1)).squeeze(1)
    union_share = torch.where(
        union_fine_trusted[changing] > 0,
        torch.full_like(marginal_loss, 1.0 - fine_weight),
        torch.ones_like(marginal_loss),
    )
    return torch.sum(union_share * (marginal_loss - fine_loss)) / supervised_tokens


def parse_union_fine_weight_schedule(spec: str) -> tuple[str, float, float]:
    """Parse ``constant:W`` or ``linear:START:END`` into (kind, start, end)."""
    fields = str(spec).split(":")
    kind = fields[0]
    if kind == "constant" and len(fields) == 2:
        weights = (fields[1], fields[1])
    elif kind == "linear" and len(fields) == 3:
        weights = (fields[1], fields[2])
    else:
        raise ValueError(
            f"union fine-weight schedule {spec!r} is neither 'constant:W' nor 'linear:START:END'"
        )
    parsed = []
    for weight in weights:
        try:
            value = float(weight)
        except ValueError:
            raise ValueError(f"union fine-weight schedule {spec!r} has a non-numeric weight") from None
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"union fine-weight schedule {spec!r}: {value} is outside the convex range [0, 1]"
            )
        parsed.append(value)
    return kind, parsed[0], parsed[1]


def union_fine_weight(schedule: tuple[str, float, float], step: int, horizon: int) -> float:
    """The weight fine-labelled rows keep on their own label at this step."""
    kind, start, end = schedule
    if kind == "constant":
        return start
    if horizon <= 0:
        raise ValueError("a linear union fine-weight fade requires a positive optimizer-step horizon")
    return start + (end - start) * min(1.0, max(0.0, step / horizon))


def effective_training_step(trainer) -> int:
    """Return the original-trajectory step, including a post-selection lap offset."""
    return int(trainer.state.global_step) + int(getattr(getattr(trainer, "args", None), "pii_step_offset", 0))


class UnionMarginalLossMixin:
    """Supervise coarse-labelled rows through the fine head's class unions.

    The correction is applied in training and in evaluation alike: a coarse row's
    fine label is latent, so an evaluation loss that scored the union head's
    representative row instead would not be the objective's own held-out value.
    """

    union_groups = None
    union_fine_weight_schedule = CONSTANT_UNION_FINE_WEIGHT
    union_fine_weight_horizon = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        union_groups = inputs.pop("union_groups", None)
        union_fine_trusted = inputs.pop("union_fine_trusted", None)
        if self.union_groups is None:
            raise RuntimeError("union-marginal supervision requires a bound member document")
        if union_groups is None or union_fine_trusted is None:
            raise ValueError(
                "union-marginal supervision requires the per-token union targets SpanDataset emits; "
                "this batch carries none"
            )
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        correction = union_marginal_correction(
            outputs.logits,
            inputs["labels"],
            union_groups,
            union_fine_trusted,
            self.union_groups.membership,
            union_fine_weight(
                self.union_fine_weight_schedule,
                effective_training_step(self),
                self.union_fine_weight_horizon,
            ),
        )
        loss = loss + correction.to(loss.dtype)
        return (loss, outputs) if return_outputs else loss


def argmax_own_head_logits(logits, labels):
    """Predicted row per token, from the model's own label space.

    A dual-head model returns both heads' logits, and the tuple's first entry
    is the model's own vocabulary -- the one every span metric, id2label and
    decode path in this trainer is written against.
    """
    del labels
    own = logits[0] if isinstance(logits, (tuple, list)) else logits
    return own.argmax(dim=-1)


def argmax_new_head_logits(logits, labels):
    """Predicted row per token from the new ontology's head.

    Two heads are returned while both exist and the added head is the second.
    Once the transition retires the old head the model is a single-head tagger
    whose own vocabulary *is* the new ontology, so its one output is what this
    reads -- the metric keeps measuring the same thing across that boundary.
    """
    del labels
    if not isinstance(logits, (tuple, list)):
        return logits.argmax(dim=-1)
    if len(logits) < 2:
        raise ValueError("new-head metrics need a model that returns both heads' logits")
    return logits[1].argmax(dim=-1)


class DualHeadLossMixin:
    """Train the new ontology from every old- and new-space row.

    This mixin *replaces* the objective rather than correcting it. During a
    transition, the loss is the convex blend the correctness map defines over
    two heads, so the base trainer's own token cross-entropy would be duplicate
    supervision on the old head. After retirement, or in a later mapped
    single-head stage, only the new head remains and old-space rows reach it
    through the same allowed-set marginal. Trusted O-token weighting is
    implemented inside that marginal objective; other token-loss reshaping
    remains incompatible.

    Training uses the transition weight for the current step; evaluation uses
    one fixed weight (``--dual-head-eval-weight``, the endpoint objective by
    default) so that eval_loss stays comparable across a run whose training
    objective is deliberately moving.

    Once the schedule has taken the old head's weight to zero for good, the old
    head is retired out of the model (see ``DualHeadRetirementCallback``) and
    the objective here is the surviving head's alone. That is the same number
    the blend gave at weight zero -- old-ontology rows still supervise the new
    head through the map -- computed without the retired head's forward pass.
    """

    correctness_map = None
    dual_head_schedule = ("constant", 1.0, 1.0, 1.0)
    dual_head_horizon = 0
    dual_head_eval_weight = 0.0
    dual_head_retired = False
    o_token_loss_weight = 1.0
    complete_presence_loss_weight = 0.0
    complete_family_loss_weight = 0.0
    family_label_groups = None
    complete_boundary_loss_weight = 0.0
    partial_o_loss_weight = 0.0
    partial_entity_pu_loss_weight = 0.0
    partial_entity_pu_positive_margin = None
    partial_expected_entity_ratio_loss_weight = 0.0
    partial_expected_entity_ratio_lower_width = 0.1
    partial_parent_presence_kl_weight = 0.0
    partial_parent_presence_teacher = None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Attach the latest mapped-objective components to trainer log events."""
        telemetry = getattr(self, "dual_head_telemetry", None)
        if telemetry:
            for name, value in telemetry.items():
                logs.setdefault(name, float(value))
        return super().log(logs, start_time)

    def apply_entity_dice(
        self,
        loss,
        logits,
        new_labels,
        old_labels,
        telemetry,
    ):
        """Blend Dice on the product head using direct-or-mapped supervision."""
        weight = float(getattr(self, "entity_dice_weight", 0.0))
        if not weight:
            return loss
        labels = head_entity_labels(
            new_labels,
            old_labels,
            own_o_label_id=self.correctness_map.new_outside_id,
            cross_o_label_id=self.correctness_map.old_outside_id,
            cross_membership=self.correctness_map.old_to_new,
        )
        dice = entity_dice_loss(logits, labels)
        telemetry.update(
            {
                "dual_entity_dice_loss": float(dice.detach()),
                "dual_entity_dice_weight": weight,
                "dual_entity_dice_tokens": float((labels != -100).sum()),
            }
        )
        return (loss + weight * dice) / (1.0 + weight)

    def bioes_loss_options(self, logits, new_labels, *, training):
        """Keep structure margins and risk weighting identical after head retirement."""
        if not training:
            return {}
        risk_gate = None
        if getattr(self, "bioes_risk_weight", 0.0):
            risk_gate = span_risk_gate(
                logits,
                new_labels,
                self.bioes_incompatible,
                self.bioes_continuation,
                outside_id=self.correctness_map.new_outside_id,
                threshold=self.bioes_risk_threshold,
                scale=self.bioes_risk_scale,
                weight=self.bioes_risk_weight,
            )
        return {
            "structure_cost_matrix": getattr(self, "bioes_structure_cost_matrix", None),
            "mapped_structure_cost_rows": getattr(self, "bioes_mapped_cost_rows", None),
            "transition_legality": getattr(self, "bioes_transition_legality", None),
            "illegal_flip_scale": getattr(self, "bioes_illegal_flip_scale", 1.0),
            "risk_gate": risk_gate,
        }

    def apply_presence_objectives(
        self,
        loss,
        logits,
        complete_presence_labels,
        complete_family_labels,
        complete_boundary_labels,
        consistency_mask,
        partial_entity_positive_mask,
        partial_entity_pu_group,
        partial_entity_pu_prior,
        partial_entity_ratio_group,
        partial_entity_ratio_prior,
        parent_logits,
        telemetry,
        logical_step_masses=None,
    ):
        """Blend product-head existence/boundary terms without changing evaluation."""
        complete_weight = float(getattr(self, "complete_presence_loss_weight", 0.0))
        family_weight = float(getattr(self, "complete_family_loss_weight", 0.0))
        boundary_weight = float(getattr(self, "complete_boundary_loss_weight", 0.0))
        partial_weight = float(getattr(self, "partial_o_loss_weight", 0.0))
        pu_weight = float(getattr(self, "partial_entity_pu_loss_weight", 0.0))
        ratio_weight = float(getattr(self, "partial_expected_entity_ratio_loss_weight", 0.0))
        parent_weight = float(getattr(self, "partial_parent_presence_kl_weight", 0.0))
        if not any(
            (
                complete_weight,
                family_weight,
                boundary_weight,
                partial_weight,
                pu_weight,
                ratio_weight,
                parent_weight,
            )
        ):
            return loss
        weighted = loss
        normalizer = 1.0
        if complete_weight:
            if complete_presence_labels is None:
                raise ValueError(
                    "complete entity-presence loss requires SpanDataset complete_presence_labels"
                )
            complete_tokens = int(torch.count_nonzero(complete_presence_labels != -100).item())
            if complete_tokens:
                complete_loss = entity_presence_loss(
                    logits,
                    complete_presence_labels,
                    self.correctness_map.new_outside_id,
                )
                telemetry.update(
                    {
                        "dual_complete_presence_loss": float(complete_loss.detach()),
                        "dual_complete_presence_weight": complete_weight,
                        "dual_complete_presence_tokens": float(complete_tokens),
                    }
                )
                complete_contribution = logical_step_mean_contribution(
                    complete_loss,
                    complete_tokens,
                    logical_step_masses,
                    "complete_presence",
                )
                weighted = weighted + complete_weight * complete_contribution
                if logical_step_masses is None:
                    normalizer += complete_weight
        if family_weight:
            if complete_family_labels is None:
                raise ValueError("complete family-presence loss requires SpanDataset complete_family_labels")
            if self.family_label_groups is None:
                raise ValueError("complete family-presence loss requires bound family label groups")
            family_tokens = int(torch.count_nonzero(complete_family_labels != -100).item())
            if family_tokens:
                family_loss = family_presence_loss(
                    logits,
                    complete_family_labels,
                    self.family_label_groups,
                )
                telemetry.update(
                    {
                        "dual_complete_family_loss": float(family_loss.detach()),
                        "dual_complete_family_weight": family_weight,
                        "dual_complete_family_tokens": float(family_tokens),
                    }
                )
                family_contribution = logical_step_mean_contribution(
                    family_loss,
                    family_tokens,
                    logical_step_masses,
                    "complete_family",
                )
                weighted = weighted + family_weight * family_contribution
                if logical_step_masses is None:
                    normalizer += family_weight
        if boundary_weight:
            if complete_boundary_labels is None:
                raise ValueError("complete boundary loss requires SpanDataset complete_boundary_labels")
            boundary_tokens = int(torch.count_nonzero(complete_boundary_labels != -100).item())
            if boundary_tokens:
                boundary_loss = annotated_boundary_loss(
                    logits,
                    complete_boundary_labels,
                    list(self.correctness_map.new_labels),
                )
                telemetry.update(
                    {
                        "dual_complete_boundary_loss": float(boundary_loss.detach()),
                        "dual_complete_boundary_weight": boundary_weight,
                        "dual_complete_boundary_tokens": float(boundary_tokens),
                    }
                )
                boundary_contribution = logical_step_mean_contribution(
                    boundary_loss,
                    boundary_tokens,
                    logical_step_masses,
                    "complete_boundary",
                )
                weighted = weighted + boundary_weight * boundary_contribution
                if logical_step_masses is None:
                    normalizer += boundary_weight
        if partial_weight:
            if consistency_mask is None:
                raise ValueError("partial O loss requires SpanDataset consistency_mask")
            partial_tokens = int(torch.count_nonzero(consistency_mask).item())
            if partial_tokens:
                unmarked_o_loss = partial_o_loss(
                    logits,
                    consistency_mask,
                    self.correctness_map.new_outside_id,
                )
                telemetry.update(
                    {
                        "dual_partial_o_loss": float(unmarked_o_loss.detach()),
                        "dual_partial_o_weight": partial_weight,
                        "dual_partial_o_tokens": float(partial_tokens),
                    }
                )
                partial_contribution = logical_step_mean_contribution(
                    unmarked_o_loss,
                    partial_tokens,
                    logical_step_masses,
                    "partial_o",
                )
                weighted = weighted + partial_weight * partial_contribution
                if logical_step_masses is None:
                    normalizer += partial_weight
        if pu_weight:
            if any(
                value is None
                for value in (
                    consistency_mask,
                    partial_entity_positive_mask,
                    partial_entity_pu_group,
                    partial_entity_pu_prior,
                )
            ):
                raise ValueError("partial entity PU loss requires SpanDataset PU masks and row metadata")
            pu = nonnegative_pu_entity_loss(
                logits,
                partial_entity_positive_mask,
                consistency_mask,
                partial_entity_pu_group,
                partial_entity_pu_prior,
                self.correctness_map.new_outside_id,
                positive_margin=self.partial_entity_pu_positive_margin,
            )
            if pu.groups:
                active_priors = partial_entity_pu_prior[partial_entity_pu_group >= 0]
                telemetry.update(
                    {
                        "dual_partial_entity_pu_loss": float(pu.loss.detach()),
                        "dual_partial_entity_pu_weight": pu_weight,
                        "dual_partial_entity_pu_positive_risk": float(pu.positive_risk.detach()),
                        "dual_partial_entity_pu_negative_risk": float(pu.negative_risk.detach()),
                        "dual_partial_entity_pu_unclipped_negative_risk": float(
                            pu.unclipped_negative_risk.detach()
                        ),
                        "dual_partial_entity_pu_positive_tokens": float(pu.positive_tokens),
                        "dual_partial_entity_pu_unlabeled_tokens": float(pu.unlabeled_tokens),
                        "dual_partial_entity_pu_groups": float(pu.groups),
                        "dual_partial_entity_pu_clipped_groups": float(pu.clipped_groups),
                        "dual_partial_entity_pu_prior_min": float(active_priors.min()),
                        "dual_partial_entity_pu_prior_max": float(active_priors.max()),
                        "dual_partial_entity_pu_positive_margin": (
                            -1.0
                            if self.partial_entity_pu_positive_margin is None
                            else float(self.partial_entity_pu_positive_margin)
                        ),
                    }
                )
                weighted = weighted + pu_weight * pu.loss
                normalizer += pu_weight
        if ratio_weight:
            if any(
                value is None
                for value in (
                    consistency_mask,
                    partial_entity_positive_mask,
                    partial_entity_ratio_group,
                    partial_entity_ratio_prior,
                )
            ):
                raise ValueError(
                    "partial expected entity-ratio loss requires SpanDataset masks and prior metadata"
                )
            ratio = expected_entity_ratio_loss(
                logits,
                partial_entity_positive_mask,
                consistency_mask,
                partial_entity_ratio_group,
                partial_entity_ratio_prior,
                self.correctness_map.new_outside_id,
                lower_width=self.partial_expected_entity_ratio_lower_width,
            )
            if ratio.tokens:
                telemetry.update(
                    {
                        "dual_partial_expected_entity_ratio_loss": float(ratio.loss.detach()),
                        "dual_partial_expected_entity_ratio_weight": ratio_weight,
                        "dual_partial_expected_entity_ratio_prediction": float(
                            ratio.predicted_ratio.detach()
                        ),
                        "dual_partial_expected_entity_ratio_lower": float(ratio.lower_ratio.detach()),
                        "dual_partial_expected_entity_ratio_upper": float(ratio.upper_ratio.detach()),
                        "dual_partial_expected_entity_ratio_tokens": float(ratio.tokens),
                        "dual_partial_expected_entity_ratio_rows": float(ratio.rows),
                    }
                )
                ratio_contribution = logical_step_mean_contribution(
                    ratio.loss,
                    ratio.tokens,
                    logical_step_masses,
                    "partial_expected_entity_ratio",
                )
                weighted = weighted + ratio_weight * ratio_contribution
                if logical_step_masses is None:
                    normalizer += ratio_weight
        if parent_weight:
            if consistency_mask is None:
                raise ValueError("parent-presence trust region requires SpanDataset consistency_mask")
            unlabeled_tokens = int(torch.count_nonzero(consistency_mask).item())
            if unlabeled_tokens:
                if parent_logits is None:
                    raise ValueError("parent-presence trust region requires parent product-head logits")
                parent_loss = parent_presence_kl_loss(
                    logits,
                    parent_logits,
                    consistency_mask,
                    self.correctness_map.new_outside_id,
                )
                telemetry.update(
                    {
                        "dual_partial_parent_presence_kl": float(parent_loss.detach()),
                        "dual_partial_parent_presence_kl_weight": parent_weight,
                        "dual_partial_parent_presence_kl_tokens": float(unlabeled_tokens),
                    }
                )
                parent_contribution = logical_step_mean_contribution(
                    parent_loss,
                    unlabeled_tokens,
                    logical_step_masses,
                    "partial_parent_presence",
                )
                weighted = weighted + parent_weight * parent_contribution
                if logical_step_masses is None:
                    normalizer += parent_weight
        return (
            weighted / logical_step_masses.primary_group_weight_sum
            if logical_step_masses is not None
            else weighted / normalizer
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logical_step_masses = (
            num_items_in_batch
            if model.training and isinstance(num_items_in_batch, LogicalStepObjectiveMasses)
            else None
        )
        if self.correctness_map is None:
            raise RuntimeError("dual-head supervision requires a bound correctness map")
        complete_presence_labels = inputs.pop("complete_presence_labels", None)
        complete_family_labels = inputs.pop("complete_family_labels", None)
        complete_boundary_labels = inputs.pop("complete_boundary_labels", None)
        consistency_mask = inputs.pop("consistency_mask", None)
        partial_entity_positive_mask = inputs.pop("partial_entity_positive_mask", None)
        partial_entity_pu_group = inputs.pop("partial_entity_pu_group", None)
        partial_entity_pu_prior = inputs.pop("partial_entity_pu_prior", None)
        partial_entity_ratio_group = inputs.pop("partial_entity_ratio_group", None)
        partial_entity_ratio_prior = inputs.pop("partial_entity_ratio_prior", None)
        primary_objective_weights = inputs.pop("primary_objective_weights", None)
        segmentation_prefix_masks = inputs.pop("segmentation_prefix_masks", None)
        secondary_labels = inputs.pop("secondary_labels", None)
        mapped_outside_allowed = inputs.pop("mapped_outside_allowed", None)
        if mapped_outside_allowed is not None and not self.dual_head_retired:
            raise ValueError("per-row mapped outside coverage requires a retired single head")
        if secondary_labels is None:
            raise ValueError(
                "dual-head supervision requires the new-space targets SpanDataset emits; "
                "this batch carries none"
            )
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        parent_logits = None
        if (
            model.training
            and self.partial_parent_presence_kl_weight
            and consistency_mask is not None
            and torch.any(consistency_mask)
        ):
            if self.partial_parent_presence_teacher is None:
                raise RuntimeError("parent-presence trust region has no bound parent model")
            with torch.no_grad():
                parent_outputs = self.partial_parent_presence_teacher(**inputs)
            parent_logits = parent_outputs.logits
        if getattr(outputs, "secondary_logits", None) is None:
            if not self.dual_head_retired:
                raise RuntimeError("dual-head supervision requires a model with a second head attached")
            o_token_weight = self.o_token_loss_weight if model.training else 1.0
            bioes_options = self.bioes_loss_options(outputs.logits, secondary_labels, training=model.training)
            terms = head_loss_terms(
                outputs.logits,
                secondary_labels,
                labels,
                self.correctness_map.old_to_new,
                segmentation_prefix_masks=segmentation_prefix_masks,
                label_names=self.correctness_map.new_labels,
                own_o_label_id=self.correctness_map.new_outside_id,
                cross_o_label_id=self.correctness_map.old_outside_id,
                o_token_weight=o_token_weight,
                token_objective_weights=primary_objective_weights,
                cross_outside_membership=mapped_outside_allowed,
                **bioes_options,
            )
            mean = terms.total / terms.normalizer if terms.covered else outputs.logits.sum() * 0.0
            inputs["labels"] = labels
            inputs["secondary_labels"] = secondary_labels
            self.dual_head_telemetry = {
                "dual_old_weight": 0.0,
                "dual_new_loss": float(mean.detach()),
                "dual_new_tokens": float(terms.covered),
                "dual_new_loss_normalizer": float(terms.normalizer.detach()),
                "dual_o_token_weight": float(o_token_weight),
                "dual_old_head_retired": 1.0,
            }
            risk_gate = bioes_options.get("risk_gate")
            if risk_gate is not None and torch.any(secondary_labels != -100):
                self.dual_head_telemetry["bioes_risk_gate_mean"] = float(
                    risk_gate[secondary_labels != -100].mean()
                )
            loss = logical_step_mean_contribution(
                mean,
                terms.normalizer,
                logical_step_masses,
                "primary",
            )
            if model.training:
                loss = self.apply_presence_objectives(
                    loss,
                    outputs.logits,
                    complete_presence_labels,
                    complete_family_labels,
                    complete_boundary_labels,
                    consistency_mask,
                    partial_entity_positive_mask,
                    partial_entity_pu_group,
                    partial_entity_pu_prior,
                    partial_entity_ratio_group,
                    partial_entity_ratio_prior,
                    parent_logits,
                    self.dual_head_telemetry,
                    logical_step_masses,
                )
                loss = self.apply_entity_dice(
                    loss,
                    outputs.logits,
                    secondary_labels,
                    labels,
                    self.dual_head_telemetry,
                )
            loss = loss.to(outputs.logits.dtype)
            return (loss, outputs) if return_outputs else loss
        weight = (
            transition_weight(
                self.dual_head_schedule,
                effective_training_step(self),
                self.dual_head_horizon,
            )
            if model.training
            else self.dual_head_eval_weight
        )
        bioes_options = self.bioes_loss_options(
            outputs.secondary_logits, secondary_labels, training=model.training
        )
        risk_gate = bioes_options.get("risk_gate")
        loss, telemetry = dual_head_loss(
            outputs.logits,
            outputs.secondary_logits,
            labels,
            secondary_labels,
            self.correctness_map,
            weight,
            self.o_token_loss_weight if model.training else 1.0,
            token_objective_weights=primary_objective_weights,
            segmentation_prefix_masks=segmentation_prefix_masks,
            **bioes_options,
        )
        if risk_gate is not None and torch.any(secondary_labels != -100):
            telemetry["bioes_risk_gate_mean"] = float(risk_gate[secondary_labels != -100].mean())
        inputs["labels"] = labels
        inputs["secondary_labels"] = secondary_labels
        self.dual_head_telemetry = telemetry
        if model.training:
            loss = self.apply_presence_objectives(
                loss,
                outputs.secondary_logits,
                complete_presence_labels,
                complete_family_labels,
                complete_boundary_labels,
                consistency_mask,
                partial_entity_positive_mask,
                partial_entity_pu_group,
                partial_entity_pu_prior,
                partial_entity_ratio_group,
                partial_entity_ratio_prior,
                parent_logits,
                telemetry,
            )
            loss = self.apply_entity_dice(
                loss,
                outputs.secondary_logits,
                secondary_labels,
                labels,
                telemetry,
            )
        loss = loss.to(outputs.logits.dtype)
        return (loss, outputs) if return_outputs else loss


class DualHeadRetirementCallback(TrainerCallback):
    """Delete the old ontology's head at the step its authority reaches zero.

    A zero blend weight already contributes no gradient, so this changes no
    number the objective produces. What it changes is everything around it: the
    retired head stops being computed forward, stops holding activations and
    optimizer moments, and stops occupying rows in whatever the run saves. A
    checkpoint written after this step is an ordinary single-head tagger over
    the new ontology -- a different kind of artifact from the two-head ones
    written while the transition was still running, and the one a finished run
    should export.

    Because the two phases produce different artifacts, best-checkpoint
    tracking restarts here: a two-head checkpoint from the fade is not a
    candidate for the exported model, however good its validation loss was.
    """

    def __init__(self, trainer, retirement_step: int):
        if retirement_step < 0:
            raise ValueError("a retirement step cannot be negative")
        self.trainer = trainer
        self.retirement_step = retirement_step
        self.retired = False

    def on_train_begin(self, args, state, control, **kwargs):
        if effective_training_step(self.trainer) >= self.retirement_step:
            self.retire(state)

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.retired and effective_training_step(self.trainer) >= self.retirement_step:
            self.retire(state)

    def retire(self, state) -> None:
        model = self.trainer.model
        optimizer = self.trainer.optimizer
        retired = model.retire_primary_head()
        evicted = 0
        if optimizer is not None:
            dropped = {id(parameter) for parameter in retired}
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) not in dropped]
            for parameter in retired:
                evicted += 1 if optimizer.state.pop(parameter, None) is not None else 0
        for parameter in retired:
            parameter.grad = None
        parameters = sum(parameter.numel() for parameter in retired)
        retired.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.trainer.dual_head_retired = True
        absolute_step = effective_training_step(self.trainer)
        model.config.pii_dual_head_retired_at_step = absolute_step
        state.best_metric = None
        state.best_model_checkpoint = None
        for callback in self.trainer.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                callback.early_stopping_patience_counter = 0
        self.retired = True
        print(
            f"TRAIN: dual-head transition complete at step {absolute_step}: "
            f"old head retired, {parameters} parameters and {evicted} optimizer states released; "
            f"checkpoints from here carry {model.config.num_labels} labels in the new ontology only",
            flush=True,
        )


class NewValidationDrawCallback(TrainerCallback):
    """Forget the inherited best checkpoint when the validation draw changed.

    Resuming trainer state into a fresh output directory is how a leg
    continues onto a different validation set. The resumed state carries the
    previous leg's best metric, which was measured on rows this leg will never
    evaluate. Left in place it wins every comparison, selection freezes on a
    checkpoint from the old draw, and the end-of-run reproduction check fails
    because the two numbers describe different evaluations. The same reset
    already happens when the objective changes at dual-head retirement.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        if state.best_metric is None and state.best_model_checkpoint is None:
            return control
        print(
            "TRAIN-VALIDATION: discarding the resumed best checkpoint "
            f"{state.best_model_checkpoint} at metric {state.best_metric!r}, which was selected "
            "on the previous validation draw and is not comparable with this one",
            flush=True,
        )
        state.best_metric = None
        state.best_model_checkpoint = None
        return control


def phase_restart_scheduler(args, optimizer, num_training_steps: int, restart_step: int):
    """The configured learning-rate schedule, run once per training phase.

    A run whose objective changes partway through is two pieces of training,
    and a single decay spends its remaining rate on the second one. Restarting
    gives the phase after the transition its own warmup and its own decay over
    the steps it actually has.

    The per-phase shape is whatever ``--sched`` names; it is read off a probe
    optimizer rather than reimplemented, so the two phases decay exactly the
    way an unsplit run of that length would.
    """
    if not 0 < restart_step < num_training_steps:
        raise ValueError(
            f"a learning-rate restart at step {restart_step} needs to fall inside the "
            f"{num_training_steps}-step horizon"
        )

    def phase_lambda(length: int):
        probe = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=probe,
            num_warmup_steps=args.get_warmup_steps(length),
            num_training_steps=length,
            scheduler_specific_kwargs=args.lr_scheduler_kwargs,
        )
        if not isinstance(scheduler, LambdaLR):
            raise ValueError(
                f"--sched {args.lr_scheduler_type} is not expressible as a per-step factor, "
                "so it cannot be restarted at the transition"
            )
        return scheduler.lr_lambdas[0]

    before = phase_lambda(restart_step)
    after = phase_lambda(num_training_steps - restart_step)

    def factor(step: int) -> float:
        return before(step) if step < restart_step else after(step - restart_step)

    return LambdaLR(optimizer, [factor] * len(optimizer.param_groups))


class TransitionRestartSchedulerMixin:
    """Restart the learning-rate schedule where the training phase changes."""

    lr_restart_step = 0

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None and self.lr_restart_step:
            self.lr_scheduler = phase_restart_scheduler(
                self.args,
                self.optimizer if optimizer is None else optimizer,
                num_training_steps,
                self.lr_restart_step,
            )
            self._created_lr_scheduler = True
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer=optimizer)


def fading_auxiliary_weight(initial_weight: float, fade_steps: int, step: int) -> float:
    """Linearly remove a training-only auxiliary over optimizer steps."""
    if initial_weight < 0 or fade_steps < 0 or step < 0:
        raise ValueError("auxiliary schedule values must be nonnegative")
    if not initial_weight:
        return 0.0
    if not fade_steps:
        raise ValueError("a positive auxiliary weight requires positive fade steps")
    return initial_weight * max(0.0, 1.0 - step / fade_steps)


class CharacterAuxiliaryLossMixin:
    """Add a fading tag loss from the character readout alone."""

    character_auxiliary_loss_weight = 0.0
    character_auxiliary_loss_fade_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weight = fading_auxiliary_weight(
            self.character_auxiliary_loss_weight,
            self.character_auxiliary_loss_fade_steps,
            effective_training_step(self),
        )
        if not model.training or not weight:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs["labels"]
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        character_logits = getattr(outputs, "character_logits", None)
        if character_logits is None:
            raise RuntimeError("character auxiliary loss requires character_logits")
        o_weight = float(getattr(self, "o_token_loss_weight", 1.0))
        auxiliary = o_weighted_token_classification_loss(
            character_logits,
            labels,
            int(getattr(self, "o_label_id", 0)),
            o_weight,
        )
        loss = loss + weight * auxiliary
        return (loss, outputs) if return_outputs else loss


def entity_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Sørensen-Dice loss for class-agnostic entity membership.

    BIOES type logits remain trained by the stock cross-entropy. This auxiliary
    term compares the summed non-O probability with entity-vs-O supervision,
    including fully supervised empty regions so false positives remain costly.
    """
    valid = labels != -100
    if not torch.any(valid):
        return logits.sum() * 0.0
    probabilities = F.softmax(logits.float(), dim=-1)
    entity_probabilities = (1.0 - probabilities[..., 0])[valid]
    entity_targets = (labels[valid] != 0).to(entity_probabilities.dtype)
    intersection = torch.sum(entity_probabilities * entity_targets)
    return 1.0 - (2.0 * intersection + smooth) / (
        torch.sum(entity_probabilities) + torch.sum(entity_targets) + smooth
    )


def symmetric_token_kl_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL between stochastic passes on trusted supervised tokens."""
    valid = labels != -100
    if not torch.any(valid):
        return (first_logits.sum() + second_logits.sum()) * 0.0
    first_log_probabilities = F.log_softmax(first_logits.float(), dim=-1)[valid]
    second_log_probabilities = F.log_softmax(second_logits.float(), dim=-1)[valid]
    first_probabilities = first_log_probabilities.exp()
    second_probabilities = second_log_probabilities.exp()
    second_to_first = F.kl_div(
        first_log_probabilities,
        second_probabilities,
        reduction="none",
    ).sum(dim=-1)
    first_to_second = F.kl_div(
        second_log_probabilities,
        first_probabilities,
        reduction="none",
    ).sum(dim=-1)
    return 0.5 * (second_to_first + first_to_second).mean()


class EntityDiceLossMixin:
    """Blend stock token CE with a class-agnostic entity Dice objective."""

    entity_dice_weight = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.entity_dice_weight or not model.training:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs["labels"]
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        dice_loss = entity_dice_loss(outputs.logits, labels)
        loss = (loss + self.entity_dice_weight * dice_loss) / (1.0 + self.entity_dice_weight)
        return (loss, outputs) if return_outputs else loss


class RDropLossMixin:
    """Regularize two dropout-perturbed passes with symmetric token KL."""

    rdrop_alpha = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.rdrop_alpha or not model.training:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        labels = inputs["labels"]
        first_loss, first_outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        second_loss, second_outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        supervised_loss = 0.5 * (first_loss + second_loss)
        consistency_loss = symmetric_token_kl_loss(first_outputs.logits, second_outputs.logits, labels)
        loss = supervised_loss + self.rdrop_alpha * consistency_loss
        return (loss, first_outputs) if return_outputs else loss


def loss_logit_gradient_norm(loss: torch.Tensor, logits: torch.Tensor) -> float:
    """Measure one objective's local gradient scale without accumulating grads."""
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    return float(torch.linalg.vector_norm(gradient.float()).detach().cpu())


def last_trainable_encoder_matrix(model):
    """Return a compact shared-encoder parameter for objective-scale telemetry."""
    matrices = [
        (name, parameter)
        for name, parameter in model.base_model.named_parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    encoder_layer_matrices = [
        item for item in matrices if "encoder.layer." in item[0] or "encoder.layers." in item[0]
    ]
    if encoder_layer_matrices:
        return encoder_layer_matrices[-1]
    non_pooler_matrices = [item for item in matrices if "pooler" not in item[0]]
    return non_pooler_matrices[-1] if non_pooler_matrices else (None, None)


def loss_parameter_gradient_norm(loss: torch.Tensor, parameter: torch.Tensor) -> float:
    """Measure one loss's gradient on a shared parameter without accumulating it."""
    gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
    if gradient is None:
        return 0.0
    return float(torch.linalg.vector_norm(gradient.float()).detach().cpu())


def span_metrics(eval_prediction, id2label):
    """Micro exact typed span metrics for Trainer's held-out windows."""
    predictions_or_logits, labels = eval_prediction
    predicted = (
        predictions_or_logits
        if predictions_or_logits.ndim == labels.ndim
        else predictions_or_logits.argmax(axis=-1)
    )
    true_positives = predictions = gold = 0
    invalid_constraints = constraints = invalid_sequences = sequence_count = 0
    for predicted_row, label_row in zip(predicted, labels):
        valid = label_row != -100
        predicted_ids = predicted_row[valid]
        predicted_spans = decode_bioes_spans(predicted_ids, id2label)
        gold_spans = decode_bioes_spans(label_row[valid], id2label)
        predicted_labels = [id2label[int(label_id)] for label_id in predicted_ids]
        row_invalid, row_constraints = count_bioes_violations(predicted_labels)
        invalid_constraints += row_invalid
        constraints += row_constraints
        invalid_sequences += int(row_invalid > 0)
        sequence_count += 1
        true_positives += len(predicted_spans & gold_spans)
        predictions += len(predicted_spans)
        gold += len(gold_spans)
    precision = true_positives / predictions if predictions else 0.0
    recall = true_positives / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "span_true_positives": true_positives,
        "span_predictions": predictions,
        "span_gold": gold,
        "raw_bioes_invalid_constraints": invalid_constraints,
        "raw_bioes_constraints": constraints,
        "raw_bioes_invalid_rate": invalid_constraints / constraints if constraints else 0.0,
        "raw_bioes_sequences_with_invalid": invalid_sequences,
        "raw_bioes_sequences": sequence_count,
        "raw_bioes_sequence_invalid_rate": invalid_sequences / sequence_count if sequence_count else 0.0,
    }


def dual_head_span_metrics(eval_prediction, id2label, new_rows=()):
    """Span metrics for the added head, over the rows annotated in its ontology.

    A row annotated in the incumbent ontology carries no new-space target on its
    entity tokens -- that supervision reaches the new head as an allowed set, not
    a label -- so scoring it here would count every prediction it makes as a
    false positive. Those rows are excluded and the incumbent head keeps its own
    native evaluation. ``new_rows`` is one flag per evaluated row, in evaluation
    order.
    """
    predictions, labels = eval_prediction
    if len(new_rows) != len(labels):
        raise ValueError(
            f"new-ontology evaluation needs one label space per evaluated row: {len(new_rows)} "
            f"declared against {len(labels)} evaluated"
        )
    selected = np.asarray(new_rows, dtype=bool)
    if not selected.any():
        raise ValueError(
            "dual-head validation has no row annotated in the new ontology, so the added head "
            "has nothing to be scored against"
        )
    return span_metrics((predictions[selected], labels[selected]), id2label)


def build_dual_head_span_metric_report(correctness_map, rows):
    """Build hard v2 metrics only when validation carries hard v2 labels.

    Old-space rows provide valid mapped marginal loss but no unique hard target
    for exact v2 span scoring. An all-old validation set therefore remains a
    valid loss-only selector for a historical suffix stage.
    """
    new_rows = [row.get("label_space", FINE_LABEL_SPACE) == NEW_LABEL_SPACE for row in rows]
    if not any(new_rows):
        return None
    return partial(
        dual_head_span_metrics,
        id2label=dict(enumerate(correctness_map.new_labels)),
        new_rows=new_rows,
    )


def union_span_metrics(eval_prediction, id2label, projected_id2label, fine_rows=()):
    """Span metrics in the coarse space, plus native fine metrics where they exist.

    A union head is decoded fine and projected, exactly as the released fine
    checkpoint is scored on a coarse evaluation view: each decoded span's fine
    class is mapped through the member inverse -- the same fallback map the corpus
    adapter inverted to build the groups -- and a fine class owned by no coarse
    class projects to O, which is what that fallback says it is. Gold comes through
    the same map, so a coarse-labelled row's latent fine target scores as its own
    class.

    Validation rows whose labels are native fine annotations are additionally
    scored in the fine vocabulary under ``v1_span_*``, keeping the checkpoint's
    native tagger quality in view while it is moved toward the coarse space.
    ``fine_rows`` is one flag per evaluated row, in evaluation order.
    """
    metrics = span_metrics(eval_prediction, projected_id2label)
    if not any(fine_rows):
        return metrics
    predictions_or_logits, labels = eval_prediction
    if len(fine_rows) != len(labels):
        raise ValueError(
            f"native fine evaluation needs one label space per evaluated row: {len(fine_rows)} "
            f"declared against {len(labels)} evaluated"
        )
    selected = np.asarray(fine_rows, dtype=bool)
    native = span_metrics((predictions_or_logits[selected], labels[selected]), id2label)
    metrics.update({f"v1_{name}": value for name, value in native.items()})
    return metrics


def project_deferred_references(row: dict) -> dict:
    """Keep source annotations separately; derive supervision for the remaining types.

    Complete rows provide O outside retained spans; partial rows retain unknown
    background. Reference-carried refinements have no targets in this view.
    """
    kept = [i for i, span in enumerate(row["spans"]) if span[2] not in DEFERRED_REFERENCE_TYPES]
    projected = {
        **row,
        "spans": [row["spans"][i] for i in kept],
        "deferred_reference_spans": [s for s in row["spans"] if s[2] in DEFERRED_REFERENCE_TYPES],
    }
    if "primary_span_objective_weights" in row:
        projected["primary_span_objective_weights"] = [row["primary_span_objective_weights"][i] for i in kept]
    for field in ("predicate_spans", "subclass_spans"):
        if field in row:
            projected[field] = [span for span in row[field] if span["type"] not in DEFERRED_REFERENCE_TYPES]
    return projected


def window_records(
    path,
    max_chars,
    sampling_pool=None,
    *,
    defer_references=False,
    context_field=None,
    tokenizer=None,
    max_tokens=None,
):
    """Pre-window docs into verbatim slices without cutting labeled spans."""
    out = []
    for line_number, line in enumerate(open(path), 1):
        r = json.loads(line)
        if not defer_references:
            r = project_reference_row(r)
        supervision = r.get("supervision", COMPLETE_SUPERVISION)
        if supervision not in SUPERVISION_MODES:
            raise ValueError(f"{path}:{line_number}: unsupported supervision mode {supervision!r}")
        primary_span_objective_weights = r.get("primary_span_objective_weights")
        if primary_span_objective_weights is not None:
            if not isinstance(primary_span_objective_weights, list) or len(
                primary_span_objective_weights
            ) != len(r["spans"]):
                raise ValueError(
                    f"{path}:{line_number}: primary_span_objective_weights must align one-to-one with spans"
                )
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
                for weight in primary_span_objective_weights
            ):
                raise ValueError(
                    f"{path}:{line_number}: primary span objective weights must be finite and nonnegative"
                )
        ignored_spans = r.get("ignored_spans", [])
        segmentation_spans = [*r["spans"], *ignored_spans]
        segmentation_spans.extend((span["start"], span["end"]) for span in r.get("predicate_spans", []))
        segmentation_spans.extend(
            (span["carrier_start"], span["carrier_end"]) for span in r.get("subclass_spans", [])
        )
        for off, seg in segments(
            r["text"], max_chars, segmentation_spans, tokenizer=tokenizer, max_tokens=max_tokens
        ):
            end = off + len(seg)
            spans = []
            window_primary_span_objective_weights = []
            for span_index, (s, e, node) in enumerate(r["spans"]):
                cs, ce = max(s, off), min(e, end)
                if cs < ce:
                    if cs != s or ce != e:
                        raise AssertionError(
                            f"{path}:{line_number}: window [{off}, {end}) cuts span [{s}, {e})"
                        )
                    spans.append([cs - off, ce - off, node])
                    if primary_span_objective_weights is not None:
                        window_primary_span_objective_weights.append(
                            primary_span_objective_weights[span_index]
                        )
            if supervision == ANNOTATED_SPANS_ONLY and not spans:
                continue
            window = {
                "text": seg,
                "spans": spans,
                "lang": r.get("lang") or "<unknown>",
                "supervision": supervision,
            }
            if tokenizer is not None:
                window["training_window"] = {
                    "source_path": str(path),
                    "source_line_1based": line_number,
                    "source_id": r.get("id"),
                    "source_text_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
                    "start": off,
                    "end": end,
                    "max_tokens": max_tokens,
                }
            if context_field and (
                r.get(context_field) is not None
                or (tokenizer is not None and (off > 0 or end < len(r["text"])))
            ):
                context = r.get(context_field)
                if context is None:
                    context = {"before": "", "after": ""}
                # A target split moves its other pieces into unsupervised context.
                # Keep their verbatim adjacency before attaching external neighbors.
                if isinstance(context, str):
                    window[context_field] = "\n\n".join(part for part in (context, r["text"][:off]) if part)
                elif isinstance(context, dict):
                    try:
                        before, after, start = document_context_parts(context)
                    except ValueError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid document context {context_field!r}"
                        ) from error
                    window[context_field] = {
                        "before": "\n\n".join(part for part in (before, r["text"][:off]) if part),
                        "after": "\n\n".join(part for part in (r["text"][end:], after) if part),
                    }
                    # Only the window that opens the row opens the document.
                    if start and off == 0:
                        window[context_field][DOCUMENT_START] = True
                else:
                    raise ValueError(f"{path}:{line_number}: invalid document context {context_field!r}")
            for provenance_field in (
                "annotator",
                "src",
                "source",
                "mix_source",
                "surface_origin",
                "predicate_seed",
                "seed_id",
                PROJECTION_FIELD,
            ):
                if provenance_field in r:
                    window[provenance_field] = r[provenance_field]
            # Part of the supervision contract rather than provenance: it says
            # which vocabulary this window's span labels are in. Absent means the
            # file is in the head's own fine vocabulary throughout.
            if "label_space" in r:
                window["label_space"] = r["label_space"]
            if "unknown_primary_types" in r:
                window["unknown_primary_types"] = r["unknown_primary_types"]
            if primary_span_objective_weights is not None:
                window["primary_span_objective_weights"] = window_primary_span_objective_weights
            rebased_ignored = []
            for start, stop, ignored_type in ignored_spans:
                if start < end and off < stop:
                    if start < off or stop > end:
                        raise AssertionError(
                            f"{path}:{line_number}: window [{off}, {end}) cuts ignored span [{start}, {stop})"
                        )
                    rebased_ignored.append([start - off, stop - off, ignored_type])
            if rebased_ignored:
                window["ignored_spans"] = rebased_ignored
            rebased_predicates = []
            for predicate_span in r.get("predicate_spans", []):
                start = predicate_span["start"]
                stop = predicate_span["end"]
                if start < end and off < stop:
                    if start < off or stop > end:
                        raise AssertionError(
                            f"{path}:{line_number}: window [{off}, {end}) cuts predicate span "
                            f"[{start}, {stop})"
                        )
                    rebased_predicate = {
                        "start": start - off,
                        "end": stop - off,
                        "type": predicate_span["type"],
                        "attrs": {
                            channel: [
                                [interval_start - off, interval_end - off]
                                for interval_start, interval_end in intervals
                            ]
                            for channel, intervals in predicate_span["attrs"].items()
                        },
                    }
                    if "objective_weights" in predicate_span:
                        rebased_predicate["objective_weights"] = predicate_span["objective_weights"]
                    rebased_predicates.append(rebased_predicate)
            if rebased_predicates:
                window["predicate_spans"] = rebased_predicates
            rebased_subclasses = []
            for component in r.get("subclass_spans", []):
                carrier_start = component["carrier_start"]
                carrier_end = component["carrier_end"]
                if carrier_start < end and off < carrier_end:
                    if carrier_start < off or carrier_end > end:
                        raise AssertionError(
                            f"{path}:{line_number}: window [{off}, {end}) cuts subclass carrier "
                            f"[{carrier_start}, {carrier_end})"
                        )
                    rebased_component = {
                        "carrier_start": carrier_start - off,
                        "carrier_end": carrier_end - off,
                        "type": component["type"],
                        "start": component["start"] - off,
                        "end": component["end"] - off,
                        "family": component["family"],
                        "value": component["value"],
                    }
                    for weight_field in ("objective_weight", "learning_weight"):
                        if weight_field in component:
                            rebased_component[weight_field] = component[weight_field]
                    rebased_subclasses.append(rebased_component)
            if rebased_subclasses:
                window["subclass_spans"] = rebased_subclasses
            for sampling_field, value in r.items():
                if sampling_field.startswith("sampling_"):
                    window[sampling_field] = value
            if sampling_pool is not None:
                existing_pool = window.get("sampling_pool")
                if existing_pool is not None and existing_pool != sampling_pool:
                    raise ValueError(
                        f"{path}:{line_number}: inline sampling_pool {existing_pool!r} "
                        f"conflicts with file pool {sampling_pool!r}"
                    )
                window["sampling_pool"] = sampling_pool
            out.append(project_deferred_references(window) if defer_references else window)
    return out


def row_source_name(row) -> str:
    """Return the supervision source identity used by PU prior assignments."""
    source = row.get("src") or row.get("source") or row.get("mix_source")
    if not isinstance(source, str) or not source:
        raise ValueError("PU rows require a nonempty src, source, or mix_source identity")
    return source


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash the exact tokenizer graph and special-token contract."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        implementation = backend.to_str()
    elif callable(getattr(tokenizer, "get_vocab", None)):
        implementation = sorted(tokenizer.get_vocab().items())
    else:
        raise ValueError("tokenizer cannot provide a stable vocabulary fingerprint")
    payload = {
        "class": type(tokenizer).__name__,
        "implementation": implementation,
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_entity_token_prior(path, tokenizer, max_len, max_chars):
    """Load and bind an immutable source/language entity-token prior receipt."""
    prior_path = Path(path)
    document = json.loads(prior_path.read_text(encoding="utf-8"))
    supported_schemas = {
        ENTITY_PU_PRIOR_SCHEMA: ENTITY_PU_PRIOR_SCHEMA_VERSION,
        ENTITY_TOKEN_PRIOR_SCHEMA: ENTITY_TOKEN_PRIOR_SCHEMA_VERSION,
    }
    schema = document.get("schema")
    if schema not in supported_schemas:
        raise ValueError(f"unsupported entity-token prior schema in {prior_path}")
    if document.get("schema_version") != supported_schemas[schema]:
        raise ValueError(f"unsupported entity-token prior schema version in {prior_path}")
    if document.get("unit") != "token":
        raise ValueError("entity-token prior must be estimated in token units")
    if document.get("max_length") != max_len:
        raise ValueError(
            f"entity-token prior max_length={document.get('max_length')!r} does not match training {max_len}"
        )
    if document.get("max_chars") != max_chars:
        raise ValueError(
            f"entity-token prior max_chars={document.get('max_chars')!r} does not match training {max_chars}"
        )
    observed_fingerprint = tokenizer_fingerprint(tokenizer)
    expected_fingerprint = document.get("tokenizer_fingerprint")
    if expected_fingerprint != observed_fingerprint:
        raise ValueError(
            "entity-token prior tokenizer fingerprint does not match the training tokenizer: "
            f"{expected_fingerprint!r} vs {observed_fingerprint!r}"
        )
    assignments = {}
    for entry in document.get("assignments", ()):
        source = entry.get("partial_source")
        language = entry.get("language")
        prior = entry.get("class_prior")
        if not isinstance(source, str) or not source or not isinstance(language, str) or not language:
            raise ValueError("entity-token prior assignments require nonempty partial_source and language")
        if not isinstance(prior, (int, float)) or not math.isfinite(prior) or not 0 < prior < 1:
            raise ValueError(f"entity-token prior for {source}/{language} must be strictly in (0, 1)")
        key = (source, language)
        if key in assignments:
            raise ValueError(f"duplicate entity-token prior assignment for {source}/{language}")
        assignments[key] = float(prior)
    if not assignments:
        raise ValueError("entity-token prior receipt has no source/language assignments")
    return document, assignments, hashlib.sha256(prior_path.read_bytes()).hexdigest()


def load_entity_pu_prior(path, tokenizer, max_len, max_chars):
    """Backward-compatible name for the shared entity-token prior loader."""
    return load_entity_token_prior(path, tokenizer, max_len, max_chars)


def resolve_training_windowing(requested: str | None, *, resume_config: object | None = None) -> str:
    """Preserve historical row order on exact resume; new stages protect token capacity."""
    saved = (
        getattr(resume_config, "pii_training_windowing", "character") if resume_config is not None else None
    )
    for value in (requested, saved):
        if value is not None and value not in {"character", "token-capacity"}:
            raise ValueError(f"unsupported training windowing: {value!r}")
    if saved is not None and requested is not None and saved != requested:
        raise ValueError("training windowing cannot change during exact resume; use --init-from-checkpoint")
    return saved or requested or "token-capacity"


def resolve_window_limits(max_chars: int, max_train_chars: int) -> tuple[int, int]:
    """Keep validation windowing stable while allowing presegmented training rows."""
    if max_chars <= 0:
        raise ValueError("--max-chars must be positive")
    if max_train_chars < 0:
        raise ValueError("--max-train-chars must be nonnegative")
    return max_train_chars or max_chars, max_chars


def window_mlm_replay_records(path, max_chars, tokenizer, max_tokens):
    """Window natural documents while preserving configured language mass.

    Every segment is a training example, so a document with ``N`` segments
    contributes ``N`` times the mass of a one-segment document before the
    per-language renormalization. The renormalization retains the packet's
    configured language targets despite language-specific length differences.
    """
    documents = []
    document_language_mass = Counter()
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            language = row.get("lang")
            weight = row.get("sampling_weight")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: MLM replay row needs nonempty string text")
            if not isinstance(language, str) or not language:
                raise ValueError(f"{path}:{line_number}: MLM replay row needs string lang")
            try:
                weight = float(weight)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: MLM replay row needs numeric sampling_weight"
                ) from error
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{path}:{line_number}: MLM replay sampling_weight must be positive")
            documents.append((row, weight))
            document_language_mass[language] += weight

    windows = []
    initial_segments = 0
    adaptive_token_splits = 0
    maximum_segment_tokens = 0
    initial_segments_by_language = Counter()
    final_segments_by_language = Counter()
    adaptive_token_splits_by_language = Counter()
    provisional_language_mass = Counter()
    for row, weight in documents:
        language = row["lang"]
        pending = list(segments(row["text"], max_chars, []))
        initial_segments += len(pending)
        initial_segments_by_language[language] += len(pending)
        accepted = []
        while pending:
            offset, text = pending.pop()
            token_count = len(tokenizer(text, truncation=False, verbose=False)["input_ids"])
            if token_count <= max_tokens:
                maximum_segment_tokens = max(maximum_segment_tokens, token_count)
                accepted.append((offset, text))
                continue
            child_max_chars = max(1, len(text) // 2)
            children = list(segments(text, child_max_chars, []))
            if len(children) < 2:
                raise ValueError(
                    f"MLM replay segment at {row.get('id')!r}+{offset} cannot be refined "
                    f"below {max_tokens} tokens"
                )
            pending.extend((offset + child_offset, child) for child_offset, child in children)
            adaptive_token_splits += 1
            adaptive_token_splits_by_language[language] += 1
        final_segments_by_language[language] += len(accepted)
        for offset, text in sorted(accepted):
            window = {
                "objective": "mlm",
                "text": text,
                "lang": language,
                "sampling_weight": weight,
                "source_document_id": row.get("id"),
                "source_offset": offset,
            }
            windows.append(window)
            provisional_language_mass[language] += weight
    if not windows:
        raise ValueError(f"{path}: MLM replay packet produced no windows")
    for window in windows:
        language = window["lang"]
        window["sampling_weight"] *= document_language_mass[language] / provisional_language_mass[language]
    return windows, {
        "documents": len(documents),
        "initial_segments": initial_segments,
        "initial_segments_by_language": dict(sorted(initial_segments_by_language.items())),
        "final_segments": len(windows),
        "final_segments_by_language": dict(sorted(final_segments_by_language.items())),
        "adaptive_token_splits": adaptive_token_splits,
        "adaptive_token_splits_by_language": dict(sorted(adaptive_token_splits_by_language.items())),
        "maximum_segment_tokens": maximum_segment_tokens,
        "max_chars": max_chars,
        "max_tokens": max_tokens,
        "segment_weighting": (
            "one source-row weight occurrence per final segment, then per-language mass renormalization; "
            "no token-count adjustment"
        ),
    }


def combine_objective_sampling_weights(
    supervised_weights,
    supervised_count,
    mlm_weights,
    mlm_probability,
):
    """Scale two positive row-weight distributions to an objective mixture."""
    if not 0 < mlm_probability < 1:
        raise ValueError("MLM replay probability must be strictly between zero and one")
    if supervised_count <= 0 or not mlm_weights:
        raise ValueError("joint objective sampling requires both supervised and MLM rows")
    supervised_weights = (
        [1.0] * supervised_count
        if supervised_weights is None
        else [float(value) for value in supervised_weights]
    )
    if len(supervised_weights) != supervised_count:
        raise ValueError("supervised sampling weights do not align with supervised rows")
    mlm_weights = [float(value) for value in mlm_weights]
    if any(not math.isfinite(value) or value <= 0 for value in supervised_weights + mlm_weights):
        raise ValueError("joint objective sampling weights must be finite and positive")
    supervised_total = sum(supervised_weights)
    mlm_total = sum(mlm_weights)
    return [
        *(value * (1 - mlm_probability) / supervised_total for value in supervised_weights),
        *(value * mlm_probability / mlm_total for value in mlm_weights),
    ]


def rebalance_family_share_within_groups(
    rows,
    weights,
    *,
    group_field,
    family_field,
    family_value,
    target_share,
):
    """Change one family's share without changing any group mass.

    A common odds multiplier is applied inside every group that contains
    both family and non-family rows. Groups containing only one side remain
    fixed. This makes a gold-share correction orthogonal to configured
    language mass instead of globally boosting gold-heavy English.
    """
    if len(rows) != len(weights):
        raise ValueError("family-share rows and weights must align")
    if not rows:
        raise ValueError("family-share rebalance requires at least one row")
    weights = [float(value) for value in weights]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("family-share weights must be finite and positive")
    if not 0 < target_share < 1:
        raise ValueError("family-share target must be strictly between zero and one")

    group_masses = {}
    for row, weight in zip(rows, weights, strict=True):
        group = str(row.get(group_field) or "<unknown>")
        family_mass, other_mass = group_masses.get(group, (0.0, 0.0))
        if row.get(family_field) == family_value:
            family_mass += weight
        else:
            other_mass += weight
        group_masses[group] = (family_mass, other_mass)

    total_mass = sum(weights)
    current_mass = sum(family_mass for family_mass, _ in group_masses.values())
    current_share = current_mass / total_mass
    target_mass = target_share * total_mass
    if math.isclose(current_mass, target_mass, rel_tol=1e-12, abs_tol=1e-15):
        return weights, {
            "prior_share": current_share,
            "target_share": target_share,
            "achieved_share": current_share,
            "odds_multiplier": 1.0,
        }

    minimum_mass = sum(
        family_mass for family_mass, other_mass in group_masses.values() if family_mass and not other_mass
    )
    maximum_mass = sum(
        family_mass + other_mass for family_mass, other_mass in group_masses.values() if family_mass
    )
    if not minimum_mass < target_mass < maximum_mass:
        raise ValueError(
            f"family-share target {target_share:.8f} is infeasible while preserving "
            f"{group_field} masses; feasible open interval is "
            f"({minimum_mass / total_mass:.8f}, {maximum_mass / total_mass:.8f})"
        )

    def mass_at(odds_multiplier):
        mass = 0.0
        for family_mass, other_mass in group_masses.values():
            if not family_mass or not other_mass:
                mass += family_mass
                continue
            scaled_family = odds_multiplier * family_mass
            mass += (family_mass + other_mass) * scaled_family / (scaled_family + other_mass)
        return mass

    if target_mass > current_mass:
        low, high = 1.0, 2.0
        while mass_at(high) < target_mass:
            high *= 2.0
    else:
        low, high = 0.5, 1.0
        while mass_at(low) > target_mass:
            low *= 0.5
    for _ in range(80):
        middle = (low + high) / 2.0
        if mass_at(middle) < target_mass:
            low = middle
        else:
            high = middle
    odds_multiplier = (low + high) / 2.0

    scales = {}
    for group, (family_mass, other_mass) in group_masses.items():
        if not family_mass or not other_mass:
            scales[group] = (1.0, 1.0)
            continue
        scaled_family = odds_multiplier * family_mass
        new_family_mass = (family_mass + other_mass) * scaled_family / (scaled_family + other_mass)
        scales[group] = (
            new_family_mass / family_mass,
            (family_mass + other_mass - new_family_mass) / other_mass,
        )

    adjusted = []
    for row, weight in zip(rows, weights, strict=True):
        group = str(row.get(group_field) or "<unknown>")
        family_scale, other_scale = scales[group]
        scale = family_scale if row.get(family_field) == family_value else other_scale
        adjusted.append(weight * scale)
    achieved_share = sum(
        weight for row, weight in zip(rows, adjusted, strict=True) if row.get(family_field) == family_value
    ) / sum(adjusted)
    return adjusted, {
        "prior_share": current_share,
        "target_share": target_share,
        "achieved_share": achieved_share,
        "odds_multiplier": odds_multiplier,
    }


def parse_train_pool_specs(specs, *, sampling_config=False):
    """Parse repeated ``NAME=PATH[:WEIGHT]`` pool declarations."""
    parsed = []
    seen = {"data"}
    for spec in specs:
        name, separator, rest = spec.partition("=")
        if not separator or not name.strip() or not rest.strip():
            raise ValueError(f"train pool must be NAME=PATH[:WEIGHT], got {spec!r}")
        name = name.strip()
        if name in seen:
            raise ValueError(f"duplicate or reserved train-pool name {name!r}")
        seen.add(name)
        path_text = rest
        weight = None
        candidate_path, colon, candidate_weight = rest.rpartition(":")
        if colon:
            try:
                weight = float(candidate_weight)
                path_text = candidate_path
            except ValueError:
                pass
        if not path_text:
            raise ValueError(f"train pool must be NAME=PATH[:WEIGHT], got {spec!r}")
        if sampling_config and weight is not None:
            raise ValueError("--sampling-config owns weights; omit :WEIGHT from --train-pool")
        if not sampling_config and weight is None:
            raise ValueError("--train-pool requires :WEIGHT unless --sampling-config is supplied")
        parsed.append((name, path_text, weight))
    if not sampling_config and parsed:
        extra_total = sum(weight for _name, _path, weight in parsed)
        if not 0.0 < extra_total < 1.0:
            raise ValueError("--train-pool weights must sum to a value strictly between zero and one")
    return parsed


def sampling_plan_for_rows(rows, config_path=""):
    """Return row weights, pool labels, and any deterministic derivation receipt.

    With no config, an inline ``sampling_weight`` must appear on every row and
    is interpreted directly (only relative values matter). A config instead
    names weighted pools by exact row-field matches; its optional
    ``example_factor_field`` supplies within-pool relative factors. An optional
    ``predicate_exposure_strata`` declaration first derives disjoint reserved
    rows for required ontology-v3 targets.
    """
    inline = ["sampling_weight" in row for row in rows]
    if not config_path:
        pool_keys = [str(row.get("sampling_pool") or "data") for row in rows]
        if not any(inline):
            return None, pool_keys, None
        if not all(inline):
            raise ValueError("inline sampling_weight must be present on every training row")
        return [float(row["sampling_weight"]) for row in rows], pool_keys, None

    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError(f"{config_path}: sampling config schema_version must be 1")
    derivation_receipt = None
    exposure_declaration = spec.get("predicate_exposure_strata")
    if exposure_declaration is not None:
        if not isinstance(exposure_declaration, dict):
            raise ValueError(f"{config_path}: predicate_exposure_strata must be an object")
        derivation_receipt = {
            "predicate_exposure_strata": assign_predicate_exposure_strata(
                rows,
                exposure_declaration,
            )
        }
    raw_pools = spec.get("pools")
    if not isinstance(raw_pools, list) or not raw_pools:
        raise ValueError(f"{config_path}: sampling config pools must be a non-empty list")
    pool_weights = {}
    pool_matches = []
    for index, pool in enumerate(raw_pools):
        if not isinstance(pool, dict):
            raise ValueError(f"{config_path}: pool {index} must be an object")
        name = pool.get("name")
        match = pool.get("match")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{config_path}: pool {index} needs a non-empty name")
        if name in pool_weights:
            raise ValueError(f"{config_path}: duplicate pool name {name!r}")
        if not isinstance(match, dict) or not match:
            raise ValueError(f"{config_path}: pool {name!r} needs a non-empty match object")
        pool_weights[name] = float(pool.get("weight"))
        pool_matches.append((name, match))

    labels = []
    for row_index, row in enumerate(rows):
        matches = [
            name
            for name, required in pool_matches
            if all(row.get(field) == value for field, value in required.items())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{config_path}: training row {row_index} matches {len(matches)} pools {matches}; expected one"
            )
        labels.append(matches[0])

    factor_field = spec.get("example_factor_field")
    factors = None
    if factor_field is not None:
        if not isinstance(factor_field, str) or not factor_field:
            raise ValueError(f"{config_path}: example_factor_field must be a non-empty string")
        missing = [index for index, row in enumerate(rows) if factor_field not in row]
        if missing:
            raise ValueError(
                f"{config_path}: example_factor_field {factor_field!r} is absent from row {missing[0]}"
            )
        factors = [float(row[factor_field]) for row in rows]
    return (
        example_weights_from_pools(labels, pool_weights, example_factors=factors),
        labels,
        derivation_receipt,
    )


def declared_subclass_exposure_targets(config_path: str) -> tuple[str, ...]:
    """Return categorical targets named by refinement-exposure sampling pools."""
    if not config_path:
        return ()
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    targets = set()
    for pool in spec.get("pools", []):
        match = pool.get("match", {})
        target = match.get("sampling_refinement_exposure")
        if isinstance(target, str) and target.startswith("subclass:"):
            targets.add(target.removeprefix("subclass:"))
    return tuple(sorted(targets))


def sampling_weights_and_pool_keys_for_rows(rows, config_path=""):
    """Return row weights and the named pool that owns each row."""
    weights, pool_keys, _receipt = sampling_plan_for_rows(rows, config_path)
    return weights, pool_keys


def sampling_weights_for_rows(rows, config_path=""):
    """Return direct or pool-compiled row weights, or ``None`` for uniform training."""
    weights, _pool_keys = sampling_weights_and_pool_keys_for_rows(rows, config_path)
    return weights


def expand_context_variants(
    rows, weights, pool_keys, context_field, configurations, configuration_weights=None
):
    """One sampler entry per distinct context variant, each with its share of the row's weight.

    Each configuration gets its normalized configuration weight's share of the
    row's weight (an equal share when configuration_weights is None). Variants that
    select identical context (a row with no previous sentence, say) merge into
    one entry, so a row's total weight is unchanged. Expanding before the
    length-windowed sampler gives every variant its own context-inclusive
    length for batching and an exact weight share, instead of a per-draw coin
    flip made after target-only length binning.
    """
    pool_keys = pool_keys if pool_keys is not None else [None] * len(rows)
    shares = configuration_weights or [1.0 / len(configurations)] * len(configurations)
    entries, entry_weights, entry_keys = [], [], []
    for row, weight, key in zip(rows, weights, pool_keys, strict=True):
        context = row.get(context_field)
        if context is None:
            entries.append(row)
            entry_weights.append(weight)
            entry_keys.append(key)
            continue
        variants = {}
        for offsets, share in zip(configurations, shares, strict=True):
            selected = select_document_context(context, offsets)
            marker = json.dumps(selected, sort_keys=True)
            variants.setdefault(marker, [selected, 0.0])[1] += share
        for selected, fraction in variants.values():
            entries.append({**row, context_field: selected})
            entry_weights.append(float(weight) * fraction)
            entry_keys.append(key)
    return entries, entry_weights, entry_keys


def _each_item_objective_masses(items, collate, objective_masses):
    return [{name: float(mass) for name, mass in objective_masses(collate([item])).items()} for item in items]


def item_objective_masses(dataset, collate, objective_masses, workers=8):
    """Every objective's mass for each dataset item, as logical-step normalization counts it.

    Each item goes through ``__getitem__`` and the training collator, and
    ``objective_masses`` is the trainer's own per-batch normalizer, so context,
    truncation, ignored spans, masking and objective weights match training.
    Masked positions, including every context token, carry no mass in any
    objective.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=workers,
        collate_fn=partial(_each_item_objective_masses, collate=collate, objective_masses=objective_masses),
    )
    masses = [item for batch in loader for item in batch]
    if len(masses) != len(dataset):
        raise RuntimeError(f"supervision audit covered {len(masses)} of {len(dataset)} items")
    return masses


def exclude_unsupervised_items(weights, pool_keys, masses):
    """Zero the sampling weight of items without supervised mass, keeping each pool's total.

    A draw that would have landed on such an item lands instead on a supervised
    item of the same pool, in proportion to their weights, so pool and branch
    shares are unchanged.
    """
    keys = pool_keys if pool_keys is not None else [None] * len(weights)
    total, kept = Counter(), Counter()
    for weight, key, mass in zip(weights, keys, masses, strict=True):
        total[key] += weight
        if mass > 0:
            kept[key] += weight
    empty = [key for key in total if total[key] > 0 and kept[key] == 0]
    if empty:
        raise ValueError(f"sampling pools with no supervised item: {empty}")
    scaled = [
        weight * total[key] / kept[key] if mass > 0 else 0.0
        for weight, key, mass in zip(weights, keys, masses, strict=True)
    ]
    excluded = [index for index, mass in enumerate(masses) if mass <= 0]
    return scaled, excluded


def parse_mlm_pool_probabilities(specs):
    """Parse repeated ``POOL=PROBABILITY`` physical-batch MLM overrides."""
    probabilities = {}
    for spec in specs:
        pool, separator, raw_probability = spec.partition("=")
        pool = pool.strip()
        if not separator or not pool or not raw_probability.strip():
            raise ValueError(f"MLM pool must be POOL=PROBABILITY, got {spec!r}")
        if pool in probabilities:
            raise ValueError(f"duplicate MLM pool override {pool!r}")
        try:
            probability = float(raw_probability)
        except ValueError as error:
            raise ValueError(f"MLM pool probability must be numeric, got {spec!r}") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"MLM pool probability must be in [0, 1], got {spec!r}")
        probabilities[pool] = probability
    return probabilities


def resolve_physical_batch_mlm_probabilities(
    pool_keys,
    weights,
    *,
    default_probability,
    overrides,
    forced_mlm_pools=(),
):
    """Resolve MLM probabilities only for pools with positive sampling mass."""
    if len(pool_keys) != len(weights):
        raise ValueError("sampling pool keys and weights must align")
    pool_mass = Counter()
    for pool, weight in zip(pool_keys, weights, strict=True):
        pool_mass[str(pool)] += float(weight)
    active_pools = {pool for pool, mass in pool_mass.items() if mass > 0.0}
    named_pools = set(overrides) | set(forced_mlm_pools)
    inactive = sorted(named_pools - active_pools)
    if inactive:
        raise ValueError(f"MLM pool names have no positive sampling mass: {inactive}")
    probabilities = {pool: float(overrides.get(pool, default_probability)) for pool in active_pools}
    for pool in forced_mlm_pools:
        probabilities[pool] = 1.0
    return probabilities, pool_mass


def summarize_weighted_values(rows, weights, field):
    """Expected sampler shares by a provenance/language field."""
    if weights is None:
        return None
    totals = Counter()
    for row, weight in zip(rows, weights):
        totals[str(row.get(field) or "<unknown>")] += float(weight)
    total = sum(totals.values())
    return {name: value / total for name, value in sorted(totals.items())}


def row_language(row):
    return row.get("lang") or "<unknown>"


def sample_rows_with_replacement(rows, count, rng):
    """Shuffle-cycle rows so every row is seen before a repeat within a cycle."""
    if count == 0:
        return []
    if not rows:
        raise ValueError(f"cannot sample {count} rows from an empty pool")
    indices = list(range(len(rows)))
    selected = []
    while len(selected) < count:
        rng.shuffle(indices)
        remaining = count - len(selected)
        selected.extend(rows[index] for index in indices[:remaining])
    return selected


def parse_language_shares(specs):
    shares = {}
    for spec in specs:
        language, separator, raw_share = spec.partition("=")
        if not separator or not language:
            raise ValueError(f"language share must be LANG=FRACTION, got {spec!r}")
        try:
            share = float(raw_share)
        except ValueError as error:
            raise ValueError(f"language share must be LANG=FRACTION, got {spec!r}") from error
        if not 0 < share < 1:
            raise ValueError(f"language share must be strictly between 0 and 1, got {spec!r}")
        if language in shares:
            raise ValueError(f"duplicate minimum language share for {language!r}")
        shares[language] = share
    if sum(shares.values()) > 1:
        raise ValueError(f"minimum language shares sum to more than 1: {sum(shares.values()):.6f}")
    return shares


def parse_partial_negative_sources(specs):
    """Parse repeated SOURCE=TYPE,TYPE declarations without label-vocabulary assumptions."""
    sources = {}
    for spec in specs:
        source, separator, raw_labels = spec.partition("=")
        source = source.strip()
        labels = tuple(label.strip() for label in raw_labels.split(",") if label.strip())
        if not separator or not source or not labels:
            raise ValueError(f"partial-negative source must be SOURCE=TYPE,TYPE, got {spec!r}")
        if source in sources:
            raise ValueError(f"duplicate partial-negative source {source!r}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate type in partial-negative source {source!r}: {labels}")
        sources[source] = labels
    return sources


def summarize_languages(rows):
    counts = Counter(row_language(row) for row in rows)
    total = len(rows)
    return {
        "windows": total,
        "counts": dict(sorted(counts.items())),
        "shares": {language: count / total for language, count in sorted(counts.items())} if total else {},
    }


def summarize_values(rows, field):
    counts = Counter(row.get(field) or "<unknown>" for row in rows)
    total = len(rows)
    return {
        "windows": total,
        "counts": dict(sorted(counts.items())),
        "shares": {value: count / total for value, count in sorted(counts.items())} if total else {},
    }


def replay_rows_at_probability(
    primary_rows,
    replay_rows,
    probability,
    seed,
    minimum_language_shares=None,
    minimum_replay_language_shares=None,
):
    """Select replay rows while satisfying final-mix and replay-tranche floors."""
    if not 0 < probability < 1:
        raise ValueError(f"replay probability must be strictly between 0 and 1, got {probability}")
    if not replay_rows:
        raise ValueError("replay corpus produced no training windows")
    target = round(len(primary_rows) * probability / (1 - probability))
    rng = random.Random(seed)
    minimum_language_shares = minimum_language_shares or {}
    minimum_replay_language_shares = minimum_replay_language_shares or {}
    if not minimum_language_shares and not minimum_replay_language_shares:
        return sample_rows_with_replacement(replay_rows, target, rng)
    if minimum_replay_language_shares and not target:
        raise ValueError("minimum replay language shares require a nonzero replay budget")

    final_count = len(primary_rows) + target
    primary_counts = Counter(row_language(row) for row in primary_rows)
    constrained_languages = set(minimum_language_shares) | set(minimum_replay_language_shares)
    replay_by_language = {
        language: [row for row in replay_rows if row_language(row) == language]
        for language in constrained_languages
    }
    required = {
        language: max(
            max(
                0,
                math.ceil(final_count * minimum_language_shares.get(language, 0)) - primary_counts[language],
            ),
            math.ceil(target * minimum_replay_language_shares.get(language, 0)),
        )
        for language in constrained_languages
    }
    if sum(required.values()) > target:
        raise ValueError(
            "minimum final/replay language shares require "
            f"{sum(required.values())} replay windows, but the replay budget is {target}"
        )

    selected = []
    for language in sorted(required):
        needed = required[language]
        if needed and not replay_by_language[language]:
            raise ValueError(
                f"minimum language share for {language!r} requires {needed} replay windows, "
                "but the replay corpus has none"
            )
        selected.extend(sample_rows_with_replacement(replay_by_language[language], needed, rng))

    # Preserve the established final-mix-floor behavior: after reserving its
    # required rows, do not spend residual budget on those languages unless
    # every replay language is constrained. Replay-tranche floors are only
    # lower bounds, so they remain eligible for natural residual sampling.
    final_mix_constrained_languages = set(minimum_language_shares)
    residual_pool = [row for row in replay_rows if row_language(row) not in final_mix_constrained_languages]
    if not residual_pool:
        residual_pool = replay_rows
    selected.extend(sample_rows_with_replacement(residual_pool, target - len(selected), rng))

    mixed_counts = Counter(row_language(row) for row in primary_rows + selected)
    for language, share in minimum_language_shares.items():
        actual = mixed_counts[language] / final_count
        if actual < share:
            raise AssertionError(f"{language} share {actual:.6f} fell below requested minimum {share:.6f}")
    replay_counts = Counter(row_language(row) for row in selected)
    for language, share in minimum_replay_language_shares.items():
        actual = replay_counts[language] / target
        if actual < share:
            raise AssertionError(
                f"{language} replay share {actual:.6f} fell below requested minimum {share:.6f}"
            )
    return selected


def token_classifier_head(model):
    """Return the conventional HF token-classification projection and its attribute name."""
    candidates = [
        name for name in ("classifier", "score") if isinstance(getattr(model, name, None), nn.Linear)
    ]
    if len(candidates) != 1:
        observed = {name: type(getattr(model, name, None)).__name__ for name in ("classifier", "score")}
        raise TypeError(f"expected exactly one linear token-classification head; got {observed}")
    name = candidates[0]
    return name, getattr(model, name)


class ExplicitMatmulLinear(nn.Linear):
    """Linear with a state-compatible forward that avoids cuBLAS-Lt dispatch."""

    def forward(self, inputs):
        shape = (*inputs.shape[:-1], self.out_features)
        output = torch.mm(inputs.reshape(-1, self.in_features), self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(shape)


def configure_stock_classifier_kernel(model, kernel):
    """Select an algebraically equivalent implementation for the stock affine."""
    if kernel not in {"framework", "explicit_mm"}:
        raise ValueError(f"unknown stock classifier kernel: {kernel!r}")
    classifier_name, classifier = token_classifier_head(model)
    if kernel == "explicit_mm" and not isinstance(classifier, ExplicitMatmulLinear):
        replacement = ExplicitMatmulLinear(
            classifier.in_features,
            classifier.out_features,
            bias=classifier.bias is not None,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        )
        replacement.load_state_dict(classifier.state_dict())
        replacement.train(classifier.training)
        setattr(model, classifier_name, replacement)
    model.config.pii_stock_classifier_kernel = kernel


def projected_warm_start_labels(source_label2id, target_label2id, source_schema=""):
    """Map source BIOES rows into the target inventory, optionally through the tagset ontology."""
    tagset = Tagset() if source_schema else None
    target_to_source = {}
    for source_label, source_index in source_label2id.items():
        target_label = source_label
        if source_label != "O" and tagset is not None:
            prefix, separator, source_type = source_label.partition("-")
            if not separator or prefix not in BOUNDARY_LABELS[1:]:
                raise ValueError(f"invalid source BIOES label: {source_label!r}")
            target_label = f"{prefix}-{tagset.project(source_schema, source_type)}"
        if target_label not in target_label2id:
            continue
        if target_label in target_to_source:
            raise ValueError(f"warm-start schema {source_schema!r} maps multiple rows to {target_label!r}")
        target_to_source[target_label] = int(source_index)
    return target_to_source


def projected_warm_start_cut_groups(source_label2id, target_label2id, source_cut):
    """Group canonical source BIOES rows by their reporting-cut target row."""
    tagset = Tagset()
    target_to_sources = {}
    for source_label, source_index in source_label2id.items():
        target_label = source_label
        if source_label != "O":
            prefix, separator, source_type = source_label.partition("-")
            if not separator or prefix not in BOUNDARY_LABELS[1:]:
                raise ValueError(f"invalid source BIOES label: {source_label!r}")
            target_label = f"{prefix}-{tagset.project_cut('canonical', source_type, source_cut)}"
        if target_label in target_label2id:
            target_to_sources.setdefault(target_label, []).append(int(source_index))
    return {
        target_label: sorted(source_indices) for target_label, source_indices in target_to_sources.items()
    }


def expand_output_labels(
    model,
    label_names,
    source_schema="",
    source_cut="",
    copy_shared_rows=True,
):
    """Replace the output map, optionally retaining semantically shared warm-start rows."""
    if source_schema and source_cut:
        raise ValueError("source_schema and source_cut are mutually exclusive")
    classifier_name, classifier = token_classifier_head(model)
    if not isinstance(classifier, nn.Linear):
        raise AssertionError("token_classifier_head returned a non-linear module")
    source_label2id = dict(model.config.label2id)
    replacement = nn.Linear(
        classifier.in_features,
        len(label_names),
        bias=classifier.bias is not None,
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    )
    nn.init.normal_(
        replacement.weight,
        mean=0.0,
        std=float(getattr(model.config, "initializer_range", 0.02)),
    )
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)

    target_label2id = {label: index for index, label in enumerate(label_names)}
    if not copy_shared_rows:
        transferred = {}
    elif source_cut:
        transferred = projected_warm_start_cut_groups(source_label2id, target_label2id, source_cut)
    else:
        transferred = {
            target_label: [source_index]
            for target_label, source_index in projected_warm_start_labels(
                source_label2id, target_label2id, source_schema
            ).items()
        }
    with torch.no_grad():
        for target_label, source_indices in transferred.items():
            target_index = target_label2id[target_label]
            replacement.weight[target_index].copy_(classifier.weight[source_indices].mean(dim=0))
            if replacement.bias is not None:
                replacement.bias[target_index].copy_(classifier.bias[source_indices].mean(dim=0))

    setattr(model, classifier_name, replacement)
    model.num_labels = len(label_names)
    model.config.num_labels = len(label_names)
    model.config.id2label = dict(enumerate(label_names))
    model.config.label2id = target_label2id
    model.config.pii_warm_start_label_schema = source_schema or None
    model.config.pii_warm_start_label_cut = source_cut or None
    model.config.pii_output_head_reinitialized = not copy_shared_rows
    model.config.pii_warm_start_copied_label_rows = len(transferred)
    model.config.pii_warm_start_source_rows_per_label = {
        target_label: len(source_indices) for target_label, source_indices in transferred.items()
    }
    if getattr(model.config, "pii_head_architecture", None):
        model.config.pii_head_parameters = model.task_head_parameter_count()
    return len(transferred)


def expand_mapped_single_head_successor(model, correctness_map, new_names):
    """Append successor rows only after proving the retired parent identity."""
    config = model.config
    own_names = [config.id2label[index] for index in range(config.num_labels)]
    if own_names == list(new_names):
        return len(own_names)
    parent_names = list(correctness_map.parent_new_labels)
    if not parent_names:
        raise ValueError("mapped single-head label expansion requires a versioned successor map")
    if own_names != parent_names:
        raise ValueError("mapped single-head checkpoint labels do not match the successor map's parent")
    expected_parent = {
        "pii_dual_head_map_sha256": correctness_map.parent_map_sha256,
        "pii_dual_head_ontology_sha256": correctness_map.parent_ontology_sha256,
    }
    mismatches = [
        f"{field}={getattr(config, field, None)!r} (checkpoint) vs {value!r} (successor parent)"
        for field, value in expected_parent.items()
        if getattr(config, field, None) != value
    ]
    if mismatches:
        raise ValueError("mapped single-head successor ancestry mismatch: " + "; ".join(mismatches))
    copied = expand_output_labels(model, new_names)
    if copied != len(parent_names):
        raise AssertionError(
            f"successor expansion copied {copied} rows, expected the complete {len(parent_names)}-row parent"
        )
    config.pii_successor_parent_map_sha256 = correctness_map.parent_map_sha256
    config.pii_successor_parent_ontology_sha256 = correctness_map.parent_ontology_sha256
    config.pii_successor_parent_output_rows = len(parent_names)
    config.pii_successor_appended_output_rows = len(new_names) - len(parent_names)
    config.pii_dual_head_map_sha256 = correctness_map.map_sha256
    config.pii_dual_head_ontology_sha256 = correctness_map.ontology_sha256
    return copied


def validate_warm_start_encoder(model, requested_model):
    """Reject a checkpoint from a different encoder shape/family."""
    requested = AutoConfig.from_pretrained(requested_model)
    fields = ("model_type", "hidden_size", "num_hidden_layers")
    mismatches = [
        f"{field}={getattr(model.config, field, None)!r} (checkpoint) vs "
        f"{getattr(requested, field, None)!r} (requested)"
        for field in fields
        if getattr(model.config, field, None) != getattr(requested, field, None)
    ]
    if mismatches:
        raise ValueError("warm-start encoder does not match --model: " + "; ".join(mismatches))


def freeze_encoder_parameters(model):
    """Freeze the pretrained encoder while leaving the task head trainable."""
    encoder = model.base_model
    if encoder is model:
        raise ValueError("token classifier does not expose a separate base encoder to freeze")
    encoder.requires_grad_(False)
    return {
        "encoder_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def successor_parent_output_rows(model) -> int:
    """Return the inherited classifier prefix declared by a successor checkpoint."""
    rows = getattr(model.config, "pii_successor_parent_output_rows", None)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("checkpoint does not declare a positive successor parent output-row count")
    _classifier_name, classifier = token_classifier_head(model)
    if rows >= classifier.out_features:
        raise ValueError(
            "successor parent output rows must be a strict classifier prefix: "
            f"rows={rows}, outputs={classifier.out_features}"
        )
    return rows


class FrozenClassifierPrefixCallback(TrainerCallback):
    """Keep inherited classifier rows bit-exact while appended rows train."""

    def __init__(self, model, rows: int) -> None:
        _classifier_name, classifier = token_classifier_head(model)
        if not 0 < rows < classifier.out_features:
            raise ValueError(
                f"frozen classifier prefix must be in [1, {classifier.out_features - 1}], got {rows}"
            )
        self.classifier = classifier
        self.rows = rows
        self.snapshots: dict[nn.Parameter, torch.Tensor] = {}
        self.gradient_hooks = []

    def _mask_prefix(self, gradient: torch.Tensor) -> torch.Tensor:
        masked = gradient.clone()
        masked[: self.rows].zero_()
        return masked

    def _restore(self) -> None:
        with torch.no_grad():
            for parameter, snapshot in self.snapshots.items():
                parameter[: self.rows].copy_(snapshot)

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        parameters = [self.classifier.weight]
        if self.classifier.bias is not None:
            parameters.append(self.classifier.bias)
        self.snapshots = {parameter: parameter[: self.rows].detach().clone() for parameter in parameters}
        self.gradient_hooks = [parameter.register_hook(self._mask_prefix) for parameter in parameters]

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        del args, state, control, kwargs
        if optimizer is None:
            raise RuntimeError("frozen classifier prefix requires the optimizer after every step")
        self._restore()
        unwrapped = getattr(optimizer, "optimizer", optimizer)
        for parameter in self.snapshots:
            for value in unwrapped.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    value[: self.rows].zero_()

    def on_train_end(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        self._restore()
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks = []


def freeze_encoder_except_top_layers(model, trainable_top_layers):
    """Freeze embeddings and lower encoder layers, leaving a trainable top stack."""
    encoder = model.base_model
    if encoder is model:
        raise ValueError("token classifier does not expose a separate base encoder to freeze")
    layer_stack = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layer_stack is None:
        layer_stack = getattr(encoder, "layers", None)
    if layer_stack is None:
        raise ValueError("base encoder does not expose an ordered layer stack")
    total_layers = len(layer_stack)
    if not 1 <= trainable_top_layers <= total_layers:
        raise ValueError(
            f"--trainable-top-encoder-layers must be in [1, {total_layers}], got {trainable_top_layers}"
        )
    encoder.requires_grad_(False)
    for layer in layer_stack[-trainable_top_layers:]:
        layer.requires_grad_(True)
    return {
        "encoder_layers": total_layers,
        "trainable_top_encoder_layers": trainable_top_layers,
        "frozen_encoder_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if not parameter.requires_grad
        ),
        "trainable_encoder_parameters": sum(
            parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_frozen_mlm_head(model_name, target_config):
    """Load the original pretrained MLM head as a fixed encoder-space anchor."""
    source = AutoModelForMaskedLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    mismatches = []
    for field in ("model_type", "hidden_size", "vocab_size"):
        source_value = getattr(source.config, field, None)
        target_value = getattr(target_config, field, None)
        if source_value != target_value:
            mismatches.append(f"{field}={source_value!r} (task model {target_value!r})")
    if mismatches:
        raise ValueError("MLM replay head does not match the task encoder: " + "; ".join(mismatches))
    head = getattr(source, "lm_head", None)
    if head is None:
        raise ValueError(f"{model_name} does not expose an lm_head supported by MLM replay")
    head.requires_grad_(False).eval()
    return head


def save_initialized_checkpoint(
    model,
    tokenizer,
    output_dir,
    provenance,
    *,
    allowed_existing=(),
):
    """Save the deterministic post-warm-start, pre-training state as a control."""
    output = Path(output_dir)
    unexpected = (
        []
        if not output.exists()
        else [path.name for path in output.iterdir() if path.name not in allowed_existing]
    )
    if unexpected:
        raise FileExistsError(
            f"initialized-only output directory contains unexpected entries: {output}: {unexpected}"
        )
    output.mkdir(parents=True, exist_ok=True)
    model.config.pii_initialized_only = True
    model.config.pii_initialization_control = provenance
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "pii_initialization_control.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_training_parameter_precision(
    requested: str | None, *, resume_config: object | None = None, evaluate_only: bool = False
) -> str:
    """Keep legacy resumes exact while new stages accumulate FP32 updates."""
    saved = (
        getattr(resume_config, "pii_training_parameter_precision", "legacy")
        if resume_config is not None
        else None
    )
    if resume_config is not None and saved not in {"float32", "legacy"}:
        raise ValueError(f"unsupported checkpoint training parameter precision: {saved!r}")
    if requested is not None and requested not in {"float32", "legacy"}:
        raise ValueError(f"unsupported training parameter precision: {requested!r}")
    if saved is not None and requested is not None and requested != saved:
        raise ValueError(
            "exact resume training parameter precision mismatch: "
            f"checkpoint={saved}, requested={requested}; "
            "use --init-from-checkpoint for a new stage"
        )
    return saved or requested or ("legacy" if evaluate_only else "float32")


def load_local_token_classifier(checkpoint, *, dtype: torch.dtype = torch.bfloat16):
    """Load either supported local task-head architecture without changing labels."""
    config = AutoConfig.from_pretrained(checkpoint)
    if getattr(config, "pii_head_architecture", None) in SUPPORTED_CONTINUOUS_CHARACTER_HEAD_ARCHITECTURES:
        return ContinuousCharacterForTokenClassification.from_local_checkpoint(
            checkpoint,
            dtype=dtype,
        )
    if getattr(config, "pii_head_architecture", None) == "concat_encoder_layers":
        return LayerConcatForTokenClassification.from_local_checkpoint(
            checkpoint,
            dtype=dtype,
        )
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        dtype=dtype,
    )
    configure_stock_classifier_kernel(
        model,
        getattr(config, "pii_stock_classifier_kernel", "framework"),
    )
    return model


def validate_dual_head_resume(
    model,
    correctness_map,
    primary_names,
    secondary_names,
    *,
    old_weight_schedule,
    initialization,
    eval_weight,
    retirement_step,
    lr_restart_step,
    horizon,
    o_token_loss_weight=1.0,
    entity_dice_loss_weight=0.0,
    partial_o_loss_weight=0.0,
    partial_entity_pu_loss_weight=0.0,
    partial_entity_pu_prior_sha256=None,
    partial_entity_pu_positive_margin=None,
    partial_parent_presence_kl_weight=0.0,
    complete_presence_loss_weight=0.0,
    complete_family_loss_weight=0.0,
    complete_boundary_loss_weight=0.0,
    projection_identity=None,
):
    """Validate and classify an exact dual-head trainer-state resume.

    A checkpoint after head retirement is intentionally a different model
    shape from the run's initialization: the new-ontology head has become the
    sole classifier. Reattaching a fresh head, or rebuilding the old head and
    relying on Trainer to overwrite it, loses that phase transition. Resume
    therefore loads the checkpoint's exact shape and proves that its saved
    objective identity is the one requested by the current command.
    """
    config = model.config
    if getattr(config, "pii_dual_head_mapped_single_head", False):
        raise ValueError(
            "this checkpoint belongs to a mapped single-head stage; resume it with "
            "--dual-head-mapped-single-head"
        )
    expected = {
        "pii_dual_head_map_sha256": correctness_map.map_sha256,
        "pii_dual_head_ontology_sha256": correctness_map.ontology_sha256,
        "pii_dual_head_old_weight_schedule": old_weight_schedule,
        "pii_dual_head_init": initialization,
        "pii_dual_head_eval_weight": eval_weight,
        "pii_dual_head_retirement_step": retirement_step,
        "pii_dual_head_lr_restart_step": lr_restart_step,
        "pii_o_token_loss_weight": o_token_loss_weight,
        "pii_entity_dice_loss_weight": entity_dice_loss_weight,
        "pii_partial_o_loss_weight": partial_o_loss_weight,
        "pii_partial_entity_pu_loss_weight": partial_entity_pu_loss_weight,
        "pii_partial_entity_pu_prior_sha256": partial_entity_pu_prior_sha256,
        "pii_partial_entity_pu_positive_margin": partial_entity_pu_positive_margin,
        "pii_partial_parent_presence_kl_weight": partial_parent_presence_kl_weight,
        "pii_complete_presence_loss_weight": complete_presence_loss_weight,
        "pii_complete_family_loss_weight": complete_family_loss_weight,
        "pii_complete_boundary_loss_weight": complete_boundary_loss_weight,
    }
    expected.update(projection_identity or {})
    mismatches = [
        f"{field}={getattr(config, field, None)!r} (checkpoint) vs {value!r} (requested)"
        for field, value in expected.items()
        if getattr(
            config,
            field,
            0.0
            if field
            in {
                "pii_complete_presence_loss_weight",
                "pii_complete_family_loss_weight",
                "pii_complete_boundary_loss_weight",
                "pii_partial_entity_pu_loss_weight",
                "pii_partial_parent_presence_kl_weight",
            }
            else None,
        )
        != value
    ]
    saved_horizon = getattr(config, "pii_dual_head_horizon", None)
    if saved_horizon is not None and saved_horizon != horizon:
        mismatches.append(f"pii_dual_head_horizon={saved_horizon!r} (checkpoint) vs {horizon!r} (requested)")
    if mismatches:
        raise ValueError("dual-head resume objective mismatch: " + "; ".join(mismatches))

    own_names = [config.id2label[index] for index in range(config.num_labels)]
    retired_at = getattr(config, "pii_dual_head_retired_at_step", None)
    if retired_at is not None:
        if retired_at != retirement_step:
            raise ValueError(
                "dual-head resume retirement mismatch: "
                f"checkpoint retired at step {retired_at}, requested transition ends at {retirement_step}"
            )
        if own_names != list(secondary_names):
            raise ValueError("retired dual-head checkpoint does not carry the requested new-ontology labels")
        if getattr(model, "secondary_classifier", None) is not None:
            raise ValueError("retired dual-head checkpoint still carries a secondary classifier")
        if getattr(config, "pii_secondary_labels", None) is not None:
            raise ValueError("retired dual-head checkpoint still declares secondary labels")
        return True

    if own_names != list(primary_names):
        raise ValueError("active dual-head checkpoint does not carry the requested old-ontology labels")
    if list(getattr(config, "pii_secondary_labels", ()) or ()) != list(secondary_names):
        raise ValueError("active dual-head checkpoint does not declare the requested new-ontology labels")
    if getattr(model, "secondary_classifier", None) is None:
        raise ValueError("active dual-head checkpoint is missing its secondary classifier")
    return False


def validate_mapped_single_head_checkpoint(
    model,
    correctness_map,
    new_names,
    *,
    exact_resume,
    old_weight_schedule,
    eval_weight,
    retirement_step,
    horizon,
    o_token_loss_weight=1.0,
    entity_dice_loss_weight=0.0,
    partial_o_loss_weight=0.0,
    partial_entity_pu_loss_weight=0.0,
    partial_entity_pu_prior_sha256=None,
    partial_entity_pu_positive_margin=None,
    partial_expected_entity_ratio_loss_weight=0.0,
    partial_expected_entity_ratio_prior_sha256=None,
    partial_expected_entity_ratio_lower_width=0.1,
    partial_parent_presence_kl_weight=0.0,
    complete_presence_loss_weight=0.0,
    complete_family_loss_weight=0.0,
    complete_boundary_loss_weight=0.0,
    resume_from_horizon=None,
    bind_native_map=False,
):
    """Validate an existing new-ontology head used with mapped supervision.

    A new data stage resets optimizer and scheduler state, but it must not
    recreate either classifier: the sole surviving head keeps learning directly
    from new-space rows and through the same correctness map from old-space
    rows. An explicitly native head may bind its first map without inventing
    a dual-head retirement history. Exact trainer-state resume additionally
    binds the current stage's objective and schedule horizon.
    """
    config = model.config
    retired_at = getattr(config, "pii_dual_head_retired_at_step", None)
    native_origin = getattr(config, "pii_mapped_single_head_origin", None) == "native"
    if bind_native_map:
        if exact_resume or retired_at is not None or native_origin:
            raise ValueError("binding a native map requires a new stage from an unmapped native checkpoint")
        if not getattr(config, "pii_native_new_label_space", False) or getattr(
            config, "pii_dual_head_map_sha256", None
        ):
            raise ValueError(
                "binding a native map requires an explicitly native checkpoint with no previous map"
            )
    if retired_at is None and not native_origin and not bind_native_map:
        raise ValueError("mapped single-head supervision needs a checkpoint whose old head is retired")
    own_names = [config.id2label[index] for index in range(config.num_labels)]
    if own_names != list(new_names):
        raise ValueError("mapped single-head checkpoint does not carry the requested new-ontology labels")
    if getattr(model, "secondary_classifier", None) is not None:
        raise ValueError("mapped single-head checkpoint unexpectedly carries a secondary classifier")
    if getattr(config, "pii_secondary_labels", None) is not None:
        raise ValueError("mapped single-head checkpoint unexpectedly declares secondary labels")

    expected = {
        "pii_dual_head_map_sha256": correctness_map.map_sha256,
        "pii_dual_head_ontology_sha256": correctness_map.ontology_sha256,
    }
    if bind_native_map:
        expected = {}
    if exact_resume:
        expected.update(
            {
                "pii_dual_head_mapped_single_head": True,
                "pii_dual_head_old_weight_schedule": old_weight_schedule,
                "pii_dual_head_eval_weight": eval_weight,
                "pii_dual_head_retirement_step": retirement_step,
                "pii_o_token_loss_weight": o_token_loss_weight,
                "pii_entity_dice_loss_weight": entity_dice_loss_weight,
                "pii_partial_o_loss_weight": partial_o_loss_weight,
                "pii_partial_entity_pu_loss_weight": partial_entity_pu_loss_weight,
                "pii_partial_entity_pu_prior_sha256": partial_entity_pu_prior_sha256,
                "pii_partial_entity_pu_positive_margin": partial_entity_pu_positive_margin,
                "pii_partial_expected_entity_ratio_loss_weight": (partial_expected_entity_ratio_loss_weight),
                "pii_partial_expected_entity_ratio_prior_sha256": (
                    partial_expected_entity_ratio_prior_sha256
                ),
                "pii_partial_expected_entity_ratio_lower_width": (partial_expected_entity_ratio_lower_width),
                "pii_partial_parent_presence_kl_weight": partial_parent_presence_kl_weight,
                "pii_complete_presence_loss_weight": complete_presence_loss_weight,
                "pii_complete_family_loss_weight": complete_family_loss_weight,
                "pii_complete_boundary_loss_weight": complete_boundary_loss_weight,
            }
        )
        expected["pii_dual_head_horizon"] = horizon if resume_from_horizon is None else resume_from_horizon
    mismatches = [
        f"{field}={getattr(config, field, None)!r} (checkpoint) vs {value!r} (requested)"
        for field, value in expected.items()
        if getattr(
            config,
            field,
            0.0
            if field
            in {
                "pii_complete_presence_loss_weight",
                "pii_complete_family_loss_weight",
                "pii_complete_boundary_loss_weight",
                "pii_partial_entity_pu_loss_weight",
                "pii_partial_expected_entity_ratio_loss_weight",
                "pii_partial_parent_presence_kl_weight",
            }
            else None,
        )
        != value
    ]
    if mismatches:
        prefix = "mapped single-head resume" if exact_resume else "mapped single-head warm start"
        raise ValueError(f"{prefix} objective mismatch: " + "; ".join(mismatches))
    if resume_from_horizon is not None and horizon <= resume_from_horizon:
        raise ValueError(
            "extended exact resume requires a new --max-steps horizon greater than "
            f"the checkpoint horizon ({resume_from_horizon})"
        )
    if bind_native_map:
        config.pii_mapped_single_head_origin = "native"


def resume_extension_restart_step(config, requested_horizon: int, enabled: bool) -> int:
    """Return the old horizon that becomes an exact-resume LR phase boundary."""
    if not enabled:
        return 0
    prior_horizon = getattr(config, "pii_dual_head_horizon", None)
    if not isinstance(prior_horizon, int) or prior_horizon <= 0:
        raise ValueError("extended exact resume requires a positive checkpoint trajectory horizon")
    if requested_horizon <= prior_horizon:
        raise ValueError(
            "--extend-resume-horizon requires --max-steps greater than the checkpoint horizon "
            f"({prior_horizon})"
        )
    return prior_horizon


def predicate_token_supervision(
    row: dict,
    offsets: list[tuple[int, int]],
    spec: PredicateSpec,
    *,
    condition_type2id: dict[str, int] | None = None,
    objective_mask_channels: frozenset[str] = frozenset(),
) -> tuple[list[list[float]], list[list[float]]] | tuple[list[list[float]], list[list[float]], list[int]]:
    """Project predicate labels and cell weights onto activating gold spans."""
    unknown_mask_channels = objective_mask_channels - set(spec.channels)
    if unknown_mask_channels:
        raise ValueError(f"unknown predicate objective mask channels: {sorted(unknown_mask_channels)}")
    targets = [[-100.0] * len(spec.channels) for _ in offsets]
    weights = [[0.0] * len(spec.channels) for _ in offsets]
    condition_ids = [-1] * len(offsets) if condition_type2id is not None else None
    primary_spans = {(start, end, label) for start, end, label in row["spans"]}
    observed_spans = set()
    channel_index = {name: index for index, name in enumerate(spec.channels)}
    for predicate_index, predicate_span in enumerate(row.get("predicate_spans", ())):
        where = f"predicate_spans[{predicate_index}]"
        if not isinstance(predicate_span, dict) or set(predicate_span) not in (
            {
                "start",
                "end",
                "type",
                "attrs",
            },
            {
                "start",
                "end",
                "type",
                "attrs",
                "objective_weights",
            },
        ):
            raise ValueError(f"{where} must contain start, end, type, attrs, and optional objective_weights")
        start = predicate_span["start"]
        end = predicate_span["end"]
        primary_type = predicate_span["type"]
        attrs = predicate_span["attrs"]
        weight_profile = predicate_span.get("objective_weights")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(row["text"])
        ):
            raise ValueError(f"{where} has invalid offsets [{start}, {end})")
        if not isinstance(primary_type, str) or not primary_type:
            raise ValueError(f"{where}.type must be a nonempty string")
        identity = (start, end, primary_type)
        if identity not in primary_spans:
            raise ValueError(f"{where} does not match an activating primary span: {identity!r}")
        if identity in observed_spans:
            raise ValueError(f"duplicate predicate supervision for primary span {identity!r}")
        observed_spans.add(identity)
        if not isinstance(attrs, dict) or not attrs:
            raise ValueError(f"{where}.attrs must be a nonempty object")
        if weight_profile is not None:
            if not isinstance(weight_profile, dict) or not {"O", "other"} <= set(weight_profile):
                raise ValueError(f"{where}.objective_weights must contain O and other")
            unknown_weight_keys = set(weight_profile) - {"O", "other", *spec.channels}
            if unknown_weight_keys:
                raise ValueError(
                    f"{where}.objective_weights has unknown keys {sorted(unknown_weight_keys)!r}"
                )
            unlabelled_weight_keys = (set(weight_profile) & set(spec.channels)) - set(attrs)
            if unlabelled_weight_keys:
                raise ValueError(
                    f"{where}.objective_weights names unknown targets {sorted(unlabelled_weight_keys)!r}"
                )
            for key, value in weight_profile.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{where}.objective_weights.{key} must be finite and nonnegative")
        token_indices = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start != token_end and token_start < end and start < token_end
        ]
        if condition_ids is not None:
            if primary_type not in condition_type2id:
                raise ValueError(f"{where}.type has no predicate condition block: {primary_type!r}")
            condition_id = condition_type2id[primary_type]
            for token_index in token_indices:
                prior = condition_ids[token_index]
                if prior not in (-1, condition_id):
                    raise ValueError(
                        "overlapping predicate carriers assign different gold primary types "
                        f"to token {token_index}"
                    )
                condition_ids[token_index] = condition_id
        for channel, intervals in attrs.items():
            if channel not in channel_index:
                raise ValueError(f"{where}.attrs has unknown channel {channel!r}")
            if primary_type not in spec.applicable_types[channel]:
                raise ValueError(f"predicate {channel!r} does not apply to primary type {primary_type!r}")
            if not isinstance(intervals, list):
                raise ValueError(f"{where}.attrs.{channel} must be a list of intervals")
            normalized = []
            for interval_index, interval in enumerate(intervals):
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or isinstance(interval[0], bool)
                    or not isinstance(interval[0], int)
                    or isinstance(interval[1], bool)
                    or not isinstance(interval[1], int)
                ):
                    raise ValueError(
                        f"{where}.attrs.{channel}[{interval_index}] must be an integer [start, end]"
                    )
                interval_start, interval_end = interval
                if not start <= interval_start < interval_end <= end:
                    raise ValueError(
                        f"{where}.attrs.{channel} interval [{interval_start}, {interval_end}) "
                        f"escapes primary span [{start}, {end})"
                    )
                if normalized and normalized[-1][1] > interval_start:
                    raise ValueError(f"{where}.attrs.{channel} intervals overlap or are unsorted")
                normalized.append((interval_start, interval_end))
            output_channel = channel_index[channel]
            for token_index in token_indices:
                token_start, token_end = offsets[token_index]
                target = float(
                    any(
                        token_start < interval_end and interval_start < token_end
                        for interval_start, interval_end in normalized
                    )
                )
                targets[token_index][output_channel] = target
                if weight_profile is None:
                    weight = 1.0
                else:
                    fallback = "other" if target else "O"
                    weight = float(weight_profile.get(channel, weight_profile[fallback]))
                if channel in objective_mask_channels:
                    weight = 0.0
                weights[token_index][output_channel] = weight
    if condition_ids is None:
        return targets, weights
    return targets, weights, condition_ids


def predicate_token_labels(
    row: dict,
    offsets: list[tuple[int, int]],
    spec: PredicateSpec,
) -> list[list[float]]:
    """Project reliable character predicates onto only their activating gold span."""
    return predicate_token_supervision(row, offsets, spec)[0]


def subclass_component_supervision(
    row: dict,
    offsets: list[tuple[int, int]],
    spec: SubclassSpec,
) -> dict[str, list]:
    """Project exact carrier/subspan categories into weighted component records."""
    raw_components = row.get("subclass_spans", [])
    if not isinstance(raw_components, list):
        raise ValueError("subclass_spans must be a list")
    primary_spans = [tuple(span) for span in row["spans"]]
    primary_index = {span: index for index, span in enumerate(primary_spans)}
    if len(primary_index) != len(primary_spans):
        raise ValueError("primary spans must be unique before subclass projection")
    primary_weights = row.get("primary_span_objective_weights")
    if primary_weights is None:
        primary_weights = [1.0] * len(primary_spans)
    elif not isinstance(primary_weights, list) or len(primary_weights) != len(primary_spans):
        raise ValueError("primary_span_objective_weights must align one-to-one with spans")
    primary_weights = [
        validate_component_weight(weight, f"primary_span_objective_weights[{index}]")
        for index, weight in enumerate(primary_weights)
    ]

    required = {
        "carrier_start",
        "carrier_end",
        "type",
        "start",
        "end",
        "family",
        "value",
    }
    optional = {"objective_weight", "learning_weight"}
    family_by_name = spec.family_by_name
    block_by_key = spec.block_by_key
    observed: dict[tuple[int, int, str, str], list[tuple[int, int, str]]] = {}
    records = []
    for index, component in enumerate(raw_components):
        where = f"subclass_spans[{index}]"
        if (
            not isinstance(component, dict)
            or not required <= set(component)
            or set(component) - required - optional
        ):
            raise ValueError(f"{where} has an invalid shape")
        carrier = (
            component["carrier_start"],
            component["carrier_end"],
            component["type"],
        )
        if carrier not in primary_index:
            raise ValueError(f"{where} has no exact activating primary span: {carrier!r}")
        family = family_by_name.get(component["family"])
        if family is None or component["value"] not in family.outcomes:
            raise ValueError(f"{where} has an unknown family or value")
        if carrier[2] not in family.applicable_types:
            raise ValueError(f"{where} family {family.name!r} does not apply to {carrier[2]!r}")
        start = component["start"]
        end = component["end"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not carrier[0] <= start < end <= carrier[1]
        ):
            raise ValueError(f"{where} has invalid component offsets [{start}, {end})")
        if family.scope == "full_primary_span" and (start, end) != carrier[:2]:
            raise ValueError(f"{where} must cover its complete primary carrier")
        family_carrier = (*carrier, family.name)
        existing = observed.setdefault(family_carrier, [])
        if family.scope == "full_primary_span" and existing:
            raise ValueError(f"{where} duplicates a whole-carrier family decision")
        if any(start < prior_end and prior_start < end for prior_start, prior_end, _ in existing):
            raise ValueError(f"{where} overlaps another component in its family")
        existing.append((start, end, component["value"]))

        objective_weight = validate_component_weight(
            component.get("objective_weight", primary_weights[primary_index[carrier]]),
            f"{where}.objective_weight",
        )
        learning_weight = validate_component_weight(
            component.get("learning_weight", 1.0),
            f"{where}.learning_weight",
        )
        token_mask = [
            int(token_start != token_end and token_start < end and start < token_end)
            for token_start, token_end in offsets
        ]
        if not any(token_mask):
            continue
        block_id, _block = block_by_key[(family.name, carrier[2])]
        records.append(
            {
                "block_id": block_id,
                "target_id": family.outcomes.index(component["value"]),
                "scope_id": SUBCLASS_SCOPE_IDS[family.scope],
                "token_mask": token_mask,
                "objective_weight": objective_weight,
                "learning_weight": learning_weight,
            }
        )
    for family_carrier, components in observed.items():
        validate_sequence_grammar(family_by_name[family_carrier[-1]], components)
    return {
        "subclass_block_ids": [record["block_id"] for record in records],
        "subclass_target_ids": [record["target_id"] for record in records],
        "subclass_scope_ids": [record["scope_id"] for record in records],
        "subclass_token_masks": [record["token_mask"] for record in records],
        "subclass_objective_weights": [record["objective_weight"] for record in records],
        "subclass_learning_weights": [record["learning_weight"] for record in records],
    }


class SpanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        label2id,
        max_len,
        include_boundary_labels=False,
        include_consistency_mask=False,
        include_complete_presence_labels=False,
        include_complete_family_labels=False,
        family_target_by_own_label=None,
        family_target_by_secondary_label=None,
        include_complete_boundary_labels=False,
        partial_entity_pu_assignments=None,
        partial_entity_ratio_assignments=None,
        partial_negative_source_groups=None,
        union_groups=None,
        secondary_label2id=None,
        sampling_weights=None,
        sampling_pool_keys=None,
        mlm_batch_probabilities=None,
        joint_objectives=False,
        surface_realizer=None,
        character_projection=None,
        character_projection_view="pair",
        o_weight_table=None,
        predicate_spec=None,
        predicate_condition_types=(),
        predicate_objective_mask_channels=(),
        subclass_spec=None,
        legacy_outside_unknown_primary_types=(),
        primary_type_learning_weights=None,
        partial_primary_objective_weight=1.0,
        partial_type_only_types=(),
        annotation_conventions=None,
        reference_type_residual_types=(),
        context_field=None,
        context_separator="\n\n",
        soft_registers=0,
        source_classes=None,
        source_dropout=0.0,
        source_condition="row",
        languages=None,
        language_dropout=0.0,
        language_condition="row",
        bias_languages=None,
        domain_posteriors=None,
        domain_scopes=(),
        domain_dropout=0.0,
        native_label_space=None,
        context_side="both",
        mapped_outside_by_row=False,
        context_configurations=None,
        context_configuration_weights=None,
        status_types=(),
        status_accepts=None,
        prompt_slots=0,
        prompt_languages=None,
        document_start_marker=False,
    ):
        # Rows in this label space are the head's own targets rather than mapped
        # or union-bound ones; the head's inventory was built from that ontology.
        self.native_label_space = native_label_space
        # Tag-status prompt slots: per primary type, whether this row's target text
        # certainly contains it (present), certainly lacks it (absent), or cannot say.
        # Old-ontology spans resolve through the correctness map's accepted type sets.
        self.status_types = tuple(status_types)
        self.status_accepts = status_accepts
        if self.status_types and status_accepts is None:
            raise ValueError("tag-status slots need the correctness map's accepted type sets")
        # Prompt slots are spliced in by the model at the target's bounds, which each
        # item reports; their width comes out of the token budget like registers do.
        self.prompt_slots = int(prompt_slots)
        self.prompt_language_index = (
            None
            if prompt_languages is None
            else {value: position for position, value in enumerate(prompt_languages)}
        )
        self.mapped_outside_by_row = mapped_outside_by_row
        self.annotation_conventions = annotation_conventions
        if annotation_conventions is not None and (
            union_groups is not None
            or partial_type_only_types
            or include_boundary_labels
            or include_complete_boundary_labels
        ):
            raise ValueError(
                "internal segmentation cannot be combined with union/type-only or exact-boundary auxiliary supervision"
            )
        # Learned register positions the encoder reads and is never scored on; the
        # dataset only reserves their slots, `pii_soft_registers` owns their values.
        self.soft_registers = int(soft_registers)
        # Which annotation pipeline labelled this row, as a class id for a conditioning
        # register slot. Dropout replaces it with `unknown` on a fraction of rows so the
        # condition production actually feeds is trained rather than extrapolated.
        self.source_classes = source_classes
        self.source_dropout = float(source_dropout)
        if not 0.0 <= self.source_dropout <= 1.0:
            raise ValueError("source dropout must be a probability")
        if self.source_dropout and self.source_classes is None:
            raise ValueError("source dropout needs a source-class vocabulary")
        if source_condition not in {"row", "unknown"}:
            raise ValueError(f"unsupported source condition {source_condition!r}")
        # Evaluation pins the condition to `unknown` rather than sampling it. Two reasons:
        # `unknown` is what production feeds, so it is the condition worth selecting a
        # checkpoint on, and a sampled condition makes evaluation nondeterministic, which
        # the trainer's own checkpoint-reproduction guard rightly rejects.
        self.source_condition = source_condition
        if source_condition == "unknown" and self.source_dropout:
            raise ValueError("a pinned unknown condition must not also sample dropout")
        # Language is the mirror image of source. Production usually knows the language, so
        # the deployable condition is the true one and dropout exists only to keep the
        # unspecified-language fallback trained, not because unknown is the normal case.
        self.languages = languages
        self.language_index = {value: position for position, value in enumerate(languages or ())}
        self.language_dropout = float(language_dropout)
        if not 0.0 <= self.language_dropout <= 1.0:
            raise ValueError("language dropout must be a probability")
        if self.language_dropout and self.languages is None:
            raise ValueError("language dropout needs a language vocabulary")
        if language_condition not in {"row", "unknown"}:
            raise ValueError(f"unsupported language condition {language_condition!r}")
        self.language_condition = language_condition
        if language_condition == "unknown" and self.language_dropout:
            raise ValueError("a pinned unknown condition must not also sample dropout")
        # Explicit routing for the per-language output bias. -1 selects the shared head,
        # which is what an unlisted language gets; there is no unknown row to learn.
        self.bias_language_index = (
            {value: position for position, value in enumerate(bias_languages)} if bias_languages else None
        )
        # Learned-domain posteriors, keyed by text hash so they join by content rather
        # than by an identifier that may not have survived materialisation. Index 0 of
        # each vector is the unknown class and stays zero, so a dropped or unmatched row
        # contributes nothing and is exactly the condition deployment can always supply.
        self.domain_posteriors = domain_posteriors
        self.domain_scopes = tuple(domain_scopes)
        self.domain_dropout = float(domain_dropout)
        if not 0.0 <= self.domain_dropout <= 1.0:
            raise ValueError("domain dropout must be a probability")
        if self.domain_scopes and self.domain_posteriors is None:
            raise ValueError("domain conditioning needs a posterior sidecar")
        self.domain_width = (
            len(next(iter(self.domain_posteriors.values()))["segment"]) if self.domain_posteriors else 0
        )
        # Neighbouring sentences the encoder may read but never predicts on. Context
        # tokens are given zero-width offsets, which every per-token channel below
        # already treats the way it treats a special token, so target-only supervision
        # needs no per-channel masking.
        self.context_field = context_field
        self.context_separator = context_separator
        self.context_side = resolve_context_side(context_side)
        self.document_start_marker = bool(document_start_marker)
        self.context_configurations = resolve_context_configurations(context_configurations)
        self.context_configuration_weights = resolve_context_weights(
            context_configuration_weights, self.context_configurations
        )
        check_context_mixture(context_field, self.context_side, self.context_configurations)
        self.o_weight_table = o_weight_table
        self.predicate_spec: PredicateSpec | None = predicate_spec
        self.predicate_condition_type2id = {
            primary_type: index for index, primary_type in enumerate(predicate_condition_types)
        }
        self.predicate_objective_mask_channels = frozenset(predicate_objective_mask_channels)
        if self.predicate_condition_type2id and self.predicate_spec is None:
            raise ValueError("predicate condition types require a predicate spec")
        if self.predicate_objective_mask_channels and self.predicate_spec is None:
            raise ValueError("predicate objective mask channels require a predicate spec")
        self.subclass_spec: SubclassSpec | None = subclass_spec
        self.legacy_outside_unknown_primary_types = frozenset(legacy_outside_unknown_primary_types)
        self.primary_type_learning_weights = {
            str(primary_type): float(weight)
            for primary_type, weight in (primary_type_learning_weights or {}).items()
        }
        if any(
            not primary_type or not math.isfinite(weight) or weight < 1.0
            for primary_type, weight in self.primary_type_learning_weights.items()
        ):
            raise ValueError("primary type learning weights must be finite, named, and at least 1")
        self.partial_primary_objective_weight = float(partial_primary_objective_weight)
        if (
            not math.isfinite(self.partial_primary_objective_weight)
            or self.partial_primary_objective_weight < 0
        ):
            raise ValueError("partial primary objective weight must be finite and nonnegative")
        self.reference_type_residual_types = tuple(reference_type_residual_types)
        if self.reference_type_residual_types and (
            len(set(self.reference_type_residual_types)) != len(self.reference_type_residual_types)
            or any(
                not isinstance(primary_type, str) or not primary_type
                for primary_type in self.reference_type_residual_types
            )
        ):
            raise ValueError("reference residual types must be unique nonempty strings")
        if self.reference_type_residual_types and secondary_label2id is None:
            raise ValueError("reference residual supervision requires mapped successor labels")
        self.rows = rows
        self.include_native_outside_allowed = self.native_label_space is not None and any(
            row.get("unknown_primary_types") for row in rows
        )
        self.include_mapped_outside_allowed = (
            mapped_outside_by_row
            and secondary_label2id is not None
            and any(
                row.get("unknown_primary_types")
                and set(row["unknown_primary_types"]) != self.legacy_outside_unknown_primary_types
                for row in rows
            )
        )
        # Types whose partial-supervision rows teach what the span is but not
        # where it ends. Indexed here so the loss can map a token back to the
        # set of BIOES tags it is allowed to choose among.
        self.partial_type_only_types = frozenset(partial_type_only_types or ())
        self.type_only_index = {
            primary_type: index for index, primary_type in enumerate(sorted(self.partial_type_only_types))
        }
        self.include_primary_objective_weights = (
            self.partial_primary_objective_weight != 1.0
            or bool(self.primary_type_learning_weights)
            or any("primary_span_objective_weights" in row for row in rows)
        )
        self.tok = tokenizer
        self.label2id = label2id
        self.id2label = {index: label for label, index in label2id.items()}
        # Register slots come out of the same length budget as the text, so the
        # target sentence keeps the room it had before registers were added.
        self.max_len = max_len - self.soft_registers - self.prompt_slots
        self.include_boundary_labels = include_boundary_labels
        self.include_consistency_mask = include_consistency_mask
        self.include_complete_presence_labels = include_complete_presence_labels
        self.include_complete_family_labels = include_complete_family_labels
        if include_complete_family_labels and (
            family_target_by_own_label is None or family_target_by_secondary_label is None
        ):
            raise ValueError("complete family labels require both family target tables")
        self.family_target_by_own_label = family_target_by_own_label
        self.family_target_by_secondary_label = family_target_by_secondary_label
        self.include_complete_boundary_labels = include_complete_boundary_labels
        self.partial_entity_pu_assignments = partial_entity_pu_assignments
        self.partial_entity_pu_groups = (
            None
            if partial_entity_pu_assignments is None
            else {key: index for index, key in enumerate(sorted(partial_entity_pu_assignments))}
        )
        self.partial_entity_ratio_assignments = partial_entity_ratio_assignments
        self.partial_entity_ratio_groups = (
            None
            if partial_entity_ratio_assignments is None
            else {key: index for index, key in enumerate(sorted(partial_entity_ratio_assignments))}
        )
        if partial_entity_pu_assignments is not None and partial_entity_ratio_assignments is not None:
            raise ValueError("PU and expected entity-ratio priors are mutually exclusive")
        self.partial_negative_source_groups = partial_negative_source_groups
        self.union_groups: UnionLabelGroups | None = union_groups
        self.secondary_label2id = secondary_label2id
        self.secondary_id2label = (
            None
            if secondary_label2id is None
            else {index: label for label, index in secondary_label2id.items()}
        )
        self.joint_objectives = joint_objectives
        self.surface_realizer = surface_realizer
        self.character_projection = character_projection
        self.character_projection_view = character_projection_view
        self.sampling_weights = None if sampling_weights is None else list(sampling_weights)
        if self.sampling_weights is not None and len(self.sampling_weights) != len(self.rows):
            raise ValueError("sampling weights must align one-to-one with dataset rows")
        self.sampling_pool_keys = (
            None if sampling_pool_keys is None else [str(key) for key in sampling_pool_keys]
        )
        if self.sampling_pool_keys is not None and len(self.sampling_pool_keys) != len(self.rows):
            raise ValueError("sampling pool keys must align one-to-one with dataset rows")
        self.mlm_batch_probabilities = (
            None
            if mlm_batch_probabilities is None
            else {str(pool): float(probability) for pool, probability in mlm_batch_probabilities.items()}
        )
        self._token_lengths = None

    def __len__(self):
        return len(self.rows)

    def length_text(self, row):
        """The text whose token count approximates this row's encoded input length.

        A row's fixed context counts toward its length. Under per-draw context
        configurations the variant is unknown in advance, so only the target counts.
        """
        context = row.get(self.context_field) if self.context_field else None
        if not context or self.context_configurations is not None:
            return row["text"]
        if isinstance(context, str):
            return context.strip() + self.context_separator + row["text"]
        before = context.get("before", "").strip()
        after = context.get("after", "").strip() if self.context_side == "both" else ""
        # The tokenizer reads a literal separator string as that special token,
        # so this counts the two document-start marker tokens.
        marker = self.tok.sep_token * 2 if self.document_start_marker and context.get(DOCUMENT_START) else ""
        return marker + (before + "\n\n" if before else "") + row["text"] + ("\n\n" + after if after else "")

    def token_lengths(self, chunk_size=4096):
        """Materialize truncated token lengths once for padding-aware sampling."""
        if self._token_lengths is None:
            lengths = []
            for start in range(0, len(self.rows), chunk_size):
                encoded = self.tok(
                    [self.length_text(row) for row in self.rows[start : start + chunk_size]],
                    truncation=True,
                    max_length=self.max_len,
                    return_length=True,
                )
                lengths.extend(int(value) for value in encoded["length"])
            self._token_lengths = lengths
        return self._token_lengths

    def _family_target(self, labels, secondary_labels, j):
        """Coarse family target for one trustworthy complete-row token.

        Own-channel (old-space) entity tokens supervise the single family their
        accepted new set covers and are masked when that set spans several
        families; new-space entity tokens supervise their type's family; every
        other complete-row token is a trustworthy ``O``.
        """
        own = labels[j]
        if own not in (-100, self.label2id["O"]):
            return self.family_target_by_own_label.get(own, -100)
        if (
            self.secondary_label2id is not None
            and secondary_labels[j] != -100
            and secondary_labels[j] != self.secondary_label2id[UNION_OUTSIDE_LABEL]
        ):
            return self.family_target_by_secondary_label.get(secondary_labels[j], -100)
        return 0

    def encode_with_context(self, r):
        """Tokenize one row, optionally behind its neighbouring sentences.

        Returns the encoding and token offsets in the *target* text's character
        coordinates. Context tokens come back as zero-width offsets, which marks them
        the way special tokens are already marked, so no span in the row needs shifting
        and no per-token supervision channel needs a context case. Truncation drops
        context first; a target that alone exceeds the length budget falls back to the
        legacy no-context truncation. New training stages must pre-window their
        targets by token capacity to preserve tail supervision.
        """
        text = r["text"]
        context = r.get(self.context_field) if self.context_field else None
        if context is not None:
            if self.context_configurations is not None:
                offsets = random.choices(
                    self.context_configurations, weights=self.context_configuration_weights
                )[0]
                context = select_document_context(context, offsets)
            contextual = encode_document_context(
                self.tok,
                text,
                context,
                self.max_len,
                self.context_separator,
                side=self.context_side,
                document_start_marker=self.document_start_marker,
            )
            if contextual is not None:
                return contextual
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        return enc, offsets

    def tag_status_ids(self, r, supervision, label_space, unknown_primary_types, offsets, special) -> dict:
        """Per-type status ids for this item: 0 unknown, 1 absent, 2 present.

        Only spans that reach a token of the target text count. A span shows its type
        present only when it can map to that type alone; absence needs complete
        supervision, a type the source annotates, and no span that could be that type.
        """
        accepts = []
        for start, end, cat in r["spans"]:
            if not any(not special[j] and a < end and start < b for j, (a, b) in enumerate(offsets)):
                continue
            if label_space == NEW_LABEL_SPACE:
                accepts.append(frozenset({cat}))
            elif cat in self.status_accepts:
                accepts.append(self.status_accepts[cat])
            else:
                raise ValueError(f"span type {cat!r} has no accepted successor types for tag-status slots")
        trusts_absence = supervision == COMPLETE_SUPERVISION
        # An old-ontology O without an explicit unknown list still means "O or one of
        # the map's legacy unknown types", so those types cannot be called absent.
        unknown = set(unknown_primary_types)
        if label_space != NEW_LABEL_SPACE and not unknown:
            unknown = set(self.legacy_outside_unknown_primary_types)
        ids = {}
        for name in self.status_types:
            if any(accept == {name} for accept in accepts):
                value = 2
            elif trusts_absence and name not in unknown and not any(name in a for a in accepts):
                value = 1
            else:
                value = 0
            ids[f"status_{name}_id"] = value
        return ids

    def reserve_register_slots(self, enc, offsets):
        if not self.soft_registers:
            return enc, offsets
        filler = placeholder_id(self.tok)
        input_ids, offsets = reserve_positions(enc["input_ids"], offsets, self.soft_registers, filler)
        enc["input_ids"] = input_ids
        if "attention_mask" in enc:
            enc["attention_mask"] = [1] * self.soft_registers + list(enc["attention_mask"])
        return enc, offsets

    def __getitem__(self, i):
        sampled_mlm = None
        surface_draw_nonce = None
        if isinstance(i, tuple):
            if len(i) == 2 and isinstance(i[1], bool):
                i, sampled_mlm = i
            elif len(i) == 3 and (i[1] is None or isinstance(i[1], bool)) and isinstance(i[2], int):
                i, sampled_mlm, surface_draw_nonce = i
            else:
                raise ValueError(f"invalid sampled dataset index {i!r}")
        r = self.rows[i]
        if self.surface_realizer is not None:
            if surface_draw_nonce is None:
                raise ValueError("sample-time surface realization requires a sampler draw nonce")
            r = self.surface_realizer.realize(r, surface_draw_nonce)
        stored_objective = r.get("objective", "tag")
        objective = "mlm" if sampled_mlm else stored_objective
        if sampled_mlm is False and stored_objective == "mlm":
            raise ValueError("stored MLM-only row was sampled with tagging objective")
        if objective == "mlm":
            if not self.joint_objectives:
                raise ValueError("MLM replay row requires joint-objective collation")
            enc = self.tok(r["text"], truncation=True, max_length=self.max_len)
            enc["pii_objective"] = "mlm"
            return enc
        if objective != "tag":
            raise ValueError(f"unsupported training objective {objective!r}")
        internal_tags = set()
        annotation_row = r
        if self.annotation_conventions is not None:
            r, internal_tags = convention_training_row(r, self.annotation_conventions)
        enc, offsets = self.reserve_register_slots(*self.encode_with_context(r))
        if self.prompt_slots:
            enc["prompt_target_start"], enc["prompt_target_end"] = prompt_target_bounds(offsets)
            if self.prompt_language_index is not None:
                enc["prompt_language_id"] = self.prompt_language_index.get(r.get("lang"), -1)
        segmentation_prefix_masks = [0] * len(offsets)
        if self.source_classes is not None:
            # Dropped per item rather than per row, so the same row is seen both with its
            # true class and as unknown across epochs; that is what trains the unknown
            # condition instead of leaving it extrapolated.
            dropped = self.source_condition == "unknown" or (
                self.source_dropout and random.random() < self.source_dropout
            )
            enc["source_id"] = (
                self.source_classes.unknown_id if dropped else self.source_classes.id_of(r.get("src"))
            )
        if self.languages is not None:
            dropped = self.language_condition == "unknown" or (
                self.language_dropout and random.random() < self.language_dropout
            )
            enc["language_id"] = 0 if dropped else self.language_index.get(r.get("lang"), 0)
        if self.bias_language_index is not None:
            enc["language_ids"] = self.bias_language_index.get(r.get("lang"), -1)
        for scope in self.domain_scopes:
            record = self.domain_posteriors.get(domain_key(r.get("text") or ""))
            values = None if record is None else record.get(scope)
            dropped = self.domain_dropout and random.random() < self.domain_dropout
            width = 1 + self.domain_width
            if values is None or dropped:
                enc[f"domain_{scope}_id"] = [1.0] + [0.0] * self.domain_width
            else:
                enc[f"domain_{scope}_id"] = [0.0] + list(values)
            assert len(enc[f"domain_{scope}_id"]) == width
        supervision = r.get("supervision", COMPLETE_SUPERVISION)
        if supervision == COMPLETE_SUPERVISION:
            labels = [self.label2id["O"]] * len(offsets)
        elif supervision == ANNOTATED_SPANS_ONLY:
            labels = [-100] * len(offsets)
        else:
            raise ValueError(f"unsupported supervision mode {supervision!r}")
        primary_objective_weights = None
        span_objective_weights = r.get("primary_span_objective_weights")
        if span_objective_weights is not None:
            if not isinstance(span_objective_weights, list) or len(span_objective_weights) != len(r["spans"]):
                raise ValueError("primary_span_objective_weights must align one-to-one with spans")
            if any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
                for weight in span_objective_weights
            ):
                raise ValueError("primary span objective weights must be finite and nonnegative")
        if self.include_primary_objective_weights:
            primary_objective_weights = [1.0 if supervision == COMPLETE_SUPERVISION else 0.0 for _ in offsets]
            if span_objective_weights is None:
                span_objective_weights = [1.0] * len(r["spans"])
        # How far to trust this row's O labels, by intake method. Only complete
        # supervision carries it: annotated_spans_only rows are already masked,
        # so there is no O to weight.
        o_weight_table = getattr(self, "o_weight_table", None)
        if o_weight_table is not None and supervision == COMPLETE_SUPERVISION:
            enc["o_weight"] = o_weight_table.weight_for(r.get("src") or r.get("source"))
        label_space = r.get("label_space", FINE_LABEL_SPACE)
        if label_space not in LABEL_SPACES:
            raise ValueError(f"unsupported label space {label_space!r}")
        if (
            label_space == UNION_LABEL_SPACE
            and label_space != self.native_label_space
            and self.union_groups is None
            and self.secondary_label2id is None
        ):
            raise ValueError(
                "a row labelled in the new ontology needs either the member document --union-members "
                "binds or the second head --dual-head-map builds"
            )
        unknown_primary_types = r.get("unknown_primary_types", [])
        if not isinstance(unknown_primary_types, list) or any(
            not isinstance(primary_type, str) or not primary_type for primary_type in unknown_primary_types
        ):
            raise ValueError("unknown_primary_types must be a list of nonempty strings")
        if len(set(unknown_primary_types)) != len(unknown_primary_types):
            raise ValueError("unknown_primary_types must not contain duplicates")
        legacy_unknown_outside = bool(unknown_primary_types)
        if legacy_unknown_outside:
            if supervision != COMPLETE_SUPERVISION:
                raise ValueError("unknown_primary_types applies only to complete rows")
            if self.native_label_space is not None:
                known_types = {label.split("-", 1)[1] for label in self.label2id if label != "O"}
                if not set(unknown_primary_types) <= known_types:
                    raise ValueError("unknown_primary_types contains a type absent from the native head")
            elif self.secondary_label2id is None:
                raise ValueError("unknown_primary_types requires mapped successor supervision")
            elif (
                not self.mapped_outside_by_row
                and set(unknown_primary_types) != self.legacy_outside_unknown_primary_types
            ):
                raise ValueError(
                    "unknown_primary_types must exactly match the successor map's legacy-O unknown set"
                )
            else:
                known_types = {label.split("-", 1)[1] for label in self.secondary_label2id if label != "O"}
                if not set(unknown_primary_types) <= known_types:
                    raise ValueError("unknown_primary_types contains a type absent from the successor head")
        if self.include_native_outside_allowed:
            enc["native_outside_allowed"] = [
                label == "O" or label.split("-", 1)[1] in unknown_primary_types
                for _index, label in sorted(self.id2label.items())
            ]
        if self.include_mapped_outside_allowed:
            enc["mapped_outside_allowed"] = [
                label == "O" or label.split("-", 1)[1] in unknown_primary_types
                for label, _index in sorted(self.secondary_label2id.items(), key=lambda item: item[1])
            ]
        type_only_labels: list[int] = [] if not self.partial_type_only_types else [-100] * len(offsets)
        union_groups: list[int] = [] if self.union_groups is None else [-1] * len(offsets)
        union_fine_trusted: list[int] = [] if self.union_groups is None else [0] * len(offsets)
        secondary_labels: list[int] = [] if self.secondary_label2id is None else [-100] * len(offsets)
        special = [a == b for a, b in offsets]
        if self.status_types:
            enc.update(
                self.tag_status_ids(r, supervision, label_space, unknown_primary_types, offsets, special)
            )
        reference_type_labels = None
        reference_type_weights = None
        if self.reference_type_residual_types:
            outside_is_known = supervision == COMPLETE_SUPERVISION and not legacy_unknown_outside
            reference_type_labels = [
                [0.0 if outside_is_known else -100.0] * len(self.reference_type_residual_types)
                for _ in offsets
            ]
            reference_type_weights = [
                [1.0 if outside_is_known else 0.0] * len(self.reference_type_residual_types) for _ in offsets
            ]
        ignored_tokens = [False] * len(offsets)
        ignored_spans = r.get("ignored_spans", [])
        for ignored in ignored_spans:
            if (
                not isinstance(ignored, (list, tuple))
                or len(ignored) != 3
                or not isinstance(ignored[0], int)
                or not isinstance(ignored[1], int)
                or not isinstance(ignored[2], str)
                or not ignored[2]
            ):
                raise ValueError(f"invalid ignored span {ignored!r}")
            start, end, _ = ignored
            if not (0 <= start < end <= len(r["text"])):
                raise ValueError(f"ignored span [{start}, {end}) is outside text of length {len(r['text'])}")
        ordered_ignored = sorted(ignored_spans, key=lambda span: (span[0], span[1], span[2]))
        for left, right in zip(ordered_ignored, ordered_ignored[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping ignored spans {left!r} and {right!r}")
        for ignored_start, ignored_end, _ in ordered_ignored:
            for span_start, span_end, span_type in r["spans"]:
                if ignored_start < span_end and span_start < ignored_end:
                    raise ValueError(
                        f"ignored span [{ignored_start}, {ignored_end}) overlaps supervised "
                        f"{span_type} span [{span_start}, {span_end})"
                    )
        for span_index, (start, end, cat) in enumerate(r["spans"]):
            toks = [j for j, (a, b) in enumerate(offsets) if not special[j] and a < end and start < b]
            if not toks:
                continue
            if cat in internal_tags:
                for j, prefix_mask in zip(toks, internal_prefix_masks(len(toks))):
                    segmentation_prefix_masks[j] = prefix_mask
            # Foreign-standard gold is reliable about what an organization is and
            # unreliable about where our annotators would have ended the span:
            # measured, it cuts outright organization misses by 15 and adds 18
            # boundary errors. Supervising the type without the boundary keeps the
            # first and discards the second. The exact BIOES target is withheld and
            # replaced by a target on the summed probability of that type's four
            # boundary tags, so the model is free to place the edges where our own
            # complete-supervision rows taught it to.
            boundary_free = supervision == ANNOTATED_SPANS_ONLY and cat in self.partial_type_only_types
            for j, name in zip(toks, bioes(cat, len(toks))):
                if boundary_free:
                    type_only_labels[j] = self.type_only_index[cat]
                    labels[j] = -100
                elif self.secondary_label2id is not None and label_space == NEW_LABEL_SPACE:
                    # A new-ontology span is this head's own target, and reaches
                    # the incumbent head only through the correctness map.
                    secondary_labels[j] = self.secondary_label2id[name]
                    labels[j] = -100
                elif self.union_groups is None:
                    labels[j] = self.label2id[name]
                elif label_space == UNION_LABEL_SPACE:
                    # The fine target is latent: the head row recorded here is the
                    # group's representative, which keeps every label-shaped
                    # consumer (boundary letters, partial masks, span decoding)
                    # working, and the loss replaces its fine term by the group's.
                    union_groups[j] = self.union_groups.group_of_union_label[name]
                    labels[j] = self.label2id[self.union_groups.representative_of_union_label[name]]
                else:
                    labels[j] = self.label2id[name]
                    group = self.union_groups.group_of_fine_label.get(name)
                    if group is not None:
                        union_groups[j] = group
                        union_fine_trusted[j] = 1
                if primary_objective_weights is not None:
                    primary_objective_weights[j] = float(span_objective_weights[span_index]) * (
                        self.primary_type_learning_weights.get(cat, 1.0)
                    )
                    if supervision == ANNOTATED_SPANS_ONLY:
                        primary_objective_weights[j] *= self.partial_primary_objective_weight
                if reference_type_labels is not None:
                    objective_weight = (
                        1.0 if span_objective_weights is None else float(span_objective_weights[span_index])
                    )
                    reference_type_labels[j] = [
                        float(cat == primary_type) for primary_type in self.reference_type_residual_types
                    ]
                    reference_type_weights[j] = [objective_weight] * len(self.reference_type_residual_types)
        # Ignored annotations preserve a known out-of-head concept without
        # turning it into trusted O. Mask any overlapping tokenizer position;
        # masking a boundary-straddling token is safer than applying either a
        # false negative or a partial positive target.
        for start, end, _ in ordered_ignored:
            for j, (a, b) in enumerate(offsets):
                if special[j] or a >= end or start >= b:
                    continue
                ignored_tokens[j] = True
                labels[j] = -100
                if primary_objective_weights is not None:
                    primary_objective_weights[j] = 0.0
                if self.secondary_label2id is not None:
                    secondary_labels[j] = -100
                if reference_type_labels is not None:
                    reference_type_labels[j] = [-100.0] * len(self.reference_type_residual_types)
                    reference_type_weights[j] = [0.0] * len(self.reference_type_residual_types)
                if self.union_groups is not None:
                    union_groups[j] = -1
                    union_fine_trusted[j] = 0
        for j, sp in enumerate(special):
            if sp:
                labels[j] = -100
                if primary_objective_weights is not None:
                    primary_objective_weights[j] = 0.0
                if self.secondary_label2id is not None:
                    secondary_labels[j] = -100
                if reference_type_labels is not None:
                    reference_type_labels[j] = [-100.0] * len(self.reference_type_residual_types)
                    reference_type_weights[j] = [0.0] * len(self.reference_type_residual_types)
        supervised = [label != -100 for label in labels]
        if self.secondary_label2id is not None:
            supervised = [own or secondary != -100 for own, secondary in zip(supervised, secondary_labels)]
        if self.partial_type_only_types:
            # A boundary-free span withholds the exact tag on purpose, so the row
            # is still supervised even though every one of its labels is masked.
            supervised = [own or kind != -100 for own, kind in zip(supervised, type_only_labels)]
        if supervision == ANNOTATED_SPANS_ONLY and not any(supervised):
            identity = r.get("id") or r.get("document_id") or f"dataset index {i}"
            raise ValueError(
                f"annotated-spans-only row {identity!r} has no token-aligned span after truncation"
            )
        enc["labels"] = labels
        if self.annotation_conventions is not None:
            enc["segmentation_prefix_masks"] = segmentation_prefix_masks
        if self.partial_type_only_types:
            enc["type_only_labels"] = type_only_labels
        if primary_objective_weights is not None:
            enc["primary_objective_weights"] = primary_objective_weights
        if self.secondary_label2id is not None:
            # Frozen-map rows agree exactly on O. Successor-stamped predecessor
            # rows instead leave the product O unset so their carrier-side O
            # reaches the O-or-new-type marginal in the extended map.
            outside_new = self.secondary_label2id[UNION_OUTSIDE_LABEL]
            outside_old = self.label2id["O"]
            enc["secondary_labels"] = [
                outside_new
                if (not legacy_unknown_outside and label_id == outside_old and secondary_labels[j] == -100)
                else secondary_labels[j]
                for j, label_id in enumerate(labels)
            ]
        if reference_type_labels is not None:
            enc["reference_type_labels"] = reference_type_labels
            enc["reference_type_weights"] = reference_type_weights
        if (
            self.partial_entity_pu_assignments is not None
            or self.partial_entity_ratio_assignments is not None
        ):
            enc["partial_entity_positive_mask"] = [
                int(supervision == ANNOTATED_SPANS_ONLY and supervised[j]) for j in range(len(labels))
            ]
        if self.partial_entity_pu_assignments is not None:
            if supervision == ANNOTATED_SPANS_ONLY:
                language = r.get("lang") or r.get("language")
                if not isinstance(language, str) or not language:
                    raise ValueError("PU partial rows require a nonempty language")
                key = (row_source_name(r), language)
                if key not in self.partial_entity_pu_assignments:
                    raise ValueError(f"PU class-prior receipt has no assignment for {key[0]}/{key[1]}")
                enc["partial_entity_pu_group"] = self.partial_entity_pu_groups[key]
                enc["partial_entity_pu_prior"] = self.partial_entity_pu_assignments[key]
            else:
                enc["partial_entity_pu_group"] = -1
                enc["partial_entity_pu_prior"] = 0.0
        if self.partial_entity_ratio_assignments is not None:
            if supervision == ANNOTATED_SPANS_ONLY:
                language = r.get("lang") or r.get("language")
                if not isinstance(language, str) or not language:
                    raise ValueError("expected entity-ratio partial rows require a nonempty language")
                key = (row_source_name(r), language)
                if key not in self.partial_entity_ratio_assignments:
                    raise ValueError(
                        f"expected entity-ratio prior receipt has no assignment for {key[0]}/{key[1]}"
                    )
                enc["partial_entity_ratio_group"] = self.partial_entity_ratio_groups[key]
                enc["partial_entity_ratio_prior"] = self.partial_entity_ratio_assignments[key]
            else:
                enc["partial_entity_ratio_group"] = -1
                enc["partial_entity_ratio_prior"] = 0.0
        if self.include_boundary_labels:
            enc["boundary_labels"] = (
                [
                    boundary_label_id(self.id2label[label_id]) if label_id != -100 else -100
                    for label_id in labels
                ]
                if supervision == ANNOTATED_SPANS_ONLY
                else [-100] * len(labels)
            )
        if self.include_consistency_mask:
            enc["consistency_mask"] = [
                int(
                    supervision == ANNOTATED_SPANS_ONLY
                    and not special[j]
                    and not ignored_tokens[j]
                    and not supervised[j]
                )
                for j in range(len(labels))
            ]
        if self.include_complete_presence_labels:
            enc["complete_presence_labels"] = [
                (
                    int(
                        (labels[j] != -100 and labels[j] != self.label2id["O"])
                        or (
                            self.secondary_label2id is not None
                            and secondary_labels[j] != -100
                            and secondary_labels[j] != self.secondary_label2id[UNION_OUTSIDE_LABEL]
                        )
                    )
                    if (
                        supervision == COMPLETE_SUPERVISION
                        and not special[j]
                        and not ignored_tokens[j]
                        and not (
                            legacy_unknown_outside
                            and labels[j] == self.label2id["O"]
                            and secondary_labels[j] == -100
                        )
                    )
                    else -100
                )
                for j in range(len(labels))
            ]
        if self.include_complete_family_labels:
            enc["complete_family_labels"] = [
                (
                    self._family_target(labels, secondary_labels, j)
                    if (
                        supervision == COMPLETE_SUPERVISION
                        and not special[j]
                        and not ignored_tokens[j]
                        and not (
                            legacy_unknown_outside
                            and labels[j] == self.label2id["O"]
                            and secondary_labels[j] == -100
                        )
                    )
                    else -100
                )
                for j in range(len(labels))
            ]
        if self.include_complete_boundary_labels:
            enc["complete_boundary_labels"] = [
                (
                    boundary_label_id(self.id2label[labels[j]])
                    if (
                        supervision == COMPLETE_SUPERVISION
                        and not special[j]
                        and labels[j] not in (-100, self.label2id["O"])
                    )
                    else boundary_label_id(self.secondary_id2label[secondary_labels[j]])
                    if (
                        supervision == COMPLETE_SUPERVISION
                        and not special[j]
                        and self.secondary_label2id is not None
                        and secondary_labels[j] not in (-100, self.secondary_label2id[UNION_OUTSIDE_LABEL])
                    )
                    else -100
                )
                for j in range(len(labels))
            ]
        if self.union_groups is not None:
            # A masked position has no target of any kind, so it keeps no group
            # either: the loss reads these two only where a label survives.
            enc["union_groups"] = [group if labels[j] != -100 else -1 for j, group in enumerate(union_groups)]
            enc["union_fine_trusted"] = [
                trusted if labels[j] != -100 else 0 for j, trusted in enumerate(union_fine_trusted)
            ]
        if self.partial_negative_source_groups is not None:
            source_group = self.partial_negative_source_groups.get(r.get("src"), -1)
            enc["partial_negative_groups"] = [
                source_group
                if (
                    source_group >= 0
                    and supervision == ANNOTATED_SPANS_ONLY
                    and not special[j]
                    and not ignored_tokens[j]
                    and not supervised[j]
                )
                else -1
                for j in range(len(labels))
            ]
        if self.predicate_spec is not None:
            supervision_fields = predicate_token_supervision(
                annotation_row,
                offsets,
                self.predicate_spec,
                condition_type2id=(
                    self.predicate_condition_type2id if self.predicate_condition_type2id else None
                ),
                objective_mask_channels=self.predicate_objective_mask_channels,
            )
            predicate_labels, predicate_weights = supervision_fields[:2]
            enc["predicate_labels"] = predicate_labels
            enc["predicate_weights"] = predicate_weights
            if self.predicate_condition_type2id:
                enc["predicate_condition_ids"] = supervision_fields[2]
        if self.subclass_spec is not None:
            enc.update(subclass_component_supervision(annotation_row, offsets, self.subclass_spec))
        if self.joint_objectives:
            enc["pii_objective"] = "tag"
        if self.character_projection is not None:
            character_inputs = project_continuous_characters(
                [r],
                torch.tensor([offsets], dtype=torch.long),
                projection=self.character_projection,
                projection_view=self.character_projection_view,
            )
            enc["character_ids"] = character_inputs.character_ids[0].tolist()
            enc["character_mask"] = character_inputs.character_mask[0].tolist()
            enc["token_offsets"] = character_inputs.token_offsets[0].tolist()
        if self._token_lengths is not None:
            # The length this item was batched by, for LengthAuditCollator to set
            # beside the length actually encoded (register positions included).
            enc["pii_binned_length"] = self._token_lengths[i] + self.soft_registers
        return enc


class PartialLabelDataCollator:
    """Pad partial-supervision and union-target fields beside the stock token batch."""

    AUXILIARY_PADDING = {
        "boundary_labels": -100,
        "complete_boundary_labels": -100,
        "complete_family_labels": -100,
        "complete_presence_labels": -100,
        "consistency_mask": 0,
        "partial_entity_positive_mask": 0,
        "partial_negative_groups": -1,
        "union_groups": -1,
        "union_fine_trusted": 0,
        "secondary_labels": -100,
        "predicate_condition_ids": -1,
        "type_only_labels": -100,
        "segmentation_prefix_masks": 0,
    }
    FLOAT_AUXILIARY_PADDING = {"primary_objective_weights": 0.0}
    MATRIX_PADDING = {
        "predicate_labels": -100.0,
        "predicate_weights": 0.0,
        "reference_type_labels": -100.0,
        "reference_type_weights": 0.0,
    }
    SUBCLASS_ID_PADDING = {
        "subclass_block_ids": -1,
        "subclass_target_ids": -1,
        "subclass_scope_ids": -1,
    }
    SUBCLASS_WEIGHT_FIELDS = (
        "subclass_objective_weights",
        "subclass_learning_weights",
    )

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base = DataCollatorForTokenClassification(tokenizer)

    def __call__(self, features):
        copied = [dict(feature) for feature in features]
        # A per-row distribution is neither a scalar nor a token-aligned sequence, so it
        # must come out before the base collator sees it: that collator pads every list it
        # is handed to the token length, which would silently reshape a domain posterior
        # into nonsense of the right dtype.
        row_vectors = {}
        for field in sorted(
            {
                field
                for feature in copied
                for field in feature
                if (field.startswith("domain_") and field.endswith("_id"))
                or field in {"native_outside_allowed", "mapped_outside_allowed"}
            }
        ):
            if not all(field in feature for feature in copied):
                raise ValueError(f"{field} is missing from part of a batch")
            row_vectors[field] = [feature.pop(field) for feature in copied]
        # Per-row metadata must survive collation beside the padded token fields.
        row_scalars = {}
        for field, dtype, convert in (
            ("o_weight", torch.float, float),
            ("partial_entity_pu_group", torch.long, int),
            ("partial_entity_pu_prior", torch.float, float),
            ("partial_entity_ratio_group", torch.long, int),
            ("partial_entity_ratio_prior", torch.float, float),
        ):
            present = [(field in feature) for feature in copied]
            if any(present):
                if not all(present):
                    raise ValueError(f"{field} is missing from part of a batch")
                row_scalars[field] = (
                    [convert(feature.pop(field)) for feature in copied],
                    dtype,
                )
        auxiliary_rows = {}
        for field in self.AUXILIARY_PADDING:
            present = [field in feature for feature in copied]
            if any(present) and not all(present):
                raise ValueError(f"partial-supervision field {field!r} is missing from part of a batch")
            if all(present):
                auxiliary_rows[field] = [feature.pop(field) for feature in copied]
        float_auxiliary_rows = {}
        for field in self.FLOAT_AUXILIARY_PADDING:
            present = [field in feature for feature in copied]
            if any(present) and not all(present):
                raise ValueError(f"primary-objective field {field!r} is missing from part of a batch")
            if all(present):
                float_auxiliary_rows[field] = [feature.pop(field) for feature in copied]
        matrix_rows = {}
        for field in self.MATRIX_PADDING:
            present = [field in feature for feature in copied]
            if any(present) and not all(present):
                raise ValueError(f"token-matrix field {field!r} is missing from part of a batch")
            if all(present):
                matrix_rows[field] = [feature.pop(field) for feature in copied]
        subclass_fields = {
            *self.SUBCLASS_ID_PADDING,
            *self.SUBCLASS_WEIGHT_FIELDS,
            "subclass_token_masks",
        }
        subclass_present = [any(field in feature for field in subclass_fields) for feature in copied]
        subclass_rows = None
        subclass_token_lengths = None
        if any(subclass_present):
            if not all(subclass_present):
                raise ValueError("subclass component fields are missing from part of a batch")
            subclass_rows = {field: [feature.pop(field) for feature in copied] for field in subclass_fields}
            subclass_token_lengths = [len(feature["input_ids"]) for feature in copied]
            for row_index in range(len(copied)):
                component_counts = {len(subclass_rows[field][row_index]) for field in subclass_fields}
                if len(component_counts) != 1:
                    raise ValueError(
                        f"subclass component fields disagree in row {row_index}: {component_counts}"
                    )
        batch = self.base(copied)
        for field, rows in row_vectors.items():
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ValueError(f"{field} rows disagree in width: {sorted(widths)}")
            batch[field] = torch.tensor(rows, dtype=torch.float)
        target_length = batch["input_ids"].shape[1]
        for field, rows in auxiliary_rows.items():
            padded = []
            for row in rows:
                padding = [self.AUXILIARY_PADDING[field]] * (target_length - len(row))
                padded.append(row + padding if self.tokenizer.padding_side == "right" else padding + row)
            batch[field] = torch.tensor(padded, dtype=torch.long)
        for field, rows in float_auxiliary_rows.items():
            padded = []
            for row in rows:
                padding = [self.FLOAT_AUXILIARY_PADDING[field]] * (target_length - len(row))
                padded.append(row + padding if self.tokenizer.padding_side == "right" else padding + row)
            batch[field] = torch.tensor(padded, dtype=torch.float)
        for field, rows in matrix_rows.items():
            widths = {len(cell) for row in rows for cell in row}
            if len(widths) != 1:
                raise ValueError(f"token-matrix field {field!r} has inconsistent channel widths")
            width = widths.pop()
            padded = []
            for row in rows:
                padding = [[self.MATRIX_PADDING[field]] * width] * (target_length - len(row))
                padded.append(row + padding if self.tokenizer.padding_side == "right" else padding + row)
            batch[field] = torch.tensor(padded, dtype=torch.float)
        for field, (values, dtype) in row_scalars.items():
            batch[field] = torch.tensor(values, dtype=dtype)
        if subclass_rows is not None:
            component_count = max(
                (len(row) for row in subclass_rows["subclass_block_ids"]),
                default=0,
            )
            for field, padding_value in self.SUBCLASS_ID_PADDING.items():
                padded = [
                    row + [padding_value] * (component_count - len(row)) for row in subclass_rows[field]
                ]
                batch[field] = torch.tensor(padded, dtype=torch.long)
            for field in self.SUBCLASS_WEIGHT_FIELDS:
                padded = [row + [0.0] * (component_count - len(row)) for row in subclass_rows[field]]
                batch[field] = torch.tensor(padded, dtype=torch.float)
            if component_count:
                padded_masks = []
                for row_index, masks in enumerate(subclass_rows["subclass_token_masks"]):
                    token_count = subclass_token_lengths[row_index]
                    padded_components = []
                    for mask in masks:
                        if len(mask) != token_count:
                            raise ValueError("subclass token mask does not align with its unpadded input")
                        token_padding = [0] * (target_length - token_count)
                        padded_components.append(
                            mask + token_padding
                            if self.tokenizer.padding_side == "right"
                            else token_padding + mask
                        )
                    padded_components.extend(
                        [[0] * target_length] * (component_count - len(padded_components))
                    )
                    padded_masks.append(padded_components)
                batch["subclass_token_masks"] = torch.tensor(
                    padded_masks,
                    dtype=torch.bool,
                )
            else:
                batch["subclass_token_masks"] = torch.zeros(
                    (len(copied), 0, target_length),
                    dtype=torch.bool,
                )
        return batch


class ContinuousCharacterDataCollator:
    """Pad whole-row characters and token offsets beside a token batch."""

    CHARACTER_FIELDS = ("character_ids", "character_mask", "token_offsets")

    def __init__(self, tokenizer, base):
        self.tokenizer = tokenizer
        self.base = base

    def __call__(self, features):
        copied = [dict(feature) for feature in features]
        present = [all(field in feature for field in self.CHARACTER_FIELDS) for feature in copied]
        if not all(present):
            raise ValueError("continuous-character fields are missing from part of a tagging batch")
        character_ids = [feature.pop("character_ids") for feature in copied]
        character_masks = [feature.pop("character_mask") for feature in copied]
        token_offsets = [feature.pop("token_offsets") for feature in copied]
        batch = self.base(copied)

        character_length = max(len(row) for row in character_ids)
        padded_character_ids = []
        padded_character_masks = []
        for ids, mask in zip(character_ids, character_masks, strict=True):
            padding = character_length - len(ids)
            padded_character_ids.append(ids + [[0, 0]] * padding)
            padded_character_masks.append(mask + [False] * padding)

        token_length = batch["input_ids"].shape[1]
        padded_offsets = []
        for offsets in token_offsets:
            padding = [[0, 0]] * (token_length - len(offsets))
            padded_offsets.append(
                offsets + padding if self.tokenizer.padding_side == "right" else padding + offsets
            )
        batch["character_ids"] = torch.tensor(padded_character_ids, dtype=torch.long)
        batch["character_mask"] = torch.tensor(padded_character_masks, dtype=torch.bool)
        batch["token_offsets"] = torch.tensor(padded_offsets, dtype=torch.long)
        return batch


class JointMlmDataCollator:
    """Collate one homogeneous tag or dynamically masked physical batch."""

    def __init__(self, tokenizer, *, tag_collator, mlm_probability, replay_seed):
        self.tokenizer = tokenizer
        self.tag_collator = tag_collator
        self.mlm_probability = mlm_probability
        self.replay_seed = replay_seed
        self._mlm_collator = None
        self._fallback_generator = None

    def mlm_collator(self):
        if self._mlm_collator is None:
            # Each deterministic DataLoader worker receives a distinct torch
            # seed. Mix that with the named replay fork so masking varies by
            # worker and repeated draw without coupling to model init.
            seed = (int(torch.initial_seed()) ^ int(self.replay_seed)) % (2**63 - 1)
            self._mlm_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                seed=seed,
            )
            self._fallback_generator = torch.Generator().manual_seed(seed ^ 0x4D4C4D)
        return self._mlm_collator

    def ensure_masked_token(self, batch):
        """Guarantee a usable MLM target in small, dynamically mixed batches."""
        if torch.count_nonzero(batch["labels"] != -100):
            return batch
        eligible = batch["attention_mask"].bool()
        for token_id in self.tokenizer.all_special_ids:
            eligible &= batch["input_ids"] != token_id
        candidates = torch.nonzero(eligible, as_tuple=False)
        if not len(candidates):
            raise ValueError("MLM replay batch contains no mask-eligible tokens")
        choice = int(
            torch.randint(
                len(candidates),
                (1,),
                generator=self._fallback_generator,
            ).item()
        )
        row, column = candidates[choice]
        original_token = batch["input_ids"][row, column].clone()
        batch["labels"][row, column] = original_token
        batch["input_ids"][row, column] = self.tokenizer.mask_token_id
        return batch

    def __call__(self, features):
        tag_features = []
        mlm_features = []
        marked_objectives = []
        for feature in features:
            copied = dict(feature)
            objective = copied.pop("pii_objective", None)
            marked_objectives.append(objective is not None)
            if objective in (None, "tag"):
                tag_features.append(copied)
            elif objective == "mlm":
                mlm_features.append(copied)
            else:
                raise ValueError(f"joint-objective feature has invalid objective {objective!r}")
        if not any(marked_objectives):
            return self.tag_collator(tag_features)
        if tag_features and mlm_features:
            raise ValueError("physical training batches must not mix tagging and MLM rows")
        batch = {}
        if tag_features:
            batch["tag_batch"] = self.tag_collator(tag_features)
        if mlm_features:
            batch["mlm_batch"] = self.ensure_masked_token(self.mlm_collator()(mlm_features))
        return batch


def masked_language_model_loss(model, mlm_head, batch):
    """Evaluate a fixed pretrained MLM head over the trainable encoder."""
    labels = batch["labels"]
    masked_tokens = int(torch.count_nonzero(labels != -100).item())
    if not masked_tokens:
        raise ValueError("MLM replay batch contains no masked tokens")
    encoder_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model.base_model(**encoder_inputs, return_dict=True)
    logits = mlm_head(outputs.last_hidden_state)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return loss, logits, masked_tokens


def physical_batch_objective_loss(tag_loss, mlm_loss, mlm_loss_weight):
    """Return one physical-slot loss without renormalizing over active MLM slots."""
    if tag_loss is not None and mlm_loss is not None:
        raise ValueError("one physical batch cannot carry both tagging and MLM losses")
    if tag_loss is not None:
        return tag_loss
    if mlm_loss is not None:
        return float(mlm_loss_weight) * mlm_loss
    raise ValueError("physical batch contains neither tagging nor MLM loss")


class MlmReplayTrainerMixin:
    """Add fixed-head multilingual masked-LM batches to token classification."""

    def __init__(self, *args, mlm_head, mlm_loss_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm_head = mlm_head.requires_grad_(False).eval().to(self.args.device)
        self.mlm_loss_weight = mlm_loss_weight
        self.mlm_gradient_probe_name, self.mlm_gradient_probe = last_trainable_encoder_matrix(self.model)
        if self.mlm_gradient_probe is None:
            raise ValueError("MLM replay requires a trainable encoder matrix for scale telemetry")
        self.mlm_scale_logged_steps = set()
        self.mlm_scale_seen_objectives = set()
        self.mlm_physical_batches_seen = 0
        self.mlm_active_physical_batches_seen = 0

    def record_mlm_scale(self, tag_loss, mlm_loss, masked_tokens, tag_rows, mlm_rows):
        self.mlm_physical_batches_seen += 1
        self.mlm_active_physical_batches_seen += int(mlm_loss is not None)
        step = int(self.state.global_step)
        logging_steps = max(1, int(self.args.logging_steps))
        observed = set()
        if tag_loss is not None:
            observed.add("tag")
        if mlm_loss is not None:
            observed.add("mlm")
        first_observation = bool(observed - self.mlm_scale_seen_objectives)
        if not first_observation and (step in self.mlm_scale_logged_steps or step % logging_steps):
            return
        tag_gradient_norm = (
            None if tag_loss is None else loss_parameter_gradient_norm(tag_loss, self.mlm_gradient_probe)
        )
        mlm_gradient_norm = (
            None if mlm_loss is None else loss_parameter_gradient_norm(mlm_loss, self.mlm_gradient_probe)
        )
        record = {
            "schema_version": 2,
            "step": step,
            "tag_loss": None if tag_loss is None else float(tag_loss.detach().float().cpu()),
            "mlm_loss": None if mlm_loss is None else float(mlm_loss.detach().float().cpu()),
            "mlm_loss_weight": float(self.mlm_loss_weight),
            "weighted_mlm_loss": (
                None if mlm_loss is None else float((self.mlm_loss_weight * mlm_loss).detach().float().cpu())
            ),
            "mlm_masked_tokens": masked_tokens,
            "tag_rows": tag_rows,
            "mlm_rows": mlm_rows,
            "mlm_row_fraction": mlm_rows / (tag_rows + mlm_rows),
            "physical_batches_seen": self.mlm_physical_batches_seen,
            "mlm_active_physical_batches_seen": self.mlm_active_physical_batches_seen,
            "mlm_active_physical_batch_fraction": (
                self.mlm_active_physical_batches_seen / self.mlm_physical_batches_seen
            ),
            "encoder_gradient_probe": self.mlm_gradient_probe_name,
            "tag_encoder_gradient_norm": tag_gradient_norm,
            "mlm_encoder_gradient_norm": mlm_gradient_norm,
            "weighted_mlm_encoder_gradient_norm": (
                None if mlm_gradient_norm is None else self.mlm_loss_weight * mlm_gradient_norm
            ),
        }
        path = Path(self.args.output_dir) / "mlm_objective_scale.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "TRAIN-MLM-SCALE: "
            f"step={step} tag-loss={record['tag_loss']} mlm-loss={record['mlm_loss']} "
            f"mlm-weight={self.mlm_loss_weight:g} rows={tag_rows}+{mlm_rows} "
            f"masked-tokens={masked_tokens} probe={self.mlm_gradient_probe_name} "
            f"tag-grad={tag_gradient_norm} mlm-grad={mlm_gradient_norm}",
            flush=True,
        )
        self.mlm_scale_logged_steps.add(step)
        self.mlm_scale_seen_objectives.update(observed)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        tag_batch = inputs.pop("tag_batch", None)
        mlm_batch = inputs.pop("mlm_batch", None)
        if tag_batch is None and mlm_batch is None:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        tag_loss = None
        tag_outputs = None
        tag_rows = 0
        if tag_batch is not None:
            tag_rows = int(tag_batch["input_ids"].shape[0])
            tag_loss, tag_outputs = super().compute_loss(
                model,
                tag_batch,
                return_outputs=True,
                num_items_in_batch=None,
            )
        mlm_loss = None
        mlm_logits = None
        mlm_rows = 0
        if mlm_batch is not None:
            mlm_rows = int(mlm_batch["input_ids"].shape[0])
            mlm_loss, mlm_logits, masked_tokens = masked_language_model_loss(
                model,
                self.mlm_head,
                mlm_batch,
            )
        self.record_mlm_scale(
            tag_loss,
            mlm_loss,
            masked_tokens if mlm_loss is not None else 0,
            tag_rows,
            mlm_rows,
        )

        loss = physical_batch_objective_loss(tag_loss, mlm_loss, self.mlm_loss_weight)
        outputs = tag_outputs if tag_outputs is not None else {"logits": mlm_logits}
        return (loss, outputs) if return_outputs else loss


class PartialSupervisionTrainer(Trainer):
    """Apply training-only objectives whose masks come from partial-label rows."""

    def __init__(
        self,
        *args,
        boundary_label_names,
        boundary_loss_weight,
        retention_teacher,
        retention_loss_weight,
        retention_temperature,
        partial_o_loss_weight,
        o_label_id,
        partial_negative_label_groups,
        partial_negative_loss_weight,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.boundary_label_names = boundary_label_names
        self.boundary_loss_weight = boundary_loss_weight
        self.retention_teacher = retention_teacher
        self.retention_loss_weight = retention_loss_weight
        self.retention_temperature = retention_temperature
        self.partial_o_loss_weight = partial_o_loss_weight
        self.o_label_id = o_label_id
        self.partial_negative_label_groups = partial_negative_label_groups
        self.partial_negative_loss_weight = partial_negative_loss_weight
        if self.retention_teacher is not None:
            self.retention_teacher.requires_grad_(False)
            self.retention_teacher.eval()
            self.retention_teacher.to(self.args.device)
        self.objective_scale_logged_steps = set()

    def record_objective_scale(
        self,
        ordinary_loss: torch.Tensor,
        boundary_loss: torch.Tensor | None,
        retention_loss: torch.Tensor | None,
        partial_o_token_loss: torch.Tensor | None,
        negative_loss: torch.Tensor | None,
        logits: torch.Tensor,
        labels: torch.Tensor,
        boundary_labels: torch.Tensor | None,
        consistency_mask: torch.Tensor | None,
        partial_negative_groups: torch.Tensor | None,
    ) -> None:
        """Sparsely persist component scales for comparable objective contrasts."""
        step = int(self.state.global_step)
        logging_steps = max(1, int(self.args.logging_steps))
        boundary_tokens = (
            int(torch.count_nonzero(boundary_labels != -100).item()) if boundary_labels is not None else 0
        )
        retention_tokens = (
            int(torch.count_nonzero(consistency_mask).item()) if consistency_mask is not None else 0
        )
        negative_tokens = (
            int(torch.count_nonzero(partial_negative_groups >= 0).item())
            if partial_negative_groups is not None
            else 0
        )
        if (
            step in self.objective_scale_logged_steps
            or step % logging_steps
            or not (boundary_tokens or retention_tokens or negative_tokens)
        ):
            return
        # Some stock Transformers heads compute ``outputs.loss`` through a
        # tensor distinct from the returned ``outputs.logits``.  Reconstruct
        # the same token CE here so its local logit gradient is observable;
        # the model-provided loss remains the actual training objective.
        ordinary_logit_loss = token_classification_loss(logits, labels)
        ordinary_gradient_norm = loss_logit_gradient_norm(ordinary_logit_loss, logits)
        record = {
            "schema_version": 4,
            "step": step,
            "ordinary_loss": float(ordinary_loss.detach().float().cpu()),
            "ordinary_logit_loss": float(ordinary_logit_loss.detach().float().cpu()),
            "ordinary_supervised_tokens": int(torch.count_nonzero(labels != -100).item()),
            "boundary_supervised_tokens": boundary_tokens,
            "retention_supervised_tokens": retention_tokens,
            "partial_negative_supervised_tokens": negative_tokens,
            "ordinary_logit_gradient_norm": ordinary_gradient_norm,
        }
        if boundary_loss is not None:
            boundary_gradient_norm = loss_logit_gradient_norm(boundary_loss, logits)
            record.update(
                {
                    "boundary_loss": float(boundary_loss.detach().float().cpu()),
                    "boundary_loss_weight": float(self.boundary_loss_weight),
                    "weighted_boundary_loss": float(
                        (self.boundary_loss_weight * boundary_loss).detach().float().cpu()
                    ),
                    "boundary_logit_gradient_norm": boundary_gradient_norm,
                    "weighted_boundary_logit_gradient_norm": (
                        self.boundary_loss_weight * boundary_gradient_norm
                    ),
                }
            )
        if retention_loss is not None:
            retention_gradient_norm = loss_logit_gradient_norm(retention_loss, logits)
            record.update(
                {
                    "retention_loss": float(retention_loss.detach().float().cpu()),
                    "retention_loss_weight": float(self.retention_loss_weight),
                    "weighted_retention_loss": float(
                        (self.retention_loss_weight * retention_loss).detach().float().cpu()
                    ),
                    "retention_logit_gradient_norm": retention_gradient_norm,
                    "weighted_retention_logit_gradient_norm": (
                        self.retention_loss_weight * retention_gradient_norm
                    ),
                }
            )
        if partial_o_token_loss is not None:
            partial_o_gradient_norm = loss_logit_gradient_norm(partial_o_token_loss, logits)
            record.update(
                {
                    "partial_o_loss": float(partial_o_token_loss.detach().float().cpu()),
                    "partial_o_loss_weight": float(self.partial_o_loss_weight),
                    "weighted_partial_o_loss": float(
                        (self.partial_o_loss_weight * partial_o_token_loss).detach().float().cpu()
                    ),
                    "partial_o_logit_gradient_norm": partial_o_gradient_norm,
                    "weighted_partial_o_logit_gradient_norm": (
                        self.partial_o_loss_weight * partial_o_gradient_norm
                    ),
                }
            )
        if negative_loss is not None:
            negative_gradient_norm = loss_logit_gradient_norm(negative_loss, logits)
            record.update(
                {
                    "partial_negative_loss": float(negative_loss.detach().float().cpu()),
                    "partial_negative_loss_weight": float(self.partial_negative_loss_weight),
                    "weighted_partial_negative_loss": float(
                        (self.partial_negative_loss_weight * negative_loss).detach().float().cpu()
                    ),
                    "partial_negative_logit_gradient_norm": negative_gradient_norm,
                    "weighted_partial_negative_logit_gradient_norm": (
                        self.partial_negative_loss_weight * negative_gradient_norm
                    ),
                }
            )
        path = Path(self.args.output_dir) / "objective_scale.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            "TRAIN-SCALE: "
            f"step={step} ordinary-loss={record['ordinary_loss']:.4f} "
            f"ordinary-tokens={record['ordinary_supervised_tokens']} "
            f"boundary-tokens={boundary_tokens} "
            f"retention-tokens={retention_tokens} "
            f"partial-negative-tokens={negative_tokens} "
            f"ordinary-logit-grad={ordinary_gradient_norm:.6f} "
            f"boundary-loss={record.get('boundary_loss', 0):.4f} "
            f"boundary-weight={self.boundary_loss_weight:g} "
            f"retention-loss={record.get('retention_loss', 0):.4f} "
            f"retention-weight={self.retention_loss_weight:g} "
            f"partial-o-loss={record.get('partial_o_loss', 0):.4f} "
            f"partial-o-weight={self.partial_o_loss_weight:g} "
            f"partial-negative-loss={record.get('partial_negative_loss', 0):.4f} "
            f"partial-negative-weight={self.partial_negative_loss_weight:g}",
            flush=True,
        )
        self.objective_scale_logged_steps.add(step)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        boundary_labels = inputs.pop("boundary_labels", None)
        consistency_mask = inputs.pop("consistency_mask", None)
        partial_negative_groups = inputs.pop("partial_negative_groups", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        ordinary_loss = outputs.loss
        loss = ordinary_loss
        if model.training:
            boundary_loss = (
                annotated_boundary_loss(
                    outputs.logits,
                    boundary_labels,
                    self.boundary_label_names,
                )
                if boundary_labels is not None
                else None
            )
            retention_loss = None
            if consistency_mask is not None and torch.any(consistency_mask):
                if self.retention_teacher is not None:
                    teacher_inputs = {
                        key: value
                        for key, value in inputs.items()
                        if key not in {"labels", "character_ids", "character_mask", "token_offsets"}
                    }
                    with torch.no_grad():
                        teacher_logits = self.retention_teacher(**teacher_inputs).logits
                    retention_loss = partial_boundary_retention_loss(
                        outputs.logits,
                        teacher_logits,
                        consistency_mask,
                        self.boundary_label_names,
                        self.retention_temperature,
                    )
            partial_o_token_loss = (
                partial_o_loss(outputs.logits, consistency_mask, self.o_label_id)
                if consistency_mask is not None and self.partial_o_loss_weight
                else None
            )
            negative_loss = (
                partial_negative_loss(
                    outputs.logits,
                    partial_negative_groups,
                    self.partial_negative_label_groups,
                )
                if partial_negative_groups is not None
                else None
            )
            self.record_objective_scale(
                ordinary_loss,
                boundary_loss,
                retention_loss,
                partial_o_token_loss,
                negative_loss,
                outputs.logits,
                labels,
                boundary_labels,
                consistency_mask,
                partial_negative_groups,
            )
            if boundary_loss is not None:
                loss = loss + self.boundary_loss_weight * boundary_loss
            if retention_loss is not None:
                loss = loss + self.retention_loss_weight * retention_loss
            if partial_o_token_loss is not None:
                loss = loss + self.partial_o_loss_weight * partial_o_token_loss
            if negative_loss is not None:
                loss = loss + self.partial_negative_loss_weight * negative_loss
        return (loss, outputs) if return_outputs else loss


class LengthAuditCollator:
    """Measure each item's encoded length against the length it was batched by.

    Length-selective physical-batch sampling is the last preparation step
    (process map, training batch preparation inside FIT); a later change of
    more than TOLERANCE of an item's length mis-bins it. Collation may run in
    data-loader workers, so the per-batch counts travel with the batch as
    `pii_length_audit` = [audited items, items beyond tolerance, max relative
    deviation] for the trainer to strip and report.
    """

    TOLERANCE = 0.10

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        copied = [dict(feature) for feature in features]
        binned = [feature.pop("pii_binned_length", None) for feature in copied]
        deviations = [
            abs(len(feature["input_ids"]) - length) / length
            for feature, length in zip(copied, binned, strict=True)
            if length
        ]
        batch = self.base(copied)
        batch["pii_length_audit"] = torch.tensor(
            [len(deviations), sum(d > self.TOLERANCE for d in deviations), max(deviations, default=0.0)],
            dtype=torch.float64,
        )
        return batch


class WeightedSamplingTrainerMixin:
    """Use row-weighted batch routing and optional padding optimization."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        audit = inputs.pop("pii_length_audit", None)
        if audit is not None:
            items, violations, deviation = audit.tolist()
            totals = self.__dict__.setdefault("_length_audit", Counter())
            totals["batches"] += 1
            totals["items"] += int(items)
            totals["violations"] += int(violations)
            totals["violating_batches"] += bool(violations)
            self._length_audit_max = max(getattr(self, "_length_audit_max", 0.0), deviation)
            if violations:
                print(
                    f"TRAIN-LENGTH-AUDIT: warning step={self.state.global_step} {int(violations)}/{int(items)} "
                    f"items changed length by more than {LengthAuditCollator.TOLERANCE:.0%} after batching "
                    f"(max {deviation:.1%})",
                    flush=True,
                )
        return super().training_step(model, inputs, num_items_in_batch)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        totals = getattr(self, "_length_audit", None)
        if totals:
            share = totals["violations"] / max(totals["items"], 1)
            status = "warning" if totals["violations"] else "ok"
            print(
                f"TRAIN-LENGTH-AUDIT: summary {status} items={totals['items']} batches={totals['batches']} "
                f"beyond {LengthAuditCollator.TOLERANCE:.0%}: items={totals['violations']} ({share:.2%}) "
                f"batches={totals['violating_batches']} max_deviation={self._length_audit_max:.1%}",
                flush=True,
            )
        return result

    def get_train_dataloader(self):
        dataset = self.train_dataset
        weights = getattr(dataset, "sampling_weights", None)
        if weights is None:
            return super().get_train_dataloader()
        if dataset is None:
            raise ValueError("Trainer: training requires a train_dataset")
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise ValueError("weighted sampling requires a sized, indexable train dataset")
        data_collator = LengthAuditCollator(
            self._get_collator_with_removed_columns(
                self.data_collator,
                description="Training",
            )
        )
        common_sampler_args = {
            "lengths": dataset.token_lengths(),
            "weights": weights,
            "batch_size": self._train_batch_size,
            "gradient_accumulation_steps": int(self.args.gradient_accumulation_steps),
            "seed": int(getattr(self.args, "pii_sampling_seed", self.args.seed)),
            "epoch_examples": int(getattr(self.args, "sampling_epoch_examples", 0) or len(dataset)),
        }
        mlm_batch_probabilities = getattr(dataset, "mlm_batch_probabilities", None)
        if getattr(dataset, "requires_draw_nonce", False):
            batch_sampler = NativeHeadBatchSampler(
                dataset,
                partial(
                    WeightedLengthBatchSampler,
                    **common_sampler_args,
                    length_window_steps=int(
                        getattr(
                            self.args, "sampling_length_window_steps", DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS
                        )
                    ),
                    batch_formation=getattr(self.args, "sampling_batch_formation", "sorted-window"),
                    padding_budget=getattr(self.args, "sampling_padding_budget", 0.10),
                    draw_policy=getattr(self.args, "sampling_draw_policy", "systematic"),
                ),
                seed=common_sampler_args["seed"],
                epoch_examples=common_sampler_args["epoch_examples"],
                batch_size=self._train_batch_size,
            )
        elif mlm_batch_probabilities is not None:
            pool_keys = getattr(dataset, "sampling_pool_keys", None)
            if pool_keys is None:
                raise ValueError("physical-batch MLM sampling requires dataset pool keys")
            batch_sampler = RecursiveWeightedBatchSampler(
                **common_sampler_args,
                pool_keys=pool_keys,
                length_bucket_width=int(
                    getattr(
                        self.args,
                        "sampling_length_bucket_width",
                        DEFAULT_RECURSIVE_LENGTH_BUCKET_WIDTH,
                    )
                ),
                variant_probability_by_pool=mlm_batch_probabilities,
            )
        else:
            batch_sampler = WeightedLengthBatchSampler(
                **common_sampler_args,
                length_window_steps=int(
                    getattr(
                        self.args,
                        "sampling_length_window_steps",
                        DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS,
                    )
                ),
                batch_formation=getattr(self.args, "sampling_batch_formation", "sorted-window"),
                padding_budget=getattr(self.args, "sampling_padding_budget", 0.10),
                draw_policy=getattr(self.args, "sampling_draw_policy", "systematic"),
            )
        if getattr(dataset, "surface_realizer", None) is not None and not getattr(
            dataset, "requires_draw_nonce", False
        ):
            batch_sampler = SurfaceDrawBatchSampler(batch_sampler)
        if not getattr(self, "_weighted_sampler_logged", False):
            print(f"TRAIN-SAMPLER: {batch_sampler.summary()}", flush=True)
            self._weighted_sampler_logged = True
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))


def differential_learning_rate_groups(
    model,
    decay_parameter_names,
    *,
    head_lr,
    encoder_lr,
    weight_decay,
    character_lr=None,
    character_output_lr=None,
    register_lr=None,
    prompt_lr=None,
    block_lr=None,
    require_encoder_parameters=True,
):
    """Split encoder, character trunk/readout, and remaining head parameters by rate."""
    # A freshly inserted residual head block needs a fresh-layer rate, not the
    # rate tuned for the already-trained head it feeds.
    block = getattr(model, "head_residual_mlp", None)
    if block_lr is not None and block is None:
        raise ValueError("a residual head block learning rate requires an installed block")
    block_parameter_ids = (
        {id(parameter) for parameter in block.parameters()} if block_lr is not None else set()
    )
    encoder = model.base_model
    if encoder is model:
        raise ValueError("differential learning rates require a separate base encoder")
    encoder_parameter_ids = {id(parameter) for parameter in encoder.parameters()}
    character_encoder = getattr(model, "character_encoder", None)
    if character_lr is not None and character_encoder is None:
        raise ValueError("a character learning rate requires model.character_encoder")
    character_parameter_ids = (
        {id(parameter) for parameter in character_encoder.parameters()} if character_lr is not None else set()
    )
    # Register parameters start at or near zero and would otherwise fall into the task-head
    # group, whose rate is tuned for a head that is already trained. Give them their own.
    registers = getattr(model, "pii_soft_registers", None)
    register_parameter_ids = (
        {id(parameter) for parameter in registers.parameters()}
        if register_lr is not None and registers is not None
        else set()
    )
    if register_lr is not None and registers is None:
        raise ValueError("a register learning rate requires installed soft registers")
    # Prompt slots are new input embeddings; like registers they need their own, higher rate.
    prompt_slots = getattr(model, "pii_prompt_slots", None)
    prompt_parameter_ids = (
        {id(parameter) for parameter in prompt_slots.parameters()}
        if prompt_lr is not None and prompt_slots is not None
        else set()
    )
    if prompt_lr is not None and prompt_slots is None:
        raise ValueError("a prompt-slot learning rate requires installed prompt slots")
    character_classifier = getattr(model, "character_classifier", None)
    character_output_parameter_ids = (
        {id(parameter) for parameter in character_classifier.parameters()}
        if character_classifier is not None
        else set()
    )
    grouped = {
        ("encoder", True): [],
        ("encoder", False): [],
        ("character", True): [],
        ("character", False): [],
        ("character-output", True): [],
        ("character-output", False): [],
        ("register", True): [],
        ("register", False): [],
        ("prompt", True): [],
        ("prompt", False): [],
        ("block", True): [],
        ("block", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in register_parameter_ids:
            scope = "register"
        elif id(parameter) in prompt_parameter_ids:
            scope = "prompt"
        elif id(parameter) in block_parameter_ids:
            scope = "block"
        elif id(parameter) in encoder_parameter_ids:
            scope = "encoder"
        elif id(parameter) in character_parameter_ids:
            scope = "character"
        elif id(parameter) in character_output_parameter_ids:
            scope = "character-output"
        else:
            scope = "head"
        decay = name in decay_parameter_names and scope not in ("character-output", "prompt")
        grouped[(scope, decay)].append(parameter)
    if require_encoder_parameters and not any(grouped[("encoder", decay)] for decay in (True, False)):
        raise ValueError("--encoder-lr requires trainable encoder parameters")
    if not any(grouped[("head", decay)] for decay in (True, False)):
        raise ValueError("--encoder-lr requires trainable task-head parameters")
    if character_lr is not None and not any(grouped[("character", decay)] for decay in (True, False)):
        raise ValueError("--continuous-character-lr requires trainable character parameters")
    if character_output_lr is not None and not grouped[("character-output", False)]:
        raise ValueError("--continuous-character-output-lr requires model.character_classifier")
    groups = []
    scope_rates = [("encoder", encoder_lr)]
    if character_lr is not None:
        scope_rates.append(("character", character_lr))
    if character_output_parameter_ids:
        scope_rates.append(
            (
                "character-output",
                character_output_lr
                if character_output_lr is not None
                else character_lr
                if character_lr is not None
                else head_lr,
            )
        )
    if register_parameter_ids:
        scope_rates.append(("register", register_lr))
    if prompt_parameter_ids:
        scope_rates.append(("prompt", prompt_lr))
    if block_parameter_ids:
        scope_rates.append(("block", block_lr))
    scope_rates.append(("head", head_lr))
    for scope, learning_rate in scope_rates:
        for decay in (True, False):
            parameters = grouped[(scope, decay)]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": weight_decay if decay else 0.0,
                        "group_name": f"{scope}-{'decay' if decay else 'no-decay'}",
                    }
                )
    return groups


def encoder_parameter_prior_anchors(model, anchor_model):
    """Copy trainable encoder parameters from a shape-compatible anchor model."""
    encoder = model.base_model
    anchor_encoder = anchor_model.base_model
    if encoder is model or anchor_encoder is anchor_model:
        raise ValueError("encoder parameter prior requires separate base encoders")
    current = dict(encoder.named_parameters())
    anchored = dict(anchor_encoder.named_parameters())
    if current.keys() != anchored.keys():
        missing = sorted(current.keys() - anchored.keys())
        extra = sorted(anchored.keys() - current.keys())
        raise ValueError(
            f"encoder parameter prior names do not match: missing={missing[:3]} extra={extra[:3]}"
        )
    anchors = {}
    for name, parameter in current.items():
        if not parameter.requires_grad:
            continue
        anchor = anchored[name]
        if parameter.shape != anchor.shape:
            raise ValueError(
                f"encoder parameter prior shape mismatch for {name}: "
                f"model={tuple(parameter.shape)} anchor={tuple(anchor.shape)}"
            )
        anchors[name] = anchor.detach().cpu().clone()
    if not anchors:
        raise ValueError("encoder parameter prior requires trainable encoder parameters")
    return anchors


def load_encoder_prior_model(checkpoint):
    """Load an auxiliary anchor without shifting the training RNG stream."""
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            return load_local_token_classifier(checkpoint)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)


def encoder_parameter_prior_loss(model, anchors):
    """Half squared L2 distance from a fixed encoder checkpoint."""
    current = dict(model.base_model.named_parameters())
    missing = sorted(set(anchors) - set(current))
    if missing:
        raise ValueError(f"encoder parameter prior is missing model parameters: {missing[:3]}")
    loss = None
    parameter_count = 0
    for name, anchor in anchors.items():
        parameter = current[name]
        difference = parameter.float() - anchor.float()
        term = difference.square().sum()
        loss = term if loss is None else loss + term
        parameter_count += parameter.numel()
    if loss is None:
        raise ValueError("encoder parameter prior has no parameters")
    return 0.5 * loss, parameter_count


def inherited_output_parameter_prior_anchors(model, anchor_model, rows):
    """Copy the inherited classifier prefix from a shape-compatible anchor."""
    classifier_name, classifier = token_classifier_head(model)
    anchor_classifier_name, anchor_classifier = token_classifier_head(anchor_model)
    if classifier_name != anchor_classifier_name:
        raise ValueError(
            "inherited-output prior classifier names do not match: "
            f"model={classifier_name} anchor={anchor_classifier_name}"
        )
    if not 0 < rows < classifier.out_features:
        raise ValueError(
            f"inherited-output prior rows must be in [1, {classifier.out_features - 1}], got {rows}"
        )
    if classifier.weight.shape != anchor_classifier.weight.shape:
        raise ValueError(
            "inherited-output prior classifier weight shape mismatch: "
            f"model={tuple(classifier.weight.shape)} anchor={tuple(anchor_classifier.weight.shape)}"
        )
    anchors = {"weight": anchor_classifier.weight[:rows].detach().cpu().clone()}
    if (classifier.bias is None) != (anchor_classifier.bias is None):
        raise ValueError("inherited-output prior classifier bias presence differs")
    if classifier.bias is not None:
        if classifier.bias.shape != anchor_classifier.bias.shape:
            raise ValueError(
                "inherited-output prior classifier bias shape mismatch: "
                f"model={tuple(classifier.bias.shape)} anchor={tuple(anchor_classifier.bias.shape)}"
            )
        anchors["bias"] = anchor_classifier.bias[:rows].detach().cpu().clone()
    return anchors


def inherited_output_parameter_prior_loss(model, anchors, rows):
    """Half squared L2 distance of inherited classifier rows from an anchor."""
    _classifier_name, classifier = token_classifier_head(model)
    parameters = {"weight": classifier.weight}
    if classifier.bias is not None:
        parameters["bias"] = classifier.bias
    if set(parameters) != set(anchors):
        raise ValueError(
            "inherited-output prior parameter inventory differs: "
            f"model={sorted(parameters)} anchor={sorted(anchors)}"
        )
    loss = None
    parameter_count = 0
    for name, parameter in parameters.items():
        anchor = anchors[name]
        if tuple(anchor.shape) != tuple(parameter[:rows].shape):
            raise ValueError(
                f"inherited-output prior shape mismatch for {name}: "
                f"model={tuple(parameter[:rows].shape)} anchor={tuple(anchor.shape)}"
            )
        difference = parameter[:rows].float() - anchor.float()
        term = difference.square().sum()
        loss = term if loss is None else loss + term
        parameter_count += difference.numel()
    if loss is None:
        raise ValueError("inherited-output prior has no parameters")
    return 0.5 * loss, parameter_count


class EncoderParameterPriorTrainerMixin:
    """Apply an L2-SP-style prior to an adapted encoder checkpoint."""

    def __init__(
        self,
        *args,
        encoder_prior_anchors,
        encoder_prior_weight,
        encoder_prior_start_step,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if encoder_prior_weight <= 0:
            raise ValueError("encoder parameter prior weight must be positive")
        self.encoder_prior_weight = float(encoder_prior_weight)
        self.encoder_prior_start_step = int(encoder_prior_start_step)
        current = dict(self.model.base_model.named_parameters())
        self.encoder_prior_anchors = {
            name: anchor.to(device=current[name].device, dtype=current[name].dtype)
            for name, anchor in encoder_prior_anchors.items()
        }
        self.encoder_prior_logged_steps = set()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        step = effective_training_step(self)
        if model.training and step >= self.encoder_prior_start_step:
            prior_loss, parameter_count = encoder_parameter_prior_loss(
                model,
                self.encoder_prior_anchors,
            )
            prior_scale = (
                1.0 / num_items_in_batch.physical_batches
                if isinstance(num_items_in_batch, LogicalStepObjectiveMasses)
                else 1.0
            )
            loss = loss + prior_scale * self.encoder_prior_weight * prior_loss
            logging_steps = max(1, int(self.args.logging_steps))
            if step not in self.encoder_prior_logged_steps and (
                step == self.encoder_prior_start_step or step % logging_steps == 0
            ):
                squared_distance = float((2.0 * prior_loss).detach().cpu())
                record = {
                    "schema_version": 1,
                    "step": step,
                    "encoder_parameters": parameter_count,
                    "squared_l2_distance": squared_distance,
                    "rms_parameter_distance": math.sqrt(squared_distance / parameter_count),
                    "prior_loss": float(prior_loss.detach().cpu()),
                    "prior_weight": self.encoder_prior_weight,
                    "weighted_prior_loss": float((self.encoder_prior_weight * prior_loss).detach().cpu()),
                    "prior_start_step": self.encoder_prior_start_step,
                }
                path = Path(self.args.output_dir) / "encoder_parameter_prior.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "TRAIN-ENCODER-PRIOR: "
                    f"step={step} weight={self.encoder_prior_weight:g} "
                    f"rms-distance={record['rms_parameter_distance']:.8g} "
                    f"weighted-loss={record['weighted_prior_loss']:.8g}",
                    flush=True,
                )
                self.encoder_prior_logged_steps.add(step)
        return (loss, outputs) if return_outputs else loss


class InheritedOutputParameterPriorTrainerMixin:
    """Apply an L2-SP-style prior to the old-the production toolkit classifier prefix."""

    def __init__(
        self,
        *args,
        inherited_output_prior_anchors,
        inherited_output_prior_rows,
        inherited_output_prior_weight,
        inherited_output_prior_start_step,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if inherited_output_prior_weight <= 0:
            raise ValueError("inherited-output parameter prior weight must be positive")
        self.inherited_output_prior_rows = int(inherited_output_prior_rows)
        self.inherited_output_prior_weight = float(inherited_output_prior_weight)
        self.inherited_output_prior_start_step = int(inherited_output_prior_start_step)
        _classifier_name, classifier = token_classifier_head(self.model)
        parameters = {"weight": classifier.weight}
        if classifier.bias is not None:
            parameters["bias"] = classifier.bias
        self.inherited_output_prior_anchors = {
            name: anchor.to(device=parameters[name].device, dtype=parameters[name].dtype)
            for name, anchor in inherited_output_prior_anchors.items()
        }
        self.inherited_output_prior_logged_steps = set()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        step = effective_training_step(self)
        if model.training and step >= self.inherited_output_prior_start_step:
            prior_loss, parameter_count = inherited_output_parameter_prior_loss(
                model,
                self.inherited_output_prior_anchors,
                self.inherited_output_prior_rows,
            )
            prior_scale = (
                1.0 / num_items_in_batch.physical_batches
                if isinstance(num_items_in_batch, LogicalStepObjectiveMasses)
                else 1.0
            )
            loss = loss + prior_scale * self.inherited_output_prior_weight * prior_loss
            logging_steps = max(1, int(self.args.logging_steps))
            if step not in self.inherited_output_prior_logged_steps and (
                step == self.inherited_output_prior_start_step or step % logging_steps == 0
            ):
                squared_distance = float((2.0 * prior_loss).detach().cpu())
                record = {
                    "schema_version": 1,
                    "step": step,
                    "inherited_output_rows": self.inherited_output_prior_rows,
                    "parameters": parameter_count,
                    "squared_l2_distance": squared_distance,
                    "rms_parameter_distance": math.sqrt(squared_distance / parameter_count),
                    "prior_loss": float(prior_loss.detach().cpu()),
                    "prior_weight": self.inherited_output_prior_weight,
                    "weighted_prior_loss": float(
                        (self.inherited_output_prior_weight * prior_loss).detach().cpu()
                    ),
                    "prior_start_step": self.inherited_output_prior_start_step,
                }
                path = Path(self.args.output_dir) / "inherited_output_parameter_prior.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                print(
                    "TRAIN-INHERITED-OUTPUT-PRIOR: "
                    f"step={step} rows={self.inherited_output_prior_rows} "
                    f"weight={self.inherited_output_prior_weight:g} "
                    f"rms-distance={record['rms_parameter_distance']:.8g} "
                    f"weighted-loss={record['weighted_prior_loss']:.8g}",
                    flush=True,
                )
                self.inherited_output_prior_logged_steps.add(step)
        return (loss, outputs) if return_outputs else loss


class DifferentialLearningRateTrainerMixin:
    """Use ``--lr`` for the task head and a separate rate for the encoder."""

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model if model is None else model
        encoder_lr = getattr(self.args, "pii_encoder_learning_rate", None)
        character_lr = getattr(self.args, "pii_character_learning_rate", None)
        character_output_lr = getattr(self.args, "pii_character_output_learning_rate", None)
        register_lr = getattr(self.args, "pii_register_learning_rate", None)
        prompt_lr = getattr(self.args, "pii_prompt_learning_rate", None)
        block_lr = getattr(self.args, "pii_head_residual_mlp_learning_rate", None)
        if (
            encoder_lr is None
            and character_lr is None
            and character_output_lr is None
            and register_lr is None
            and prompt_lr is None
            and block_lr is None
        ):
            return super().create_optimizer(model=model)
        optimizer_grouped_parameters = differential_learning_rate_groups(
            opt_model,
            self.get_decay_parameter_names(opt_model),
            head_lr=float(self.args.learning_rate),
            register_lr=getattr(self.args, "pii_register_learning_rate", None),
            prompt_lr=prompt_lr,
            block_lr=None if block_lr is None else float(block_lr),
            encoder_lr=float(encoder_lr if encoder_lr is not None else self.args.learning_rate),
            weight_decay=float(self.args.weight_decay),
            character_lr=None if character_lr is None else float(character_lr),
            character_output_lr=(None if character_output_lr is None else float(character_output_lr)),
            require_encoder_parameters=encoder_lr is not None,
        )
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        unsupported = {"params", "model", "optimizer_dict"}.intersection(optimizer_kwargs)
        if unsupported:
            raise ValueError(
                "--encoder-lr is incompatible with optimizer-supplied parameter groups: "
                + ", ".join(sorted(unsupported))
            )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


class WeightedSamplingTrainer(WeightedSamplingTrainerMixin, Trainer):
    pass


class WeightedPartialSupervisionTrainer(WeightedSamplingTrainerMixin, PartialSupervisionTrainer):
    pass


class RollingTrainLossCallback(TrainerCallback):
    """Record recent loss averaged over an approximate sampler-epoch window."""

    def __init__(self, sampling_epoch_steps: int, window_epochs: float = 1.0):
        if sampling_epoch_steps <= 0:
            raise ValueError("sampling epoch steps must be positive")
        if window_epochs <= 0:
            raise ValueError("train-loss window epochs must be positive")
        self.sampling_epoch_steps = int(sampling_epoch_steps)
        self.window_steps = max(1, round(self.sampling_epoch_steps * float(window_epochs)))
        self.intervals = []
        self.last_step = 0

    def observe(self, step: int, loss: float):
        step = int(step)
        if step <= self.last_step:
            return None
        self.intervals.append((self.last_step, step, float(loss)))
        self.last_step = step
        cutoff = step - self.window_steps
        self.intervals = [entry for entry in self.intervals if entry[1] > cutoff]
        weighted_loss = 0.0
        coverage_steps = 0
        for start, end, value in self.intervals:
            overlap = end - max(start, cutoff)
            weighted_loss += overlap * value
            coverage_steps += overlap
        return weighted_loss / coverage_steps, coverage_steps

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        for entry in state.log_history:
            if "loss" in entry and "step" in entry:
                self.observe(entry["step"], entry["loss"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        del control, kwargs
        if not logs or "loss" not in logs:
            return
        observed = self.observe(state.global_step, logs["loss"])
        if observed is None:
            return
        rolling_loss, coverage_steps = observed
        logs["rolling_sampling_epoch_loss"] = rolling_loss
        logs["rolling_sampling_epoch_coverage"] = coverage_steps / self.sampling_epoch_steps
        point = {
            "schema_version": 1,
            "step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "recent_loss": float(logs["loss"]),
            "rolling_sampling_epoch_loss": rolling_loss,
            "rolling_sampling_epoch_coverage": coverage_steps / self.sampling_epoch_steps,
            "sampling_epoch_steps": self.sampling_epoch_steps,
            "window_steps": self.window_steps,
        }
        path = Path(args.output_dir) / "training_loss_curve.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(point, sort_keys=True) + "\n")


class HeadlineCallback(TrainerCallback):
    """Overwrite AGENTCTL_HEADLINE_FILE with live progress (repo run policy)."""

    def __init__(self, tag, head_spec, effective_batch):
        self.tag = tag
        self.head_spec = head_spec
        self.effective_batch = effective_batch

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            rolling = logs.get("rolling_sampling_epoch_loss")
            rolling_text = "" if rolling is None else f" epoch-loss~={rolling:.4f}"
            log_format.headline(
                f"{self.tag}: step={state.global_step}/{state.max_steps} "
                f"epoch={state.epoch:.2f} loss={logs['loss']:.4f}{rolling_text} "
                f"lr={logs.get('learning_rate', 0):.2e} {self.head_spec} "
                f"eff-batch={self.effective_batch}"
            )


class ExtraSaveStepsCallback(TrainerCallback):
    """Request save-only checkpoints without changing evaluation cadence."""

    def __init__(self, steps):
        self.steps = frozenset(int(step) for step in steps)
        if any(step <= 0 for step in self.steps):
            raise ValueError("extra save steps must be positive optimizer-step numbers")

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.steps:
            control.should_save = True
        return control


class StopAfterStepCallback(TrainerCallback):
    """Save, evaluate, and stop at one scheduler-consistent screen boundary."""

    def __init__(self, step):
        self.step = int(step)
        if self.step <= 0:
            raise ValueError("stop-after step must be a positive optimizer-step number")

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step == self.step:
            control.should_evaluate = True
            control.should_save = True
            control.should_training_stop = True
        return control


class FinalSelectionCheckpointCallback(TrainerCallback):
    """Guarantee one aligned eval/save event at the natural training endpoint."""

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


class LearningCurveCallback(TrainerCallback):
    """Persist validation quality against elapsed training budget."""

    def __init__(self, train_windows, resume=False):
        self.train_windows = train_windows
        self.resume = resume
        self.started = None
        self.elapsed_offset = 0.0
        self.peak_allocated_prior = 0.0
        self.peak_reserved_prior = 0.0
        self.learning_rate_integral_by_group = None
        self.learning_rate_integral_since_resume_by_group = None
        self.learning_rate_integral_complete = True
        self.last_learning_rates = None
        self.learning_rate_group_names = None

    def on_train_begin(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / "learning_curve.jsonl"
        if path.is_file():
            if not self.resume:
                raise FileExistsError(f"existing learning curve requires checkpoint resume: {path}")
            points = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if points:
                self.elapsed_offset = float(points[-1]["elapsed_s"])
                self.peak_allocated_prior = float(points[-1].get("peak_memory_allocated_mib", 0.0))
                self.peak_reserved_prior = float(points[-1].get("peak_memory_reserved_mib", 0.0))
                prior_integrals = points[-1].get("learning_rate_integral_by_group")
                if prior_integrals is None:
                    self.learning_rate_integral_complete = False
                else:
                    self.learning_rate_integral_by_group = [float(value) for value in prior_integrals]
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_optimizer_step(self, args, state, control, optimizer=None, **kwargs):
        """Integrate the learning rates actually applied at each optimizer step."""
        del args, state, control, kwargs
        if optimizer is None or not optimizer.param_groups:
            raise RuntimeError("learning-rate integration requires an optimizer with parameter groups")
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        group_names = [
            str(group.get("group_name", f"group-{index}"))
            for index, group in enumerate(optimizer.param_groups)
        ]
        if self.learning_rate_group_names is None:
            self.learning_rate_group_names = group_names
        elif group_names != self.learning_rate_group_names:
            raise RuntimeError("optimizer parameter-group names changed during learning-rate integration")
        if not self.learning_rate_integral_by_group:
            self.learning_rate_integral_by_group = [0.0] * len(learning_rates)
        if not self.learning_rate_integral_since_resume_by_group:
            self.learning_rate_integral_since_resume_by_group = [0.0] * len(learning_rates)
        if len(learning_rates) != len(self.learning_rate_integral_by_group):
            raise RuntimeError("optimizer parameter-group count changed during learning-rate integration")
        self.learning_rate_integral_by_group = [
            integral + learning_rate
            for integral, learning_rate in zip(
                self.learning_rate_integral_by_group,
                learning_rates,
                strict=True,
            )
        ]
        self.learning_rate_integral_since_resume_by_group = [
            integral + learning_rate
            for integral, learning_rate in zip(
                self.learning_rate_integral_since_resume_by_group,
                learning_rates,
                strict=True,
            )
        ]
        self.last_learning_rates = learning_rates

    @staticmethod
    def common_group_value(values):
        if not values:
            return 0.0
        if all(math.isclose(value, values[0], rel_tol=1e-12, abs_tol=0.0) for value in values[1:]):
            return values[0]
        return None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.started is None:
            raise RuntimeError("learning-curve evaluation occurred before training began")
        point = {
            "schema_version": 2,
            "step": state.global_step,
            "epoch": state.epoch,
            "windows_seen": round(state.epoch * self.train_windows),
            "elapsed_s": self.elapsed_offset + time.perf_counter() - self.started,
            "learning_rate": self.common_group_value(self.last_learning_rates),
            "learning_rate_by_group": self.last_learning_rates or [],
            "learning_rate_group_names": self.learning_rate_group_names or [],
            "learning_rate_integral": (
                self.common_group_value(self.learning_rate_integral_by_group)
                if self.learning_rate_integral_complete
                else None
            ),
            "learning_rate_integral_by_group": (
                (self.learning_rate_integral_by_group or []) if self.learning_rate_integral_complete else None
            ),
            "learning_rate_integral_since_resume": self.common_group_value(
                self.learning_rate_integral_since_resume_by_group
            ),
            "learning_rate_integral_since_resume_by_group": (
                self.learning_rate_integral_since_resume_by_group or []
            ),
            "peak_memory_allocated_mib": max(
                self.peak_allocated_prior,
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
            ),
            "peak_memory_reserved_mib": max(
                self.peak_reserved_prior,
                torch.cuda.max_memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0,
            ),
        }
        for key, value in (metrics or {}).items():
            try:
                point[key] = float(value)
            except (TypeError, ValueError):
                continue
        path = Path(args.output_dir) / "learning_curve.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(point, sort_keys=True) + "\n")


def record_training_objective_config(config, args, partial_negative_sources):
    """Make saved objective provenance describe this stage, not its parent."""
    # Warm starts inherit config metadata as well as weights. Record every
    # objective unconditionally so a zero-valued continuation cannot retain
    # stale nonzero loss settings from its parent checkpoint.
    config.pii_initialized_only = False
    config.pii_annotated_boundary_loss_weight = args.annotated_boundary_loss
    config.pii_annotated_boundary_telemetry = args.annotated_boundary_telemetry
    config.pii_partial_boundary_retention_loss_weight = args.partial_boundary_retention_loss
    config.pii_partial_boundary_retention_temperature = args.partial_boundary_retention_temperature
    config.pii_partial_o_loss_weight = args.partial_o_loss
    config.pii_partial_entity_pu_loss_weight = args.partial_entity_pu_loss
    config.pii_partial_entity_pu_prior = (
        None if args.partial_entity_pu_prior is None else str(args.partial_entity_pu_prior)
    )
    config.pii_partial_entity_pu_prior_sha256 = (
        None
        if args.partial_entity_pu_prior is None
        else hashlib.sha256(args.partial_entity_pu_prior.read_bytes()).hexdigest()
    )
    config.pii_partial_entity_pu_positive_margin = args.partial_entity_pu_positive_margin
    config.pii_partial_expected_entity_ratio_loss_weight = args.partial_expected_entity_ratio_loss
    config.pii_partial_expected_entity_ratio_prior = (
        None
        if args.partial_expected_entity_ratio_prior is None
        else str(args.partial_expected_entity_ratio_prior)
    )
    config.pii_partial_expected_entity_ratio_prior_sha256 = (
        None
        if args.partial_expected_entity_ratio_prior is None
        else hashlib.sha256(args.partial_expected_entity_ratio_prior.read_bytes()).hexdigest()
    )
    config.pii_partial_expected_entity_ratio_lower_width = args.partial_expected_entity_ratio_lower_width
    config.pii_partial_parent_presence_kl_weight = args.partial_parent_presence_kl
    config.pii_partial_negative_loss_weight = args.partial_negative_loss
    config.pii_partial_negative_sources = {
        source: list(label_types) for source, label_types in partial_negative_sources.items()
    }
    config.pii_coarse_agreement_weight_20 = args.coarse_agreement_weight_20
    config.pii_coarse_agreement_weight_9 = args.coarse_agreement_weight_9
    config.pii_coarse_agreement_weight_presence_20 = args.coarse_agreement_weight_presence_20
    config.pii_coarse_agreement_reduction = args.coarse_agreement_reduction
    config.pii_fine_label_loss_weight = args.fine_label_loss_weight
    config.pii_o_token_loss_weight = args.o_token_loss_weight
    config.pii_entity_dice_loss_weight = args.entity_dice_loss
    config.pii_complete_presence_loss_weight = args.complete_presence_loss
    config.pii_complete_family_loss_weight = args.complete_family_loss
    config.pii_complete_boundary_loss_weight = args.complete_boundary_loss
    config.pii_reference_primary_positive_loss_weight = args.reference_primary_positive_loss_weight
    config.pii_partial_primary_objective_weight = args.partial_primary_objective_weight
    config.pii_native_new_label_space = bool(args.native_new_label_space)
    config.pii_logical_step_objective_normalization = args.logical_step_objective_normalization
    # Prediction reads this to encode flagged document starts the way training did.
    config.pii_document_start_marker = bool(args.document_start_marker)
    config.pii_accumulation_loss = (
        "logical-step-mass" if args.logical_step_objective_normalization else args.accumulation_loss
    )
    config.pii_max_grad_norm = args.max_grad_norm
    config.pii_batch_formation = args.batch_formation
    config.pii_batch_padding_budget = args.batch_padding_budget
    config.pii_draw_policy = args.draw_policy
    config.pii_predicate_loss_weight = args.predicate_loss_weight
    config.pii_reference_type_residual_loss_weight = args.reference_type_residual_loss_weight
    balance_path = getattr(args, "ont3_added_head_balance", None)
    config.pii_ont3_added_head_balance = None if balance_path is None else str(balance_path)
    config.pii_ont3_added_head_balance_sha256 = (
        None if balance_path is None else hashlib.sha256(balance_path.read_bytes()).hexdigest()
    )
    config.pii_subclass_loss_weight = args.subclass_loss_weight
    config.pii_rdrop_alpha = args.rdrop_alpha
    config.pii_mlm_replay_probability = args.mlm_replay_prob
    config.pii_mlm_replay_gold_policy = args.mlm_replay_gold_policy
    config.pii_mlm_physical_batch_probability = args.mlm_physical_batch_prob
    config.pii_mlm_pool_probabilities = dict(args.mlm_pool_probabilities)
    config.pii_mlm_loss_weight = args.mlm_loss_weight
    config.pii_mlm_probability = args.mlm_probability
    config.pii_mlm_head_model = args.mlm_head_model or args.model if args.use_mlm_objective else None
    config.pii_head_learning_rate = args.lr
    config.pii_encoder_learning_rate = args.encoder_lr
    config.pii_character_learning_rate = args.continuous_character_lr
    config.pii_character_output_learning_rate = args.continuous_character_output_lr
    config.pii_character_auxiliary_loss_weight = args.continuous_character_auxiliary_loss_weight
    config.pii_character_auxiliary_loss_fade_steps = args.continuous_character_auxiliary_loss_fade_steps
    config.pii_encoder_prior_checkpoint = args.encoder_prior_checkpoint or None
    config.pii_encoder_prior_weight = args.encoder_prior_weight
    config.pii_encoder_prior_start_step = args.encoder_prior_start_step
    config.pii_inherited_output_prior_checkpoint = (
        None
        if args.inherited_output_prior_checkpoint is None
        else str(args.inherited_output_prior_checkpoint)
    )
    config.pii_inherited_output_prior_weight = args.inherited_output_prior_weight
    config.pii_inherited_output_prior_start_step = args.inherited_output_prior_start_step
    config.pii_validation_selection_policy = args.val_selection
    config.pii_validation_selection_limit = args.max_val_windows
    config.pii_selection_metric = args.selection_metric
    config.pii_early_stopping_patience = args.patience
    config.pii_early_stopping_threshold = args.early_stopping_threshold
    config.pii_train_loss_window_epochs = args.train_loss_window_epochs
    config.pii_victory_lap_lr_scale = args.victory_lap_lr_scale
    config.pii_victory_lap = None


def rows_sha256(rows) -> str:
    """Hash an ordered sequence of window records canonically."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validation_selection_receipt(universe_rows, selected_rows, *, policy, seed, limit):
    """Describe the exact early-stopping draw used by this selection run."""

    def counts(rows, field, fallback=None, default="?"):
        return dict(
            sorted(
                Counter(
                    str(row.get(field) or (row.get(fallback) if fallback else None) or default)
                    for row in rows
                ).items()
            )
        )

    return {
        "schema_version": 2,
        "policy": policy,
        "seed": int(seed),
        "limit": int(limit),
        "universe_windows": len(universe_rows),
        "selected_windows": len(selected_rows),
        "universe_sha256": rows_sha256(universe_rows),
        "selected_sha256": rows_sha256(selected_rows),
        "universe_supervision": counts(
            universe_rows,
            "supervision",
            default=COMPLETE_SUPERVISION,
        ),
        "selected_supervision": counts(
            selected_rows,
            "supervision",
            default=COMPLETE_SUPERVISION,
        ),
        "universe_label_spaces": counts(
            universe_rows,
            "label_space",
            default=FINE_LABEL_SPACE,
        ),
        "selected_label_spaces": counts(
            selected_rows,
            "label_space",
            default=FINE_LABEL_SPACE,
        ),
        "selected_languages": counts(selected_rows, "lang"),
        "selected_sources": counts(selected_rows, "src", "source"),
    }


def persist_validation_selection(output_dir, receipt, *, resume):
    """Persist one selection receipt and reject a changed draw on resume.

    Returns the receipt path and whether this run established the draw rather
    than inheriting it. Resuming trainer state into a fresh output directory is
    the supported way to continue a leg onto a different validation set, and
    the caller needs to know it happened: a best-metric carried over from the
    old draw was measured on different rows and cannot be compared with
    anything this run will produce.
    """
    path = Path(output_dir) / "validation_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and resume:
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != receipt:
            raise ValueError(
                "validation selection changed while resuming this run; start a new output "
                "directory or restore the original data, seed, policy, and limit"
            )
        return path, False
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, True


def victory_lap_training_arguments(
    base_args,
    *,
    output_dir,
    learning_rate_scale,
    selected_step,
    validation_seed,
    validation_windows,
):
    """Build a fresh one-epoch base schedule for post-selection validation refit."""
    if learning_rate_scale <= 0:
        raise ValueError("victory-lap learning-rate scale must be positive")
    if selected_step < 0:
        raise ValueError("selected checkpoint step must be nonnegative")
    if validation_windows <= 0:
        raise ValueError("victory lap requires a nonempty validation draw")
    lap_args = replace(
        base_args,
        output_dir=str(output_dir),
        logging_dir=str(Path(output_dir) / "logs"),
        learning_rate=float(base_args.learning_rate) * learning_rate_scale,
        num_train_epochs=1.0,
        max_steps=-1,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        save_total_limit=None,
        data_seed=int(validation_seed),
    )
    lap_args.sampling_epoch_examples = int(validation_windows)
    lap_args.pii_sampling_seed = int(validation_seed)
    lap_args.pii_step_offset = int(selected_step)
    for field in (
        "pii_encoder_learning_rate",
        "pii_character_learning_rate",
        "pii_character_output_learning_rate",
    ):
        value = getattr(base_args, field, None)
        setattr(lap_args, field, None if value is None else float(value) * learning_rate_scale)
    return lap_args


def load_victory_lap_only_selection(receipt_path, *, init_checkpoint, validation_sha256):
    """Validate the frozen selection identity used by a lap-only arm."""
    path = Path(receipt_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "selected_checkpoint",
        "selected_step",
        "selected_eval_loss",
        "selected_metric_name",
        "selected_metric_value",
        "validation_selected_sha256",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise ValueError(f"lap-only selection receipt is missing fields: {missing}")
    if receipt.get("status") != "completed":
        raise ValueError(f"lap-only selection receipt is not completed: {receipt.get('status')!r}")
    if Path(receipt["selected_checkpoint"]).resolve() != Path(init_checkpoint).resolve():
        raise ValueError(
            "lap-only initialization differs from the frozen selected checkpoint: "
            f"{init_checkpoint!r} != {receipt['selected_checkpoint']!r}"
        )
    if receipt["validation_selected_sha256"] != validation_sha256:
        raise ValueError(
            "lap-only validation draw differs from the frozen selection draw: "
            f"{validation_sha256} != {receipt['validation_selected_sha256']}"
        )
    if not isinstance(receipt["selected_step"], int) or receipt["selected_step"] < 0:
        raise ValueError("lap-only selection receipt has an invalid selected_step")
    return receipt


def validate_victory_lap_only_configuration(receipt, base_args):
    """Require the lap-only arm to reuse the selected run's base schedule."""
    scheduler_name = getattr(base_args.lr_scheduler_type, "value", str(base_args.lr_scheduler_type))
    base_rates = {
        "head": float(base_args.learning_rate),
        "encoder": getattr(base_args, "pii_encoder_learning_rate", None),
        "character": getattr(base_args, "pii_character_learning_rate", None),
        "character_output": getattr(base_args, "pii_character_output_learning_rate", None),
    }
    if receipt.get("base_learning_rates") != base_rates:
        raise ValueError(
            "lap-only base learning rates differ from the frozen selected run: "
            f"{base_rates!r} != {receipt.get('base_learning_rates')!r}"
        )
    if receipt.get("base_scheduler") != scheduler_name:
        raise ValueError(
            "lap-only scheduler differs from the frozen selected run: "
            f"{scheduler_name!r} != {receipt.get('base_scheduler')!r}"
        )
    if not math.isclose(
        float(receipt.get("warmup_ratio", -1.0)),
        float(base_args.warmup_ratio),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "lap-only warmup ratio differs from the frozen selected run: "
            f"{base_args.warmup_ratio!r} != {receipt.get('warmup_ratio')!r}"
        )


def attach_calibrated_head_input_norm(model, train, tokenizer, args) -> None:
    """Calibrate head-feature moments on weighted training draws and insert the norm.

    Every kind starts as the identity in evaluation mode at these moments, so
    all arms of the BatchNorm contrast begin from the same served function.
    The draw is seeded from ``--seed`` alone, so paired arms calibrate on the
    same items.
    """
    if not isinstance(model, LayerConcatForTokenClassification):
        raise ValueError("--head-input-norm requires a concatenated-layer task head")
    weights = getattr(train, "sampling_weights", None) or [1.0] * len(train)
    rng = random.Random(f"head-input-norm-calibration:{args.seed}")
    indices = rng.choices(range(len(weights)), weights=weights, k=args.head_input_norm_calibration_items)
    items = [train[index] for index in indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    moments = head_input_calibration_moments(model, items, tokenizer, device=device)
    if args.head_input_norm != "none":
        spec = {
            "kind": args.head_input_norm,
            "momentum": args.head_input_norm_momentum,
        }
        model.attach_head_input_norm(spec, moments["mean"], moments["var"])
    if args.head_residual_mlp != "off":
        if args.head_input_norm != "none":
            raise ValueError("calibrate the residual block after head input normalization is not supported")
        model.attach_head_residual_mlp(
            {
                "hidden": args.head_residual_mlp_hidden,
                "norm": args.head_residual_mlp,
                "momentum": args.head_residual_mlp_momentum,
            },
            moments["mean"],
            moments["var"],
        )
    receipt = {
        "schema": "pii-head-input-norm-calibration-v1",
        "spec": {
            "head_input_norm": None if model.head_input_norm is None else model.head_input_norm.config(),
            "head_residual_mlp": None
            if model.head_residual_mlp is None
            else model.head_residual_mlp.config(),
        },
        "items": moments["items"],
        "tokens": moments["tokens"],
        "indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "mean_sha256": hashlib.sha256(moments["mean"].cpu().numpy().tobytes()).hexdigest(),
        "var_sha256": hashlib.sha256(moments["var"].cpu().numpy().tobytes()).hexdigest(),
        "median_channel_sd": float(moments["var"].sqrt().median()),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "head_input_norm_calibration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"TRAIN-HEAD-INPUT-NORM: kind={args.head_input_norm} momentum={args.head_input_norm_momentum} "
        f"residual-mlp={args.head_residual_mlp} hidden={args.head_residual_mlp_hidden} "
        f"items={moments['items']} tokens={moments['tokens']} "
        f"median-channel-sd={receipt['median_channel_sd']:.4f} mean-sha256={receipt['mean_sha256'][:12]}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="pii_assemble_corpus output dir")
    ap.add_argument(
        "--native-heads-config",
        type=Path,
        help="hash-pinned source-native gold heads, coverage and fixed auxiliary weights",
    )
    ap.add_argument(
        "--language-round",
        type=Path,
        default=DEFAULT_LANGUAGE_ROUND_PATH,
        help="core-language inventory and minimum expected sampler share",
    )
    ap.add_argument(
        "--language-support-split-manifest",
        type=Path,
        help=(
            "aggregate route manifest for an explicitly language-support-split model; "
            "requires --language-support-split-component"
        ),
    )
    ap.add_argument(
        "--language-support-split-component",
        help=(
            "route name trained by this component; its manifest must collectively cover the language round"
        ),
    )
    ap.add_argument(
        "--train-pool",
        action="append",
        default=[],
        metavar="NAME=PATH[:WEIGHT]",
        help=(
            "additional train JSONL pool; repeatable. With no --sampling-config, WEIGHT is its "
            "target mass and the --data train pool receives the remainder"
        ),
    )
    ap.add_argument("--out", required=True)
    add_checkpoint_mirror_args(ap)
    ap.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "evaluate the exact --init-from-checkpoint weights on the configured validation set, "
            "write evaluation_only.json under --out, and perform no optimizer step or model export"
        ),
    )
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-large")
    ap.add_argument(
        "--continuous-character-pretrain",
        type=Path,
        help=(
            "continuous-character pretraining artifact appended to a final-layer affine "
            "--init-from-checkpoint before joint token classification"
        ),
    )
    ap.add_argument(
        "--continuous-character-initialization",
        choices=("pretrained", "random", "tagger"),
        default="pretrained",
        help=(
            "initialize the character encoder from its masked-character endpoint, deterministic "
            "ancestor, or an isolated tagger"
        ),
    )
    ap.add_argument(
        "--continuous-character-tagger",
        type=Path,
        help="isolated tagger artifact supplying a compatible character encoder and/or readout",
    )
    ap.add_argument(
        "--continuous-character-logit-initialization",
        choices=("zero", "tagger"),
        default="zero",
        help="initialize the bias-free character-to-tag block exactly at zero or from the tagger",
    )
    ap.add_argument(
        "--continuous-character-logit-scale",
        type=float,
        default=1.0,
        help="held-out calibrated multiplier applied when importing the isolated tagger readout",
    )
    ap.add_argument(
        "--continuous-character-pooling",
        choices=("max", "mean", "max-mean"),
        default="max",
        help="pool whole-segment character states onto each tokenizer offset interval",
    )
    ap.add_argument(
        "--continuous-character-lr",
        type=float,
        help="optional independent learning rate for the continuous character encoder",
    )
    ap.add_argument(
        "--continuous-character-output-lr",
        type=float,
        help=(
            "optional learning rate for the bias-free character-to-tag block; it never receives weight decay"
        ),
    )
    ap.add_argument(
        "--continuous-character-auxiliary-loss-weight",
        type=float,
        default=0.0,
        help="initial training-only tag-loss weight on character logits alone",
    )
    ap.add_argument(
        "--continuous-character-auxiliary-loss-fade-steps",
        type=int,
        default=0,
        help="optimizer steps over which the character-only tag loss fades linearly to zero",
    )
    ap.add_argument(
        "--surface-realization-pool",
        type=Path,
        help="natural-surface pool used to re-realize selected training rows after sampling",
    )
    ap.add_argument(
        "--surface-realization-predicate-pool",
        type=Path,
        help="predicate-aware primary-type surface pool used by ont3 rows after sampling",
    )
    ap.add_argument(
        "--surface-realization-predicate-policy",
        type=Path,
        help="origin, support, compatibility, and backoff policy bound to the predicate pool",
    )
    ap.add_argument(
        "--surface-realization-recipe",
        type=Path,
        help="versioned language/tag surface-mixture recipe for sample-time realization",
    )
    ap.add_argument(
        "--surface-realization-locale-profile",
        type=Path,
        help="locale-rendering profile used by sample-time realization",
    )
    ap.add_argument(
        "--surface-realization-version",
        default="",
        help="required provenance version for sample-time-realized training draws",
    )
    ap.add_argument(
        "--surface-realization-context-generator",
        type=Path,
        help="optional fitted context-character generator used before recipe fallback",
    )
    ap.add_argument(
        "--surface-realization-context-generator-rate",
        type=float,
        default=1.0,
        help=(
            "deterministic per-entity probability of attempting the fitted context-character "
            "generator before recipe fallback (default: 1)"
        ),
    )
    ap.add_argument(
        "--surface-realization-context-generator-route",
        choices=("auto", "exact", "exact+p20+p9-additive", "intrinsic-winner"),
        default="auto",
        help=(
            "language-local bundle route; auto selects the intrinsic winner per language and "
            "is the only setting accepted by a single shared checkpoint"
        ),
    )
    ap.add_argument(
        "--surface-realization-equals",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="realize only rows with this exact top-level string value; repeatable with AND semantics",
    )
    ap.add_argument(
        "--surface-realization-unlocalized-categorical-policy",
        choices=("keep", "drop"),
        default="keep",
    )
    freeze_group = ap.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="freeze every pretrained encoder parameter and train only the task head",
    )
    freeze_group.add_argument(
        "--trainable-top-encoder-layers",
        type=int,
        help="freeze encoder embeddings and lower layers; train only this many top layers plus the task head",
    )
    ap.add_argument(
        "--freeze-inherited-output-rows",
        action="store_true",
        help=(
            "keep the successor checkpoint's declared parent classifier prefix bit-exact while "
            "training appended primary rows and independent heads; encoder updates may still change "
            "inherited predictions"
        ),
    )
    ap.add_argument(
        "--inherited-output-prior-checkpoint",
        type=Path,
        help=(
            "checkpoint anchoring the successor model's declared inherited classifier prefix; "
            "all prefix rows remain trainable"
        ),
    )
    ap.add_argument(
        "--inherited-output-prior-weight",
        type=float,
        default=0.0,
        help="positive multiplier on half the squared inherited-prefix distance from its anchor",
    )
    ap.add_argument(
        "--inherited-output-prior-start-step",
        type=int,
        default=0,
        help="optimizer step at which the inherited-output parameter prior begins",
    )
    ap.add_argument("--decoder", choices=("linear", "crf"), default="linear")
    ap.add_argument(
        "--head-kind",
        choices=("stock", "affine", "factorized_linear", "mlp"),
        default="stock",
        help="token head; non-stock heads concatenate --encoder-layers",
    )
    ap.add_argument(
        "--stock-classifier-kernel",
        choices=("framework", "explicit_mm"),
        default="framework",
        help=(
            "implementation of the stock affine projection; explicit_mm preserves its parameters "
            "and state dict while avoiding the framework's cuBLAS-Lt F.linear dispatch"
        ),
    )
    ap.add_argument(
        "--encoder-layers",
        default="-1",
        help=(
            "comma-separated hidden states: 0 is the embedding output, positive values are "
            "1-based encoder layers, and negatives count backward through encoder layers"
        ),
    )
    ap.add_argument(
        "--token-offsets",
        default="0",
        help=(
            "comma-separated relative token states concatenated before an affine head; "
            "-1,0,1 means preceding, current, and following token"
        ),
    )
    ap.add_argument("--head-rank", type=int, default=256, help="bottleneck rank for factorized/MLP heads")
    ap.add_argument("--classifier-dropout", type=float, default=0.1)
    ap.add_argument(
        "--annotated-boundary-loss",
        type=float,
        default=0.0,
        help=(
            "training-only O/B/I/E/S auxiliary-loss weight on annotated-spans-only rows; "
            "zero preserves the stock objective and inference graph"
        ),
    )
    ap.add_argument(
        "--coarse-agreement-weight-20",
        type=float,
        default=0.0,
        help=(
            "weight on the redaction_20_v1 cut-marginal CE term (bucket probability = sum of "
            "member fine probabilities); zero preserves the stock objective"
        ),
    )
    ap.add_argument(
        "--coarse-agreement-weight-9",
        type=float,
        default=0.0,
        help="weight on the redaction_9_v1 cut-marginal CE term; zero preserves the stock objective",
    )
    ap.add_argument(
        "--coarse-agreement-weight-presence-20",
        type=float,
        default=0.0,
        help=(
            "weight on the boundary-free redaction_20_v1 bucket-presence term: partial credit "
            "for any B/I/E/S member of the gold bucket, giving within-bucket boundary drift a "
            "reward the exact terms cannot"
        ),
    )
    ap.add_argument(
        "--fine-label-loss-weight",
        type=float,
        default=1.0,
        help=(
            "weight on native fine-label CE when a coarse-agreement term is active; zero trains "
            "initialized fine rows only as latent components of the coarse bucket, and one "
            "preserves the historical objective"
        ),
    )
    ap.add_argument(
        "--coarse-agreement-reduction",
        choices=("sum", "mean"),
        default="sum",
        help=(
            "pool member logits by probability sum (historical objective) or subtract log member "
            "count to give each bucket a uniform within-bucket component prior"
        ),
    )
    ap.add_argument(
        "--union-members",
        type=Path,
        help=(
            "member document published beside a union-labels training corpus "
            "(members.json): supervise its coarse-labelled rows through the union of each class's "
            "fine head rows instead of through any one row, leaving the head a native fine tagger"
        ),
    )
    ap.add_argument(
        "--union-v1-fine-weight-schedule",
        default=DEFAULT_UNION_FINE_WEIGHT_SCHEDULE,
        metavar="constant:W|linear:START:END",
        help=(
            "weight a fine-labelled row keeps on its own label, the remainder going to the same "
            "union-marginal term the coarse rows use; linear fades over the --max-steps horizon and "
            "the default constant:1.0 leaves fine supervision exactly as it was. This fades label "
            "precision only: sampling weights are untouched"
        ),
    )
    ap.add_argument(
        "--dual-head-map",
        type=Path,
        help=(
            "v1/v2 correctness map (scripts/pii_v1_v2_correctness_map.py): by default add a second "
            "head over the new ontology and train both from every row, each row supervising its own "
            "head directly and the other through the map's allowed set; with "
            "--dual-head-mapped-single-head, keep a retired new-ontology head and map old-space "
            "rows into that sole head"
        ),
    )
    ap.add_argument(
        "--dual-head-mapped-single-head",
        action="store_true",
        help=(
            "continue a checkpoint whose old head is already retired as one new-ontology head, "
            "while --dual-head-map still maps old-space rows into allowed-set supervision. This "
            "supports a new data stage with fresh optimizer and scheduler state; it never creates "
            "or reinitializes a classifier and requires --dual-head-old-weight-schedule constant:0"
        ),
    )
    ap.add_argument(
        "--dual-head-bind-native-map",
        action="store_true",
        help="start mapped single-head supervision from an exactly matching native head; bind a new map without changing classifier weights",
    )
    ap.add_argument(
        "--dual-head-old-weight-schedule",
        default="linear:1.0:0.0",
        metavar="constant:W|linear:START:END[:SPAN]",
        help=(
            "authority the incumbent head keeps in the training objective; the remainder goes to "
            "the new head, a linear schedule moves over the --max-steps horizon, and the default "
            "ends the run training the new head alone. SPAN compresses the handover into that "
            "leading fraction of the horizon (linear:1.0:0.0:0.25 finishes it in the first "
            "quarter and trains the new head alone for the rest)"
        ),
    )
    ap.add_argument(
        "--dual-head-keep-old-head",
        action="store_true",
        help=(
            "keep the incumbent head in the model after its authority reaches zero. By default it "
            "is deleted at that step, which costs nothing in the objective -- a zero weight was "
            "already contributing no gradient -- and leaves a single-head tagger over the new "
            "ontology that needs no promotion pass. Keep it to write two-head checkpoints "
            "throughout, for instance to compare the two vocabularies on one encoder afterwards"
        ),
    )
    ap.add_argument(
        "--dual-head-restart-lr-after-transition",
        action="store_true",
        help=(
            "give the phase after the handover its own warmup and decay instead of continuing one "
            "schedule across both. Worth setting whenever the handover finishes early: otherwise "
            "the new head trains alone on the decayed tail of a rate chosen for the whole run"
        ),
    )
    ap.add_argument(
        "--dual-head-init",
        choices=("projected", "fallback", "fresh"),
        default="fallback",
        help=(
            "initialize the new-ontology head by source-normalized accepted-set projection, by the "
            "legacy destination mean over fallback edges, or from fresh random parameters"
        ),
    )
    ap.add_argument(
        "--dual-head-init-counts",
        type=Path,
        help=(
            "training-only raw v1/v2 aligned-span counts for --dual-head-init projected; omission "
            "makes every accepted edge equiprobable before optional P20 smoothing"
        ),
    )
    ap.add_argument(
        "--dual-head-init-add-k",
        type=float,
        default=1.0,
        help="positive add-k mass per accepted edge for projected dual-head initialization",
    )
    ap.add_argument(
        "--dual-head-init-p20-alpha",
        type=float,
        default=0.0,
        help="nonnegative P20 pooled-prior pseudocount mass for projected initialization",
    )
    ap.add_argument(
        "--dual-head-init-p20-cut",
        default=DEFAULT_P20_CUT,
        help="v1 tagset cut that defines the pooled bridge for projected initialization",
    )
    ap.add_argument(
        "--dual-head-init-tagset",
        type=Path,
        default=Path(TAGSET_PATH),
        help="v1 tagset whose source projections and P20 cut define affine support",
    )
    ap.add_argument(
        "--dual-head-eval-weight",
        type=float,
        default=0.0,
        help=(
            "fixed incumbent-head authority used for validation loss, so checkpoint selection "
            "compares one objective across a run whose training blend is moving; the default is "
            "the endpoint objective"
        ),
    )
    ap.add_argument(
        "--o-weight-config",
        default=None,
        help=(
            "path to scripts/pii_o_weight.yaml. Sets the O-label trust PER SOURCE, so one physical "
            "batch can mix a gold row at full O weight with a lower-trust complete row and an "
            "intersection-consensus row at near 0. A class weight cannot express that, which is "
            "why this is separate from --o-token-loss-weight. Positive-only teacher rows instead "
            "mask O and never consult this table. Omitted leaves every complete row at full O trust. "
            "Sources absent from the config take its default, and the resolved weight is reported."
        ),
    )
    ap.add_argument(
        "--o-token-loss-weight",
        type=float,
        default=1.0,
        help=(
            "relative supervised CE weight for trusted O targets; entity targets retain weight 1, "
            "and 1 preserves the stock objective"
        ),
    )
    ap.add_argument(
        "--bioes-margin-boundary",
        type=float,
        default=0.0,
        help=(
            "BIOES structure-cost margin (softmax-margin) on the new head: extra logit margin the gold "
            "label must win by against every alternative whose BIOES position differs; 0 disables"
        ),
    )
    ap.add_argument(
        "--bioes-margin-type",
        type=float,
        default=0.0,
        help="companion margin against alternatives whose entity type differs (both costs add); 0 disables",
    )
    ap.add_argument(
        "--bioes-margin-illegal-scale",
        type=float,
        default=1.0,
        help=(
            "multiply a flip's margin cost by this when the flip is illegal next to the gold "
            "neighbours (the constrained decoder would repair it); 1 keeps every flip at full cost"
        ),
    )
    ap.add_argument(
        "--bioes-risk-weight",
        type=float,
        default=0.0,
        help=(
            "span-risk gate strength: tokens of a gold span whose coarse-incompatible margin is small "
            "get objective weight 1 + w * sigmoid((threshold - risk) / scale); 0 disables"
        ),
    )
    ap.add_argument(
        "--bioes-risk-threshold", type=float, default=1.0, help="risk-gate margin threshold in logits"
    )
    ap.add_argument("--bioes-risk-scale", type=float, default=0.5, help="risk-gate sigmoid scale in logits")
    ap.add_argument(
        "--bioes-risk-cut",
        default=DEFAULT_P20_CUT,
        help="reporting cut whose buckets define coarse-incompatible alternatives for the risk gate",
    )
    ap.add_argument(
        "--bioes-bucket-map",
        default="",
        help=(
            "JSON coarse cut (pii-ontology-v3-coarse-cut) giving every entity type a bucket; when set, "
            "the risk gate uses these buckets instead of --bioes-risk-cut and the margin's type cost is "
            "scaled by --bioes-margin-same-bucket-scale for flips inside the gold's bucket"
        ),
    )
    ap.add_argument(
        "--bioes-margin-same-bucket-scale",
        type=float,
        default=1.0,
        help="multiplier on the type cost for a same-bucket type flip (minor error); 1 keeps every flip severe",
    )
    ap.add_argument(
        "--entity-dice-loss",
        type=float,
        default=0.0,
        help=(
            "weight on class-agnostic entity-vs-O soft Dice loss, blended with token CE after "
            "normalizing by total objective weight; zero preserves the stock objective"
        ),
    )
    ap.add_argument(
        "--complete-presence-loss",
        type=float,
        default=0.0,
        help=(
            "training-only entity-vs-O cross-entropy weight on complete rows, projected from the "
            "native-v2 product head and normalized with its typed objective; requires --dual-head-map"
        ),
    )
    ap.add_argument(
        "--complete-family-loss",
        type=float,
        default=0.0,
        help=(
            "training-only O-vs-coarse-family cross-entropy weight on complete rows, projected "
            "from the native-v2 product head over ontology families and normalized with its typed "
            "objective; old-space tokens whose accepted set spans several families are masked; "
            "requires --dual-head-map"
        ),
    )
    ap.add_argument(
        "--complete-boundary-loss",
        type=float,
        default=0.0,
        help=(
            "training-only class-agnostic BIOES cross-entropy weight on annotated entity "
            "tokens in complete rows, projected from the native-v2 product head and normalized "
            "with its typed objective; excludes O tokens and requires --dual-head-map"
        ),
    )
    ap.add_argument(
        "--reference-primary-positive-loss-weight",
        type=float,
        default=0.0,
        help=(
            "relative weight on a separately normalized ordinary primary BIOES loss over "
            "explicitly supervised person_reference and organization_reference positive "
            "tokens; untagged partial-row tokens remain masked"
        ),
    )
    ap.add_argument(
        "--partial-primary-objective-weight",
        type=float,
        default=1.0,
        help=(
            "training-only multiplier on ordinary primary BIOES objective mass from explicitly "
            "tagged tokens in annotated-spans-only rows; one preserves the current objective, "
            "zero retains their other supervised channels but removes primary-label influence"
        ),
    )
    ap.add_argument(
        "--partial-type-only",
        action="append",
        default=[],
        metavar="TYPE",
        help=(
            "on annotated-spans-only rows, teach this primary type without teaching where its "
            "spans end: the target becomes the summed probability of the type's B, I, E and S "
            "tags instead of one exact tag (repeatable). Foreign-standard gold is reliable about "
            "what an organization is and carries its own convention about whether a leading "
            "article belongs inside the span, which at full dose overwrites ours"
        ),
    )
    ap.add_argument(
        "--partial-type-only-weight",
        type=float,
        default=1.0,
        help="loss weight on the boundary-free term introduced by --partial-type-only",
    )
    ap.add_argument(
        "--logical-step-objective-normalization",
        action="store_true",
        help=(
            "normalize each data objective by its total supervised mass over every physical "
            "batch in one gradient-accumulated optimizer step"
        ),
    )
    ap.add_argument(
        "--document-start-marker",
        action="store_true",
        help=(
            "encode a context flagged document_start (the target is its document's first sentence) as "
            "<s></s></s> target, distinct from the bare input that means no previous sentence is given; "
            "the flag is kept only by context configurations that select the previous neighbor (-1)"
        ),
    )
    ap.add_argument(
        "--exclude-unsupervised-items",
        action="store_true",
        help=(
            "encode every weighted training item once, give items with no supervised primary "
            "mass zero weight (their pool's other items absorb it), and write "
            "supervision-mass-audit.json.gz with each item's branch, weight, mass and batching length"
        ),
    )
    ap.add_argument(
        "--supervision-audit-only",
        action="store_true",
        help="stop after --exclude-unsupervised-items writes its audit, without training",
    )
    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help=(
            "gradient-norm clipping threshold per optimizer step. It applies to the step loss "
            "that --accumulation-loss defines: with sum, gradients are about --grad-accum times "
            "larger than with mean, so a threshold T under sum matches T/--grad-accum under mean"
        ),
    )
    ap.add_argument(
        "--accumulation-loss",
        choices=["mean", "sum"],
        default="mean",
        help=(
            "combine the physical-batch losses of one optimizer step by their mean (the usual "
            "Transformers behavior) or their sum. sum reproduces every run before this option "
            "existed: those summed --grad-accum batch means and, at --max-grad-norm 1.0, clipped "
            "essentially every step, which equals mean with --max-grad-norm 1/--grad-accum. "
            "--logical-step-objective-normalization defines its own step mean and requires mean"
        ),
    )
    ap.add_argument(
        "--predicate-loss-weight",
        type=float,
        default=0.0,
        help=(
            "relative weight on token-mean masked BCE for tag-specific predicates; only "
            "explicitly labeled token/channel cells inside an activating primary span contribute"
        ),
    )
    ap.add_argument(
        "--defer-reference-training",
        action="store_true",
        help=(
            "exclude person/organization references from primary logits and project their spans "
            "and refinements out of train/validation supervision; preserve complete/partial "
            "background semantics; requires zero reference-specific loss weights"
        ),
    )
    ap.add_argument(
        "--predicate-spec",
        type=Path,
        default=DEFAULT_PREDICATE_SPEC_PATH,
        help="channel order, activating primary types, and three-state target contract",
    )
    ap.add_argument(
        "--predicate-conditioning",
        choices=("none", "primary-type"),
        default="none",
        help=(
            "use one predicate-logit block per semantic gold primary type; only that block "
            "receives objective weight (BIOES positions share their type's block)"
        ),
    )
    ap.add_argument(
        "--predicate-objective-mask-channel",
        action="append",
        default=None,
        help=(
            "predicate channel whose annotated cells retain their targets but receive zero "
            "training objective weight; repeat to mask multiple channels"
        ),
    )
    ap.add_argument(
        "--ont3-added-head-balance",
        type=Path,
        help=(
            "frozen non-proof-derived positive learning multipliers for appended reference "
            "types and primary-type-conditioned predicate channels"
        ),
    )
    ap.add_argument(
        "--reference-type-residual-loss-weight",
        type=float,
        default=0.0,
        help=(
            "relative weight on separately masked binary semantic scores for the added reference "
            "types; each score is added to all four BIOES rows of its type"
        ),
    )
    ap.add_argument(
        "--subclass-loss-weight",
        type=float,
        default=0.0,
        help=(
            "relative weight on carrier-conditioned categorical subclass components; "
            "each component's own objective and learning weights multiply this global weight"
        ),
    )
    ap.add_argument(
        "--subclass-spec",
        type=Path,
        default=DEFAULT_SUBCLASS_SPEC_PATH,
        help="closed family, carrier, outcome, scope, sequence-grammar, and head-block contract",
    )
    ap.add_argument(
        "--head-input-norm",
        choices=("none", "fixed", "batch", "renorm"),
        default="none",
        help=(
            "insert per-channel normalization before the head dropout of a concatenated-layer "
            "affine head (new --init-from-checkpoint stage only): fixed calibration moments, "
            "live BatchNorm, or Batch Renormalization; every kind starts as the identity at "
            "calibration moments (gaps/sketches/batchnorm-continuation.md)"
        ),
    )
    ap.add_argument(
        "--head-input-norm-momentum",
        type=float,
        default=0.1,
        help="--head-input-norm running-moment update rate per physical training batch",
    )
    ap.add_argument(
        "--head-input-norm-calibration-items",
        type=int,
        default=2048,
        help="--head-input-norm and --head-residual-mlp weighted training draws used for calibration moments",
    )
    ap.add_argument(
        "--head-residual-mlp",
        choices=("off", "none", "layer", "batch"),
        default="off",
        help=(
            "insert a zero-initialized residual GELU block x + up(GELU(down(N(x)))) on the head "
            "features of a concatenated-layer head (new --init-from-checkpoint stage only); the "
            "value selects its pre-norm N: none, LayerNorm, or masked BatchNorm"
        ),
    )
    ap.add_argument(
        "--head-residual-mlp-hidden", type=int, default=1024, help="--head-residual-mlp hidden width"
    )
    ap.add_argument(
        "--head-residual-mlp-lr",
        type=float,
        default=None,
        help="--head-residual-mlp block learning rate (default: the head rate --lr)",
    )
    ap.add_argument(
        "--head-residual-mlp-momentum",
        type=float,
        default=0.1,
        help="--head-residual-mlp batch pre-norm running-moment update rate per physical batch",
    )
    ap.add_argument(
        "--rdrop-alpha",
        type=float,
        default=0.0,
        help=(
            "symmetric token-KL weight between two dropout-perturbed training passes; "
            "zero preserves single-pass stock training and evaluation always remains single-pass"
        ),
    )
    ap.add_argument(
        "--annotated-boundary-telemetry",
        action="store_true",
        help=(
            "record sparse component-loss, supervised-token, and logit-gradient scales; "
            "with zero boundary-loss weight this leaves the stock training objective unchanged"
        ),
    )
    ap.add_argument(
        "--partial-boundary-retention-loss",
        type=float,
        default=0.0,
        help=(
            "teacher-to-student BIOES KL weight on unannotated tokens of partial-label rows; "
            "requires --init-from-checkpoint and leaves the inference graph unchanged"
        ),
    )
    ap.add_argument(
        "--partial-boundary-retention-temperature",
        type=float,
        default=1.0,
        help="softmax temperature for --partial-boundary-retention-loss",
    )
    ap.add_argument(
        "--partial-o-loss",
        type=float,
        default=0.0,
        help=(
            "O cross-entropy weight on otherwise-unannotated tokens of partial-label rows; "
            "zero leaves teacher omissions unknown and does not affect trusted O tokens"
        ),
    )
    ap.add_argument(
        "--partial-entity-pu-loss",
        type=float,
        default=0.0,
        help=(
            "weight for source/language-stratified non-negative PU risk on binary entity "
            "presence; tagged partial tokens are positive and untagged partial tokens are "
            "unlabeled; requires --partial-entity-pu-prior and mapped native-v2 supervision"
        ),
    )
    ap.add_argument(
        "--partial-entity-pu-prior",
        type=Path,
        help="frozen trustworthy-complete token-prior receipt for --partial-entity-pu-loss",
    )
    ap.add_argument(
        "--partial-entity-pu-positive-margin",
        type=float,
        default=None,
        help=(
            "replace the PU positive logistic term with a hinge that stops pushing once this "
            "entity-vs-O log-odds margin is reached"
        ),
    )
    ap.add_argument(
        "--partial-expected-entity-ratio-loss",
        type=float,
        default=0.0,
        help=(
            "weight for a batch entity-rate interval hinge on all real tokens from "
            "annotated-spans-only rows; known spans keep ordinary BIOES supervision and "
            "untagged tokens remain latent; requires --partial-expected-entity-ratio-prior"
        ),
    )
    ap.add_argument(
        "--partial-expected-entity-ratio-prior",
        type=Path,
        help=(
            "frozen trustworthy-complete source/language token-prior receipt for "
            "--partial-expected-entity-ratio-loss"
        ),
    )
    ap.add_argument(
        "--partial-expected-entity-ratio-lower-width",
        type=float,
        default=0.1,
        help=(
            "subtract this width from each complete-data entity-token prior to form the "
            "lower interval edge; the unmodified prior is the upper edge"
        ),
    )
    ap.add_argument(
        "--partial-parent-presence-kl",
        type=float,
        default=0.0,
        help=(
            "weak parent-to-student Bernoulli entity-presence KL weight on untagged partial "
            "tokens; requires the mapped single-head PU arm and --init-from-checkpoint"
        ),
    )
    ap.add_argument(
        "--partial-negative-loss",
        type=float,
        default=0.0,
        help=(
            "weight for source-aware negative evidence outside spans on annotated-spans-only rows; "
            "requires one or more --partial-negative-source declarations"
        ),
    )
    ap.add_argument(
        "--partial-negative-source",
        action="append",
        default=[],
        metavar="SOURCE=TYPE,TYPE",
        help=(
            "source and exhaustively annotated types whose BIOES labels are absent outside its spans; "
            "repeat for multiple partial-label sources"
        ),
    )
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help="optional encoder rate; when set, --lr applies only to the task head",
    )
    ap.add_argument(
        "--encoder-prior-checkpoint",
        default="",
        help=(
            "fixed checkpoint whose encoder parameters define an L2-SP-style prior; "
            "the task head remains unregularized"
        ),
    )
    ap.add_argument(
        "--encoder-prior-weight",
        type=float,
        default=0.0,
        help="positive multiplier on half the squared encoder distance from --encoder-prior-checkpoint",
    )
    ap.add_argument(
        "--encoder-prior-start-step",
        type=int,
        default=0,
        help="completed optimizer steps before the encoder prior becomes active",
    )
    ap.add_argument("--sched", default="cosine")
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument(
        "--max-train-chars",
        type=int,
        default=0,
        help=(
            "training/replay character ceiling; 0 reuses --max-chars. Set this above the "
            "largest independently token-gated sentence to prevent a second fixed-character split "
            "without changing validation windowing"
        ),
    )
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument(
        "--training-windowing",
        choices=("character", "token-capacity"),
        default=None,
        help="pre-window training, replay and validation to the tokenizer budget; new stages default "
        "to token-capacity, exact resumes inherit the checkpoint (historical missing means character)",
    )
    ap.add_argument(
        "--soft-registers",
        type=int,
        default=0,
        help="learned register positions prepended to every input; the encoder attends to "
        "them and is never scored on them, and they need a trainable encoder to be useful",
    )
    ap.add_argument(
        "--soft-registers-input-only",
        action="store_true",
        help="omit the per-layer re-injection, leaving input-only soft prompt tokens",
    )
    ap.add_argument(
        "--soft-register-source-classes",
        nargs="?",
        const=str(DEFAULT_SOURCE_CLASS_SPEC),
        help="condition the first register slot on the row's annotation-convention class, "
        "read from this class spec (default spec when the flag is given without a value)",
    )
    ap.add_argument(
        "--soft-register-source-dropout",
        type=float,
        default=0.0,
        help="probability of replacing a row's source class with unknown during training, "
        "so the condition production feeds is trained rather than extrapolated",
    )
    ap.add_argument(
        "--soft-register-language",
        action="store_true",
        help="condition a register slot on the row's language; unlike source, the true value "
        "is available at inference, so this is a deployable signal rather than a nuisance absorber",
    )
    ap.add_argument(
        "--soft-register-language-dropout",
        type=float,
        default=0.0,
        help="probability of replacing a row's language with unspecified during training, so the "
        "model keeps working when the language is unknown or unreliable",
    )
    ap.add_argument(
        "--prune-early-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after selection, delete numbered checkpoints older than the selected one and "
        "its immediate predecessor; the predecessor is kept for checkpoint averaging and "
        "everything after the selection is kept as the post-selection trajectory",
    )
    ap.add_argument(
        "--thin-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after training, delete optimizer.pt from every numbered checkpoint except the "
        "terminal one; continuations load weights only, and exact resume needs only the last",
    )
    ap.add_argument(
        "--soft-register-domain",
        help="posterior sidecar from pii_domain_model.py assign; conditions a register slot "
        "on a learned domain, which unlike annotation source is computable at inference",
    )
    ap.add_argument(
        "--soft-register-domain-scope",
        choices=("segment", "document", "both"),
        default="segment",
        help="which granularity conditions the model: the sentence being tagged, the document "
        "it was drawn from, or one slot each",
    )
    ap.add_argument(
        "--soft-register-domain-dropout",
        type=float,
        default=0.0,
        help="probability of replacing a domain posterior with unspecified during training, so "
        "deployment can classify at either granularity, both, or neither",
    )
    ap.add_argument(
        "--language-bias",
        action="store_true",
        help="add a zero-initialized per-language output bias on the BIOES rows. Unlike an "
        "input token this cannot perturb the encoder's representation, so it tests language "
        "conditioning without the distraction an input token may cause",
    )
    ap.add_argument(
        "--language-bias-strength",
        type=float,
        default=1.0,
        help="scalar on the language bias; 0 reproduces the shared head exactly",
    )
    ap.add_argument(
        "--soft-register-lr",
        type=float,
        help="learning rate for the register parameters alone; they start at or near zero and "
        "otherwise inherit the task-head rate, which is tuned for an already-trained head",
    )
    ap.add_argument(
        "--tag-status-slots",
        choices=("neutral", "gold"),
        help="learned prompt slots spliced into the input: a visible 'maybe' slot and a "
        "training-only slot per primary type. gold fills the training-only slot with the "
        "row's present/absent status for that type; neutral leaves it a plain learned "
        "vector. Training-only slots are hidden at evaluation and inference, and for "
        "unknown status in training, without moving positions",
    )
    ap.add_argument(
        "--tag-status-layout",
        choices=("maybe-first", "gold-first", "adjacent-pairs", "grouped"),
        default="maybe-first",
        help="with --tag-status-slots: maybe-first = maybe | context | target | training-only; "
        "gold-first = training-only | context | maybe | target; adjacent-pairs = context | "
        "(maybe, training-only) pairs | target; grouped = all slots before the input",
    )
    ap.add_argument(
        "--tag-status-language",
        action="store_true",
        help="with --tag-status-slots: add a row-language slot to the maybe group and a "
        "training-only one to the other group, over the frozen register language list",
    )
    ap.add_argument(
        "--tag-status-lr",
        type=float,
        default=3e-4,
        help="with --tag-status-slots: learning rate for the prompt slots, which start from "
        "type-name embeddings and need a higher rate than the pretrained encoder",
    )
    ap.add_argument(
        "--context-field",
        help="row field holding {before: text, after: text} neighboring segments, or a legacy "
        "prefix string; context uses spare token capacity and only the target is supervised",
    )
    ap.add_argument(
        "--context-side",
        choices=("both", "previous"),
        help="neighbor sides available to the encoder after target splitting; new stages default "
        "to both, exact resumes inherit the saved policy",
    )
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument(
        "--context-configurations",
        type=json.loads,
        help="training-only uniform per-draw context mixture, e.g. [[-1,0,1],[-1,0],[0]]; "
        "requires --context-field; with --context-side previous no configuration may hold 1; "
        "validation keeps every allowed side; "
        "only immediate neighbors are supported; exact resumes inherit the saved mixture",
    )
    ap.add_argument(
        "--context-configuration-weights",
        type=json.loads,
        help="relative sampling weights, one per --context-configurations entry "
        "(e.g. [4,1] with [[0],[-1,0]] trains 80%% isolated); default equal",
    )
    ap.add_argument(
        "--extra-save-step",
        action="append",
        type=int,
        default=[],
        help=(
            "save a checkpoint at this optimizer step without evaluating or changing model-selection "
            "cadence; repeat for multiple steps"
        ),
    )
    ap.add_argument(
        "--stop-after-step",
        type=int,
        default=0,
        help=(
            "save, evaluate, and exit after this optimizer step while retaining --max-steps as "
            "the learning-rate schedule horizon; resume from the saved checkpoint without this flag"
        ),
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "validation checks without a qualifying loss improvement before stopping; "
            f"default {DEFAULT_EARLY_STOPPING_PATIENCE} tolerates roughly half a 256k-draw "
            "sampling epoch when evaluation runs every 250 steps"
        ),
    )
    ap.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.001,
        help=(
            "minimum selected-metric improvement that resets early-stopping patience "
            "(loss decrease or span-F1 increase)"
        ),
    )
    ap.add_argument(
        "--save-limit",
        type=int,
        default=5,
        help="checkpoint retention limit; 1 deletes non-best checkpoints at training completion",
    )
    ap.add_argument(
        "--keep-final-checkpoint",
        action="store_true",
        help="require the final resumable checkpoint in addition to validation best; requires --save-limit >= 2",
    )
    ap.add_argument("--max-train-windows", type=int, default=0)
    ap.add_argument("--max-val-windows", type=int, default=0)
    ap.add_argument(
        "--val-selection",
        choices=("shuffle", "head"),
        default="shuffle",
        help=(
            "choose a deterministic shuffled validation draw before --max-val-windows; "
            "head preserves the legacy order-sensitive behavior explicitly"
        ),
    )
    ap.add_argument(
        "--selection-metric",
        choices=("loss", "span-f1"),
        default="loss",
        help=(
            "checkpoint and early-stopping criterion: loss preserves the historical full-draw "
            "objective; span-f1 requires hard validation labels and maximizes exact typed span F1"
        ),
    )
    ap.add_argument(
        "--train-loss-window-epochs",
        type=float,
        default=1.0,
        help="sampling-epoch fraction used for rolling training-loss telemetry",
    )
    ap.add_argument(
        "--victory-lap-lr-scale",
        type=float,
        default=DEFAULT_VICTORY_LAP_LR_SCALE,
        help=(
            "positive fraction of each base learning rate for one post-selection pass over "
            "the early-stopping draw; zero disables the lap"
        ),
    )
    ap.add_argument(
        "--victory-lap-only-receipt",
        type=Path,
        help=(
            "skip trajectory fitting and run only the configured victory lap from "
            "--init-from-checkpoint; require a completed prior lap receipt that binds the "
            "identical selected checkpoint and validation draw"
        ),
    )
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument(
        "--memory-metrics",
        action="store_true",
        help="record Trainer CPU/GPU allocation and peak-memory deltas",
    )
    ap.add_argument(
        "--sampling-config",
        default="",
        help=(
            "JSON pool config compiled to per-row sampling weights; without it, inline "
            "sampling_weight values are used when present on every train row"
        ),
    )
    ap.add_argument(
        "--sampling-epoch-windows",
        type=int,
        default=0,
        help="weighted draws per sampler epoch; zero uses distinct supervised inputs plus MLM rows when sharing annotations, otherwise the train row count",
    )
    ap.add_argument(
        "--annotation-variant-weighting",
        choices=("share", "legacy"),
        default=None,
        help="share one encoder-input sampling weight equally across repeated annotations (new-stage default); exact resume inherits its checkpoint policy; legacy reproduces historical recipes",
    )
    ap.add_argument(
        "--annotation-conventions",
        type=Path,
        help="annotator/language/tag internal-segmentation policy JSON, standalone or embedded in a correctness map; exact resume inherits the saved policy",
    )
    ap.add_argument(
        "--sampling-length-window-steps",
        type=int,
        default=DEFAULT_WEIGHTED_LENGTH_WINDOW_STEPS,
        help=(
            "optimizer steps pooled before weighted draws are length-sorted into physical batches; "
            "used only when weighted sampling is enabled"
        ),
    )
    ap.add_argument(
        "--batch-formation",
        choices=list(BATCH_FORMATIONS),
        default="band-random",
        help=(
            "how weighted draws become physical batches. band-random: each batch is a random draw plus "
            "occupants drawn at random from the remaining draws within --batch-padding-budget of its "
            "length, with sparse tails and final stragglers batched in length order. sorted-window: "
            "occupants are length-sorted neighbours (every run before this option). Both deal batches "
            "round-robin over optimizer steps in length order and leave each draw's weight unchanged"
        ),
    )
    ap.add_argument(
        "--batch-padding-budget",
        type=float,
        default=0.10,
        help=(
            "band-random only: the largest share of a physical batch's padded token slots that may be "
            "padding. Co-occupants lie within length*(1 +- budget/2) of the batch's seed; a smaller "
            "budget means less padding and less random co-occupancy"
        ),
    )
    ap.add_argument(
        "--draw-policy",
        choices=list(DRAW_POLICIES),
        default="carried",
        help=(
            "how each epoch's weighted draws are made. carried: one systematic wheel continued across "
            "epochs, so every row is drawn the floor or ceiling of its cumulative expected count "
            "(weighted draws without replacement over the run). systematic: exact per-epoch counts "
            "re-randomized each epoch (every run before this option). independent: with replacement"
        ),
    )
    ap.add_argument(
        "--sampling-length-bucket-width",
        type=int,
        default=DEFAULT_RECURSIVE_LENGTH_BUCKET_WIDTH,
        help=(
            "token-width of dynamic nearest-filled length buckets used only to reduce padding in "
            "physical-batch MLM sampling; zero disables this optimization, while narrower sparse "
            "bins can overexpose adjacent fill bins"
        ),
    )
    ap.add_argument(
        "--training-parameter-precision",
        choices=("float32", "legacy"),
        default=None,
        help=(
            "float32 retains FP32 parameters and Adam moments under BF16 autocast; "
            "legacy preserves architecture-specific BF16/mixed parameter loading. "
            "New training defaults to float32; exact resume inherits the saved choice "
            "(legacy for old checkpoints); evaluation-only keeps legacy loading by default"
        ),
    )
    ap.add_argument(
        "--init-from-checkpoint",
        default="",
        help=(
            "selected task checkpoint for a new training stage; retains encoder/head weights and "
            "copies shared label rows when the new corpus expands the label vocabulary"
        ),
    )
    ap.add_argument(
        "--warm-start-label-schema",
        default="",
        help=(
            "source schema in scripts/pii_tagset.yaml for semantic BIOES-row transfer when "
            "--init-from-checkpoint uses a different label vocabulary"
        ),
    )
    ap.add_argument(
        "--warm-start-label-cut",
        choices=Tagset().cut_names(),
        default="",
        help=(
            "reporting cut used to pre-collapse a canonical warm-start head; each target BIOES row "
            "is initialized from the mean of all source rows assigned to it"
        ),
    )
    ap.add_argument(
        "--reset-output-head",
        action="store_true",
        help=(
            "load the encoder from --init-from-checkpoint but deterministically reinitialize every "
            "task-output row; intended for standardized frozen-encoder probe comparisons"
        ),
    )
    ap.add_argument(
        "--mlm-replay-data",
        default="",
        help="natural-text JSONL packet used for a joint masked-language-model replay objective",
    )
    ap.add_argument(
        "--mlm-replay-prob",
        type=float,
        default=0.0,
        help="target fraction of sampled training rows drawn from --mlm-replay-data",
    )
    ap.add_argument(
        "--mlm-replay-gold-policy",
        choices=("preserve", "scale"),
        default="preserve",
        help=(
            "preserve keeps gold's absolute joint share by rebalancing within languages; "
            "scale uniformly shrinks every supervised pool when MLM replay is inserted"
        ),
    )
    ap.add_argument(
        "--mlm-physical-batch-prob",
        type=float,
        default=0.0,
        help=(
            "default probability that a supervised physical batch is replaced by an MLM view; "
            "frequency controls compute independently of --mlm-loss-weight"
        ),
    )
    ap.add_argument(
        "--mlm-pool",
        action="append",
        default=[],
        metavar="POOL=PROBABILITY",
        help=(
            "override --mlm-physical-batch-prob for one named sampling pool; repeatable, with "
            "POOL from --train-pool or --sampling-config"
        ),
    )
    ap.add_argument(
        "--mlm-loss-weight",
        type=float,
        default=0.0,
        help=(
            "required positive multiplier on an active physical batch's masked-language-model loss; "
            "inactive slots remain in the logical-update average"
        ),
    )
    ap.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="token masking probability within masked-language-model replay rows",
    )
    ap.add_argument(
        "--mlm-head-model",
        default="",
        help="pretrained model supplying the fixed MLM head; defaults to --model",
    )
    ap.add_argument("--replay-data", default="", help="prior corpus mixed into this stage's train split")
    ap.add_argument(
        "--replay-prob",
        type=float,
        default=0.1,
        help="target share of --replay-data windows in the mixed training corpus",
    )
    ap.add_argument(
        "--min-language-share",
        action="append",
        default=[],
        metavar="LANG=FRACTION",
        help=(
            "minimum language share in the final post-windowing train mix; repeatable and "
            "enforced by stratifying --replay-data without changing --replay-prob"
        ),
    )
    ap.add_argument(
        "--min-replay-language-share",
        action="append",
        default=[],
        metavar="LANG=FRACTION",
        help=(
            "minimum language share within the selected --replay-data tranche; repeatable and "
            "enforced without changing --replay-prob"
        ),
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="'auto' uses the newest checkpoint under --out; otherwise provide an explicit checkpoint path",
    )
    ap.add_argument(
        "--extend-resume-horizon",
        action="store_true",
        help=(
            "continue an exact mapped-single-head resume beyond its saved --max-steps while "
            "preserving optimizer/RNG state; the saved horizon becomes a learning-rate phase "
            "boundary and the configured schedule restarts over the added steps"
        ),
    )
    ap.add_argument(
        "--save-initialized-only",
        action="store_true",
        help=(
            "save the deterministic post-warm-start, pre-training model to --out and exit; "
            "requires --init-from-checkpoint"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "legacy single seed for model and data sampling; explicit use preserves shared-seed coupling. "
            "Omit to fork the configured reproducibility root"
        ),
    )
    ap.add_argument("--reproducibility-config", type=Path, default=REPRODUCIBILITY_CONFIG_PATH)
    ap.add_argument("--reproducibility-seed", type=int, default=None)
    ap.add_argument(
        "--language-weights",
        type=Path,
        help="JSON object of per-language sampling weights multiplied into each row's "
        "weight; an unlisted language weighs 1. Compute a set meeting a minimum share "
        "with scripts/pii_language_weights.py",
    )
    ap.add_argument(
        "--minimum-language-share",
        type=float,
        default=0.0,
        help="refuse to train unless every language reaches this share of the draw; "
        "checks the supplied weights rather than adjusting them",
    )
    ap.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help="seed for choosing which validation rows are scored, independent of --seed; "
        "pin it across an arm family so a seed change moves the model and not the test",
    )
    ap.add_argument(
        "--native-new-label-space",
        action="store_true",
        help="labels.json lists the new-ontology primary types, so a row in that label "
        "space is the head's own target. Excludes --dual-head-map and --union-members: "
        "no old-ontology supervision, no correctness map, no second head",
    )
    args = ap.parse_args()
    if args.keep_final_checkpoint and args.save_limit < 2:
        ap.error(
            "--keep-final-checkpoint requires --save-limit >= 2 to retain both final and validation best"
        )
    if args.native_new_label_space and (args.dual_head_map is not None or args.union_members is not None):
        ap.error(
            "--native-new-label-space trains the new ontology directly; drop --dual-head-map and --union-members"
        )

    native_head_specs, native_head_rows = [], {}
    if args.native_heads_config is not None:
        if args.continuous_character_pretrain or args.decoder == "crf":
            ap.error(
                "--native-heads-config requires layered token classification without MLM or character objectives"
            )
        try:
            native_head_specs, native_head_rows = load_native_head_config(args.native_heads_config)
            validate_training_inputs([Path(spec["train"]) for spec in native_head_specs])
        except (ValueError, KeyError) as error:
            ap.error(str(error))

    if args.evaluate_only:
        if not args.init_from_checkpoint:
            ap.error("--evaluate-only requires --init-from-checkpoint")
        if args.resume_from_checkpoint:
            ap.error("--evaluate-only cannot resume trainer state")
        if args.save_initialized_only:
            ap.error("--evaluate-only cannot be combined with --save-initialized-only")
        if args.victory_lap_only_receipt is not None or args.victory_lap_lr_scale:
            ap.error("--evaluate-only requires --victory-lap-lr-scale=0 and no lap-only receipt")
        if args.stop_after_step or args.extra_save_step:
            ap.error("--evaluate-only cannot save or stop at optimizer steps")
    if args.freeze_inherited_output_rows and args.save_initialized_only:
        ap.error("--freeze-inherited-output-rows applies to optimization, not --save-initialized-only")
    if args.inherited_output_prior_checkpoint:
        if not args.inherited_output_prior_checkpoint.is_dir():
            ap.error(
                "--inherited-output-prior-checkpoint is not a directory: "
                f"{args.inherited_output_prior_checkpoint}"
            )
        if args.inherited_output_prior_weight <= 0:
            ap.error("--inherited-output-prior-weight must be positive")
        if args.inherited_output_prior_start_step < 0:
            ap.error("--inherited-output-prior-start-step must be nonnegative")
        if args.freeze_inherited_output_rows:
            ap.error(
                "--inherited-output-prior-checkpoint cannot be combined with --freeze-inherited-output-rows"
            )
        if args.save_initialized_only:
            ap.error("--inherited-output-prior-checkpoint applies to optimization")
    elif args.inherited_output_prior_weight or args.inherited_output_prior_start_step:
        ap.error(
            "--inherited-output-prior-weight/--inherited-output-prior-start-step require "
            "--inherited-output-prior-checkpoint"
        )
    if bool(args.language_support_split_manifest) != bool(args.language_support_split_component):
        ap.error(
            "--language-support-split-manifest and --language-support-split-component "
            "must be supplied together"
        )
    if args.save_initialized_only and args.language_support_split_manifest:
        ap.error("language-support split declarations apply to training, not --save-initialized-only")

    surface_realization_requested = any(
        (
            args.surface_realization_pool,
            args.surface_realization_predicate_pool,
            args.surface_realization_predicate_policy,
            args.surface_realization_recipe,
            args.surface_realization_locale_profile,
            args.surface_realization_version,
            args.surface_realization_equals,
            args.surface_realization_context_generator,
        )
    )
    legacy_surface_configuration = (
        args.surface_realization_pool,
        args.surface_realization_recipe,
        args.surface_realization_locale_profile,
    )
    if any(legacy_surface_configuration) and not all(legacy_surface_configuration):
        ap.error(
            "legacy sample-time realization requires --surface-realization-pool, "
            "--surface-realization-recipe, and --surface-realization-locale-profile together"
        )
    if bool(args.surface_realization_predicate_pool) != bool(args.surface_realization_predicate_policy):
        ap.error(
            "predicate sample-time realization requires --surface-realization-predicate-pool "
            "and --surface-realization-predicate-policy together"
        )
    if surface_realization_requested and not (
        args.surface_realization_version and args.surface_realization_equals
    ):
        ap.error(
            "sample-time realization requires --surface-realization-version and --surface-realization-equals"
        )
    if surface_realization_requested and not (
        args.surface_realization_pool or args.surface_realization_predicate_pool
    ):
        ap.error("sample-time realization requires a legacy or predicate surface pool")
    if not 0.0 <= args.surface_realization_context_generator_rate <= 1.0:
        ap.error("--surface-realization-context-generator-rate must be in [0, 1]")
    if (
        args.surface_realization_context_generator is None
        and args.surface_realization_context_generator_rate != 1.0
    ):
        ap.error(
            "--surface-realization-context-generator-rate requires --surface-realization-context-generator"
        )
    if (
        args.surface_realization_context_generator is None
        and args.surface_realization_context_generator_route != "auto"
    ):
        ap.error(
            "--surface-realization-context-generator-route requires --surface-realization-context-generator"
        )
    if args.surface_realization_context_generator is not None and not args.surface_realization_pool:
        ap.error("surface context generation requires the legacy surface pool configuration")
    try:
        surface_realization_equalities = (
            parse_string_equalities(args.surface_realization_equals)
            if surface_realization_requested
            else None
        )
    except ValueError as error:
        ap.error(str(error))

    language_weights: dict[str, float] = {}
    if args.language_weights:
        try:
            raw = json.loads(Path(args.language_weights).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            ap.error(f"--language-weights: {error}")
        if not isinstance(raw, dict) or not raw:
            ap.error("--language-weights must be a non-empty JSON object of language to weight")
        for language, weight in raw.items():
            value = float(weight)
            if not math.isfinite(value) or value < 0:
                ap.error(f"--language-weights[{language}] must be finite and nonnegative")
            language_weights[str(language)] = value
    try:
        training_seeds = resolve_training_seeds(
            args.seed,
            args.reproducibility_config,
            args.reproducibility_seed,
            args.validation_seed,
        )
    except ValueError as error:
        ap.error(str(error))
    model_seed = training_seeds["model"]
    sampling_seed = training_seeds["sampling"]
    replay_seed = training_seeds["replay"]
    validation_seed = training_seeds["validation"]
    # Capping the validation view is what makes the seed decide membership rather than
    # only order. Unpinned, that turns a seed sweep into a sweep over different tests, so
    # refuse the combination instead of reporting curves that cannot be compared.
    if args.max_val_windows and not training_seeds["validation_pinned"] and args.val_selection != "head":
        ap.error(
            "--max-val-windows selects a validation subset by seed; pass --validation-seed "
            "to fix which rows are scored, or --val-selection head to take the first rows"
        )

    try:
        train_pool_specs = parse_train_pool_specs(
            args.train_pool,
            sampling_config=bool(args.sampling_config),
        )
        args.mlm_pool_probabilities = parse_mlm_pool_probabilities(args.mlm_pool)
        train_max_chars, validation_max_chars = resolve_window_limits(
            args.max_chars,
            args.max_train_chars,
        )
    except ValueError as error:
        ap.error(str(error))
    if not args.save_initialized_only:
        training_input_paths = [Path(args.data) / "train.jsonl"]
        training_input_paths.extend(Path(path) for _name, path, _weight in train_pool_specs)
        if args.replay_data:
            training_input_paths.append(Path(args.replay_data) / "train.jsonl")
        try:
            validate_training_inputs(training_input_paths)
        except (OSError, ValueError) as error:
            ap.error(str(error))
    if not 0.0 <= args.mlm_physical_batch_prob <= 1.0:
        ap.error("--mlm-physical-batch-prob must be in [0, 1]")
    if args.sampling_length_bucket_width < 0:
        ap.error("--sampling-length-bucket-width must be nonnegative")
    if args.max_val_windows < 0:
        ap.error("--max-val-windows must be nonnegative")
    if args.patience <= 0:
        ap.error("--patience must be positive")
    if args.early_stopping_threshold < 0:
        ap.error("--early-stopping-threshold must be nonnegative")
    if args.train_loss_window_epochs <= 0:
        ap.error("--train-loss-window-epochs must be positive")
    if args.victory_lap_lr_scale < 0:
        ap.error("--victory-lap-lr-scale must be nonnegative")
    if args.victory_lap_only_receipt is not None:
        if not args.init_from_checkpoint:
            ap.error("--victory-lap-only-receipt requires --init-from-checkpoint")
        if args.victory_lap_lr_scale <= 0:
            ap.error("--victory-lap-only-receipt requires a positive --victory-lap-lr-scale")
        if args.resume_from_checkpoint:
            ap.error("--victory-lap-only-receipt cannot resume trainer state")
        if args.save_initialized_only or args.stop_after_step:
            ap.error("--victory-lap-only-receipt cannot be combined with initialized-only or staged stopping")
    if args.stop_after_step < 0:
        ap.error("--stop-after-step must be nonnegative")
    if args.stop_after_step and args.max_steps <= 0:
        ap.error("--stop-after-step requires a positive --max-steps schedule horizon")
    if args.stop_after_step > args.max_steps > 0:
        ap.error("--stop-after-step cannot exceed --max-steps")
    args.use_mlm_objective = bool(
        args.mlm_replay_data or args.mlm_physical_batch_prob or any(args.mlm_pool_probabilities.values())
    )
    args.physical_batch_mlm = bool(args.mlm_replay_data or args.mlm_physical_batch_prob or args.mlm_pool)
    if train_pool_specs and args.replay_data:
        ap.error("--train-pool generalizes replay pools and cannot be combined with --replay-data")

    # TrainingArguments seeds Trainer later, after the task head has already
    # been initialized. Seed here so architecture contrasts share head init.
    set_seed(model_seed)

    nodes = json.load(open(os.path.join(args.data, "labels.json")))["labels"]
    names = ["O"] + [f"{p}-{n}" for n in nodes for p in "BIES"]
    label2id = {n: i for i, n in enumerate(names)}
    predicate_spec = None
    predicate_condition_types = ()
    predicate_objective_mask_channels = tuple(sorted(set(args.predicate_objective_mask_channel or ())))
    if args.predicate_loss_weight:
        try:
            predicate_spec = load_predicate_spec(args.predicate_spec)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
        active_types = set().union(*predicate_spec.applicable_types.values()) & set(nodes)
        if not active_types:
            ap.error("--predicate-spec has no activating type in this corpus label inventory")
        if args.predicate_conditioning == "primary-type":
            predicate_condition_types = tuple(sorted(set().union(*predicate_spec.applicable_types.values())))
        unknown_mask_channels = set(predicate_objective_mask_channels) - set(predicate_spec.channels)
        if unknown_mask_channels:
            ap.error(
                f"--predicate-objective-mask-channel names unknown channels: {sorted(unknown_mask_channels)}"
            )
    elif args.predicate_conditioning != "none":
        ap.error("--predicate-conditioning requires a positive --predicate-loss-weight")
    elif predicate_objective_mask_channels:
        ap.error("--predicate-objective-mask-channel requires a positive --predicate-loss-weight")
    added_head_balance_spec = None
    if args.ont3_added_head_balance is not None:
        if predicate_spec is None or args.predicate_conditioning != "primary-type":
            ap.error(
                "--ont3-added-head-balance requires positive predicate loss with "
                "--predicate-conditioning primary-type"
            )
        if args.dual_head_map is None:
            ap.error("--ont3-added-head-balance requires --dual-head-map")
        try:
            added_head_balance_spec = load_added_head_balance_spec(
                args.ont3_added_head_balance,
                predicate_spec,
                predicate_condition_types,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
    reference_type_residual_types = ()
    if args.reference_type_residual_loss_weight:
        if added_head_balance_spec is None:
            ap.error("--reference-type-residual-loss-weight requires --ont3-added-head-balance")
        if not args.dual_head_mapped_single_head:
            ap.error(
                "--reference-type-residual-loss-weight currently requires --dual-head-mapped-single-head"
            )
        if not args.predicate_loss_weight or not args.subclass_loss_weight:
            ap.error(
                "--reference-type-residual-loss-weight currently requires the joint predicate "
                "and subclass objectives"
            )
        reference_type_residual_types = tuple(sorted(added_head_balance_spec.reference_positive_weights))
    subclass_spec = None
    if args.subclass_loss_weight:
        try:
            subclass_spec = load_subclass_spec(args.subclass_spec)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
        active_types = set().union(*(family.applicable_types for family in subclass_spec.families)) & set(
            nodes
        )
        if not active_types:
            ap.error("--subclass-spec has no activating type in this corpus label inventory")
    encoder_layers = [int(value) for value in args.encoder_layers.split(",") if value.strip()]
    token_offsets = [int(value) for value in args.token_offsets.split(",") if value.strip()]
    try:
        resolved_token_offsets = resolve_token_offsets(token_offsets)
    except ValueError as error:
        ap.error(str(error))
    head_rank = args.head_rank if args.head_kind in ("factorized_linear", "mlp") else 0
    if args.stock_classifier_kernel != "framework" and (
        args.decoder != "linear" or args.head_kind != "stock"
    ):
        ap.error("--stock-classifier-kernel explicit_mm requires --decoder linear --head-kind stock")

    if args.init_from_checkpoint and args.resume_from_checkpoint:
        ap.error("--init-from-checkpoint starts a new stage and cannot be combined with trainer-state resume")
    if args.head_input_norm != "none":
        if not args.init_from_checkpoint:
            ap.error("--head-input-norm requires a new --init-from-checkpoint stage")
        if args.rdrop_alpha:
            ap.error("--head-input-norm would update running moments on both R-Drop passes")
        if not 0.0 < args.head_input_norm_momentum <= 1.0:
            ap.error("--head-input-norm-momentum must be in (0, 1]")
        if args.head_input_norm_calibration_items < 2:
            ap.error("--head-input-norm-calibration-items must be at least 2")
    if args.head_residual_mlp != "off":
        if not args.init_from_checkpoint:
            ap.error("--head-residual-mlp requires a new --init-from-checkpoint stage")
        if args.head_residual_mlp == "batch" and args.rdrop_alpha:
            ap.error("--head-residual-mlp batch would update running moments on both R-Drop passes")
        if args.head_residual_mlp_hidden <= 0:
            ap.error("--head-residual-mlp-hidden must be positive")
        if not 0.0 < args.head_residual_mlp_momentum <= 1.0:
            ap.error("--head-residual-mlp-momentum must be in (0, 1]")
    if args.head_residual_mlp_lr is not None:
        if args.head_residual_mlp == "off":
            ap.error("--head-residual-mlp-lr requires --head-residual-mlp")
        if args.head_residual_mlp_lr <= 0:
            ap.error("--head-residual-mlp-lr must be positive")
    if args.defer_reference_training:
        if not args.dual_head_mapped_single_head or args.head_kind != "affine" or args.decoder != "linear":
            ap.error("--defer-reference-training requires an affine mapped single-head linear decoder")
        if args.reference_type_residual_loss_weight or args.reference_primary_positive_loss_weight:
            ap.error("--defer-reference-training requires zero reference-specific loss weights")
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_checkpoint = (
            get_last_checkpoint(args.out)
            if args.resume_from_checkpoint == "auto"
            else args.resume_from_checkpoint
        )
        if args.resume_from_checkpoint == "auto" and resume_checkpoint is None:
            print(f"TRAIN: no checkpoint found under {args.out}; starting from step 0", flush=True)
        if (
            args.resume_from_checkpoint != "auto"
            and Path(args.resume_from_checkpoint).resolve().parent != Path(args.out).resolve()
            and args.save_limit < 2
        ):
            ap.error(
                "--save-limit must be at least 2 when explicit --resume-from-checkpoint is "
                "outside --out; otherwise best-checkpoint rotation can delete the new resume milestone"
            )
    resume_config = AutoConfig.from_pretrained(resume_checkpoint) if resume_checkpoint else None
    try:
        annotation_conventions = resolve_annotation_conventions(
            args.annotation_conventions, resume_config=resume_config, correctness_map=args.dual_head_map
        )
    except ValueError as error:
        ap.error(str(error))
    if annotation_conventions is not None:
        if (
            args.decoder != "linear"
            or args.union_members
            or args.partial_type_only
            or args.annotated_boundary_loss
            or args.annotated_boundary_telemetry
            or args.complete_boundary_loss
            or args.coarse_agreement_weight_20
            or args.coarse_agreement_weight_9
            or args.bioes_risk_weight
            or (
                (args.bioes_margin_boundary or args.bioes_margin_type)
                and args.bioes_margin_illegal_scale != 1.0
            )
            or args.native_heads_config
        ):
            ap.error(
                "annotation conventions require linear native/mapped supervision without union/type-only, exact-boundary auxiliaries, risk gating, or context-dependent margins"
            )
    try:
        args.training_windowing = resolve_training_windowing(
            args.training_windowing, resume_config=resume_config
        )
    except ValueError as error:
        ap.error(str(error))
    if resume_config is not None and hasattr(resume_config, "pii_context_field"):
        saved_context_field = resume_config.pii_context_field
        if args.context_field not in (None, saved_context_field):
            ap.error("context field cannot change during exact resume; start a new initialization stage")
        args.context_field = saved_context_field
    try:
        args.context_side = resolve_context_side(
            args.context_side,
            getattr(resume_config, "pii_context_side", "both"),
            exact_resume=resume_config is not None,
        )
    except ValueError as error:
        ap.error(str(error))
    if args.context_side == "previous" and not args.context_field:
        ap.error("--context-side previous requires --context-field")
    try:
        args.context_configurations = resolve_context_configurations(
            args.context_configurations,
            getattr(resume_config, "pii_context_configurations", None),
            exact_resume=resume_config is not None,
        )
        args.context_configuration_weights = resolve_context_weights(
            args.context_configuration_weights,
            args.context_configurations,
            getattr(resume_config, "pii_context_configuration_weights", None),
            exact_resume=resume_config is not None,
        )
    except ValueError as error:
        ap.error(str(error))
    try:
        check_context_mixture(args.context_field, args.context_side, args.context_configurations)
    except ValueError as error:
        ap.error(str(error))
    if resume_config is not None:
        saved_variant_policy = getattr(resume_config, "pii_annotation_variant_weighting", "legacy")
        if args.annotation_variant_weighting not in (None, saved_variant_policy):
            ap.error(
                "annotation weighting cannot change during exact resume; start a new initialization stage"
            )
        args.annotation_variant_weighting = saved_variant_policy
    else:
        args.annotation_variant_weighting = args.annotation_variant_weighting or "share"
    try:
        training_parameter_precision = resolve_training_parameter_precision(
            args.training_parameter_precision,
            resume_config=resume_config,
            evaluate_only=args.evaluate_only,
        )
    except ValueError as error:
        ap.error(str(error))
    parameter_dtype = torch.float32 if training_parameter_precision == "float32" else torch.bfloat16
    if args.extend_resume_horizon:
        if resume_checkpoint is None:
            ap.error("--extend-resume-horizon requires a resolved --resume-from-checkpoint")
        if not args.dual_head_mapped_single_head:
            ap.error("--extend-resume-horizon currently requires --dual-head-mapped-single-head")
    use_continuous_character = args.continuous_character_pretrain is not None
    if use_continuous_character:
        if args.predicate_loss_weight or args.subclass_loss_weight:
            ap.error("predicate and subclass heads do not support continuous-character concatenation")
        if not args.init_from_checkpoint:
            ap.error("--continuous-character-pretrain requires --init-from-checkpoint")
        if not args.continuous_character_pretrain.is_dir():
            ap.error(
                f"--continuous-character-pretrain is not a directory: {args.continuous_character_pretrain}"
            )
        if args.decoder != "linear" or args.head_kind != "affine":
            ap.error("continuous-character concatenation requires --decoder linear --head-kind affine")
        if len(encoder_layers) != 1 or resolved_token_offsets != (0,) or head_rank:
            ap.error(
                "continuous-character concatenation requires one final encoder layer, "
                "--token-offsets=0, and a full affine"
            )
        if args.continuous_character_lr is not None and args.continuous_character_lr <= 0:
            ap.error("--continuous-character-lr must be positive")
        if args.continuous_character_output_lr is not None and args.continuous_character_output_lr <= 0:
            ap.error("--continuous-character-output-lr must be positive")
        uses_character_tagger = (
            args.continuous_character_initialization == "tagger"
            or args.continuous_character_logit_initialization == "tagger"
        )
        if uses_character_tagger:
            if args.continuous_character_tagger is None:
                ap.error("tagger initialization requires --continuous-character-tagger")
            if not args.continuous_character_tagger.is_dir():
                ap.error(
                    f"--continuous-character-tagger is not a directory: {args.continuous_character_tagger}"
                )
        elif args.continuous_character_tagger is not None:
            ap.error("--continuous-character-tagger requires a tagger encoder or logit initialization")
        if args.continuous_character_logit_scale <= 0:
            ap.error("--continuous-character-logit-scale must be positive")
        if (
            args.continuous_character_logit_initialization == "zero"
            and args.continuous_character_logit_scale != 1.0
        ):
            ap.error("--continuous-character-logit-scale only applies to tagger logits")
        if args.continuous_character_auxiliary_loss_weight < 0:
            ap.error("--continuous-character-auxiliary-loss-weight must be nonnegative")
        if args.continuous_character_auxiliary_loss_fade_steps < 0:
            ap.error("--continuous-character-auxiliary-loss-fade-steps must be nonnegative")
        if bool(args.continuous_character_auxiliary_loss_weight) != bool(
            args.continuous_character_auxiliary_loss_fade_steps
        ):
            ap.error(
                "--continuous-character-auxiliary-loss-weight and "
                "--continuous-character-auxiliary-loss-fade-steps must be supplied together"
            )
    elif args.continuous_character_initialization != "pretrained":
        ap.error("--continuous-character-initialization requires --continuous-character-pretrain")
    elif args.continuous_character_pooling != "max":
        ap.error("--continuous-character-pooling requires --continuous-character-pretrain")
    elif args.continuous_character_lr is not None:
        ap.error("--continuous-character-lr requires --continuous-character-pretrain")
    elif args.continuous_character_tagger is not None:
        ap.error("--continuous-character-tagger requires --continuous-character-pretrain")
    elif args.continuous_character_logit_initialization != "zero":
        ap.error("--continuous-character-logit-initialization requires --continuous-character-pretrain")
    elif args.continuous_character_logit_scale != 1.0:
        ap.error("--continuous-character-logit-scale requires --continuous-character-pretrain")
    elif args.continuous_character_output_lr is not None:
        ap.error("--continuous-character-output-lr requires --continuous-character-pretrain")
    elif args.continuous_character_auxiliary_loss_weight:
        ap.error("--continuous-character-auxiliary-loss-weight requires --continuous-character-pretrain")
    elif args.continuous_character_auxiliary_loss_fade_steps:
        ap.error("--continuous-character-auxiliary-loss-fade-steps requires --continuous-character-pretrain")
    if args.save_initialized_only and not args.init_from_checkpoint:
        ap.error("--save-initialized-only requires --init-from-checkpoint")
    if args.warm_start_label_schema and not args.init_from_checkpoint:
        ap.error("--warm-start-label-schema requires --init-from-checkpoint")
    if args.warm_start_label_cut and not args.init_from_checkpoint:
        ap.error("--warm-start-label-cut requires --init-from-checkpoint")
    if args.warm_start_label_schema and args.warm_start_label_cut:
        ap.error("--warm-start-label-schema and --warm-start-label-cut are mutually exclusive")
    if args.reset_output_head and not args.init_from_checkpoint:
        ap.error("--reset-output-head requires --init-from-checkpoint")
    if args.reset_output_head and (args.warm_start_label_schema or args.warm_start_label_cut):
        ap.error("--reset-output-head cannot be combined with warm-start label projection")
    if args.mlm_replay_data:
        if not Path(args.mlm_replay_data).is_file():
            ap.error(f"--mlm-replay-data is not a file: {args.mlm_replay_data}")
        if not 0 < args.mlm_replay_prob < 1:
            ap.error("--mlm-replay-prob must be strictly between 0 and 1")
    elif args.mlm_replay_prob:
        ap.error("--mlm-replay-prob requires --mlm-replay-data")
    if args.use_mlm_objective and native_head_specs:
        ap.error("--native-heads-config cannot combine with physical-batch MLM")
    if args.use_mlm_objective:
        if args.mlm_loss_weight <= 0:
            ap.error("--mlm-loss-weight must be positive when physical-batch MLM is enabled")
        if not 0 < args.mlm_probability < 1:
            ap.error("--mlm-probability must be strictly between 0 and 1")
        if args.freeze_encoder:
            ap.error("physical-batch MLM requires a trainable encoder")
        if args.save_initialized_only:
            ap.error("physical-batch MLM cannot be used with --save-initialized-only")
    elif args.mlm_head_model:
        ap.error("--mlm-head-model requires an enabled MLM objective")
    if args.encoder_lr is not None and args.encoder_lr <= 0:
        ap.error("--encoder-lr must be positive")
    if args.encoder_lr is not None and args.freeze_encoder:
        ap.error("--encoder-lr cannot be combined with --freeze-encoder")
    if args.encoder_prior_checkpoint:
        if not Path(args.encoder_prior_checkpoint).is_dir():
            ap.error(f"--encoder-prior-checkpoint is not a directory: {args.encoder_prior_checkpoint}")
        if args.encoder_prior_weight <= 0:
            ap.error("--encoder-prior-weight must be positive")
        if args.encoder_prior_start_step < 0:
            ap.error("--encoder-prior-start-step must be nonnegative")
        if args.freeze_encoder:
            ap.error("--encoder-prior-checkpoint cannot be combined with --freeze-encoder")
    elif args.encoder_prior_weight or args.encoder_prior_start_step:
        ap.error("--encoder-prior-weight/--encoder-prior-start-step require --encoder-prior-checkpoint")
    if args.annotated_boundary_loss < 0:
        ap.error("--annotated-boundary-loss must be nonnegative")
    coarse_agreement_weights = (
        args.coarse_agreement_weight_20,
        args.coarse_agreement_weight_9,
        args.coarse_agreement_weight_presence_20,
    )
    if any(weight < 0 for weight in coarse_agreement_weights):
        ap.error("coarse-agreement weights must be nonnegative")
    if args.fine_label_loss_weight < 0:
        ap.error("--fine-label-loss-weight must be nonnegative")
    if args.fine_label_loss_weight != 1.0 and not any(coarse_agreement_weights):
        ap.error("--fine-label-loss-weight requires a positive coarse-agreement weight")
    if args.fine_label_loss_weight == 0.0 and not any(coarse_agreement_weights):
        ap.error("at least one token-classification objective must have positive weight")
    if args.o_token_loss_weight <= 0:
        ap.error("--o-token-loss-weight must be positive")
    if args.entity_dice_loss < 0:
        ap.error("--entity-dice-loss must be nonnegative")
    if args.complete_presence_loss < 0:
        ap.error("--complete-presence-loss must be nonnegative")
    if args.complete_presence_loss and args.dual_head_map is None:
        ap.error("--complete-presence-loss requires --dual-head-map")
    if args.complete_family_loss < 0:
        ap.error("--complete-family-loss must be nonnegative")
    if args.complete_family_loss and args.dual_head_map is None:
        ap.error("--complete-family-loss requires --dual-head-map")
    if args.complete_boundary_loss < 0:
        ap.error("--complete-boundary-loss must be nonnegative")
    if args.complete_boundary_loss and args.dual_head_map is None:
        ap.error("--complete-boundary-loss requires --dual-head-map")
    if (
        not math.isfinite(args.reference_primary_positive_loss_weight)
        or args.reference_primary_positive_loss_weight < 0
    ):
        ap.error("--reference-primary-positive-loss-weight must be finite and nonnegative")
    if args.reference_primary_positive_loss_weight:
        if args.dual_head_map is None or not args.dual_head_mapped_single_head:
            ap.error(
                "--reference-primary-positive-loss-weight requires "
                "--dual-head-map and --dual-head-mapped-single-head"
            )
        if not args.predicate_loss_weight or not args.subclass_loss_weight:
            ap.error(
                "--reference-primary-positive-loss-weight currently requires the joint "
                "predicate and subclass objectives"
            )
    if not math.isfinite(args.partial_primary_objective_weight) or args.partial_primary_objective_weight < 0:
        ap.error("--partial-primary-objective-weight must be finite and nonnegative")
    if args.partial_primary_objective_weight != 1.0 and (
        args.dual_head_map is None or not args.dual_head_mapped_single_head
    ):
        ap.error(
            "--partial-primary-objective-weight other than one requires "
            "--dual-head-map and --dual-head-mapped-single-head"
        )
    if args.logical_step_objective_normalization and args.accumulation_loss == "sum":
        ap.error(
            "--logical-step-objective-normalization defines its own step mean; drop --accumulation-loss sum"
        )
    if args.document_start_marker and not args.context_field:
        ap.error("--document-start-marker requires --context-field")
    if args.supervision_audit_only and not args.exclude_unsupervised_items:
        ap.error("--supervision-audit-only requires --exclude-unsupervised-items")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        ap.error("--max-grad-norm must be finite and positive")
    if args.logical_step_objective_normalization:
        if args.dual_head_map is None or not args.dual_head_mapped_single_head:
            ap.error(
                "--logical-step-objective-normalization currently requires "
                "--dual-head-map and --dual-head-mapped-single-head"
            )
        if not args.predicate_loss_weight or not args.subclass_loss_weight:
            ap.error(
                "--logical-step-objective-normalization currently requires the joint "
                "predicate and subclass objectives"
            )
        if args.partial_entity_pu_loss:
            ap.error("--logical-step-objective-normalization does not yet support PU risk")
        if args.entity_dice_loss:
            ap.error("--logical-step-objective-normalization does not support entity Dice")
    if not math.isfinite(args.predicate_loss_weight) or args.predicate_loss_weight < 0:
        ap.error("--predicate-loss-weight must be finite and nonnegative")
    if args.predicate_loss_weight:
        if not args.predicate_spec.is_file():
            ap.error(f"--predicate-spec is not a file: {args.predicate_spec}")
        if args.decoder != "linear" or args.head_kind == "stock":
            ap.error(
                "--predicate-loss-weight needs a concatenated-layer head: "
                "--decoder linear --head-kind affine/mlp"
            )
    if (
        not math.isfinite(args.reference_type_residual_loss_weight)
        or args.reference_type_residual_loss_weight < 0
    ):
        ap.error("--reference-type-residual-loss-weight must be finite and nonnegative")
    if not math.isfinite(args.subclass_loss_weight) or args.subclass_loss_weight < 0:
        ap.error("--subclass-loss-weight must be finite and nonnegative")
    if args.subclass_loss_weight:
        if not args.subclass_spec.is_file():
            ap.error(f"--subclass-spec is not a file: {args.subclass_spec}")
        if args.decoder != "linear" or args.head_kind == "stock":
            ap.error(
                "--subclass-loss-weight needs a concatenated-layer head: "
                "--decoder linear --head-kind affine/mlp"
            )
    if args.rdrop_alpha < 0:
        ap.error("--rdrop-alpha must be nonnegative")
    if args.partial_boundary_retention_loss < 0:
        ap.error("--partial-boundary-retention-loss must be nonnegative")
    if args.partial_boundary_retention_temperature <= 0:
        ap.error("--partial-boundary-retention-temperature must be positive")
    if args.partial_o_loss < 0:
        ap.error("--partial-o-loss must be nonnegative")
    if not math.isfinite(args.partial_entity_pu_loss) or args.partial_entity_pu_loss < 0:
        ap.error("--partial-entity-pu-loss must be finite and nonnegative")
    if (
        not math.isfinite(args.partial_expected_entity_ratio_loss)
        or args.partial_expected_entity_ratio_loss < 0
    ):
        ap.error("--partial-expected-entity-ratio-loss must be finite and nonnegative")
    if (
        not math.isfinite(args.partial_expected_entity_ratio_lower_width)
        or not 0 <= args.partial_expected_entity_ratio_lower_width < 1
    ):
        ap.error("--partial-expected-entity-ratio-lower-width must be finite and in [0, 1)")
    if args.partial_entity_pu_positive_margin is not None and (
        not math.isfinite(args.partial_entity_pu_positive_margin)
        or args.partial_entity_pu_positive_margin < 0
    ):
        ap.error("--partial-entity-pu-positive-margin must be finite and nonnegative")
    if not math.isfinite(args.partial_parent_presence_kl) or args.partial_parent_presence_kl < 0:
        ap.error("--partial-parent-presence-kl must be finite and nonnegative")
    if args.partial_entity_pu_loss:
        if args.partial_entity_pu_prior is None or not args.partial_entity_pu_prior.is_file():
            ap.error("--partial-entity-pu-loss requires an existing --partial-entity-pu-prior")
        if args.dual_head_map is None:
            ap.error("--partial-entity-pu-loss requires --dual-head-map")
        if args.partial_o_loss:
            ap.error("--partial-entity-pu-loss and --partial-o-loss are matched alternatives")
    elif (
        args.partial_entity_pu_prior is not None
        or args.partial_entity_pu_positive_margin is not None
        or args.partial_parent_presence_kl
    ):
        ap.error(
            "--partial-entity-pu-prior, --partial-entity-pu-positive-margin, and "
            "--partial-parent-presence-kl require a positive --partial-entity-pu-loss"
        )
    if args.partial_expected_entity_ratio_loss:
        if (
            args.partial_expected_entity_ratio_prior is None
            or not args.partial_expected_entity_ratio_prior.is_file()
        ):
            ap.error(
                "--partial-expected-entity-ratio-loss requires an existing "
                "--partial-expected-entity-ratio-prior"
            )
        if args.dual_head_map is None or not args.dual_head_mapped_single_head:
            ap.error(
                "--partial-expected-entity-ratio-loss requires --dual-head-map and "
                "--dual-head-mapped-single-head"
            )
        if not args.logical_step_objective_normalization:
            ap.error("--partial-expected-entity-ratio-loss requires --logical-step-objective-normalization")
        if args.partial_entity_pu_loss or args.partial_o_loss or args.partial_parent_presence_kl:
            ap.error(
                "--partial-expected-entity-ratio-loss is a matched alternative to PU, "
                "partial-O, and parent-presence objectives"
            )
    elif args.partial_expected_entity_ratio_prior is not None:
        ap.error(
            "--partial-expected-entity-ratio-prior requires a positive --partial-expected-entity-ratio-loss"
        )
    if args.partial_parent_presence_kl and (
        not args.dual_head_mapped_single_head or not args.init_from_checkpoint
    ):
        ap.error(
            "--partial-parent-presence-kl requires --dual-head-mapped-single-head and --init-from-checkpoint"
        )
    if args.partial_negative_loss < 0:
        ap.error("--partial-negative-loss must be nonnegative")
    try:
        partial_negative_sources = parse_partial_negative_sources(args.partial_negative_source)
    except ValueError as error:
        ap.error(str(error))
    if args.partial_negative_loss and not partial_negative_sources:
        ap.error("--partial-negative-loss requires --partial-negative-source")
    if partial_negative_sources and not args.partial_negative_loss:
        ap.error("--partial-negative-source requires a positive --partial-negative-loss")
    try:
        union_fine_weight_schedule = parse_union_fine_weight_schedule(args.union_v1_fine_weight_schedule)
    except ValueError as error:
        ap.error(str(error))
    if args.union_members is not None:
        if not Path(args.union_members).is_file():
            ap.error(f"--union-members is not a file: {args.union_members}")
        if union_fine_weight_schedule[0] == "linear" and args.max_steps <= 0:
            ap.error(
                "a linear --union-v1-fine-weight-schedule fades over the optimizer-step horizon and "
                "so requires a positive --max-steps"
            )
        if args.predicate_loss_weight:
            ap.error(
                "--predicate-loss-weight requires a native primary head and cannot be combined "
                "with --union-members"
            )
        # The union term rewrites the target of every supervised token and is
        # normalized by the stock token mean's own denominator. An objective that
        # reweights or duplicates that same mean would silently rescale it, so the
        # combination is refused rather than approximated.
        for flag, conflicting in (
            ("--coarse-agreement-weight-20", args.coarse_agreement_weight_20),
            ("--coarse-agreement-weight-9", args.coarse_agreement_weight_9),
            ("--coarse-agreement-weight-presence-20", args.coarse_agreement_weight_presence_20),
            ("--fine-label-loss-weight", args.fine_label_loss_weight != 1.0),
            ("--o-token-loss-weight", args.o_token_loss_weight != 1.0),
            ("--o-weight-config", args.o_weight_config is not None),
            ("--entity-dice-loss", args.entity_dice_loss),
            ("--rdrop-alpha", args.rdrop_alpha),
        ):
            if conflicting:
                ap.error(
                    f"--union-members reshapes the token cross-entropy and cannot be combined with {flag}"
                )
    elif union_fine_weight_schedule != CONSTANT_UNION_FINE_WEIGHT:
        ap.error("--union-v1-fine-weight-schedule requires --union-members")
    dual_head_schedule = None
    dual_head_retirement_step = None
    dual_head_lr_restart_step = 0
    if args.dual_head_bind_native_map and not (
        args.dual_head_mapped_single_head
        and args.dual_head_map
        and args.init_from_checkpoint
        and not args.resume_from_checkpoint
    ):
        ap.error(
            "--dual-head-bind-native-map requires a new --init-from-checkpoint stage with --dual-head-map and --dual-head-mapped-single-head"
        )
    if args.dual_head_map is not None:
        if not Path(args.dual_head_map).is_file():
            ap.error(f"--dual-head-map is not a file: {args.dual_head_map}")
        if args.union_members is not None:
            ap.error(
                "--union-members and --dual-head-map are two different answers to the same "
                "question: the first reads new classes off the incumbent head's rows, the second "
                "gives the new ontology its own head. Choose one"
            )
        if args.dual_head_init != "projected" and (
            args.dual_head_init_counts is not None or args.dual_head_init_p20_alpha != 0.0
        ):
            ap.error(
                "--dual-head-init-counts and nonzero --dual-head-init-p20-alpha require "
                "--dual-head-init projected"
            )
        if not math.isfinite(args.dual_head_init_add_k) or args.dual_head_init_add_k <= 0:
            ap.error("--dual-head-init-add-k must be finite and positive")
        if not math.isfinite(args.dual_head_init_p20_alpha) or args.dual_head_init_p20_alpha < 0:
            ap.error("--dual-head-init-p20-alpha must be finite and nonnegative")
        if args.dual_head_init == "projected" and not args.dual_head_init_tagset.is_file():
            ap.error(f"--dual-head-init-tagset is not a file: {args.dual_head_init_tagset}")
        if args.dual_head_init_counts is not None and not args.dual_head_init_counts.is_file():
            ap.error(f"--dual-head-init-counts is not a file: {args.dual_head_init_counts}")
        try:
            dual_head_schedule = parse_transition_schedule(args.dual_head_old_weight_schedule)
        except ValueError as error:
            ap.error(str(error))
        if args.dual_head_mapped_single_head:
            if not (args.init_from_checkpoint or args.resume_from_checkpoint):
                ap.error(
                    "--dual-head-mapped-single-head requires --init-from-checkpoint for a new "
                    "stage or --resume-from-checkpoint for exact same-stage resume"
                )
            if dual_head_schedule[:3] != ("constant", 0.0, 0.0):
                ap.error(
                    "--dual-head-mapped-single-head has no old head and therefore requires "
                    "--dual-head-old-weight-schedule constant:0"
                )
            if args.dual_head_keep_old_head or args.dual_head_restart_lr_after_transition:
                ap.error(
                    "--dual-head-mapped-single-head cannot keep a retired head or restart the "
                    "learning rate after an already-complete transition"
                )
            if (
                args.dual_head_init != "fallback"
                or args.dual_head_init_counts is not None
                or args.dual_head_init_p20_alpha != 0.0
            ):
                ap.error(
                    "--dual-head-mapped-single-head inherits its sole classifier exactly and "
                    "cannot use --dual-head-init, --dual-head-init-counts, or "
                    "--dual-head-init-p20-alpha"
                )
        if dual_head_schedule[0] == "linear" and args.max_steps <= 0:
            ap.error(
                "a linear --dual-head-old-weight-schedule moves over the optimizer-step horizon "
                "and so requires a positive --max-steps"
            )
        if not 0.0 <= args.dual_head_eval_weight <= 1.0:
            ap.error("--dual-head-eval-weight is a convex blend weight in [0, 1]")
        if args.decoder != "linear" or args.head_kind == "stock":
            ap.error(
                "--dual-head-map needs a concatenated-layer head: --decoder linear --head-kind affine/mlp"
            )
        # The dual-head objective replaces token CE outright. O-token weighting,
        # complete presence, partial O, and entity Dice are implemented on its
        # direct/mapped product-head terms; the other options below still reshape
        # a loss this trainer no longer computes.
        for flag, conflicting in (
            ("--coarse-agreement-weight-20", args.coarse_agreement_weight_20),
            ("--coarse-agreement-weight-9", args.coarse_agreement_weight_9),
            ("--coarse-agreement-weight-presence-20", args.coarse_agreement_weight_presence_20),
            ("--fine-label-loss-weight", args.fine_label_loss_weight != 1.0),
            ("--o-weight-config", args.o_weight_config is not None),
            ("--rdrop-alpha", args.rdrop_alpha),
            ("--annotated-boundary-loss", args.annotated_boundary_loss),
            ("--partial-boundary-retention-loss", args.partial_boundary_retention_loss),
            ("--partial-negative-loss", args.partial_negative_loss),
            ("--continuous-character-auxiliary-loss-weight", args.continuous_character_auxiliary_loss_weight),
            ("--mlm-replay-prob", args.mlm_replay_prob),
        ):
            if conflicting:
                ap.error(
                    f"--dual-head-map replaces the token cross-entropy and cannot be combined with {flag}"
                )
        if not args.dual_head_keep_old_head:
            dual_head_retirement_step = transition_completion_step(dual_head_schedule, args.max_steps)
        if args.dual_head_restart_lr_after_transition:
            completion = transition_completion_step(dual_head_schedule, args.max_steps)
            if completion is None:
                ap.error(
                    "--dual-head-restart-lr-after-transition needs a schedule that reaches zero: "
                    f"{args.dual_head_old_weight_schedule} never finishes handing over"
                )
            if completion == 0:
                ap.error(
                    "--dual-head-restart-lr-after-transition needs a handover to happen during the "
                    f"run: {args.dual_head_old_weight_schedule} starts at zero, so there is one "
                    "phase and the stock schedule already covers it"
                )
            if completion >= args.max_steps:
                ap.error(
                    "--dual-head-restart-lr-after-transition needs steps left after the handover; "
                    f"{args.dual_head_old_weight_schedule} finishes at step {completion} of "
                    f"{args.max_steps}. Compress it with a horizon span, as in "
                    f"{args.dual_head_old_weight_schedule}:0.25"
                )
            dual_head_lr_restart_step = completion
    elif (
        args.dual_head_old_weight_schedule != "linear:1.0:0.0"
        or args.dual_head_eval_weight != 0.0
        or args.dual_head_mapped_single_head
        or args.dual_head_keep_old_head
        or args.dual_head_restart_lr_after_transition
    ):
        ap.error(
            "--dual-head-old-weight-schedule, --dual-head-eval-weight, "
            "--dual-head-mapped-single-head, --dual-head-keep-old-head, and "
            "--dual-head-restart-lr-after-transition require --dual-head-map"
        )
    if (
        args.annotated_boundary_loss
        or args.annotated_boundary_telemetry
        or args.partial_boundary_retention_loss
        or args.partial_o_loss
        or args.partial_entity_pu_loss
        or args.partial_expected_entity_ratio_loss
        or args.partial_parent_presence_kl
        or args.partial_negative_loss
        or args.o_token_loss_weight != 1.0
        or args.o_weight_config is not None
        or args.entity_dice_loss
        or args.complete_boundary_loss
        or args.reference_primary_positive_loss_weight
        or args.partial_primary_objective_weight != 1.0
        or args.predicate_loss_weight
        or args.subclass_loss_weight
        or args.rdrop_alpha
        or args.fine_label_loss_weight != 1.0
        or args.union_members is not None
    ) and args.decoder != "linear":
        ap.error("auxiliary training objectives only support --decoder linear")
    if args.partial_boundary_retention_loss and not args.init_from_checkpoint:
        ap.error("--partial-boundary-retention-loss requires --init-from-checkpoint")
    if args.replay_data and not 0 < args.replay_prob < 1:
        ap.error("--replay-prob must be strictly between 0 and 1")
    if not args.replay_data and args.replay_prob != 0.1:
        ap.error("--replay-prob requires --replay-data")
    if args.min_language_share and not args.replay_data:
        ap.error("--min-language-share requires --replay-data")
    if args.min_replay_language_share and not args.replay_data:
        ap.error("--min-replay-language-share requires --replay-data")
    try:
        minimum_language_shares = parse_language_shares(args.min_language_share)
        minimum_replay_language_shares = parse_language_shares(args.min_replay_language_share)
    except ValueError as error:
        ap.error(str(error))
    unknown_partial_negative_types = sorted(
        {
            label_type
            for label_types in partial_negative_sources.values()
            for label_type in label_types
            if label_type not in nodes
        }
    )
    if unknown_partial_negative_types:
        ap.error(
            "--partial-negative-source types are absent from the corpus label vocabulary: "
            + ", ".join(unknown_partial_negative_types)
        )
    partial_negative_source_groups = {
        source: group_id for group_id, source in enumerate(sorted(partial_negative_sources))
    }
    partial_negative_label_groups = [
        [
            label2id[f"{prefix}-{label_type}"]
            for label_type in partial_negative_sources[source]
            for prefix in "BIES"
        ]
        for source in sorted(partial_negative_sources)
    ]
    union_groups = None
    if args.union_members is not None:
        try:
            union_groups = load_union_members(args.union_members, nodes, label2id)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
    if resolved_token_offsets != (0,):
        if args.decoder != "linear" or args.head_kind != "affine":
            ap.error("multi-token-state concatenation requires --decoder linear --head-kind affine")
        if len(encoder_layers) != 1:
            ap.error("multi-token-state concatenation requires exactly one --encoder-layers value")

    tokenizer_source = resume_checkpoint or args.init_from_checkpoint or args.model
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    entity_pu_prior_document = None
    entity_pu_prior_assignments = None
    entity_pu_prior_sha256 = None
    entity_ratio_prior_document = None
    entity_ratio_prior_assignments = None
    entity_ratio_prior_sha256 = None
    if args.partial_entity_pu_loss:
        try:
            (
                entity_pu_prior_document,
                entity_pu_prior_assignments,
                entity_pu_prior_sha256,
            ) = load_entity_pu_prior(
                args.partial_entity_pu_prior,
                tok,
                args.max_len,
                train_max_chars,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
    if args.partial_expected_entity_ratio_loss:
        try:
            (
                entity_ratio_prior_document,
                entity_ratio_prior_assignments,
                entity_ratio_prior_sha256,
            ) = load_entity_token_prior(
                args.partial_expected_entity_ratio_prior,
                tok,
                args.max_len,
                train_max_chars,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
    shared_warm_start_labels = 0
    character_projection = None
    if resume_checkpoint:
        checkpoint_config = AutoConfig.from_pretrained(resume_checkpoint)
        is_layered = getattr(checkpoint_config, "pii_head_architecture", None) == "concat_encoder_layers"
        if is_layered != (args.head_kind != "stock"):
            ap.error("resume checkpoint and requested --head-kind disagree on stock vs custom head")
        if is_layered:
            model = load_local_token_classifier(resume_checkpoint, dtype=parameter_dtype)
            requested_layers = resolve_encoder_layers(encoder_layers, model.config.num_hidden_layers)
            expected = (
                args.head_kind,
                requested_layers,
                resolved_token_offsets,
                head_rank,
            )
            observed = (
                model.config.pii_head_kind,
                tuple(model.config.pii_encoder_layers),
                tuple(getattr(model.config, "pii_token_offsets", [0])),
                int(model.config.pii_head_rank),
            )
            if expected != observed:
                ap.error(f"resume custom head mismatch: requested={expected}, checkpoint={observed}")
        elif args.decoder == "crf":
            model = MMBertCrfForTokenClassification.from_local_checkpoint(
                resume_checkpoint,
                encoder_dtype=parameter_dtype,
            )
        else:
            if args.head_kind != "stock" or encoder_layers != [-1] or resolved_token_offsets != (0,):
                ap.error("stock resume requires --head-kind stock --encoder-layers=-1 --token-offsets=0")
            model = load_local_token_classifier(resume_checkpoint, dtype=parameter_dtype)
        validate_warm_start_encoder(model, args.model)
    elif args.init_from_checkpoint:
        if args.decoder == "crf":
            ap.error("--init-from-checkpoint does not yet support --decoder crf")
        checkpoint_config = AutoConfig.from_pretrained(args.init_from_checkpoint)
        is_layered = getattr(checkpoint_config, "pii_head_architecture", None) == "concat_encoder_layers"
        if is_layered != (args.head_kind != "stock"):
            ap.error("warm-start checkpoint and requested --head-kind disagree on stock vs custom head")
        if is_layered:
            model = load_local_token_classifier(args.init_from_checkpoint, dtype=parameter_dtype)
            requested_layers = resolve_encoder_layers(encoder_layers, model.config.num_hidden_layers)
            expected = (
                args.head_kind,
                requested_layers,
                resolved_token_offsets,
                head_rank,
            )
            observed = (
                model.config.pii_head_kind,
                tuple(model.config.pii_encoder_layers),
                tuple(getattr(model.config, "pii_token_offsets", [0])),
                int(model.config.pii_head_rank),
            )
            if expected != observed:
                ap.error(f"warm-start custom head mismatch: requested={expected}, checkpoint={observed}")
        else:
            if args.head_kind != "stock" or encoder_layers != [-1] or resolved_token_offsets != (0,):
                ap.error("stock warm start requires --head-kind stock --encoder-layers=-1 --token-offsets=0")
            model = load_local_token_classifier(args.init_from_checkpoint, dtype=parameter_dtype)
        validate_warm_start_encoder(model, args.model)
        if args.dual_head_mapped_single_head:
            shared_warm_start_labels = model.config.num_labels
        else:
            shared_warm_start_labels = expand_output_labels(
                model,
                names,
                source_schema=args.warm_start_label_schema,
                source_cut=args.warm_start_label_cut,
                copy_shared_rows=not args.reset_output_head,
            )
        if use_continuous_character:
            model = ContinuousCharacterForTokenClassification.from_incumbent(
                model,
                args.continuous_character_pretrain,
                initialization=args.continuous_character_initialization,
                pooling=args.continuous_character_pooling,
                character_tagger=args.continuous_character_tagger,
                logit_initialization=args.continuous_character_logit_initialization,
                logit_scale=args.continuous_character_logit_scale,
            )
            character_projection = load_character_projection(
                args.continuous_character_pretrain / CHARACTER_PROJECTION_FILENAME
            )
    elif args.decoder == "crf":
        if args.head_kind != "stock" or encoder_layers != [-1] or resolved_token_offsets != (0,):
            ap.error("--decoder crf only supports --head-kind stock --encoder-layers=-1 --token-offsets=0")
        model = MMBertCrfForTokenClassification.from_encoder_pretrained(
            args.model,
            names,
            dtype=parameter_dtype,
        )
    elif args.head_kind != "stock":
        model = LayerConcatForTokenClassification.from_encoder_pretrained(
            args.model,
            names,
            layers=encoder_layers,
            token_offsets=resolved_token_offsets,
            head_kind=args.head_kind,
            rank=head_rank,
            dtype=parameter_dtype,
            dropout=args.classifier_dropout,
        )
    else:
        if encoder_layers != [-1] or resolved_token_offsets != (0,):
            ap.error("--head-kind stock only supports --encoder-layers=-1 --token-offsets=0")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model,
            num_labels=len(names),
            id2label=dict(enumerate(names)),
            label2id=label2id,
            dtype=parameter_dtype,
            ignore_mismatched_sizes=True,
        )
    if resume_checkpoint or args.init_from_checkpoint:
        saved_classifier_kernel = getattr(
            model.config,
            "pii_stock_classifier_kernel",
            "framework",
        )
        if saved_classifier_kernel != args.stock_classifier_kernel:
            ap.error(
                "checkpoint stock classifier kernel mismatch: "
                f"checkpoint={saved_classifier_kernel!r}, requested={args.stock_classifier_kernel!r}"
            )
    if args.stock_classifier_kernel != "framework":
        configure_stock_classifier_kernel(model, args.stock_classifier_kernel)
    tag_status_types = ()
    prompt_slots = None
    prompt_languages = None
    if args.tag_status_slots:
        if args.soft_registers:
            ap.error("--tag-status-slots and --soft-registers both claim input positions; use one")
        tag_status_types = tuple(
            sorted(
                {label.split("-", 1)[1] for label in model.config.id2label.values() if label != "O"}
                - set(DEFERRED_REFERENCE_TYPES)
            )
        )
        if args.tag_status_language:
            language_spec = json.loads(Path(DEFAULT_REGISTER_LANGUAGE_SPEC).read_text(encoding="utf-8"))
            prompt_languages = tuple(language_spec["languages"])
        vocabulary = model.base_model.get_input_embeddings().weight

        def embed_text(text):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"no tokens to initialize a prompt slot from {text!r}")
            return vocabulary[ids].detach().float().mean(0).cpu()

        slots = layout_slots(
            tag_status_types, args.tag_status_layout, args.tag_status_slots == "gold", bool(prompt_languages)
        )
        prompt_slots = PromptSlots(
            model.config.hidden_size,
            slots,
            tag_status_types,
            prompt_languages or (),
            initial=torch.stack([embed_text(slot["label"].replace("_", " ")) for slot in slots]),
            language_initial=(
                torch.stack([embed_text(PROMPT_LANGUAGE_NAMES[code]) for code in prompt_languages])
                if prompt_languages
                else None
            ),
        ).to(next(model.parameters()).device)
        install_prompt_slots(model, prompt_slots)
        print(
            json.dumps(
                {
                    "tag_status_prompt": {
                        "slots": args.tag_status_slots,
                        "layout": args.tag_status_layout,
                        "width": prompt_slots.width,
                        "types": len(tag_status_types),
                        "languages": len(prompt_languages or ()),
                        "anchors": [slot["anchor"] for slot in slots],
                        "lr": args.tag_status_lr,
                        "parameters": sum(p.numel() for p in prompt_slots.parameters()),
                    }
                }
            ),
            flush=True,
        )
    elif args.tag_status_language:
        ap.error("--tag-status-language needs --tag-status-slots")
    source_classes = (
        SourceClasses.load(args.soft_register_source_classes) if args.soft_register_source_classes else None
    )
    if source_classes is not None and not args.soft_registers:
        ap.error("--soft-register-source-classes needs --soft-registers to condition")
    if args.soft_register_source_dropout and source_classes is None:
        ap.error("--soft-register-source-dropout needs --soft-register-source-classes")
    if args.soft_register_language and not args.soft_registers:
        ap.error("--soft-register-language needs --soft-registers to condition")
    if args.soft_register_language_dropout and not args.soft_register_language:
        ap.error("--soft-register-language-dropout needs --soft-register-language")
    domain_scopes = ()
    domain_posteriors = None
    if args.soft_register_domain:
        domain_posteriors = load_domain_posteriors(args.soft_register_domain)
        domain_scopes = (
            ("segment", "document")
            if args.soft_register_domain_scope == "both"
            else (args.soft_register_domain_scope,)
        )
        if not args.soft_registers:
            ap.error("--soft-register-domain needs --soft-registers to condition")
    register_languages = None
    if args.soft_register_language:
        # Frozen in a spec rather than collected from the rows, which are not loaded yet at
        # this point and would in any case let the vocabulary drift with the pool.
        spec = json.loads(Path(DEFAULT_REGISTER_LANGUAGE_SPEC).read_text(encoding="utf-8"))
        if spec.get("schema") != "pii-register-languages/v1":
            ap.error(f"unsupported language spec schema {spec.get('schema')!r}")
        register_languages = (spec["unknown"], *spec["languages"])
    if args.soft_registers:
        register_conditions = ()
        if source_classes is not None:
            register_conditions += (("source", tuple(source_classes.names)),)
        if register_languages is not None:
            register_conditions += (("language", register_languages),)
        for scope in domain_scopes:
            width = len(next(iter(domain_posteriors.values()))["segment"])
            register_conditions += (
                (f"domain_{scope}", (UNKNOWN_CONDITION, *(f"d{index}" for index in range(width)))),
            )
        soft_registers = install_soft_registers(
            model,
            args.soft_registers,
            layerwise=not args.soft_registers_input_only,
            conditions=register_conditions,
        )
        print(
            json.dumps(
                {
                    "soft_registers": args.soft_registers,
                    "layerwise": not args.soft_registers_input_only,
                    "conditions": [[name, len(values)] for name, values in register_conditions],
                    "source_dropout": args.soft_register_source_dropout,
                    "language_dropout": args.soft_register_language_dropout,
                    "register_lr": args.soft_register_lr,
                    "parameters": sum(p.numel() for p in soft_registers.parameters()),
                }
            ),
            flush=True,
        )
    if args.language_bias:
        # Same frozen vocabulary the register condition uses, minus its `unknown` slot:
        # routing is explicit here and -1 selects the shared head for anything unlisted.
        spec = json.loads(Path(DEFAULT_REGISTER_LANGUAGE_SPEC).read_text(encoding="utf-8"))
        bias_languages = list(spec["languages"])
        model.attach_language_bias(bias_languages, strength=args.language_bias_strength)
        print(
            json.dumps(
                {
                    "language_bias": len(bias_languages),
                    "strength": args.language_bias_strength,
                    "parameters": model.language_bias.weight.numel(),
                }
            ),
            flush=True,
        )
    else:
        bias_languages = None
    resume_from_horizon = None
    if args.extend_resume_horizon:
        try:
            resume_from_horizon = resume_extension_restart_step(
                model.config,
                args.max_steps,
                enabled=True,
            )
        except ValueError as error:
            ap.error(str(error))
        dual_head_lr_restart_step = resume_from_horizon
    elif resume_checkpoint and args.dual_head_mapped_single_head:
        saved_restart_step = getattr(model.config, "pii_dual_head_lr_restart_step", 0) or 0
        if not isinstance(saved_restart_step, int) or saved_restart_step < 0:
            ap.error("resume checkpoint has an invalid pii_dual_head_lr_restart_step")
        dual_head_lr_restart_step = saved_restart_step
    if resume_checkpoint:
        # Checkpoints from before these options trained summed and sorted-window.
        saved_accumulation = getattr(model.config, "pii_accumulation_loss", None) or "sum"
        requested_accumulation = (
            "logical-step-mass" if args.logical_step_objective_normalization else args.accumulation_loss
        )
        saved_normalized = bool(getattr(model.config, "pii_logical_step_objective_normalization", False))
        if not saved_normalized and saved_accumulation != requested_accumulation:
            ap.error(
                f"exact resume accumulation-loss mismatch: checkpoint={saved_accumulation}, "
                f"requested={requested_accumulation}; pass --accumulation-loss {saved_accumulation}"
            )
        saved_formation = getattr(model.config, "pii_batch_formation", None) or "sorted-window"
        if saved_formation != args.batch_formation:
            ap.error(
                f"exact resume batch-formation mismatch: checkpoint={saved_formation}, "
                f"requested={args.batch_formation}; pass --batch-formation {saved_formation}"
            )
        saved_draw_policy = getattr(model.config, "pii_draw_policy", None) or "systematic"
        if saved_draw_policy != args.draw_policy:
            ap.error(
                f"exact resume draw-policy mismatch: checkpoint={saved_draw_policy}, "
                f"requested={args.draw_policy}; pass --draw-policy {saved_draw_policy}"
            )
        saved_logical_step_normalization = bool(
            getattr(model.config, "pii_logical_step_objective_normalization", False)
        )
        if saved_logical_step_normalization != args.logical_step_objective_normalization:
            ap.error(
                "exact resume logical-step objective normalization mismatch: "
                f"checkpoint={saved_logical_step_normalization}, "
                f"requested={args.logical_step_objective_normalization}"
            )
        saved_reference_primary_weight = float(
            getattr(model.config, "pii_reference_primary_positive_loss_weight", 0.0) or 0.0
        )
        if saved_reference_primary_weight != args.reference_primary_positive_loss_weight:
            ap.error(
                "exact resume reference-primary-positive objective mismatch: "
                f"checkpoint={saved_reference_primary_weight:g}, "
                f"requested={args.reference_primary_positive_loss_weight:g}"
            )
        saved_partial_primary_weight = float(
            getattr(model.config, "pii_partial_primary_objective_weight", 1.0)
        )
        if saved_partial_primary_weight != args.partial_primary_objective_weight:
            ap.error(
                "exact resume partial-primary objective mismatch: "
                f"checkpoint={saved_partial_primary_weight:g}, "
                f"requested={args.partial_primary_objective_weight:g}"
            )
        saved_predicate_weight = float(getattr(model.config, "pii_predicate_loss_weight", 0.0) or 0.0)
        if saved_predicate_weight != args.predicate_loss_weight:
            ap.error(
                "exact resume predicate objective mismatch: "
                f"checkpoint={saved_predicate_weight:g}, requested={args.predicate_loss_weight:g}"
            )
        saved_predicate_mask_channels = tuple(
            getattr(model.config, "pii_predicate_objective_mask_channels", ()) or ()
        )
        if saved_predicate_mask_channels != predicate_objective_mask_channels:
            ap.error(
                "exact resume predicate objective mask mismatch: "
                f"checkpoint={list(saved_predicate_mask_channels)}, "
                f"requested={list(predicate_objective_mask_channels)}"
            )
        saved_subclass_weight = float(getattr(model.config, "pii_subclass_loss_weight", 0.0) or 0.0)
        if saved_subclass_weight != args.subclass_loss_weight:
            ap.error(
                "exact resume subclass objective mismatch: "
                f"checkpoint={saved_subclass_weight:g}, requested={args.subclass_loss_weight:g}"
            )
        saved_reference_residual_weight = float(
            getattr(model.config, "pii_reference_type_residual_loss_weight", 0.0) or 0.0
        )
        if saved_reference_residual_weight != args.reference_type_residual_loss_weight:
            ap.error(
                "exact resume reference residual objective mismatch: "
                f"checkpoint={saved_reference_residual_weight:g}, "
                f"requested={args.reference_type_residual_loss_weight:g}"
            )
        saved_balance_sha256 = getattr(model.config, "pii_ont3_added_head_balance_sha256", None)
        requested_balance_sha256 = None if added_head_balance_spec is None else added_head_balance_spec.sha256
        if saved_balance_sha256 != requested_balance_sha256:
            ap.error(
                "exact resume added-head balance mismatch: "
                f"checkpoint={saved_balance_sha256!r}, requested={requested_balance_sha256!r}"
            )
    if predicate_spec is not None:
        if not isinstance(model, LayerConcatForTokenClassification):
            ap.error("predicate heads require LayerConcatForTokenClassification")
        saved_spec_sha256 = getattr(model.config, "pii_predicate_spec_sha256", None)
        if saved_spec_sha256 is not None and saved_spec_sha256 != predicate_spec.sha256:
            ap.error(
                f"predicate spec mismatch: checkpoint={saved_spec_sha256}, requested={predicate_spec.sha256}"
            )
        try:
            attached_predicate_head = model.attach_predicate_head(
                list(predicate_spec.channels),
                condition_types=list(predicate_condition_types),
            )
        except ValueError as error:
            ap.error(str(error))
        if resume_checkpoint and attached_predicate_head:
            ap.error(
                "exact resume cannot add a predicate head; start a new stage with --init-from-checkpoint"
            )
        model.config.pii_predicate_spec = str(predicate_spec.path)
        model.config.pii_predicate_spec_sha256 = predicate_spec.sha256
        model.config.pii_predicate_applicable_types = {
            channel: sorted(predicate_spec.applicable_types[channel]) for channel in predicate_spec.channels
        }
        model.config.pii_predicate_objective_mask_channels = list(predicate_objective_mask_channels)
    if added_head_balance_spec is not None:
        model.config.pii_ont3_added_head_balance = str(added_head_balance_spec.path)
        model.config.pii_ont3_added_head_balance_sha256 = added_head_balance_spec.sha256
        model.config.pii_ont3_reference_positive_weights = added_head_balance_spec.reference_positive_weights
        model.config.pii_ont3_predicate_positive_weights = added_head_balance_spec.predicate_positive_weights
        model.config.pii_predicate_conditioning = (
            "primary_type" if args.predicate_conditioning == "primary-type" else "none"
        )
        model.config.pii_predicate_condition_types = list(predicate_condition_types)
    if subclass_spec is not None:
        if not isinstance(model, LayerConcatForTokenClassification):
            ap.error("subclass heads require LayerConcatForTokenClassification")
        saved_spec_sha256 = getattr(model.config, "pii_subclass_spec_sha256", None)
        if saved_spec_sha256 is not None and saved_spec_sha256 != subclass_spec.sha256:
            ap.error(
                f"subclass spec mismatch: checkpoint={saved_spec_sha256}, requested={subclass_spec.sha256}"
            )
        try:
            attached_subclass_head = model.attach_subclass_head(
                subclass_spec.config_blocks(),
                spec_sha256=subclass_spec.sha256,
            )
        except ValueError as error:
            ap.error(str(error))
        if resume_checkpoint and attached_subclass_head:
            ap.error("exact resume cannot add a subclass head; start a new stage with --init-from-checkpoint")
        model.config.pii_subclass_spec = str(subclass_spec.path)
        model.config.pii_subclass_spec_sha256 = subclass_spec.sha256
        model.config.pii_subclass_blocks = subclass_spec.config_blocks()
    retention_teacher = None
    if args.partial_boundary_retention_loss:
        retention_teacher = load_local_token_classifier(args.init_from_checkpoint)
        validate_warm_start_encoder(retention_teacher, args.model)
        teacher_names = [
            retention_teacher.config.id2label[index] for index in range(retention_teacher.config.num_labels)
        ]
        if teacher_names != names:
            ap.error("partial boundary retention currently requires an exact parent/student label vocabulary")
    correctness_map = None
    affine_projection = None
    affine_projection_receipt_sha256 = None
    projection_identity = {}
    mapped_single_head = args.dual_head_mapped_single_head
    resumed_retired_dual_head = False
    if args.dual_head_map is not None:
        try:
            map_document = load_correctness_map(args.dual_head_map)
            secondary_names = new_space_labels(map_document["ontology"]["primary_types"])
            correctness_map = build_correctness_map(
                map_document,
                names,
                secondary_names,
                map_path=str(args.dual_head_map),
            )
            if added_head_balance_spec is not None:
                successor_primary_types = {
                    label.partition("-")[2] for label in secondary_names if label != "O"
                }
                missing_reference_types = sorted(
                    set(added_head_balance_spec.reference_positive_weights) - successor_primary_types
                )
                if missing_reference_types:
                    raise ValueError(
                        "added-head balance references are absent from the successor ontology: "
                        + ", ".join(missing_reference_types)
                    )
            if args.dual_head_init == "projected":
                affine_projection = load_affine_projection(
                    map_document,
                    names,
                    secondary_names,
                    map_path=args.dual_head_map,
                    count_path=args.dual_head_init_counts,
                    tagset_path=args.dual_head_init_tagset,
                    add_k=args.dual_head_init_add_k,
                    alpha=args.dual_head_init_p20_alpha,
                    p20_cut=args.dual_head_init_p20_cut,
                )
                projection_identity = {
                    "pii_dual_head_init_projection_sha256": semantic_sha256(affine_projection.receipt),
                    "pii_dual_head_init_counts_sha256": affine_projection.receipt["counts"]["sha256"],
                    "pii_dual_head_init_tagset_sha256": affine_projection.receipt["tagset"]["sha256"],
                    "pii_dual_head_init_p20_cut_sha256": affine_projection.receipt["p20"]["cut_sha256"],
                    "pii_dual_head_init_add_k": args.dual_head_init_add_k,
                    "pii_dual_head_init_p20_alpha": args.dual_head_init_p20_alpha,
                }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            ap.error(str(error))
        seeded_rows = 0
        if mapped_single_head:
            if resume_checkpoint is None:
                try:
                    shared_warm_start_labels = expand_mapped_single_head_successor(
                        model,
                        correctness_map,
                        secondary_names,
                    )
                except ValueError as error:
                    ap.error(str(error))
            try:
                validate_mapped_single_head_checkpoint(
                    model,
                    correctness_map,
                    secondary_names,
                    exact_resume=resume_checkpoint is not None,
                    old_weight_schedule=args.dual_head_old_weight_schedule,
                    eval_weight=args.dual_head_eval_weight,
                    retirement_step=dual_head_retirement_step,
                    horizon=args.max_steps,
                    o_token_loss_weight=args.o_token_loss_weight,
                    entity_dice_loss_weight=args.entity_dice_loss,
                    partial_o_loss_weight=args.partial_o_loss,
                    partial_entity_pu_loss_weight=args.partial_entity_pu_loss,
                    partial_entity_pu_prior_sha256=entity_pu_prior_sha256,
                    partial_entity_pu_positive_margin=args.partial_entity_pu_positive_margin,
                    partial_expected_entity_ratio_loss_weight=(args.partial_expected_entity_ratio_loss),
                    partial_expected_entity_ratio_prior_sha256=entity_ratio_prior_sha256,
                    partial_expected_entity_ratio_lower_width=(
                        args.partial_expected_entity_ratio_lower_width
                    ),
                    partial_parent_presence_kl_weight=args.partial_parent_presence_kl,
                    complete_presence_loss_weight=args.complete_presence_loss,
                    complete_family_loss_weight=args.complete_family_loss,
                    complete_boundary_loss_weight=args.complete_boundary_loss,
                    resume_from_horizon=resume_from_horizon,
                    bind_native_map=args.dual_head_bind_native_map,
                )
            except ValueError as error:
                ap.error(str(error))
            resumed_retired_dual_head = True
        elif resume_checkpoint:
            try:
                resumed_retired_dual_head = validate_dual_head_resume(
                    model,
                    correctness_map,
                    names,
                    secondary_names,
                    old_weight_schedule=args.dual_head_old_weight_schedule,
                    initialization=args.dual_head_init,
                    eval_weight=args.dual_head_eval_weight,
                    retirement_step=dual_head_retirement_step,
                    lr_restart_step=dual_head_lr_restart_step,
                    horizon=args.max_steps,
                    o_token_loss_weight=args.o_token_loss_weight,
                    entity_dice_loss_weight=args.entity_dice_loss,
                    partial_o_loss_weight=args.partial_o_loss,
                    partial_entity_pu_loss_weight=args.partial_entity_pu_loss,
                    partial_entity_pu_prior_sha256=entity_pu_prior_sha256,
                    partial_entity_pu_positive_margin=args.partial_entity_pu_positive_margin,
                    partial_parent_presence_kl_weight=args.partial_parent_presence_kl,
                    complete_presence_loss_weight=args.complete_presence_loss,
                    complete_family_loss_weight=args.complete_family_loss,
                    complete_boundary_loss_weight=args.complete_boundary_loss,
                    projection_identity=projection_identity,
                )
            except ValueError as error:
                ap.error(str(error))
        else:
            if getattr(model.config, "pii_dual_head_retired_at_step", None) is not None:
                ap.error(
                    "this warm-start checkpoint is already past its transition: its old head was "
                    f"retired at step {model.config.pii_dual_head_retired_at_step}. Use "
                    "--dual-head-mapped-single-head for a new mapped-supervision stage, or "
                    "--resume-from-checkpoint with the original dual-head recipe"
                )
            attach_secondary_head = getattr(model, "attach_secondary_head", None)
            if not callable(attach_secondary_head):
                raise AssertionError("dual-head model cannot attach a secondary classifier")
            attach_secondary_head(secondary_names)
            primary_classifier = getattr(model, "classifier", None)
            secondary_classifier = getattr(model, "secondary_classifier", None)
            if not isinstance(primary_classifier, nn.Linear) or not isinstance(
                secondary_classifier, nn.Linear
            ):
                raise AssertionError("dual-head initialization requires affine classifiers")
            if args.dual_head_init == "projected":
                if affine_projection is None:
                    raise AssertionError("projected initialization was not constructed")
                initialize_affine_classifier(
                    secondary_classifier,
                    primary_classifier,
                    affine_projection,
                )
                seeded_rows = len(secondary_names)
                affine_projection_receipt_sha256 = write_affine_projection_receipt(
                    Path(args.out) / "pii_dual_head_affine_projection.json",
                    affine_projection,
                    primary_classifier,
                    secondary_classifier,
                )
            elif args.dual_head_init == "fallback":
                sources = fallback_source_rows(map_document, names, secondary_names)
                with torch.no_grad():
                    for new_id, rows in enumerate(sources):
                        if not rows:
                            continue
                        secondary_classifier.weight[new_id] = primary_classifier.weight[rows].mean(dim=0)
                        secondary_classifier.bias[new_id] = primary_classifier.bias[rows].mean()
                        seeded_rows += 1
        head_state = (
            "mapped-single-head-resume"
            if mapped_single_head and resume_checkpoint
            else "mapped-single-head-new-stage"
            if mapped_single_head
            else "resumed-retired-new-head"
            if resumed_retired_dual_head
            else "resumed-active-two-head"
            if resume_checkpoint
            else "initialized-two-head"
        )
        initialization_status = (
            f"new-head-init=inherited-retired preserved-new-rows={len(secondary_names)}"
            if mapped_single_head
            else f"new-head-init={args.dual_head_init} seeded-new-rows={seeded_rows}/{len(secondary_names)}"
        )
        print(
            "TRAIN: dual-head supervision active: "
            f"map={args.dual_head_map} map-sha256={correctness_map.map_sha256[:12]} "
            f"ontology-sha256={correctness_map.ontology_sha256[:12]} "
            f"incumbent-rows={len(names)} new-rows={len(secondary_names)} "
            f"empirically-admitted-targets={correctness_map.empirical_targets} "
            f"head-state={head_state} "
            f"{initialization_status} "
            f"old-weight-schedule={args.dual_head_old_weight_schedule} "
            f"eval-weight={args.dual_head_eval_weight:g} horizon={args.max_steps} "
            f"retire-old-head-at="
            f"{'never' if dual_head_retirement_step is None else dual_head_retirement_step} "
            f"lr-restart-at={dual_head_lr_restart_step or 'none'}",
            flush=True,
        )
    if reference_type_residual_types:
        if not isinstance(model, LayerConcatForTokenClassification):
            ap.error("reference-type residuals require LayerConcatForTokenClassification")
        if set(reference_type_residual_types) != set(correctness_map.legacy_outside_unknown_primary_types):
            ap.error(
                "reference-type residual inventory must equal the correctness map's "
                "legacy-O unknown primary types"
            )
        try:
            attached_reference_type_residual = model.attach_reference_type_residual(
                list(reference_type_residual_types)
            )
        except ValueError as error:
            ap.error(str(error))
        if resume_checkpoint and attached_reference_type_residual:
            ap.error(
                "exact resume cannot add a reference-type residual; start a new stage with "
                "--init-from-checkpoint"
            )
    saved_reference_deferral = bool(getattr(model.config, "pii_defer_reference_training", False))
    if resume_checkpoint and saved_reference_deferral != args.defer_reference_training:
        ap.error("exact resume reference deferral differs; start a new stage to change projection")
    if args.defer_reference_training or saved_reference_deferral:
        if not isinstance(model, LayerConcatForTokenClassification):
            ap.error("reference deferral requires LayerConcatForTokenClassification")
        model.defer_reference_training(args.defer_reference_training)
        print(f"TRAIN: defer-reference-training={args.defer_reference_training}", flush=True)
    partial_parent_presence_teacher = None
    if args.partial_parent_presence_kl:
        partial_parent_presence_teacher = load_encoder_prior_model(args.init_from_checkpoint)
        parent_names = [
            partial_parent_presence_teacher.config.id2label[index]
            for index in range(partial_parent_presence_teacher.config.num_labels)
        ]
        expected_parent_names = list(correctness_map.parent_new_labels or correctness_map.new_labels)
        if parent_names != expected_parent_names:
            ap.error("parent-presence trust region requires the mapped successor's exact parent product head")
        if getattr(partial_parent_presence_teacher, "secondary_classifier", None) is not None:
            ap.error("parent-presence trust region requires a retired mapped single-head parent")
        partial_parent_presence_teacher.requires_grad_(False)
        partial_parent_presence_teacher.eval()
    record_training_objective_config(model.config, args, partial_negative_sources)
    # Recorded unconditionally for the same reason as every objective above: a warm
    # start inherits config metadata, so a continuation without the flag must not
    # keep its parent's union settings.
    model.config.pii_dual_head_map = None if args.dual_head_map is None else str(args.dual_head_map)
    model.config.pii_dual_head_map_sha256 = (
        correctness_map.map_sha256 if correctness_map is not None else None
    )
    model.config.pii_dual_head_ontology_sha256 = (
        correctness_map.ontology_sha256 if correctness_map is not None else None
    )
    model.config.pii_dual_head_old_weight_schedule = (
        args.dual_head_old_weight_schedule if correctness_map is not None else None
    )
    if not mapped_single_head:
        model.config.pii_dual_head_init = args.dual_head_init if correctness_map is not None else None
        model.config.pii_dual_head_init_counts = (
            str(args.dual_head_init_counts)
            if correctness_map is not None and args.dual_head_init_counts is not None
            else None
        )
        model.config.pii_dual_head_init_tagset = (
            str(args.dual_head_init_tagset) if affine_projection is not None else None
        )
        model.config.pii_dual_head_init_p20_cut = (
            args.dual_head_init_p20_cut if affine_projection is not None else None
        )
        projection_config = {
            "pii_dual_head_init_projection_sha256": None,
            "pii_dual_head_init_counts_sha256": None,
            "pii_dual_head_init_tagset_sha256": None,
            "pii_dual_head_init_p20_cut_sha256": None,
            "pii_dual_head_init_add_k": None,
            "pii_dual_head_init_p20_alpha": None,
        }
        projection_config.update(projection_identity)
        for field, value in projection_config.items():
            setattr(model.config, field, value)
        model.config.pii_dual_head_init_receipt_sha256 = (
            getattr(model.config, "pii_dual_head_init_receipt_sha256", None)
            if resume_checkpoint
            else affine_projection_receipt_sha256
        )
    model.config.pii_dual_head_mapped_single_head = (
        mapped_single_head if correctness_map is not None else None
    )
    model.config.pii_dual_head_eval_weight = (
        args.dual_head_eval_weight if correctness_map is not None else None
    )
    model.config.pii_dual_head_retirement_step = (
        dual_head_retirement_step if correctness_map is not None else None
    )
    model.config.pii_dual_head_lr_restart_step = (
        dual_head_lr_restart_step if correctness_map is not None else None
    )
    model.config.pii_dual_head_horizon = args.max_steps if correctness_map is not None else None
    if resume_checkpoint and (getattr(model.config, "pii_native_head_training", None) or native_head_specs):
        if getattr(model.config, "pii_native_head_training", None) != native_head_specs:
            ap.error(
                "native-head resume requires the checkpoint's exact source configuration and objective weights"
            )
    if native_head_specs:
        if not isinstance(model, LayerConcatForTokenClassification):
            ap.error("--native-heads-config requires a LayerConcatForTokenClassification checkpoint")
        for spec in native_head_specs:
            native_labels = new_space_labels(spec["types"])
            coverage = native_head_coverage(spec)
            existing = model.config.pii_native_heads.get(spec["name"])
            if existing is None:
                model.attach_native_head(spec["name"], native_labels, coverage=coverage)
            elif existing != {"labels": native_labels, "coverage": coverage}:
                ap.error(f"native head {spec['name']}: checkpoint inventory or coverage mismatch")
    if not args.evaluate_only:
        model.config.pii_native_head_training = native_head_specs
        if training_parameter_precision == "float32":
            model.float()
            if isinstance(model, ContinuousCharacterForTokenClassification):
                model.config.pii_classifier_dtype = "float32"
                model.config.pii_character_parameter_dtype = "float32"
                model.config.pii_character_classifier_dtype = "float32"
        model.config.pii_training_parameter_precision = training_parameter_precision
    model.config.pii_resume_extended_from_horizon = resume_from_horizon
    model.config.pii_union_members = None if args.union_members is None else str(args.union_members)
    model.config.pii_union_members_sha256 = None if union_groups is None else union_groups.members_sha256
    model.config.pii_union_v1_fine_weight_schedule = (
        args.union_v1_fine_weight_schedule if union_groups is not None else None
    )
    model.config.pii_encoder_frozen = bool(args.freeze_encoder)
    model.config.pii_trainable_top_encoder_layers = args.trainable_top_encoder_layers
    if args.freeze_encoder:
        freeze_summary = freeze_encoder_parameters(model)
    elif args.trainable_top_encoder_layers is not None:
        freeze_summary = freeze_encoder_except_top_layers(model, args.trainable_top_encoder_layers)
    else:
        freeze_summary = None
    try:
        frozen_inherited_output_rows = (
            successor_parent_output_rows(model) if args.freeze_inherited_output_rows else 0
        )
    except ValueError as error:
        ap.error(str(error))
    saved_frozen_inherited_output_rows = int(
        getattr(model.config, "pii_frozen_inherited_output_rows", 0) or 0
    )
    if resume_checkpoint and saved_frozen_inherited_output_rows != frozen_inherited_output_rows:
        ap.error(
            "exact resume inherited-output freeze mismatch: "
            f"checkpoint={saved_frozen_inherited_output_rows}, "
            f"requested={frozen_inherited_output_rows}"
        )
    model.config.pii_frozen_inherited_output_rows = frozen_inherited_output_rows
    if freeze_summary is not None:
        freeze_summary["frozen_inherited_output_rows"] = frozen_inherited_output_rows
    mlm_head = (
        load_frozen_mlm_head(args.mlm_head_model or args.model, model.config)
        if args.use_mlm_objective
        else None
    )
    encoder_prior_anchors = None
    if args.encoder_prior_checkpoint:
        encoder_prior_model = load_encoder_prior_model(args.encoder_prior_checkpoint)
        validate_warm_start_encoder(encoder_prior_model, args.model)
        encoder_prior_anchors = encoder_parameter_prior_anchors(model, encoder_prior_model)
        del encoder_prior_model
    inherited_output_prior_anchors = None
    inherited_output_prior_rows = 0
    if args.inherited_output_prior_checkpoint:
        inherited_output_prior_model = load_encoder_prior_model(args.inherited_output_prior_checkpoint)
        validate_warm_start_encoder(inherited_output_prior_model, args.model)
        try:
            inherited_output_prior_rows = successor_parent_output_rows(model)
            model_names = [model.config.id2label[index] for index in range(model.config.num_labels)]
            anchor_names = [
                inherited_output_prior_model.config.id2label[index]
                for index in range(inherited_output_prior_model.config.num_labels)
            ]
            if anchor_names != model_names:
                raise ValueError("inherited-output prior requires the anchor's exact ordered label inventory")
            inherited_output_prior_anchors = inherited_output_parameter_prior_anchors(
                model,
                inherited_output_prior_model,
                inherited_output_prior_rows,
            )
        except ValueError as error:
            ap.error(str(error))
        finally:
            del inherited_output_prior_model
    model.config.pii_inherited_output_prior_rows = inherited_output_prior_rows

    model.config.pii_context_field = args.context_field
    model.config.pii_context_side = args.context_side
    model.config.pii_context_configurations = args.context_configurations
    model.config.pii_context_configuration_weights = args.context_configuration_weights
    model.config.pii_context_max_length = args.max_len
    if args.save_initialized_only:
        labels_path = Path(args.data) / "labels.json"
        classifier_name, classifier = token_classifier_head(model)
        provenance = {
            "schema_version": 1,
            "parent_checkpoint": args.init_from_checkpoint,
            "warm_start_label_schema": args.warm_start_label_schema or None,
            "warm_start_label_cut": args.warm_start_label_cut or None,
            "output_head_reinitialized": args.reset_output_head,
            "seed": model_seed,
            "reproducibility": training_seeds,
            "labels_path": str(labels_path),
            "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "bioes_outputs": int(model.config.num_labels),
            "copied_output_rows": shared_warm_start_labels,
            "new_output_rows": int(model.config.num_labels) - shared_warm_start_labels,
            "classifier_attribute": classifier_name,
            "classifier_in_features": classifier.in_features,
            "classifier_parameters": sum(parameter.numel() for parameter in classifier.parameters()),
            "predicate_head": (
                {
                    "channels": list(predicate_spec.channels),
                    "spec_path": str(predicate_spec.path),
                    "spec_sha256": predicate_spec.sha256,
                    "parameters": sum(
                        parameter.numel() for parameter in model.predicate_classifier.parameters()
                    ),
                }
                if predicate_spec is not None
                else None
            ),
            "subclass_head": (
                {
                    "spec_path": str(subclass_spec.path),
                    "spec_sha256": subclass_spec.sha256,
                    "blocks": subclass_spec.config_blocks(),
                    "parameters": sum(
                        parameter.numel() for parameter in model.subclass_classifier.parameters()
                    ),
                }
                if subclass_spec is not None
                else None
            ),
            "continuous_character": (
                {
                    "pretrain": str(args.continuous_character_pretrain.resolve()),
                    "initialization": args.continuous_character_initialization,
                    "pooling": args.continuous_character_pooling,
                    "tagger": (
                        str(args.continuous_character_tagger.resolve())
                        if args.continuous_character_tagger is not None
                        else None
                    ),
                    "logit_initialization": args.continuous_character_logit_initialization,
                    "logit_scale": args.continuous_character_logit_scale,
                    "projection_sha256": model.config.pii_character_projection_sha256,
                }
                if use_continuous_character
                else None
            ),
        }
        save_initialized_checkpoint(
            model,
            tok,
            args.out,
            provenance,
            allowed_existing=(
                ("pii_dual_head_affine_projection.json",)
                if affine_projection_receipt_sha256 is not None
                else ()
            ),
        )
        print(
            f"TRAIN: initialized-only control saved to {args.out}; "
            f"copied={shared_warm_start_labels} "
            f"new={int(model.config.num_labels) - shared_warm_start_labels}",
            flush=True,
        )
        log_format.headline(
            f"{os.path.basename(args.out.rstrip('/'))}: initialized-only control saved; "
            f"copied={shared_warm_start_labels} "
            f"new={int(model.config.num_labels) - shared_warm_start_labels}"
        )
        return

    window_tokenizer = tok if args.training_windowing == "token-capacity" else None
    prompt_width = prompt_slots.width if prompt_slots is not None else 0
    window_max_tokens = (
        args.max_len - args.soft_registers - prompt_width if window_tokenizer is not None else None
    )
    model.config.pii_training_windowing = args.training_windowing
    print(f"TRAIN-WINDOWING: policy={args.training_windowing} max_tokens={window_max_tokens}", flush=True)
    train_rows = window_records(
        os.path.join(args.data, "train.jsonl"),
        train_max_chars,
        sampling_pool="data" if train_pool_specs else None,
        defer_references=args.defer_reference_training,
        context_field=args.context_field,
        tokenizer=window_tokenizer,
        max_tokens=window_max_tokens,
    )
    for pool_name, pool_path, _pool_weight in train_pool_specs:
        train_rows.extend(
            window_records(
                pool_path,
                train_max_chars,
                sampling_pool=pool_name,
                defer_references=args.defer_reference_training,
                context_field=args.context_field,
                tokenizer=window_tokenizer,
                max_tokens=window_max_tokens,
            )
        )
    val_universe_rows = window_records(
        os.path.join(args.data, "val.jsonl"),
        validation_max_chars,
        defer_references=args.defer_reference_training,
        context_field=args.context_field,
        tokenizer=window_tokenizer,
        max_tokens=window_max_tokens,
    )
    replay_windows = 0
    primary_train_rows = list(train_rows)
    replay = []
    if args.replay_data:
        if args.max_train_windows:
            primary_budget = round(args.max_train_windows * (1 - args.replay_prob))
            train_rows = train_rows[:primary_budget]
        primary_train_rows = list(train_rows)
        replay_source = window_records(
            os.path.join(args.replay_data, "train.jsonl"),
            train_max_chars,
            defer_references=args.defer_reference_training,
            context_field=args.context_field,
            tokenizer=window_tokenizer,
            max_tokens=window_max_tokens,
        )
        replay = replay_rows_at_probability(
            train_rows,
            replay_source,
            args.replay_prob,
            replay_seed,
            minimum_language_shares=minimum_language_shares,
            minimum_replay_language_shares=minimum_replay_language_shares,
        )
        replay_windows = len(replay)
        train_rows.extend(replay)
    elif args.max_train_windows:
        train_rows = train_rows[: args.max_train_windows]
    val_rows = select_validation_rows(
        val_universe_rows,
        limit=args.max_val_windows,
        seed=validation_seed,
        policy=args.val_selection,
    )
    if not val_rows:
        ap.error("validation selection is empty")
    validation_receipt = validation_selection_receipt(
        val_universe_rows,
        val_rows,
        policy=args.val_selection,
        seed=validation_seed,
        limit=args.max_val_windows,
    )
    validation_receipt_path, validation_draw_is_new = persist_validation_selection(
        args.out,
        validation_receipt,
        resume=resume_checkpoint is not None,
    )
    validation_draw_changed_on_resume = validation_draw_is_new and resume_checkpoint is not None
    model.config.pii_validation_selection = validation_receipt
    print(
        "TRAIN-VALIDATION: "
        f"policy={args.val_selection} seed={validation_seed} "
        f"selected={len(val_rows)}/{len(val_universe_rows)} "
        f"sha256={validation_receipt['selected_sha256'][:12]} "
        f"supervision={validation_receipt['selected_supervision']} "
        f"label-spaces={validation_receipt['selected_label_spaces']} "
        f"languages={validation_receipt['selected_languages']} "
        f"receipt={validation_receipt_path}",
        flush=True,
    )
    supervised_train_rows = list(train_rows)
    primary_objective_weight_rows = [
        row for row in supervised_train_rows if "primary_span_objective_weights" in row
    ]
    if primary_objective_weight_rows and correctness_map is None and not args.native_new_label_space:
        ap.error(
            "primary_span_objective_weights currently require mapped successor supervision "
            "through --dual-head-map or a native new-ontology head (--native-new-label-space)"
        )
    model.config.pii_primary_span_objective_weights = bool(
        primary_objective_weight_rows or added_head_balance_spec is not None
    )
    if entity_pu_prior_assignments is not None or entity_ratio_prior_assignments is not None:
        complete_rows = [
            row
            for row in supervised_train_rows
            if row.get("supervision", COMPLETE_SUPERVISION) == COMPLETE_SUPERVISION
        ]
        partial_rows = [
            row
            for row in supervised_train_rows
            if row.get("supervision", COMPLETE_SUPERVISION) == ANNOTATED_SPANS_ONLY
        ]
        if not complete_rows or not partial_rows:
            ap.error("entity-token prior objectives require both complete and partial supervision rows")
        observed_prior_keys = {
            (row_source_name(row), row.get("lang") or row.get("language")) for row in partial_rows
        }
        prior_arms = (
            (
                "PU",
                entity_pu_prior_assignments,
                args.partial_entity_pu_prior,
                entity_pu_prior_sha256,
            ),
            (
                "EXPECTED-ENTITY-RATIO",
                entity_ratio_prior_assignments,
                args.partial_expected_entity_ratio_prior,
                entity_ratio_prior_sha256,
            ),
        )
        for prior_name, assignments, receipt_path, receipt_sha256 in prior_arms:
            if assignments is None:
                continue
            missing_keys = sorted(observed_prior_keys - set(assignments))
            if not missing_keys:
                print(
                    f"TRAIN-{prior_name}-PRIOR: "
                    f"receipt={receipt_path} sha256={receipt_sha256[:12]} "
                    f"complete-windows={len(complete_rows)} partial-windows={len(partial_rows)} "
                    f"active-strata={len(observed_prior_keys)}",
                    flush=True,
                )
                continue
            ap.error(
                f"{prior_name.lower()} prior receipt is missing training source/language strata: "
                + ", ".join(f"{source}/{language}" for source, language in missing_keys)
            )
    sampling_derivation_receipt = None
    if train_pool_specs and not args.sampling_config:
        pool_weights = {name: weight for name, _path, weight in train_pool_specs}
        pool_weights["data"] = 1.0 - sum(pool_weights.values())
        supervised_pool_keys = [str(row["sampling_pool"]) for row in supervised_train_rows]
        supervised_sampling_weights = example_weights_from_pools(
            supervised_pool_keys,
            pool_weights,
        )
    else:
        (
            supervised_sampling_weights,
            supervised_pool_keys,
            sampling_derivation_receipt,
        ) = sampling_plan_for_rows(supervised_train_rows, args.sampling_config)
    predicate_exposure_receipt = (
        None
        if sampling_derivation_receipt is None
        else sampling_derivation_receipt.get("predicate_exposure_strata")
    )
    if predicate_exposure_receipt is not None:
        if predicate_spec is None:
            ap.error("predicate exposure strata require an active --predicate-spec")
        declared_channels = tuple(predicate_exposure_receipt["predicate_targets"])
        if declared_channels != predicate_spec.channels:
            ap.error(
                "predicate exposure channels must exactly match the active predicate spec: "
                f"declared={list(declared_channels)} active={list(predicate_spec.channels)}"
            )
        if correctness_map is None:
            ap.error("predicate exposure reference targets require an active --dual-head-map")
        declared_primary = set(predicate_exposure_receipt["primary_targets"])
        expected_primary = set(correctness_map.legacy_outside_unknown_primary_types)
        if declared_primary != expected_primary:
            ap.error(
                "predicate exposure primary targets must exactly match the successor reference types: "
                f"declared={sorted(declared_primary)} expected={sorted(expected_primary)}"
            )
    use_mlm_objective = args.use_mlm_objective
    mlm_rows = []
    mlm_windowing = None
    mlm_sampling_weights = []
    joint_supervised_sampling_weights = supervised_sampling_weights
    gold_rebalance = None
    if args.mlm_replay_data:
        mlm_rows, mlm_windowing = window_mlm_replay_records(
            args.mlm_replay_data,
            train_max_chars,
            tok,
            args.max_len,
        )
        for row in mlm_rows:
            row["sampling_pool"] = "mlm-replay"
        mlm_sampling_weights = [row["sampling_weight"] for row in mlm_rows]
        base_supervised_weights = (
            [1.0] * len(supervised_train_rows)
            if supervised_sampling_weights is None
            else supervised_sampling_weights
        )
        base_family_shares = summarize_weighted_values(
            supervised_train_rows,
            base_supervised_weights,
            "sampling_family",
        )
        base_gold_share = base_family_shares.get("gold", 0.0)
        if base_gold_share and args.mlm_replay_gold_policy == "preserve":
            supervised_gold_target = base_gold_share / (1 - args.mlm_replay_prob)
            joint_supervised_sampling_weights, gold_rebalance = rebalance_family_share_within_groups(
                supervised_train_rows,
                base_supervised_weights,
                group_field="lang",
                family_field="sampling_family",
                family_value="gold",
                target_share=supervised_gold_target,
            )
    annotation_variant_sampling = None
    if args.annotation_variant_weighting == "share":
        joint_supervised_sampling_weights, annotation_variant_sampling = share_annotation_sampling_mass(
            supervised_train_rows, joint_supervised_sampling_weights
        )
        supervised_sampling_weights = joint_supervised_sampling_weights
        print(
            "TRAIN-ANNOTATION-SHARING: " + json.dumps(annotation_variant_sampling, sort_keys=True),
            flush=True,
        )
    model.config.pii_annotation_variant_weighting = args.annotation_variant_weighting
    model.config.pii_annotation_conventions = (
        annotation_conventions.document if annotation_conventions else None
    )
    model.config.pii_annotation_conventions_sha256 = (
        annotation_conventions.sha256 if annotation_conventions else None
    )
    if annotation_variant_sampling is not None and not args.sampling_epoch_windows:
        args.sampling_epoch_windows = annotation_variant_sampling["distinct_inputs"] + len(mlm_rows)
    if args.mlm_replay_data:
        sampling_weights = combine_objective_sampling_weights(
            joint_supervised_sampling_weights,
            len(supervised_train_rows),
            mlm_sampling_weights,
            args.mlm_replay_prob,
        )
        train_rows = [*supervised_train_rows, *mlm_rows]
    else:
        sampling_weights = supervised_sampling_weights
    if args.physical_batch_mlm and sampling_weights is None:
        sampling_weights = [1.0] * len(train_rows)
    if args.mlm_replay_data:
        sampling_pool_keys = [*supervised_pool_keys, *["mlm-replay"] * len(mlm_rows)]
    else:
        sampling_pool_keys = supervised_pool_keys
    sampling_pool_mass = Counter()
    if sampling_weights is not None:
        for pool, weight in zip(sampling_pool_keys, sampling_weights, strict=True):
            sampling_pool_mass[pool] += float(weight)
    mlm_batch_probabilities = None
    if args.physical_batch_mlm:
        try:
            mlm_batch_probabilities, sampling_pool_mass = resolve_physical_batch_mlm_probabilities(
                sampling_pool_keys,
                sampling_weights,
                default_probability=args.mlm_physical_batch_prob,
                overrides=args.mlm_pool_probabilities,
                forced_mlm_pools=("mlm-replay",) if args.mlm_replay_data else (),
            )
        except ValueError as error:
            ap.error(str(error))
    sampling_pool_total = sum(sampling_pool_mass.values())
    expected_batch_pool_shares = {
        pool: mass / sampling_pool_total for pool, mass in sorted(sampling_pool_mass.items())
    }
    expected_mlm_batch_fraction = (
        sum(sampling_pool_mass[pool] * probability for pool, probability in mlm_batch_probabilities.items())
        / sampling_pool_total
        if mlm_batch_probabilities and sampling_pool_total
        else 0.0
    )
    predicate_exposure_sampling = None
    if predicate_exposure_receipt is not None:
        if supervised_sampling_weights is None:
            ap.error("predicate exposure strata require weighted trainer sampling")
        exposure_field = predicate_exposure_receipt["field"]
        expected_stratum_shares = summarize_weighted_values(
            train_rows,
            sampling_weights,
            exposure_field,
        )
        required_targets = sorted(predicate_exposure_receipt["targets"])
        missing_target_shares = [
            target for target in required_targets if expected_stratum_shares.get(target, 0.0) <= 0.0
        ]
        if missing_target_shares:
            ap.error("predicate exposure targets have no sampling mass: " + ", ".join(missing_target_shares))
        sampling_epoch_windows = args.sampling_epoch_windows or len(train_rows)
        expected_target_draws = {
            target: expected_stratum_shares[target] * sampling_epoch_windows for target in required_targets
        }
        guaranteed_target_draw_floor = {
            target: math.floor(expected + 1e-12) for target, expected in expected_target_draws.items()
        }
        uncovered = [target for target, floor in guaranteed_target_draw_floor.items() if floor < 1]
        if uncovered:
            ap.error(
                "sampling epoch is too short to guarantee every reserved predicate exposure row: "
                + ", ".join(uncovered)
            )
        predicate_exposure_sampling = {
            "assignment": predicate_exposure_receipt,
            "expected_stratum_shares": expected_stratum_shares,
            "expected_target_draws_per_sampler_epoch": expected_target_draws,
            "systematic_target_draw_floor_per_sampler_epoch": guaranteed_target_draw_floor,
            "minimum_systematic_target_draw_floor": min(guaranteed_target_draw_floor.values()),
            "sampler_epoch_windows": sampling_epoch_windows,
            "guarantee_basis": (
                "one distinct row per target; randomized systematic sampling selects each row at "
                "least floor(epoch_windows * normalized_row_weight) times"
            ),
        }
    joint_family_shares = summarize_weighted_values(
        train_rows,
        sampling_weights,
        "sampling_family",
    )
    surface_realizer = None
    surface_realization_receipt = None
    if surface_realization_requested:
        if sampling_weights is None:
            ap.error("sample-time surface realization requires weighted trainer sampling")
        if args.surface_realization_predicate_pool:
            if predicate_spec is None:
                ap.error("predicate surface realization requires an active --predicate-spec")
        surface_seed = SeedTree(sampling_seed).fork("sample-time-surface-realization")
        try:
            surface_realizer = SampleTimeSurfaceRealizer(
                surface_pool_path=args.surface_realization_pool,
                surface_pool_split="train",
                surface_recipe_path=args.surface_realization_recipe,
                locale_profile_path=args.surface_realization_locale_profile,
                materialization_version=args.surface_realization_version,
                seed=surface_seed,
                row_string_equals=surface_realization_equalities,
                reject_unlocalized_categorical=(
                    args.surface_realization_unlocalized_categorical_policy == "drop"
                ),
                context_generator_path=args.surface_realization_context_generator,
                context_generator_rate=args.surface_realization_context_generator_rate,
                context_generator_route=args.surface_realization_context_generator_route,
                predicate_surface_pool_path=args.surface_realization_predicate_pool,
                predicate_surface_policy_path=args.surface_realization_predicate_policy,
            )
            if surface_realizer.predicate_surface_policy is not None:
                surface_realizer.predicate_surface_policy.validate_channels(predicate_spec.channels)
        except (OSError, RuntimeError, ValueError) as error:
            ap.error(str(error))
        surface_realization_receipt = surface_realizer.receipt(train_rows)
        if not surface_realization_receipt["eligible_rows"]:
            ap.error("sample-time surface-realization predicates matched no post-windowing train rows")
    effective_language_shares = (
        summarize_weighted_values(
            train_rows,
            sampling_weights,
            "lang",
        )
        or summarize_languages(train_rows)["shares"]
    )
    # Language weights are an input, not a policy computed here. Each row's weight is
    # multiplied by its language's weight, an unlisted language weighs 1, and the receipt
    # reports the share every language ended up with beside the share it had before. A
    # weight set by accident therefore shows up as a share nobody intended, which is the
    # point of stating them rather than deriving them: see scripts/pii_language_weights.py
    # for the tool that computes a set meeting a minimum-share policy.
    language_weight_receipt = None
    if language_weights:
        languages = [str(row.get("lang") or "<unknown>") for row in train_rows]
        unknown = sorted(set(language_weights) - set(languages))
        if unknown:
            ap.error(f"--language-weights names languages with no rows: {unknown}")
        if sampling_weights is None:
            sampling_weights = [1.0] * len(train_rows)
        before = dict(effective_language_shares)
        sampling_weights = [
            weight * float(language_weights.get(language, 1.0))
            for weight, language in zip(sampling_weights, languages, strict=True)
        ]
        total = math.fsum(sampling_weights)
        if total <= 0:
            ap.error("--language-weights zeroes every training row")
        sampling_weights = [weight / total for weight in sampling_weights]
        effective_language_shares = summarize_weighted_values(train_rows, sampling_weights, "lang")
        language_weight_receipt = {
            "weights": {k: float(v) for k, v in sorted(language_weights.items())},
            "share_before": {k: round(v, 8) for k, v in sorted(before.items())},
            "share_after": {k: round(v, 8) for k, v in sorted(effective_language_shares.items())},
        }
    if native_head_specs:
        combined_language_mass = Counter(effective_language_shares)
        for spec in native_head_specs:
            rows = native_head_rows[spec["name"]]
            for language, count in Counter(row["lang"] for row in rows).items():
                combined_language_mass[language] += spec["probability"] * count / len(rows)
        total_mass = sum(combined_language_mass.values())
        effective_language_shares = {
            language: mass / total_mass for language, mass in sorted(combined_language_mass.items())
        }
    # Say what is about to be checked, always. Four separate fixes to the language floor
    # reported the same two failing numbers, and there was no way to tell from the log
    # whether the floor had run, whether it had moved anything, or whether the quantity
    # being checked was the one being adjusted.
    # The logger is not configured in this context, so a log line here is invisible while
    # an error is not. Four fixes to the language floor reported the same two failing
    # numbers and nothing distinguished the floor not running from it adjusting a
    # quantity other than the one checked, so the diagnostic goes where it will be seen.
    if args.minimum_language_share:
        _low = sorted(effective_language_shares.items(), key=lambda kv: kv[1])[:4]
        if min(effective_language_shares.values()) + 1e-12 < args.minimum_language_share:
            ap.error(
                "the supplied language weights do not meet the declared minimum share: "
                f"minimum={args.minimum_language_share} "
                f"weighted={language_weight_receipt is not None} "
                f"weighted={sampling_weights is not None} "
                f"languages={len(effective_language_shares)} "
                f"lowest={[(k, round(v, 6)) for k, v in _low]} "
                + (
                    "before="
                    + str(
                        sorted((k, round(v, 6)) for k, v in language_weight_receipt["share_before"].items())[
                            :4
                        ]
                    )
                    if language_weight_receipt
                    else "no-receipt"
                )
            )
    try:
        core_language_support = validate_core_language_support(
            effective_language_shares,
            language_round=args.language_round,
            split_manifest=args.language_support_split_manifest,
            split_component=args.language_support_split_component,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        ap.error(str(error))
    model.config.pii_core_language_support = core_language_support
    # A multiplier states intent and the share is the consequence; both travel with the
    # checkpoint so a later reader can tell what this run actually trained on.
    model.config.pii_language_weights = language_weight_receipt
    model.config.pii_mlm_gold_rebalance = gold_rebalance
    language_mix = {
        "reproducibility": training_seeds,
        "native_heads": {
            "configuration": native_head_specs,
            "source_sentences": {name: len(rows) for name, rows in native_head_rows.items()},
            "combined_expected_example_language_shares": effective_language_shares,
            "includes_zero_weight_control_forwards": True,
        }
        if native_head_specs
        else None,
        "windowing": {
            "training_max_characters": train_max_chars,
            "validation_max_characters": validation_max_chars,
        },
        "core_language_support": core_language_support,
        "minimum_language_shares": minimum_language_shares,
        "minimum_replay_language_shares": minimum_replay_language_shares,
        "primary": summarize_languages(primary_train_rows),
        "primary_sources": summarize_values(primary_train_rows, "src"),
        "selected_replay": summarize_languages(replay),
        "selected_replay_mix_sources": summarize_values(replay, "mix_source"),
        "selected_replay_sources": summarize_values(replay, "src"),
        "mixed": summarize_languages(supervised_train_rows),
        "mixed_sources": summarize_values(supervised_train_rows, "src"),
        "supervision": dict(sorted(Counter(row["supervision"] for row in supervised_train_rows).items())),
        "sampling": {
            "enabled": supervised_sampling_weights is not None,
            "annotation_variant_weighting": args.annotation_variant_weighting,
            "annotation_variant_sharing": annotation_variant_sampling,
            "config": args.sampling_config or None,
            "epoch_windows": args.sampling_epoch_windows or len(train_rows),
            "expected_batch_pool_shares": expected_batch_pool_shares,
            "expected_language_shares": summarize_weighted_values(
                supervised_train_rows,
                supervised_sampling_weights,
                "lang",
            ),
            "expected_mix_source_shares": summarize_weighted_values(
                supervised_train_rows,
                supervised_sampling_weights,
                "mix_source",
            ),
            "expected_source_shares": summarize_weighted_values(
                supervised_train_rows,
                supervised_sampling_weights,
                "src",
            ),
            "expected_file_pool_shares": summarize_weighted_values(
                supervised_train_rows,
                supervised_sampling_weights,
                "sampling_pool",
            ),
            "expected_family_shares": summarize_weighted_values(
                supervised_train_rows,
                supervised_sampling_weights,
                "sampling_family",
            ),
            "predicate_exposure": predicate_exposure_sampling,
            "mlm_adjusted_expected_family_shares": summarize_weighted_values(
                supervised_train_rows,
                joint_supervised_sampling_weights,
                "sampling_family",
            ),
        },
        "mlm_replay": {
            "enabled": bool(args.mlm_replay_data),
            "objective_enabled": use_mlm_objective,
            "path": args.mlm_replay_data or None,
            "sha256": (
                hashlib.sha256(Path(args.mlm_replay_data).read_bytes()).hexdigest()
                if args.mlm_replay_data
                else None
            ),
            "windows": len(mlm_rows),
            "windowing": mlm_windowing,
            "languages": summarize_languages(mlm_rows),
            "expected_language_shares": summarize_weighted_values(
                mlm_rows,
                mlm_sampling_weights or None,
                "lang",
            ),
            "sample_probability": args.mlm_replay_prob,
            "gold_policy": args.mlm_replay_gold_policy,
            "default_physical_batch_probability": args.mlm_physical_batch_prob,
            "physical_batch_probability_by_pool": mlm_batch_probabilities,
            "loss_weight": args.mlm_loss_weight,
            "mask_probability": args.mlm_probability,
            "head_model": (args.mlm_head_model or args.model) if use_mlm_objective else None,
        },
        "joint_objective_sampling": {
            "unit": "physical_batch_slot",
            "expected_tag_share": 1 - expected_mlm_batch_fraction,
            "expected_mlm_share": expected_mlm_batch_fraction,
            "mlm_gradient_scale_per_logical_step": (args.mlm_loss_weight * expected_mlm_batch_fraction),
            "measured_active_batch_compute_overhead_prior": MLM_ACTIVE_BATCH_COMPUTE_OVERHEAD_PRIOR,
            "estimated_logical_step_compute_overhead": (
                MLM_ACTIVE_BATCH_COMPUTE_OVERHEAD_PRIOR * expected_mlm_batch_fraction
            ),
            "length_bucket_width": args.sampling_length_bucket_width,
            "length_binning_reference": "independent_weighted_rows_within_selected_pool",
            "expected_language_shares": summarize_weighted_values(
                train_rows,
                sampling_weights,
                "lang",
            ),
            "expected_family_shares": joint_family_shares,
            "expected_gold_share": (joint_family_shares or {}).get("gold", 0.0),
            "gold_rebalance": gold_rebalance,
        },
        "sample_time_surface_realization": {
            "enabled": surface_realizer is not None,
            "receipt": surface_realization_receipt,
        },
        "primary_span_objective_weights": {
            "enabled": bool(primary_objective_weight_rows or added_head_balance_spec is not None),
            "rows": len(primary_objective_weight_rows),
            "span_weights": dict(
                sorted(
                    Counter(
                        str(float(weight))
                        for row in primary_objective_weight_rows
                        for weight in row["primary_span_objective_weights"]
                    ).items()
                )
            ),
        },
        "ont3_added_head_balance": (
            {
                "path": str(added_head_balance_spec.path),
                "sha256": added_head_balance_spec.sha256,
                "reference_positive_weights": (added_head_balance_spec.reference_positive_weights),
                "predicate_positive_weights": (added_head_balance_spec.predicate_positive_weights),
            }
            if added_head_balance_spec is not None
            else None
        ),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if entity_pu_prior_document is not None:
        prior_copy_path = Path(args.out) / "partial_entity_pu_prior.json"
        prior_copy = json.dumps(entity_pu_prior_document, indent=2, sort_keys=True) + "\n"
        if prior_copy_path.exists() and prior_copy_path.read_text(encoding="utf-8") != prior_copy:
            raise ValueError(f"existing bound PU prior differs: {prior_copy_path}")
        prior_copy_path.write_text(prior_copy, encoding="utf-8")
    if entity_ratio_prior_document is not None:
        prior_copy_path = Path(args.out) / "partial_expected_entity_ratio_prior.json"
        prior_copy = json.dumps(entity_ratio_prior_document, indent=2, sort_keys=True) + "\n"
        if prior_copy_path.exists() and prior_copy_path.read_text(encoding="utf-8") != prior_copy:
            raise ValueError(f"existing bound expected entity-ratio prior differs: {prior_copy_path}")
        prior_copy_path.write_text(prior_copy, encoding="utf-8")
    if added_head_balance_spec is not None:
        balance_copy_path = Path(args.out) / "ont3_added_head_balance.json"
        balance_copy = json.dumps(added_head_balance_spec.document, indent=2, sort_keys=True) + "\n"
        if balance_copy_path.exists() and balance_copy_path.read_text(encoding="utf-8") != balance_copy:
            raise ValueError(f"existing bound added-head balance differs: {balance_copy_path}")
        balance_copy_path.write_text(balance_copy, encoding="utf-8")
    (Path(args.out) / "training_language_mix.json").write_text(
        json.dumps(language_mix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    use_boundary_loss = args.annotated_boundary_loss > 0 or args.annotated_boundary_telemetry
    use_retention_loss = args.partial_boundary_retention_loss > 0
    use_partial_o_loss = args.partial_o_loss > 0
    use_partial_entity_pu = args.partial_entity_pu_loss > 0
    use_partial_expected_entity_ratio = args.partial_expected_entity_ratio_loss > 0
    use_partial_parent_presence_kl = args.partial_parent_presence_kl > 0
    use_complete_presence_loss = args.complete_presence_loss > 0
    use_complete_family_loss = args.complete_family_loss > 0
    use_complete_boundary_loss = args.complete_boundary_loss > 0
    use_reference_primary_positive = args.reference_primary_positive_loss_weight > 0
    use_predicate_loss = args.predicate_loss_weight > 0
    use_reference_type_residual = args.reference_type_residual_loss_weight > 0
    use_subclass_loss = args.subclass_loss_weight > 0
    use_negative_loss = args.partial_negative_loss > 0
    partial_type_only_types = tuple(sorted(set(args.partial_type_only)))
    unknown_type_only = [
        primary_type
        for primary_type in partial_type_only_types
        if not any(label.endswith(f"-{primary_type}") for label in label2id)
    ]
    if unknown_type_only:
        ap.error(f"--partial-type-only names types outside the label space: {unknown_type_only}")
    type_only_label_ids = [
        [index for label, index in sorted(label2id.items()) if label.endswith(f"-{primary_type}")]
        for primary_type in partial_type_only_types
    ]
    family_label_groups = None
    family_target_by_own_label = None
    family_target_by_secondary_label = None
    if use_complete_family_loss:
        family_label_groups, family_target_by_own_label, family_target_by_secondary_label = (
            build_family_presence_projection(correctness_map)
        )
        print(
            f"FAMILY-PRESENCE groups={len(family_label_groups)} "
            f"own-targets={len(family_target_by_own_label)}/{len(correctness_map.old_labels)} "
            f"secondary-targets={len(family_target_by_secondary_label)}",
            flush=True,
        )
    use_partial_objective = use_boundary_loss or use_retention_loss or use_partial_o_loss or use_negative_loss
    use_union_objective = union_groups is not None
    use_dual_head = correctness_map is not None
    o_weight_table = None
    if args.o_weight_config:
        from pii_o_weight import OWeightTable

        o_weight_table = OWeightTable.load(args.o_weight_config)
        observed = sorted({(row.get("src") or row.get("source") or "?") for row in train_rows})
        print(
            f"O-WEIGHT config={args.o_weight_config} scaling={o_weight_table.scaling_factor:g} "
            f"default={o_weight_table.default:g}",
            flush=True,
        )
        for source in observed:
            detail = o_weight_table.explain(source)
            print(
                f"O-WEIGHT   {source[:44]:46s} -> {detail['effective_weight']:.3f} "
                f"(matched {detail['matched']})",
                flush=True,
            )

    status_accepts = None
    if tag_status_types:
        if correctness_map is None:
            ap.error("--tag-status-slots needs --dual-head-map to resolve old-ontology spans")
        successor_types = {label.split("-", 1)[1] for label in correctness_map.new_labels if label != "O"}
        if not set(tag_status_types) <= successor_types:
            ap.error("tag-status types are not all successor-head types")
        # Old type -> every successor type (or O) its BIOES labels may become.
        status_accepts = {}
        for old_id, old_label in enumerate(correctness_map.old_labels):
            if old_label == "O":
                continue
            allowed = torch.nonzero(correctness_map.old_to_new[old_id]).flatten().tolist()
            status_accepts.setdefault(old_label.split("-", 1)[1], set()).update(
                "O"
                if correctness_map.new_labels[j] == "O"
                else correctness_map.new_labels[j].split("-", 1)[1]
                for j in allowed
            )
        status_accepts = {name: frozenset(types) for name, types in status_accepts.items()}
    tag_status_dataset_args = {
        "status_types": tag_status_types,
        "status_accepts": status_accepts,
        "prompt_slots": prompt_width,
        "prompt_languages": prompt_languages,
    }

    train_entries, entry_weights, entry_pool_keys = train_rows, sampling_weights, sampling_pool_keys
    dataset_context_configurations = args.context_configurations
    if args.context_configurations is not None and sampling_weights is not None:
        if surface_realizer is not None:
            ap.error("--context-configurations with sample-time surface realization is not supported")
        train_entries, entry_weights, entry_pool_keys = expand_context_variants(
            train_rows,
            sampling_weights,
            sampling_pool_keys,
            args.context_field,
            args.context_configurations,
            args.context_configuration_weights,
        )
        dataset_context_configurations = None
        print(
            f"TRAIN-CONTEXT-VARIANTS: rows={len(train_rows)} entries={len(train_entries)} "
            f"configurations={args.context_configurations}",
            flush=True,
        )

    train = SpanDataset(
        train_entries,
        tok,
        label2id,
        args.max_len,
        include_boundary_labels=use_boundary_loss,
        include_consistency_mask=(
            use_retention_loss
            or use_partial_o_loss
            or use_partial_entity_pu
            or use_partial_expected_entity_ratio
            or use_partial_parent_presence_kl
        ),
        include_complete_presence_labels=use_complete_presence_loss,
        include_complete_family_labels=use_complete_family_loss,
        family_target_by_own_label=family_target_by_own_label,
        family_target_by_secondary_label=family_target_by_secondary_label,
        include_complete_boundary_labels=use_complete_boundary_loss,
        partial_entity_pu_assignments=(entity_pu_prior_assignments if use_partial_entity_pu else None),
        partial_entity_ratio_assignments=(
            entity_ratio_prior_assignments if use_partial_expected_entity_ratio else None
        ),
        partial_negative_source_groups=partial_negative_source_groups if use_negative_loss else None,
        union_groups=union_groups,
        secondary_label2id=(
            None
            if correctness_map is None
            else {label: index for index, label in enumerate(correctness_map.new_labels)}
        ),
        sampling_weights=entry_weights,
        sampling_pool_keys=entry_pool_keys,
        native_label_space=NEW_LABEL_SPACE if args.native_new_label_space else None,
        mapped_outside_by_row=mapped_single_head,
        mlm_batch_probabilities=mlm_batch_probabilities,
        joint_objectives=use_mlm_objective,
        surface_realizer=surface_realizer,
        character_projection=character_projection,
        character_projection_view=(
            model.config.pii_character_projection_view if use_continuous_character else "pair"
        ),
        o_weight_table=o_weight_table,
        predicate_spec=predicate_spec,
        predicate_condition_types=predicate_condition_types,
        predicate_objective_mask_channels=predicate_objective_mask_channels,
        subclass_spec=subclass_spec,
        legacy_outside_unknown_primary_types=(
            correctness_map.legacy_outside_unknown_primary_types if correctness_map is not None else ()
        ),
        primary_type_learning_weights=(
            added_head_balance_spec.reference_positive_weights
            if added_head_balance_spec is not None
            else None
        ),
        partial_primary_objective_weight=args.partial_primary_objective_weight,
        partial_type_only_types=partial_type_only_types,
        annotation_conventions=annotation_conventions,
        reference_type_residual_types=reference_type_residual_types,
        context_field=args.context_field,
        context_side=args.context_side,
        document_start_marker=args.document_start_marker,
        context_configurations=dataset_context_configurations,
        context_configuration_weights=(
            args.context_configuration_weights if dataset_context_configurations is not None else None
        ),
        soft_registers=args.soft_registers,
        source_classes=source_classes,
        source_dropout=args.soft_register_source_dropout,
        languages=register_languages,
        language_dropout=args.soft_register_language_dropout,
        bias_languages=bias_languages,
        domain_posteriors=domain_posteriors,
        domain_scopes=domain_scopes,
        domain_dropout=args.soft_register_domain_dropout,
        **tag_status_dataset_args,
    )
    # Held separately because a native-head run rebinds `train` to a wrapper, and
    # the collator choice below has to ask the span dataset what fields it emits.
    train_span_dataset = train
    if args.exclude_unsupervised_items and train.sampling_weights is None:
        ap.error("--exclude-unsupervised-items requires weighted sampling")
    native_datasets = {}
    if native_head_specs and not args.evaluate_only:
        for spec in native_head_specs:
            native_rows = native_head_rows[spec["name"]]
            validate_native_tokenization(native_rows, tok, args.max_len)
            native_labels = new_space_labels(spec["types"])
            native_datasets[spec["name"]] = SpanDataset(
                native_rows, tok, {label: index for index, label in enumerate(native_labels)}, args.max_len
            )
        train = NativeHeadDataset(train)
        native_sampler_recipe = {
            "schema": "pii-native-epoch-sampler/v1",
            "seed": sampling_seed,
            "epoch_examples": args.sampling_epoch_windows or len(train),
            "batch_size": args.batch,
            "gradient_accumulation_steps": args.grad_accum,
            "length_window_steps": args.sampling_length_window_steps,
            "max_length": args.max_len,
            "primary_rows_sha256": semantic_sha256(train.rows),
            "sampling_weights_sha256": semantic_sha256(train.sampling_weights),
            "surface_realization_sha256": semantic_sha256(surface_realization_receipt),
        }
        # Recipes recorded before batch formation existed omit it: sorted-window.
        if args.batch_formation != "sorted-window":
            native_sampler_recipe["batch_formation"] = args.batch_formation
            native_sampler_recipe["padding_budget"] = args.batch_padding_budget
        if args.draw_policy != "systematic":
            native_sampler_recipe["draw_policy"] = args.draw_policy
        if (
            resume_checkpoint
            and getattr(model.config, "pii_native_head_sampler", None) != native_sampler_recipe
        ):
            ap.error("native-head resume sampler or primary data differs from its checkpoint")
        model.config.pii_native_head_sampler = native_sampler_recipe
    val = SpanDataset(
        val_rows,
        tok,
        label2id,
        args.max_len,
        include_boundary_labels=use_boundary_loss,
        include_consistency_mask=(
            use_retention_loss
            or use_partial_o_loss
            or use_partial_entity_pu
            or use_partial_expected_entity_ratio
            or use_partial_parent_presence_kl
        ),
        include_complete_presence_labels=use_complete_presence_loss,
        include_complete_family_labels=use_complete_family_loss,
        family_target_by_own_label=family_target_by_own_label,
        family_target_by_secondary_label=family_target_by_secondary_label,
        include_complete_boundary_labels=use_complete_boundary_loss,
        partial_entity_pu_assignments=(entity_pu_prior_assignments if use_partial_entity_pu else None),
        partial_entity_ratio_assignments=None,
        partial_negative_source_groups=partial_negative_source_groups if use_negative_loss else None,
        union_groups=union_groups,
        native_label_space=NEW_LABEL_SPACE if args.native_new_label_space else None,
        mapped_outside_by_row=mapped_single_head,
        secondary_label2id=(
            None
            if correctness_map is None
            else {label: index for index, label in enumerate(correctness_map.new_labels)}
        ),
        character_projection=character_projection,
        character_projection_view=(
            model.config.pii_character_projection_view if use_continuous_character else "pair"
        ),
        o_weight_table=o_weight_table,
        predicate_spec=predicate_spec,
        predicate_condition_types=predicate_condition_types,
        predicate_objective_mask_channels=predicate_objective_mask_channels,
        subclass_spec=subclass_spec,
        legacy_outside_unknown_primary_types=(
            correctness_map.legacy_outside_unknown_primary_types if correctness_map is not None else ()
        ),
        reference_type_residual_types=reference_type_residual_types,
        context_field=args.context_field,
        context_side=args.context_side,
        document_start_marker=args.document_start_marker,
        soft_registers=args.soft_registers,
        source_classes=source_classes,
        source_condition="unknown",
        languages=register_languages,
        bias_languages=bias_languages,
        domain_posteriors=domain_posteriors,
        domain_scopes=domain_scopes,
        **tag_status_dataset_args,
    )
    eff = args.batch * args.grad_accum
    head_spec = (
        f"head={args.head_kind} layers={encoder_layers} "
        f"training-parameter-precision={training_parameter_precision} "
        f"stock-classifier-kernel={args.stock_classifier_kernel} "
        f"token-offsets={list(resolved_token_offsets)} rank={head_rank} "
        f"encoder-frozen={args.freeze_encoder} "
        f"trainable-top-encoder-layers={args.trainable_top_encoder_layers or 'all'} "
        f"annotated-boundary-loss={args.annotated_boundary_loss:g} "
        f"boundary-telemetry={args.annotated_boundary_telemetry} "
        f"partial-boundary-retention-loss={args.partial_boundary_retention_loss:g} "
        f"retention-temperature={args.partial_boundary_retention_temperature:g} "
        f"partial-o-loss={args.partial_o_loss:g} "
        f"partial-entity-pu-loss={args.partial_entity_pu_loss:g} "
        f"partial-entity-pu-positive-margin={args.partial_entity_pu_positive_margin} "
        f"partial-expected-entity-ratio-loss={args.partial_expected_entity_ratio_loss:g} "
        f"partial-expected-entity-ratio-lower-width="
        f"{args.partial_expected_entity_ratio_lower_width:g} "
        f"partial-parent-presence-kl={args.partial_parent_presence_kl:g} "
        f"partial-negative-loss={args.partial_negative_loss:g} "
        f"partial-negative-sources={partial_negative_sources} "
        f"fine-label-loss-weight={args.fine_label_loss_weight:g} "
        f"union-members={args.union_members or 'none'} "
        f"union-v1-fine-weight-schedule="
        f"{args.union_v1_fine_weight_schedule if use_union_objective else 'none'} "
        f"coarse-agreement-reduction={args.coarse_agreement_reduction} "
        f"o-token-loss-weight={args.o_token_loss_weight:g} "
        f"entity-dice-loss={args.entity_dice_loss:g} "
        f"complete-presence-loss={args.complete_presence_loss:g} "
        f"complete-family-loss={args.complete_family_loss:g} "
        f"complete-boundary-loss={args.complete_boundary_loss:g} "
        f"reference-primary-positive-loss-weight={args.reference_primary_positive_loss_weight:g} "
        f"partial-primary-objective-weight={args.partial_primary_objective_weight:g} "
        f"logical-step-objective-normalization={args.logical_step_objective_normalization} "
        f"predicate-loss-weight={args.predicate_loss_weight:g} "
        f"predicate-channels={list(predicate_spec.channels) if predicate_spec is not None else []} "
        f"predicate-objective-mask-channels={list(predicate_objective_mask_channels)} "
        f"predicate-conditioning={args.predicate_conditioning} "
        f"reference-type-residual-loss-weight={args.reference_type_residual_loss_weight:g} "
        f"reference-type-residual-types={list(reference_type_residual_types)} "
        f"subclass-loss-weight={args.subclass_loss_weight:g} "
        f"subclass-spec-sha256={subclass_spec.sha256[:12] if subclass_spec is not None else 'none'} "
        f"rdrop-alpha={args.rdrop_alpha:g} "
        f"mlm-replay-prob={args.mlm_replay_prob:g} "
        f"mlm-physical-batch-prob={args.mlm_physical_batch_prob:g} "
        f"mlm-pools={args.mlm_pool_probabilities} "
        f"mlm-loss-weight={args.mlm_loss_weight:g} "
        f"mlm-mask-prob={args.mlm_probability:g} "
        f"mlm-head={args.mlm_head_model or args.model if use_mlm_objective else 'none'} "
        f"head-lr={args.lr:g} encoder-lr={args.encoder_lr if args.encoder_lr is not None else args.lr:g} "
        f"encoder-prior={args.encoder_prior_checkpoint or 'none'} "
        f"encoder-prior-weight={args.encoder_prior_weight:g} "
        f"encoder-prior-start-step={args.encoder_prior_start_step} "
        f"inherited-output-prior={args.inherited_output_prior_checkpoint or 'none'} "
        f"inherited-output-prior-rows={inherited_output_prior_rows} "
        f"inherited-output-prior-weight={args.inherited_output_prior_weight:g} "
        f"inherited-output-prior-start-step={args.inherited_output_prior_start_step} "
        f"continuous-character-pretrain={args.continuous_character_pretrain or 'none'} "
        f"continuous-character-initialization={args.continuous_character_initialization} "
        f"continuous-character-pooling={args.continuous_character_pooling} "
        f"continuous-character-lr={args.continuous_character_lr or args.lr:g} "
        f"continuous-character-output-lr={args.continuous_character_output_lr or args.continuous_character_lr or args.lr:g} "
        f"continuous-character-logits={args.continuous_character_logit_initialization} "
        f"continuous-character-logit-scale={args.continuous_character_logit_scale:g} "
        f"continuous-character-aux={args.continuous_character_auxiliary_loss_weight:g}/"
        f"{args.continuous_character_auxiliary_loss_fade_steps}"
    )
    if frozen_inherited_output_rows:
        active_auxiliary_heads = "+".join(
            name
            for enabled, name in (
                (use_predicate_loss, "predicate"),
                (use_reference_type_residual, "reference-residual"),
                (use_subclass_loss, "subclass"),
            )
            if enabled
        )
        training_scope = "appended-primary" + (
            f"+{active_auxiliary_heads}-only" if active_auxiliary_heads else "-only"
        )
    elif args.freeze_encoder:
        training_scope = "head-only"
    elif args.trainable_top_encoder_layers is not None:
        training_scope = f"top-{args.trainable_top_encoder_layers}-encoder+head"
    else:
        training_scope = "full-FT"
    tag = os.path.basename(args.out.rstrip("/"))
    output_label_count = int(model.config.num_labels)
    print(
        f"TRAIN: {tag}: model={args.model} decoder={args.decoder} {head_spec} "
        f"training-labels={len(names)} output-labels={output_label_count} "
        f"train={len(train)} windows val={len(val)} lr={args.lr} "
        f"eff-batch={eff} ({args.batch}x{args.grad_accum}) epochs={args.epochs} "
        f"warm-start-copied-label-rows={shared_warm_start_labels} "
        f"output-head-reinitialized={args.reset_output_head} "
        f"warm-start-label-schema={args.warm_start_label_schema or 'literal'} "
        f"warm-start-label-cut={args.warm_start_label_cut or 'none'} "
        f"freeze-summary={freeze_summary or {}} "
        f"replay-windows={replay_windows} "
        f"mlm-replay-windows={len(mlm_rows)} "
        f"weighted-sampling={sampling_weights is not None} "
        f"memory-metrics={args.memory_metrics} "
        f"seeds={training_seeds} "
        f"language-shares={effective_language_shares} "
        f"file-pool-shares={expected_batch_pool_shares}",
        flush=True,
    )
    log_format.headline(
        f"{tag}: {training_scope} {args.model} decoder={args.decoder} "
        f"{head_spec} lr={args.lr} {args.sched} eff-batch={eff}, "
        f"{len(train)} windows, {len(names)} training labels, "
        f"{output_label_count} output labels"
    )

    selection_metric = "eval_loss" if args.selection_metric == "loss" else "eval_span_f1"
    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.sched,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        # Every real run is on an accelerator and wants bf16. Asking for it
        # without one is a hard error in TrainingArguments, which blocks the
        # cheap CPU smoke that data-path changes are supposed to be checked
        # with, so follow the device rather than assert it.
        bf16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model=selection_metric,
        greater_is_better=args.selection_metric == "span-f1",
        save_total_limit=args.save_limit,
        dataloader_num_workers=4,
        seed=model_seed,
        data_seed=sampling_seed,
        report_to=[],
        skip_memory_metrics=not args.memory_metrics,
        remove_unused_columns=not (
            use_partial_objective
            or annotation_conventions is not None
            or native_head_specs
            or use_mlm_objective
            or use_continuous_character
            or use_union_objective
            or use_dual_head
            or use_predicate_loss
            or use_reference_type_residual
            or use_subclass_loss
            or train_span_dataset.include_primary_objective_weights
            or train_span_dataset.include_native_outside_allowed
        ),
    )
    if use_dual_head:
        # Span metrics score the new-ontology output, so the labels handed to
        # them are new-space targets whether that output is a second head or the
        # sole classifier left after retirement.
        targs.label_names = ["secondary_labels"]
    targs.sampling_epoch_examples = args.sampling_epoch_windows or len(train)
    targs.sampling_length_window_steps = args.sampling_length_window_steps
    targs.sampling_batch_formation = args.batch_formation
    targs.sampling_padding_budget = args.batch_padding_budget
    targs.sampling_draw_policy = args.draw_policy
    targs.sampling_length_bucket_width = args.sampling_length_bucket_width
    targs.pii_sampling_seed = sampling_seed
    targs.pii_encoder_learning_rate = args.encoder_lr
    targs.pii_register_learning_rate = args.soft_register_lr
    targs.pii_head_residual_mlp_learning_rate = args.head_residual_mlp_lr
    targs.pii_prompt_learning_rate = args.tag_status_lr if args.tag_status_slots else None
    targs.pii_character_learning_rate = (
        args.continuous_character_lr
        if args.continuous_character_lr is not None
        else args.lr
        if use_continuous_character
        else None
    )
    targs.pii_character_output_learning_rate = (
        args.continuous_character_output_lr
        if args.continuous_character_output_lr is not None
        else targs.pii_character_learning_rate
    )
    model.config.pii_character_learning_rate = targs.pii_character_learning_rate
    model.config.pii_character_output_learning_rate = targs.pii_character_output_learning_rate
    if sampling_weights is not None or native_head_specs:
        base_trainer_class = (
            WeightedPartialSupervisionTrainer if use_partial_objective else WeightedSamplingTrainer
        )
    else:
        base_trainer_class = PartialSupervisionTrainer if use_partial_objective else Trainer
    trainer_class = base_trainer_class
    native_objective = False
    if not use_dual_head:
        if args.decoder == "linear" and not use_union_objective:
            try:
                native_primary_objective = resolve_native_primary_objective(
                    {
                        "boundary": args.bioes_margin_boundary,
                        "type": args.bioes_margin_type,
                        "illegal_scale": args.bioes_margin_illegal_scale,
                        "same_bucket_scale": args.bioes_margin_same_bucket_scale,
                        "bucket_map_sha256": (
                            hashlib.sha256(Path(args.bioes_bucket_map).read_bytes()).hexdigest()
                            if args.bioes_bucket_map
                            else None
                        ),
                    },
                    resume_config=resume_config,
                )
            except ValueError as error:
                ap.error(str(error))
            native_objective = native_primary_objective["version"] != "legacy"
            if annotation_conventions is not None and not native_objective:
                ap.error("internal segmentation requires a new native-objective initialization stage")
            if not args.evaluate_only:
                model.config.pii_native_primary_objective = native_primary_objective
            if not native_objective and (args.bioes_margin_boundary or args.bioes_margin_type):
                print(
                    "TRAIN: historical exact resume preserves inactive native BIOES margins; "
                    "use --init-from-checkpoint for the corrected objective",
                    flush=True,
                )
        elif args.bioes_margin_boundary or args.bioes_margin_type:
            ap.error("native BIOES margins require a linear decoder without union supervision")
        trainer_class = o_token_weighted_trainer_class(
            trainer_class,
            o_weight_config=args.o_weight_config,
            o_token_loss_weight=args.o_token_loss_weight,
            native_objective=native_objective,
        )
    coarse_agreement = []
    for weight, cut, boundary_free in (
        (args.coarse_agreement_weight_20, "redaction_20_v1", False),
        (args.coarse_agreement_weight_9, "redaction_9_v1", False),
        (args.coarse_agreement_weight_presence_20, "redaction_20_v1", True),
    ):
        if weight:
            index_groups, remap = build_coarse_cut_groups(names, cut, boundary_free)
            coarse_agreement.append((weight, index_groups, remap))
    if coarse_agreement:
        trainer_class = type(
            "CoarseAgreement" + trainer_class.__name__,
            (CoarseAgreementMixin, trainer_class),
            {},
        )
    if partial_type_only_types:
        trainer_class = type(
            "TypeOnly" + trainer_class.__name__,
            (TypeOnlySupervisionMixin, trainer_class),
            {
                "type_only_label_ids": type_only_label_ids,
                "type_only_loss_weight": args.partial_type_only_weight,
            },
        )
        print(
            "TRAIN: boundary-free partial supervision active: "
            f"types={list(partial_type_only_types)} weight={args.partial_type_only_weight}",
            flush=True,
        )
    if args.entity_dice_loss:
        trainer_class = type(
            "EntityDice" + trainer_class.__name__,
            (EntityDiceLossMixin, trainer_class),
            {},
        )
    if args.continuous_character_auxiliary_loss_weight:
        trainer_class = type(
            "CharacterAuxiliary" + trainer_class.__name__,
            (CharacterAuxiliaryLossMixin, trainer_class),
            {},
        )
    if args.rdrop_alpha:
        trainer_class = type(
            "RDrop" + trainer_class.__name__,
            (RDropLossMixin, trainer_class),
            {},
        )
    if use_union_objective:
        # Wrapped before the MLM mixin so it sits inside it: that mixin unpacks the
        # physical batch and calls down with the tagging batch, which is where the
        # per-token union targets are.
        trainer_class = type(
            "UnionMarginal" + trainer_class.__name__,
            (UnionMarginalLossMixin, trainer_class),
            {},
        )
    if use_dual_head:
        trainer_class = type(
            "DualHead" + trainer_class.__name__,
            (DualHeadLossMixin, trainer_class),
            {},
        )
    if use_predicate_loss and use_subclass_loss:
        trainer_class = type(
            "PredicateSubclass" + trainer_class.__name__,
            (PredicateSubclassLossMixin, trainer_class),
            {},
        )
    elif use_predicate_loss:
        trainer_class = type(
            "Predicate" + trainer_class.__name__,
            (PredicateLossMixin, trainer_class),
            {},
        )
    elif use_subclass_loss:
        trainer_class = type(
            "Subclass" + trainer_class.__name__,
            (SubclassLossMixin, trainer_class),
            {},
        )
    if dual_head_lr_restart_step:
        trainer_class = type(
            "TransitionRestart" + trainer_class.__name__,
            (TransitionRestartSchedulerMixin, trainer_class),
            {},
        )
    if use_mlm_objective:
        trainer_class = type(
            "MlmReplay" + trainer_class.__name__,
            (MlmReplayTrainerMixin, trainer_class),
            {},
        )
    if args.encoder_lr is not None or use_continuous_character:
        trainer_class = type(
            "DifferentialLearningRate" + trainer_class.__name__,
            (DifferentialLearningRateTrainerMixin, trainer_class),
            {},
        )
    if encoder_prior_anchors is not None:
        trainer_class = type(
            "EncoderParameterPrior" + trainer_class.__name__,
            (EncoderParameterPriorTrainerMixin, trainer_class),
            {},
        )
    if inherited_output_prior_anchors is not None:
        trainer_class = type(
            "InheritedOutputParameterPrior" + trainer_class.__name__,
            (InheritedOutputParameterPriorTrainerMixin, trainer_class),
            {},
        )
    if native_head_specs:
        trainer_class = type("NativeHeads" + trainer_class.__name__, (NativeHeadLossMixin, trainer_class), {})
    trainer_kwargs = (
        {
            "boundary_label_names": names,
            "boundary_loss_weight": args.annotated_boundary_loss,
            "retention_teacher": retention_teacher,
            "retention_loss_weight": args.partial_boundary_retention_loss,
            "retention_temperature": args.partial_boundary_retention_temperature,
            "partial_o_loss_weight": args.partial_o_loss,
            "o_label_id": label2id["O"],
            "partial_negative_label_groups": partial_negative_label_groups,
            "partial_negative_loss_weight": args.partial_negative_loss,
        }
        if use_partial_objective
        else {}
    )
    if use_mlm_objective:
        trainer_kwargs.update(
            {
                "mlm_head": mlm_head,
                "mlm_loss_weight": args.mlm_loss_weight,
            }
        )
    if encoder_prior_anchors is not None:
        trainer_kwargs.update(
            {
                "encoder_prior_anchors": encoder_prior_anchors,
                "encoder_prior_weight": args.encoder_prior_weight,
                "encoder_prior_start_step": args.encoder_prior_start_step,
            }
        )
    if inherited_output_prior_anchors is not None:
        trainer_kwargs.update(
            {
                "inherited_output_prior_anchors": inherited_output_prior_anchors,
                "inherited_output_prior_rows": inherited_output_prior_rows,
                "inherited_output_prior_weight": args.inherited_output_prior_weight,
                "inherited_output_prior_start_step": args.inherited_output_prior_start_step,
            }
        )
    if native_head_specs:
        trainer_kwargs.update(
            native_head_specs=native_head_specs,
            native_logical_normalization=args.logical_step_objective_normalization,
        )
    # Which collator to use is not a question about which losses are enabled; it
    # is a question about which fields the dataset actually emits, because the
    # stock token-classification collator pads only input_ids and labels and
    # hands anything else straight to tensor construction. Per-span objective
    # weights are emitted by the rows themselves, so a run can enable no partial,
    # union, dual-head, predicate or subclass loss at all and still need the
    # padding collator. Deriving the choice from the dataset keeps the two in
    # step; deriving it from the loss flags is what let a run reach its first
    # batch and die on an unpadded float sequence.
    emits_primary_objective_weights = bool(
        getattr(train_span_dataset, "include_primary_objective_weights", False)
    )
    tag_collator = (
        PartialLabelDataCollator(tok)
        if (
            use_partial_objective
            or annotation_conventions is not None
            or use_union_objective
            or use_dual_head
            or use_predicate_loss
            or use_subclass_loss
            or emits_primary_objective_weights
        )
        else DataCollatorForTokenClassification(tok)
    )
    if use_continuous_character:
        tag_collator = ContinuousCharacterDataCollator(tok, tag_collator)
    data_collator = (
        JointMlmDataCollator(
            tok,
            tag_collator=tag_collator,
            mlm_probability=args.mlm_probability,
            replay_seed=replay_seed,
        )
        if use_mlm_objective
        else tag_collator
    )
    if native_head_specs:
        data_collator = NativeHeadCollator(
            data_collator, tok, native_head_specs, native_datasets, seed=sampling_seed
        )
    span_metric_report = (
        build_dual_head_span_metric_report(correctness_map, val_rows)
        if correctness_map is not None
        else partial(
            union_span_metrics,
            id2label=model.config.id2label,
            projected_id2label=union_groups.projected_id2label,
            # Evaluation order is the validation dataset's own order, so one flag
            # per row lines up with one evaluated row.
            fine_rows=[row.get("label_space", FINE_LABEL_SPACE) == FINE_LABEL_SPACE for row in val_rows],
        )
        if union_groups is not None
        else partial(span_metrics, id2label=model.config.id2label)
    )
    if correctness_map is not None and span_metric_report is None:
        print(
            "TRAIN: validation contains no hard new-ontology rows; using mapped eval_loss "
            "without exact new-ontology span metrics",
            flush=True,
        )
    has_span_metrics = args.decoder != "crf" and span_metric_report is not None
    if args.selection_metric == "span-f1" and not has_span_metrics:
        ap.error(
            "--selection-metric span-f1 requires hard validation labels in the product head's "
            "ontology and a decoder that emits span metrics"
        )
    sampler_epoch_steps = math.ceil((args.sampling_epoch_windows or len(train)) / eff)
    orderly_checkpoint = OrderlyCheckpointCallback()
    checkpoint_mirror = CheckpointMirrorCallback(args)
    callbacks = []
    if not args.evaluate_only:
        callbacks.extend(
            [
                EarlyStoppingCallback(
                    early_stopping_patience=args.patience,
                    early_stopping_threshold=args.early_stopping_threshold,
                ),
                RollingTrainLossCallback(sampler_epoch_steps, args.train_loss_window_epochs),
                HeadlineCallback(tag, head_spec, eff),
                LearningCurveCallback(
                    args.sampling_epoch_windows or len(train),
                    resume=resume_checkpoint is not None,
                ),
                FinalSelectionCheckpointCallback(),
                orderly_checkpoint,
                checkpoint_mirror,
                RunMetricsCallback("train"),
            ]
        )
        if frozen_inherited_output_rows:
            callbacks.append(FrozenClassifierPrefixCallback(model, frozen_inherited_output_rows))
        if args.extra_save_step:
            callbacks.append(ExtraSaveStepsCallback(args.extra_save_step))
        if args.stop_after_step:
            callbacks.append(StopAfterStepCallback(args.stop_after_step))

    def bind_task_configuration(
        target,
        *,
        dual_head_retired,
        include_transition_restart,
    ):
        # Transformers divides each physical batch's loss by the accumulation
        # steps only when it believes the model does not normalize its own loss.
        # The model's forward takes **kwargs, which made it believe the model
        # does, so the recipe summed its batch means before --accumulation-loss.
        # Logical-step normalization divides by the step's supervised mass itself.
        target.model_accepts_loss_kwargs = (
            args.logical_step_objective_normalization or args.accumulation_loss == "sum"
        )
        if coarse_agreement:
            target.coarse_agreement = coarse_agreement
            target.fine_label_loss_weight = args.fine_label_loss_weight
            target.normalize_coarse_groups = args.coarse_agreement_reduction == "mean"
        if correctness_map is not None:
            target.correctness_map = correctness_map
            target.dual_head_schedule = dual_head_schedule
            target.dual_head_horizon = args.max_steps
            target.dual_head_eval_weight = args.dual_head_eval_weight
            target.dual_head_retired = bool(dual_head_retired)
            if include_transition_restart and dual_head_lr_restart_step:
                target.lr_restart_step = dual_head_lr_restart_step
            if dual_head_retirement_step is not None and not target.dual_head_retired:
                target.add_callback(DualHeadRetirementCallback(target, dual_head_retirement_step))
        if validation_draw_changed_on_resume:
            target.add_callback(NewValidationDrawCallback())
        if union_groups is not None:
            target.union_groups = union_groups
            target.union_fine_weight_schedule = union_fine_weight_schedule
            target.union_fine_weight_horizon = args.max_steps
        if native_objective or args.o_token_loss_weight != 1.0 or args.o_weight_config is not None:
            target.o_label_id = label2id["O"]
            target.o_token_loss_weight = args.o_token_loss_weight
        bucket_of = load_bucket_map(args.bioes_bucket_map) if args.bioes_bucket_map else None
        if (correctness_map is not None or native_objective) and (
            args.bioes_margin_boundary or args.bioes_margin_type
        ):
            margin_labels = list(correctness_map.new_labels) if correctness_map is not None else list(names)
            if bucket_of is not None:
                target.bioes_structure_cost_matrix = bucketed_structure_cost_matrix(
                    margin_labels,
                    bucket_of,
                    boundary_cost=args.bioes_margin_boundary,
                    type_cost=args.bioes_margin_type,
                    same_bucket_scale=args.bioes_margin_same_bucket_scale,
                )
            else:
                target.bioes_structure_cost_matrix = bioes_structure_cost_matrix(
                    margin_labels,
                    boundary_cost=args.bioes_margin_boundary,
                    type_cost=args.bioes_margin_type,
                )
            if correctness_map is not None:
                # Only a mapped run has old-head rows to project the cost onto.
                target.bioes_mapped_cost_rows = mapped_structure_cost_rows(
                    target.bioes_structure_cost_matrix, correctness_map.old_to_new
                )
            target.bioes_transition_legality = bioes_transition_legality(margin_labels)
            target.bioes_illegal_flip_scale = args.bioes_margin_illegal_scale
            print(
                "TRAIN: BIOES structure-cost margin active: "
                f"boundary={args.bioes_margin_boundary} type={args.bioes_margin_type} "
                f"illegal-flip-scale={args.bioes_margin_illegal_scale} "
                f"labels={len(margin_labels)} "
                f"mapped-rows={'yes' if correctness_map is not None else 'no'}",
                flush=True,
            )
        if args.bioes_risk_weight and correctness_map is not None:
            target.bioes_incompatible = (
                bucketed_incompatibility_matrix(list(correctness_map.new_labels), bucket_of)
                if bucket_of is not None
                else coarse_incompatibility_matrix(list(correctness_map.new_labels), args.bioes_risk_cut)
            )
            target.bioes_continuation = continuation_label_mask(list(correctness_map.new_labels))
            target.bioes_risk_weight = args.bioes_risk_weight
            target.bioes_risk_threshold = args.bioes_risk_threshold
            target.bioes_risk_scale = args.bioes_risk_scale
        if args.entity_dice_loss:
            target.entity_dice_weight = args.entity_dice_loss
        if correctness_map is not None:
            target.partial_o_loss_weight = args.partial_o_loss
            target.partial_entity_pu_loss_weight = args.partial_entity_pu_loss
            target.partial_entity_pu_positive_margin = args.partial_entity_pu_positive_margin
            target.partial_expected_entity_ratio_loss_weight = args.partial_expected_entity_ratio_loss
            target.partial_expected_entity_ratio_lower_width = args.partial_expected_entity_ratio_lower_width
            target.partial_parent_presence_kl_weight = args.partial_parent_presence_kl
            if partial_parent_presence_teacher is not None:
                partial_parent_presence_teacher.to(target.args.device)
                target.partial_parent_presence_teacher = partial_parent_presence_teacher
            target.complete_presence_loss_weight = args.complete_presence_loss
            target.complete_family_loss_weight = args.complete_family_loss
            target.family_label_groups = family_label_groups
            target.complete_boundary_loss_weight = args.complete_boundary_loss
        if use_reference_primary_positive:
            target.reference_primary_positive_loss_weight = args.reference_primary_positive_loss_weight
        if args.logical_step_objective_normalization:
            target.logical_step_objective_normalization = True
        if args.rdrop_alpha:
            target.rdrop_alpha = args.rdrop_alpha
        if use_predicate_loss:
            target.predicate_loss_weight = args.predicate_loss_weight
            target.predicate_exposure_channels = predicate_spec.channels
            target.predicate_exposure_condition_types = predicate_condition_types
            target.primary_exposure_labels = tuple(
                model.config.id2label[index] for index in range(model.config.num_labels)
            )
            reference_types = (
                ()
                if correctness_map is None
                else tuple(sorted(correctness_map.legacy_outside_unknown_primary_types))
            )
            reference_label2id = (
                {}
                if correctness_map is None
                else {label: index for index, label in enumerate(correctness_map.new_labels)}
            )
            target.predicate_exposure_reference_label_ids = {
                primary_type: tuple(
                    reference_label2id[f"{boundary}-{primary_type}"] for boundary in ("B", "I", "E", "S")
                )
                for primary_type in reference_types
            }
            target.predicate_positive_weights = (
                None
                if added_head_balance_spec is None
                else torch.tensor(
                    [
                        [
                            added_head_balance_spec.predicate_positive_weights[primary_type][channel]
                            for channel in predicate_spec.channels
                        ]
                        for primary_type in predicate_condition_types
                    ],
                    dtype=torch.float,
                )
            )
        if use_reference_type_residual:
            target.reference_type_residual_loss_weight = args.reference_type_residual_loss_weight
            target.reference_type_residual_types = reference_type_residual_types
            target.reference_type_residual_positive_weights = torch.tensor(
                [
                    added_head_balance_spec.reference_positive_weights[primary_type]
                    for primary_type in reference_type_residual_types
                ],
                dtype=torch.float,
            )
        if use_subclass_loss:
            target.subclass_loss_weight = args.subclass_loss_weight
            target.subclass_blocks = subclass_spec.config_blocks()
            target.subclass_exposure_blocks = tuple(
                {
                    **block,
                    "outcomes": subclass_spec.family_by_name[block["family"]].outcomes,
                }
                for block in target.subclass_blocks
            )
        if args.continuous_character_auxiliary_loss_weight:
            target.character_auxiliary_loss_weight = args.continuous_character_auxiliary_loss_weight
            target.character_auxiliary_loss_fade_steps = args.continuous_character_auxiliary_loss_fade_steps

    if args.head_input_norm != "none" or args.head_residual_mlp != "off":
        attach_calibrated_head_input_norm(model, train, tok, args)
    trainer = trainer_class(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=span_metric_report if has_span_metrics else None,
        preprocess_logits_for_metrics=(argmax_new_head_logits if use_dual_head else argmax_own_head_logits)
        if has_span_metrics
        else None,
        **trainer_kwargs,
    )
    bind_task_configuration(
        trainer,
        dual_head_retired=resumed_retired_dual_head,
        include_transition_restart=True,
    )
    if coarse_agreement:
        print(
            "TRAIN: coarse-agreement terms active: "
            + ", ".join(f"w={w} groups={len(g)}" for w, g, _ in coarse_agreement)
            + f"; fine-label-weight={args.fine_label_loss_weight:g} "
            + f"reduction={args.coarse_agreement_reduction}",
            flush=True,
        )
    if correctness_map is not None:
        dual_rows = Counter(row.get("label_space", FINE_LABEL_SPACE) for row in train_rows)
        print(
            "TRAIN: dual-head objective bound: "
            f"train-windows-by-space={dict(sorted(dual_rows.items()))} "
            f"old-weight-schedule={args.dual_head_old_weight_schedule} "
            f"eval-weight={args.dual_head_eval_weight:g} horizon={args.max_steps} "
            f"resumed-retired={resumed_retired_dual_head} "
            f"retire-old-head-at="
            f"{'never' if dual_head_retirement_step is None else dual_head_retirement_step} "
            f"lr-restart-at={dual_head_lr_restart_step or 'none'}",
            flush=True,
        )
    if union_groups is not None:
        union_rows = Counter(row.get("label_space", FINE_LABEL_SPACE) for row in train_rows)
        print(
            "TRAIN: union-marginal supervision active: "
            f"members={args.union_members} sha256={union_groups.members_sha256[:12]} "
            f"classes={len(union_groups.union_classes)} groups={len(union_groups.names)} "
            f"fine-rows={len(union_groups.fine_labels)} "
            f"outside-members={list(union_groups.outside_members)} "
            f"train-windows-by-space={dict(sorted(union_rows.items()))} "
            f"v1-fine-weight-schedule={args.union_v1_fine_weight_schedule} "
            f"horizon={args.max_steps}",
            flush=True,
        )
    if args.o_token_loss_weight != 1.0:
        print(f"TRAIN: O-token loss weight active: w={args.o_token_loss_weight:g}", flush=True)
    if args.entity_dice_loss:
        print(f"TRAIN: entity-dice loss active: w={args.entity_dice_loss:g}", flush=True)
    if args.complete_presence_loss:
        print(
            f"TRAIN: complete-row entity-vs-O loss active on product head: w={args.complete_presence_loss:g}",
            flush=True,
        )
    if args.complete_boundary_loss:
        print(
            "TRAIN: complete-row entity-boundary loss active on product head: "
            f"w={args.complete_boundary_loss:g} O-tokens=excluded",
            flush=True,
        )
    if args.partial_o_loss and correctness_map is not None:
        print(
            f"TRAIN: positive-only unmarked-token O loss active on product head: w={args.partial_o_loss:g}",
            flush=True,
        )
    if args.partial_entity_pu_loss:
        print(
            "TRAIN: binary entity-presence nnPU active on partial rows: "
            f"w={args.partial_entity_pu_loss:g} "
            f"positive-margin={args.partial_entity_pu_positive_margin} "
            f"prior-sha256={entity_pu_prior_sha256[:12]}",
            flush=True,
        )
    if args.partial_expected_entity_ratio_loss:
        print(
            "TRAIN: expected entity-ratio hinge active on partial-row real tokens: "
            f"w={args.partial_expected_entity_ratio_loss:g} "
            f"lower-width={args.partial_expected_entity_ratio_lower_width:g} "
            f"prior-sha256={entity_ratio_prior_sha256[:12]}",
            flush=True,
        )
    if args.partial_parent_presence_kl:
        print(
            "TRAIN: parent entity-presence trust region active on untagged partial tokens: "
            f"w={args.partial_parent_presence_kl:g} parent={args.init_from_checkpoint}",
            flush=True,
        )
    if use_predicate_loss:
        print(
            "TRAIN: tag-specific predicate loss active: "
            f"w={args.predicate_loss_weight:g} channels={list(predicate_spec.channels)} "
            f"objective-mask-channels={list(predicate_objective_mask_channels)} "
            f"spec-sha256={predicate_spec.sha256[:12]}",
            flush=True,
        )
    if args.logical_step_objective_normalization:
        print(
            "TRAIN: data objectives normalized by supervised mass over each complete "
            f"optimizer step ({args.grad_accum} physical batches), "
            f"max-grad-norm={args.max_grad_norm:g}",
            flush=True,
        )
    if use_reference_primary_positive:
        print(
            "TRAIN: separately normalized primary reference-positive loss active: "
            f"w={args.reference_primary_positive_loss_weight:g} "
            "untagged-partial-tokens=masked",
            flush=True,
        )
    if use_reference_type_residual:
        print(
            "TRAIN: reference-type semantic residual active: "
            f"w={args.reference_type_residual_loss_weight:g} "
            f"types={list(reference_type_residual_types)} initialization=zero",
            flush=True,
        )
    if use_subclass_loss:
        print(
            "TRAIN: carrier-conditioned categorical subclass loss active: "
            f"w={args.subclass_loss_weight:g} rows={subclass_spec.head_rows} "
            f"families={[family.name for family in subclass_spec.families]} "
            f"spec-sha256={subclass_spec.sha256[:12]}",
            flush=True,
        )
    if args.rdrop_alpha:
        print(f"TRAIN: R-Drop active: alpha={args.rdrop_alpha:g}", flush=True)
    if args.continuous_character_auxiliary_loss_weight:
        print(
            "TRAIN: character-only tag loss active: "
            f"weight={args.continuous_character_auxiliary_loss_weight:g} "
            f"fade-steps={args.continuous_character_auxiliary_loss_fade_steps}",
            flush=True,
        )
    print(
        "TRAIN-STOPPING: "
        f"patience={args.patience} threshold={args.early_stopping_threshold:g} "
        f"selection-metric={selection_metric} "
        f"eval-steps={args.eval_steps} sampler-epoch-steps~={sampler_epoch_steps} "
        f"train-loss-window-epochs={args.train_loss_window_epochs:g} "
        f"victory-lap-lr-scale={args.victory_lap_lr_scale:g}",
        flush=True,
    )
    if args.evaluate_only:
        evaluation = trainer.evaluate()
        checkpoint = Path(args.init_from_checkpoint).resolve()
        weights_path = checkpoint / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"--evaluate-only checkpoint has no model.safetensors: {weights_path}")
        with weights_path.open("rb") as source:
            weights_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        receipt_path = output / "evaluation_only.json"
        if receipt_path.exists():
            raise FileExistsError(f"evaluation-only receipt already exists: {receipt_path}")
        receipt = {
            "schema": "pii-token-classifier-evaluation-only",
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "weights_sha256": weights_sha256,
            "validation_selection_receipt": str(validation_receipt_path),
            "validation_selected_sha256": validation_receipt["selected_sha256"],
            "validation_windows": len(val),
            "metrics": evaluation,
            "optimizer_steps": 0,
            "model_exported": False,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "TRAIN-EVALUATE-ONLY: "
            f"checkpoint={checkpoint} validation-windows={len(val)} "
            f"optimizer-steps=0 receipt={receipt_path} metrics={evaluation}",
            flush=True,
        )
        log_format.headline(
            f"{tag}: evaluated {checkpoint.name} on {len(val)} validation windows; optimizer-steps=0"
        )
        return
    if args.exclude_unsupervised_items:
        if not hasattr(trainer, "_batch_objective_masses"):
            ap.error("--exclude-unsupervised-items requires the joint predicate and subclass objectives")
        # Measured through the training collator and the trainer's own
        # per-objective normalizers. An item is excluded only when every
        # objective's mass is zero: a positive-only row with zero primary weight
        # may still carry subclass or predicate supervision.
        item_masses = item_objective_masses(
            train_span_dataset,
            LengthAuditCollator(trainer.data_collator),
            trainer._batch_objective_masses,
        )
        supervised_mass = [sum(item.values()) for item in item_masses]
        original_weights = list(train_span_dataset.sampling_weights)
        train_span_dataset.sampling_weights, excluded = exclude_unsupervised_items(
            original_weights, train_span_dataset.sampling_pool_keys, supervised_mass
        )
        rows = train_span_dataset.rows
        branches = [str(row.get("sampling_branch", "")) for row in rows]
        by_branch = {}
        for branch, weight, item, total in zip(
            branches, original_weights, item_masses, supervised_mass, strict=True
        ):
            entry = by_branch.setdefault(
                branch, {"items": 0, "weight": 0.0, "weighted_mass": {}, "excluded": 0}
            )
            entry["items"] += 1
            entry["weight"] += weight
            entry["excluded"] += int(total <= 0)
            for name, mass in item.items():
                entry["weighted_mass"][name] = entry["weighted_mass"].get(name, 0.0) + weight * mass
        for branch, entry in sorted(by_branch.items()):
            per_draw = " ".join(
                f"{name}={mass / entry['weight']:.2f}"
                for name, mass in sorted(entry["weighted_mass"].items())
                if mass
            )
            print(
                f"TRAIN-SUPERVISION: branch={branch or '-'} items={entry['items']} "
                f"excluded={entry['excluded']} draw_weight={entry['weight']:.6g} mass_per_draw: {per_draw}",
                flush=True,
            )
        print(
            f"TRAIN-SUPERVISION: excluded {len(excluded)} of {len(item_masses)} items with no mass in any objective",
            flush=True,
        )
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(args.out) / "supervision-mass-audit.json.gz", "wt") as handle:
            json.dump(
                {
                    "schema": "pii-supervision-mass-audit/v2",
                    "o_token_loss_weight": args.o_token_loss_weight,
                    "by_branch": by_branch,
                    # Training windows keep provenance but not row ids.
                    "excluded": [
                        {"src": rows[index].get("src"), "text": rows[index]["text"][:200]}
                        for index in excluded
                    ],
                    "branch": branches,
                    "weight": original_weights,
                    "mass": [item["primary"] for item in item_masses],
                    "supervised_mass": supervised_mass,
                    "objective_mass": item_masses,
                    "binned_length": list(train_span_dataset.token_lengths()),
                },
                handle,
            )
        if args.supervision_audit_only:
            print("TRAIN-SUPERVISION: audit only; not training", flush=True)
            return
    lap_only_selection = None
    if args.victory_lap_only_receipt is not None:
        lap_only_selection = load_victory_lap_only_selection(
            args.victory_lap_only_receipt,
            init_checkpoint=args.init_from_checkpoint,
            validation_sha256=validation_receipt["selected_sha256"],
        )
        validate_victory_lap_only_configuration(lap_only_selection, targs)
        selected_checkpoint = str(Path(args.init_from_checkpoint).resolve())
        selected_step = int(lap_only_selection["selected_step"])
        selected_loss = float(lap_only_selection["selected_eval_loss"])
        selected_metric_name = str(lap_only_selection["selected_metric_name"])
        selected_metric_value = float(lap_only_selection["selected_metric_value"])
        print(
            "TRAIN-SELECTION: "
            f"checkpoint={selected_checkpoint} step={selected_step} "
            f"metric={selected_metric_name} best={selected_metric_value:.8g} "
            f"eval-loss={selected_loss:.8g} verified=receipt "
            f"lap-only-receipt={args.victory_lap_only_receipt}",
            flush=True,
        )
    else:
        with orderly_checkpoint.signal_handlers():
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        if use_predicate_loss:
            training_exposure = trainer.predicate_training_exposure_receipt()
            training_exposure.update(
                {
                    "global_step_after_training": int(trainer.state.global_step),
                    "process_index": int(trainer.args.process_index),
                    "world_size": int(trainer.args.world_size),
                    "sampling_config": args.sampling_config or None,
                    "sampling_config_sha256": (
                        hashlib.sha256(Path(args.sampling_config).read_bytes()).hexdigest()
                        if args.sampling_config
                        else None
                    ),
                    "sampling_predicate_exposure": predicate_exposure_sampling,
                    "predicate_objective_mask_channels": list(predicate_objective_mask_channels),
                    "partial_primary_objective_weight": args.partial_primary_objective_weight,
                }
            )
            if use_reference_type_residual:
                residual_exposure = trainer.reference_type_residual_training_exposure_receipt()
                training_exposure["reference_type_residual"] = residual_exposure
                training_exposure["missing_positive_reference_type_residuals"] = sorted(
                    primary_type
                    for primary_type, values in residual_exposure["types"].items()
                    if values["positive_tokens"] <= 0
                )
            if use_subclass_loss:
                subclass_exposure = trainer.subclass_training_exposure_receipt()
                declared_subclass_targets = declared_subclass_exposure_targets(args.sampling_config)
                training_exposure["subclass"] = subclass_exposure
                training_exposure["declared_positive_subclass_targets"] = list(declared_subclass_targets)
                training_exposure["missing_positive_subclass_targets"] = [
                    target
                    for target in declared_subclass_targets
                    if subclass_exposure["by_family_target"].get(target, {}).get("components", 0) <= 0
                ]
            predicate_positive = training_exposure["predicate"]["positive_token_cells"]
            reference_positive = {
                primary_type: values["positive_tokens"]
                for primary_type, values in training_exposure["reference"].items()
            }
            training_exposure["missing_positive_predicates"] = sorted(
                channel
                for channel, cells in predicate_positive.items()
                if cells <= 0 and channel not in predicate_objective_mask_channels
            )
            training_exposure["missing_positive_reference_types"] = sorted(
                primary_type for primary_type, tokens in reference_positive.items() if tokens <= 0
            )
            exposure_path = Path(args.out) / "training_predicate_exposure.json"
            exposure_path.write_text(
                json.dumps(training_exposure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                "TRAIN-PREDICATE-EXPOSURE: "
                f"windows={training_exposure['windows']} "
                f"missing-predicates={training_exposure['missing_positive_predicates']} "
                f"missing-references={training_exposure['missing_positive_reference_types']} "
                f"missing-reference-residuals="
                f"{training_exposure.get('missing_positive_reference_type_residuals', [])} "
                f"missing-subclasses="
                f"{training_exposure.get('missing_positive_subclass_targets', [])} "
                f"receipt={exposure_path}",
                flush=True,
            )
        interrupted_checkpoint = orderly_checkpoint.require_resumable_checkpoint(args.out)
        if interrupted_checkpoint is not None:
            print(f"TRAIN: orderly stop saved resumable checkpoint {interrupted_checkpoint}", flush=True)
            orderly_checkpoint.raise_if_terminated()

        if args.keep_final_checkpoint:
            final_checkpoint = Path(args.out) / f"checkpoint-{trainer.state.global_step}"
            if not is_valid_trainer_checkpoint(final_checkpoint):
                raise RuntimeError(
                    f"--keep-final-checkpoint: missing resumable final checkpoint {final_checkpoint}"
                )
            print(f"TRAIN-CHECKPOINTS: retained final checkpoint {final_checkpoint}", flush=True)

        selected_checkpoint = trainer.state.best_model_checkpoint
        selected_step = checkpoint_step(selected_checkpoint) if selected_checkpoint else None
        if selected_checkpoint is None or selected_step is None:
            raise RuntimeError(
                "training finished without a selected checkpoint; victory-lap and final export "
                "require at least one aligned evaluation/save event"
            )
        selected_eval = trainer.evaluate()
        selected_loss = selected_eval.get("eval_loss")
        selected_metric_name = str(targs.metric_for_best_model)
        selected_metric_value = selected_eval.get(selected_metric_name)
        recorded_best_metric = trainer.state.best_metric
        if (
            selected_loss is None
            or selected_metric_value is None
            or recorded_best_metric is None
            or not math.isclose(
                float(selected_metric_value),
                float(recorded_best_metric),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise RuntimeError(
                "in-memory model does not reproduce the selected checkpoint metric: "
                f"name={selected_metric_name} selected={recorded_best_metric!r} "
                f"recheck={selected_metric_value!r} checkpoint={selected_checkpoint}"
            )
        print(
            "TRAIN-SELECTION: "
            f"checkpoint={selected_checkpoint} step={selected_step} "
            f"metric={selected_metric_name} best={float(recorded_best_metric):.8g} "
            f"reloaded={float(selected_metric_value):.8g} "
            f"eval-loss={float(selected_loss):.8g} verified=true",
            flush=True,
        )
        print(f"TRAIN: {tag}: SELECTED eval {selected_eval}", flush=True)

    best_output = Path(args.out) / "best"
    if args.victory_lap_lr_scale > 0:
        lap_output_dir = Path(args.out) / "victory-lap"
        if lap_output_dir.exists():
            raise FileExistsError(f"victory-lap output already exists: {lap_output_dir}")
        lap_output_dir.mkdir(parents=True)
        lap_targs = victory_lap_training_arguments(
            targs,
            output_dir=lap_output_dir,
            learning_rate_scale=args.victory_lap_lr_scale,
            selected_step=selected_step,
            validation_seed=validation_seed,
            validation_windows=len(val),
        )
        lap_sampler_epoch_steps = math.ceil(len(val) / eff)
        lap_orderly_checkpoint = OrderlyCheckpointCallback()
        lap_callbacks = [
            RollingTrainLossCallback(lap_sampler_epoch_steps, 1.0),
            HeadlineCallback(f"{tag}-victory-lap", head_spec, eff),
            lap_orderly_checkpoint,
            RunMetricsCallback("victory_lap"),
        ]
        lap_trainer = trainer_class(
            model=trainer.model,
            args=lap_targs,
            train_dataset=val,
            processing_class=tok,
            data_collator=data_collator,
            callbacks=lap_callbacks,
            **trainer_kwargs,
        )
        bind_task_configuration(
            lap_trainer,
            dual_head_retired=getattr(trainer, "dual_head_retired", False),
            include_transition_restart=False,
        )
        scheduler_name = getattr(targs.lr_scheduler_type, "value", str(targs.lr_scheduler_type))
        base_rates = {
            "head": float(targs.learning_rate),
            "encoder": getattr(targs, "pii_encoder_learning_rate", None),
            "character": getattr(targs, "pii_character_learning_rate", None),
            "character_output": getattr(targs, "pii_character_output_learning_rate", None),
        }
        lap_rates = {
            "head": float(lap_targs.learning_rate),
            "encoder": getattr(lap_targs, "pii_encoder_learning_rate", None),
            "character": getattr(lap_targs, "pii_character_learning_rate", None),
            "character_output": getattr(lap_targs, "pii_character_output_learning_rate", None),
        }
        lap_receipt = {
            "schema_version": 2,
            "status": "running",
            "selected_checkpoint": str(selected_checkpoint),
            "selected_step": selected_step,
            "selected_eval_loss": float(selected_loss),
            "selected_metric_name": selected_metric_name,
            "selected_metric_value": float(selected_metric_value),
            "validation_selection_receipt": str(validation_receipt_path),
            "validation_selected_sha256": validation_receipt["selected_sha256"],
            "validation_windows": len(val),
            "epochs": 1.0,
            "learning_rate_scale": args.victory_lap_lr_scale,
            "base_learning_rates": base_rates,
            "lap_learning_rates": lap_rates,
            "base_scheduler": scheduler_name,
            "warmup_ratio": float(targs.warmup_ratio),
            "schedule_origin_step": 0,
            "trajectory_step_offset": selected_step,
            "early_stopping_enabled": False,
            "evaluation_enabled": False,
            "lap_only": lap_only_selection is not None,
            "source_selection_receipt": (
                str(args.victory_lap_only_receipt.resolve())
                if args.victory_lap_only_receipt is not None
                else None
            ),
        }
        lap_receipt_path = lap_output_dir / "victory_lap.json"
        lap_receipt_path.write_text(
            json.dumps(lap_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trainer.model.config.pii_victory_lap = lap_receipt
        print(
            "TRAIN-VICTORY-LAP: "
            f"start selected={selected_checkpoint} step={selected_step} "
            f"validation-windows={len(val)} epochs=1 lr-scale={args.victory_lap_lr_scale:g} "
            f"base-scheduler={scheduler_name} base-rates={base_rates} lap-rates={lap_rates} "
            "early-stopping=false evaluation=false",
            flush=True,
        )
        with lap_orderly_checkpoint.signal_handlers():
            lap_result = lap_trainer.train()
        interrupted_checkpoint = lap_orderly_checkpoint.require_resumable_checkpoint(lap_output_dir)
        if interrupted_checkpoint is not None:
            print(
                f"TRAIN-VICTORY-LAP: orderly stop saved resumable checkpoint {interrupted_checkpoint}",
                flush=True,
            )
            lap_orderly_checkpoint.raise_if_terminated()
        lap_epoch = float(lap_trainer.state.epoch or 0.0)
        if lap_trainer.state.global_step != lap_trainer.state.max_steps or not math.isclose(
            lap_epoch,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "victory lap did not complete exactly one pass: "
                f"step={lap_trainer.state.global_step}/{lap_trainer.state.max_steps} epoch={lap_epoch}"
            )
        lap_receipt.update(
            {
                "status": "completed",
                "optimizer_steps": int(lap_trainer.state.global_step),
                "completed_epoch": lap_epoch,
                "training_loss": float(lap_result.training_loss),
                "former_validation_consumed_for_training": True,
            }
        )
        lap_receipt_path.write_text(
            json.dumps(lap_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lap_trainer.model.config.pii_victory_lap = lap_receipt
        # The lap model is not any numbered checkpoint, so `best` becomes a real
        # directory again; detach first or the save follows a symlink into one.
        detach_best_directory(best_output)
        lap_trainer.save_model(best_output)
        tok.save_pretrained(best_output)
        print(
            "TRAIN-VICTORY-LAP: "
            f"complete optimizer-steps={lap_trainer.state.global_step} epoch={lap_epoch:g} "
            f"training-loss={lap_result.training_loss:.8g} receipt={lap_receipt_path} "
            f"export={best_output}",
            flush=True,
        )
        log_format.headline(
            f"{tag}: victory lap done; selected {selected_metric_name}="
            f"{float(selected_metric_value):.4f} -> {best_output}"
        )
    else:
        trainer.model.config.pii_victory_lap = None
        if selected_checkpoint is not None:
            link_best_to_checkpoint(best_output, Path(selected_checkpoint))
            print(
                f"TRAIN-BEST: {best_output} -> {Path(selected_checkpoint).name} "
                "(symlink; the copy was byte-identical)",
                flush=True,
            )
        else:
            detach_best_directory(best_output)
            trainer.save_model(best_output)
            tok.save_pretrained(best_output)
        print(f"TRAIN-VICTORY-LAP: disabled lr-scale=0 export={best_output}", flush=True)
        log_format.headline(
            f"{tag}: done; selected {selected_metric_name}="
            f"{float(selected_metric_value):.4f} -> {best_output}"
        )

    if args.prune_early_checkpoints and selected_step is not None:
        dropped, freed = prune_checkpoints_before_selected(Path(args.out), int(selected_step))
        if dropped:
            print(
                f"TRAIN-CHECKPOINTS: dropped {len(dropped)} before the selection "
                f"({dropped[0]}..{dropped[-1]}), freed {freed / 2**30:.1f} GiB; "
                f"kept the selection and its predecessor plus everything after it",
                flush=True,
            )
    if args.thin_optimizer_state and trainer.is_world_process_zero():
        thinned, freed = thin_optimizer_state(Path(args.out))
        if thinned:
            print(
                f"TRAIN-CHECKPOINTS: removed optimizer state from {len(thinned)} non-terminal "
                f"checkpoints ({thinned[0]}..{thinned[-1]}), freed {freed / 2**30:.1f} GiB; "
                f"the terminal checkpoint stays resumable",
                flush=True,
            )

    checkpoint_mirror.finish(args.out, is_world_process_zero=trainer.is_world_process_zero())


if __name__ == "__main__":
    main()
