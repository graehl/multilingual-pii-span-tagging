"""GLiNER2 processor and loss integration, preserving stock checkpoint weights."""

from types import MethodType

import torch
from gliner2.processor import SchemaTransformer

from scripts.pii_gliner2_mapped_loss import acceptable_span_loss


class MappedSchemaTransformer(SchemaTransformer):
    def _transform_record(self, record, max_len=None, *, build_targets=False):
        schema = record["schema"]
        targets = schema.get("pii_acceptable_targets")
        if targets is None:
            raise ValueError("acceptable-label training requires target metadata on every row")
        fields = list(schema["entities"])
        sampling = self.sampling_config
        if sampling.shuffle_entities or sampling.remove_entities_prob or sampling.remove_entity_prob:
            raise ValueError("mapped targets require preserved entity field membership and order")
        transformed = super()._transform_record(record, max_len, build_targets=build_targets)
        if transformed.task_types != ["entities"]:
            raise ValueError("mapped objective supports only the entity task")
        structure = transformed.structure_labels[0]
        if len(structure[1][0]) != len(fields):
            raise ValueError("processor changed mapped field order or membership")
        groups = [
            {**group, "labels": [fields.index(label) for label in group["labels"]]}
            for group in targets["groups"]
        ]
        for group in groups:
            if group["end"] >= len(transformed.text_tokens):
                raise ValueError("processor truncated a mapped target")
        known = [targets["complete"] and label not in targets["unknown_types"] for label in fields]
        structure.append({"groups": groups, "known_negative": known})
        return transformed


def mapped_struct_loss(self, span_rep, schema_emb, structure, span_mask, masking_rate=0.5):
    if len(structure) != 3 or structure[0] != 1:
        raise ValueError("missing acceptable-label structure metadata")
    projection = self.count_embed(schema_emb[1:], 1)
    scores = torch.einsum("lkd,bpd->bplk", span_rep, projection)
    targets = structure[2]
    return acceptable_span_loss(
        scores,
        targets["groups"],
        targets["known_negative"],
        ~span_mask[0],
        masking_rate=masking_rate if self.training else 0.0,
    )


def install_mapped_training(model):
    """Replace only processor target handling and the entity loss."""
    original = model.processor
    processor = MappedSchemaTransformer(
        tokenizer=original.tokenizer,
        sampling_config=original.sampling_config,
        token_pooling=original.token_pooling,
        word_splitter=original.word_splitter,
    )
    core = model.model if hasattr(model, "model") else model
    core._pii_original_forward = core.forward
    core.forward = MethodType(checked_forward, core)
    core.compute_struct_loss = MethodType(mapped_struct_loss, core)
    core.processor = processor
    model.processor = processor
    core.config.pii_training_objective = "acceptable-span-bernoulli-union-v1"


def checked_forward(self, batch, *args, **kwargs):
    result = self._pii_original_forward(batch, *args, **kwargs)
    if result["batch_size"] != len(batch):
        raise ValueError("GLiNER2 discarded a sample while computing the mapped objective")
    return result


def training_records(records, mapping):
    """Use explicit schemas so the stock dataset preserves target metadata."""
    result = []
    for row in records:
        targets = row["acceptable_targets"]
        groups = [{**g, "labels": [mapping[label] for label in g["labels"]]} for g in targets["groups"]]
        entities = {mapping[label]: surfaces for label, surfaces in row["entities"].items()}
        for group in groups:
            for label in group["labels"]:
                entities.setdefault(label, [])
        result.append(
            {
                "input": row["text"],
                "output": {
                    "entities": entities,
                    "pii_acceptable_targets": {
                        **targets,
                        "groups": groups,
                        "unknown_types": [mapping[label] for label in targets["unknown_types"]],
                    },
                },
            }
        )
    return result
