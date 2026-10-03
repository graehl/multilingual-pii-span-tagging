"""Bijective Ont3 prompt names and auditable input-embedding transfer.

Donor associations initialize tokens; they never relax the supervised objective
or scoring. Existing vocabulary names keep their tokenization and embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CONFIG_KEY = "pii_ont3_label_transfer"


def load_transfer(path: Path) -> dict:
    from scripts.pii_ontology_v2 import load_ontology
    from scripts.pii_reference_projection import REFERENCE_BASE_TYPES

    spec = json.loads(path.read_text())
    if spec.get("schema") != "pii-gliner2-label-transfer/v1":
        raise ValueError("unsupported label-transfer schema")
    entries = spec["labels"]
    names = [entry["prompt"] for entry in entries.values()]
    if not entries or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("prompt names must be nonempty and one-to-one")
    ontology = load_ontology()
    if set(entries) != set(ontology.primary_types) | set(REFERENCE_BASE_TYPES):
        raise ValueError("transfer inventory must cover exactly the 31 Ont3 primary types")
    for canonical, entry in entries.items():
        donors = entry["donors"]
        if not donors:
            raise ValueError(f"{canonical}: no initialization donors")
        for donor in donors:
            if donor["role"] not in {"preferred", "allowed", "lexical"}:
                raise ValueError(f"{canonical}: unknown donor role")
            if not donor["name"] or not math.isfinite(donor["weight"]) or donor["weight"] <= 0:
                raise ValueError(f"{canonical}: invalid donor weight/name")
            if donor["role"] != "lexical":
                target = REFERENCE_BASE_TYPES.get(canonical, canonical)
                if target not in ontology.source_acceptable("fastino_42", donor["name"]):
                    raise ValueError(f"{canonical}: donor {donor['name']} is not an allowed source mapping")
    return spec


def prompt_mapping(model, labels: list[str]) -> dict[str, str]:
    spec = getattr(model.config, CONFIG_KEY, None)
    if spec is None:
        return {label: label for label in labels}
    if set(labels) != set(spec["labels"]):
        raise ValueError("requested canonical inventory differs from checkpoint transfer inventory")
    return {label: spec["labels"][label]["prompt"] for label in labels}


def translate_records(records: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [
        {**row, "entities": {mapping[label]: surfaces for label, surfaces in row["entities"].items()}}
        for row in records
    ]


def initialize_transfer(model, spec: dict) -> dict:
    """Initialize only newly added rows, before optimizer construction.

    Compute every donor from the original tokenizer and embedding table before
    adding any token, so iteration order cannot change donor representations.
    Store the contract in model config, which every upstream checkpoint saves.
    """
    import torch
    from tokenizers import AddedToken

    if getattr(model.config, CONFIG_KEY, None) is not None:
        raise ValueError("checkpoint already has label transfer; do not reinitialize it")
    tokenizer = model.processor.tokenizer
    original = model.encoder.get_input_embeddings().weight
    old_size = original.shape[0]
    targets = {}
    receipt = {}
    for canonical, entry in spec["labels"].items():
        name = entry["prompt"]
        pieces = tokenizer.tokenize(name)
        ids = tokenizer.convert_tokens_to_ids(pieces)
        donors = []
        vectors = []
        for donor in entry["donors"]:
            donor_pieces = tokenizer.tokenize(donor["name"])
            donor_ids = tokenizer.convert_tokens_to_ids(donor_pieces)
            if not donor_ids or tokenizer.unk_token_id in donor_ids:
                raise ValueError(f"unrepresentable donor {donor['name']}")
            vectors.append(original.detach()[donor_ids].float().mean(dim=0) * donor["weight"])
            donors.append({**donor, "tokens": donor_pieces, "ids": donor_ids})
        # SentencePiece can encode an existing bare token as a word-start piece
        # plus that token (e.g. age). add_tokens would reuse its old row rather
        # than allocate a new one. Preserve those names and natural singletons.
        reused = name in tokenizer.get_vocab() or (len(ids) == 1 and ids[0] != tokenizer.unk_token_id)
        if not reused:
            targets[name] = torch.stack(vectors).sum(dim=0) / sum(d["weight"] for d in donors)
        receipt[canonical] = dict(prompt=name, reused=reused, original_ids=ids, donors=donors)
    added = tokenizer.add_tokens([AddedToken(name, normalized=False, single_word=True) for name in targets])
    if added != len(targets):
        raise ValueError("not every requested atomic prompt token was newly added")
    model.encoder.resize_token_embeddings(len(tokenizer))
    embedding = model.encoder.get_input_embeddings().weight
    with torch.no_grad():
        for name, vector in targets.items():
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(name))
            if len(ids) != 1 or ids[0] < old_size:
                raise ValueError(f"new prompt name {name} is not one new token")
            embedding[ids[0]].copy_(vector.to(embedding))
    model.processor._tokenize_cached.cache_clear()
    for canonical, entry in spec["labels"].items():
        receipt[canonical]["ids"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(entry["prompt"]))
    setattr(model.config, CONFIG_KEY, spec)
    return {
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "original_vocabulary_size": old_size,
        "vocabulary_size": len(tokenizer),
        "added_tokens": added,
        "labels": receipt,
    }
