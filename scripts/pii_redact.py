#!/usr/bin/env python3
"""Tag and redact raw text with a trained Ont3 checkpoint.

Input is plain text (one document per line) or JSONL with `text` and optional
`id` and `lang`. Output JSONL keeps each document's typed spans and a redacted
copy in which every span becomes `[label]`. Decoding is the model's zero
outside-bias operating point with maximal person and organization spans.

The paper serves O4 with three further stages, in this order, each optional
here: character-level boundary refinement (`--refiner`), name-component
postprocessing (`--name-kind-bundle`) and regular-expression supplementation
of structured identifiers where the model found nothing (`--regex`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import acli

REFERENCES = {"person_reference", "organization_reference"}
NAME_CONFIG = ROOT / "research/pii/frontier/models/name-components/name-postprocessor-name-kind.json"
REGEX_RULES = ROOT / "scripts/pii_regex_tags_v1.json"
REGEX_POLICY = ROOT / "scripts/pii_regex_policy_ont3_v1.json"


def read_documents(path: Path, lang: str) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = []
    for index, line in enumerate(lines, start=1):
        row = json.loads(line) if path.suffix == ".jsonl" else {"text": line}
        documents.append(
            {"id": str(row.get("id", index)), "text": row["text"], "lang": row.get("lang", lang)}
        )
    if not documents:
        raise ValueError(f"no documents in {path}")
    return documents


def redacted(text: str, spans: list[dict]) -> str:
    pieces, cursor = [], 0
    for span in sorted(spans, key=lambda span: (span["start"], -span["end"])):
        if span["start"] < cursor:
            continue
        pieces.extend((text[cursor : span["start"]], f"[{span['label']}]"))
        cursor = span["end"]
    return "".join(pieces) + text[cursor:]


def refine(documents: list[dict], rows: list[dict], refiner: Path) -> dict:
    """Boundary refinement of primary spans; reference spans pass through unchanged."""
    import torch

    from scripts.pii_character_boundary_refiner import load_refiner, refine_prediction_rows

    device = torch.device("cpu")
    model, config = load_refiner(refiner, device)
    primary = [{**row, "preds": [p for p in row["preds"] if p["label"] not in REFERENCES]} for row in rows]
    refined, telemetry = refine_prediction_rows(documents, primary, model, config, device=device)
    for row, result in zip(rows, refined, strict=True):
        row["preds"] = result["preds"] + [p for p in row["preds"] if p["label"] in REFERENCES]
    return telemetry.get("aggregate", {})


def name_kinds(documents: list[dict], rows: list[dict], bundle: Path) -> int:
    from scripts.pii_name_annotation_qc import NameAnnotationQc, load_config
    from scripts.pii_name_role_publish import NameKindOnnxDeployment

    engine = NameAnnotationQc(
        load_config(NAME_CONFIG),
        mode="insert-name-kinds",
        override_name_kinds=False,
        name_kind_deployment=NameKindOnnxDeployment.load(bundle),
    )
    changed = 0
    for document, row in zip(documents, rows, strict=True):
        analysis = engine.analyze(
            row_id=document["id"], text=document["text"], language=document["lang"], annotations=row["preds"]
        )
        preds = engine.inference_fields(analysis)["preds"]
        changed += preds != row["preds"]
        row["preds"] = preds
    return changed


def regex(documents: list[dict], rows: list[dict]) -> dict:
    from scripts.pii_regex_tag import load_policy, load_rules, tag_rows

    rules, families, _ = load_rules(REGEX_RULES, "ont3")
    policy, arbitration, _ = load_policy(REGEX_POLICY)
    return tag_rows(
        rows,
        rules,
        span_keys=["preds"],
        policy=policy,
        families=families,
        scores=arbitration["scores"],
        gates=arbitration["gates"],
        texts=[d["text"] for d in documents],
        languages=[d["lang"] for d in documents],
    )


def run(args) -> dict:
    from scripts.pii_ont3_eval import predict_rows

    config = json.loads((args.checkpoint / "config.json").read_text())
    documents = read_documents(args.input, args.lang)
    predictions, summary = predict_rows(
        args.checkpoint,
        documents,
        config.get("pii_predicate_channels", []),
        merge_adjacent_types=frozenset({"person_name", "organization"}),
    )
    rows = [
        {"id": d["id"], "preds": p["reference_aware_preds"]}
        for d, p in zip(documents, predictions, strict=True)
    ]
    stages = {}
    if args.refiner:
        stages["boundary_refinement"] = refine(documents, rows, args.refiner)
    if args.name_kind_bundle:
        stages["name_kind_changed_rows"] = name_kinds(documents, rows, args.name_kind_bundle)
    if args.regex:
        stages["regex"] = regex(documents, rows)
    allowed = set(args.types.split(",")) if args.types else None
    emitted = withheld = 0
    with args.out.open("x", encoding="utf-8") as stream:
        for document, row in zip(documents, rows, strict=True):
            spans = [
                {**span, "text": document["text"][span["start"] : span["end"]]}
                for span in row["preds"]
                if allowed is None or span["label"] in allowed
            ]
            emitted += len(spans)
            withheld += len(row["preds"]) - len(spans)
            record = {**document, "spans": spans, "redacted": redacted(document["text"], spans)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "ok": True,
        "out": str(args.out.resolve()),
        "documents": len(documents),
        "spans": emitted,
        "spans_outside_types": withheld,
        "types": sorted(allowed) if allowed else "all",
        "serving_stages": stages,
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


def main() -> None:
    parser = acli.argument_parser(description=__doc__, capabilities=("complete",))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Text (one document per line) or JSONL")
    parser.add_argument("--out", type=Path, required=True, help="New JSONL with spans and redacted text")
    parser.add_argument("--lang", default="en", help="Language code for rows that do not name one")
    parser.add_argument(
        "--types",
        help="Comma-separated Ont3 types to emit. A model is unreliable on types its training data never "
        "supervised; restrict output to the types it learned.",
    )
    parser.add_argument(
        "--refiner", type=Path, help="Boundary-refiner directory (config.json, model.safetensors)"
    )
    parser.add_argument(
        "--name-kind-bundle", type=Path, help="Name-kind bundle config (built by the names command)"
    )
    parser.add_argument(
        "--regex", action="store_true", help="Add regex spans for identifiers the model missed"
    )
    acli.add_standard_args(parser)
    acli.maybe_complete(parser)
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, KeyError) as error:
        acli.die(str(error), acli.ExitCode.SOFTWARE)
    acli.emit(result, fmt=acli.resolve_format(args))


if __name__ == "__main__":
    main()
