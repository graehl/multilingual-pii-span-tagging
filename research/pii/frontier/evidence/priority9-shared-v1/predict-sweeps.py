"""One-off saved-score producer for the five-model priority-nine comparison.

Run from the repository root with MODEL INPUT OUTPUT arguments; MODEL is one
of ont1, ont2, ont3, gliner2 or gliner2-tuned. Worker paths are pinned below.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from pii_eval import (
    GLINER2_FRONTIER_REVISION,
    GLINER2_ID,
    GLINER2_LABELS,
    _predict_hf_tokcls_biases,
    dedupe_window_preds,
    windows,
)
from pii_ont3_eval import _decode_window, merge_adjacent_same_type, predict_rows

GRID = [*range(-8, 17), 24, 32]
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
MODELS = {
    "ont1": "/scratch/paper-priority9-ont1-model",
    "ont2": "/scratch/artifacts/pii-redaction-frontier/models/paper-ont2-checkpoint",
    "ont3": "/scratch/artifacts/pii-redaction-frontier/models/pii-ont3-v35-xlmr-large-allgold-r2-seed173-v1/best",
    "ont3-context": "/scratch/artifacts/pii-redaction-frontier/models/pii-ont3-context-large-pretrained-masked-both-seed173-v1/checkpoint-14250",
    "gliner2-tuned": "/scratch/paper-priority9-gliner2",
}


def main():
    import torch

    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=[*MODELS, "gliner2"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model-path",
        action="append",
        help="explicit checkpoint for matched initial/trained comparisons; repeat (ont3 kinds) to "
        "decode the mean of the models' primary logits (a logit-blend ensemble)",
    )
    parser.add_argument("--context-side", choices=["both", "previous", "none"], default="both")
    parser.add_argument("--labels", type=Path, default=Path("/scratch/paper-priority9-labels.json"))
    args = parser.parse_args()
    kind = args.kind
    path, output = args.input, args.output
    rows = [json.loads(line) for line in path.open()]
    assert rows and len(rows) == len({r["text"] for r in rows}) == len({r["id"] for r in rows})
    assert not output.exists()
    points = {}
    synthetic_suffix = dict(clipped=0, omitted=0)
    model_paths = args.model_path or []
    if len(model_paths) > 1 and kind not in {"ont3", "ont3-context"}:
        parser.error("several --model-path values (a logit blend) need an ont3 kind")
    model_source = (model_paths[0] if len(model_paths) == 1 else model_paths) or MODELS.get(kind)
    mapping = None
    if kind in {"ont1", "ont2"}:
        results = _predict_hf_tokcls_biases(
            [r["text"] for r in rows],
            print,
            model_id=model_source,
            bioes_project_legal=True,
            o_logit_biases=GRID,
            bucket_compat="redaction_20_v1" if kind == "ont1" else "fine",
            hf_windowing="token-capacity",
        )
        for bias, predictions in results.items():
            points[str(bias)] = [
                dict(id=row["id"], preds=spans) for row, (spans, _) in zip(rows, predictions, strict=True)
            ]
    elif kind in {"ont3", "ont3-context"}:
        if kind == "ont3-context":
            assert all("document_context" in row for row in rows)
        members = model_source if isinstance(model_source, list) else [model_source]
        caches, configs = [], []
        for member in members:
            config = json.loads((Path(member) / "config.json").read_text())
            member_cache = []

            def sink(key, logits, offsets, member_cache=member_cache):
                member_cache.append((key, logits, offsets))

            predict_rows(
                member,
                rows,
                config.get("pii_predicate_channels", []),
                context_field="document_context"
                if kind == "ont3-context" and args.context_side != "none"
                else "",
                context_side="previous" if args.context_side == "previous" else "both",
                window_length=512,
                primary_window_sink=sink,
            )
            caches.append(member_cache)
            configs.append(config)
        config = configs[0]
        # A logit blend decodes the mean primary logits; every member must share the
        # label space and produce the same windows (same tokenizer and context).
        if any(c["id2label"] != config["id2label"] for c in configs[1:]):
            parser.error("blended checkpoints have different label spaces")
        cache = []
        for windows_by_member in zip(*caches, strict=True):
            key, _, offsets = windows_by_member[0]
            if any(k != key or list(o) != list(offsets) for k, _, o in windows_by_member[1:]):
                raise SystemExit("blended checkpoints produced different windows")
            stacked = [lg.float() if hasattr(lg, "float") else lg for _, lg, _ in windows_by_member]
            cache.append((key, sum(stacked) / len(stacked), offsets))
        labels = {int(k): v for k, v in config["id2label"].items()}
        outside = [k for k, v in labels.items() if v == "O"]
        assert len(outside) == 1
        for bias in GRID:
            predictions = {r["id"]: [] for r in rows}
            for key, logits, offsets in cache:
                spans, _, _ = _decode_window(logits, offsets, labels, column_biases={outside[0]: bias})
                predictions[key].extend(spans)
            points[str(float(bias))] = []
            for row in rows:
                spans, _ = merge_adjacent_same_type(
                    dedupe_window_preds(predictions[row["id"]]),
                    row["text"],
                    frozenset({"person_name", "organization"}),
                )
                points[str(float(bias))].append(dict(id=row["id"], preds=spans))
    else:
        from gliner2 import GLiNER2

        if kind == "gliner2":
            from huggingface_hub import snapshot_download

            model_source = model_source or snapshot_download(GLINER2_ID, revision=GLINER2_FRONTIER_REVISION)
            # The published privacy inventory omits organizations, which are
            # required in this comparison. Preserve the published names and
            # explicitly request this additional primary type.
            labels = [*GLINER2_LABELS, "organization"]
        else:
            assert kind == "gliner2-tuned"
            from pii_gliner2_ont3_predict import ont3_labels

            labels = ont3_labels(args.labels)
        model = GLiNER2.from_pretrained(model_source).to("cuda").eval()
        from pii_gliner2_label_transfer import prompt_mapping

        mapping = prompt_mapping(model, labels)
        reverse = {prompt: canonical for canonical, prompt in mapping.items()}
        labels = list(mapping.values())
        candidates = []
        for index, row in enumerate(rows):
            found = {}
            for offset, text in windows(row["text"], max_chars=1200, overlap=200):
                entities = model.extract_entities(
                    text, labels, threshold=min(THRESHOLDS), include_confidence=True, include_spans=True
                )["entities"]
                for label, values in entities.items():
                    for value in values:
                        start, end = value["start"], value["end"]
                        # GLiNER2 appends a period before splitting unfinished
                        # sentences. Predictions must refer only to real input.
                        processed_length = len(text) + int(not text.endswith((".", "!", "?")))
                        if not 0 <= start < end <= processed_length:
                            raise ValueError(f"GLiNER2 returned invalid offsets: {value}")
                        if start >= len(text):
                            synthetic_suffix["omitted"] += 1
                            continue
                        if end > len(text):
                            synthetic_suffix["clipped"] += 1
                            end = len(text)
                        key = (start + offset, end + offset, reverse[label])
                        found[key] = max(found.get(key, 0), float(value["confidence"]))
            candidates.append(
                dict(
                    id=row["id"],
                    preds=[
                        dict(start=a, end=b, label=t, confidence=c) for (a, b, t), c in sorted(found.items())
                    ],
                )
            )
            if index % 100 == 0:
                print(f"[predict] {kind} {index}/{len(rows)}", flush=True)
        for threshold in THRESHOLDS:
            points[str(threshold)] = [
                dict(id=r["id"], preds=[p for p in r["preds"] if p["confidence"] >= threshold])
                for r in candidates
            ]
    receipt = dict(
        model=kind,
        model_path=model_source,
        context_side=args.context_side if kind == "ont3-context" else "none",
        input=str(path),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=len(rows),
        requested_labels=labels if kind.startswith("gliner2") else None,
        canonical_to_prompt=mapping,
        synthetic_suffix=synthetic_suffix if kind.startswith("gliner2") else None,
        points=points,
    )
    output.write_text(json.dumps(receipt) + "\n")
    print(f"[complete] {kind} {len(points)} points x {len(rows)} inputs -> {output}", flush=True)


if __name__ == "__main__":
    main()
