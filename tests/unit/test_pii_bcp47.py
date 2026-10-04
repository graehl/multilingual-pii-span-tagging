import hashlib
import json

import pytest

from scripts.pii_bcp47 import materialize_jsonl_sidecar


def test_materialize_jsonl_sidecar_prefers_explicit_bcp47_and_receipts_identity(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "bcp47": "EN-us", "lang": "en"}),
                json.dumps({"id": "two", "lang": "zh-hant-tw"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "input.bcp"
    receipt_path = tmp_path / "input.bcp.receipt.json"

    receipt = materialize_jsonl_sidecar(source, output, receipt_path)

    assert output.read_text(encoding="utf-8") == "en-US\nzh-Hant-TW\n"
    assert receipt["source_fields"] == {"bcp47": 1, "lang": 1}
    assert receipt["tag_counts"] == {"en-US": 1, "zh-Hant-TW": 1}
    assert receipt["input"]["ordered_id_sha256_with_lf"] == hashlib.sha256(b"one\ntwo\n").hexdigest()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"id": "one", "text": "missing language"}], "missing bcp47 and lang"),
        ([{"id": "one", "lang": "en"}, {"id": "one", "lang": "en"}], "duplicate id"),
    ],
)
def test_materialize_jsonl_sidecar_rejects_ambiguous_rows(tmp_path, rows, message):
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        materialize_jsonl_sidecar(source, tmp_path / "input.bcp", tmp_path / "receipt.json")
