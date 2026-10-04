import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.pii_eval import (
    _align_gold_to_predictions,
    _cache_document_languages,
    _cache_document_texts,
    _canonicalize_prediction_labels,
    _maximum_matches,
    _resolve_suppressed_primary_columns,
    _tokenizer_capacity_windows,
    _with_o_logit_bias,
    _with_suppressed_primary_columns,
    apply_name_oracle_to_rows,
    bucket_compat_map,
    character_logits_score_window,
    covered_word_boundary_offsets,
    decode_bioes_labels,
    decode_span_score_cache,
    decode_token_score_cache,
    load_aqmar_splits,
    load_character_onnx_bundle,
    load_character_token_onnx_bundle,
    load_hiner_splits,
    load_idner_news_splits,
    load_mapa_test,
    load_openner_core_splits,
    load_ordered_label_vocabulary,
    load_span_score_cache,
    load_token_score_cache,
    load_unified_eval,
    load_wojood_splits,
    o_logit_bias_slug,
    overlaps,
    preponderance_token_weights,
    score_one,
    subset_token_score_cache,
    token_logits_score_window,
    unicode_whitespace_split_boundaries,
    unicode_word_boundary_positions,
    write_span_score_cache,
    write_token_score_cache,
)
from scripts.pii_name_annotation_qc import NameAnnotationQc
from scripts.pii_name_annotation_qc import load_config as load_name_qc_config
from scripts.pii_projector import Tagset


def test_script_entrypoint_exposes_repo_package_to_late_imports():
    repo = Path(__file__).resolve().parents[2]
    probe = (
        f"import runpy,sys;repo={str(repo)!r};"
        "sys.path=[repo+'/scripts']+[path for path in sys.path if path!=repo];"
        "runpy.run_path(repo+'/scripts/pii_eval.py',run_name='pii_eval_probe');"
        "import pii_continuous_character_cnn"
    )

    subprocess.run([sys.executable, "-c", probe], cwd=repo, check=True)


def test_projector_canonical_and_shared_schema_image():
    tagset = Tagset()
    assert tagset.project("canonical", "given_name") == "given_name"
    assert tagset.project("openmed_54", "BANKACCOUNT") == "account_number"
    assert tagset.project("openmed_nemotron_55", "company_name") == "organization"
    assert tagset.shared_image("openmed_nemotron_55", "nemotron_pii") == tagset.schema_image("nemotron_pii")
    with pytest.raises(ValueError, match="unknown label"):
        tagset.project("openmed_54", "COMPANYNAME")


def test_reporting_cuts_are_exhaustive_and_preserve_key_redaction_distinctions():
    tagset = Tagset()
    for cut in tagset.cut_names():
        assert {tagset.project_canonical_cut(node, cut) for node in tagset.nodes}
        assert all(tagset.project_canonical_cut(node, cut) for node in tagset.nodes)

    assert tagset.project_cut("authored_v1", "last_name", "redaction_20_v1") == "family_name"
    assert tagset.project_cut("authored_v1", "first_name", "redaction_20_v1") == "person_name"
    assert tagset.project_cut("authored_v1", "last_name", "redaction_9_v1") == "name"
    for label in ("medical_record_number", "health_plan_beneficiary_number", "employee_id"):
        assert tagset.project_cut("authored_v1", label, "redaction_20_v1") == "org_issued_id"
        assert tagset.project_cut("authored_v1", label, "redaction_9_v1") == "unique_identifier"
    assert tagset.project_cut("authored_v1", "unique_id", "redaction_9_v1") == "other_sensitive"


def test_native_cut_schema_projects_only_to_equal_or_coarser_cuts():
    tagset = Tagset()
    assert tagset.is_cut_schema("redaction_20_v1")
    assert tagset.cut_targets("redaction_20_v1") == set(tagset.cuts["redaction_20_v1"])
    assert tagset.project_cut("redaction_20_v1", "family_name", "redaction_20_v1") == "family_name"
    assert tagset.project_cut("redaction_20_v1", "family_name", "redaction_9_v1") == "name"
    assert tagset.cut_image("redaction_20_v1", "redaction_9_v1") == set(tagset.cuts["redaction_9_v1"])
    with pytest.raises(ValueError, match="would require refinement"):
        tagset.project_cut("redaction_9_v1", "name", "redaction_20_v1")
    with pytest.raises(ValueError, match="fine canonical image is undefined"):
        tagset.schema_image("redaction_20_v1")
    with pytest.raises(ValueError, match="redaction_20_v1: unknown label"):
        tagset.project_cut("redaction_20_v1", "last_name", "redaction_20_v1")


def test_authored_schema_covers_fresh_evaluation_refinements():
    tagset = Tagset()
    expected = {
        "document_code": "document_code",
        "job_area": "job_area",
        "ipv6": "ipv6",
        "monetary_amount": "monetary_amount",
    }
    assert {label: tagset.project("authored_v1", label) for label in expected} == expected


def test_maximum_matching_is_one_to_one_not_independent_coverage():
    gold = [{"start": 0, "end": 2}, {"start": 2, "end": 4}]
    pred = [{"start": 1, "end": 3}]
    assert _maximum_matches(gold, pred, lambda a, b: a["start"] < b["end"] and b["start"] < a["end"]) == 1


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((0, 10), (2, 10), True),  # exactly 80% of the longer span
        ((0, 10), (2, 12), True),  # exactly 80% of both spans
        ((0, 100), (0, 80), True),  # containment still covers 80% of the longer span
        ((0, 10), (0, 20), False),  # full coverage of one span is only 50% of the other
        ((0, 10), (3, 10), False),  # 70% of the longer span
        ((0, 10), (10, 20), False),  # touching endpoints have zero intersection
        ((0, 0), (0, 1), False),  # empty spans never match
    ],
)
def test_overlap_requires_symmetric_80_percent_coverage(left, right, expected):
    a = {"start": left[0], "end": left[1]}
    b = {"start": right[0], "end": right[1]}
    assert overlaps(a, b) is expected
    assert overlaps(b, a) is expected


def test_o_logit_bias_changes_only_outside_decisions():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 0.2, 0.1], [0.0, 0.1, 0.4]])
    original = logits.clone()

    assert _with_o_logit_bias(logits, 0, 0.0) is logits
    biased = _with_o_logit_bias(logits, 0, 0.3)
    assert biased.argmax(-1).tolist() == [0, 2]
    assert torch.equal(logits, original)


def test_primary_type_suppression_resolves_complete_families_and_masks_before_decode():
    torch = pytest.importorskip("torch")
    id2label = {
        0: "O",
        1: "B-person_reference",
        2: "I-person_reference",
        3: "E-person_reference",
        4: "S-person_reference",
        5: "B-person_name",
        6: "I-person_name",
        7: "E-person_name",
        8: "S-person_name",
    }
    columns = _resolve_suppressed_primary_columns(id2label, ["person_reference"])
    logits = torch.tensor([[0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]])
    masked = _with_suppressed_primary_columns(logits, columns)

    assert columns == (1, 2, 3, 4)
    assert masked.argmax(-1).tolist() == [5]
    assert torch.isneginf(masked[:, list(columns)]).all()
    assert torch.equal(logits, torch.tensor([[0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]]))


def test_primary_type_suppression_fails_closed_on_schema_mismatch():
    complete = {
        0: "O",
        1: "B-person_reference",
        2: "I-person_reference",
        3: "E-person_reference",
        4: "S-person_reference",
    }
    with pytest.raises(ValueError, match="duplicate suppressed"):
        _resolve_suppressed_primary_columns(
            complete,
            ["person_reference", "person_reference"],
        )
    with pytest.raises(ValueError, match="absent from model schema"):
        _resolve_suppressed_primary_columns(complete, ["organization_reference"])
    with pytest.raises(ValueError, match="complete BIOES family"):
        _resolve_suppressed_primary_columns(
            {0: "O", 1: "B-person_reference", 2: "E-person_reference"},
            ["person_reference"],
        )


def test_tokenizer_capacity_windows_preserve_document_offsets_and_fill_capacity():
    class FakeTokenizer:
        kwargs = None

        @staticmethod
        def num_special_tokens_to_add(pair=False):
            assert pair is False
            return 2

        def __call__(self, text, **kwargs):
            assert text == "abcdefgh"
            self.kwargs = kwargs
            return {
                "input_ids": [[0, 11, 12, 13, 2], [0, 13, 14, 2]],
                "attention_mask": [[1, 1, 1, 1, 1], [1, 1, 1, 1]],
                "offset_mapping": [
                    [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)],
                    [(0, 0), (4, 6), (6, 8), (0, 0)],
                ],
                "overflow_to_sample_mapping": [0, 0],
            }

    tokenizer = FakeTokenizer()
    stride, model_windows = _tokenizer_capacity_windows(tokenizer, "abcdefgh", max_length=6)

    assert stride == 1
    assert tokenizer.kwargs == {
        "return_offsets_mapping": True,
        "truncation": True,
        "max_length": 6,
        "stride": 1,
        "return_overflowing_tokens": True,
        "verbose": False,
    }
    assert model_windows == [
        (
            {"input_ids": [0, 11, 12, 13, 2], "attention_mask": [1, 1, 1, 1, 1]},
            [(0, 0), (0, 2), (2, 4), (4, 6), (0, 0)],
        ),
        (
            {"input_ids": [0, 13, 14, 2], "attention_mask": [1, 1, 1, 1]},
            [(0, 0), (4, 6), (6, 8), (0, 0)],
        ),
    ]
    with pytest.raises(ValueError, match="leaves no capacity"):
        _tokenizer_capacity_windows(tokenizer, "abcdefgh", max_length=2)


def test_character_logits_score_window_uses_identity_codepoint_offsets():
    np = pytest.importorskip("numpy")
    labels = ["O", "B-given_name", "E-given_name", "S-email"]
    logits = np.asarray(
        [
            [0.5, 1.5, 0.0, -1.0],
            [0.3, -1.0, 2.0, 0.0],
            [3.0, 0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    window = character_logits_score_window(logits, "Aé中", labels)

    assert window["token_start"] == [0, 1, 2]
    assert window["token_end"] == [1, 2, 3]
    assert window["top_non_o_label"] == [1, 2, 3]
    assert window["nfc_char_count"] == [1, 1, 1]
    assert window["full_logits"] == logits.tolist()
    with pytest.raises(ValueError, match="shape"):
        character_logits_score_window(logits[:2], "Aé中", labels)
    logits[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        character_logits_score_window(logits, "Aé中", labels)


def test_token_logits_score_window_preserves_native_intervals():
    np = pytest.importorskip("numpy")
    labels = ["O", "B-name", "E-name", "S-email"]
    logits = np.arange(12, dtype=np.float32).reshape(3, 4)
    offsets = [(0, 3), (2, 5), (6, 7)]

    window = token_logits_score_window(logits, "abc dé!", labels, offsets)

    assert window["token_start"] == [0, 2, 6]
    assert window["token_end"] == [3, 5, 7]
    assert window["nfc_char_count"] == [3.0, 3.0, 1.0]
    assert window["full_logits"] == logits.tolist()
    with pytest.raises(ValueError, match="outside"):
        token_logits_score_window(logits, "abc dé!", labels, [(0, 3), (2, 5), (6, 8)])
    with pytest.raises(ValueError, match="shape"):
        token_logits_score_window(logits[:2], "abc dé!", labels, offsets)


def test_ordered_label_vocabulary_requires_contiguous_unique_rows(tmp_path):
    vocabulary = tmp_path / "model.vcb"
    vocabulary.write_text("0\tO\n1\tB-name\n2\tE-name\n3\tS-email\n", encoding="utf-8")
    assert load_ordered_label_vocabulary(str(vocabulary)) == ["O", "B-name", "E-name", "S-email"]

    vocabulary.write_text("0\tO\n2\tB-name\n3\tE-name\n4\tS-email\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_ordered_label_vocabulary(str(vocabulary))


def test_character_onnx_bundle_requires_one_char_ids_input(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("model.onnx", "character_projection.json"):
        (bundle / name).write_bytes(b"fixture")
    (bundle / "model.vcb").write_text(
        "0\tO\n1\tB-name\n2\tE-name\n3\tS-email\n",
        encoding="utf-8",
    )

    class FakeSession:
        input_names = ["char_ids"]

        def __init__(self, _path, providers):
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [
                SimpleNamespace(name=name, type="tensor(int64)", shape=["batch", "characters"])
                for name in self.input_names
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(
                    name="logits",
                    type="tensor(float)",
                    shape=["batch", "characters", 4],
                )
            ]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    import scripts.pii_character_projection as character_projection

    expected_projection = object()
    monkeypatch.setattr(character_projection, "load_character_projection", lambda _path: expected_projection)

    session, projection, labels = load_character_onnx_bundle(str(bundle))
    assert isinstance(session, FakeSession)
    assert projection is expected_projection
    assert labels == ["O", "B-name", "E-name", "S-email"]

    FakeSession.input_names = ["char_ids", "attention_mask"]
    with pytest.raises(ValueError, match="sole ONNX input 'char_ids'"):
        load_character_onnx_bundle(str(bundle))


def test_character_token_onnx_bundle_requires_exact_three_input_contract(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("model.onnx", "character_projection.json", "tokenizer.json"):
        (bundle / name).write_bytes(b"fixture")
    (bundle / "model.vcb").write_text(
        "0\tO\n1\tB-name\n2\tE-name\n3\tS-email\n",
        encoding="utf-8",
    )

    class FakeSession:
        input_names = ["char_ids", "token_starts", "token_ends"]

        def __init__(self, _path, providers):
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            axes = {
                "char_ids": "characters",
                "token_starts": "tokens",
                "token_ends": "tokens",
            }
            return [
                SimpleNamespace(name=name, type="tensor(int64)", shape=["batch", axes[name]])
                for name in self.input_names
            ]

        def get_outputs(self):
            return [
                SimpleNamespace(
                    name="logits",
                    type="tensor(float)",
                    shape=["batch", "tokens", 4],
                )
            ]

    class FakeTokenizer:
        @staticmethod
        def from_file(path):
            assert path == str(bundle / "tokenizer.json")
            return "tokenizer"

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))
    import scripts.pii_character_projection as character_projection

    expected_projection = object()
    monkeypatch.setattr(character_projection, "load_character_projection", lambda _path: expected_projection)

    session, tokenizer, projection, labels = load_character_token_onnx_bundle(str(bundle))
    assert isinstance(session, FakeSession)
    assert tokenizer == "tokenizer"
    assert projection is expected_projection
    assert labels == ["O", "B-name", "E-name", "S-email"]

    FakeSession.input_names = ["char_ids", "token_starts"]
    with pytest.raises(ValueError, match="expected ONNX inputs"):
        load_character_token_onnx_bundle(str(bundle))


def test_token_score_cache_subset_reindexes_documents_windows_and_tokens():
    np = pytest.importorskip("numpy")
    cache = {
        "schema_version": np.asarray([1], dtype=np.int32),
        "calibration_family": np.asarray(["greedy_global_o_bias"]),
        "labels": np.asarray(["O", "S-email"]),
        "document_ids": np.asarray(["a", "b", "c"]),
        "document_datasets": np.asarray(["x", "y", "z"]),
        "window_document": np.asarray([0, 1, 1, 2], dtype=np.int32),
        "window_token_start": np.asarray([0, 1, 3, 4, 6], dtype=np.int64),
        "token_start": np.arange(6, dtype=np.int32),
        "token_end": np.arange(1, 7, dtype=np.int32),
        "top_non_o_label": np.ones(6, dtype=np.uint16),
        "o_minus_top_logit": np.arange(6, dtype=np.float32),
        "token_logits": np.arange(12, dtype=np.float32).reshape(6, 2),
    }

    subset = subset_token_score_cache(cache, [2, 1])

    assert subset["document_ids"].tolist() == ["c", "b"]
    assert subset["document_datasets"].tolist() == ["z", "y"]
    assert subset["window_document"].tolist() == [0, 1, 1]
    assert subset["window_token_start"].tolist() == [0, 2, 4, 5]
    assert subset["token_start"].tolist() == [4, 5, 1, 2, 3]
    assert subset["token_logits"].tolist() == [
        [8.0, 9.0],
        [10.0, 11.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ]
    with pytest.raises(ValueError, match="must be unique"):
        subset_token_score_cache(cache, [1, 1])
    with pytest.raises(ValueError, match="out of range"):
        subset_token_score_cache(cache, [3])


def test_o_logit_bias_slug_is_stable_and_rejects_lossy_values():
    assert [o_logit_bias_slug(value) for value in (-0.25, 0, 0.25, 0.5)] == [
        "m025",
        "p000",
        "p025",
        "p050",
    ]
    with pytest.raises(ValueError, match="two-decimal"):
        o_logit_bias_slug(0.125)


def test_entity_compat_repairs_mixed_continuations_but_never_explicit_boundaries():
    entity = bucket_compat_map("entity", {"given_name", "family_name", "street_address"})

    adjacent = decode_bioes_labels(
        ["B-given_name", "E-given_name", "B-family_name", "E-family_name"],
        [(0, 2), (2, 4), (4, 6), (6, 8)],
        bucket_of=entity,
        span_type="preponderance",
    )
    assert adjacent == [
        {"start": 0, "end": 4, "label": "given_name"},
        {"start": 4, "end": 8, "label": "family_name"},
    ]

    mixed = ["B-given_name", "E-street_address"]
    offsets = [(0, 2), (2, 4)]
    assert decode_bioes_labels(mixed, offsets, bucket_of=entity) == [
        {"start": 0, "end": 4, "label": "given_name"}
    ]
    assert decode_bioes_labels(
        mixed,
        offsets,
        bucket_of=bucket_compat_map("redaction_9_v1"),
    ) == [
        {"start": 0, "end": 2, "label": "given_name"},
        {"start": 2, "end": 4, "label": "street_address"},
    ]


def test_bucket_compat_projects_a_native_redaction_cut_inventory():
    native_redaction_20 = Tagset().cut_targets("redaction_20_v1")
    redaction_9 = bucket_compat_map("redaction_9_v1", native_redaction_20)

    assert set(redaction_9) == native_redaction_20
    assert redaction_9["family_name"] == "name"
    assert redaction_9["government_id"] == "unique_identifier"
    assert redaction_9["other_sensitive"] == "other_sensitive"


def test_bucket_compat_projects_ontology_v2_families_and_references():
    families = bucket_compat_map(
        "ontology_v2_family",
        {"locality", "organization_reference", "person_name", "person_reference"},
    )

    assert families == {
        "locality": "location",
        "organization_reference": "organization",
        "person_name": "person",
        "person_reference": "person",
    }
    with pytest.raises(ValueError, match="no family"):
        bucket_compat_map("ontology_v2_family", {"not_a_real_type"})


def test_preponderance_weight_changes_only_the_repaired_span_label():
    text = "e\u0301xy"
    labels = ["B-given_name", "I-family_name", "E-family_name"]
    offsets = [(0, 2), (2, 3), (3, 4)]
    entity = bucket_compat_map("entity", {"given_name", "family_name"})

    codepoint = decode_bioes_labels(
        labels,
        offsets,
        bucket_of=entity,
        span_type="preponderance",
        token_weights=preponderance_token_weights(text, offsets, "codepoint"),
    )
    nfc_char = decode_bioes_labels(
        labels,
        offsets,
        bucket_of=entity,
        span_type="preponderance",
        token_weights=preponderance_token_weights(text, offsets, "nfc-char"),
    )

    assert codepoint == [{"start": 0, "end": 4, "label": "given_name"}]
    assert nfc_char == [{"start": 0, "end": 4, "label": "family_name"}]


def test_load_unified_eval_splits_languages_and_preserves_provenance(tmp_path):
    source = tmp_path / "val.jsonl"
    rows = [
        {
            "lang": "en",
            "src": "native",
            "mix_source": "primary",
            "supervision": "complete",
            "text": "Jason",
            "spans": [[0, 5, "given_name"]],
        },
        {
            "lang": "ja",
            "src": "transport",
            "supervision": "complete",
            "text": "東京",
            "spans": [[0, 2, "city"]],
        },
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    shards = load_unified_eval(source, "final20-val")

    assert list(shards) == ["en", "ja"]
    assert shards["en"] == [
        {
            "id": "final20-val:0",
            "text": "Jason",
            "spans": [{"start": 0, "end": 5, "type": "given_name"}],
            "meta": {
                "lang": "en",
                "src": "native",
                "mix_source": "primary",
                "supervision": "complete",
            },
        }
    ]
    assert shards["ja"][0]["id"] == "final20-val:1"


def test_load_unified_eval_rejects_partial_supervision(tmp_path):
    source = tmp_path / "val.jsonl"
    source.write_text(
        json.dumps(
            {
                "lang": "en",
                "supervision": "annotated_spans_only",
                "text": "Acme",
                "spans": [[0, 4, "organization"]],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="complete supervision"):
        load_unified_eval(source, "val")


def test_sparse_token_score_cache_redecodes_o_bias_without_logits(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-person_name", "E-person_name", "S-organization"]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0, 6, 12],
                    "token_end": [5, 11, 16],
                    "top_non_o_label": [1, 2, 3],
                    "o_minus_top": [-0.4, -0.2, 0.1],
                    "top3_non_o_label": [[1, 2, 3], [2, 1, 3], [3, 1, 2]],
                    "o_minus_top3": [[-0.4, 0.5, 1.0], [-0.2, 0.4, 0.8], [0.1, 0.5, 0.9]],
                    "nfc_char_count": [5, 5, 4],
                }
            ]
        ],
    )

    cache = load_token_score_cache(cache_path)
    default = decode_token_score_cache(cache, 0.0)
    conservative = decode_token_score_cache(cache, 0.5)

    assert default == [
        {
            "id": "doc-1",
            "dataset": "cal-en",
            "preds": [{"start": 0, "end": 11, "label": "person_name"}],
        }
    ]
    assert conservative == [{"id": "doc-1", "dataset": "cal-en", "preds": []}]


def test_legacy_token_cache_reconstructs_nfc_vote_mass_from_gold(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-given_name", "I-family_name", "E-family_name"]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0, 2, 3],
                    "token_end": [2, 3, 4],
                    "top_non_o_label": [1, 2, 3],
                    "o_minus_top": [-1.0, -1.0, -1.0],
                    "top3_non_o_label": [[1, 2, 3], [2, 3, 1], [3, 2, 1]],
                    "o_minus_top3": [[-1.0, 0.0, 0.0]] * 3,
                    "nfc_char_count": [1, 1, 1],
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)
    cache.pop("token_nfc_char_count")
    entity = bucket_compat_map("entity", {"given_name", "family_name"})

    assert _cache_document_texts(cache, [{"id": "doc-1", "text": "e\u0301xy"}]) == ["e\u0301xy"]
    assert decode_token_score_cache(
        cache,
        0.0,
        bucket_of=entity,
        span_type="preponderance",
        preponderance_weight="nfc-char",
        document_texts=["e\u0301xy"],
    )[0]["preds"] == [{"start": 0, "end": 4, "label": "family_name"}]


def test_full_logit_cache_supports_exact_bucket_viterbi_and_top_k_study(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    logits = [[-10.0] * len(labels) for _ in range(3)]
    for row in logits:
        row[0] = 0.0
    logits[0][1] = 5.0
    logits[1][5] = 5.0
    logits[1][6] = 4.0
    logits[2][3] = 5.0
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0, 1, 2],
                    "token_end": [1, 2, 3],
                    "top_non_o_label": [1, 5, 3],
                    "o_minus_top": [-5.0, -5.0, -5.0],
                    "top3_non_o_label": [[1, 2, 3], [5, 6, 2], [3, 2, 1]],
                    "o_minus_top3": [[-5.0, 10.0, 10.0], [-5.0, -4.0, 10.0], [-5.0, 10.0, 10.0]],
                    "nfc_char_count": [1, 1, 1],
                    "full_logits": logits,
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)
    entity = bucket_compat_map("entity", {"name", "city"})

    exact = decode_token_score_cache(
        cache,
        0.0,
        bucket_of=entity,
        span_type="preponderance",
        preponderance_weight="nfc-char",
        bioes_search="viterbi-max",
    )
    top2 = decode_token_score_cache(
        cache,
        0.0,
        bucket_of=entity,
        span_type="preponderance",
        preponderance_weight="nfc-char",
        bioes_search="viterbi-max",
        bioes_search_top_k=2,
    )

    assert cache["token_logits"].shape == (3, len(labels))
    assert exact[0]["preds"] == [{"start": 0, "end": 3, "label": "name"}]
    assert top2 == exact


def test_full_logit_cache_supports_lazy_fine_legal_projection(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    logits = [
        [0.0, 5.0, -10.0, -10.0, 4.0],
        [5.0, -10.0, -10.0, 4.5, 0.0],
    ]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0, 1],
                    "token_end": [1, 2],
                    "top_non_o_label": [1, 3],
                    "o_minus_top": [-5.0, 0.5],
                    "top3_non_o_label": [[1, 4, 2], [3, 4, 1]],
                    "o_minus_top3": [[-5.0, -4.0, 10.0], [0.5, 5.0, 15.0]],
                    "nfc_char_count": [1, 1],
                    "full_logits": logits,
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)

    decoded = decode_token_score_cache(cache, 0.0, bioes_search="lazy-fine-legal")

    assert decoded[0]["preds"] == [{"start": 0, "end": 2, "label": "name"}]
    with pytest.raises(ValueError, match="without bucket compatibility"):
        decode_token_score_cache(
            cache,
            0.0,
            bucket_of={"name": "entity"},
            bioes_search="lazy-fine-legal",
        )


def test_unicode_word_boundaries_preserve_apostrophes_and_split_punctuation():
    assert unicode_word_boundary_positions("John O'Connor.") == frozenset({0, 4, 5, 13})


def test_tokens_cover_word_boundaries_that_fall_inside_their_offsets():
    starts, ends = covered_word_boundary_offsets([0, 2], [2, 5], {0, 4})

    assert starts == [0, 4]
    assert ends == [None, 4]


def test_whitespace_split_boundary_includes_attached_punctuation():
    assert unicode_whitespace_split_boundaries(
        "William\u00a0Tambellini",
        [0, 8],
        [7, 18],
    ) == [True]
    assert unicode_whitespace_split_boundaries(
        "William, Tambellini",
        [0, 9],
        [8, 19],
    ) == [False]
    assert unicode_whitespace_split_boundaries("WilliamTambellini", [0, 7], [7, 17]) == [False]
    assert unicode_whitespace_split_boundaries("مشهد يظهر", [0, 0], [1, 4]) == [False]


def test_whitespace_split_cost_and_name_oracle_adjust_final_inference(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-person_name", "I-person_name", "E-person_name", "S-person_name"]
    text = "Please call William Tambellini"
    logits = [
        [0.0, 7.5, -20.0, -20.0, 10.0],
        [0.0, -20.0, -20.0, 7.5, 10.0],
    ]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [12, 20],
                    "token_end": [19, 30],
                    "top_non_o_label": [4, 4],
                    "o_minus_top": [-10.0, -10.0],
                    "top3_non_o_label": [[4, 1, 2], [4, 3, 1]],
                    "o_minus_top3": [[-10.0, -7.5, 20.0], [-10.0, -7.5, 20.0]],
                    "nfc_char_count": [7, 10],
                    "full_logits": logits,
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)

    below_crossing = decode_token_score_cache(
        cache,
        0.0,
        document_texts=[text],
        bioes_search="lazy-fine-legal",
        bioes_whitespace_split_cost=4.0,
    )
    at_crossing = decode_token_score_cache(
        cache,
        0.0,
        document_texts=[text],
        bioes_search="lazy-fine-legal",
        bioes_whitespace_split_cost=5.0,
    )

    assert below_crossing[0]["preds"] == [
        {"start": 12, "end": 19, "label": "person_name"},
        {"start": 20, "end": 30, "label": "person_name"},
    ]
    assert at_crossing[0]["preds"] == [{"start": 12, "end": 30, "label": "person_name"}]

    punctuated_cache = {key: value.copy() for key, value in cache.items()}
    punctuated_cache["token_start"][:] = [12, 21]
    punctuated_cache["token_end"][:] = [20, 31]
    punctuated = decode_token_score_cache(
        punctuated_cache,
        0.0,
        document_texts=["Please call William, Tambellini"],
        bioes_search="lazy-fine-legal",
        bioes_whitespace_split_cost=5.0,
    )
    assert punctuated[0]["preds"] == [
        {"start": 12, "end": 20, "label": "person_name"},
        {"start": 21, "end": 31, "label": "person_name"},
    ]

    root = Path(__file__).resolve().parents[2]
    oracle = NameAnnotationQc(
        load_name_qc_config(root / "scripts" / "pii_name_annotation_qc_profiles_v2.json")
    )
    adjusted = apply_name_oracle_to_rows(at_crossing, [text], ["en-US"], oracle)

    assert adjusted[0]["preds"] == [{"start": 12, "end": 30, "label": "person_name"}]
    assert adjusted[0]["subclass_spans"] == [
        {
            "carrier_start": 12,
            "carrier_end": 30,
            "type": "person_name",
            "start": 12,
            "end": 19,
            "family": "name_component",
            "value": "given_name",
        },
        {
            "carrier_start": 12,
            "carrier_end": 30,
            "type": "person_name",
            "start": 20,
            "end": 30,
            "family": "name_component",
            "value": "family_name",
        },
    ]
    receipt = dict(adjusted[0]["name_oracle"])
    # The digest names the profile file these kinds came from, so it changes whenever
    # that file is edited. Pin its shape; pinning one value would date the test.
    config_sha256 = receipt.pop("config_sha256")
    assert len(config_sha256) == 64 and set(config_sha256) <= set("0123456789abcdef")
    assert receipt == {
        "mode": "insert-name-kinds",
        "max_name_gap_chars": 5,
        "apply_name_grammar": True,
        "override_name_kinds": False,
        "profile": "given_first",
        "language_rule": "en",
        "status": "proposed",
        "review_required": 0,
    }


def test_name_oracle_language_alignment_accepts_override_and_row_tags():
    np = pytest.importorskip("numpy")
    cache = {"document_ids": np.asarray(["a", "b"])}
    gold = [{"id": "a", "bcp47": "en"}, {"id": "b", "bcp47": "ja-JP"}]

    assert _cache_document_languages(cache, gold) == ["en", "ja-JP"]
    assert _cache_document_languages(cache, gold, "fr") == ["fr", "fr"]


def test_full_logit_cache_supports_word_constrained_fine_projection(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    logits = [
        [0.0, 4.0, -10.0, -10.0, 10.0],
        [0.0, -10.0, -10.0, 4.0, -10.0],
    ]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0, 2],
                    "token_end": [2, 4],
                    "top_non_o_label": [4, 3],
                    "o_minus_top": [-10.0, -4.0],
                    "top3_non_o_label": [[4, 1, 2], [3, 4, 1]],
                    "o_minus_top3": [[-10.0, -4.0, 10.0], [-4.0, 10.0, 10.0]],
                    "nfc_char_count": [2, 2],
                    "full_logits": logits,
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)

    decoded = decode_token_score_cache(
        cache,
        0.0,
        document_texts=["John"],
        bioes_search="word-fine-legal",
    )

    assert decoded[0]["preds"] == [{"start": 0, "end": 4, "label": "name"}]
    with pytest.raises(ValueError, match="requires document_texts"):
        decode_token_score_cache(cache, 0.0, bioes_search="word-fine-legal")


def test_word_constrained_projection_realizes_a_boundary_inside_a_token(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = ["O", "B-name", "I-name", "E-name", "S-name"]
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0],
                    "token_end": [5],
                    "top_non_o_label": [4],
                    "o_minus_top": [-10.0],
                    "top3_non_o_label": [[4, 1, 2]],
                    "o_minus_top3": [[-10.0, 10.0, 10.0]],
                    "nfc_char_count": [5],
                    "full_logits": [[0.0, -10.0, -10.0, -10.0, 10.0]],
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)

    decoded = decode_token_score_cache(
        cache,
        0.0,
        document_texts=["John."],
        bioes_search="word-fine-legal",
    )

    assert decoded[0]["preds"] == [{"start": 0, "end": 4, "label": "name"}]


def test_full_logit_cache_supports_temperature_bucket_reduction(tmp_path):
    cache_path = tmp_path / "scores.npz"
    labels = [
        "O",
        "B-name",
        "I-name",
        "E-name",
        "S-name",
        "B-city",
        "I-city",
        "E-city",
        "S-city",
    ]
    logits = [[-10.0] * len(labels)]
    logits[0][0] = 1.3
    logits[0][4] = 1.0
    logits[0][8] = 1.0
    write_token_score_cache(
        cache_path,
        labels,
        ["doc-1"],
        ["cal-en"],
        [
            [
                {
                    "token_start": [0],
                    "token_end": [1],
                    "top_non_o_label": [4],
                    "o_minus_top": [0.3],
                    "top3_non_o_label": [[4, 8, 1]],
                    "o_minus_top3": [[0.3, 0.3, 11.3]],
                    "nfc_char_count": [1],
                    "full_logits": logits,
                }
            ]
        ],
    )
    cache = load_token_score_cache(cache_path)
    entity = bucket_compat_map("entity", {"name", "city"})

    maximum = decode_token_score_cache(
        cache,
        0.0,
        bucket_of=entity,
        span_type="preponderance",
        bioes_search="viterbi-max",
    )
    soft_mass = decode_token_score_cache(
        cache,
        0.0,
        bucket_of=entity,
        span_type="preponderance",
        bioes_search="viterbi-max",
        bioes_bucket_reduction="logsumexp",
        bioes_bucket_temperature=1.0,
    )

    assert maximum[0]["preds"] == []
    assert soft_mass[0]["preds"] == [{"start": 0, "end": 1, "label": "name"}]


def test_span_score_cache_redecodes_threshold_and_keeps_documents(tmp_path):
    cache_path = tmp_path / "span-scores.npz"
    write_span_score_cache(
        cache_path,
        ["doc-1", "doc-2"],
        ["cal-en", "cal-en"],
        [
            [
                {"start": 0, "end": 5, "label": "first_name", "confidence": 0.8},
                {"start": 8, "end": 12, "label": "city", "confidence": 0.4},
            ],
            [],
        ],
    )

    cache = load_span_score_cache(cache_path)

    assert decode_span_score_cache(cache, 0.5) == [
        {
            "id": "doc-1",
            "dataset": "cal-en",
            "preds": [{"start": 0, "end": 5, "label": "first_name"}],
        },
        {"id": "doc-2", "dataset": "cal-en", "preds": []},
    ]
    assert decode_span_score_cache(cache, 0.9) == [
        {"id": "doc-1", "dataset": "cal-en", "preds": []},
        {"id": "doc-2", "dataset": "cal-en", "preds": []},
    ]


def test_score_preserves_legacy_regions_and_adds_typed_pairwise_metrics():
    gold = [
        {
            "id": "one",
            "text": " Alice 4111 ",
            "spans": [
                {"start": 1, "end": 6, "type": "first_name"},
                {"start": 7, "end": 11, "type": "credit_debit_card"},
            ],
        }
    ]
    pred = [
        {
            "id": "one",
            "preds": [
                {"start": 0, "end": 7, "label": "FIRSTNAME"},
                {"start": 7, "end": 11, "label": "CREDITCARD"},
                {"start": 7, "end": 11, "label": "CREDITCARD"},
            ],
        }
    ]
    score = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="nemotron_pii",
        pred_schema="openmed_54",
    )
    assert score["n_pred"] == 1  # legacy whitespace-joined redaction region
    individual = score["individual_span_one_to_one"]
    assert individual["n_pred"] == 3
    assert individual["class_agnostic_exact"]["F1"] == 0.8
    assert individual["annotated_recall_by_gold_type"] == {
        "credit_debit_card": {
            "n": 1,
            "exact": 1,
            "overlap": 1,
            "exact_R": 1.0,
            "overlap_R": 1.0,
        },
        "first_name": {
            "n": 1,
            "exact": 1,
            "overlap": 1,
            "exact_R": 1.0,
            "overlap_R": 1.0,
        },
    }
    typed = individual["typed_fine_intersection"]
    assert typed["universe_size"] == 30
    assert typed["exact"]["F1"] == 0.8
    assert typed["overlap"]["F1"] == 0.8
    assert typed["excluded_gold"] == typed["excluded_pred"] == 0


def test_character_redaction_gives_proportional_credit_to_boundary_near_miss():
    gold = [
        {
            "id": "one",
            "text": "abcdefghijXYZ",
            "spans": [{"start": 0, "end": 10, "type": "first_name"}],
        }
    ]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 8, "label": "first_name"}]}]

    individual = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="authored_v1",
        pred_schema="authored_v1",
    )["individual_span_one_to_one"]

    assert individual["class_agnostic_exact"]["tp"] == 0
    assert individual["class_agnostic_overlap"]["tp"] == 1
    assert individual["character_redaction"] == {
        "tp": 8,
        "fp": 0,
        "fn": 2,
        "tn": 3,
        "wrong_type": 0,
        "correct": 11,
        "n_pred": 8,
        "n_gold": 10,
        "n_total": 13,
        "excluded": 0,
        "P": 1.0,
        "R": 0.8,
        "F1": 0.8889,
        "accuracy": 0.8462,
        "specificity": 1.0,
        "balanced_accuracy": 0.9,
    }


def test_character_labels_score_wrong_fine_type_but_accept_shared_coarse_type():
    gold = [
        {
            "id": "one",
            "text": "abcdefghijXYZ",
            "spans": [{"start": 0, "end": 10, "type": "first_name"}],
        }
    ]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 10, "label": "last_name"}]}]

    individual = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="authored_v1",
        pred_schema="authored_v1",
        ontology_cuts=("redaction_20_v1", "redaction_9_v1"),
    )["individual_span_one_to_one"]

    assert individual["character_redaction"]["accuracy"] == 1.0
    fine = individual["typed_fine_intersection"]["character_labels"]
    assert fine["wrong_type"] == 10
    assert fine["accuracy"] == 0.2308
    assert fine["F1"] == 0.0
    assert individual["typed_redaction_9_v1_intersection"]["character_labels"]["accuracy"] == 1.0


def test_typed_character_metrics_can_be_skipped_for_overlapping_gold():
    gold = [
        {
            "id": "one",
            "text": "Alice",
            "spans": [
                {"start": 0, "end": 5, "type": "last_name"},
                {"start": 0, "end": 3, "type": "first_name"},
            ],
        }
    ]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 5, "label": "last_name"}]}]

    with pytest.raises(ValueError, match="conflicting labels"):
        score_one(
            gold,
            pred,
            tagset=Tagset(),
            gold_schema="authored_v1",
            pred_schema="authored_v1",
        )

    individual = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="authored_v1",
        pred_schema="authored_v1",
        character_label_projections=(),
    )["individual_span_one_to_one"]

    assert individual["class_agnostic_exact"]["tp"] == 1
    assert individual["character_redaction"]["tp"] == 5
    assert individual["typed_fine_intersection"]["exact"]["tp"] == 1
    assert "character_labels" not in individual["typed_fine_intersection"]


def test_typed_intersection_excludes_inexpressible_spans_and_fails_unknown():
    gold = [{"id": "one", "text": "Peru", "spans": [{"start": 0, "end": 4, "type": "country"}]}]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 4, "label": "ORDINALDIRECTION"}]}]
    typed = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="nemotron_pii",
        pred_schema="openmed_54",
    )["individual_span_one_to_one"]["typed_fine_intersection"]
    assert typed["excluded_gold"] == typed["excluded_pred"] == 1
    assert typed["n_gold"] == typed["n_pred"] == 0

    pred[0]["preds"][0]["label"] = "COMPANYNAME"
    with pytest.raises(ValueError, match="openmed_54: unknown label"):
        score_one(
            gold,
            pred,
            tagset=Tagset(),
            gold_schema="nemotron_pii",
            pred_schema="openmed_54",
        )


def test_canonicalized_predictions_keep_gold_types_absent_from_source_schema():
    tagset = Tagset()
    gold = [
        {
            "id": "one",
            "text": "asthma Acme",
            "spans": [
                {"start": 0, "end": 6, "type": "medical_condition"},
                {"start": 7, "end": 11, "type": "company_name"},
            ],
        }
    ]
    source_predictions = [
        {
            "id": "one",
            "preds": [{"start": 7, "end": 11, "label": "company_name"}],
        }
    ]
    canonical_predictions = _canonicalize_prediction_labels(
        source_predictions,
        "openmed_nemotron_55",
        tagset,
    )

    assert source_predictions[0]["preds"][0]["label"] == "company_name"
    assert canonical_predictions[0]["preds"][0]["label"] == "organization"
    typed = score_one(
        gold,
        canonical_predictions,
        tagset=tagset,
        gold_schema="authored_v1",
        pred_schema="canonical",
    )["individual_span_one_to_one"]["typed_fine_intersection"]
    assert typed["universe_size"] == len(tagset.schema_image("authored_v1"))
    assert typed["n_gold"] == 2
    assert typed["n_pred"] == 1
    assert typed["overlap"]["F1"] == 0.6667


def test_cut_projection_expands_only_the_declared_common_schema_universe():
    gold = [
        {
            "id": "one",
            "text": "asthma",
            "spans": [{"start": 0, "end": 6, "type": "medical_condition"}],
        }
    ]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 6, "label": "medical_condition"}]}]
    individual = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="authored_v1",
        pred_schema="canonical",
        ontology_cuts=("redaction_20_v1", "redaction_9_v1"),
        comparison_schemas=("openmed_nemotron_55",),
    )["individual_span_one_to_one"]

    fine = individual["typed_fine_intersection"]
    assert fine["comparison_schemas"] == ["openmed_nemotron_55"]
    assert fine["excluded_gold"] == fine["excluded_pred"] == 1

    coarse20 = individual["typed_redaction_20_v1_intersection"]
    assert coarse20["universe_size"] == 17
    assert coarse20["excluded_gold"] == coarse20["excluded_pred"] == 0
    assert coarse20["overlap"]["F1"] == 1.0


def test_native_cut_predictions_score_geometry_and_valid_cuts_without_fine_projection():
    gold = [
        {
            "id": "one",
            "text": "Kim",
            "spans": [{"start": 0, "end": 3, "type": "last_name"}],
        }
    ]
    pred = [{"id": "one", "preds": [{"start": 0, "end": 3, "label": "family_name"}]}]
    individual = score_one(
        gold,
        pred,
        tagset=Tagset(),
        gold_schema="authored_v1",
        pred_schema="redaction_20_v1",
        ontology_cuts=("redaction_20_v1", "redaction_9_v1"),
        comparison_schemas=("authored_v1",),
    )["individual_span_one_to_one"]

    assert "typed_fine_intersection" not in individual
    assert individual["class_agnostic_overlap"]["F1"] == 1.0
    assert individual["typed_redaction_20_v1_intersection"]["overlap"]["F1"] == 1.0
    assert individual["typed_redaction_9_v1_intersection"]["overlap"]["F1"] == 1.0


def test_load_mapa_test_preserves_document_units_and_source_labels(tmp_path):
    shard = tmp_path / "test" / "cs.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "cs:doc:1",
        "text": "Jane Prague",
        "spans": [
            {
                "start": 0,
                "end": 4,
                "label": "person_name",
                "source_label": "PERSON",
            }
        ],
        "lang": "cs",
        "metadata": {"file_name": "doc"},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")

    assert load_mapa_test(tmp_path, ["cs"]) == {
        "mapa-cs-test": [
            {
                "id": "cs:doc:1",
                "document_id": "cs:doc",
                "text": "Jane Prague",
                "spans": [{"start": 0, "end": 4, "type": "PERSON"}],
            }
        ]
    }


def test_load_idner_news_splits_preserves_upstream_split_and_source_labels(tmp_path):
    shard = tmp_path / "validation" / "id.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "id:validation:1",
        "text": "Joko di Jakarta",
        "spans": [
            {
                "start": 0,
                "end": 4,
                "label": "person_name",
                "source_label": "PER",
            },
            {
                "start": 8,
                "end": 15,
                "label": "location",
                "source_label": "LOC",
            },
        ],
        "lang": "id",
        "metadata": {"sentence_number": 1},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")

    assert load_idner_news_splits(tmp_path, ["validation"]) == {
        "idner-validation": [
            {
                "id": "id:validation:1",
                "text": "Joko di Jakarta",
                "spans": [
                    {"start": 0, "end": 4, "type": "PER"},
                    {"start": 8, "end": 15, "type": "LOC"},
                ],
            }
        ]
    }


def test_load_hiner_splits_preserves_upstream_split_and_source_labels(tmp_path):
    shard = tmp_path / "validation" / "hi.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "hi:validation:1",
        "text": "भारत में हिन्दी",
        "spans": [
            {
                "start": 0,
                "end": 4,
                "label": "location",
                "source_label": "LOCATION",
            },
            {
                "start": 8,
                "end": 14,
                "label": "language_spoken",
                "source_label": "LANGUAGE",
            },
        ],
        "lang": "hi",
        "metadata": {"sentence_number": 1},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")

    assert load_hiner_splits(tmp_path, ["validation"]) == {
        "hiner-validation": [
            {
                "id": "hi:validation:1",
                "text": "भारत में हिन्दी",
                "spans": [
                    {"start": 0, "end": 4, "type": "LOCATION"},
                    {"start": 8, "end": 14, "type": "LANGUAGE"},
                ],
            }
        ]
    }


def test_load_aqmar_splits_preserves_corrected_source_labels(tmp_path):
    shard = tmp_path / "test" / "ar.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "ar:aqmar:test:7",
        "text": "أحمد في القاهرة",
        "spans": [
            {
                "start": 0,
                "end": 4,
                "label": "person_name",
                "source_label": "PER",
            },
            {
                "start": 8,
                "end": 15,
                "label": "location",
                "source_label": "LOC",
            },
        ],
        "lang": "ar",
        "metadata": {"bio_provenance": "Liu_et_al_2019_corrected_via_OpenNER_1.0"},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")

    assert load_aqmar_splits(tmp_path, ["test"]) == {
        "aqmar-test": [
            {
                "id": "ar:aqmar:test:7",
                "text": "أحمد في القاهرة",
                "spans": [
                    {"start": 0, "end": 4, "type": "PER"},
                    {"start": 8, "end": 15, "type": "LOC"},
                ],
            }
        ]
    }


def test_load_openner_core_splits_preserves_language_and_source_labels(tmp_path):
    shard = tmp_path / "validation" / "ja.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "ja:openner:japanese_gsd-jap:validation:3",
        "text": "東京 の 太郎",
        "spans": [
            {"start": 0, "end": 2, "label": "location", "source_label": "LOC"},
            {"start": 5, "end": 7, "label": "person_name", "source_label": "PER"},
        ],
        "lang": "ja",
        "metadata": {"source_component": "Japanese GSD NER"},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    assert load_openner_core_splits(tmp_path, ["validation"], ["ja"]) == {
        "openner-ja-validation": [
            {
                "id": "ja:openner:japanese_gsd-jap:validation:3",
                "text": "東京 の 太郎",
                "spans": [
                    {"start": 0, "end": 2, "type": "LOC"},
                    {"start": 5, "end": 7, "type": "PER"},
                ],
            }
        ]
    }


def test_load_wojood_splits_flattens_nested_spans_longest_first(tmp_path):
    shard = tmp_path / "validation" / "ar.jsonl.gz"
    shard.parent.mkdir()
    row = {
        "id": "ar:validation:1",
        "text": "جامعة بيرزيت 2025",
        "spans": [
            {
                "start": 0,
                "end": 12,
                "label": "organization",
                "source_label": "ORG",
            },
            {
                "start": 6,
                "end": 12,
                "label": "location",
                "source_label": "GPE",
            },
            {
                "start": 13,
                "end": 17,
                "label": "date",
                "source_label": "DATE",
            },
        ],
        "lang": "ar",
        "metadata": {"sentence_number": 1},
    }
    with gzip.open(shard, "wt", encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")

    assert load_wojood_splits(tmp_path, ["validation"]) == {
        "wojood-validation": [
            {
                "id": "ar:validation:1",
                "text": "جامعة بيرزيت 2025",
                "spans": [
                    {"start": 0, "end": 12, "type": "ORG"},
                    {"start": 13, "end": 17, "type": "DATE"},
                ],
            }
        ]
    }


def test_score_alignment_reorders_by_id_and_rejects_implicit_prefix():
    gold = [{"id": "a"}, {"id": "b"}]
    assert _align_gold_to_predictions(gold, [{"id": "b"}, {"id": "a"}]) == [
        {"id": "b"},
        {"id": "a"},
    ]
    with pytest.raises(ValueError, match="1 missing predictions"):
        _align_gold_to_predictions(gold, [{"id": "a"}])
    assert _align_gold_to_predictions(
        gold,
        [{"id": "a"}],
        allow_prediction_subset=True,
    ) == [{"id": "a"}]
