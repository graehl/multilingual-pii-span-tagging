import numpy as np

from scripts.pii_bioes import constrained_backoff_bioes_decode, interpolate_bioes_sibling_scores
from scripts.pii_eval import decode_token_score_cache, load_token_score_cache, write_token_score_cache


def test_cached_backoff_uses_exact_fine_path_and_exposes_endpoint_ties(tmp_path):
    labels = ["O"] + [f"{p}-{t}" for t in ("generic", "healthcare") for p in "BIES"]
    logits = np.full((3, len(labels)), -6.0)
    logits[:, 0] = 2.5
    for i, (prefix, kind) in enumerate(zip("BIE", ("healthcare", "generic", "healthcare"))):
        logits[i, labels.index(f"{prefix}-{kind}")] = 4
    path = tmp_path / "cache.npz"
    winners = logits[:, 1:].argmax(axis=1) + 1
    top3 = np.argsort(logits[:, 1:], axis=1)[:, -3:][:, ::-1] + 1
    write_token_score_cache(
        path,
        labels,
        ["example"],
        ["test"],
        [
            [
                {
                    "token_start": [0, 3, 8],
                    "token_end": [2, 7, 16],
                    "top_non_o_label": winners.tolist(),
                    "o_minus_top": [-1.5] * 3,
                    "top3_non_o_label": top3.tolist(),
                    "o_minus_top3": (2.5 - np.take_along_axis(logits, top3, axis=1)).tolist(),
                    "nfc_char_count": [2, 4, 8],
                    "full_logits": logits.tolist(),
                }
            ]
        ],
    )
    cache = load_token_score_cache(path)
    kwargs = dict(
        bucket_of={"generic": "org", "healthcare": "org"},
        span_type="preponderance",
        preponderance_weight="nfc-char",
        bioes_search="viterbi-backoff",
        bioes_bucket_temperature=0.5,
    )
    basic = decode_token_score_cache(cache, 0, bioes_backoff_alpha=0, **kwargs)
    backed = decode_token_score_cache(cache, 0, bioes_backoff_alpha=0.95, **kwargs)
    tied = decode_token_score_cache(cache, 0, bioes_backoff_alpha=1, **kwargs)
    assert basic[0]["preds"] == []
    assert backed[0]["preds"] == [{"start": 0, "end": 16, "label": "healthcare"}]
    assert tied[0]["preds"] == [{"start": 0, "end": 16, "label": "generic"}]


def test_backoff_normalization_shift_invariance_and_validation():
    labels = dict(enumerate(["O"] + [f"{p}-{t}" for t in ("a", "b") for p in "BIES"]))
    buckets = {"a": "group", "b": "group"}
    scores = np.random.default_rng(9).normal(size=(4, len(labels)))
    blended = interpolate_bioes_sibling_scores(scores, labels, buckets, alpha=0.7, temperature=0.5)
    shifted = interpolate_bioes_sibling_scores(scores + 17, labels, buckets, alpha=0.7, temperature=0.5)
    np.testing.assert_allclose(shifted, blended + 17)
    np.testing.assert_array_equal(blended[:, 0], scores[:, 0])
    same = np.full_like(scores, 3.0)
    np.testing.assert_allclose(interpolate_bioes_sibling_scores(same, labels, buckets, alpha=1), same)
    for alpha in (-0.1, 1.1, np.nan):
        with np.testing.assert_raises(ValueError):
            constrained_backoff_bioes_decode(scores, labels, buckets, alpha=alpha)
