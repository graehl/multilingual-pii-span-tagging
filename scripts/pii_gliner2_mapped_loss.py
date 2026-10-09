"""Partial-label supervision for GLiNER2's independent Bernoulli span scores.

A positive span requires at least one accepted type. Unknown-coverage types
receive no negative supervision outside annotated spans. Singleton targets
recover the stock binary loss, including its negative-span sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def window_targets(row, window, splitter):
    """Carry the original map through the unchanged admitted word windows."""
    offset = window["provenance"]["window_start"]
    stop = window["provenance"]["window_end"]
    tokens = list(splitter(window["input"]))
    groups = []
    seen = set()
    for start, end, accepted in row["_mapped_targets"]:
        if not (offset <= start and end <= stop):
            continue
        positions = [i for i, (_, a, b) in enumerate(tokens) if a < end - offset and start - offset < b]
        if not positions:
            raise ValueError("mapped target disappeared from admitted window")
        key = (positions[0], positions[-1], tuple(accepted))
        if key not in seen:
            groups.append({"start": key[0], "end": key[1], "labels": accepted})
            seen.add(key)
    return {
        "groups": groups,
        "unknown_types": row.get("unknown_primary_types", []),
        "complete": (row.get("supervision") or "complete") == "complete",
    }


def acceptable_span_loss(scores, groups, known_negative, valid_spans, *, masking_rate=0.0):
    """Sum binary negatives and one Bernoulli-union positive per gold span."""
    if scores.ndim != 4 or scores.shape[0] != 1:
        raise ValueError("mapped entity loss requires one entity structure")
    positive = torch.zeros_like(scores, dtype=torch.bool)
    terms = []
    for group in groups:
        start, end, labels = group["start"], group["end"], group["labels"]
        width = end - start
        if not labels or not (0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]):
            raise ValueError("unreachable acceptable-label target")
        positive[0, labels, start, width] = True
        logits = scores[0, labels, start, width].float()
        # Disjoint first-success events avoid cancellation when every p is tiny.
        log_failure = F.logsigmoid(-logits)
        prefix_failure = torch.cat((logits.new_zeros(1), log_failure.cumsum(0)[:-1]))
        terms.append(-torch.logsumexp(F.logsigmoid(logits) + prefix_failure, dim=0))
    negative = ~positive
    if masking_rate:
        negative = negative & (torch.rand_like(scores) >= masking_rate)
    negative = negative & torch.tensor(known_negative, device=scores.device)[None, :, None, None]
    negative = negative & valid_spans.reshape(1, 1, *scores.shape[2:])
    loss = (F.softplus(scores.float()) * negative).sum()
    return loss + (torch.stack(terms).sum() if terms else scores.sum() * 0)
