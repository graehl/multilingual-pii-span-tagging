# Name-kind character model

Status: **terminal inference model; recipe selection is closed and CPU ONNX
parity is accepted.** The final fit consumed the former train, development,
and test partitions, so only the pre-final-fit scores below are
generalization evidence.

## Purpose and boundary

This small character model supplies lexical evidence that one component inside
a complete `person_name` carrier is a `given_name` or `family_name`. It does
not find the carrier and does not independently predict `middle_name` or
`neither`. The language-configurable grammar owns carrier coalescing,
punctuation, family/given order, consecutive middle or other material,
initials, particles, and the final legal component sequence.

The model can be used after primary-span decoding or as the initialized
character branch of a jointly trained token model. In the integrated design,
the CNN and its subtype head retain these trained weights; only the new affine
map into XLM-R residuals is random-initialized. A direct subtype loss remains
active only where the gold carrier is `person_name`.

A surface does not have one intrinsic role: many names are legitimately used
both ways. Globally ambiguous normalized surfaces are excluded from role
supervision rather than assigned an arbitrary target.

Implementation: [trainer and evaluator](../../../../../scripts/pii_name_role_char_pilot.py).
Terminal fit/export: [publisher](../../../../../scripts/pii_name_role_publish.py).
Grammar/QC configuration:
[BCP 47 profiles](../../../../../scripts/pii_name_annotation_qc_profiles_v2.json).

## ONNX inputs and output

The `name-kind.onnx` graph has dynamic batch size and these fixed
non-batch dimensions:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `char_ids` | `int64` | `[batch, 48]` | Normalized component encoded with the ordered exported vocabulary |
| `language_probs` | `float32` | `[batch, 35]` | Fuzzy base-language distribution; all zero means unavailable |
| `mixture_alpha` | `float32` | `[batch, 1]` | Per-row shared-versus-language-conditioned probability mass |
| `log_probabilities` | `float32` | `[batch, 2]` | Log probabilities ordered `given`, `family` |

With the ordinary `mixture_alpha=0.2`, output probabilities are

```text
0.2 * p(role | surface) + 0.8 * p(role | surface, language distribution).
```

The configured language order is `ar`, `bn`, `cs`, `da`, `de`, `el`, `en`,
`es`, `fa`, `fi`, `fil`, `fr`, `he`, `hi`, `hr`, `id`, `it`, `ja`, `ko`,
`ms`, `nl`, `no`, `pl`, `pt`, `ro`, `ru`, `sv`, `ta`, `te`, `th`, `tr`,
`uk`, `ur`, `vi`, `zh`. These are base BCP 47 language tags; locale-specific
extensions belong to the grammar sidecar. There is no nation input.

## Text normalization and character alphabet

Apply Unicode NFKC, replace every Unicode-whitespace run with one ASCII space,
and trim edge whitespace. Split identity additionally case-folds that form,
but model input preserves case. The training/export environment uses Python
Unicode 15.1. Informative mixed case such as `DeMarcus` and internal
punctuation such as the apostrophe in `O'Connor` remain input features.

Publisher-wide uppercase from Census and INSEE sources is matched to mixed-case
observations from other sources when possible and otherwise title-cased before
augmentation. Raw publisher uppercase is not treated as natural casing.
Training samples natural/reconstructed, uppercase, and lowercase forms with
weights `0.7 / 0.2 / 0.1`.

The ordered vocabulary has 3,910 entries:

| Alphabet class | Count | Treatment |
|---|---:|---|
| Special tokens | 4 | `<PAD>`, `<BOS>`, `<EOS>`, `<TRUNC>` |
| Literal codepoints | 3,437 | Retained exactly, including case |
| SCRIPT-block unknown kinds | 468 | One token per SCRIPT v1 block |
| Unassigned SCRIPT fallback | 1 | Scalars outside the projection |

A 48-position input reserves beginning/end markers. A longer surface retains
22 leading and 23 trailing codepoints with `<TRUNC>` between them. Character
counts are measured on normalized, publisher-case-reconstructed natural
training surfaces. The compact recipe uses hard `k=1` backoff: codepoints seen
at most once are replaced by their SCRIPT-block kind.

| Quantity | Count |
|---|---:|
| Natural codepoint occurrences | 1,573,014 |
| Natural distinct codepoints | 4,788 |
| Singleton codepoints replaced | 1,351 |
| Literal codepoints retained | 3,437 |
| SCRIPT blocks exercised by replaced singletons | 52 |
| Replaced singletons unassigned by the projection | 3 |

Unseen characters use the same SCRIPT-block fallback. The projection is
`SCRIPT v1`'s block/index representation, built from Unicode 16.0
`Scripts.txt`. The vendored projection
`scripts/vendor/script_bpe_v1/script_encoding_v1.json` has SHA-256
`895fc5cd38a93f8d64509215d2741a4f93f2df722fd5404c8fd27bc9774121fc`;
the published runtime copy is `name-kind-script-encoding.json` beside the
graph and deployment JSON. It derives from `sanderland/script_bpe` commit
`0fda2bc14109350040274f9ec26997655220533c`. The source `Scripts.txt`
SHA-256 is
`9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039`.

## Architecture

The model has 108,732 trainable parameters:

- 24-dimensional character embeddings;
- three bias-free one-dimensional convolutions of widths 2, 3, and 4 with 64
  channels each, followed by ReLU and global max pooling;
- an 8-dimensional projection of the 35-way language distribution;
- independent shared and language-conditioned affine two-class heads; and
- a convex probability mixture of those heads, 20% shared and 80%
  conditioned by default.

During source-list training, the language vector is dropped for 25% of
examples. Retained vectors receive up to 15% diffuse language noise. This
supports cross-language name occurrence while preserving explicit inference
mixture intent.

## Source-labeled data and language composition

These are publisher/category role labels, not manually reviewed running-text
gold. The v4 input has SHA-256
`133c5292d2b1621879d10dcfdd4beb266ae2ac0960e69a2833091c4c1ccaeb21`.
Its 913,649 rows normalize to 782,980 keys. Global ambiguity excludes 47,707
keys and 123,143 rows, leaving 783,075 unambiguous surface-language examples:
419,425 given and 363,650 family.

The principal sources are U.S. Census family names, INSEE French given names,
an SSA-data mirror, EDRDG JMnedict, English Wiktionary categories for the
supported languages/scripts, and low-weight Faker locale supplements for
sparse cells. Source language/nation metadata describes the publishing corpus;
it is not a claim about a name's intrinsic origin or its bearer's nationality.

The table gives the full all-data fit pool, the capped pre-lap selection split,
and expected all-data optimization mass. Weighting combines Final35 language
importance with support, capped so a well-sourced language receives at most
twice its otherwise assigned mass. Sparse low-weight pools still receive some
exposure.

| Language | All-data given | All-data family | Pre-lap train given | Pre-lap train family | Expected mass |
|---|---:|---:|---:|---:|---:|
| `ar` | 526 | 54 | 429 | 45 | 3.47% |
| `bn` | 537 | 122 | 442 | 98 | 1.77% |
| `cs` | 692 | 6,823 | 548 | 5,416 | 2.27% |
| `da` | 432 | 180 | 349 | 150 | 1.84% |
| `de` | 1,876 | 2,888 | 1,524 | 2,309 | 4.64% |
| `el` | 930 | 1,465 | 740 | 1,154 | 1.99% |
| `en` | 86,834 | 137,692 | 40,000 | 40,000 | 13.51% |
| `es` | 1,281 | 2,265 | 1,032 | 1,797 | 4.75% |
| `fa` | 580 | 266 | 460 | 208 | 1.85% |
| `fi` | 698 | 5,397 | 542 | 4,300 | 2.28% |
| `fil` | 264 | 1,619 | 214 | 1,304 | 1.91% |
| `fr` | 151,375 | 2,107 | 40,000 | 1,701 | 6.76% |
| `he` | 625 | 415 | 492 | 330 | 1.80% |
| `hi` | 882 | 147 | 725 | 123 | 1.81% |
| `hr` | 764 | 1,341 | 619 | 1,065 | 2.31% |
| `id` | 849 | 533 | 685 | 441 | 2.07% |
| `it` | 840 | 10,622 | 664 | 8,492 | 2.36% |
| `ja` | 160,902 | 158,614 | 40,000 | 40,000 | 3.38% |
| `ko` | 13 | 46 | 9 | 39 | 3.40% |
| `ms` | 34 | 1 | 29 | 1 | 1.69% |
| `nl` | 650 | 1,811 | 516 | 1,478 | 2.00% |
| `no` | 537 | 231 | 422 | 181 | 1.86% |
| `pl` | 595 | 14,897 | 466 | 11,918 | 2.19% |
| `pt` | 936 | 677 | 748 | 534 | 4.48% |
| `ro` | 441 | 9,344 | 349 | 7,443 | 1.86% |
| `ru` | 573 | 892 | 445 | 718 | 2.17% |
| `sv` | 555 | 545 | 453 | 434 | 1.99% |
| `ta` | 148 | 4 | 119 | 4 | 1.69% |
| `te` | 218 | 159 | 175 | 124 | 1.82% |
| `th` | 531 | 253 | 445 | 203 | 1.73% |
| `tr` | 2,543 | 99 | 2,032 | 81 | 1.76% |
| `uk` | 384 | 842 | 309 | 680 | 1.97% |
| `ur` | 168 | 17 | 132 | 13 | 1.70% |
| `vi` | 207 | 90 | 164 | 79 | 3.53% |
| `zh` | 5 | 1,192 | 4 | 947 | 3.39% |

Tiny or one-role cells are genuine limitations. Repeated sampling preserves
exposure but cannot create missing distinctions, and their per-language scores
are not broad quality estimates.

## Selection evidence and terminal fit

Normalized surface keys were assigned once by SHA-256 to disjoint
80%/10%/10% partitions. Per-cell caps yielded 270,092 pre-lap training
examples, 57,032 development examples, and 57,606 diagnostic test examples,
with no normalized-key overlap.

The selected c64 run trained from scratch with AdamW, initial LR `0.003`,
weight decay `1e-4`, batch size 1,024, 120,000 samples per epoch, a cosine
schedule over a 300-epoch ceiling, and development patience 10. It stopped
after epoch 64 and restored epoch 54. Before any evaluation rows entered
training, natural-case language-conditioned macro F1 was:

| Split | Rows | Macro F1 | Use |
|---|---:|---:|---|
| Development | 57,032 | 0.871381 | checkpoint selection |
| Test | 57,606 | 0.871253 | one diagnostic after selection |

The deployable terminal fit initialized from that selected checkpoint and used
all 783,075 admissible examples, including both former evaluation partitions.
It ran a fixed 120 epochs at AdamW LR `0.002`, cosine-decayed to exactly zero,
with 120,000 samples per epoch: 14.4 million sampled rows and 14,160 optimizer
steps. Terminal weights are the last epoch; there is no post-merge selector.
The Safetensors model SHA-256 is
`83b51feddd13c6e19167c1b5816ec33293940b87c16b2aa0dc3be0d7ec232549`;
the source training-config SHA-256 is
`51a72bd9388b602d6b1a6f527e108c1b25b4043f0f3cbcd391ea816c1be8483a`.
The published deployment JSON adds the relative ONNX tensor contract and
SCRIPT projection path; its hash is recorded in `name-kind.manifest.json`.
No post-lap score is fresh generalization evidence and none is reported here.

For retraining after a materially changed architecture or blend, keep the
split fixed and bracket LR multiplicatively, reject unstable loss, and run
stable candidates through genuine development flattening. For a compatible
continuation, use a new output directory and include the inherited checkpoint
as selectable epoch zero. Same-stage interruption recovery must restore model,
AdamW, scheduler, sampler, augmentation, and random state; the terminal fitter
checkpoints these after every completed epoch.

## Cost and size

On the measured 16-core x86 host, the selected pre-lap run used four training
threads and accumulated 21m07s across its 64 train/evaluate epochs. The
terminal all-data fit used four threads, had an 8.90-second median epoch, and
completed its 120 epochs in 18m51s. These are shared-host diagnostics, not
service-level guarantees.

The model has 108,732 parameters and a 435,712-byte Safetensors checkpoint.
The source training config is 54,095 bytes; the relocatable deployment JSON is
54,378 bytes. The ONNX graph is 438,138 bytes. Pre-lap PyTorch
encode-plus-forward timing was 359 microseconds for batch 1 and 37.3
microseconds per name at batch 1,024 on one CPU inference thread.

## ONNX validation and compatibility

The graph was checked with ONNX 1.22.0 and evaluated with ONNX Runtime 1.29.0
on `CPUExecutionProvider`, against PyTorch 2.10.0. A six-row parity batch
covering ordinary, all-uppercase, mixed-case, punctuation, Han, and Hangul
surfaces had exact argmax agreement, maximum absolute log-probability
difference `5.722046e-6`, mean difference `1.453484e-6`, and passed
`rtol=1e-5, atol=1e-6`.

The Python equivalent loads the published deployment JSON, resolves and
hash-checks both relative sidecars, applies its NFKC/whitespace and SCRIPT
encoding, validates ONNX tensor shapes, and runs CPU inference through
`scripts.pii_name_role_publish.NameKindOnnxDeployment`. Its `infer-onnx`
command is copied into the published `the production toolkitConfig-name-kind.yml` as the the production toolkit
implementer's behavioral oracle.

The published family is basename-matched:

```text
name-kind.onnx
name-kind.README.md
name-kind.config.json
name-kind-script-encoding.json
name-kind.manifest.json
```

The manifest hashes every member, including this card. The graph is a
standalone name-component classifier. The published
`the production toolkitConfig-name-kind.yml` points at `name-postprocessor-name-kind.json`.
That sidecar has been advanced in place from the working the production toolkit 20.0.15 version-3
contract to the explicit version-4 implementer contract; do not claim current
the production toolkit compatibility until its v4 parser lands. The original version-2
`name-postprocessor.json` remains unchanged for existing the production toolkit configurations.
Python loads the v4 sidecar, resolves the graph and runtime sidecars, and is the
combined grammar-plus-neural behavioral oracle.

## Grammar integration and known limits

The [promoted grammar configuration](name-postprocessor-name-kind.json) sets
`languages.en.scoring.weights.middle_presence` to **1.5** (2026-09-15).
This optional weight defaults to zero and rewards a candidate containing any
middle component once, independently of its length. Ranking and review margins
use the same adjusted score. The [multilingual replay](../../evidence/multilingual-middle-bias-v1.md)
records explicit zero-retention decisions for 17 additional languages, the
held Italian .1 candidate and unqualified remaining languages. This does not
claim zero is optimal or a new nonzero promotion. Other languages and `und` retain zero; the normal
script-gated grammar fallback still applies. Future name-kind-p evaluations
use this configuration as their baseline and record its hash.
The [calibration receipt](../../evidence/english-middle-bias-v2.md) records the
explicit judge-name fields, natural-text controls and reserved audit.
The [remaining gap](../../gaps/english-name-frequency-and-middle-supervision.md)
tracks justified thresholds for other languages, frequency-based source repair
and inclusion of the middle-labelled source in the next neural training round.
The graph itself is unchanged. Existing generated training labels are not
retroactively rewritten; a refreshed view must record the new config identity.

The grammar consumes lexical `given`/`family` scores over subspans of one
complete `person_name`. Whole-name output remains one primary span with
categorical component subspans. Given and family subspans may not alternate;
middle/other material is one consecutive run, and locale profiles decide
given-first versus family-first order and comma inversion.

The current generated component-sequence language is made explicit in
`scripts/pii_name_component_grammars_v1.json`: after removing `Q` and mapping
given/middle/family to `G/M/F`, it is
`(?:G|M|F|GM?F|FGM?)`. The same file separates this legality rule from order
preference and candidate-partition breadth: neutral `und` has no order bonus
and considers every family prefix/suffix, while `given_first` and
`family_first` retain their named order bonus and current
default-plus-lexicon partition search. The deployed neural v4 sidecar now uses
those explicit fields directly and declares its internal apostrophe/hyphen
atom joiners as inclusive Unicode ranges. Python uses the same atom ranges,
filters candidates by the regex before batching their given/family surfaces
through ONNX, and adds centered role log-probabilities. The deterministic v2
sidecar retains the legacy enum.

- The model does not find or type the enclosing `person_name` span.
- It has no intrinsic middle-name classifier; `middle_name` is a grammar
  outcome.
- Several languages have sparse or single-role source support.
- Official lists and synthetic locale supplements differ from running text.
- There is no untouched quality split after the all-data terminal fit.
- ONNX service throughput has not yet been benchmarked.
- There is no language or nation/locale prediction head.
- Complete-name transition learning is deferred to the
  [sequence-aware model gap](../../gaps/sequence-aware-name-component-model.md),
  and language/nation prediction to the
  [language/nation gap](../../gaps/name-language-nation-prediction.md).
