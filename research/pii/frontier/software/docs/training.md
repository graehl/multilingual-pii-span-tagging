# Training an O4-style tagger

This document describes the model architecture, the training recipes the
driver offers, the meaning of O4's recorded trainer options, and what
separates a fresh fit from public data from the published O4 model. Paths are
relative to the package root. Data preparation is in [data.md](data.md), the
label inventory in [ontology.md](ontology.md), and scoring in
[evaluation.md](evaluation.md).

## The model

| Component | O4 setting |
|---|---|
| Encoder | `FacebookAI/xlm-roberta-large` (24 layers), fully fine-tuned |
| Primary head | One affine layer (weight and bias) over the last encoder layer's token states, 125 outputs: B/I/E/S for each of 31 Ont3 types, plus `O` |
| Decoding | Per-token argmax; if the result violates BIOES order, a constrained decoder picks the best legal sequence |
| Language bias | A zero-initialized output bias per language on the BIOES logits, selected by the row's `lang`; languages outside `scripts/pii_register_languages_v1.json` use no bias |
| Auxiliary heads | Predicate and subclass outputs ([ontology.md](ontology.md#refinement-channels)); they do not affect primary decoding |
| Input | Windows of at most 900 characters and 512 tokens; no neighboring-sentence context |

The affine head is deliberately simple: a token's label depends only on its
own final hidden state and the language bias. Span structure comes from the
BIOES labels and the constrained decoder, not from a CRF or span classifier.

The trainer is `scripts/pii_encoder_train.py`; run
`python scripts/pii_encoder_train.py -h` in the installed environment for
every option.

## Recipes

`python pii-reproduce.py train --data DIR --out OUT --recipe R` runs one of
three recipes. `DIR` must contain `train.jsonl`, `val.jsonl` and
`labels.json`; `OUT` must not exist.

**`affine`** (default) is a minimal fit: XLM-R large with an affine BIOES head
over whatever labels `labels.json` lists, length-matched batching and span-F1
checkpoint selection. It has none of O4's mapping, auxiliary objectives or
loss settings. Use it for plumbing checks or for a tagger over your own
inventory.

**`o4`** replays O4's recorded trainer options, read at run time from
`research/pii/frontier/software/records/paper-run-records.json` (the record
for run `pii-gs30-titles-v6-g50-seed2-4000`). O4 continued an existing
checkpoint, so this recipe requires `--init-from-checkpoint PARENT`. That
parent must already have an Ont3 head. O4's own parent is not distributed.

**`o4-fresh`** uses the same options starting from the base encoder, with the
learning rates that O4's earliest ancestor used when it was first fit from
pretrained XLM-R (head 5e-5, encoder 3e-5, instead of O4's 2e-5 and 1e-5).
It runs in two steps:

1. **Head creation.** The mapped-single-head mode only continues an existing
   Ont3 head; it never creates one. The driver therefore first trains one
   update on `DIR/root/` (base rows labeled natively in Ont3, written by
   `mixture`) with `--native-new-label-space`, which creates the 125-output
   affine head. This single update is not meant to teach anything; it
   produces a checkpoint with the right head shape.
2. **Fit.** The driver then continues from that checkpoint with the O4
   options, the mixture's `mapping.json`, and `--dual-head-bind-native-map`,
   which attaches the map to the existing head without changing its weights.

Both `o4` and `o4-fresh` need `DIR/mapping.json` and replace O4's
35-language declaration with `DIR/language-round.yaml` when present
([data.md](data.md#how-the-mixture-is-compiled)).

A complete public run:

```bash
python pii-reproduce.py install
python pii-reproduce.py data
python pii-reproduce.py human-gold
python pii-reproduce.py mixture --gold-share 0.5 --o4-membership
python pii-reproduce.py train --data work/mixture --out work/o4-fresh --recipe o4-fresh --steps 12000
python pii-reproduce.py evaluate --checkpoint work/o4-fresh/fit/model/checkpoint-12000
```

`python pii-reproduce.py demo` runs the same chain at minutes scale (three
small corpora, 200 updates, batch 4 x accumulation 2) and then redacts two
sample sentences. It proves the code runs; its scores mean nothing.

### Options the driver owns

The driver sets these itself and drops any recorded value, so a replay cannot
silently reuse the original machine's paths: `--data`, `--out`, `--model`,
`--init-from-checkpoint`, `--dual-head-map`, `--max-steps`, `--eval-steps`,
`--batch`, `--grad-accum`, `--seed` and `--keep-final-checkpoint`. The
driver's defaults equal O4's values (validation every 2,000 updates, batch 8,
gradient accumulation 8, seed 20260927), except the update budget: 4,000 as
in O4 for `o4` and `affine`, 12,000 for `o4-fresh` (see the budget note
below).

`--step-scale F` multiplies the update budget and `--max-steps N` caps it;
the smaller wins. Further native trainer options may follow `--`, but they
cannot override driver-owned or recipe options. Each run writes
`OUT/.../receipt.json` with the exact command, requested and effective
steps, and exit status.

## O4's recorded options

The table groups O4's trainer options by purpose. Values are from the shipped
run record.

### Objective

| Option | Value | Meaning |
|---|---|---|
| `--dual-head-map` | the mixture's `mapping.json` | Native-label rows supervise the Ont3 head through accepted sets |
| `--dual-head-mapped-single-head` | on | Only the Ont3 head exists ([ontology.md](ontology.md#the-mapped-single-head)) |
| `--dual-head-old-weight-schedule` | `constant:0` | No objective weight for any older head |
| `--o-token-loss-weight` | 0.75 | Trusted `O` targets count 0.75 relative to entity targets |
| `--partial-primary-objective-weight` | 1.0 | Primary loss on `annotated_spans_only` rows is kept at full weight |
| `--subclass-spec`, `--subclass-loss-weight` | `scripts/pii_subclass_families_v3.json`, 1.0 | Subclass families trained where rows annotate them |
| `--predicate-spec`, `--predicate-loss-weight` | `scripts/pii_ont3_bernoulli_predicate_channels_v1.json`, 1.0 | Predicate head present |
| `--predicate-objective-mask-channel` | all six channels | Every predicate channel receives zero objective weight |
| `--predicate-conditioning` | `primary-type` | One predicate block per primary type |
| `--bioes-margin-boundary`, `--bioes-margin-type` | 0, 0 | BIOES margin losses off |
| `--bioes-risk-weight` | 0 | Span-risk reweighting off |

**O-token weight.** Most tokens are `O`. Weighting trusted `O` targets below
1 shifts the model toward predicting entities, trading some precision for
recall. For redaction a missed identifier is usually costlier than an extra
masked word. The trainer applies this weight inside the mapped accepted-set
loss as well. Tuning this weight at training time, rather than only moving
the decision threshold afterwards, is the preferred way to set the
precision-recall trade-off; the evaluation's `O`-logit sweep
([evaluation.md](evaluation.md#o-logit-bias-sweep)) then describes the curve
around the trained operating point.

**BIOES margins.** The trainer can require the gold label to beat
alternatives by an extra logit margin when they differ in BIOES position
(`--bioes-margin-boundary`) or in type (`--bioes-margin-type`), scaled down
for flips the constrained decoder would repair anyway
(`--bioes-margin-illegal-scale`). O4 did not use them; they are available
for boundary-sensitive applications.

### Sampling and batching

| Option | Value | Meaning |
|---|---|---|
| `sampling_weight` (per row) | from the mixture | Draw probability; branch shares are set by the compiler |
| `--sampling-epoch-windows` | 64,000 | Draws per sampler epoch; 4,000 updates x 64 windows = 4 epochs |
| `--sampling-length-window-steps` | 1,000 | Draws for 1,000 updates are pooled, sorted by encoded length and cut into physical batches |
| `--annotation-variant-weighting` | `legacy` | The trainer does not re-share repeated annotations; the compiler already has |
| `--language-round` | O4: its 35-language file; replay: the mixture's | Core languages that must reach their minimum sampling share |
| `--minimum-language-share` | 0 | No additional per-language floor |
| `--training-windowing` | `token-capacity` | Windows fit the real tokenizer budget without cutting any span; an unsplittable over-long span is an error |
| `--max-chars`, `--max-len` | 900, 512 | Window limits in characters and tokens |
| `--context-field`, `--context-side`, `--context-configurations` | `document_context`, `previous`, `[[0]]` | Context available, but every training draw uses the target sentence alone |

**Length-matched batching.** Draws are weighted-random, so a batch could mix
a 20-token and a 500-token window and waste most of its compute on padding.
The sampler draws the rows for many updates at once, sorts them by encoded
length and forms each physical batch from neighbors, so batch members have
similar lengths while every row keeps its sampling probability. When a row
has variants of different length (with and without context, for example),
each variant enters the pool as its own item with its share of the row's
weight, so batching sees the length actually encoded. The trainer checks
this on every batch and prints `TRAIN-LENGTH-AUDIT` lines when an item's
encoded length differs from its batching length by more than 10%.

**Context.** O4 was trained and is served on isolated windows. In the
paper's development runs, mixing in previous-sentence context during training
made isolated inference slightly worse; the recipe keeps training isolated.

### Optimization

| Option | Value | Meaning |
|---|---|---|
| `--lr`, `--encoder-lr` | 2e-5, 1e-5 (`o4-fresh`: 5e-5, 3e-5) | Head and encoder learning rates |
| `--sched` | cosine | Cosine decay after the default 3% warmup |
| `--batch`, `--grad-accum` | 8, 8 | 64 windows per update |
| `--max-grad-norm` | 0.125 | Gradient-norm clipping threshold per update |
| `--training-parameter-precision` | `float32` | FP32 parameters and Adam moments under BF16 autocast |
| `--seed`, `--validation-seed` | 20260927, 20260912 | Training draws; choice of validation rows |
| `--victory-lap-lr-scale` | 0 | No extra low-rate pass after selection |
| `--soft-registers` | 0 | No learned prefix positions |

**Gradient clipping.** The per-update loss is the mean over the eight
accumulated batches (`--accumulation-loss mean`, the default). A threshold of
0.125 = 1/8 under the mean equals a threshold of 1.0 on the summed loss that
earlier runs in this lineage used, where clipping was active on almost every
step. O4 keeps that behavior.

**FP32 parameters.** With BF16 parameters, updates smaller than BF16
resolution are lost, which matters at learning rates of 1e-5 and below over
thousands of steps. Keeping parameters and optimizer state in FP32 while
computing the forward pass in BF16 avoids that at the cost of more memory.
Evaluation loads checkpoints in BF16 on GPU.

### Validation and checkpoints

| Option | Value | Meaning |
|---|---|---|
| `--selection-metric` | `span-f1` | Checkpoints are ranked by exact typed span F1 on validation |
| `--eval-steps` | 2,000 | Validation at updates 2,000 and 4,000 |
| `--max-val-windows`, `--val-selection` | 0, `shuffle` | All validation rows, in a seeded order |
| `--patience`, `--early-stopping-threshold` | 1000, 0.001 | Early stopping effectively disabled for this length |
| `--keep-final-checkpoint`, `--save-limit`, `--no-prune-early-checkpoints` | on, 12 | Keep the final checkpoint besides the validation best |

Span F1 is computed only on validation rows labeled in Ont3
(`label_space: v2`). Rows supervised through the map carry a set of
acceptable types rather than one label, so they cannot be scored exactly and
are excluded. A mixture without Ont3-labeled validation rows cannot select by
span F1; the public compiler relabels assembler validation rows for this
reason ([data.md](data.md#how-the-mixture-is-compiled)).

**The paper's O4 is the final checkpoint** (`checkpoint-4000`) of its
4,000-update schedule, not the trainer's validation-best checkpoint. Choosing
among candidate runs used development scores on the paper's evaluation
populations, with human gold carrying most of the weight, rather than the
trainer's internal metric; this is why the human-gold numbers are qualified
as development results ([evaluation.md](evaluation.md#qualifications)).

## Hardware and cost

The O4 continuation ran on one GPU with 96 GB of memory (the run record
names the device). Our fresh fits (`o4-fresh`, 12,000 updates, batch 8 x
accumulation 8) ran on the same kind of GPU at about 2 updates per second
alone and 1.5 with two fits sharing it, 2 hours 15 minutes each, peaking
near 13 GB of GPU memory per fit. Reduce `--batch` and raise `--grad-accum`
to fit a smaller GPU with the same effective batch.

## What separates a fresh public fit from O4

A run of `o4-fresh` on public data uses O4's objective, sampling, optimizer
and selection settings, and O4's exact human-gold rows with
`--o4-membership`. It differs in:

1. **Base data.** About half of O4's draws came from teacher-annotated web
   text and other private or partly public pools covering 35 languages and all
   31 types. The public base branch has a few coarse types in a handful of
   languages, all partially supervised ([data.md](data.md#what-o4-was-trained-on)).
2. **Training history.** O4 is 4,000 updates on top of a parent that had
   itself been trained on earlier Ont3 mixtures. A fresh fit starts from
   pretrained XLM-R with a one-update head.
3. **Budget.** The recipe's 4,000 updates were chosen for a continuation.
   In our fresh fit with teacher-labeled web text, human-gold zero-bias F1
   was 86.3 after 2,000 updates, 88.1 after 4,000, 88.6 after 6,000 and 89.4
   after 12,000; we recommend `--steps 12000` from scratch.
4. **Validation.** O4's validation rows were Ont3-annotated; the public
   compiler's are relabeled publisher rows.

The result is a recipe reproduction, not a recreation of O4's weights or
scores. Expect the largest gaps on types and languages the public corpora do
not annotate.

## Adapting the recipe

- **More annotated data.** Add Ont3-labeled rows to the base branch with
  `scripts/pii_public_mixture.py --annotated FILE`, after deduplicating them
  against evaluation data.
- **Another label inventory.** See
  [ontology.md](ontology.md#migrating-to-a-new-inventory).
- **Precision-recall trade-off.** Change `--o-token-loss-weight`, keep
  decoding and selection fixed, and compare on the same evaluation rows.
- **Human-gold share.** `mixture --gold-share F` changes the branch balance.
  More publisher gold improves robustness on those publishers' text but
  pulls type choices and extents toward their conventions; a gain on the
  human-gold evaluation is partly in-distribution, because the same corpora
  supply the training branch.
- **Smaller or faster models.** The driver's `--model` accepts another
  Hugging Face encoder. The `o4` recipes read the last of 24 encoder layers
  (`--encoder-layers 24`) and the driver does not let extra options override
  recipe settings, so for an encoder of another depth use the `affine` recipe
  or call `scripts/pii_encoder_train.py` directly.
