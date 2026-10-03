# Character-boundary refiner

A small ranker that moves each predicted span's start and end by at most one
character when the tokenizer's subword boundaries put them in the wrong place,
for example a span that swallows a trailing period or stops one character
short of a word end. It is the first of the paper's serving stages for O4.

| File | Content |
|---|---|
| `config.json` | Recipe: radius 1 character, 3 characters of context, 262,144 hashed feature buckets, movement penalty 2.0, supported languages and labels |
| `model.safetensors` | One weight vector over hashed character-context features (1 MB) |

**Input and output.** Text, language and predicted primary spans in; the same
spans with possibly adjusted endpoints out. Reference types
(`person_reference`, `organization_reference`) are passed through unchanged.
Use it with `pii-reproduce.py redact --serve`, or call
`scripts/pii_character_boundary_refiner.py apply-sweep` on a prediction
sweep.

**Provenance.** Fitted on Ont3-annotated training rows of the paper's program
(35 languages), including annotations that are not released. The weights are
hashed feature scores; they contain no text. Continued from the serving
parent for 36 epochs, epoch 5 selected; seed 173.

**Validated effect.** On the paper's 1,201-segment Ont3 development pool at
zero bias, O4 with refinement against O4 without it: +2.49 exact-region F1
[1.58, 3.50] and +1.68 exact typed-span F1 [1.01, 2.41]; on human gold,
−0.07 [−0.23, 0.00] (`records/receipts/o4-boundary-summary.json`). On a
model fitted from scratch with this package (`train --recipe o4-fresh`, with
teacher-labeled web text), the same weights add +2.18 exact-region F1
[1.28, 3.18] and +1.50 typed F1 [0.88, 2.18] on that pool, changing about 3%
of predicted spans.

**Training your own.** `pii-reproduce.py train-refiner --checkpoint TAGGER
--annotated ROWS` fits a refiner from complete Ont3 rows with the same code.
In our test, with the tagger's own teacher rows as data, selection kept the
untrained starting point: on rows the tagger was trained on its boundaries
are already nearly right, so learned moves only lowered validation F1. These
weights came from a longer continued recipe with a calibrated movement
penalty; prefer them unless your tagger or languages differ substantially.

**Limits.** Radius one character only. Languages outside `config.json`'s list
get no refinement. The development pool was reused during selection.
