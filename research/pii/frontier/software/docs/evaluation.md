# Evaluation: the paper's scoring views and how to check them

This document explains how the paper scores a tagger, why each scoring rule
exists, how to score your own checkpoint in the same view, and how to
recompute the paper's reported numbers from the shipped receipts. Paths are
relative to the package root.

## Two scorers

| Command | What it does | Use it for |
|---|---|---|
| `python pii-reproduce.py evaluate --checkpoint CKPT` | The paper's human-gold and Ont3 views: bias sweep, the paper's serving stages (`--raw` skips them), coverage masks, optional references, title policy, redaction regions and Ont3 typed spans, paired against O4; then `calibrate` fixes an operating point by the paper's rule | Comparing with the paper's O4 numbers |
| `python pii-reproduce.py score-jsonl --input GOLD --checkpoint CKPT --out DIR` | Generic prediction and scoring of any JSONL with `id`, `text` and `spans` (`start`, `end`, `type`) | Your own test data in the model's own labels |

The generic scorer (`scripts/pii_eval.py`) has none of the paper's corpus
masks or title rules. Its numbers are not comparable with the paper's.

## Redaction regions

The paper's headline metric asks whether the right characters were hidden,
not whether each span received the right type. Gold spans and predicted spans
are each converted into **untyped maximal regions**: trim surrounding
whitespace, then merge spans that overlap or are separated only by
whitespace (`merge_regions` in `scripts/pii_eval.py`).

```text
text:          Dr. Ada Mensah moved from Eastfield to Port Alder.
gold spans:    "Dr. Ada Mensah" person_name; "Eastfield" locality; "Port Alder" locality
predictions:   "Dr." demographic_attribute; "Ada Mensah" person_name;
               "Eastfield" location; "Port Alder" locality
gold regions:  "Dr. Ada Mensah", "Eastfield", "Port Alder"
pred regions:  "Dr. Ada Mensah" (two predictions merged across a space),
               "Eastfield", "Port Alder"        -> 3 of 3 regions match
```

A type error on a correctly hidden span (`location` for a town) does not
count against redaction, because the redacted output is the same; neither
does splitting one span into adjacent pieces. Typed scores are reported separately
on the Ont3 development set.

**Matching.** A gold region and a predicted region match when their
intersection covers at least a set fraction of **both** regions. The paper
reports 80% overlap as the main view and 100% (exact regions) alongside it.
Matching is one-to-one: the scorer computes a maximum bipartite matching
(`_maximum_matches` in `scripts/pii_eval.py`). True positives, predicted
regions and gold regions give precision, recall and F1.

The 80% threshold tolerates a missing punctuation mark or a one-word
disagreement on a long span, which publishers resolve differently. It does
not tolerate a prediction that covers half a name.

**Pooling.** Counts `[true positives, predicted, gold]` are kept per input
row, summed over the population, and F1 is computed once from the sums.
Language panels sum the same counts by language. The paper never averages
per-row or per-language F1.

## The human-gold population

The human-gold population is 1,283 rows from the publishers' **test** splits
of four public corpora:

| Corpus | Rows | | Language | Rows |
|---|---|---|---|---|
| OpenNER (non-NC subset) | 904 | | German | 196 |
| MAPA | 189 | | Spanish | 195 |
| AQMAR | 134 | | Chinese | 195 |
| Wojood sample | 56 | | Portuguese | 192 |
| | | | Arabic | 190 |
| | | | English | 176 |
| | | | French | 139 |

`python pii-reproduce.py human-gold` rebuilds these rows after `data`. The
list `research/pii/frontier/evidence/human-gold-v1/selected-ids.json` names
each row by corpus, source id, line position and SHA-256 of its text; the
rebuild (`scripts/pii_public_gold.py eval-rebuild`) fails if any text hash or
position differs, so a successful rebuild proves you are scoring the same
text. The list also names 30 private rows, which are skipped. `--demo`
rebuilds only the AQMAR and Wojood rows and marks the result as a subset.

The output directory holds `evaluation.jsonl` (rows with gold),
`inputs.jsonl` (the same rows without gold, for prediction) and a receipt with
counts and hashes.

## Scoring rules

`project()` in `research/pii/frontier/evidence/human-gold-v1/score.py`
applies the following rules, in order, to each row before region matching.
`score_points()` in `scripts/pii_paper_o4_eval.py` runs it over every bias
setting and both overlap thresholds.

### Coverage masks

Each corpus annotates only some types. A prediction of a type the corpus
never labels, on text the corpus left unlabeled, is not evidence of an error:
the corpus is silent there. Such predictions would otherwise be scored as
false positives, penalizing a model for knowing more types than the corpus.

For each row the scorer therefore:

1. expands each gold span to all Ont3 types its native label may be (the
   accepted sets of [ontology.md](ontology.md#native-source-labels-and-the-many-to-many-map));
2. keeps a predicted span's type only if the corpus can express it and either
   (a) the corpus annotates that type exhaustively
   (`research/pii/frontier/evidence/four-corpus-v1/negative-coverage-v1.json`),
   or (b) the prediction overlaps a gold span that accepts that type;
3. drops ("masks") predictions with no remaining type, and counts them.

Rule 2(b) keeps the prediction's full extent when it overlaps a gold span, so
incomplete coverage cannot forgive a boundary error on an annotated entity.
Examples: OpenNER has no date label, so a predicted `date` on an OpenNER row
is masked. Wojood's `OCC` label accepts `demographic_attribute`, but Wojood
does not annotate that type exhaustively; a predicted `demographic_attribute`
on unlabeled Wojood text is masked, while one overlapping an `OCC` span is
scored with its full extent, overhang included.

### Optional references

Unless a score is explicitly labeled as including references,
`person_reference` and `organization_reference` are optional: they are
removed from gold and predictions, and a `person_name` or `organization`
prediction that exactly matches a gold reference is neutral (no credit, no
penalty). Wrong families and shifted boundaries remain errors. Required gold
wins over an overlapping optional reference.

The reason is that corpora and annotators disagree on whether "the
defendant" or "the company" should be marked, and leaving such wording
visible usually does not expose the identity. The optional view neither
rewards nor punishes a model for its reference policy. The public corpora
contain no reference labels, so on human gold this rule mainly removes the
model's own reference predictions. Policy identifier:
`optional-reference-neutral-exact-v1`.

### Title policy

Ont3 puts a job title beside a name in a separate `demographic_attribute`
span ("Senator" + "Ada Mensah"); MAPA puts it inside the person span; other
corpora leave it unlabeled. Policy `title-extents/v1` removes that
disagreement from scoring:

- Title extents beside or inside a gold person span are cut out of the gold
  and predicted spans. The remaining name part is always required.
- A **scored title** (one the corpus labels: MAPA's fine `ROLE` and
  `PROFESSION` runs inside `PERSON`, Wojood `OCC` beside `PERS`) becomes its
  own region, covered by any prediction over it, for a model that can express
  `demographic_attribute`.
- A **neutral** extent (MAPA's fine `TITLE` honorifics, and descriptors beside
  a name in corpora that never label them) costs nothing when redacted and is
  never required.

Redacting a title together with a name therefore neither helps nor hurts.
Neutral extents for corpora without a fine layer come from a sidecar file of
character offsets. `evaluate` passes
`research/pii/frontier/software/records/title-extents-human-gold.jsonl` when
that file is present; without it, OpenNER and AQMAR rows have no neutral
extents and the result is flagged as not paper-comparable.

### Effect of the title sidecar and serving stages on O4

| Condition (O4, human gold, 80% overlap) | Curve maximum | Zero bias |
|---|---|---|
| Paper view (with title sidecar) | 88.83 | 88.16 |
| Without title sidecar | 88.27 | 87.54 |

The paper scores each system under its own serving policy. For O4 that adds
three stages after the tagger: character-level boundary refinement,
name-component postprocessing and regular-expression supplementation for
structured identifiers where the model found nothing. On the human-gold
population at 80% overlap these stages leave O4's score unchanged: raw and
served predictions both give 88.83 and 88.16. On the Ont3 development set
the stages do matter: boundary refinement alone adds 2.5 exact-region and
1.7 exact typed-span F1 points
(`research/pii/frontier/software/records/receipts/o4-boundary-summary.json`).
`evaluate` applies the same stages by default: the shipped boundary refiner,
name-component postprocessing once `names` has built the name-kind model,
and regex supplementation; `--raw` scores the tagger's own output.

## O-logit bias sweep

Precision and recall can be traded at decode time by adding a constant `b` to
the `O` logit of every token before the argmax: positive `b` predicts fewer
entities (higher precision), negative `b` more (higher recall). The paper's
prediction script
(`research/pii/frontier/evidence/priority9-shared-v1/predict-sweeps.py`)
decodes every row at `b` = -8, -7, ..., 16, 24 and 32 in one pass and
scores each.

Three numbers appear per system:

- **Curve maximum**: the best F1 over the sweep. It is chosen on the
  evaluation rows themselves, so it is **descriptive**: it shows what the
  model could reach with a tuned threshold, not an honest held-out estimate.
  O4's human-gold maximum is at `b` = 1.
- **Fixed zero bias**: F1 at `b` = 0, declared in advance. The paired
  intervals in the receipts' main comparison and `evaluate`'s comparison
  with O4 use it. Systems with a confidence threshold instead of a bias
  (GLiNER2) use a fixed 0.5.
- **Operating point fixed on Silver-dev**: the point the paper's main
  comparison reports. Every grid setting is scored on Silver-dev (the 659
  `ont3` selection rows) alone; its source documents are resampled 10,000
  times (seed 20261009), and a setting is near-optimal when the 95% interval
  of its F1 difference from the best setting includes zero. The operating
  point is the median setting of the contiguous near-optimal range around
  the best, and it is applied unchanged to Gold-7 (`human`) and Silver-test
  (`heldout`). The grids are finer than the sweep above: quarter steps of
  `b` from -4 to 4 for O3, O4 and OpenMed Privacy Filter, GLiNER2 confidence
  steps of 0.05 and Presidio score steps of 0.025; O2 keeps its saved
  integer grid. O4's operating point is `b` = 0, so its zero-bias numbers
  are also its Silver-dev-fixed numbers (88.16 F1 on Gold-7 and 79.88 on
  Silver-test at 80% overlap).

## Paired bootstrap intervals

A paired comparison of two systems (O4 vs. O3, for example) resamples the
evaluation population with replacement 10,000 times, recomputes both pooled
F1s on each resample and reports the 2.5% and 97.5% quantiles of their
difference. Resampling is by source group: rows from the same document are
drawn together, so correlated sentences do not look like independent
evidence. The human-gold rows carry no document identifiers, so each row is
its own group (1,283 groups); the Ont3 development pool has 692 groups for
1,201 rows. Both systems see the same resampled rows (the pairing), which
removes row difficulty from the variance. Paired intervals use exact (100%)
regions at the fixed operating point. The random generator is seeded
(20260929), so intervals recompute exactly.

The intervals are conditional on the selected recipes. They do not account
for the number of recipes tried during development.

## Evaluating your own checkpoint

```bash
python pii-reproduce.py evaluate --checkpoint work/o4-fresh/fit/model/checkpoint-12000 \
  --name my-model
```

This predicts on `WORK/human-gold/inputs.jsonl` and the two Ont3 collections
with isolated windows over the whole bias grid, applies the paper's serving
stages unless `--raw`, scores with the paper's code, and writes
`WORK/evaluation/summary.json`. Its `human` entry holds:

- `maximum` and `fixed_zero_bias`: precision, recall and F1 at 80% overlap;
  `fixed_default` is the same at the default setting (zero bias, or
  confidence 0.5 for GLiNER2);
- `paper_o4`: the paper's O4 values for the same view;
- `comparable_to_paper`: true only for the full 1,283 rows with the title
  sidecar present;
- `paired_versus_o4`: when comparable, a paired bootstrap of your model
  against O4 on the same 1,283 rows at the default setting and exact
  regions, computed with the paper's code from O4's shipped receipt.

Its `ont3` entry gives the same view per collection and pooled, with exact
typed spans. `WORK/evaluation/scores-human.json.gz` and `scores-ont3.json.gz`
keep per-row counts at both overlap thresholds (and typed, on Ont3) for every
setting, in the schema of the paper's score archives (from which the shipped
receipts were derived), so you can compute your own panels and intervals.

`--grid fine` adds the finer settings the paper fixed operating points on
(quarter biases from -4 to 4; GLiNER2 confidence steps of 0.05); use it
before `calibrate`. A GLiNER2 checkpoint from `train-gliner2` is recognized
from its configuration and scored on its own output, as the paper scored GL4,
at confidence thresholds instead of biases; its `--name` must start with
`gliner2` (default `gliner2-model`), which is how the paper's code knows its
default setting is 0.5.

## Fixing an operating point

```bash
python pii-reproduce.py evaluate --checkpoint CKPT --grid fine --out work/evaluation-mine
python pii-reproduce.py calibrate --evaluation work/evaluation-mine
```

`calibrate` fixes your model's bias (or GLiNER2 confidence threshold) on one
population and reports that setting unchanged on the others, by the rule the
paper used for every system
([the bias sweep](#o-logit-bias-sweep); `select_threshold` in
`scripts/pii_paper_o4_figures.py`, the same code): take the setting with the
best pooled 80%-overlap region F1 on the development population, resample
its source documents 10,000 times, keep the contiguous range of settings
whose F1 difference from the best has a 95% interval including zero, and
take the median setting of that range. Choosing the middle of the
near-optimal range rather than the single best point keeps the choice from
chasing noise in a few hundred rows.

`--development silver-dev` (the default, as in the paper) chooses on the 659
Ont3 selection rows; `gold-7` or `silver-test` choose on those instead. The
result, `EVALUATION/calibration-DEVELOPMENT.json`, gives for every population
the scores at the selected setting, at the default and at the development
argmax (regions at 80% and exact overlap, and typed spans on Ont3), marks
which population was the development set, pairs your model against O4 at
O4's own operating point (bias 0; 80%-overlap regions, like the paper's
operating-point comparisons), and lists the paper's O4 (and, for a
GLiNER2 model, GL4) at their operating points. Only the held-out populations'
scores are estimates of what to expect at that setting. Applied to the
shipped receipt curves of O3 and GL4, `calibrate` reproduces their published
operating points and scores exactly (`tests/unit/test_pii_software_calibrate.py`).

To tag or redact raw text:

```bash
python pii-reproduce.py redact --checkpoint CKPT --input docs.txt --out redacted.jsonl --lang de
```

Input is one document per line, or JSONL with `text` and optional `id` and
`lang`. Each output line has typed spans and a copy of the text with each
span replaced by `[type]`. Decoding is at zero bias with adjacent person and
organization pieces merged, followed by the paper's serving stages unless
`--raw`. Pass the right
`--lang`, because the model's language bias is chosen by it. A model is
unreliable on types its training data never supervised: with `--work WORK`,
`redact` emits only the types `WORK/mixture` supervised, and `--types`
names an explicit list.

## Verifying the paper's numbers

```bash
python pii-reproduce.py verify [--details]
```

This recomputes every reported F1 maximum, fixed-bias F1 and paired interval
from `research/pii/frontier/software/records/receipts/` using the paper's own
pooling and resampling code (`scripts/pii_software_receipts.py verify`), and
fails if any value differs from the shipped summaries by more than 1e-12.

The receipts contain, for each system, population, bias setting and overlap
threshold, the per-row counts `[true positives, predicted, gold]` plus row
identifiers, languages and source groups. They contain no text, predictions
or spans. When the receipts were built, every count was checked to expand back
to the original score archive, and every receipt string of document-like
length was checked not to occur in any evaluation text. SHA-256 hashes bind
each receipt to the withheld score archive, gold files and prediction files.

| Receipt | Content |
|---|---|
| `o4-comparison` | 13 systems on human gold and the Ont3 development pool; paired O4 vs. O3 and GLiNER2 comparisons |
| `o4-boundary` | O4 with and without boundary refinement at zero bias |
| `gliner-trajectory` | GLiNER2 adaptation checkpoints on the paper populations (development diagnostic) |
| `operating-point-presidio`, `operating-point-refined-grid`, `operating-point-fine-bias` | The finer threshold and bias grids behind the Silver-dev operating points |
| `operating-points-trust-region.json` | Each system's Silver-dev operating point, its near-optimal range and its scores on Gold-7 and Silver-test |

For the operating points, `verify` merges the finer grids over the main
comparison, reruns the paper's selection rule
(`operating_points` in `scripts/pii_paper_o4_figures.py`), compares every
value of `operating-points-trust-region.json` with the result, and checks
each system's chosen setting and its rounded F1 against the paper. Its
result line reads, for example, `O4 bias 0: Gold-7 88.2, Silver-test 79.9`.

Selected values recomputed by `verify` (F1, percent):

| System | Population | 80% max | 80% zero bias | 100% max | 100% zero bias |
|---|---|---|---|---|---|
| O4 | human gold | 88.83 | 88.16 | 88.53 | 87.80 |
| O3 | human gold | 87.34 | 86.91 | 86.88 | 86.55 |
| O4 | Ont3 development | 82.13 | 82.13 | 80.20 | 80.20 |
| O3 | Ont3 development | 81.80 | 81.80 | 79.65 | 79.65 |

O4 minus O3 at zero bias, exact regions: +1.25 points on human gold (95%
interval +0.19 to +2.34) and +0.55 on Ont3 development (-0.62 to +1.74);
exact typed spans on Ont3: +1.79 (+0.51 to +3.13).

What `verify` establishes: the reported numbers follow from the per-row
counts by the stated pooling and resampling. What it cannot establish: that
the counts came from the stated models and gold. You can close that gap for
the evaluation data yourself: rebuild the human-gold rows, and score a model
on the shipped Ont3 populations (`data/ont3-evaluation/`), which hold the
rows, text and references O4 was scored on. O4's weights are not released,
so its own counts remain receipts.

## Qualifications

- **Human-gold rows never train.** The 1,283 rows are a subset of the
  publishers' test splits. Training uses only the same corpora's train
  splits, and O4's training rows also passed a near-duplicate screen against
  the evaluation rows ([data.md](data.md#how-the-mixture-is-compiled)).
- **Human gold is a development set.** Those rows were evaluated repeatedly
  while recipes were compared and O4 was chosen, so they are not an untouched
  final test. We do not believe this selection meaningfully overfit the O4
  recipe to them. The paper's evaluation on fresh data supports this: on 542
  Ont3 segments, freshly annotated from documents that never train and never
  used to select O4, O4 leads the adapted GLiNER2 baseline by 12.8 [10.0,
  15.6] fine-type character F1 and 9.3 [6.7, 12.1] redaction-character F1
  (86.8 and 89.3 F1 for O4, each system at its Silver-dev operating point;
  the paper's GLiNER2-adaptation appendix). An
  unselected model also matches it: the package's fresh fit, 12,000 updates
  of O4's recipe from pretrained XLM-R with no checkpoint or run selected on
  human gold, scores +0.9 [−0.3, 2.0] exact-region F1 over O4 there (served;
  `research/pii/frontier/software/records/receipts/fresh-fit-summary.json`),
  which indicates that selection has not inflated O4's human-gold score.
- **Human gold is in-distribution.** Every scored corpus also supplies
  training rows from its train split. As with any split of one identically
  annotated corpus into train and test, the test rows share the training
  rows' text sources and annotation conventions, so scores there overstate
  what to expect out of distribution. Keep this in mind when judging the
  tagger on other domains or annotation styles.
- **Publisher splits are not all document-disjoint.** OpenNER re-split
  AnCora by sentence, so most AnCora test documents also contribute training
  sentences ([data.md](data.md#known-limitations)).
- **The Ont3 development pool is partly reused.** Its 1,201 rows cover 35
  languages (plus 12 rows of undetermined language) and come from two
  collections, both shipped in `data/ont3-evaluation/`: 659 rows whose
  references were settled by two annotation passes (exact agreements kept,
  disputes adjudicated) and then manually revised, the subset used to select
  O4; and 542 fresh training-disjoint rows with single-teacher references,
  never used for selection. Report both collections and the pool.
- **Coarse corpora bound what human gold can show.** Its corpora annotate a
  few types; performance on the other Ont3 types is visible only in the Ont3
  development scores.
- **Operating points are fixed on Silver-dev.** Curve maxima are chosen on
  the scored rows and only describe each curve. The paper compares systems
  at points fixed on Silver-dev by the near-optimal-range rule of
  [the bias sweep](#o-logit-bias-sweep), never on Gold-7 or Silver-test
  rows; `verify` recomputes them from
  `research/pii/frontier/software/records/receipts/operating-points-trust-region.json`
  and the `operating-point-*` grid receipts. Silver-dev also selected O4,
  so it is not a held-out set for O4.
