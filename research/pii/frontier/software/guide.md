# Multilingual PII tagging and redaction: software

This package trains and evaluates the span tagger described in the paper: an
XLM-RoBERTa-large encoder with an affine BIOES head over a 31-type privacy
ontology (Ont3), trained on a mix of public human-annotated corpora and
teacher-annotated text. It lets you

- **run it**: a minutes-scale demo that fetches public data, trains, evaluates
  and redacts text;
- **build one**: fit a tagger with the paper's final (O4) recipe from public
  corpora, score it on the paper's public human-gold evaluation, and compare
  it row by row with O4;
- **check the paper**: recompute every reported comparison score and
  confidence interval from shipped text-free receipts.

The paper is at https://arxiv.org/html/2609.38630.

No corpus text, annotation labels or trained weights are included. Public
corpora are downloaded from their pinned upstream revisions under their own
licenses.

## Quick start

Linux with an NVIDIA GPU (CUDA 13 driver) and [Pixi](https://pixi.sh).

```bash
python pii-reproduce.py install   # pinned environment
python pii-reproduce.py verify    # recompute the paper's numbers (CPU)
python pii-reproduce.py licenses  # every data source and its license terms
python pii-reproduce.py licenses --approve aqmar-openner \
    --approve wojood-sample --approve idner-news-2k
python pii-reproduce.py demo      # three small corpora end to end
```

Commands that download data refuse any source whose license terms you have
not approved. `licenses` lists each source's recorded license, terms URL and
required citations; `--approve SOURCE` records that you accept those terms in
`license-approvals.json`, and `--approve-all` approves every listed source
except those under commercial terms. An approval lapses if the recorded
terms change.

On our GPU host, from a fresh extraction of this archive with warm download
caches, `install` took 12 seconds, `verify` 12 seconds and `demo` about two
minutes; a first `install` downloads the pinned PyTorch stack.

`demo` writes `work-demo/`: converted data, a 200-update model, its score on
the 190 human-gold rows its corpora cover, and `demo-redacted.jsonl` with
redacted sample sentences. The demo proves that the pipeline runs; its model
is badly undertrained (about 65 F1 on that subset, with visibly wrong
redactions), and its scores are not quality results.

## Build a model like O4

Every step writes under one work directory (default `work/`) and logs its
command, settings and exit status to `work/logs/<step>/receipt.json`. First
approve the six corpora `data` fetches (openner-commercial-core, mapa,
aqmar-openner, wojood-sample, hiner, idner-news-2k), and fineweb and
fineweb-2 if you will use `fetch-web`, or `licenses --approve-all` once you
have reviewed the list.

```bash
python pii-reproduce.py data                  # fetch and convert public corpora
python pii-reproduce.py human-gold            # rebuild the 1,283-row evaluation
python pii-reproduce.py mixture               # compile the training directory
python pii-reproduce.py train --recipe o4-fresh --data work/mixture \
    --out work/model --steps 12000   # about 2 hours on one large GPU
python pii-reproduce.py evaluate --checkpoint work/model/fit/model/checkpoint-12000
python pii-reproduce.py redact --work work \
    --checkpoint work/model/fit/model/checkpoint-12000 \
    --input my-documents.txt --out my-documents.redacted.jsonl
```

- `data` fetches OpenNER commercial core, MAPA, AQMAR, Wojood, HiNER and
  IDNER, converts them with their label maps, and assembles their natural
  realizations as base rows. Compare `work/onboarded/*/manifest.json` with
  the shipped `data/pii-onboarded/*/manifest.json` to confirm identical data.
- `human-gold` rebuilds the paper's public human-gold population from the
  corpora's test splits and checks every row against a pinned text hash.
- `mixture` compiles O4's human-gold branch exactly as O4 used it
  (`--o4-membership` keeps O4's overlap-screened rows) and fills the other
  half of the sampling mass with base rows. `--annotated FILE` adds your own
  teacher-annotated rows in Ont3 labels, for example `annotate` output.
- `train --recipe o4-fresh` fits from the base encoder with O4's recorded
  trainer options: one update creates the Ont3 head, then the full fit binds
  the label map. `--recipe o4 --init-from-checkpoint DIR` continues an
  existing Ont3 checkpoint instead.
- `evaluate` runs the paper's prediction sweep and scorer on both of the
  paper's evaluations: the rebuilt human gold and the shipped 31-type Ont3
  populations (`data/ont3-evaluation/`: 659 rows O4 was selected on, 542
  never used for selection). It reports zero-bias F1 for each and a paired
  comparison with O4's receipts. Like O4's reported numbers, it scores
  served output by default; `--raw` scores the tagger's own spans and pairs
  them with O4 without refinement. `--population human|ont3` limits it to
  one evaluation.
- `calibrate --evaluation DIR` fixes the model's bias on Silver-dev by the
  paper's operating-point rule and reports that setting on Gold-7 and
  Silver-test, next to O4's paired scores; run `evaluate --grid fine` first
  for the paper's finer bias grid
  ([docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md#fixing-an-operating-point)).
- `redact` emits typed spans and a redacted copy of each input line. With
  `--work` it only emits types the training data supervised. By default it
  applies the paper's serving stages: the shipped character-boundary refiner
  (`research/pii/frontier/software/models/boundary-refiner/`, worth about
  +1.5 to 2.5 F1 on Ont3, nothing on human gold), name-component
  postprocessing when you have built the name-kind model, and regex
  supplementation for identifiers; `--raw` skips them. `train-refiner` fits
  your own refiner.

The paper names its evaluation sets differently from this package's code,
files and receipts. "Ont3" in the paper is only the 31-type label set.

| Paper | Rows | Code, receipts and sweep files | Data |
|---|---|---|---|
| Gold-7 | 1,283 | `human` ("human gold") | rebuilt by `data` from publisher test splits |
| Silver-dev | 659 | `ont3`, or "selection" | `data/ont3-evaluation/selection-inputs.jsonl` and `selection-references.jsonl` |
| Silver-test | 542 | `heldout` | `data/ont3-evaluation/heldout.jsonl` |
| Silver | 1,201 | both Ont3 collections pooled ("Ont3 development pool") | both of the above |

The paper fixes each system's operating point on Silver-dev and reports it
on Gold-7 and Silver-test.

O4 also trained on about 50,000 sentences of public FineWeb text labeled by
an LLM teacher. We do not release those labels, but you can recover the
exact text and label it with your own teacher:

```bash
python pii-reproduce.py fetch-web             # exact O4 web text, hash-verified
python pii-reproduce.py annotate --prompt-revision paper-teacher \
    --input work/web/rows.jsonl --web-receipt work/web/receipt.json --out work/teacher
python pii-reproduce.py mixture --annotated work/teacher/training-rows.jsonl \
    --out work/mixture-teacher   # then train with --data work/mixture-teacher
```

`fetch-web` finds each row's FineWeb or FineWeb-2 record and keeps only rows
whose SHA-256 equals O4's training text. `annotate --prompt-revision
paper-teacher` labels them with the prompt O4's teacher used for its training
rows; without that option, `annotate` uses the prompt behind the paper's
prompted-LLM result instead (see "Other commands").

Choose the teacher by the quality you need. O4's teacher was Luna:
`--codex-model gpt-6-luna` (or `--luna`) sends the same prompt through your
`codex login`, in a context-isolated home the driver builds, and is the
route to annotation like the paper's; see
[docs/data.md](research/pii/frontier/software/docs/data.md#choosing-a-teacher).
Gemma-4 31B is the default only for convenience: it runs locally without an
account, in-process, or for all ~50,000 rows behind a batching server from
the separate pinned vLLM environment (`pixi run serve-teacher` in
`research/pii/frontier/software/serve/`, then add `--server
http://127.0.0.1:8000/v1`): about an hour and a half on one 96 GB GPU
instead of days. `--model` picks another Hugging Face teacher. Any teacher
that labels all 31 types can take its place, and its quality will shape the
result.
[docs/data.md](research/pii/frontier/software/docs/data.md) explains what
each source contributes and how to screen and annotate your own text
(`screen`, then `annotate --dedup-receipt`).

`data` downloads about 300 MB and took one minute on our host with warm
download caches; `mixture` took under two minutes.

## What a fresh fit reaches

We ran the commands above on one 96 GB GPU: `mixture --annotated` with the
teacher-labeled web text (our cached labels standing in for yours), then
`train --recipe o4-fresh --steps 12000` from pretrained XLM-R large, about
2 hours on a GPU shared with another fit. `mixture` removed 316 of its
209,593 distinct training texts as overlapping an evaluation row and capped
Hindi at 8% of sampling. All scores below come from `evaluate`: "served" is
its default (the paper's serving stages, as in O4's reported numbers; here
the boundary refiner and regex, since we did not build the name-kind model),
"raw" is `evaluate --raw`, paired against O4 without refinement. F1 at zero
bias; brackets are paired 95% intervals.

Human gold (1,283 rows):

| Model | Output | Max, 80% regions | 80% regions | Exact regions | Fresh − O4, exact regions |
|---|---|---|---|---|---|
| O4 (paper) | served | 88.8 | 88.2 | 87.8 | |
| O4 (paper) | raw | | 88.2 | 87.9 | |
| Fresh fit | served | 89.2 | 88.9 | 88.7 | +0.9 [−0.3, 2.0] |
| Fresh fit | raw | 89.2 | 88.9 | 88.7 | +0.9 [−0.3, 2.0] |

"Max" is the best point of each bias curve on these same rows, which only
describes the curve. The paper reports O4 at the bias it fixed on Silver-dev,
which for O4 is zero: 88.2 redaction F1 on Gold-7 (human gold), the
zero-bias column above.

Ont3, exact regions / exact typed spans (selection: the 659 rows O4 was
selected on; held-out: 542 rows never used for selection):

| Model | Output | Selection | Held-out | Pooled 1,201 |
|---|---|---|---|---|
| O4 (paper) | served | 81.8 / 76.9 | 77.0 / 75.2 | 80.2 / 76.3 |
| O4 (paper) | raw | 78.9 / 74.8 | 75.5 / 74.2 | 77.7 / 74.6 |
| Fresh fit | served | 80.6 / 76.1 | 79.4 / 76.1 | 80.2 / 76.1 |
| Fresh fit | raw | 77.7 / 74.0 | 78.0 / 75.4 | 77.8 / 74.5 |
| Fresh − O4, served, regions | | −1.3 [−2.8, 0.2] | +2.3 [0.3, 4.2] | 0.0 [−1.3, 1.2] |
| Fresh − O4, served, typed | | −0.8 [−2.5, 0.9] | +0.9 [−1.2, 2.9] | −0.2 [−1.5, 1.2] |

The fresh fit is level with O4 on human gold and on the pooled Ont3
evaluation, leads it on the held-out Ont3 rows for redaction regions, and
trails it on the selection rows, not significantly. Ont3 accuracy in a
language rises with more Ont3 annotation of its text by Luna or a stronger
teacher; O4's training included such annotations of further text sources,
which this package does not ship. The fit matched O4 on human gold after
4,000 updates (88.0 at zero bias) and reached 88.9 at 12,000. The serving
stages are worth about 1 to 3 points on Ont3, whose conventions put span
edges at characters the tokenizer splits poorly, and nothing on human gold.
The public corpora never annotate most of Ont3's types; the teacher-labeled
text is what teaches those types and supplies trusted negatives for them.
Human gold comes from the publishers' test
splits, which never train the model: training uses the same corpora's train
splits, screened against the evaluation rows. It was reused during
development, though we do not believe O4 was meaningfully over-selected on
it: on 542 freshly annotated segments never used to select it, O4 keeps a
9 to 13 F1 lead over the adapted GLiNER2 baseline, and the fresh fit above,
with no checkpoint or run selected on human gold, scores +0.9 [−0.3, 2.0]
exact-region F1 over O4 there. As with any train/test
split of identically annotated data, human gold is
in-distribution and overstates accuracy on other domains; see
[docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md#qualifications).
Details, including the intermediate checkpoints and the 31-type results:
`research/pii/frontier/software/records/receipts/fresh-fit-summary.json`.

## Check the paper's numbers

`verify` expands the receipts in `research/pii/frontier/software/records/receipts/`
and recomputes every maximum, fixed-bias score and paired bootstrap interval
reported for the main comparison and the boundary-refinement ablation with
the paper's own pooling and resampling code. It also reruns the paper's
Silver-dev operating-point selection on the receipts and checks each
system's chosen setting and its Gold-7 and Silver-test F1 against the paper
([docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md#o-logit-bias-sweep)).
It fails on any mismatch.
[records/README.md](research/pii/frontier/software/records/README.md) describes the receipts, O4's
text-free training membership, the run records and what they cannot show.

## Learn the method

| Document | Covers |
|---|---|
| [docs/ontology.md](research/pii/frontier/software/docs/ontology.md) | The 31 Ont3 types, reference types, native corpus labels and the many-to-many label map, auxiliary channels, migrating to a new inventory |
| [docs/data.md](research/pii/frontier/software/docs/data.md) | Public sources and licenses, supervision conventions (complete versus partial annotation, type masks, ignored spans), sampling branches and gold share |
| [docs/training.md](research/pii/frontier/software/docs/training.md) | The O4 recipe flag by flag, the fresh two-step fit, sampling, validation and selection, the GLiNER2 baseline |
| [docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md) | Redaction regions, overlap matching, the bias sweep, coverage and title policies, pooling and paired intervals, fixing an operating point |

## Other commands

`names` fetches the public name lexicons behind the name-component
postprocessor (U.S. Census surnames, INSEE given names, the SSA baby-name
mirror, JMnedict, Wiktionary name categories, Faker), builds the role
inventory and trains and exports the small name-kind model on CPU into
`WORK/name-kind`; `redact` and `evaluate` then use it. In our run it took about
half an hour and reached the paper's development-stage quality (0.871 test
macro F1). The paper's deployed copy was also continued on private text, so
yours will differ somewhat. Some publishers block some networks (the Census
server rejected our host); `names` then stops and tells you where to place a
file downloaded in a browser, and checks its hash against the paper's input.
`train-refiner` fits a character-boundary refiner for your tagger from
complete Ont3 rows; the shipped refiner is usually the better choice (see
its model card).
`train-gliner2` rebuilds the paper's adapted GLiNER2 baseline (GL4) on a
`mixture` directory by its recorded recipe, about 10 minutes of training on
one large GPU; `evaluate` and `calibrate` score its checkpoints. It is the
paper's baseline, not a recommended model: on Gold-7 the adapted model
scored below the published GLiNER2 model, which is scored only on the types
it has labels for. GL4 shuffles each training prompt's entity types, an
option GLiNER2's trainer leaves off; without it (`--unshuffled`, the paper's
first run) training collapses after a few hundred updates
([docs/training.md](research/pii/frontier/software/docs/training.md#the-gliner2-baseline-gl4)).
`select` draws new unlabeled candidate text from FineWeb and FineWeb-2: by
default the paper's source-only training draw, or, optionally, paragraphs
matching rare-identifier patterns (`--needles`) or resembling example text
from your domain (`--domain`); see
[docs/data.md](research/pii/frontier/software/docs/data.md#selecting-new-web-text).
`screen` compares your own candidate text with the human-gold evaluation,
any data you name and itself, using the paper's near-duplicate detector, and
writes the admission receipt `annotate` requires. `annotate` labels admitted
text with a Hugging Face teacher (Gemma-4 31B unless `--model` names
another) in-process or on an OpenAI-compatible server (`--server`), or with
an OpenAI model through your `codex login` (`--codex-model MODEL`; `--luna`
means `--codex-model gpt-6-luna`). Luna
(GPT-5.6-Luna and GPT-6-Luna) labeled O4's training rows: for annotation
quality like the paper's, use it. Gemma is the default only for convenience,
since it runs locally without an account. `--prompt-revision` selects one of
the paper's two prompts. `paper-prompted` (the default) is the prompt behind
the paper's prompted Gemma-4 31B result (`prompts/pii-label/paper-eval/`).
Served from `research/pii/frontier/software/serve/`, this command rebuilt
that run's prompts byte for byte on our private development rows and scored
within 0.3 F1 of the published numbers. The paper's prompted result was not
rerun with the later teacher prompt (its gain, one to two typed F1 points and
not statistically resolved, does not reverse any of the paper's
recommendations). `paper-teacher` is the prompt O4's teacher annotated its
training rows with, the one to use when building training data. Each run
writes `prompts-by-language.json`, the exact rendered prompt every input
language received.
`score-jsonl` scores any gold JSONL with a simpler generic evaluator. `fetch`,
`prepare` and `assemble` are the single-source steps behind `data`. `stage`
exports this package from its source repository.

Commands print one JSON result on stdout (`--pretty` for indented JSON).
A missing dependency or failed step is an error; the driver never silently
switches data, models or recipes.

## License and scope

The software and documentation are MIT licensed (`LICENSE.md`). Corpora,
model weights and third-party components keep their upstream terms. A fresh
fit on public data reproduces the recipe, not O4's weights: O4 also used
private training data and a longer training history.
