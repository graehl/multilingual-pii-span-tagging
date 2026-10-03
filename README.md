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
- `evaluate` runs the paper's prediction sweep and scorer on the human-gold
  view and reports the curve maximum and the fixed zero-bias point beside
  O4's 88.8 and 88.2 F1.
- `redact` emits typed spans and a redacted copy of each input line. With
  `--work` it only emits types the training data supervised; `--serve` adds
  the paper's serving stages: the shipped character-boundary refiner
  (`research/pii/frontier/software/models/boundary-refiner/`, worth about
  +2 region F1 on our 31-type set for a fresh model), name-component
  postprocessing when you have built the name-kind model, and regex
  supplementation for identifiers. `train-refiner` fits your own refiner.

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

We ran this pipeline from a staged copy of this package on one 96 GB GPU:
`train --recipe o4-fresh --steps 12000` from pretrained XLM-R large, about
2 hours 15 minutes per fit with two fits sharing the GPU. Human-gold F1 at 80%
overlap, the paper's view:

| Training data | Max | Zero bias | Versus O4 at zero bias, exact regions |
|---|---|---|---|
| O4 (paper) | 88.8 | 88.2 | — |
| Public gold + public corpora + O4's web text with teacher labels | 89.5 | 89.4 | +1.4 [0.3, 2.5] |
| Public gold + public corpora only | 86.6 | 82.2 | −6.2 [−8.1, −4.4] |

The web-text run used our cached teacher labels as a stand-in for yours. It
already matched O4 after 4,000 updates (88.1 at zero bias) and kept improving
to 12,000. On our private 31-type development set the same model matches O4
without character-boundary refinement and trails served O4 by about 2.7
points, the size of that refinement's gain. Public gold alone reaches its best
score only at a strongly shifted O-logit bias and fails on types the public
corpora never annotate: the teacher-labeled text supplies both the missing
types and the trusted negatives. Human gold comes from the publishers' test
splits, which never train the model: training uses the same corpora's train
splits, screened against the evaluation rows. It was reused during
development, though we do not believe O4 was meaningfully over-selected on
it: on 542 freshly annotated segments never used to select it, O4 keeps a
10 to 12 F1 lead over the adapted GLiNER2 baseline. As with any train/test
split of identically annotated data, human gold is
in-distribution and overstates accuracy on other domains; see
[docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md#qualifications).
Details, including the intermediate checkpoints and the 31-type results:
`research/pii/frontier/software/records/receipts/fresh-fit-summary.json`.

## Check the paper's numbers

`verify` expands the receipts in `research/pii/frontier/software/records/receipts/`
and recomputes every maximum, fixed-bias score and paired bootstrap interval
reported for the main comparison and the boundary-refinement ablation with
the paper's own pooling and resampling code. It fails on any mismatch.
[records/README.md](research/pii/frontier/software/records/README.md) describes the receipts, O4's
text-free training membership, the run records and what they cannot show.

## Learn the method

| Document | Covers |
|---|---|
| [docs/ontology.md](research/pii/frontier/software/docs/ontology.md) | The 31 Ont3 types, reference types, native corpus labels and the many-to-many label map, auxiliary channels, migrating to a new inventory |
| [docs/data.md](research/pii/frontier/software/docs/data.md) | Public sources and licenses, supervision conventions (complete versus partial annotation, type masks, ignored spans), sampling branches and gold share |
| [docs/training.md](research/pii/frontier/software/docs/training.md) | The O4 recipe flag by flag, the fresh two-step fit, sampling, validation and selection |
| [docs/evaluation.md](research/pii/frontier/software/docs/evaluation.md) | Redaction regions, overlap matching, the bias sweep, coverage and title policies, pooling and paired intervals |

## Other commands

`names` fetches the public name lexicons behind the name-component
postprocessor (U.S. Census surnames, INSEE given names, the SSA baby-name
mirror, JMnedict, Wiktionary name categories, Faker), builds the role
inventory and trains and exports the small name-kind model on CPU into
`WORK/name-kind`; `redact --serve` then uses it. In our run it took about
half an hour and reached the paper's development-stage quality (0.871 test
macro F1). The paper's deployed copy was also continued on private text, so
yours will differ somewhat. Some publishers block some networks (the Census
server rejected our host); `names` then stops and tells you where to place a
file downloaded in a browser, and checks its hash against the paper's input.
`train-refiner` fits a character-boundary refiner for your tagger from
complete Ont3 rows; the shipped refiner is usually the better choice (see
its model card).
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

## Command reference

```text
usage: pii-reproduce.py [-h] [--no-commentary] [--text] [--verbose N] [-v]
                        [--acli-quiet]
                        [--format {compact,jsonl,pretty,toon,text} | --compact | --json | --pretty | --toon]
                        [--full]
                        {doctor,readme,files,stage,package,publish,install,fetch,prepare,annotate,codex-home,train,score-jsonl,licenses,data,human-gold,mixture,fetch-web,select,screen,names,train-refiner,evaluate,redact,verify,demo,fetch-model,assemble}
                        ...

Portable PII reproduction workflow.

positional arguments:
  {doctor,readme,files,stage,package,publish,install,fetch,prepare,annotate,codex-home,train,score-jsonl,licenses,data,human-gold,mixture,fetch-web,select,screen,names,train-refiner,evaluate,redact,verify,demo,fetch-model,assemble}
    doctor              Report local prerequisites without loading models.
    readme              Write this help as the staged package README.md.
    files               List the local Python dependency subset (no data or builds).
    stage               Stage repository-relative sources and README; optionally anonymize for review.
    package             Stage into OUT/software, build OUT/software.tgz and verify a fresh extraction.
    publish             Stage the public release into a git repository for review and commit (no commit made).
    install             Create the pinned Pixi environment (Linux, CUDA 13).
    fetch               Fetch a pinned source.
    prepare             Convert a source with its existing ontology map.
    annotate            Annotate with a Gemma-4 31B teacher (default; in-process or --server) or a --codex-model such as Luna, O4's teacher.
    codex-home          Build or refresh the isolated Codex home annotate --codex-model uses (it runs this itself).
    train               Fit the existing affine BIOES encoder on a prepared training directory.
    score-jsonl         Predict and score any gold JSONL with the generic evaluator (not the paper's view).
    licenses            List every data source's license terms and record your approvals; fetches require approval.
    data                Fetch and convert the public corpora into WORK/onboarded and assemble base rows in WORK/base.
    human-gold          Rebuild the paper's 1,283-row public human-gold evaluation in WORK/human-gold.
    mixture             Compile an O4-style training directory in WORK/mixture.
    fetch-web           Recover the exact public web text of O4's teacher-annotated rows into WORK/web for your own annotation.
    select              Draw new unlabeled candidate text from FineWeb / FineWeb-2 into WORK/select/NNN for screen.
    screen              Screen your own text against evaluation and training data; writes the receipt annotate requires.
    names               Fetch approved public name lexicons and build the name-kind model into WORK/name-kind (CPU).
    train-refiner       Fit a character-boundary refiner for your tagger from complete Ont3 rows (e.g. annotate output).
    evaluate            Score a checkpoint on the paper's human-gold view and compare with O4.
    redact              Tag and redact raw text (one document per line, or JSONL).
    verify              Recompute the paper's reported scores and intervals from the text-free receipts.
    demo                Run the whole pipeline at minutes scale on three small corpora; proves the code works.
    fetch-model         Prefetch model weights into the Hugging Face cache (optional).
    assemble            Assemble selected prepared sources into train/validation JSONL and labels.

options:
  -h, --help            show this help message and exit
  --no-commentary       Omit commentary metadata and standalone JSONL commentary records; keep ordinary result data.
  --text                Prefer concise readable text; verbs without a text renderer may still output JSON. Explicit encoding flags take precedence.
  --verbose N           Verbosity level (default: 0); silently ignored by tools without verbosity support.
  -v                    Set --verbose=1.
  --acli-quiet          Suppress the `# acli: ...` stderr banner (env: ACLI_QUIET).
  --format {compact,jsonl,pretty,toon,text}
                        Output format. compact/jsonl is JSON Lines; pretty is indented JSON; toon is flat-table TOON; text prefers readable output with JSON fallback.
  --compact             Output compact JSON Lines.
  --json                Output compact JSON Lines (accepted even when already the default).
  --pretty              Output indented JSON.
  --toon                Output TOON; valid only for table-producing subcommands.
  --full                Include the full structured schema instead of the minimal default.
```

acli: 1 complete
