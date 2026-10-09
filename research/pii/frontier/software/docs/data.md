# Data: sources, supervision conventions and the training mixture

This document explains which corpora the pipeline uses, how their labels are
converted, how the training mixture is compiled, and what O4's own training
data consisted of. Paths are relative to the package root. The package ships
no corpus text or labels; every corpus is fetched from its upstream source
under that source's terms.

## Record formats

**Prepared (onboarded) corpora.** `fetch` downloads a pinned upstream
revision; `prepare` converts it to gzip JSONL shards at
`<root>/<corpus>/<split>/<lang>.jsonl.gz` plus a `manifest.json` with the
upstream revision, license, required citations, label counts and shard
hashes. Each record is:

```json
{"id": "...", "text": "...", "lang": "de",
 "spans": [{"start": 0, "end": 9, "label": "person_name", "source_label": "PER"}],
 "metadata": {...}}
```

`source_label` keeps the publisher's own label. The manifests produced for
the paper are shipped (text-free) under `data/pii-onboarded/*/manifest.json`,
so you can compare your prepared shards against them.

**Training rows.** The trainer reads `train.jsonl`, `val.jsonl` and
`labels.json` from one directory. A training row is a text window of at most
900 characters and 512 XLM-R tokens, with character-offset spans:

| Field | Meaning |
|---|---|
| `text`, `lang`, `id` | Window text, language code, identifier |
| `spans` | `[start, end, label]` triples; label is an Ont3 type or a map node |
| `label_space` | `v2` for rows labeled in Ont3 types; `v1` for rows supervised through the map |
| `supervision` | `complete` or `annotated_spans_only` (below) |
| `unknown_primary_types` | Ont3 types the row's source never annotates (complete rows only) |
| `ignored_spans` | `[start, end, reason]` character ranges that receive no loss |
| `sampling_weight` | Relative probability of drawing this row |
| `sampling_branch`, `sampling_pool` | Which branch and pool the row belongs to |
| `document_context` | `{before, after}` neighboring text; O4 trained without it |

## Supervision conventions

The central question for every row is: which unlabeled tokens are true
negatives? Getting this wrong is the most damaging data error in span
tagging.

### Illusory negatives

Suppose a corpus annotates only persons, organizations and locations:

```text
Contact Ana Ruiz (PER) at ana.ruiz@example.net before 4 June.
```

The email address and the date are unlabeled because the corpus has no such
categories, not because they are not personal data. If those tokens are
trained as `O`, the model is taught that emails and dates are not PII. These
are **illusory negatives**. Their effect is systematic: every row from the
corpus pushes the unannotated types toward `O`, and recall on them falls in
proportion to how much of the training mass the corpus carries.

The pipeline therefore records, per row, how far its `O` can be trusted.

### `annotated_spans_only`

Labeled spans are trusted; every unlabeled token is unknown and receives no
loss. Only the positive spans teach anything. This is the safe default for a
corpus whose coverage has not been examined. The cost is that such rows teach
no boundaries against `O` and no precision.

The trainer's `--partial-primary-objective-weight` scales the primary loss
from these rows (O4: 1.0, unchanged).

### Complete supervision with unknown types

Labeled spans are trusted, and unlabeled tokens are trusted as negatives for
the types the corpus annotates exhaustively. For every other type they are
unknown. The row lists those in `unknown_primary_types`. For an unlabeled
token the training target becomes the set {`O`, and any B/I/E/S label of an
unknown type}, and the loss rewards probability mass anywhere in that set.
Evidence against an annotated type is kept; nothing is claimed about the
others.

In the example above, if the corpus annotates `person_name`, `organization`
and the place types exhaustively, the token "example" is a trusted negative
for those types and unknown for `email`.

Which types each public human-gold corpus annotates exhaustively is recorded
in `research/pii/frontier/evidence/four-corpus-v1/negative-coverage-v1.json`:

| Corpus | Trusted negative types |
|---|---|
| OpenNER (non-NC subset), AQMAR | person_name, organization, admin_area, locality, location, street_address, postal_code, gps_coordinates |
| MAPA | the above plus date, date_of_birth, time, monetary_amount, quantity |
| Wojood (sample) | the OpenNER set plus date, date_of_birth, time, monetary_amount, url |

This is a judgment about each corpus, not a guarantee. For Wojood,
`demographic_attribute` is deliberately left unknown: its `OCC` and
`LANGUAGE` labels do not establish that every occupation or nationality word
is annotated.

### Ignored spans

`ignored_spans` removes loss from a character range regardless of label. Two
uses in the shipped pipeline:

- **Alternate nested positives.** Wojood has nested annotations (an
  organization inside a longer organization name). BIOES cannot represent
  nesting, so the compiler splits the spans into non-overlapping "flat views"
  and emits one row per view. Text covered by a positive in another view is
  ignored in this view, so the model is never told that a real entity is `O`.
- **MAPA titles.** <a id="mapa-titles"></a>MAPA's `PERSON` spans include job
  titles and roles ("Advocate General Doria Venn" as one span), while Ont3
  makes a title a
  separate `demographic_attribute`. The compiler uses MAPA's fine-grained
  layer to cut `ROLE` and `PROFESSION` runs out of person spans and marks them
  ignored; the remaining name keeps supervised boundaries. Honorifics (fine
  `TITLE`) stay inside the name, as the Ont3 rule says. The transform is
  `research/pii/frontier/evidence/title-extents-v1/mapa-train-titles.py`.

## Sources

`scripts/pii_onboard_sources.py` pins every source in its `SOURCES` table.
License terms below are copied from that table; check the upstream terms
yourself before redistributing anything derived from a corpus. The software's
MIT license does not cover data or model weights.

| Source | Upstream (revision prefix) | Languages | Terms recorded | Role |
|---|---|---|---|---|
| `openner-commercial-core` | `bltlab/open-ner-core-types` on Hugging Face (`59ce4c55fc54`) | de, en, es, ja, pt, sv, zh | CC-BY-4.0 collection; components CC-BY-4.0 or CC-BY-SA-4.0; cite OpenNER 1.0 | Human-gold branch; base (natural); evaluation |
| `mapa` | `joelniklaus/mapa` on Hugging Face (`bbb2a0157b76`) | 21 EU languages | CC-BY-4.0 | Human-gold branch; evaluation |
| `aqmar-openner` | `bltlab/open-ner-core-types`, AQMAR subset (`59ce4c55fc54`) | ar | Citation required (Mohit et al., 2012) | Human-gold branch; base (natural); evaluation |
| `wojood-sample` | `SinaLab/ArabicNER` on GitHub (`ef3a7f4e806a`) | ar | MIT | Human-gold branch; evaluation |
| `hiner` | `cfiltnlp/HiNER` on GitHub (`fdec0c85a6c3`) | hi | CC-BY-SA-4.0 | Base (natural) |
| `idner-news-2k` | `khairunnisaor/idner-news-2k` on GitHub (`625ff40f85b0`) | id | MIT | Base (natural); demo |
| `klue-ner` | `KLUE-benchmark/KLUE` on GitHub (`3efd98708a40`) | ko | CC-BY-SA-4.0; cite Park et al. (2021) | Available adapter; not in the default pipeline |
| `nemotron-pii` | `nvidia/Nemotron-PII` on Hugging Face (`b70ffaf5ff39`) | en | CC-BY-4.0 | Available adapter; not in the default pipeline |
| `openpii-1m` | `ai4privacy/pii-masking-openpii-1m` on Hugging Face (`ecfdc547f4a0`) | 23 languages | CC-BY-4.0 | Available adapter; not in the default pipeline |
| `ai4privacy-health-phi-400k-sample-1k` | `ai4privacy/pii-masking-health-phi-400k` (`f1c06d3062df`) | 30 languages | ai4privacy commercial terms; local file only | Available adapter; requires a separately obtained file |

We use seven components of OpenNER's core-types release (source slug
`openner-commercial-core`), all licensed CC BY 4.0 or CC BY-SA 4.0;
noncommercial (-NC) components are excluded. They are AnCora (Spanish),
GermEval 2014
(German), Japanese GSD, and the Universal NER English EWT, Portuguese Bosque,
Swedish Talbanken and Simplified Chinese GSD treebanks; each component's
license and attribution are listed in `OPENNER_COMMERCIAL_CORE_COMPONENTS`.
Some adapters drop labels outside the PII scope (for example HiNER's
`FESTIVAL`, `GAME`, `LITERATURE`); see each entry's `ignored_labels`.

The base model is `FacebookAI/xlm-roberta-large` from Hugging Face; it keeps
its own license.

## Commands

All commands run through `python pii-reproduce.py` and write under one work
directory (`--work`, default `work/`), with a log directory and a receipt per
step.

**`data [--demo]`** fetches and prepares the four human-gold corpora
(OpenNER's non-NC subset, MAPA, AQMAR, Wojood) and the two base-only corpora
(HiNER, IDNER) into `WORK/onboarded`, then assembles the base rows from the
natural training splits of OpenNER, AQMAR, HiNER and IDNER into `WORK/base`.
"Natural" means the publisher's own sentences and labels, as opposed to
translated or teacher-annotated text. `--demo` uses AQMAR, Wojood and IDNER
only. Already prepared corpora are skipped.

The lower-level steps are also exposed for one source at a time:
`fetch --source S --source-root DIR --log-dir L`, `prepare --source S
--source-root DIR --output-root OUT --log-dir L [--max-records-per-source N]`
and `assemble --source NAME --prepared-root OUT --out DIR`.

**`human-gold [--demo]`** rebuilds the 1,283 public evaluation rows; see
[evaluation.md](evaluation.md#the-human-gold-population).

**`mixture [--gold-share F] [--o4-membership] [--annotated FILE] [--demo]`**
compiles an O4-style training directory in `WORK/mixture` with
`scripts/pii_public_mixture.py`. It needs `data` and `human-gold` first.
`--annotated` adds your own teacher-annotated rows (below).

## How the mixture is compiled

O4 samples every training draw from one of two **branches**: the human-gold
branch (publisher-annotated public corpora converted through the map) or the
base branch (everything else). `--gold-share` is the probability of drawing
from the human-gold branch; O4 used 0.5.

**Human-gold branch.** For each of the four gold corpora, the compiler reads
the publisher **train** split in stable order and:

1. converts each native label to its map node, for example `openner_core__LOC`
   ([ontology.md](ontology.md#native-source-labels-and-the-many-to-many-map));
   the row is labeled `label_space: v1`;
2. cuts it into windows of at most 900 characters and 512 tokens without
   splitting a span, and fails if a span is lost;
3. splits nested annotations into flat views, ignoring alternate positives;
4. marks each window `complete` with that corpus's `unknown_primary_types`;
5. for MAPA, cuts titles out of person spans as described above;
6. removes any row whose normalized text equals a human-gold evaluation row.

**Base branch.** The assembled rows in `WORK/base/train.jsonl` must all be
`annotated_spans_only`; the compiler refuses otherwise, because it cannot
verify their negatives. Labels that are not Ont3 primary types (HiNER's
language and religion labels, for example) are dropped, which is harmless for
partial rows: removing a positive creates no negative. The same exact-text
screen applies.

The natural base rows are converted with the older unified tagset
(`scripts/pii_tagset.yaml`), which maps each `LOC` to the single Ont3 type
`location`, not to the six-type accepted set. A city in these rows is
therefore trained as `location`. The same public corpora appear with the
many-to-many map in the human-gold branch. No measurement of this mismatch's
effect is included in the package.

`mixture --annotated FILE` adds teacher-annotated rows
labeled in Ont3 types to the base branch. It refuses rows carrying other
labels, since such rows may be complete supervision and dropping a positive
would teach a false negative.

**Weights.** Within each branch, rows start with equal weight. Rows with the
same language and exact text (repeated annotations, or the flat views of one
nested row) share one input's weight instead of each receiving a full share,
so reannotating data does not multiply its exposure
(`scripts/pii_annotation_sampling.py`). Each branch's weights are then scaled
to sum to its share.

**Language round.** The trainer refuses to start unless every declared core
language reaches a minimum share of sampling mass. O4's recorded declaration
names the 35 languages of its full training pool; public data covers far
fewer. The compiler therefore writes `WORK/mixture/language-round.yaml`
declaring the languages whose realized share is at least 0.5%
(`--language-floor`), and the training recipes substitute it for O4's.

**Validation.** `val.jsonl` holds up to 2,000 windows of the assembler's
validation split, relabeled as Ont3 (`label_space: v2`) so the trainer can
select checkpoints by span F1 in the head's own inventory. These rows are
split from the base by row, not by document, and carry only their
publisher's types, so their scores are a relative selection signal.

**Root rows.** With base rows present, the compiler also writes
`WORK/mixture/root/`, the base rows alone labeled natively in Ont3. The
`o4-fresh` recipe uses them to create the Ont3 head
([training.md](training.md#recipes)).

**`--o4-membership`** keeps exactly the human-gold training rows that O4
used, read from `research/pii/frontier/software/records/o4-training-membership.csv`.
O4's rows passed a semantic near-duplicate screen against evaluation data,
which is stronger than the exact-text screen applied otherwise.

`WORK/mixture/receipt.json` records row counts per branch and corpus, removed
rows, dropped labels, realized branch probabilities, language shares and the
hashes of every input and output.

## What O4 was trained on

O4 is a continuation: 4,000 updates from a parent checkpoint that had itself
been trained on earlier mixtures. The continuation pool is described, without
text or labels, by `research/pii/frontier/software/records/o4-training-membership.csv`
(one line per training window: identifiers, sampling weight, branch, language,
window offsets and SHA-256 hashes of the text). It has 171,168 windows in 35
languages. Shares of sampling mass, computed from that file:

| Branch | Content | Windows | Share of draws |
|---|---|---|---|
| Human gold | OpenNER non-NC subset, train split | 67,344 | 41.8% |
| Human gold | MAPA, train split | 11,312 | 7.0% |
| Human gold | AQMAR, train split | 1,263 | 0.8% |
| Human gold | Wojood sample, train split | 910 | 0.5% |
| Base | Public web text (FineWeb-2, FineWeb) with private teacher annotations | 50,369 | 24.2% |
| Base | Public corpora: natural OpenNER, HiNER, IDNER, AQMAR; TAB; Nemotron-PII; MEDDOCAN | 25,070 | 11.5% |
| Base | Other locally identified pools (reviewed and adjudicated annotation sets, translated transfers, earlier annotation intake) | 14,900 | 14.4% |

Only the human-gold half can be rebuilt exactly from public sources: the
public corpora, the map and `--o4-membership` reproduce it. For the web-text
rows the record gives the upstream record and the row's offsets inside it,
so `fetch-web` recovers the exact text (below), but the Ont3 annotations were
produced by a commercial LLM teacher and are not shipped. The remaining rows have local identities only; a hash
lets you verify a text you obtained but does not retrieve it. A mixture built
from public data alone therefore replaces roughly half of O4's draws with a
much smaller, coarser base branch: a few label types, fewer languages, and
partial supervision only.

## Annotating O4's web text yourself

```bash
python pii-reproduce.py fetch-web [--language el ...]
python pii-reproduce.py annotate --prompt-revision paper-teacher \
    --input work/web/rows.jsonl --web-receipt work/web/receipt.json \
    --out work/teacher [--limit N]
python pii-reproduce.py mixture --annotated work/teacher/training-rows.jsonl --out work/mixture-teacher
```

`fetch-web` finds each row's upstream record, cuts the row out by the
recorded offsets and keeps it only if its SHA-256 equals the recorded
training-text hash. It first streams each needed FineWeb or FineWeb-2
configuration and replays the recorded seeded draws, which finds most records
quickly; records still missing are then located exactly by reading only the
`id` column of every data file of that configuration (`--lookup-threads`).
Some intake batches recorded offsets shifted by a character or two, or on
NFKC-normalized text (common for Chinese, Japanese and Thai); the command then
tries nearby slices and the normalized form, again accepting only an exact
hash match, and counts these repairs in its receipt. Rows without recorded
offsets cannot be recovered.

`annotate --web-receipt` accepts only rows listed in that receipt, since
these texts already passed the paper's screening as O4 training rows. With
`--prompt-revision paper-teacher` it runs the prompt O4's teacher used
(`prompts/pii-label/`; the default revision is the paper's prompted-evaluation
prompt in `prompts/pii-label/paper-eval/`, not the teacher's) and writes
`training-rows.jsonl`: Ont3 spans, marked as complete supervision because the
prompt asks for every type, as O4's teacher rows were. Rows whose output could
not be parsed cleanly are left out. Any teacher that labels all 31 types in
this format can be used.

### Choosing a teacher

The teacher's quality bounds the tagger's. O4's teacher rows came from Luna
(GPT-5.6-Luna and GPT-6-Luna). On the same development segments, prompted
Gemma-4 31B emitted only 82% as many spans as Luna (the paper's teacher
density table), and on the Ont3 evaluation rows the trained O4 tagger scored
about 5 or more F1 above every prompted Gemma-4 31B variant we ran, including
the later prompt. For annotation like the paper's, use Luna through `codex`.
Gemma-4 31B is the default only because it runs locally without an account.

**Luna or another OpenAI model.** `--codex-model MODEL` (`--luna` for
`gpt-6-luna`) sends each row to a fresh Codex session at low reasoning
effort, six at a time by default (`--concurrency`), as the paper's Luna
batches ran. It needs only an ordinary `codex login`:

```bash
codex login            # once; or: printenv OPENAI_API_KEY | codex login --with-api-key
python pii-reproduce.py annotate --codex-model gpt-6-luna --prompt-revision paper-teacher \
    --input work/web/rows.jsonl --web-receipt work/web/receipt.json --out work/teacher
```

The annotator must not see your own Codex instructions, memories, plugins or
skills, so `annotate` runs it in a separate home, `~/.codex-pii-annotate`,
which it builds or refreshes before every run (`python pii-reproduce.py
codex-home` does only that step). By hand, the same home is:

```bash
mkdir -p -m 700 ~/.codex-pii-annotate
touch ~/.codex-pii-annotate/.pii-reproduce-codex-home       # marks it driver-managed
cp scripts/codex-annotation-home.config.toml ~/.codex-pii-annotate/config.toml
install -m 600 "${CODEX_HOME:-$HOME/.codex}/auth.json" ~/.codex-pii-annotate/auth.json
rm -r ~/.codex-pii-annotate/{memories,plugins,skills} ~/.codex-pii-annotate/AGENTS*.md  # if present
```

The configuration gives the model no tools, memory, web search or external
servers. Nothing else from your own Codex home is copied; Codex recreates
some of that context while it runs, which the refresh clears and the runner
refuses (`--codex-home DIR` uses a home you maintain yourself, under the
same refusal). A login refreshed in `~/.codex` reaches the annotator at the
next run.

At GPT-6-Luna's list prices the paper's whole annotation volume, about
50,000 requests, would cost about $53 (the paper's annotation appendix). The
paper's batches also grouped consecutive sentences of one document into a
single session to share the prompt; the shipped runner supports that
(`scripts/pii_api_label.py --session-field`), but `annotate` sends one row per
session.

**A local Hugging Face model.** By default the teacher (Gemma-4 31B, or the
model `--model` names) runs in-process, one document at a time (about 2.5
labeled tokens per second on one GPU). For the full web set, serve the
teacher with a batching server and pass its URL:

```bash
cd research/pii/frontier/software/serve && pixi install --locked
pixi run serve-teacher      # vllm serve google/gemma-4-31B-it --port 8000 ...
# in another shell, from the main environment:
python pii-reproduce.py annotate --server http://127.0.0.1:8000/v1 \
    --prompt-revision paper-teacher \
    --input work/web/rows.jsonl --web-receipt work/web/receipt.json --out work/teacher
```

`--server` sends the same prompt with thinking disabled and greedy decoding
to any OpenAI-compatible endpoint, `--concurrency` requests at a time
(default 64), and parses the answers exactly as the in-process route does.
The prompt (tag catalog and examples) is about 6,700 tokens, so the server
needs a context of at least 8,300 tokens; it is identical across rows, so a
server with prefix caching (vLLM's default) computes it once. On one 96 GB
GPU this annotated about 10 rows per second (512 English rows in 53
seconds), so the whole web set takes about an hour and a half.

## Selecting new web text

`select` draws fresh, unlabeled candidate text from the pinned FineWeb
(English) and FineWeb-2 snapshots into `WORK/select/NNN/rows.jsonl`, ready
for `screen` and then `annotate`:

```bash
python pii-reproduce.py select --language el,ko --rows 200
python pii-reproduce.py screen --input work/select/001/rows.jsonl
python pii-reproduce.py annotate --codex-model gpt-6-luna --prompt-revision paper-teacher \
    --input work/screen/rows/retained.jsonl --dedup-receipt work/screen/rows/receipt.json \
    --out work/teacher-new
```

By default it is the paper's own training draw: it streams documents in a
seeded order, keeps only documents whose hash falls in the training buckets,
segments them into sentences and samples sentences per language in three
strata (10% containing an identifier-like pattern, 20% with several
reference cues such as pronouns or role nouns, 70% anything else). This
default is intentionally source-only: it decides from the crawl text alone and
never looks at evaluation text or labels. Every run excludes the rebuilt
human-gold evaluation, `fetch-web` rows and earlier selections (add more with
`--exclude`), so repeated runs return new text.

Two optional selectors target text instead. Both read further into the
stream on each run: a cursor and a used-region ledger under `WORK/select/state`
record which documents and paragraphs are taken.

- `--needles [FILE[:WEIGHT]]` keeps paragraphs that match rare-identifier
  surface patterns (IBANs, card numbers, IP addresses and the like, with
  checksum and nearby-cue checks), and emits every sentence of each kept
  paragraph. The default set, `data/pii-needles/rare-type-needles-v1.json`,
  targets types that were thin in the paper's training data. Pass several
  files, repeated or comma-separated; they are concatenated, and a file's
  weight multiplies its needles' weights. We used needles to add training
  text but did not test whether they improve the tagger, so treat them as
  a suggestion to try when adapting to a domain whose identifiers you can
  describe as patterns. Selection is slower than the default draw, since
  most documents contain no match.
- `--domain [FILE[:WEIGHT]]` keeps the paragraphs most similar to example
  text from the domain you care about: JSONL rows with `text` (and optional
  `lang`), or plain text with one example per line. Without a file it uses
  the rebuilt human-gold evaluation text, which steers selection toward the
  evaluation domains; it reads only that text, not its labels, but it does
  deliberately select text resembling the evaluation, so keep `screen` in the
  path. Each paragraph is scored by the mean multilingual-E5 cosine of its five
  nearest examples, minus its mean cosine to its five nearest fellow
  candidates (generic text such as site navigation is near everything and
  would otherwise rank first). Examples in the paragraph's own language are
  used when any exist, else all examples; the cross-language fallback picks
  noticeably less on-domain text, so give examples in each language you
  select. Several files are pooled; a file's
  weight counts each of its examples that many times among the five nearest,
  so weight 2 behaves like listing its lines twice and 0.5 like half a copy.
  This is the paper's domain-near retrieval. Its benefit was not established
  either: in our one contrast, adding domain-targeted rather than random
  text moved human-gold F1 by under half a point, in both directions.

`--scan` bounds the documents read per language in one run. Selection only
proposes text; screening and annotation decide what becomes training data.

## Annotating your own text

`annotate` refuses text that has not been screened against evaluation data.
For your own candidate rows (JSONL with `id`, `text`, `lang`):

```bash
python pii-reproduce.py screen --input my-candidates.jsonl \
    [--compare training=my-train=work/mixture/train.jsonl ...]
python pii-reproduce.py annotate --prompt-revision paper-teacher \
    --input work/screen/my-candidates/retained.jsonl \
    --dedup-receipt work/screen/my-candidates/receipt.json --out work/teacher-mine
```

`screen` runs the paper's near-duplicate detector: character n-gram F1 at
least 0.30 or multilingual-E5 cosine at least 0.875, among each candidate's
three nearest neighbors in either view. It compares candidates with the
rebuilt human-gold evaluation (`WORK/human-gold`, always included) and any
`--compare ROLE=NAME=PATH` sources you name, and within the batch itself.
Candidates that overlap anything are dropped, as is every member but the first
of a within-batch near-duplicate group. It writes `retained.jsonl` and the
receipt that `annotate` checks. The receipt lists which roles you declared no
data for; screening only proves separation from what you named.

Our fresh-fit experiment replayed our own cached teacher labels for these
rows; results are in the guide's "What a fresh fit reaches".

## Practices for adding data

These practices protect evaluation validity and label quality. The shipped
code enforces some of them.

- **Deduplicate before annotating or admitting data.** Exact hashes miss
  near-duplicates such as re-punctuated or lightly edited copies. The paper's
  pipeline compared each candidate with existing training, validation and
  evaluation text using lexical and multilingual-embedding nearest neighbors
  (`scripts/pii_overlap_neighbors.py`, `scripts/pii_overlap_filter.py`) and
  recorded the decisions in a receipt. `screen` runs that detector for you;
  `annotate` refuses to run without its receipt (`--dedup-receipt`), a
  `fetch-web` receipt, or an explicit reannotation or evaluation-replay
  receipt.
- **Repair contaminated validation by removing rows from validation, not
  from training.** Training data is scarce; a validation split is cheap to
  rebuild.
- **Keep the provenance of labels.** The pipeline keeps labels from local LLM
  teachers out of training until a separate quality check admits them; an
  annotation run does not by itself admit its output.
- **Check language support on the realized mix.** Count sampled tokens per
  language after weighting, not rows before it.

## Known limitations

- Publisher splits are inherited, not re-split by document. OpenNER divided
  AnCora by sentence, so most AnCora evaluation documents also contribute
  training sentences; the EWT, Bosque, AQMAR and MAPA splits are
  document-disjoint. Several corpora (GermEval, Japanese and Chinese GSD, the
  Wojood sample, HiNER, IDNER) have no document identifiers at all.
- The exact-text screen does not catch near-duplicates between training and
  evaluation; use `--o4-membership` for the human-gold branch and `screen`
  for text you add.
- Negative coverage is a per-corpus judgment and may be wrong for rare
  subtypes.
