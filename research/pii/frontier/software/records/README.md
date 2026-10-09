# Evidence shipped with the software

Everything here is text-free: no corpus text, annotation labels, span
surfaces or model weights. A SHA-256 hash commits to a withheld file without
disclosing it. [CLAIMS.md](CLAIMS.md) maps each headline number in the paper
to its evidence and says whether `verify` recomputes it.

## Score receipts: check the paper's numbers

`receipts/` holds the per-document counts behind the paper's comparisons.
Every reported score is pooled from `[true positives, predicted regions, gold
regions]` per document before F1 is computed, so these counts determine the
maxima, fixed-bias scores and paired bootstrap intervals exactly.

| Receipt | What the paper reports from it |
|---|---|
| `o4-comparison.json.gz` | Main comparison: 13 systems on 1,283 human-gold and 1,201 Ont3 development segments; O3 versus O4 intervals |
| `o4-boundary.json.gz` | Character-boundary refinement on and off for O4 |
| `gliner-trajectory.json.gz`, `gliner-trajectory-unshuffled.json.gz` | GL4's checkpoints (shuffled type order) and the first, unshuffled run's, on Gold-7 and Silver-dev (development trajectories) |
| `operating-point-presidio.json.gz`, `operating-point-refined-grid.json.gz`, `operating-point-fine-bias.json.gz` | Finer threshold and bias grids for Presidio, GLiNER2, GL4, O3, O4 and OpenMed Privacy Filter |
| `operating-points-trust-region.json` | Each main-comparison system's operating point fixed on Silver-dev, and its scores there and on Gold-7 and Silver-test |

Each receipt stores document ids, languages and source groups once per
population and one flat count array per system, threshold and matching view.
Ids of human-gold rows are the public corpora's own test-split ids; other ids
are opaque hashes. Prediction files, gold files and checkpoints appear only as
basename and SHA-256. `receipts.json` binds each receipt to the hash of the
score archive it was derived from. Building the receipts proves that they
expand back to exactly the archived counts and that no content-like string in
them occurs in any evaluated document.

```bash
python pii-reproduce.py verify            # recompute the reported numbers
python pii-reproduce.py verify --details  # list each recomputed value
```

`verify` uses the paper's own pooling (`pool_ont3`) and resampling (`paired`,
10,000 resamples of source groups, fixed seed) from
`scripts/pii_paper_o4_figures.py`, and fails if any number differs from the
`*-summary.json` values the paper was written from. It also merges the finer
grids over `o4-comparison`, reruns the paper's Silver-dev operating-point
rule (`operating_points` there), and fails unless the result equals
`operating-points-trust-region.json` and gives the paper's chosen settings
and Gold-7 and Silver-test F1.

What a receipt cannot show: that the withheld gold labels are correct, or
that the counts came from the stated predictions. For the human-gold
population a reader can close most of that gap: `human-gold` rebuilds the
exact 1,283 evaluated rows from public test splits, and `evaluate` scores any
checkpoint with the same code into the same per-document schema, so a
reader's own model lines up with O4 row by row.

## Title-extent sidecar

`title-extents-human-gold.jsonl` lists, for 958 public test rows, character
offsets of job titles and descriptors beside person names (`kind: neutral`)
that the paper's title policy leaves unscored. It holds ids and offsets only.
Its SHA-256 equals the `title_sidecar` hash in the O4 receipts, so
`evaluate` scores your model exactly as the paper scored O4 and can pair the
two.

## Training membership of O4

`o4-training-membership.csv` has one entry per O4 training row, in input
order: training id, sampling weight and branch, source id, window offsets and
the SHA-256 of the row text. It distributes no text or labels. Training ids
repeat in the input; `pool_row_1based` distinguishes rows. The adjacent JSON
binds the index to the original pool hash.

Of 171,168 rows, 50,369 carry an upstream pointer (public FineWeb or
FineWeb-2 dataset, configuration, revision, split and record id); 50,036 of
those also give the character offsets of the row inside the record
(`upstream_document_start`, `upstream_document_end`). `pii-reproduce.py
fetch-web` uses them to recover the exact training text, keeping a row only
when its SHA-256 matches `training_text_sha256`. On our host it recovered
48,308 of the 50,036 offset-bearing rows (96.5%, 1,226 after repairing
offsets shifted by a character or two) in about nine minutes. Most misses are
English FineWeb records outside the first 500,000 streamed (raise
`--max-scan`) and Chinese, Japanese and Thai rows whose offsets do not
reproduce the hashed text. The other 120,799 rows have local identities only. The
human-gold rows among the latter are public: their training ids
(`<corpus>-train:<line>`) name lines of the public corpora's train splits,
and `pii-reproduce.py mixture --o4-membership` rebuilds exactly O4's 80,829
human-gold training windows from them. The remaining rows come from
annotation of non-public or not yet re-locatable text; a hash verifies text
you obtain but does not retrieve it.

## Run records

`paper-run-records.json` contains selected saved run records, including
O4's final training stage and GL4's runs (training windows, selector
windows, training with shuffled type order), the first GL4's unshuffled
training run, and those of the accepted-label variant. It keeps
structured commands, metrics and source provenance and omits free-text logs.
`pii-reproduce.py train --recipe o4` reads O4's trainer options from these
records, and `train-gliner2` GL4's. Historical absolute paths describe the
original run and must be rebound to local inputs.

## Source manifests

`data/pii-onboarded/<corpus>/manifest.json` (shipped at those paths) records,
for each public corpus, the pinned upstream revision, license, label
projection hashes, counts and the SHA-256 of every converted shard. After
`pii-reproduce.py data`, compare your `WORK/onboarded/<corpus>/manifest.json`
with the shipped one to confirm that you converted identical data.

## Limits

The receipts and records do not reconstruct O4's full ancestor trajectory,
and a fresh fit on public data is a reproduction of the recipe, not of O4's
weights. Corpora and model weights keep their upstream terms; the software's
MIT license does not relicense them.
