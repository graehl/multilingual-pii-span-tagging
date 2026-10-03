# Where each headline number comes from

Paths are relative to `research/pii/frontier/software/records/`. **Recomputed**
means `python pii-reproduce.py verify` rebuilds the number from per-document
counts with the paper's code and fails on any difference. **Recorded** means
the shipped file holds the number and its inputs' hashes but `verify` does
not recompute it; the file's per-item counts, where present, let you do so.
**Reproducible** means you can regenerate the number yourself from public
data with this software.

Human gold is 1,283 publisher test segments in seven languages from four
public corpora, reused during development. Ont3 is 1,201 development segments
with our annotations (not released). Scores are untyped redaction-region F1
at 80% overlap unless marked typed; "max" is the best point on each system's
confidence or O-bias curve, "fixed" the predeclared operating point.

| Claim | Value | Evidence | Status |
|---|---|---|---|
| O4 human-gold F1 (max / fixed) | 88.83 / 88.16 | `receipts/o4-comparison.json.gz` | Recomputed; reproducible: `human-gold` + `evaluate` on O4-style models gives the same per-document schema |
| O4 Ont3 F1 | 82.13 | same | Recomputed |
| GLiNER2 human gold / Ont3 | 69.08 / 66.00 | same | Recomputed |
| GLiNER2 adapted to Ont3 (GL4) human gold / Ont3 | 67.79 / 61.94 | same | Recomputed |
| Presidio human gold / Ont3 | 57.27 / 44.27 | same | Recomputed |
| OpenMed multilingual v2 human gold / Ont3 | 35.80 / 39.41 | same | Recomputed |
| O3, Ont1, Ont2 and the other privacy filters | see `verify --details` | same | Recomputed |
| O4 minus O3, exact regions, human gold | +1.25 [0.19, 2.34] | `receipts/o4-comparison-summary.json` `paired` | Recomputed (10,000 source-group resamples) |
| O4 minus O3, exact typed spans, Ont3 | +1.79 [0.51, 3.13] | same | Recomputed |
| O4 minus O3, exact regions, Ont3 | +0.55 [−0.62, 1.74] | same | Recomputed |
| Character-boundary refinement, typed Ont3 | +1.68 [1.01, 2.41] | `receipts/o4-boundary.json.gz`, `o4-boundary-summary.json` | Recomputed |
| Boundary refinement, regions Ont3 / human gold | +2.49 [1.58, 3.50] / −0.07 [−0.23, 0.00] | same | Recomputed |
| GLiNER2 with its full inventory, no type exemption | 68.8 | `receipts/gliner-full-inventory.json` `full_inventory.maximum` | Recorded |
| Mapped human-gold replay gains | per mixture | `receipts/coverage-mixtures-summary.json` | Recorded |
| Local LLMs as annotators and frozen-encoder heads | per model | `receipts/local-llm-summary.json`, per-row counts in `local-llm-per-input.json.gz` | Recorded; recomputable from the per-row counts |
| Human inter-annotator agreement on TAB | 85.5% over 84 cases | `receipts/tab-agreement.json` (per-case counts, public TAB case ids) | Recorded; recomputable |
| O4 training volume: human-gold rows / unique texts | 80,829 / 80,099 | `receipts/annotation-volume.json`; `o4-training-membership.csv` | Recorded; unique counts recomputable from the membership hashes |
| O4 human-gold training windows | 80,829 | `mixture --o4-membership` | Reproducible exactly from public corpora |
| Teacher annotation cost | aggregate | `receipts/annotation-cost.json` | Recorded (totals only; raw responses withheld) |
| CPU serving speed versus GLiNER2 | — | not shipped | Timing measured on an internal serving build; not verifiable from this package |

## Beyond the paper: a fresh fit with this package

| Claim | Value | Evidence | Status |
|---|---|---|---|
| O4 recipe from scratch on public gold, public corpora and O4's web text with teacher labels, human gold (max / zero bias) | 89.46 / 89.35; +1.41 [0.30, 2.53] vs O4 | `receipts/fresh-fit-summary.json` | Recorded; reproducible with `fetch-web`, `annotate` (your teacher) and the pipeline |
| Same recipe on public data only | 86.62 / 82.15; −6.21 [−8.05, −4.40] vs O4 | same | Recorded; reproducible with the pipeline |

What `verify` cannot establish: that withheld gold labels are correct or
that counts came from the stated predictions. The human-gold rows are public,
so for that population you can rebuild the rows and score your own models
with identical code. The Ont3 population is not released.
