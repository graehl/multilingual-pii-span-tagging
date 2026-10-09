# Ont3 evaluation populations

The paper's 31-type (Ont3) dev and test sets: 1,201 segments of public web
text in 35 languages with the authors' reference annotations, included so
that other taggers can be scored the way the paper scores O4. Only these
evaluation sets ship; none of the roughly 50,000 Ont3-annotated training
rows do. `pii-reproduce.py evaluate`
(default `--population all`) predicts on these inputs, applies the serving
stages unless `--raw`, and pairs the result against O4's receipts with
`scripts/pii_software_ont3.py`. The rows, text, document context and
references are the ones the paper's predictions were made and scored on; the
scorer checks each sweep's recorded input hash against these files.

| File | Rows | Content |
|---|---|---|
| `selection-inputs.jsonl` | 659 | Prediction input for the selection collection (`id`, `text`, `lang`, document context). Its `spans` are an earlier revision; score against `selection-references.jsonl`. |
| `selection-references.jsonl` | 659 | References: two independent annotation passes, exact agreements kept (`review_basis: exact_two_pass_agreement`, 383) and disputes adjudicated (`adjudicated_after_dispute`, 276), then manually revised (r4). |
| `selection-language-review.json` | — | Language corrections applied to scoring, by row-id prefix. |
| `heldout.jsonl` | 542 | Input and references: needle-selected paragraphs from training-disjoint documents, one Luna teacher pass. |

**Selection status.** O4 was selected on the 659 selection rows, so they are
a reused development set; the 542 held-out rows were never used to select
it. Report both and the pooled 1,201.

**Paper names.** The paper calls the selection rows Silver-dev, the held-out
rows Silver-test, and all 1,201 rows Silver; receipts and sweep files call
them `ont3`, `heldout`, and the pooled Ont3 development set.

**Spans** are `[start, end, type]` character offsets into `text` over the 31
Ont3 types (`docs/ontology.md`); `*_reference` types are optional references,
neutral in scoring.

**Source.** The text comes from FineWeb (English) and FineWeb-2, which
redistribute Common Crawl data under ODC-By 1.0; their terms and Common
Crawl's terms of use apply to the text.
