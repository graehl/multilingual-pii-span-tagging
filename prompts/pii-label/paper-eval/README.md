# Paper prompted-evaluation prompt (ctx-v6)

The prompt behind the paper's prompted Gemma-4 31B annotation result
(69.71 exact typed span F1, 83.98 redaction-character F1 on the 659 Ont3
evaluation rows; `paper-o4-v1/local-llm-summary.json`), frozen here because
the working copies have changed since.

- `task.txt` — `task-ont3-primary-annotate-context-v6.txt` (unchanged).
- `catalog.md` — `catalog-ontology-v3-primary-v2.md` at commit `8dbe21911`,
  the run's commit.
- `examples.json` — `examples-reviewed-final35-recall-ont3-v1.json` at
  commit `aba47287f` (2026-09-07).

Recovered 2026-10-02. With the run's 31-tag inventory (`--fmt json-seq`,
legacy span policy, plain input rendering, each row's `annotation_guidance`
as guidance) these three files make `build_prompt` reproduce all 1,120 D3
prompts of run `pii-final35-d3-gemma4-31b-fp8-ctx-v6-v1` byte for byte,
checked against the SHA-256 the run stored per row. The run's prompt
contract records a different examples-file hash; the rendered prompts are
identical, so that difference is in bytes that never reach the prompt.

The original request: one user message, temperature 0, no thinking kwarg
(served alias `gemma4-31b-fp8`), up to 3,072 completion tokens, one format
retry. Its outputs were routed by the rule in
`research/pii/frontier/evidence/paper-ont3-format-fallback-v1.md`
(`research/pii/frontier/evidence/gemma31-precision-v1/route.py`).
