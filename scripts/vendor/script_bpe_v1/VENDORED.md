# SCRIPT v1 character projection

## Upstream

- Repository: <https://github.com/sanderland/script_bpe>
- Commit: `0fda2bc14109350040274f9ec26997655220533c`
- Commit date and subject: 2025-06-01, `publish code and tokenizers`
- Source artifact:
  `results/tokenizers/CulturaX-subsample-100-bal2/n256000/scriptenc_cb.json.gz`
- Source artifact SHA-256:
  `a139b14421646e1541c15cd67cc1afdd2ffa4854091da9aef8f8bc526b28622f`
- Unicode 16.0.0 `Scripts.txt` SHA-256 used upstream:
  `9e88f0a677df47311106340be8ede2ecdacd9c1c931831218d2be6d5508e0039`

## Vendored

2026-08-13.

## License

The pinned upstream repository carries the Apache License 2.0. Its complete
`LICENSE` is retained here.

## Vendored files

- `LICENSE`:
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- `script_encoding_v1.json`:
  `895fc5cd38a93f8d64509215d2741a4f93f2df722fd5404c8fd27bc9774121fc`

## Local changes

`script_encoding_v1.json` is a deterministic extraction rather than a verbatim
copy. It retains only the published pretokenizer's SCRIPT v1 version, block and
index counts, diagnostics, construction settings, and complete block table. It
drops BPE merge rules and all other tokenizer metadata, writes compact
uncompressed JSON, and adds the source hashes above. Character-to-pair
assignments are unchanged.

## Re-sync

Resolve the pinned repository with `librarian`, extract
`pretokenizer.config` from the named gzip-compressed JSON artifact, retain
`version`, `num_index_tokens`, `num_blocks`, `stats`, `settings`, and `blocks`,
then add the `source` object recorded above and serialize compact UTF-8 JSON
with a trailing newline. Verify:

```bash
sha256sum LICENSE script_encoding_v1.json
```

The hashes must match the Vendored files section unless this record is
deliberately re-pinned and reviewed.
