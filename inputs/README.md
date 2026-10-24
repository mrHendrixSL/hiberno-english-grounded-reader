# Local inputs (not distributed)

This directory holds the real inputs the pipeline notebook needs before
it can run. Place exactly these files here:

| File | What it is | Tracked in this repo? |
|---|---|---|
| `source_witness.docx` | The dictionary source document itself. | **No** -- gitignored. Copyrighted book text. |
| `sample_disagreement.parquet` | Per-entry reader-disagreement scores from a historical design snapshot (`entry_id`, `n_readers`, `disagreement`), used by the E390 stratified sample design. | Yes -- no dictionary text, numeric scores only. |
| `eval390_reference.jsonl` | The 390 manually adjudicated entries corresponding to E390's membership. Used only to build the adjudicated-precedent bank the grounded reader conditions retrieve from -- never treated as ground truth for scoring. | **No** -- gitignored. Contains full adjudicated definitions/quotes/etymologies, i.e. substantial verbatim dictionary content, the same as the source document itself. |
| `h390_reference.jsonl` | The held-out H390 sample's adjudication scaffold: one empty `Entry`-schema skeleton per H390 entry_id, every extraction field at its empty default, filled in later by independent human adjudication. Never filled from rule-based or LLM output. | **No** -- gitignored (pre-emptively -- once filled in, it carries the same verbatim content as `eval390_reference.jsonl`). |

The notebook checks each file's presence before using it, and stops at
that cell rather than substituting an invented or regenerated stand-in
if one is missing.

## Why some *generated* files are also excluded

Some files the pipeline itself produces under `data/` are excluded from
this repository for the identical reason as `eval390_reference.jsonl`
above -- not because they're large, but because they contain the
dictionary's own text, just re-encoded rather than reworded:

- `data/01_ingest/runs.parquet` -- its `text` column reconstructs the
  full document body, run by run.
- `data/03_segmentation/entries.parquet` -- its `raw` column is every
  entry's printed text verbatim.
- `data/08_rule_based/rb.jsonl` -- a complete structured transcription
  (headword/definition/senses/quotes/etymology) of all 3,899 entries.

Every other stage output under `data/` (features, the E390 sample
design, the retrieval index and table, the precedent-bank content hash,
the manifest) is structural, numeric, or a fingerprint -- no dictionary
prose -- and is tracked normally.
