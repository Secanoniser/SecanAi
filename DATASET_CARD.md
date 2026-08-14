# Dataset Card — SecanAi Instruction Data

## Status

**Release status: blocked pending source provenance review.** The repository includes `json info/sft_thinking_dataset.jsonl` and `json info/data_export.json`, but does not document their authors, licence, collection date, or redistribution/training rights.

## Intended use

The processed instruction data is intended only for local experimentation with small instruction-following models. It is not approved for production, high-stakes, or public deployment.

## Processing

`prepare_sft_data.py` normalizes text, attempts safe repair of common mojibake, removes exact normalized instruction/completion duplicates, and deterministically assigns train/validation/test splits. `validate_data.py` records validity, missing fields, duplication, encoding indicators, and split distribution.

## Known limitations

- The source file has substantial prompt duplication.
- Many source rows contain likely text-encoding corruption.
- The material is dominated by short math and Q&A records.
- The source `data_export.json` is mostly empty and is not part of the training pipeline.
- Exact deduplication is not semantic/near-duplicate deduplication.

## Required before release

Fill in source owner, licence, source URL or collection method, collection date, personal-data review, language coverage, and known bias/safety assessment. Retain a versioned preprocessing report for each trained model.
