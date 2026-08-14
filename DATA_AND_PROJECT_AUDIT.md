# SecanAi Data and Project Audit

**Audit date:** 2026-08-14  
**Scope:** Repository contents, configuration, Python syntax, packaged datasets, and dependency declarations. This is a static audit; model training and inference were not run.

## Executive summary

SecanAi is a promising local-LLM prototype with three overlapping paths: training a Llama-style model from scratch, fine-tuning SmolLM2, and serving a SmolLM2 chat UI. The current repository is not yet reproducible on a new machine, and its largest included SFT dataset needs quality work before it is used for training.

The highest-value next steps are:

1. Make all paths configurable and relative to the repository instead of using a previous developer's absolute Windows directory.
2. Repair or remove text-encoding corruption, then deduplicate and split the SFT data before training.
3. Declare and lock all runtime dependencies.
4. Choose and document one supported workflow (recommended: fine-tune and serve a base model) rather than treating pre-training and serving as a single pipeline.
5. Add a small evaluation suite and automated checks.

## Data inventory

| Asset | Size | Structure | Findings |
| --- | ---: | --- | --- |
| `json info/sft_thinking_dataset.jsonl` | 42,532,098 bytes | 100,012 valid JSONL rows; fields: `prompt`, `completion`, `instruction`, `thinking`, `response`, `type` | 60,000 `math` and 40,012 `qa` records. No empty core fields or invalid JSON lines. |
| `json info/data_export.json` | 389,294 bytes | 51 JSON objects; fields: `page`, `title`, `authors`, `abstract`, `section_1`, `note` | 50 of 51 rows have an empty value for every field except `page`; the file does not presently look useful as a training source. |

### SFT data-quality measurements

| Measure | Result | Risk |
| --- | ---: | --- |
| Duplicate prompt rows | 81,021 of 100,012 | High: prompt-level duplication will overweight a small number of examples and distort evaluation. |
| Rows with likely mojibake (`Â`, `Ã`, or `â€`) | 69,092 | High: broken Unicode can teach malformed text and appears in repository examples as `aÂ²`. |
| Median prompt length | 42 characters | Medium: prompts are very short, limiting multi-step instruction behavior. |
| Median completion length | 121 characters | Medium: completions are brief; this is unsuitable as the sole source for rich assistants. |
| Median response length | 8 characters | High: the relation between `completion` and `response` needs a documented schema before either is used as the target. |

### Required data actions

1. Define one canonical training target: either chat messages or an `instruction`/`response` pair. Do not concatenate six fields without a schema.
2. Normalize source text to UTF-8 and reject or fix mojibake before tokenization.
3. Deduplicate by normalized prompt and normalized target, retaining provenance and a count for intentional variants.
4. Create deterministic train/validation/test splits before any training. Keep near-duplicates in the same split to avoid leakage.
5. Add a data-validation script that checks JSONL parsing, required fields, Unicode quality, length bounds, duplicate rate, and split overlap.
6. Record source, licence, collection date, and permitted use for every dataset. The current data files do not provide enough provenance for a production dataset card.

## Project findings

### Critical reproducibility gaps

- `train.py`, `train_tokenizer.py`, `sft_train.py`, `scrape_data.py`, `main.py`, `eval.py`, `query_ai.py`, and `test_inference.py` hard-code `C:\\Users\\Nyxentra\\Desktop\\local_llm`. They will not use this cloned repository without edits.
- `requirements.txt` omits packages directly imported by the project: `datasets`, `fastapi`, `uvicorn`, `requests`, `pydantic`, and `tokenizers`.
- Dependency versions are only lower bounds, so a fresh installation is not reproducible. A lock file or tested constraint set is needed.

### Architecture and workflow ambiguity

- `train.py` initializes a Llama model from scratch, whereas `sft_train.py` fine-tunes `HuggingFaceTB/SmolLM2-135M`, and `server.py` serves `HuggingFaceTB/SmolLM2-135M-Instruct` directly. These are distinct model lifecycles and output formats.
- `server.py` never loads the locally fine-tuned output, so completing `sft_train.py` does not change what the web UI serves.
- The server downloads and loads the model at import time, which makes health checks, tests, and startup failure handling difficult.

### Serving and security reliability

- `/api/chat` accepts unbounded prompts and performs generation directly in the async handler. Add request-size limits, generation limits, a concurrency policy, and request timeouts.
- The API returns raw exception text in HTTP 500 responses. Log diagnostics server-side and return a generic message to clients.
- `FileResponse("index.html")` depends on the process working directory; resolve it relative to `server.py`.
- The server currently exposes no health endpoint, model metadata endpoint, authentication, or rate limiting. Authentication is optional for a localhost-only demo, but should be required before network exposure.

### Training and evaluation gaps

- The custom pre-training path has no validation set, data collator, resume strategy, seed, run metadata, or loss/perplexity report.
- The SFT path trains labels over padded tokens and user-prompt tokens. Mask padding and, for instruction tuning, mask the prompt so loss focuses on the assistant answer.
- `eval.py` is a set of generated examples, not a benchmark. Add held-out exact-answer tasks, instruction-following cases, toxicity/safety checks appropriate to the intended use, and regression thresholds.
- All Python files parsed successfully in this audit; that does not verify imports, CUDA setup, model availability, or runtime behavior.

## Recommended delivery sequence

1. Add a central settings module or CLI arguments for repository-relative `data`, `artifacts`, and `models` paths.
2. Fix `requirements.txt`, create a tested environment specification, and add a one-command smoke test.
3. Build `validate_data.py`; normalize, deduplicate, split, and document the JSONL dataset.
4. Select the supported model route and align training output, inference scripts, and `server.py` around it.
5. Add evaluation and CI checks: syntax, data validation, unit tests, and an API smoke test with model loading mocked.
6. Only then run a small reproducible fine-tune and compare it to the untouched base model on the held-out set.

## Acceptance criteria for the next iteration

- A new contributor can follow the README from clone to a passing local smoke test without editing source paths.
- A data report produced by a script records row counts, duplicate rate, Unicode issues, schema checks, and split sizes.
- The served model is explicitly identified and can be selected via configuration.
- Evaluation results compare a baseline with the trained model on a held-out dataset.
- The project documents dataset provenance and intended use.
