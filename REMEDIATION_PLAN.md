# SecanAi Remediation Plan

**Created:** 2026-08-14  
**Basis:** `DATA_AND_PROJECT_AUDIT.md`  
**Goal:** Make SecanAi reproducible, data-safe, testable, and capable of serving the model it trains.

## Guiding decisions

Before implementation, adopt these boundaries:

1. **Supported path:** Fine-tune an existing small open model and serve that saved fine-tuned artifact. Keep from-scratch pre-training as an explicitly separate experimental path.
2. **Data target:** Use one canonical instruction dataset format, with a documented `messages` representation or a clear `instruction`/`response` pair. Assistant text is the supervised target.
3. **Local-first operation:** Bind the server to localhost by default. Network exposure is a later, explicit deployment feature.
4. **No training on unvalidated data:** Data validation, normalization, deduplication, and split creation are release gates for training.

## Phase 0 — Preserve a baseline

**Purpose:** Keep a known-good reference before modifying the pipeline.

- Record the current commit hash and current environment details.
- Keep `DATA_AND_PROJECT_AUDIT.md` as the baseline evidence.
- Add a short decision log documenting the chosen supported path and model ID.
- Do not delete the existing from-scratch scripts; mark them as experimental until they are separately repaired.

**Done when:** Baseline behavior and the intended product path are written down.

## Phase 1 — Make the project runnable from a clone

**Problems addressed:** Hard-coded paths, incomplete dependencies, unclear setup.

1. Add a central settings layer using repository-relative paths:
   - `data/raw`
   - `data/processed`
   - `artifacts/tokenizer`
   - `artifacts/models`
   - `artifacts/runs`
2. Replace every `C:\\Users\\Nyxentra\\...` path with settings, CLI options, or environment variables.
3. Split dependencies into runtime and development requirements, including all imported packages.
4. Pin a tested dependency range and document the supported Python and PyTorch/CUDA combinations.
5. Rewrite the README with one supported quick-start route: create environment, install, validate data, run a smoke test, train, evaluate, serve.
6. Add a `.env.example` for optional configuration; never commit actual secrets.

**Verification:** A clean virtual environment can run a non-networked smoke test from any working directory.

## Phase 2 — Repair and govern the data

**Problems addressed:** Duplicates, mojibake, empty export data, unclear schema, missing provenance, leakage risk.

1. Create `data_contract.md` defining every accepted field, its type, required status, allowed values, and canonical target format.
2. Add `validate_data.py` to report:
   - JSON/JSONL validity
   - required fields and empty values
   - Unicode/encoding anomalies
   - length distribution and outliers
   - duplicate and near-duplicate rates
   - category distribution
   - train/validation/test split overlap
3. Build `prepare_sft_data.py` that:
   - decodes and normalizes text to UTF-8
   - flags or fixes known mojibake only when unambiguous
   - normalizes whitespace
   - removes exact duplicates by normalized prompt and target
   - retains source ID and a rejection reason for excluded records
   - creates deterministic splits using a fixed seed
4. Quarantine `data_export.json` until its schema and source quality are confirmed; do not train on mostly empty records.
5. Add a dataset card with source, licence, collection date, intended use, known limitations, and processing history.
6. Save the validator output as versioned run metadata, not as an untracked manual observation.

**Verification:** Processed data has zero invalid records, no known encoding corruption, documented provenance, and zero exact prompt-target overlap across splits.

## Phase 3 — Align training around one artifact

**Problems addressed:** Disconnected pre-training/SFT/serving paths; incorrect label masking; missing reproducibility controls.

1. Refactor SFT into a CLI with explicit arguments for base model, input data, output directory, seed, hyperparameters, and device policy.
2. Convert canonical examples into the selected model's chat template.
3. Construct labels so padding and user/system prompt tokens use `-100`; calculate loss only over assistant tokens.
4. Add validation evaluation during training, checkpoint retention, resume support, and saved run configuration.
5. Save tokenizer, model, generation configuration, dataset version, and metrics together in one model artifact directory.
6. Keep `train.py` under `experiments/pretraining/` or label it clearly as experimental; it must have its own data and evaluation plan before use.

**Verification:** A training run produces one self-contained artifact that can be loaded without relying on a developer-specific path.

## Phase 4 — Establish evaluation before tuning further

**Problems addressed:** No benchmark, no baseline comparison, no regression protection.

1. Create a held-out evaluation set from the processed test split; never use it for training choices.
2. Define task-appropriate checks:
   - exact-answer math questions
   - factual and explanatory prompts
   - instruction-following format checks
   - refusal/safety checks appropriate to intended use
   - response-length and repetition checks
3. Evaluate both the untouched base model and the fine-tuned artifact using identical prompts and deterministic generation settings where possible.
4. Report quantitative results plus a small reviewed qualitative sample.
5. Define promotion thresholds: a tuned model must improve target metrics without unacceptable regression on safety, repetition, or response quality.

**Verification:** `evaluate.py` generates a machine-readable report comparing baseline and candidate models.

## Phase 5 — Serve the trained model safely and reliably

**Problems addressed:** Server ignores trained output, import-time loading, unsafe errors, unbounded requests, fragile static-file path.

1. Add server configuration for `MODEL_PATH` or `MODEL_ID`; default it explicitly and expose the active model in a metadata endpoint.
2. Load the model during the FastAPI lifespan/startup phase, not at import time.
3. Resolve `index.html` from the script directory.
4. Add `/health` and `/api/model` endpoints.
5. Validate requests with maximum prompt length and generation limits.
6. Serialize or limit generation concurrency to match available hardware, and use a defined timeout policy.
7. Log detailed exceptions locally; return safe generic errors to clients.
8. Keep host `127.0.0.1` by default. If network serving is requested later, add authentication, rate limiting, CORS policy, and deployment documentation.

**Verification:** The API smoke test starts the application, verifies health/model endpoints, sends valid and invalid requests, and confirms the configured local artifact is used.

## Phase 6 — Automate quality gates

**Problems addressed:** No tests or CI.

1. Add unit tests for settings, data validation, data preparation, label masking, and API request validation.
2. Add integration tests with a tiny local/mock model so tests do not require a large download or GPU.
3. Add formatting, linting, static checks, and dependency auditing.
4. Add CI to run syntax checks, unit tests, dataset schema checks, and the mocked API smoke test on each change.
5. Add release checks that require a model card, dataset card, evaluation report, and reproducible configuration for any published artifact.

**Verification:** CI passes from a clean checkout and fails deliberately on invalid data or missing required metadata.

## Implementation order and checkpoints

| Checkpoint | Deliverables | Gate |
| --- | --- | --- |
| A: Foundation | Settings, dependencies, README, smoke test | New clone runs without source edits. |
| B: Trusted data | Data contract, validator, processor, dataset card, deterministic splits | Validation passes and data report is produced. |
| C: Coherent training | SFT CLI, masked labels, checkpoint/config metadata | Artifact reloads and validation metrics are saved. |
| D: Evidence | Baseline-vs-tuned evaluation report | Candidate meets defined promotion thresholds. |
| E: Local product | Configured FastAPI server, health/model endpoints, API tests | UI serves the evaluated local artifact. |
| F: Maintenance | Test suite, linting, CI | All gates pass on a clean checkout. |

## Explicit non-goals for this remediation cycle

- Training a foundation model from scratch at large scale.
- Exposing the API publicly.
- Adding chain-of-thought training data or displaying private reasoning traces.
- Treating scraped text as usable training data without licence and provenance review.

## First implementation batch

Start with Checkpoint A and B only:

1. Centralize settings and remove absolute paths.
2. Correct and pin dependencies.
3. Add the data contract and validator.
4. Add the processing pipeline and deterministic splits.
5. Update the README and add tests for the new foundation.

This order prevents spending compute on a data pipeline that cannot yet be trusted or reproduced.
