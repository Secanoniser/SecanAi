# SecanAi

SecanAi is a local instruction-model project. Its supported workflow fine-tunes a small Hugging Face causal-language model on validated JSONL data, evaluates the resulting artifact, and serves it through a localhost FastAPI chat UI.

## Requirements

- Python 3.10–3.12 is the supported range for the dependency set.
- PyTorch installed for your CPU or CUDA environment.
- Sufficient disk space and network access for the selected base model on the first run.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_sft_data.py
python validate_data.py data/processed/sft.jsonl --canonical --output artifacts/runs/data_validation.json
python sft_train.py --epochs 1
python eval.py
python server.py
```

Then open `http://127.0.0.1:8000`. The server prefers `artifacts/models/sft`; if it is not present it explicitly falls back to the configured base model.

## Project layout

- `settings.py` — repository-relative paths and environment configuration.
- `prepare_sft_data.py` — normalize, deduplicate, and split source JSONL.
- `validate_data.py` — data-quality report and schema checks.
- `sft_train.py` — supported fine-tuning workflow with assistant-only loss masking.
- `eval.py` — base-versus-candidate held-out evaluation.
- `server.py` — local FastAPI server with health and model metadata endpoints.
- `DATA_CONTRACT.md` — required training-data schema.
- `DATASET_CARD.md` — source/provenance documentation to complete before model release.

## Configuration

Copy `.env.example` values into your shell environment as required. Every path can be relative to the repository or absolute. The most useful settings are:

- `SECANAI_MODEL_PATH` — local model artifact to serve or evaluate.
- `SECANAI_BASE_MODEL` — fallback/base Hugging Face model ID.
- `SECANAI_DATA_DIR` and `SECANAI_ARTIFACTS_DIR` — data and run locations.
- `SECANAI_HOST`, `SECANAI_PORT`, prompt and generation limits.

The default host is `127.0.0.1`. Do not expose the server to a network without adding authentication, rate limiting, CORS policy, and deployment controls.

## Data workflow

The supplied source dataset is retained in `json info/`. It is not training-ready: the audit found many duplicate prompts and likely encoding corruption. Always run `prepare_sft_data.py` and `validate_data.py` before training. The processor writes a canonical format to `data/processed/sft.jsonl` and retains a processing report.

See `DATA_CONTRACT.md`, `DATASET_CARD.md`, and `DATA_AND_PROJECT_AUDIT.md` for constraints and known limitations.

## Tests

Run the dependency-free checks with:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
```

The server requires the packages in `requirements.txt`; model training and full API inference require a downloaded model and appropriate hardware.

## Experimental scripts

`train.py`, `train_tokenizer.py`, `generate_corpus.py`, and `scrape_data.py` are an experimental from-scratch pre-training path. It is separate from the supported fine-tuning workflow and should not be treated as production-ready without its own provenance, validation, and evaluation work.
