# Model Card: smollm2-baseline

- **Checkpoint source:** `smollm2-baseline`
- **Model identifier:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Card generated:** 2026-08-16T12:12:22.801140+00:00

## Overview

A small locally runnable causal language model served by the SecanAi chat stack (FastAPI + Transformers). It is intended for educational use, local prototyping, and experiments with retrieval-augmented chat on a single CPU/GPU machine.

## Architecture

- Parameters: **134.5M**
- Model type: `llama`
- Hidden size: `576`
- Layers: `30`
- Attention heads: `9`
- Context window: `8192` tokens
- Chat template: `yes`

## Training data

- `smollm2-baseline`: `HuggingFaceTB/SmolLM2-135M-Instruct` (Hugging Face), served as-is.
- `local-checkpoint`: locally trained/fine-tuned weights in `output_model/` built from `scrape_data.py` (Wikipedia article introductions) plus `json info/sft_thinking_dataset.jsonl`. See `train.py` / `sft_train.py` for the exact recipe used for this checkpoint.

## Evaluation

- Timestamp: 2026-08-16T12:12:07.057389+00:00
- Keyword hit rate: **100.0%** (5/5)
- Refusals: 0/5

| Category | Hits | Total |
| --- | --- | --- |
| python | 5 | 5 |

Run `python eval_harness.py --source <source>` to refresh these numbers.

## Intended use

- Local, offline chat experiments (see `server.py` + `index.html`).
- Studying how retrieval (RAG) changes answer quality for small models.
- Benchmarking SFT vs. baseline behavior with `eval_harness.py`.

## Limitations

- Very small parameter count: limited factual recall, weak long-form reasoning.
- CPU-only serving in this repo; generation is slow for long replies.
- The safety filter is a transparent first-pass blocklist, not a classifier.
- The local checkpoint was trained on a small curated corpus; it should not be treated as a general-knowledge assistant.
