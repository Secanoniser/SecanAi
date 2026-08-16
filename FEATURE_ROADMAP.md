# Feature Roadmap

This is the project-local implementation roadmap derived from the supplied feature review. It complements the existing training plans rather than replacing them.

## Diagnosis log (2026-08-16) - garbage output root cause

Symptom: the chat UI produced byte-level garbage (e.g. "Gånd... Gß...") for
simple prompts, even at low temperature.

Debug order followed (raw checkpoint -> tokenizer -> raw generation -> template):
1. Test B (tokenizer round-trip): `'What is Python?'` -> `ĠW h at Ġis ĠPython <unk>`
   - the `?` collapsed to `<unk>`, spaces leaked as raw `Ġ` byte markers.
2. Vocab inspection: `output_model/config.json` reported `vocab_size: 751`
   (256 bytes + ~491 merges). `min_frequency=2` on a ~7 KB corpus starved the BPE.
3. Test C (raw generation, no RAG/template/streaming): garbage at greedy,
   temp 0.3, and temp 0.7 -> the defect is in the checkpoint, not the pipeline.
4. Control: SmolLM2-135M-Instruct round-trips and generates cleanly.

Root causes (in order of impact):
- Broken tokenizer: no `ByteLevel` decoder registered (raw `Ġ`/`Ċ` markers),
  no `byte_fallback` (`?` -> unk), 751-token vocab from a ~7 KB corpus.
- Undertrained model: ~8K training tokens total - no language to learn.
- Temperature 1.5 in the UI amplified randomness.

Fixes applied:
- [x] `train_tokenizer.py` rewritten: `decoder = ByteLevel()`,
      `byte_fallback=True`, `vocab_size=8192`, round-trip self-check
      (verified: id-stable, lossless).
- [x] `scrape_data.py` rewritten: full article text, append-safe, dedupe by
      article marker, rate-limited. Corpus: 7.3 KB -> 1.1 MB (150x).
- [x] `model_router.py`: baseline is the default again; the local checkpoint
      is served only via explicit `MODEL_SOURCE` until it passes the eval
      quality gate (eval_harness.py hit rate on the fixed suite).
- [x] `server.py` + `index.html`: temperature clamped to 0..1 (default 0.7).
- [x] `/api/model` now reports vocab_size, tokenizer, dtype, device.
- [x] Retraining the 105M model on the 1.1 MB corpus (background run,
      ~19 min on CPU, saves at step 500). Result: generation is now readable
      English (tokenizer fix confirmed) but the model fixates on "Fourier
      transform" - that single 257 KB article is ~23% of the corpus, and
      1 epoch is not enough. Eval quality gate: 0/40 keyword hits
      (eval_20260816_125045_local-checkpoint). The baseline stays the
      default; the local checkpoint remains behind explicit MODEL_SOURCE.
      Next training pass should balance the corpus (cap per-article size,
      re-run scrape for the rate-limited topics) and run 2-3 epochs.

## Foundation hardening (2026-08-16, second pass)

- [x] `scrape_data.py` v2: exponential-backoff retry honoring `Retry-After`,
      contact email in User-Agent, per-article size cap (100K chars),
      `--rebalance` mode. Result: 83/83 articles, 0 failures,
      corpus 1.1 MB -> 3.8 MB balanced.
- [x] `train.py`: relative paths, `TRAIN_RESUME` (safe resume from latest
      checkpoint), `TRAIN_LR`/`TRAIN_EPOCHS`/`TRAIN_SAVE_STEPS` knobs.
- [x] `train_tokenizer.py`: retrained at vocab 16384 on the balanced corpus;
      id-stable lossless round-trip verified.
- [x] Pre-training relaunched: 128.99M params (16K vocab), 2 epochs,
      ~3190 steps, save every 1000 steps, ~2.3h on CPU (detached via WMI so
      it survives shell interruption). Checkpoints land in `output_model/`.
- [x] `sft_train.py` rewritten (level-up prep): local checkpoint by default,
      trains on the 100K-row thinking dataset with prompt/label masking,
      chat-template aware, stratified CPU-friendly subsampling
      (--max-examples, default 5000).
- [ ] After pre-training: eval `local-checkpoint` (quality gate), then
      `python sft_train.py`, then re-eval and adopt if hit rate is close to
      baseline.

## Current architecture (verified 2026-08-16)

The repository has two distinct model paths:

| Area | Current implementation | Implication |
| --- | --- | --- |
| Training | `train.py` creates and saves a from-scratch `LlamaForCausalLM` to `output_model/` | This is the experimental local model. |
| Inference test | `test_inference.py` loads `output_model/` directly | This is the correct place to judge the trained checkpoint. |
| Web chat | `server.py` decides at load time via `model_router.py` (Layer 1) | `MODEL_SOURCE` env var; local checkpoint preferred when servable. |
| Frontend | `index.html` streams replies, persists conversation memory, shows active model | The UI displays the real model identity from `/api/model`. |

The first product decision is therefore explicit: the server must declare whether it is running the local checkpoint or the SmolLM2 baseline. The baseline should remain available as a comparison, but never be presented as the locally trained model.

## Priority order

### P0 - Make the system honest and runnable

- [x] Add a `MODEL_SOURCE` configuration (`smollm2-baseline` or `local-checkpoint`) to `server.py`. (extracted into `model_router.py`)
- [x] For `local-checkpoint`, load `output_model/` with `AutoTokenizer` and `AutoModelForCausalLM`; fail clearly if it has not been trained yet.
- [x] Add `GET /api/model` and show the active model source/name in the UI.
- [x] Declare the FastAPI runtime dependencies in `requirements.txt`.
- [ ] Pin tested dependency versions or add a lockfile once the environment is stable.

Acceptance: opening the UI makes it unambiguous which checkpoint generated every response, and `pip install -r requirements.txt` installs the server's imports.

### P1 - Improve the local model before adding product complexity

This maps directly to `EXPERT_ACTION_PLAN.md` and is the prerequisite for a useful local-checkpoint mode.

- [ ] Run `debug_gen.py` and save representative raw generations.
- [x] Expand `json info/sft_thinking_dataset.jsonl` to at least 50 diverse, reviewed examples. (dataset is now 42 MB)
- [ ] Train the 80-100M configuration in `train.py` on substantially more clean text.
- [x] Ensure every generation path supplies `attention_mask`, `eos_token_id`, `pad_token_id`, and `no_repeat_ngram_size=3`.
- [x] Record the training command, data revision, model parameters, and result in a model card. (`model_cards.py`)

Note: `scrape_data.py` already implements the first Wikipedia-fetching version of Task 5, but it overwrites `corpus.txt` and fetches only six article introductions. Expand it into a reproducible, append-safe data pipeline before treating it as a large corpus.

### P2 - Build a usable chat contract

- [x] Replace the single `prompt` field with a typed `messages` array while retaining a short-lived compatibility path for `prompt`.
- [x] Add a configurable system prompt declaring the model's identity, limitations, and safety boundaries.
- [x] Make `max_new_tokens`, `temperature`, `top_p`, `top_k`, and repetition penalty optional request settings with bounded defaults.
- [x] Add token streaming with `TextIteratorStreamer` and FastAPI `StreamingResponse`.
- [x] Persist browser-side conversation history per chat, then send it on each request.

Acceptance: a follow-up question can reference a previous turn, the model identity is consistent, and replies appear while they are generating.

### P3 - Give a small model external knowledge and feedback loops

- [x] Chunk the cleaned scraped corpus and create a local embedding index (lexical fallback shipped; FAISS semantic index optional via `requirements-rag.txt` + `build_rag_index.py`).
- [x] Retrieve a small number of relevant chunks at answer time and label them as supplied context.
- [x] Create a fixed 30-50 prompt evaluation suite across Python, math, science, AI, and general help. (`eval_harness.py`, 40 prompts)
- [x] Save dated evaluation outputs and compare them before adopting a checkpoint. (`eval_results/`, `--compare`)
- [x] Add basic request telemetry: active model, input length, output length, and latency. Do not log user content by default. (`logs/requests.jsonl`)

### P4 - Deferred platform work

- [ ] Define a forward-compatible tool-call response shape.
- [ ] Evaluate 8-bit/GGUF loading on the intended Windows hardware before standardizing on a quantization path.
- [x] Maintain `MODEL_CARD.md` files for released checkpoints: data sources, date, parameter count, eval result, intended use, and limitations. (`model_cards.py` -> `model_cards/MODEL_CARD_*.md`)

## Suggested first implementation slice

1. Add server dependencies and model-source configuration.
2. Expose the active model through `/api/model` and the UI.
3. Add the `messages` contract and system prompt.
4. Verify both model-source branches with one smoke-test request each.

Do not begin retrieval or tool use until the local-checkpoint branch has coherent output on the fixed evaluation prompts.

## Relationship to existing documents

- `IMPROVEMENT_PLAN.md` owns data collection, tokenizer, and model scaling.
- `EXPERT_ACTION_PLAN.md` owns the immediate generation-quality diagnosis and fixes.
- This file owns the serving/product integration work and establishes the dependency order across those plans.